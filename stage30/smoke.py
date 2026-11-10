"""Check FP16 Qwen generation and one packed MLP during prefill and decode."""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_files import MODEL_DIR, MODEL_ID, REVISION
from .qwen_mlp import PackedMLP, ReconstructedMLP, load_selected_kernel, pack_mlp


def peak():
    torch.cuda.synchronize()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16, use_safetensors=True, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa",
    ).eval().to("cuda")
    assert model.config.model_type == "qwen3" and next(model.parameters()).dtype == torch.float16
    return model, tokenizer


def error(actual, expected):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm() / expected.norm().clamp_min(1e-12)).item(),
                top1_match=bool(torch.equal(actual.argmax(-1), expected.argmax(-1))))


def trace(model, prompt, steps, continuation=None):
    logits, tokens = [], []
    x = prompt
    cache = None
    for step in range(steps):
        result = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        assert cache.get_seq_length() == prompt.shape[-1] + step
        last = result.logits[:, -1, :]
        logits.append(last)
        x = last.argmax(-1, keepdim=True) if continuation is None else continuation[:, step:step+1]
        tokens.append(x)
    return logits, torch.cat(tokens, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(Path(__file__).with_name("model-receipt.json").read_text())
    assert receipt["revision"] == REVISION and receipt["model"] == MODEL_ID
    assert all((MODEL_DIR / name).stat().st_size == size for name, size in receipt["file_bytes"].items())
    gemm, kernel_hash = load_selected_kernel()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model()
    load_memory = peak()
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt",
            ).to("cuda")
            layer = model.model.layers[0]
            original = layer.mlp
            inputs = []
            handle = original.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach()))
            torch.cuda.reset_peak_memory_stats()
            try:
                fp16, continuation = trace(model, prompt, 8)
            finally:
                handle.remove()
            fp16_memory = peak()
            weights = pack_mlp(original)
            packed, expanded = PackedMLP(weights, gemm), ReconstructedMLP(weights)
            block = []
            for x in inputs:
                actual, expected = packed(x), expanded(x)
                torch.testing.assert_close(actual, expected, atol=.005, rtol=.005)
                block.append(dict(input_shape=list(x.shape), **error(actual, expected)))
            try:
                layer.mlp = expanded
                reference, _ = trace(model, prompt, 8, continuation)
                layer.mlp = packed
                actual, _ = trace(model, prompt, 8, continuation)
            finally:
                layer.mlp = original
            logits = []
            for index, (got, ref, fp) in enumerate(zip(actual, reference, fp16)):
                metrics = error(got, ref)
                assert metrics["relative_l2"] <= .005, (index, metrics)
                logits.append(dict(phase="prefill" if index == 0 else "decode", step=index,
                                   packed_vs_reconstructed=metrics,
                                   reconstructed_vs_fp16=error(ref, fp)))
            bytes_packed = sum((w.packed.numel() * w.packed.element_size()
                                + w.scales.numel() * w.scales.element_size()) for w in weights.values())
            bytes_fp16 = sum(getattr(original, name + "_proj").weight.numel() * 2 for name in weights)
            report = dict(status="PASS", model=MODEL_ID, revision=REVISION,
                          transformers=transformers.__version__, torch=torch.__version__,
                          gpu=torch.cuda.get_device_name(), kernel_sha256=kernel_hash,
                          backend=model.config._attn_implementation, layers=len(model.model.layers),
                          replaced_layer=0, group=128, dtype="float16", prompt_tokens=prompt.shape[-1],
                          fp16_continuation=tokenizer.decode(continuation[0], skip_special_tokens=True),
                          fp16_load_memory=load_memory, fp16_trace_memory=fp16_memory,
                          block_fp16_weight_bytes=bytes_fp16, block_packed_weight_bytes=bytes_packed,
                          block=block, logits=logits)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("block", "logits")}, indent=2))


if __name__ == "__main__":
    main()
