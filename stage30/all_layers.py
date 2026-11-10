"""Compare packed and reconstructed FFNs in all Qwen layers."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .model_files import MODEL_ID, REVISION
from .qwen_mlp import PackedMLP, ReconstructedMLP, load_selected_kernel, pack_mlp
from .smoke import load_model, peak, trace


OUT = Path(__file__).resolve().parent / "all-layers"
VARIANTS = ("fp16", "reconstructed", "packed")
MAX_RELATIVE_L2 = .02
STEPS = 8


def configure_mlp(model, variant):
    if variant == "fp16":
        return None, None
    if variant not in ("reconstructed", "packed"):
        raise ValueError(variant)
    gemm, kernel_hash = load_selected_kernel()
    packed_hash = hashlib.sha256()
    for layer in model.model.layers:
        weights = pack_mlp(layer.mlp)
        for weight in weights.values():
            packed_hash.update(weight.packed.cpu().contiguous().numpy().tobytes())
            packed_hash.update(weight.scales.cpu().contiguous().numpy().tobytes())
        layer.mlp = ReconstructedMLP(weights) if variant == "reconstructed" else PackedMLP(weights, gemm)
        del weights
    return packed_hash.hexdigest(), kernel_hash


def resident_bytes(modules):
    storages = {}
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_cuda and tensor.numel():
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def fixed_tokens(tokenizer, text, length):
    values = tokenizer.encode(text, add_special_tokens=False)
    assert values
    return (values * ((length + len(values) - 1) // len(values)))[:length]


def workloads(tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )
    continuation = torch.tensor([fixed_tokens(tokenizer, "2 + 2 = 4. The answer is four.", STEPS)])
    yield "chat", prompt, continuation
    passages = (
        "The GPU reads the packed weights and reconstructs their values. ",
        "The transformer caches keys and values from earlier tokens. ",
    )
    batch = torch.tensor([fixed_tokens(tokenizer, text, 128 + STEPS) for text in passages])
    yield "batch2", batch[:, :128].contiguous(), batch[:, 128:].contiguous()


def run(variant):
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / f"{variant}.json"
    tensor_path = OUT / f"{variant}.safetensors"
    if report_path.exists() or tensor_path.exists():
        raise FileExistsError(report_path)
    model, tokenizer = load_model()
    assert len(model.model.layers) == 28
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tensors = {}
    try:
        with torch.inference_mode():
            quantized_hash, kernel_hash = configure_mlp(model, variant)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report = dict(status="PASS", variant=variant, model=MODEL_ID, revision=REVISION,
                          torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                          layers=len(model.model.layers), group=128, dtype="float16",
                          backend=model.config._attn_implementation,
                          quantized_sha256=quantized_hash, kernel_sha256=kernel_hash,
                          model_tensor_bytes=resident_bytes([model]),
                          ffn_tensor_bytes=resident_bytes([layer.mlp for layer in model.model.layers]),
                          resident=peak(), cases={})
            for name, prompt, continuation in workloads(tokenizer):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logits, _ = trace(model, prompt.cuda(), STEPS, continuation.cuda())
                report["cases"][name] = dict(batch=prompt.shape[0], prompt_tokens=prompt.shape[1],
                                              decode_steps=STEPS-1, memory=peak())
                for step, row in enumerate(logits):
                    assert torch.isfinite(row).all()
                    tensors[f"{name}.step{step}"] = row.cpu().contiguous()
                tensors[f"{name}.prompt"] = prompt.contiguous()
                tensors[f"{name}.continuation"] = continuation.contiguous()
                del logits
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
    save_file(tensors, tensor_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def compare_rows(actual, expected, labels):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    relative = delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)
    loss_actual = F.cross_entropy(actual, labels, reduction="none")
    loss_expected = F.cross_entropy(expected, labels, reduction="none")
    return dict(max_abs=delta.abs().max().item(), max_row_relative_l2=relative.max().item(),
                top1_matches=int((actual.argmax(-1) == expected.argmax(-1)).sum()),
                rows=actual.shape[0], mean_next_token_nll_delta=(loss_actual-loss_expected).mean().item())


def summarize():
    reports = {name: json.loads((OUT / f"{name}.json").read_text()) for name in VARIANTS}
    assert all(row["status"] == "PASS" for row in reports.values())
    for key in ("model", "revision", "torch", "gpu", "layers", "group", "dtype", "backend"):
        assert len({row[key] for row in reports.values()}) == 1, key
    assert reports["packed"]["quantized_sha256"] == reports["reconstructed"]["quantized_sha256"]
    assert reports["packed"]["kernel_sha256"] == reports["reconstructed"]["kernel_sha256"]
    values = {name: load_file(OUT / f"{name}.safetensors") for name in VARIANTS}
    rows = []
    for case in reports["fp16"]["cases"]:
        for field in ("prompt", "continuation"):
            key = f"{case}.{field}"
            assert all(torch.equal(value[key], values["fp16"][key]) for value in values.values())
        for step in range(STEPS):
            key = f"{case}.step{step}"
            labels = values["fp16"][f"{case}.continuation"][:, step]
            rows.append(dict(case=case, step=step, phase="prefill" if step == 0 else "decode",
                packed_vs_reconstructed=compare_rows(values["packed"][key], values["reconstructed"][key], labels),
                reconstructed_vs_fp16=compare_rows(values["reconstructed"][key], values["fp16"][key], labels)))
    largest = max(row["packed_vs_reconstructed"]["max_row_relative_l2"] for row in rows)
    status = "PASS" if largest <= MAX_RELATIVE_L2 else "FAIL"
    report = dict(status=status, max_row_relative_l2=largest, gate=MAX_RELATIVE_L2,
                  variants=reports, comparisons=rows)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=status, max_row_relative_l2=largest,
                          top1_matches=sum(row["packed_vs_reconstructed"]["top1_matches"] for row in rows),
                          rows=sum(row["packed_vs_reconstructed"]["rows"] for row in rows),
                          resident_mib={name: r["model_tensor_bytes"]/2**20 for name, r in reports.items()},
                          ffn_mib={name: r["ffn_tensor_bytes"]/2**20 for name, r in reports.items()}), indent=2))
    if status != "PASS":
        raise AssertionError("Full-model relative logit L2 exceeded the gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--variant", choices=VARIANTS)
    mode.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.variant)
