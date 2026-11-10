"""Profile one selected W4A16 down projection from real cached Qwen decode."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import StaticCache

from qstream import dequantize
from .all_layers import configure_mlp
from .benchmark import save_report, workload
from .graph_decode import forward, mask_for, prefill
from .smoke import load_model


HERE = Path(__file__).resolve().parent
OUT = HERE / "down-profile" / "target.json"


def capture_down(model, prompt, token, cache, positions):
    prompt_length = prompt.shape[1]
    prefill(model, cache, prompt, positions[:prompt_length], positions)
    position = positions[prompt_length:prompt_length + 1]
    mlp = model.model.layers[0].mlp
    gemm = mlp.gemm
    captured = []

    def probe(x, weights):
        if weights.shape == (1024, 3072):
            captured.append((x, weights))
        return gemm(x, weights)

    mlp.gemm = probe
    try:
        logits = forward(model, token, cache, position, mask_for(position, positions)).logits
    finally:
        mlp.gemm = gemm

    assert cache.get_seq_length() == prompt_length + 1
    assert torch.isfinite(logits).all().item()
    [(activation, weights)] = captured
    assert activation.shape == (1, 3072) and weights.group_size == 128
    return gemm, activation, weights


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    graph = json.loads((HERE / "graph-profile/summary.json").read_text())
    assert graph["status"] == "PASS" and graph["batch"] == 1
    model, tokenizer = load_model()
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            weight_hash, kernel_hash = configure_mlp(model, "packed")
            assert weight_hash == graph["quantized_sha256"]
            assert kernel_hash == graph["kernel_sha256"]
            prompt, continuation, input_hash = workload(tokenizer, 1, graph["prompt_tokens"])
            assert input_hash == graph["input_sha256"]
            capacity = prompt.shape[1] + len(continuation)
            assert capacity == graph["cache_capacity"]
            cache = StaticCache(config=model.config, max_cache_len=capacity)
            positions = torch.arange(capacity, device="cuda")
            gemm, x, weights = capture_down(model, prompt, continuation[0], cache, positions)

            expected = F.linear(x, dequantize(weights))
            actual, compiled = gemm(x, weights, return_compiled=True)
            torch.testing.assert_close(actual, expected, atol=.005, rtol=.005)
            for _ in range(50):
                gemm(x, weights)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                profiled = gemm(x, weights)
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            torch.testing.assert_close(profiled, expected, atol=.005, rtol=.005)
            report = dict(
                status="PASS", gpu=torch.cuda.get_device_name(), layer=0,
                decode_position=prompt.shape[1], cache_capacity=capacity,
                shape=[x.shape[0], *weights.shape, weights.group_size],
                kernel_sha256=kernel_hash, quantized_sha256=weight_hash, input_sha256=input_hash,
                activation_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
                max_abs_error=(profiled.float() - expected.float()).abs().max().item(),
                atol=.005, rtol=.005, warmup_launches=50, registers=compiled.n_regs,
                spills=compiled.n_spills, shared_bytes=compiled.metadata.shared,
            )
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
    save_report(OUT, report)
    print(f"PASS: Qwen layer 0 down projection {report['shape']}")


if __name__ == "__main__":
    main()
