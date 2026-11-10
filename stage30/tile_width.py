"""Check and time a narrower output tile in packed Qwen batch-1 decode."""
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
import random
from statistics import median

import torch
import torch.nn.functional as F
from transformers import StaticCache
import triton

from qstream import dequantize
from .all_layers import configure_mlp
from .benchmark import events, save_report, workload
from .graph_decode import cache_pointers, checked_sequence, forward, mask_for, measure, prefill
from .profile_qwen_down import capture_down
from .qwen_mlp import load_selected_kernel_module
from .smoke import error, load_model
from .telemetry import snapshot


HERE = Path(__file__).resolve().parent
OUT = HERE / "tile-width" / "summary.json"
ORIGINAL = (64, 32, 4, 1)
MANUAL = (32, 32, 4, 1)
PAIRS = 10


def configured_down(module, config, x, weight, *, return_compiled=False):
    if (config == ORIGINAL or x.shape != (1, 3072) or weight.shape != (1024, 3072)
            or weight.group_size != 128):
        return module.w4a16_gemm(x, weight, return_compiled=return_compiled)
    bn, bk, warps, stages = config
    m, n, k, group = module._validate(x, weight)
    with torch.cuda.device(x.device):
        y = torch.empty((m, n), device=x.device, dtype=torch.float16)
        compiled = module._w4a16_kernel[
            (triton.cdiv(m, module.BLOCK_M), triton.cdiv(n, bn))
        ](
            x, weight.packed, weight.scales, y,
            m, n, k, group,
            *x.stride(), *weight.packed.stride(), *weight.scales.stride(), *y.stride(),
            module.BLOCK_M, bn, bk,
            num_warps=warps, num_stages=stages,
        )
    return (y, compiled) if return_compiled else y


@contextmanager
def use_gemm(model, gemm):
    adapters = [layer.mlp for layer in model.model.layers]
    original = [adapter.gemm for adapter in adapters]
    try:
        for adapter in adapters:
            adapter.gemm = gemm
        yield
    finally:
        for adapter, fn in zip(adapters, original):
            adapter.gemm = fn


def capture(execute, stage_inputs, token, position):
    stage_inputs(token, position)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            execute()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = execute()

    def replay():
        graph.replay()
        return result

    return replay


def decode_buffers(model, prompt, continuation):
    prompt_length = prompt.shape[1]
    capacity = prompt_length + len(continuation)
    cache = StaticCache(config=model.config, max_cache_len=capacity)
    positions = torch.arange(capacity, device="cuda")
    token_buffer = torch.empty_like(continuation[0])
    position_buffer = torch.empty_like(positions[:1])

    def stage_inputs(token, position):
        token_buffer.copy_(token)
        position_buffer.copy_(position)

    def eager():
        return forward(model, token_buffer, cache, position_buffer,
                       mask_for(position_buffer, positions)).logits
    return cache, positions, stage_inputs, eager


def prepare_graphs(model, module, configs, prompt, continuation):
    prompt_length = prompt.shape[1]
    cache, positions, stage_inputs, eager = decode_buffers(model, prompt, continuation)
    first_position = positions[prompt_length:prompt_length + 1]
    _, x, weights = capture_down(model, prompt, continuation[0], cache, positions)
    reference = F.linear(x, dequantize(weights))
    projection = {}
    methods = {}
    pointers = cache_pointers(cache)
    for name, config in configs.items():
        result, compiled = configured_down(module, config, x, weights, return_compiled=True)
        torch.testing.assert_close(result, reference, atol=.005, rtol=.005)
        projection[name] = dict(
            blocks=triton.cdiv(weights.shape[0], config[0]), max_abs_error=error(result, reference)["max_abs"],
            registers=compiled.n_regs, spills=compiled.n_spills, shared_bytes=compiled.metadata.shared,
        )
        prefill(model, cache, prompt, positions[:prompt_length], positions)
        with use_gemm(model, partial(configured_down, module, config)):
            methods[name] = capture(eager, stage_inputs, continuation[0], first_position)
    return cache, positions, pointers, stage_inputs, methods, projection


def compare_workload(model, module, report, configs, prompt, continuation):
    cache, positions, pointers, stage_inputs, methods, projection = prepare_graphs(
        model, module, configs, prompt, continuation
    )

    baseline, challenger = methods
    checks = []
    alternate = (prompt.flip(-1).contiguous(), list(reversed(continuation)))
    for label, (prefix, tokens) in (("benchmark", (prompt, continuation)), ("alternate", alternate)):
        results = {
            name: checked_sequence(model, cache, prefix, tokens, positions, positions,
                                   stage_inputs, replay, pointers)
            for name, replay in methods.items()
        }
        for step, (expected, actual) in enumerate(zip(results[baseline], results[challenger])):
            metrics = error(actual, expected)
            assert metrics["relative_l2"] <= .02 and metrics["top1_match"], (label, step, metrics)
            checks.append(dict(sequence=label, step=step, **metrics))
    report.update(cache_capacity=positions.numel(), decode_steps=len(continuation), projection=projection,
                  correctness=checks)
    print(f"Correctness PASS: {len(checks)} model logits comparisons", flush=True)

    timers = events()
    orders = [list(methods), list(reversed(methods))] * (PAIRS // 2)
    random.Random(20261109).shuffle(orders)
    rows = []
    for order in orders:
        samples = {
            name: measure(model, cache, prompt, continuation, positions, positions,
                          stage_inputs, methods[name], timers)
            for name in order
        }
        rows.append(dict(order=order, samples=samples))
    assert cache_pointers(cache) == pointers and cache.get_seq_length() == positions.numel()
    medians = {
        name: {key: median(row["samples"][name][key] for row in rows) / len(continuation)
               for key in ("event_ms", "wall_ms")}
        for name in methods
    }
    ratios = {
        key: median(row["samples"][baseline][key] / row["samples"][challenger][key] for row in rows)
        for key in ("event_ms", "wall_ms")
    }
    wins = sum(row["samples"][challenger]["wall_ms"] < row["samples"][baseline]["wall_ms"] for row in rows)
    report.update(status="PASS", samples=rows, medians_ms_per_step=medians,
                  median_paired_speedup=ratios, challenger_wall_wins=wins)


def compare(model, tokenizer, module, kernel_hash, graph_report, report, configs):
    weight_hash, selected_hash = configure_mlp(model, "packed")
    assert selected_hash == kernel_hash == graph_report["kernel_sha256"]
    assert weight_hash == graph_report["quantized_sha256"]
    prompt, continuation, input_hash = workload(tokenizer, 1, graph_report["prompt_tokens"])
    assert input_hash == graph_report["input_sha256"]
    assert prompt.shape[1] + len(continuation) == graph_report["cache_capacity"]
    report.update(kernel_sha256=kernel_hash, quantized_sha256=weight_hash, input_sha256=input_hash)
    compare_workload(model, module, report, configs, prompt, continuation)


def run(output, configs):
    if output.exists():
        raise FileExistsError(output)
    graph_report = json.loads((HERE / "graph-profile/summary.json").read_text())
    assert graph_report["status"] == "PASS" and graph_report["batch"] == 1
    module, kernel_hash = load_selected_kernel_module()
    assert (module.BLOCK_N, module.BLOCK_K, module.NUM_WARPS, module.NUM_STAGES) == ORIGINAL
    model, tokenizer = load_model()
    report = dict(status="RUNNING", gpu=torch.cuda.get_device_name(), pairs=PAIRS, configurations=configs,
                  experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  telemetry=[snapshot("start")])
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            compare(model, tokenizer, module, kernel_hash, graph_report, report, configs)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        report["telemetry"].append(snapshot("end"))
        if report["status"] == "PASS":
            save_report(output, report)
    print(json.dumps({key: report[key] for key in ("status", "medians_ms_per_step",
                                                   "median_paired_speedup", "challenger_wall_wins")},
                     indent=2))
    return report


def main():
    run(OUT, {"bn64": ORIGINAL, "bn32": MANUAL})


if __name__ == "__main__":
    main()
