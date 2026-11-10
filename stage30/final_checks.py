"""Run matched FP16 and INT4 graphs and a balanced tile-by-stage experiment."""
import argparse
from contextlib import nullcontext
from functools import partial
import gc
import json
import os
from pathlib import Path
import random
from statistics import median

import torch

from .all_layers import configure_mlp, resident_bytes
from .benchmark import events, save_report, workload
from .decode_search import OPTIONS, read_json
from .graph_decode import cache_pointers, checked_sequence, measure, prefill
from .qwen_mlp import load_selected_kernel_module
from .smoke import error, load_model, peak
from .telemetry import snapshot
from .tile_width import MANUAL, capture, configured_down, decode_buffers, prepare_graphs, use_gemm


HERE = Path(__file__).resolve().parent
OUT = HERE / "final-checks"
CONTEXTS = (128, 512)


def clean_allocator():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def graph_runner(model, prompt, continuation, gemm=None):
    cache, positions, stage, eager = decode_buffers(model, prompt, continuation)
    length = prompt.shape[1]
    prefill(model, cache, prompt, positions[:length], positions)
    with use_gemm(model, gemm) if gemm is not None else nullcontext():
        replay = capture(eager, stage, continuation[0], positions[length:length+1])
    return cache, positions, stage, eager, replay


def measure_graph(model, runner, prompt, continuation, timers):
    cache, positions, stage, _, replay = runner
    return measure(model, cache, prompt, continuation, positions, positions, stage, replay, timers)


def check_graph(model, runner, prompt, continuation, gemm):
    cache, positions, stage, eager, replay = runner
    pointers = cache_pointers(cache)
    with use_gemm(model, gemm) if gemm is not None else nullcontext():
        direct = checked_sequence(model, cache, prompt, continuation, positions, positions, stage, eager, pointers)
    captured = checked_sequence(model, cache, prompt, continuation, positions, positions, stage, replay, pointers)
    for actual, expected in zip(captured, direct):
        torch.testing.assert_close(actual, expected, atol=.005, rtol=.005)
    return captured, max(error(a, b)["relative_l2"] for a, b in zip(captured, direct))


def time_orders(calls, orders, steps):
    rows = [dict(order=order, samples={name: calls[name]() for name in order}) for order in orders]
    medians = {name: {key: median(row["samples"][name][key] for row in rows) / steps
                      for key in ("event_ms", "wall_ms")} for name in calls}
    return rows, medians


def paired(rows, baseline, trial):
    ratios = [row["samples"][trial]["wall_ms"] / row["samples"][baseline]["wall_ms"] for row in rows]
    return dict(wall_change_pct=(median(ratios) - 1) * 100, trial_wins=sum(ratio < 1 for ratio in ratios))


def print_precision_summary(report):
    print(f"QStream-Agent | {report['gpu']} | batch 1 CUDA Graph decode")
    print(f"{'Prompt':>8} {'FP16 ms/token':>15} {'INT4 ms/token':>15} "
          f"{'INT4 latency':>14} {'FP16 peak MiB':>15} {'INT4 peak MiB':>15}")
    for case in report["timing"]:
        length = case["prompt_tokens"]
        times = case["medians_ms_per_step"]
        fp16, int4 = (times[name]["wall_ms"] for name in ("fp16", "kimi_int4"))
        peaks = [report["memory"][name]["cases"][str(length)]["peak_allocated_bytes"] / 2**20
                 for name in ("fp16", "kimi_int4")]
        change = case["int4_vs_fp16"]["wall_change_pct"]
        print(f"{length:8d} {fp16:15.3f} {int4:15.3f} {change:+13.2f}% {peaks[0]:15.2f} {peaks[1]:15.2f}")


def memory_cases(model, tokenizer, gemm=None):
    rows = {}
    timers = events()
    for length in CONTEXTS:
        clean_allocator()
        prompt, continuation, digest = workload(tokenizer, 1, length)
        runner = graph_runner(model, prompt, continuation, gemm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measure_graph(model, runner, prompt, continuation, timers)
        rows[str(length)] = dict(input_sha256=digest, **peak())
        del runner, prompt, continuation
    return dict(model_storage_bytes=resident_bytes((model,)), cases=rows)


def pack_verified(model, kernel_hash, heldout):
    weights, source = configure_mlp(model, "packed")
    assert source == kernel_hash == heldout["kernel_sha256"]
    assert weights == heldout["quantized_sha256"]
    return weights


def precision(module, kernel_hash, heldout, config, report):
    fp16, tokenizer = load_model()
    memory = {"fp16": memory_cases(fp16, tokenizer)}
    clean_allocator()
    packed, _ = load_model()
    report["quantized_sha256"] = pack_verified(packed, kernel_hash, heldout)
    gemm = partial(configured_down, module, config)
    variants = {"fp16": (fp16, None), "kimi_int4": (packed, gemm)}
    report.update(memory=memory, timing=[])
    for length in CONTEXTS:
        prompt, continuation, digest = workload(tokenizer, 1, length)
        runners = {name: graph_runner(model, prompt, continuation, fn) for name, (model, fn) in variants.items()}
        checks = {name: check_graph(model, runners[name], prompt, continuation, fn)
                  for name, (model, fn) in variants.items()}
        cross = [error(int4, stock) for int4, stock in zip(checks["kimi_int4"][0], checks["fp16"][0])]
        same_precision = {name: value[1] for name, value in checks.items()}
        del checks
        timers = events()
        calls = {name: partial(measure_graph, variants[name][0], runner, prompt, continuation, timers)
                 for name, runner in runners.items()}
        orders = [list(calls), list(reversed(calls))] * 5
        random.Random(20261109 + length).shuffle(orders)
        rows, medians = time_orders(calls, orders, len(continuation))
        case = dict(prompt_tokens=length, input_sha256=digest, medians_ms_per_step=medians,
                    int4_vs_fp16=paired(rows, "fp16", "kimi_int4"), samples=rows,
                    graph_vs_own_eager_max_relative_l2=same_precision,
                    int4_vs_fp16_logit_top1=sum(row["top1_match"] for row in cross),
                    logits_compared=len(cross), int4_vs_fp16_max_relative_l2=max(row["relative_l2"] for row in cross))
        report["timing"].append(case)
        print(json.dumps(dict(prompt_tokens=length, medians=medians, comparison=case["int4_vs_fp16"])), flush=True)
        del runners, calls
        clean_allocator()
    del variants, fp16
    clean_allocator()
    memory["kimi_int4"] = memory_cases(packed, tokenizer, gemm)
    for case in report["timing"]:
        for variant in memory.values():
            assert variant["cases"][str(case["prompt_tokens"])]["input_sha256"] == case["input_sha256"]


def factorial(module, kernel_hash, heldout, config, report):
    configs = {
        "k32_s1": MANUAL, "k32_s2": (32, 32, 4, 2),
        "k64_s1": (32, 64, 4, 1), "k64_s2": config,
    }
    assert config == (32, 64, 4, 2)
    model, tokenizer = load_model()
    report["quantized_sha256"] = pack_verified(model, kernel_hash, heldout)
    prompt, continuation, digest = workload(tokenizer, 1, 128)
    cache, positions, pointers, stage, methods, projection = prepare_graphs(
        model, module, configs, prompt, continuation
    )
    logits = {name: checked_sequence(model, cache, prompt, continuation, positions, positions, stage,
                                     replay, pointers) for name, replay in methods.items()}
    checks = {name: [error(a, b) for a, b in zip(values, logits["k32_s1"])] for name, values in logits.items()}
    assert all(row["relative_l2"] <= .02 and row["top1_match"] for values in checks.values() for row in values)
    del logits
    timers = events()
    calls = {name: partial(measure, model, cache, prompt, continuation, positions, positions, stage, replay, timers)
             for name, replay in methods.items()}
    rng = random.Random(20261109)
    orders = []
    for _ in range(3):
        block = list(calls)
        rng.shuffle(block)
        orders.extend(block[i:] + block[:i] for i in range(len(block)))
    rng.shuffle(orders)
    rows, medians = time_orders(calls, orders, len(continuation))
    assert cache_pointers(cache) == pointers and cache.get_seq_length() == positions.numel()
    effects = {f"k{k}_s2_vs_s1": paired(rows, f"k{k}_s1", f"k{k}_s2") for k in (32, 64)}
    report.update(configurations=configs, input_sha256=digest, prompt_tokens=128, projection=projection,
                  correctness=checks, medians_ms_per_step=medians, stage_effects=effects, samples=rows)
    print(json.dumps(dict(medians=medians, stage_effects=effects)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("precision", "factorial"))
    parser.add_argument("--output", type=Path, help="Write a new receipt at this path")
    args = parser.parse_args()
    output = args.output or OUT / f"{args.experiment}.json"
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if os.environ.get("INCO_API_KEY"):
        raise ValueError("Unset INCO_API_KEY before the offline run")
    heldout = read_json(HERE / "heldout/summary.json")
    assert heldout["status"] == "PASS" and heldout["quality"]["passed"]
    config = OPTIONS["n32-k64-w4-s2"]
    assert heldout["configurations"]["challenger"] == list(config)
    module, kernel_hash = load_selected_kernel_module()
    report = dict(status="RUNNING", gpu=torch.cuda.get_device_name(), kernel_sha256=kernel_hash,
                  configuration=config, telemetry=[snapshot("start")])
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            experiment = {"precision": precision, "factorial": factorial}[args.experiment]
            experiment(module, kernel_hash, heldout, config, report)
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAILED"
        raise
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        report["telemetry"].append(snapshot("end"))
        save_report(output, report)
    if args.experiment == "precision":
        print_precision_summary(report)
    print(f"Receipt: {output}")


if __name__ == "__main__":
    main()
