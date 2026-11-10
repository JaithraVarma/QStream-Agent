# Qwen static-cache decode with CUDA Graph replay

PASS. The pinned Qwen3 model used the selected packed kernel in all 28
FFNs, FP16 activations, and SDPA. Batch size was 1, prompt length was 128,
and each trial ran 16 fixed cached decode steps. The static cache had
capacity for 144 tokens. Both paths used the same cache storage and
attention mask. Prefill and capture were outside decode timing.

| Decode metric | Static eager | CUDA Graph replay |
|---|---:|---:|
| Median synchronized wall ms per step | 37.33 | 7.65 |
| Median CUDA-event ms per step | 37.33 | 7.65 |
| Wall throughput at batch 1, tokens/s | 26.8 | 130.7 |

Replay was faster in all ten pairs, with paired wall-time speedups of
4.71–5.38×. The ratio of medians was 4.88×. Each mode staged the current
token and position in the timed loop. Prefill reset the static cache
before every trial, preserving all key and value addresses.

Across two token sequences, all 34 checked graph outputs matched
eager static logits exactly. Compared with the dynamic-cache packed
reference, the largest relative L2 error was 0.2901%, below the 2% gate,
and all 34 top tokens matched. Live PyTorch allocation increased by
8.42 MiB after capture. The process peak reserved memory counter did not
increase across those snapshots, but it includes earlier setup and is
not an isolated measurement of the graph pool.

Graph replay reduced full-step time without changing the packed GEMM.
This comparison does not isolate Python execution, driver submission,
and launch scheduling from one another. It also does not measure
prefill speedup, arbitrary cache capacities, or benefit at other batch
sizes. Use this graph baseline before evaluating candidate kernels.

[Raw results](summary.json) · [Commands](../REPRODUCE.md)
