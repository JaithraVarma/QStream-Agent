# Qwen batch-1 down projection

The probe reused the pinned packed Qwen model, the static KV cache of capacity 144, the 128-token prompt, and the first fixed continuation token from the graph replay workload. It captured layer 0's down-projection activation at decode position 128. Model weights, selected kernel, and input token hashes matched the graph run. The original layer callable was restored immediately after capture.

The target was `M=1, N=1024, K=3072, G=128`. Both the unprofiled and profiled outputs passed the PyTorch linear comparison with reconstructed INT4 weights at `atol=rtol=0.005`. Maximum absolute error was 0.0001220703125. Nsight collected one selected launch in 38 profiling passes after 50 warmup launches. Clocks and caches used their normal controls. The observed 47.84 µs is an isolated, profiled duration and should be kept separate from graph replay latency. See the [correctness receipt](target.json) and [collection commands](../REPRODUCE.md).

| Observation | Value |
| --- | ---: |
| Blocks / SMs | 16 / 30 |
| Threads / block | 128 |
| Theoretical / achieved occupancy | 83.33% / 8.32% |
| Active / eligible warps per scheduler | 1.00 / 0.19 |
| Scheduler cycles with no eligible warp | 81.50% |
| Long scoreboard / barrier / short scoreboard cycles per issued instruction | 1.49 / 0.85 / 0.65 |
| LG throttle / MIO throttle cycles per issued instruction | 0 / 0 |
| L2 hit rate / DRAM throughput relative to peak | 98.88% / 0.49% |
| Registers / spills / kernel shared bytes per block | 48 per thread / 0 / 2,048 |

Each block produces 64 output features for a tile of 16 input rows. At batch-1 decode only one input row is valid, so the grid contains one row tile and `1024 / 64 = 16` output tiles. That gives at most 16 SMs work, with four warps per block. The theoretical occupancy reflects what resources could fit with a larger grid. In this launch, eligible warps were scarce, and long scoreboard was the largest recorded stall reason. Long scoreboard tracks dependencies on L1/TEX operations. Low DRAM utilization and the high L2 hit rate characterize this warmed launch. They leave the in-model cache regime to the graph experiment. Nsight also flagged excessive sectors for global accesses. That aggregate does not identify which operand should be redesigned. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

The first hypothesis is to reduce the output tile width from 64 to 32, doubling the grid to 32 blocks without introducing a second reduction launch. That could put work on all 30 SMs, at the cost of potentially more activation reads and different memory access efficiency. Because `G=128` is divisible by the current reduction tile width of 32, the existing kernel already uses the aligned one-scale path. A candidate should be checked against the current kernel on real inputs and retained only if it helps paired full-model graph replay. No Inco request was made for this profiling step.
