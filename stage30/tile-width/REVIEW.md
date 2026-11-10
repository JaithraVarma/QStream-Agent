# Manual output tile baseline for Qwen decode

The earlier Nsight profile showed that the batch-1 down projection launched 16 thread blocks on a 30-SM RTX 3060 Laptop GPU. It also found few eligible warps. The manual experiment changes the output tile width from 64 to 32 for the exact `M=1, N=1024, K=3072, G=128` case in all 28 Qwen layers. This doubles the grid to 32 blocks while retaining the selected kernel's source, row and reduction tiles, warps, stages, weight format, and arithmetic. Gate, up, and prefill use the original launch. The narrower variant is currently local to the experiment.

Both widths passed the isolated check against PyTorch using reconstructed INT4 weights and an activation taken from real Qwen decode. The maximum absolute projection error was 0.0001220703125 for both at `atol=rtol=0.005`. All 34 full-model logit outputs matched exactly across the two graphs on the benchmark and reversed token sequences. Both graphs reused the same static cache and staging buffers. The timing harness reset and refilled the cache before each measurement, then measured the same 16 fixed decode steps. Five pairs ran width 64 first and five ran width 32 first, in shuffled order.

| Result | 64 outputs per block | 32 outputs per block |
| --- | ---: | ---: |
| Thread blocks | 16 | 32 |
| Registers per thread | 48 | 40 |
| Kernel shared memory per block | 2,048 B | 1,024 B |
| Register spills | 0 | 0 |
| Median wall ms per decode step | 7.662 | 7.261 |
| Median CUDA event ms per decode step | 7.654 | 7.254 |

Width 32 won all ten wall-clock pairs. Its median paired wall speedup was 1.0552×, or about 5.2% lower latency. Increasing the grid gives the GPU enough blocks to distribute work across all 30 SMs. The new launch also used fewer registers and less shared memory per block. The benchmark did not measure the new variant's scheduler stalls or separate the effects of grid size, memory access, and compiler resource use. This result applies to batch-1 graph decode after a 128-token prompt on this GPU. Other workload shapes remain to be tested.

Use this manual variant as the comparison for the profiling-guided Inco search. The target for an agent candidate is full-model decode latency after correctness checks. See the [raw receipt](summary.json) and [command log](../REPRODUCE.md).
