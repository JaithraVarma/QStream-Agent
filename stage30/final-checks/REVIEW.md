# Final Qwen CUDA Graph checks

## Stock FP16 and selected packed INT4

The comparison uses batch 1, prompt lengths 128 and 512, and 16 fixed cached decode steps per trial on the RTX 3060 Laptop GPU. Both modes use the same pinned Qwen3-0.6B checkpoint, token IDs, static KV cache design, attention backend, and graph replay timing procedure. The INT4 graph uses the validated Kimi launch for the single-row down projections. Each graph must match its own static-cache eager result before timing. Cross-precision logit comparisons are descriptive because the models use different weight representations.

Ten same-session latency pairs at each context use five trials in each execution order. The GPU event and wall interval includes token and position staging and decode replay. Prefill and graph capture are outside that interval. Both models are resident during the paired latency trials. Peak allocated memory is measured separately with one model resident at a time, after graph capture. It includes model storage, the static cache, captured graph allocations, and the peak during one prefill plus 16 replays. Model loading, INT4 conversion, graph capture itself, and allocator reservations are outside the reported peak.

| Prompt tokens | FP16 ms/token | Kimi INT4 ms/token | Paired latency reduction | FP16 peak allocated MiB | INT4 peak allocated MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.340 | 7.137 | 2.76% | 1,180.65 | 860.66 |
| 512 | 8.519 | 8.449 | 0.87% | 1,259.54 | 948.08 |

INT4 won all ten pairs at both contexts. Unique model storage decreased from 1,136.88 to 766.75 MiB. The measured peak fell by 319.99 MiB at 128 and 311.46 MiB at 512. Each graph exactly matched its own eager reference. FP16 and INT4 agreed on all 34 top predictions in this timing workload, while maximum relative logit L2 across precisions was about 39.5%. The independent [text regression check](../heldout/REVIEW.md) also found higher next-token loss for INT4. See the [precision receipt](precision.json).

## Reduction width and pipeline stages

The four configurations use the same packed model, token sequence, cache and staging buffers, a 32-output tile, and four warps. They vary reduction width (32 or 64) and stage count (one or two). Projection and full-model correctness checks gate timing. The 128-token graph workload is run in 12 blocks, with each configuration timed once in each block and equally often in each timing position overall. Comparing stages within each reduction width isolates that launch-setting change in this workload. It does not identify the hardware stall mechanism.

| Reduction width | Stages | Median wall ms/token | Registers per thread | Shared bytes per block |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 7.257 | 40 | 1,024 |
| 32 | 2 | 7.446 | 46 | 1,664 |
| 64 | 1 | 7.198 | 51 | 2,048 |
| 64 | 2 | 7.120 | 53 | 3,200 |

At width 32, adding the second stage increased paired latency by 2.61% and lost all 12 rounds. At width 64, it reduced paired latency by 1.11% and won all 12 rounds. The 64-wide, two-stage launch reduced paired latency by 1.89% versus the manual 32-wide, one-stage launch, winning all 12 rounds. Every launch used 32 thread blocks, passed the real-activation projection check, and retained all 17 top predictions. The maximum relative logit L2 versus the manual launch was 0.311%. The resource footprint grew with the extra stage at both widths. The opposite timing effects show that the choice depends on reduction width for this workload. No new Nsight counters were collected to attribute the difference to specific pipeline behavior. See the [four-way receipt](factorial.json) and [command log](../REPRODUCE.md).
