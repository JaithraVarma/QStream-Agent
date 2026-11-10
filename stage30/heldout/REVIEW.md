# Held-out validation of the Kimi decode launch

The selected Kimi proposal was fixed before validation. It uses a 32-wide output tile, a 64-wide reduction tile, four warps, and two stages. It is compared with the manual 32-output, 32-reduction, four-warp, one-stage launch on the same packed Qwen3-0.6B and RTX 3060 Laptop GPU. The dispatch is restricted to the batch-1 down projection. Prefill and the gate and up projections use the same kernels in both modes.

Timing uses 64-, 256-, and 512-token prompts and the same 16 fixed continuation tokens for each paired comparison. Ten pairs per context share a static cache and staging buffers, with five trials in each execution order. Prefill and cache reset run outside the timed interval. The historical 128-token context was the search workload and is excluded from held-out timing. Each context is checked for numerical correctness before its timing trials. The provisional screen remains a paired latency reduction of at least 1% and eight wins out of ten.

The text check uses eight short locally authored passages, with the first 32 tokens as context and the next 32 as fixed prediction targets. The manual and Kimi paths replay decode while consuming the same ground-truth tokens. Stock FP16 scores the same targets in a teacher-forced full forward as context for the quantized model results. The packed launch passes if the maximum relative logit L2 is at most 2%, top predictions agree on at least 99% of scored positions, and absolute mean next-token loss difference is at most 0.01 nats. Perplexity is the exponential of average next-token loss. These passages are a regression check on this launch change. They do not establish general model quality.

## Measured results

| Prompt tokens | Manual ms/token | Kimi ms/token | Paired latency reduction | Kimi wins |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 6.941 | 6.809 | 1.90% | 10/10 |
| 256 | 7.676 | 7.541 | 1.75% | 10/10 |
| 512 | 8.601 | 8.456 | 1.63% | 10/10 |

All three contexts passed correctness and cleared the performance screen. Across the 256 text targets, the two packed variants agreed on every top prediction. Maximum relative logit L2 was 0.437%, and the Kimi launch added 0.000520 nats to average next-token loss.

| Text score | Stock FP16 | Manual INT4 | Kimi INT4 |
| --- | ---: | ---: | ---: |
| Average next-token loss, nats | 3.98838 | 4.19160 | 4.19212 |
| Perplexity | 53.97 | 66.13 | 66.16 |

The stock score includes the difference between full-forward and cached execution in addition to the change in weight precision. It is a separate reference. The two INT4 paths use the same packed weights and cached graph procedure, so their difference evaluates the launch change directly.

Keep the selected Kimi launch for the batch-1 decode demonstration on this GPU. The measured benefit is relative to the manual packed launch and applies to the tested graph workloads. This validation did not test whether it changes batch-4 performance or prefill latency. See the [raw results](summary.json) and [command log](../REPRODUCE.md).
