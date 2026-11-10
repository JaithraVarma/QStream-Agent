# Bounded Kimi K3 launch search

The search targets the down projection in Qwen3-0.6B batch-1 decode after a 128-token prompt. The historical manual result was 7.261 ms per token for a 32-output tile, down from 7.662 ms per token with 64 outputs per block. Every candidate is remeasured against the manual launch in the same process. A fixed reduction tile of 64 supplies a separate reference for interpreting agent results.

Kimi chooses one configuration ID per round from seven allowed launch settings and gives a short hypothesis. The local evaluator compiles that configuration using the same validated Triton source. The search varies output width, reduction width, warps, and pipeline stages. It keeps the row tile, packing, numerical policy, and application dispatch scope fixed. Its request includes dimensions, kernel behavior, aggregate GPU counters, and benchmark outcomes. Model weights, prompts, token sequences, activations, and credentials are excluded from the prompt. Each of the three rounds can use one API request.

An accepted configuration first passes a real-activation PyTorch projection comparison and the 34-logit comparison on two model sequences. Timing then uses ten paired graph trials in balanced order. A provisional improvement over the manual baseline requires at least 1% lower paired latency and eight wall-time wins out of ten. Any eventual candidate selection should be confirmed on held-out workloads. This feedback-driven search measures progress against fixed references. It does not isolate the causal effect of providing feedback, which would require a separate matched no-feedback search.

The API receipt records client wall time and provider token usage when available. Completion tokens per client second uses total provider-reported completion tokens, which can include reasoning, divided by client wall time. Those API measurements are separate from Qwen GPU replay latency.

## Fixed reduction-tile result

| Same-session batch-1 replay | Manual K tile 32 | Fixed K tile 64 |
| --- | ---: | ---: |
| Median wall ms per decode token | 7.255 | 7.205 |
| Registers per thread | 40 | 51 |
| Kernel shared bytes per block | 1,024 | 2,048 |
| Thread blocks | 32 | 32 |

Both launches passed the same projection tolerance. All 34 logit comparisons retained the same top-1 result, with a maximum relative L2 difference of 0.004604. The fixed tile won ten of ten pairs with a median paired speedup of 1.00683×. The approximately 0.68% latency reduction is below the predeclared 1% screen. A wider K tile halves the number of K loop iterations while raising registers and shared memory. The benchmark did not isolate which effects account for the small net gain.

The four initial offline proposal tests passed, and the original round-one payload was generated locally.

## Model route recovery

The original round-one request returned HTTP 404. The public Inco catalog no longer listed `kimi-k3` and listed `kimi-k3.1-pro:standard` instead. The client now checks the authenticated catalog before sending the replacement request, retains the original 404 receipt, and writes the one-time recovery under `round-1-retry`. Seven offline tests passed. See the updated preview, [fixed comparison receipt](fixed.json), and [commands](../REPRODUCE.md). [Inco API reference](https://platform.inco.ai/docs)

## Round one

Kimi proposed `n32-k64-w4-s2`. Relative to the earlier fixed configuration, it kept the same output tile, reduction tile, and warp count, and raised the pipeline stage count from one to two. Kimi hypothesized that the deeper pipeline could overlap loading with dequantization and dot-product work, while consuming more shared memory and registers.

| Same-session batch-1 replay | Manual N32/K32/4 warps/1 stage | Kimi N32/K64/4 warps/2 stages |
| --- | ---: | ---: |
| Median wall ms per decode token | 7.223 | 7.090 |
| Median CUDA event ms per decode token | 7.214 | 7.082 |
| Registers per thread | 40 | 53 |
| Kernel shared bytes per block | 1,024 | 3,200 |
| Thread blocks / spills | 32 / 0 | 32 / 0 |

The real-activation projection check passed with the same maximum absolute error of 0.0001220703125 for both settings. All 34 full-model logit comparisons kept the same top prediction, with worst relative L2 difference 0.004604. The proposal won ten of ten pairs with a median paired speedup of 1.01844×, about 1.81% lower wall latency. It clears the provisional screen. The comparison against the manual baseline changed both the reduction width and pipeline stage count. The fixed one-stage K64 result came from a separate paired session. The experiment has yet to isolate the effect of the extra stage or measure load overlap directly.

Inco returned 958 prompt tokens and 5,260 completion tokens, with zero cached prompt tokens. The client measured 27.26 seconds, or 192.94 reported completion tokens per client second. The round-two preview incorporates the results and removes this configuration from the option list. See the [proposal](round-1-retry/proposal.json), [API receipt](round-1-retry/api.json), and [GPU receipt](round-1-retry/evaluation.json).

## Round two

Inco returned finish reason `length` after using the 8,192 completion-token cap. The response contained reasoning text and an empty answer field. Inco did not report a reasoning-token breakdown. The parser saved no proposal, so the round produced no GPU result. The client observed 44.86 seconds, 1,088 prompt tokens including 512 cached tokens, and 182.60 reported completion tokens per client second.

This round counts toward the bounded search. The round-three preview carries the successful round-one measurement and the invalid-proposal status from round two. Round one remains the provisional winner. See the [round-two API receipt](round-2/api.json).

## Round three and search outcome

Round three also ended with `length` at the 8,192-token cap. The response contained reasoning text and no final answer. Inco reported 1,108 prompt tokens including 512 cached tokens. Client wall time was 43.28 seconds, or 189.29 reported completion tokens per client second. The parser saved no proposal and there was no GPU evaluation. See the [round-three API receipt](round-3/api.json).

| Round | Client seconds | Completion tokens | Usable configuration | Paired GPU result |
| --- | ---: | ---: | --- | --- |
| 1 | 27.26 | 5,260 | n32-k64-w4-s2 | 1.81% lower latency, 10/10 wins |
| 2 | 44.86 | 8,192 | None | No evaluation |
| 3 | 43.28 | 8,192 | None | No evaluation |

The three completed API responses produced one usable proposal and 21,644 reported completion tokens in 115.40 client seconds. This excludes the initial HTTP 404, which produced no completion. The first proposal remains the only candidate to advance, with 7.090 ms per token versus the 7.223 ms same-session manual baseline. This is a provisional result on the search workload. The two truncated rounds supply no evidence about whether later feedback improved candidate quality. The next comparison is held-out validation of the first proposal.

## Higher-cap follow-up

This separate response-reliability experiment repeated the saved round-three payload with its completion cap raised from 8,192 to 16,384. The preview differs only in `max_tokens`. Inco returned an accepted proposal in 34.00 client seconds using 6,363 completion tokens. It would have fit under the old cap, so one successful repeat does not establish that the larger cap caused the successful finish. The repeat received 1,024 cached prompt tokens versus 512 originally. The API timings should not be compared as a controlled test of the cap.

The proposal, `n32-k32-w4-s2`, changed the manual baseline from one stage to two while keeping the output width, reduction width, warp count, and grid fixed. Both versions passed the projection comparison. All 34 full-model logits matched exactly.

| Same-session batch-1 replay | Manual one stage | Kimi two stages |
| --- | ---: | ---: |
| Median wall ms per decode token | 7.234 | 7.427 |
| Registers per thread | 40 | 46 |
| Kernel shared bytes per block | 1,024 | 1,664 |
| Thread blocks / spills | 32 / 0 | 32 / 0 |

The two-stage variant lost all ten pairs, with roughly 2.6% higher paired wall latency. The resource differences show a cost of the new schedule, but the run did not collect counters that explain the slowdown. Round one's K64/two-stage variant remains the provisional winner. See the [API receipt](round-3-higher-cap/api.json), [proposal](round-3-higher-cap/proposal.json), and [GPU receipt](round-3-higher-cap/evaluation.json).

Round two's higher-cap preview also changes only `max_tokens` from its own original payload. Eight offline tests passed. Inco returned the same `n32-k32-w4-s2` configuration in 15.47 client seconds using 3,279 completion tokens, with 1,024 cached prompt tokens versus 512 originally. See the [API receipt](round-2-higher-cap/api.json) and [proposal](round-2-higher-cap/proposal.json).

An independent timing session replicated the result. All 34 full-model logit outputs matched exactly. The compiler again used 46 registers and 1,664 shared bytes for the proposed variant, versus 40 registers and 1,024 shared bytes for the manual setting. The proposal increased median wall time from 7.240 to 7.427 ms per token, lost all ten pairs, and had a median paired latency regression of 2.58%. See the [GPU receipt](round-2-higher-cap/evaluation.json).

The two higher-cap requests yielded two usable replies and one distinct configuration, whose regression held in both sessions. Both replies used fewer than 8,192 tokens, and each prompt had only one repeat. The observations therefore do not isolate the effect of the cap from variation across requests. Round one's K64/two-stage launch remains the candidate for held-out validation.
