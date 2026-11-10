# Qwen CUDA Graph replay profile

PASS. The graph uses the validated packed Qwen model, batch 1, the
128-token benchmark prompt, and a static cache with room for 16 decode
steps. The replayed logits matched eager static execution exactly
on all 34 checks before profiling.

| Replay-only measurement | Median |
|---|---:|
| CUDA event with CUPTI off | 7.656 ms |
| CUDA event with CUPTI on | 7.992 ms |
| Traced first-to-last kernel span | 7.936 ms |
| Traced GPU kernel activity | 7.530 ms |
| Traced gaps between kernels | 0.406 ms |
| Packed GEMM activity | 2.224 ms |
| Gate and up packed GEMMs | 0.957 ms |
| Down packed GEMMs | 1.266 ms |

Each of the 16 replay windows contained 1,756 kernel records, including
56 packed gate/up and 28 packed down launches. All used one CUDA stream.
There were 28,096 records, no dropped or malformed records, and no kernels
outside the replay windows. Independent medians need not sum exactly.
Token staging, prefill, capture, and final flush are outside these windows.

Packed GEMMs account for a median 28.0% of the traced span and 29.5% of
kernel activity. Down alone accounts for roughly 16% of the traced span.
The CUPTI-enabled event median exceeded its replay-only unprofiled
control by 0.336 ms. That difference is descriptive rather than an exact
instrumentation correction.

The earlier eager trace covered a different execution and cache regime,
so use this trace for target selection rather than subtracting its gaps
from the eager trace. The next candidate target is the down projection
with M1,N1024,K3072,G128. If its traced work were halved and all other
work stayed unchanged, the reduction would be about 0.63 ms, or 8% of
this traced span. That is an illustration, not an expected speedup.
Profile this exact projection with Nsight Compute before asking Kimi to
optimize it. Compare any accepted proposal with the full graph decode
baseline under unprofiled, paired conditions.

[Raw results](summary.json) · [Commands](../REPRODUCE.md)
