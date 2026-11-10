# Reproduction notes

Use the repo-local WSL environment described in the [README](../README.md). The release includes the measured JSON and excludes credentials, model weights, caches, raw profiler traces, and full API responses. The original search is documented with compact request, proposal, and outcome receipts.

The demo runs the matched stock FP16 and selected INT4 comparison. It saves a new JSON receipt, so the checked-in result stays intact.

```bash
bash stage30/demo.sh
```

The four-way check varies the reduction width and pipeline stage count while holding the packed model, output width, and warp count fixed. Use a new output filename. The plot reproduces the figure from the included baseline receipts and requires Matplotlib.

```bash
env -u INCO_API_KEY .venv/bin/python -B -m stage30.final_checks factorial --output stage30/final-checks/factorial-local.json
.venv/bin/python -m pip install matplotlib
.venv/bin/python -B -m stage30.plot_inference_results
```

The offline Inco preview reconstructs the final prompt from the included history without calling the provider.

```bash
env -u INCO_API_KEY .venv/bin/python -B -m stage30.decode_search preview --round 3
```

[Precision receipt](final-checks/precision.json) · [Four-way receipt](final-checks/factorial.json) · [Held-out receipt](heldout/summary.json) · [Inco search history](decode-search/REVIEW.md)
