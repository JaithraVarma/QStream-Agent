# QStream-Agent

QStream-Agent tests how packed INT4 weights and profiling-guided Triton tuning affect transformer inference. It runs Qwen3-0.6B with a stock FP16 path and a packed W4A16 path. The INT4 path keeps activations in FP16 and applies groupwise weight quantization to the SwiGLU projections. The selected batch-1 down-projection launch was proposed by Kimi through Inco and validated on the local GPU.

## Results

Both models used CUDA Graphs on an RTX 3060 Laptop GPU. Each context used ten paired trials and 16 fixed cached decode steps per trial.

| Prompt | Stock FP16 | Selected INT4 | Paired latency reduction | Peak allocated memory, FP16 → INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 7.340 ms/token | 7.137 ms/token | 2.76% | 1,181 → 861 MiB |
| 512 tokens | 8.519 ms/token | 8.449 ms/token | 0.87% | 1,260 → 948 MiB |

The two INT4 launch configurations agreed on all 256 top predictions in a small fixed-text check. Stock FP16 scored better on that set. [Technical report](stage30/TECHNICAL-REPORT.md) · [Results figure](stage30/final-checks/inference-results.png) · [Raw benchmark receipt](stage30/final-checks/precision.json)

## SwiGLU and quantization baselines

1. **Build the FP16 reference.** SwiGLU projects each input through gate and up matrices independently, multiplies `SiLU(gate)` by `up`, and feeds that result to the down projection. The FP16 reference was checked against FP32 arithmetic. The independent gate and up branches made it possible to test whether two CUDA streams could help.
2. **Pack groupwise INT4 weights.** Within each group along a weight row, one FP32 scale maps the values to signed codes from −7 to 7. Two codes fit in one byte. With 128 weights per scale, the packed weights occupied **26.56%** of FP16 weight storage, including scales. Exact nibble round trips, zero groups, 45 reconstruction cases, and eight invalid-input cases passed. [Packing implementation](qstream.py) · [Correctness evidence](results/packing-correctness.json)
3. **Measure the costs of reconstruction and concurrency.** Using the existing PyTorch GEMMs, batch-1 FP16 SwiGLU took **0.0851 ms** sequentially and **0.1505 ms** with two streams. Reconstructing INT4 weights on each call took **1.0598 ms** and added **33 GPU kernels** in the captured implementation. CUPTI recorded real cross-stream overlap at larger batch sizes, while the tested eager FFN still took longer with two streams. That motivated moving reconstruction inside the projection and evaluating scheduling separately. [Baseline measurements](results/swiglu-baseline.json) · [Concurrency study](results/stream-and-group-study.json) · [Trace summary](results/gpu-trace-summary.json)

A synthetic group-size sweep also showed the storage and error tradeoff. For normal weights, mean FFN relative L2 increased from **16.84%** at group size 32 to **21.68%** at 256, while storage compression improved from **3.20×** to **3.88×**. With a deliberate outlier in each weight row, error increased from **23.84%** to **56.37%**. These are synthetic FFN measurements. [Group-size results](results/group-size-study.json)

## Triton kernel and profiling-guided optimization

1. **Fuse weight reconstruction into each projection.** The Triton W4A16 kernel loads packed bytes and scales, reconstructs one weight tile, and feeds it directly to the dot product. It avoids materializing a full FP16 weight matrix for each call. The gate, up, and down projections still run as separate launches. Loading unique packed bytes before splitting the nibbles let the compiler emit wider loads. The revised kernel was **1.59–2.80× faster** than the first custom kernel across 15 eager cases. Optimized cached FP16 remained faster in the reported comparisons.
2. **Compare Kimi with and without GPU feedback.** Two matched prompt pairs shared the same kernel and task. One arm also received GPU timing, compiler resource counts, and load-instruction evidence. The feedback candidates reused two scales when a tile crossed a group boundary. For the G48 case, static scalar global-load instruction sites dropped from **16 to 2**. The selected feedback paths improved held-out graph latency by **1.10–1.15×** in geometric mean across the two pairs and runs, relative to the code-only selections. These two pairs were exploratory. [Matched held-out results, pair one](results/kernel-feedback-holdout-1.json) · [Pair two](results/kernel-feedback-holdout-2.json)
3. **Select the combined kernel.** We combined the two-scale lookup with a cheaper signed-nibble decode. On a crossing-group G48 case, graph replay fell from **64.88–65.54 µs** with two-scale lookup alone to **49.15–50.05 µs** with both changes. The combined kernel passed 71 numerical checks, 26 invalid-input checks, five offset-view checks, and CUDA memory and synchronization checks. [Selected kernel](stage23/combined/candidate.py) · [Validation receipt](stage23/combined/full-receipt.json)

## Full-model inference and launch selection

1. **Integrate into Qwen and establish the comparison.** The packed path replaces the SwiGLU projections in all 28 Qwen3-0.6B layers. Stock FP16 was faster in the original eager prefill and decode tests. For the packed model, switching from static-cache eager decode to CUDA Graph replay reduced the batch-1 step from **37.334 to 7.652 ms**. [Initial model timings](stage30/timing/REVIEW.md) · [Graph replay comparison](stage30/graph-decode/REVIEW.md)
2. **Profile and tune batch-1 decode.** The down projection used **16 blocks for 30 SMs**. Changing the output tile from 64 to 32 doubled the grid and reduced decode latency by **5.2%** versus the original packed graph. Inco then evaluated Kimi as a launch-configuration selector. The winning proposal used 32 output features per block, a 64-wide reduction tile, four warps, and two stages. It reduced paired latency from **7.223 to 7.090 ms/token** versus the manual packed launch. The original search yielded one usable proposal in three attempts. Two higher-cap repeats returned the same slower configuration. [Down-projection profile](stage30/down-profile/REVIEW.md) · [Inco search](stage30/decode-search/REVIEW.md)
3. **Confirm the result on new inputs.** The winner reduced latency by **1.63–1.90%** across held-out prompts of 64, 256, and 512 tokens versus manual INT4. In a same-session four-way test, adding a second stage made reduction width 32 **2.61% slower** and width 64 **1.11% faster**. The stock FP16 comparison at the top measures the resulting full-model tradeoff with both paths using CUDA Graphs. [Held-out checks](stage30/heldout/REVIEW.md) · [Matched precision and four-way checks](stage30/final-checks/REVIEW.md)

## Set up and run

Use WSL 2 Ubuntu with NVIDIA GPU access and Python 3.10 or compatible. The measured configuration used Ubuntu 22.04, driver 596.08, CUDA PyTorch 2.9.1, Triton 3.5.1, and Transformers 4.57.6. The setup downloads the pinned public Qwen3-0.6B weights and verifies their SHA-256. The demo runs offline after setup and needs no Inco API key. [PyTorch CUDA wheel instructions](https://pytorch.org/get-started/previous-versions/#v291)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r stage30/requirements.txt
.venv/bin/python -B -m stage30.model_files
bash stage30/demo.sh
```

The first commands create a repo-local Python environment and install the pinned runtime. The downloader puts model files under `stage30/.model`. The demo prints a side-by-side table and writes a fresh receipt under `stage30/final-checks/demo.*/summary.json`.

If you already have a verified copy of the pinned model revision, set `QSTREAM_MODEL_DIR` to its WSL path. You can also set `QSTREAM_PYTHON` to a Python executable with the dependencies installed. Peak allocated memory is measured after graph capture with each model resident on its own. The timed interval measures replay and input staging after prefill.

## Code and evidence

- `qstream.py`, `validate.py`, and `validate_edges.py` cover packing, reconstruction, and their correctness checks.
- `stage23/combined/` holds the selected Triton source and its recorded validation receipts.
- `stage30/` contains the Qwen adapter, graph benchmarks, held-out checks, Inco search code, and the demo. The [Inco results](stage30/decode-search/REVIEW.md) describe the recorded API attempts. Raw API response bodies are omitted from this release.
- `stage30/final-checks/` holds the matched FP16 comparison, the four-way tile-by-stage comparison, and the report figure.

Run the offline search-client tests with `.venv/bin/python -B -m unittest -v stage30.test_decode_search`. Run the GPU packing checks with `.venv/bin/python -B validate.py` and `.venv/bin/python -B validate_edges.py`. See the [reproduction notes](stage30/REPRODUCE.md) for additional commands and receipt locations.
