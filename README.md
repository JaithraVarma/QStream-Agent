# QStream-Agent

QStream-Agent tests how packed INT4 weights and profiling-guided Triton tuning affect transformer inference. It runs Qwen3-0.6B with a stock FP16 path and a packed W4A16 path. The INT4 path keeps activations in FP16 and applies groupwise weight quantization to the SwiGLU projections. The selected batch-1 down-projection launch was proposed by Kimi through Inco and validated on the local GPU.

## Results

Both models used CUDA Graphs on an RTX 3060 Laptop GPU. Each context used ten paired trials and 16 fixed cached decode steps per trial.

| Prompt | Stock FP16 | Selected INT4 | Paired latency reduction | Peak allocated memory, FP16 → INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 7.340 ms/token | 7.137 ms/token | 2.76% | 1,181 → 861 MiB |
| 512 tokens | 8.519 ms/token | 8.449 ms/token | 0.87% | 1,260 → 948 MiB |

The two INT4 launch configurations agreed on all 256 top predictions in a small fixed-text check. Stock FP16 scored better on that set. The full report describes the quality tradeoff, the four-way pipeline comparison, the Inco search, and its limits. [Technical report](stage30/CHECKPOINT-3.md) · [Results figure](stage30/final-checks/checkpoint3.png) · [Raw benchmark receipt](stage30/final-checks/precision.json)

## Set up and run

Use WSL 2 Ubuntu with NVIDIA GPU access and Python 3.10 or compatible. The measured configuration used Ubuntu 22.04, driver 596.08, CUDA PyTorch 2.9.1, Triton 3.5.1, and Transformers 4.57.6. The setup downloads the pinned public Qwen3-0.6B checkpoint and verifies the weights SHA-256. The demo runs offline after setup and needs no Inco API key. [PyTorch CUDA wheel instructions](https://pytorch.org/get-started/previous-versions/#v291)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r stage30/requirements.txt
.venv/bin/python -B -m stage30.model_files
bash stage30/demo.sh
```

The first commands create a repo-local Python environment and install the pinned runtime. The downloader puts model files under `stage30/.model`. The demo prints a side-by-side table and writes a fresh receipt under `stage30/final-checks/demo.*/summary.json`.

If you already have a verified copy of the exact checkpoint, set `QSTREAM_MODEL_DIR` to its WSL path. You can also set `QSTREAM_PYTHON` to a Python executable with the dependencies installed. Peak allocated memory is measured after graph capture with each model resident on its own. The timed interval measures replay and input staging after prefill.

## Code and evidence

- `qstream.py`, `validate.py`, and `validate_edges.py` cover packing, reconstruction, and their correctness checks.
- `stage23/combined/` holds the selected Triton source and its recorded validation receipts.
- `stage30/` contains the Qwen adapter, graph benchmarks, held-out checks, Inco search code, and the demo. The [Inco results](stage30/decode-search/REVIEW.md) describe the recorded API attempts. Raw API response bodies are omitted from this release.
- `stage30/final-checks/` holds the matched FP16 comparison, the four-way tile-by-stage comparison, and the report figure.

Run the offline search-client tests with `.venv/bin/python -B -m unittest -v stage30.test_decode_search`. Run the GPU packing checks with `.venv/bin/python -B validate.py` and `.venv/bin/python -B validate_edges.py`. See the [reproduction notes](stage30/REPRODUCE.md) for additional commands and receipt locations.
