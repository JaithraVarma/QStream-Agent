#!/usr/bin/env bash
# Rerun the verified FP16 versus Kimi INT4 comparison on this WSL workstation.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"

if (( $# > 1 )); then
    printf 'Usage: bash stage30/demo.sh [new-receipt.json]\n' >&2
    exit 2
fi
if (( $# == 1 )); then
    output="$1"
else
    output="$(mktemp -d stage30/final-checks/demo.XXXXXX)/summary.json"
fi

export HF_HOME="$repo_dir/stage30/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$repo_dir/stage30/.triton-cache"
export CUDA_CACHE_PATH="$repo_dir/stage30/.cuda-cache"

exec env -u INCO_API_KEY "${QSTREAM_PYTHON:-.venv/bin/python}" -B -m stage30.final_checks precision --output "$output"
