"""Run Qwen's SwiGLU projections with packed or reconstructed weights."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from qstream import Int4Weight, dequantize, quantize


PROJECTIONS = ("gate", "up", "down")


def load_selected_kernel_module():
    folder = Path(__file__).resolve().parents[1] / "stage23" / "combined"
    expected = json.loads((folder / "summary.json").read_text())
    path = folder / "candidate.py"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected["selected"] == "combined" and expected["hashes"]["combined"] == actual
    for check in ("full", "memcheck", "synccheck"):
        receipt = json.loads((folder / f"{check}-receipt.json").read_text())
        assert receipt["status"] == "PASS" and receipt["source_sha256"] == actual
    spec = importlib.util.spec_from_file_location("qstream_checkpoint3_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual


def load_selected_kernel():
    module, actual = load_selected_kernel_module()
    return module.w4a16_gemm, actual


def pack_mlp(mlp, group=128):
    weights = {}
    for name in PROJECTIONS:
        projection = getattr(mlp, name + "_proj")
        if projection.bias is not None or projection.weight.dtype != torch.float16:
            raise ValueError("Expected a bias-free FP16 Qwen projection")
        weights[name] = quantize(projection.weight.detach(), group)
    return weights


class PackedMLP(nn.Module):
    def __init__(self, weights, gemm):
        super().__init__()
        self.gemm = gemm
        self.metadata = {}
        for name, weight in weights.items():
            self.register_buffer(name + "_bytes", weight.packed)
            self.register_buffer(name + "_scales", weight.scales)
            self.metadata[name] = (weight.shape, weight.group_size)

    def project(self, x, name):
        shape, group = self.metadata[name]
        weight = Int4Weight(getattr(self, name + "_bytes"),
                            getattr(self, name + "_scales"), shape, group)
        return self.gemm(x, weight)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        activated = F.silu(self.project(flat, "gate")) * self.project(flat, "up")
        result = self.project(activated, "down")
        return result.reshape(*x.shape[:-1], result.shape[-1])


class ReconstructedMLP(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, weight in weights.items():
            self.register_buffer(name, dequantize(weight))

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)
