"""Auditable PyTorch references; INT4 baseline explicitly reconstructs weights."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Int4Weight:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple
    group_size: int

def quantize(w, group_size=128):
    if w.ndim != 2 or group_size <= 0 or w.shape[1] % group_size or group_size % 2:
        raise ValueError('Require a matrix and an even group size dividing input width')
    if not torch.isfinite(w).all():
        raise ValueError('Weights must be finite')
    groups = w.float().reshape(w.shape[0], -1, group_size)
    scales = groups.abs().amax(-1, keepdim=True) / 7
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = torch.round(groups / scales).clamp(-7, 7).to(torch.int8).reshape(w.shape)
    nibble = q.to(torch.uint8) & 15
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    return Int4Weight(packed, scales.squeeze(-1), tuple(w.shape), group_size)

def unpack(weight):
    low = weight.packed & 15
    high = weight.packed >> 4
    q = torch.stack((low, high), dim=-1).reshape(weight.shape).to(torch.int16)
    return torch.where(q >= 8, q - 16, q).to(torch.int8)

def dequantize(weight, dtype=torch.float16):
    q = unpack(weight).float().reshape(weight.shape[0], -1, weight.group_size)
    return (q * weight.scales.unsqueeze(-1)).reshape(weight.shape).to(dtype)

def swiglu(x, gate, up, down):
    """W are [output, input]: down(silu(gate(x)) * up(x))."""
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

def quantized_swiglu(x, gate, up, down):
    return swiglu(x, *(dequantize(w, x.dtype) for w in (gate, up, down)))

def quantized_swiglu_two_stream(x, gate, up, down, streams):
    """Existing kernels: reconstruct independent weights on separate streams."""
    parent = torch.cuda.current_stream(x.device)
    for stream in streams:
        stream.wait_stream(parent)
    with torch.cuda.stream(streams[0]):
        g = F.linear(x, dequantize(gate, x.dtype))
    with torch.cuda.stream(streams[1]):
        u = F.linear(x, dequantize(up, x.dtype))
    d = dequantize(down, x.dtype)
    for stream in streams:
        parent.wait_stream(stream)
    g.record_stream(parent)
    u.record_stream(parent)
    return F.linear(F.silu(g) * u, d)
