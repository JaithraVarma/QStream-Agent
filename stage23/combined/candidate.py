"""Fused tile-local INT4 reconstruction and FP16-input/FP32-accumulator GEMM.

Input format is qstream.Int4Weight, unchanged. This is inference-only and accepts
contiguous CUDA tensors. No full reconstructed weight tensor is allocated.
"""
import torch
import triton
import triton.language as tl

from qstream import Int4Weight

BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32
NUM_WARPS, NUM_STAGES = 4, 1


@triton.jit
def _w4a16_kernel(
    X, P, S, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    SXM: tl.constexpr, SXK: tl.constexpr,
    SPN: tl.constexpr, SPK: tl.constexpr,
    SSN: tl.constexpr, SSG: tl.constexpr,
    SYM: tl.constexpr, SYN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Compile-time fast path: when G is a multiple of BK, each aligned K tile
    # belongs entirely to one scale group (K % G == 0 is a wrapper precondition,
    # so tiles are always BK-aligned or a partial tail within one group).
    ALIGNED: tl.constexpr = (G % BK) == 0
    # When G > BK but not a multiple of it, a BK-wide tile spans at most two
    # consecutive scale groups, so two affine [BN] loads plus a select replace
    # the non-affine per-element scale gather.
    TWO_GROUP: tl.constexpr = ((G % BK) != 0) and (G > BK)

    # Program IDs select output tiles, not individual CUDA threads.
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + rk  # absolute K indices, required for group selection

        # [BM,BK]: pointer arithmetic counts FP16 elements.
        x_ptrs = X + m[:, None] * SXM + k[None, :] * SXK
        a = tl.load(x_ptrs, mask=(m[:, None] < M) & (k[None, :] < K), other=0.0)

        # Load each unique byte once, then recover the original K order.
        kb = tile * (BK // 2) + tl.arange(0, BK // 2)
        p_ptrs = P + n[:, None] * SPN + kb[None, :] * SPK
        byte = tl.load(p_ptrs, mask=(n[:, None] < N) & (kb[None, :] < K // 2), other=0).to(tl.int32)
        low = byte & 15
        high = (byte >> 4) & 15
        unsigned = tl.reshape(tl.interleave(low, high), (BN, BK))
        signed = ((unsigned + 8) & 15) - 8
        valid_w = (n[:, None] < N) & (k[None, :] < K)

        if ALIGNED:
            # Whole tile shares one group; tile*BK < K always, so the group
            # index is in range even for a partial K tail. [BN] load,
            # broadcast across K — one scale fetch per (n, tile) instead of
            # one per (n, k).
            group = (tile * BK) // G
            s_ptrs = S + n * SSN + group * SSG
            scale_row = tl.load(s_ptrs, mask=n < N, other=0.0)
            scale = tl.broadcast_to(scale_row[:, None], (BN, BK))
        elif TWO_GROUP:
            # G > BK: the tile overlaps at most groups group0 and group0+1.
            # Both loads are affine in n, so they coalesce into vector loads.
            group0 = (tile * BK) // G
            in0 = (k // G) == group0  # [BK] select between the two groups
            s0 = tl.load(S + n * SSN + group0 * SSG, mask=n < N, other=0.0)
            # group0+1 is only valid if its first element is < K; masked lanes
            # get 0.0 and are never selected for valid k.
            s1 = tl.load(
                S + n * SSN + (group0 + 1) * SSG,
                mask=(n < N) & ((group0 + 1) * G < K),
                other=0.0,
            )
            scale = tl.where(in0[None, :], s0[:, None], s1[:, None])
        else:
            # Each k uses its own group's scale; groups may cross BK boundaries.
            s_ptrs = S + n[:, None] * SSN + (k[None, :] // G) * SSG
            scale = tl.load(s_ptrs, mask=valid_w, other=0.0)
        # Preserve qstream.dequantize's FP32 multiply then FP16 rounding.
        w = (signed.to(tl.float32) * scale).to(tl.float16)
        # a [BM,BK] @ w.T [BK,BN], with FP32 running accumulation.
        acc = tl.dot(a, tl.trans(w), acc)

    y_ptrs = Y + m[:, None] * SYM + n[None, :] * SYN
    tl.store(y_ptrs, acc.to(tl.float16), mask=(m[:, None] < M) & (n[None, :] < N))


def _validate(x, weight):
    """Metadata-only checks; no hidden copies or device-wide value scans."""
    if not isinstance(x, torch.Tensor):
        raise TypeError('X must be a torch.Tensor')
    if x.ndim != 2:
        raise ValueError('X must have shape [M,K]')
    if x.dtype != torch.float16:
        raise TypeError('X must be FP16')
    if not x.is_cuda:
        raise ValueError('X must be on CUDA')
    if not x.is_contiguous():
        raise ValueError('X must be contiguous')
    if x.requires_grad:
        raise ValueError('This wrapper is inference-only')
    if not isinstance(weight, Int4Weight):
        raise TypeError('weight must be qstream.Int4Weight')
    shape, group = weight.shape, weight.group_size
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError('weight.shape must be [N,K]')
    if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in shape):
        raise ValueError('Weight dimensions must be positive integers')
    n, k = shape
    m, xk = x.shape
    if m <= 0 or xk != k:
        raise ValueError('X must have positive M and match weight K')
    if not isinstance(group, int) or isinstance(group, bool) or group <= 0 or group % 2 or k % group:
        raise ValueError('Group size must be a positive even integer dividing K')
    for tensor, dtype, expected, name in (
        (weight.packed, torch.uint8, (n, k // 2), 'packed'),
        (weight.scales, torch.float32, (n, k // group), 'scales'),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f'{name} must be a {dtype} tensor')
        if tensor.device != x.device:
            raise ValueError(f'{name} must be on the same CUDA device as X')
        if tuple(tensor.shape) != expected or not tensor.is_contiguous():
            raise ValueError(f'{name} must be contiguous with shape {expected}')
        if tensor.requires_grad:
            raise ValueError('This wrapper is inference-only')
    # Finite values are a caller precondition, not an extra synchronization pass.
    return m, n, k, group


def w4a16_gemm(x, weight, *, return_compiled=False):
    """Compute X @ reconstructed_weight.T; optionally return compiler evidence.

    Requires finite FP16 activations and finite FP32 scales. FP16 output can
    overflow for large mathematical results, exactly as in a reference GEMM.
    return_compiled is for correctness/resource inspection, not autotuning.
    """
    m, n, k, group = _validate(x, weight)
    with torch.cuda.device(x.device):
        y = torch.empty((m, n), device=x.device, dtype=torch.float16)
        compiled = _w4a16_kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            x, weight.packed, weight.scales, y,
            m, n, k, group,
            *x.stride(), *weight.packed.stride(), *weight.scales.stride(), *y.stride(),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
    return (y, compiled) if return_compiled else y
