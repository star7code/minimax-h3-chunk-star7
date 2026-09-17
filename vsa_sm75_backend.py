"""SM75 Video Sparse Attention over the precompiled Star7 CUDA core."""

from __future__ import annotations

import ctypes

import torch


BLOCK = 64
HEAD_DIM = 128


def _load_sm75_backend():
    try:
        from . import sm75_backend
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        module_name = "star7_sm75_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        path = Path(__file__).with_name("sm75_backend.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load SM75 backend from {path}")
        sm75_backend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = sm75_backend
        spec.loader.exec_module(sm75_backend)
    return sm75_backend


def _library():
    library = _load_sm75_backend()._load()
    try:
        launch = library.star7_vsa_sm75_launch_all_int8
    except AttributeError as exc:
        raise RuntimeError(
            "The installed SM75 binary predates Star7 VSA support. Install the "
            "matching node release; no dense fallback was attempted."
        ) from exc
    launch.argtypes = [
        *([ctypes.c_uint64] * 10),
        *([ctypes.c_int] * 5),
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    launch.restype = ctypes.c_int
    pack_lut = library.star7_sol_sm75_pack_lut
    pack_lut.argtypes = [
        *([ctypes.c_uint64] * 3),
        *([ctypes.c_int] * 4),
        ctypes.c_uint64,
    ]
    pack_lut.restype = ctypes.c_int
    return library


def availability() -> tuple[bool, str]:
    available, reason = _load_sm75_backend().availability()
    if not available:
        return False, reason
    try:
        _library()
    except Exception as exc:
        return False, str(exc)
    return True, "precompiled SM75 VSA Q64/K64 All-INT8 fine kernel"


def block_means(value: torch.Tensor, block_len: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[-1] != HEAD_DIM:
        raise ValueError("SM75 VSA block means require [B,H,T,128]")
    if value.shape[2] % BLOCK:
        raise ValueError("SM75 VSA padded sequence must be a multiple of 64")
    blocks = value.shape[2] // BLOCK
    if block_len.shape != (blocks,) or block_len.dtype != torch.int32:
        raise ValueError(f"SM75 VSA block_len must be int32 [{blocks}]")
    lengths = block_len.to(torch.float32).view(1, 1, blocks, 1)
    return value.view(*value.shape[:2], blocks, BLOCK, HEAD_DIM).sum(
        dim=3, dtype=torch.float32,
    ) / lengths


def build_topk_routing(
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    *,
    topk_ratio: float,
    prefix_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if q_mean.shape != k_mean.shape or q_mean.ndim != 4:
        raise ValueError("SM75 VSA routing requires equal [B,H,N,128] means")
    batch, heads, blocks, head_dim = q_mean.shape
    if head_dim != HEAD_DIM or not 0.0 < topk_ratio < 1.0:
        raise ValueError("SM75 VSA routing parameters are invalid")
    prefix_blocks = min(max(0, int(prefix_blocks)), blocks)
    ranked = torch.matmul(q_mean, k_mean.transpose(-1, -2))
    if prefix_blocks:
        ranked[..., :prefix_blocks] = float("-inf")
    candidates = blocks - prefix_blocks
    keep = max(0, min(candidates - 1, max(1, round(topk_ratio * candidates))))
    if keep:
        threshold = ranked.topk(keep, dim=-1).values[..., -1:]
        exact = ranked >= threshold
    else:
        exact = torch.zeros_like(ranked, dtype=torch.bool)
    index = torch.arange(blocks, device=q_mean.device)
    exact |= (index[:, None] - index[None, :]).abs().le(1).view(
        1, 1, blocks, blocks,
    )
    if prefix_blocks:
        exact[..., :prefix_blocks] = True
        exact[..., :prefix_blocks, :] = True
    row_count = exact.sum(dim=-1, dtype=torch.int32)
    lut_stride = int(row_count.max().item())
    if q_mean.is_cuda and torch.cuda.get_device_capability(q_mean.device) == (7, 5):
        exact_u8 = exact.to(torch.uint8).contiguous()
        packed = torch.empty(
            (batch, heads, blocks, lut_stride),
            dtype=torch.int32, device=q_mean.device,
        )
        code = int(_library().star7_sol_sm75_pack_lut(
            exact_u8.data_ptr(), row_count.data_ptr(), packed.data_ptr(),
            batch * heads * blocks, blocks, lut_stride, 0,
            torch.cuda.current_stream(q_mean.device).cuda_stream,
        ))
        if code:
            raise RuntimeError(f"SM75 VSA LUT packing failed with code={code}")
    else:
        keys = torch.arange(blocks, dtype=torch.int32, device=q_mean.device).view(
            1, 1, 1, blocks,
        )
        packed = torch.where(exact, keys, blocks).sort(dim=-1).values[..., :lut_stride]
    density = float(row_count.float().mean().item() / blocks)
    return row_count.contiguous(), packed.contiguous(), density


def quantize_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    k_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    native = _load_sm75_backend()
    q_int8, q_scale = native.quantize(q, BLOCK // 4, 1.0)
    center = k_mean.mean(dim=2, keepdim=True).to(torch.float16).contiguous()
    k_int8, k_scale = native.quantize(k, BLOCK, 1.0, mean=center)
    return q_int8, k_int8, q_scale, k_scale


def run_fine(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | list[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    row_count: torch.Tensor,
    lut: torch.Tensor,
    block_len: torch.Tensor,
) -> torch.Tensor:
    consume_v = isinstance(v, list)
    if consume_v:
        if len(v) != 1:
            raise ValueError("SM75 VSA consuming V input requires one tensor")
        v = v.pop()
    tensors = (q, k, v, q_scale, k_scale, row_count, lut, block_len)
    if any(not value.is_cuda or value.device != q.device for value in tensors):
        raise ValueError("SM75 VSA tensors must share one CUDA device")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float16:
        raise TypeError("SM75 VSA fine attention requires INT8 Q/K and FP16 V")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("SM75 VSA Q/K scales must be FP32")
    if row_count.dtype != torch.int32 or lut.dtype != torch.int32:
        raise TypeError("SM75 VSA routes must be INT32")
    if block_len.dtype != torch.int32:
        raise TypeError("SM75 VSA block_len must be INT32")
    if any(not value.is_contiguous() for value in tensors):
        raise ValueError("SM75 VSA tensors must be contiguous")

    batch, heads, length, head_dim = q.shape
    blocks = length // BLOCK
    if length % BLOCK or head_dim != HEAD_DIM or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("SM75 VSA requires equal padded [B,H,T,128] Q/K/V")
    if row_count.shape != (batch, heads, blocks):
        raise ValueError("SM75 VSA row_count shape is invalid")
    if lut.ndim != 4 or lut.shape[:3] != row_count.shape:
        raise ValueError("SM75 VSA LUT shape is invalid")
    if block_len.shape != (blocks,):
        raise ValueError("SM75 VSA block_len shape is invalid")
    if int(block_len.min().item()) < 1 or int(block_len.max().item()) > BLOCK:
        raise ValueError("SM75 VSA block_len values must be in [1,64]")
    if q_scale.shape != (batch, heads, blocks * 4):
        raise ValueError("SM75 VSA Q scale shape is invalid")
    if k_scale.shape != (batch, heads, blocks):
        raise ValueError("SM75 VSA K scale shape is invalid")
    lut_stride = int(lut.shape[-1])
    if int(row_count.min().item()) <= 0 or int(row_count.max().item()) > lut_stride:
        raise ValueError("SM75 VSA row_count exceeds LUT stride")

    library = _library()
    stream = torch.cuda.current_stream(q.device).cuda_stream
    v_int8 = torch.empty(
        (batch, heads, head_dim, length), dtype=torch.int8, device=q.device,
    )
    v_scale = torch.empty(
        (batch, heads, head_dim), dtype=torch.float32, device=q.device,
    )
    code = int(library.star7_sla_sm75_quant_v_int8(
        v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(),
        batch, heads, length, head_dim, length,
        v.stride(0), v.stride(1), v.stride(2), 1, stream,
    ))
    if code:
        raise RuntimeError(f"SM75 VSA V quantization failed with code={code}")
    if consume_v:
        del tensors, v
        output = torch.empty(
            (batch, heads, length, head_dim),
            dtype=torch.float16, device=q.device,
        )
    else:
        output = torch.empty_like(v)
    code = int(library.star7_vsa_sm75_launch_all_int8(
        q.data_ptr(), k.data_ptr(), v_int8.data_ptr(),
        q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        row_count.data_ptr(), lut.data_ptr(), block_len.data_ptr(),
        output.data_ptr(), batch, heads, length, length, lut_stride,
        head_dim ** -0.5, stream,
    ))
    if code:
        raise RuntimeError(
            f"SM75 VSA fine CUDA launch failed with cudaError={code}; "
            "no dense fallback was attempted"
        )
    return output


def add_coarse_(
    output: torch.Tensor,
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    v_mean: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    batch, heads, blocks, head_dim = q_mean.shape
    flat = lambda value: value.reshape(batch * heads, blocks, head_dim)
    scores = torch.bmm(flat(q_mean), flat(k_mean).transpose(1, 2)) * head_dim ** -0.5
    coarse = torch.bmm(torch.softmax(scores, dim=-1), flat(v_mean))
    coarse = coarse.view(batch, heads, blocks, head_dim).permute(0, 2, 1, 3)
    output_bthd = output.transpose(1, 2)
    output_bthd.view(batch, blocks, BLOCK, heads, head_dim).addcmul_(
        gate.view(batch, blocks, BLOCK, heads, head_dim),
        coarse[:, :, None],
    )
    return output
