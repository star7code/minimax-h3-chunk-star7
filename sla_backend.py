"""Strict LightX2V-style dynamic sparse attention for MiniMax H3.

The routing contract follows ModelTC/LightX2V's ``dynamic_sparse_attn``
operator: smooth K for routing, mean-pool Q/K blocks, and retain the top key
blocks for every query block. The attention kernel is Star7-specific. Both
SM75 and SM80+ use INT8 QK, FP16 PV, and FP32 online-softmax state.

This module deliberately has no SageAttention dependency and no fallback.
Callers selecting it explicitly either execute this kernel or receive an
exception.
"""

from __future__ import annotations

import threading
import importlib.util
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


_LOG = logging.getLogger("MiniMaxH3ActivationChunkStar7")


def _configure_triton_cache() -> None:
    """Keep first-run compilation out of a potentially locked user cache."""
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    try:
        import folder_paths

        cache_root = Path(folder_paths.get_temp_directory())
    except Exception:
        cache_root = Path(tempfile.gettempdir())
    cache_dir = cache_root / "star7-triton-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_configure_triton_cache()


SM75_BACKEND_NAME = "sla_sm75_qk_int8_pv_fp16"
SM75_ALL_INT8_BACKEND_NAME = "sla_sm75_all_int8_experimental"
SM80PLUS_BACKEND_NAME = "sla_sm80+_qk_int8_pv_fp16"
STRICT_SLA_LABEL = "Star7 strict SLA"
BLOCK_Q = 128
BLOCK_K = 64
HEAD_DIM = 128
DEFAULT_SPARSITY = 0.85
_LOG2E = 1.4426950408889634


class SLAUnavailableError(RuntimeError):
    """The strict SLA backend cannot run in the current environment."""


@dataclass
class SLAResult:
    output: torch.Tensor
    query_blocks: int
    key_blocks: int
    selected_key_blocks: int
    effective_sparsity: float
    implementation: str
    protected_query_blocks: int = 0
    dense_guard_status: str = "not-requested"


_TRITON_IMPORT_ERROR = None
_SM75_TORCH_PREPROCESS = False
try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


def _load_sm75_backend():
    try:
        from . import sm75_backend
        return sm75_backend
    except ImportError:
        module_name = "star7_sm75_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        path = Path(__file__).with_name("sm75_backend.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Star7 SM75 backend from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


if triton is not None:

    @triton.jit
    def _mean_pool_kernel(
        X,
        X_MEAN,
        OUT,
        length: tl.constexpr,
        head_dim: tl.constexpr,
        block: tl.constexpr,
        subtract_mean: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        bh_index = tl.program_id(1).to(tl.int64)
        token_offsets = block_index * block + tl.arange(0, block)
        dim_offsets = tl.arange(0, head_dim)
        base = bh_index * length * head_dim
        pointers = X + base + token_offsets[:, None] * head_dim + dim_offsets[None, :]
        valid = token_offsets[:, None] < length
        value = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
        if subtract_mean:
            mean_value = tl.load(X_MEAN + bh_index * head_dim + dim_offsets)
            value = tl.where(valid, value - mean_value[None, :], 0.0)
        count = tl.minimum(block, length - block_index * block)
        pooled = tl.sum(value, axis=0) / count
        out_blocks = tl.cdiv(length, block)
        out_ptr = OUT + (bh_index * out_blocks + block_index) * head_dim + dim_offsets
        tl.store(out_ptr, pooled.to(OUT.type.element_ty))


    @triton.jit
    def _quantize_per_block_int8_kernel(
        X,
        X_MEAN,
        OUT,
        SCALE,
        length: tl.constexpr,
        head_dim: tl.constexpr,
        block: tl.constexpr,
        multiplier: tl.constexpr,
        subtract_mean: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        bh_index = tl.program_id(1).to(tl.int64)
        token_offsets = block_index * block + tl.arange(0, block)
        dim_offsets = tl.arange(0, head_dim)
        base = bh_index * length * head_dim
        pointers = X + base + token_offsets[:, None] * head_dim + dim_offsets[None, :]
        valid = token_offsets[:, None] < length
        value = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
        if subtract_mean:
            mean_value = tl.load(X_MEAN + bh_index * head_dim + dim_offsets)
            value = tl.where(valid, value - mean_value[None, :], 0.0)
        value *= multiplier
        scale = tl.maximum(tl.max(tl.abs(value)) / 127.0, 1.0e-8)
        quantized = value / scale
        quantized += 0.5 * tl.where(quantized >= 0.0, 1.0, -1.0)
        tl.store(OUT + base + token_offsets[:, None] * head_dim + dim_offsets[None, :],
                 quantized.to(tl.int8), mask=valid)
        blocks = tl.cdiv(length, block)
        tl.store(SCALE + bh_index * blocks + block_index, scale)


    @triton.jit
    def _sparse_qk_int8_pv_fp16_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        LUT,
        OUT,
        length: tl.constexpr,
        query_blocks: tl.constexpr,
        compute_query_blocks: tl.constexpr,
        key_blocks: tl.constexpr,
        selected_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        route_block_q: tl.constexpr,
        compute_block_q: tl.constexpr,
        block_k: tl.constexpr,
    ):
        compute_query_block = tl.program_id(0)
        query_block = compute_query_block * compute_block_q // route_block_q
        bh_index = tl.program_id(1).to(tl.int64)
        query_offsets = compute_query_block * compute_block_q + tl.arange(0, compute_block_q)
        key_offsets = tl.arange(0, block_k)
        dim_offsets = tl.arange(0, head_dim)
        base = bh_index * length * head_dim

        q_ptr = Q + base + query_offsets[:, None] * head_dim + dim_offsets[None, :]
        q = tl.load(q_ptr, mask=query_offsets[:, None] < length, other=0).to(tl.int8)
        q_scale = tl.load(Q_SCALE + bh_index * query_blocks + query_block)
        lut_base = (bh_index * query_blocks + query_block) * selected_blocks

        row_max = tl.full([compute_block_q], -float("inf"), tl.float32)
        row_sum = tl.zeros([compute_block_q], tl.float32)
        accumulator = tl.zeros([compute_block_q, head_dim], tl.float32)

        for selected_index in tl.range(selected_blocks):
            key_block = tl.load(LUT + lut_base + selected_index)
            key_token_offsets = key_block * block_k + key_offsets
            key_valid = key_token_offsets < length
            k_ptr = K + base + key_token_offsets[None, :] * head_dim + dim_offsets[:, None]
            k = tl.load(k_ptr, mask=key_valid[None, :], other=0).to(tl.int8)
            k_scale = tl.load(K_SCALE + bh_index * key_blocks + key_block)

            # Keep the form accepted by Triton's SM75 integer-dot lowering.
            # Supplying an explicit int32 output type currently trips the
            # Triton-Windows 3.5 accelerator pass on Turing.
            score = tl.dot(q, k).to(tl.float32)
            score *= q_scale * k_scale
            score = tl.where(key_valid[None, :], score, -float("inf"))

            local_max = tl.max(score, axis=1)
            new_max = tl.maximum(row_max, local_max)
            probability = tl.math.exp2(score - new_max[:, None])
            local_sum = tl.sum(probability, axis=1)
            correction = tl.math.exp2(row_max - new_max)
            accumulator *= correction[:, None]

            v_ptr = V + base + key_token_offsets[:, None] * head_dim + dim_offsets[None, :]
            value = tl.load(v_ptr, mask=key_valid[:, None], other=0.0).to(tl.float16)
            accumulator += tl.dot(probability.to(tl.float16), value, out_dtype=tl.float32)
            row_sum = row_sum * correction + local_sum
            row_max = new_max

        accumulator /= row_sum[:, None]
        out_ptr = OUT + base + query_offsets[:, None] * head_dim + dim_offsets[None, :]
        tl.store(out_ptr, accumulator.to(OUT.type.element_ty), mask=query_offsets[:, None] < length)


def _require_environment(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not torch.cuda.is_available() or not q.is_cuda:
        raise SLAUnavailableError(f"{STRICT_SLA_LABEL} requires an NVIDIA CUDA tensor")
    if q.device != k.device or q.device != v.device:
        raise SLAUnavailableError("SLA Q/K/V must be on the same CUDA device")
    if q.dtype is not torch.float16 or k.dtype is not torch.float16 or v.dtype is not torch.float16:
        raise SLAUnavailableError(
            f"{STRICT_SLA_LABEL} requires FP16 input; got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise SLAUnavailableError("SLA expects contiguous [B,H,L,D] Q/K/V tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise SLAUnavailableError(f"SLA requires equal Q/K/V shapes; got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[0] != 1:
        raise SLAUnavailableError(f"SLA currently supports batch=1; got batch={q.shape[0]}")
    if q.shape[-1] != HEAD_DIM:
        raise SLAUnavailableError(f"SLA requires head_dim={HEAD_DIM}; got {q.shape[-1]}")
    capability = torch.cuda.get_device_capability(q.device)
    if triton is None and capability != (7, 5):
        raise SLAUnavailableError(
            f"{STRICT_SLA_LABEL} SM80+ requires Triton; import failed: "
            f"{_TRITON_IMPORT_ERROR}"
        )
    applied_priority_ranges: tuple[tuple[int, int], ...] = ()
    if capability == (7, 5):
        available, reason = _load_sm75_backend().availability()
        if not available:
            raise SLAUnavailableError(
                f"{SM75_BACKEND_NAME} native CUDA core is unavailable: {reason}. "
                "No fallback was attempted."
            )
    elif capability < (8, 0):
        raise SLAUnavailableError(
            f"{STRICT_SLA_LABEL} requires SM75 native CUDA or SM80+ Triton; "
            f"got SM{capability[0]}{capability[1]}. No fallback was attempted."
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise SLAUnavailableError("SLA requires contiguous [B,H,L,D] Q/K/V tensors")


def backend_name_for_capability(capability: tuple[int, int]) -> str:
    return SM75_BACKEND_NAME if capability == (7, 5) else SM80PLUS_BACKEND_NAME


def check_runtime_support(
    device: torch.device | int | None = None,
    requested_backend: str | None = None,
) -> tuple[int, int]:
    """Fail before model patching when the strict backend cannot possibly run."""
    if not torch.cuda.is_available():
        raise SLAUnavailableError(f"{STRICT_SLA_LABEL} requires NVIDIA CUDA")
    capability = torch.cuda.get_device_capability(device)
    if triton is None and capability != (7, 5):
        raise SLAUnavailableError(
            f"{STRICT_SLA_LABEL} SM80+ requires Triton; import failed: "
            f"{_TRITON_IMPORT_ERROR}"
        )
    if requested_backend is None:
        requested_backend = backend_name_for_capability(capability)
    valid_names = {
        SM75_BACKEND_NAME, SM75_ALL_INT8_BACKEND_NAME, SM80PLUS_BACKEND_NAME
    }
    if requested_backend not in valid_names:
        raise SLAUnavailableError(f"unknown strict SLA backend: {requested_backend}")
    if requested_backend in {SM75_BACKEND_NAME, SM75_ALL_INT8_BACKEND_NAME} and capability != (7, 5):
        raise SLAUnavailableError(
            f"{SM75_BACKEND_NAME} requires exactly SM75; this GPU is "
            f"SM{capability[0]}{capability[1]}. No fallback was attempted."
        )
    if requested_backend == SM80PLUS_BACKEND_NAME and capability < (8, 0):
        raise SLAUnavailableError(
            f"{SM80PLUS_BACKEND_NAME} requires SM80 or newer; this GPU is "
            f"SM{capability[0]}{capability[1]}. No fallback was attempted."
        )
    if capability == (7, 5):
        available, reason = _load_sm75_backend().availability()
        if not available:
            raise SLAUnavailableError(
                f"{SM75_BACKEND_NAME} native CUDA core is unavailable: {reason}. "
                "No fallback was attempted."
            )
    elif capability < (8, 0):
        raise SLAUnavailableError(
            f"{STRICT_SLA_LABEL} requires SM75 native CUDA or SM80+ Triton; "
            f"this GPU is SM{capability[0]}{capability[1]}. No fallback was attempted."
        )
    return capability


def _activate_sm75_torch_preprocess(reason: BaseException | str) -> None:
    global _SM75_TORCH_PREPROCESS
    if _SM75_TORCH_PREPROCESS:
        return
    _SM75_TORCH_PREPROCESS = True
    _LOG.warning(
        "[Star7 H3 Chunk] SM75 Triton preprocessing is unavailable (%s); "
        "using bounded-memory PyTorch routing/quantization with the same "
        "native CUDA SLA core.",
        reason,
    )


def _mean_pool_torch(
    x: torch.Tensor, block: int, mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bounded-memory SM75 fallback; never materializes full FP32 Q/K."""
    batch, heads, length, head_dim = x.shape
    blocks = (length + block - 1) // block
    output = torch.empty(
        (batch, heads, blocks, head_dim), dtype=x.dtype, device=x.device
    )
    blocks_per_chunk = max(1, 4096 // block)
    mean_fp32 = None if mean is None else mean.float()
    for block_start in range(0, blocks, blocks_per_chunk):
        block_end = min(blocks, block_start + blocks_per_chunk)
        token_start = block_start * block
        token_end = min(length, block_end * block)
        valid_tokens = token_end - token_start
        padded_tokens = (block_end - block_start) * block
        values = x[:, :, token_start:token_end].float()
        if valid_tokens < padded_tokens:
            values = torch.nn.functional.pad(
                values, (0, 0, 0, padded_tokens - valid_tokens)
            )
        values = values.view(
            batch, heads, block_end - block_start, block, head_dim
        )
        if mean_fp32 is not None:
            values.sub_(mean_fp32.unsqueeze(-2))
            if valid_tokens < padded_tokens:
                values[:, :, -1, valid_tokens % block :] = 0.0
        sums = values.sum(dim=-2)
        counts = torch.full(
            (block_end - block_start,), block,
            dtype=torch.float32, device=x.device,
        )
        if block_end == blocks and length % block:
            counts[-1] = length % block
        output[:, :, block_start:block_end].copy_(
            (sums / counts.view(1, 1, -1, 1)).to(x.dtype)
        )
    return output


def _quantize_torch(
    x: torch.Tensor,
    block: int,
    multiplier: float,
    mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, head_dim = x.shape
    blocks = (length + block - 1) // block
    quantized = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(
        (batch, heads, blocks), dtype=torch.float32, device=x.device
    )
    blocks_per_chunk = max(1, 4096 // block)
    mean_fp32 = None if mean is None else mean.float()
    for block_start in range(0, blocks, blocks_per_chunk):
        block_end = min(blocks, block_start + blocks_per_chunk)
        token_start = block_start * block
        token_end = min(length, block_end * block)
        valid_tokens = token_end - token_start
        padded_tokens = (block_end - block_start) * block
        values = x[:, :, token_start:token_end].float()
        if valid_tokens < padded_tokens:
            values = torch.nn.functional.pad(
                values, (0, 0, 0, padded_tokens - valid_tokens)
            )
        values = values.view(
            batch, heads, block_end - block_start, block, head_dim
        )
        if mean_fp32 is not None:
            values.sub_(mean_fp32.unsqueeze(-2))
            if valid_tokens < padded_tokens:
                values[:, :, -1, valid_tokens % block :] = 0.0
        values.mul_(multiplier)
        local_scale = values.abs().amax(dim=(-1, -2)).div_(127.0).clamp_min_(1.0e-8)
        values.div_(local_scale[..., None, None])
        values.add_(torch.where(values >= 0.0, 0.5, -0.5))
        values.trunc_().clamp_(-127.0, 127.0)
        packed = values.to(torch.int8).view(
            batch, heads, padded_tokens, head_dim
        )[:, :, :valid_tokens]
        quantized[:, :, token_start:token_end].copy_(packed)
        scale[:, :, block_start:block_end].copy_(local_scale)
    return quantized, scale


def _mean_pool(x: torch.Tensor, block: int, mean: torch.Tensor | None = None) -> torch.Tensor:
    if _SM75_TORCH_PREPROCESS or triton is None:
        return _mean_pool_torch(x, block, mean)
    batch, heads, length, head_dim = x.shape
    blocks = triton.cdiv(length, block)
    output = torch.empty((batch, heads, blocks, head_dim), dtype=x.dtype, device=x.device)
    placeholder = x if mean is None else mean
    try:
        _mean_pool_kernel[(blocks, batch * heads)](
            x,
            placeholder,
            output,
            length,
            head_dim,
            block,
            subtract_mean=mean is not None,
            num_warps=8 if block == BLOCK_Q else 4,
        )
    except Exception as exc:
        if torch.cuda.get_device_capability(x.device) != (7, 5):
            raise
        del output
        _activate_sm75_torch_preprocess(exc)
        return _mean_pool_torch(x, block, mean)
    return output


def build_routing_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    sparsity: float = DEFAULT_SPARSITY,
    query_priority_ranges: tuple[tuple[int, int], ...] = (),
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Return LightX2V-compatible block routing and the K mean used by quantization."""
    if not 0.0 <= float(sparsity) < 1.0:
        raise ValueError(f"SLA sparsity must be in [0,1); got {sparsity}")
    k_mean = k.mean(dim=-2, keepdim=True, dtype=torch.float32).to(k.dtype)
    pooled_q = _mean_pool(q, BLOCK_Q)
    # A 128-row router block can straddle the packed audio/video boundary.
    # Without this override the first video rows dominate routing for the last
    # right-channel audio frames, which produces the characteristic tail burst.
    for range_start, range_end in query_priority_ranges:
        first_block = max(0, range_start // BLOCK_Q)
        last_block = min(pooled_q.shape[-2], (range_end + BLOCK_Q - 1) // BLOCK_Q)
        for block_index in range(first_block, last_block):
            block_start = block_index * BLOCK_Q
            block_end = min(block_start + BLOCK_Q, q.shape[-2])
            start = max(block_start, range_start)
            end = min(block_end, range_end)
            if start < end and (start != block_start or end != block_end):
                pooled_q[:, :, block_index] = q[:, :, start:end].mean(
                    dim=-2, dtype=torch.float32
                ).to(q.dtype)
    pooled_k = _mean_pool(k, BLOCK_K, k_mean)
    scores = pooled_q @ pooled_k.transpose(-1, -2)
    key_blocks = scores.shape[-1]
    selected = max(1, min(key_blocks, int((1.0 - float(sparsity)) * key_blocks)))
    lut = torch.topk(scores, selected, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
    if debug:
        _log_debug_tensor("routing pooled_q", pooled_q)
        _log_debug_tensor("routing pooled_k", pooled_k)
        _log_debug_tensor("routing scores", scores)
        lut_min = int(lut.min().item()) if lut.numel() else -1
        lut_max = int(lut.max().item()) if lut.numel() else -1
        out_of_range = int(((lut < 0) | (lut >= key_blocks)).sum().item())
        _LOG.info(
            "[Star7 H3 Chunk] SLA debug | LUT min=%d max=%d "
            "out-of-range=%d selected=%d",
            lut_min, lut_max, out_of_range, selected,
        )
    return lut, k_mean, scores.shape[-2], key_blocks, selected


def _quantize(
    x: torch.Tensor,
    block: int,
    multiplier: float,
    mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _SM75_TORCH_PREPROCESS or triton is None:
        return _quantize_torch(x, block, multiplier, mean)
    batch, heads, length, head_dim = x.shape
    blocks = triton.cdiv(length, block)
    quantized = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((batch, heads, blocks), dtype=torch.float32, device=x.device)
    placeholder = x if mean is None else mean
    try:
        _quantize_per_block_int8_kernel[(blocks, batch * heads)](
            x,
            placeholder,
            quantized,
            scale,
            length,
            head_dim,
            block,
            multiplier=multiplier,
            subtract_mean=mean is not None,
            num_warps=8 if block == BLOCK_Q else 4,
        )
    except Exception as exc:
        if torch.cuda.get_device_capability(x.device) != (7, 5):
            raise
        del quantized, scale
        _activate_sm75_torch_preprocess(exc)
        return _quantize_torch(x, block, multiplier, mean)
    return quantized, scale


def _log_debug_tensor(stage: str, value: torch.Tensor) -> None:
    """Synchronizing tensor summary used only by the targeted debug block."""
    finite = torch.isfinite(value)
    nan_count = int(torch.isnan(value).sum().item())
    inf_count = int(torch.isinf(value).sum().item())
    all_finite = bool(finite.all().item())
    if all_finite and value.numel():
        minimum = float(value.min().item())
        maximum = float(value.max().item())
        max_abs = float(value.abs().max().item())
    else:
        minimum = maximum = max_abs = float("nan")
    _LOG.info(
        "[Star7 H3 Chunk] SLA debug | %s | dtype=%s shape=%s "
        "nan=%d inf=%d min=%.6g max=%.6g max-abs=%.6g",
        stage, value.dtype, tuple(value.shape), nan_count, inf_count,
        minimum, maximum, max_abs,
    )


def _audio_guard_status(
    requested_ranges: tuple[tuple[int, int], ...],
    applied_ranges: tuple[tuple[int, int], ...],
) -> str:
    if not requested_ranges:
        return "not-requested"
    if applied_ranges:
        return "full-attention-applied"
    return "routing-priority-only"


def _run_raw_impl(
    owned_qkv: list[torch.Tensor],
    sparsity: float,
    query_priority_ranges: tuple[tuple[int, int], ...] = (),
    all_int8: bool = False,
    release_inputs: bool = False,
    debug: bool = False,
) -> SLAResult:
    if len(owned_qkv) != 3:
        raise ValueError("SLA requires exactly Q, K, and V tensors")
    q, k, v = owned_qkv
    if release_inputs:
        owned_qkv.clear()
    _require_environment(q, k, v)
    lut, k_mean, query_blocks, key_blocks, selected = build_routing_lut(
        q, k, sparsity, query_priority_ranges,
        debug=debug,
    )
    capability = torch.cuda.get_device_capability(q.device)
    applied_priority_ranges: tuple[tuple[int, int], ...] = ()
    if capability == (7, 5):
        # Match the query warp consumed by the SM75 kernel. Per-16-row scales
        # avoid a mixed audio/video CTA allowing much larger video Q values to
        # erase the last audio queries during INT8 quantization.
        # Quantize Q and K one at a time. Keeping both full FP16 sources alive
        # while allocating both INT8 copies creates an avoidable peak for long
        # H3 sequences. The kernels only need the quantized tensors after this
        # point, so release each FP16 source immediately after its copy exists.
        q_int8, q_scale = _quantize(q, 16, multiplier=1.0)
        del q
        required_q_scales = query_blocks * 8
        if q_scale.shape[-1] < required_q_scales:
            q_scale = torch.nn.functional.pad(
                q_scale, (0, required_q_scales - q_scale.shape[-1]), value=1.0
            )
        k_int8, k_scale = _quantize(k, BLOCK_K, multiplier=1.0, mean=k_mean)
        del k, k_mean
        v_input: torch.Tensor | list[torch.Tensor] = v
        if release_inputs:
            # Routing and Q/K quantization have finished. Drop the two large
            # FP16 sources before allocating the native output and, in the
            # All-INT8 mode, transfer sole V ownership to the native wrapper so
            # it can recycle V storage immediately after V quantization.
            if all_int8:
                v_input = [v]
                del v
        output = _load_sm75_backend().run(
            q_int8, k_int8, v_input, q_scale, k_scale, lut,
            dense_query_ranges=query_priority_ranges,
            all_int8=all_int8,
        )
        applied_priority_ranges = query_priority_ranges
        implementation = (
            "cuda-sm75-sparse-all-int8-tensorcore+audio-guard"
            if all_int8 else
            "cuda-sm75-sparse-qk-int8-pv-fp16-tensorcore+audio-guard"
        )
        if _SM75_TORCH_PREPROCESS:
            implementation += "+torch-preprocess"
        dense_guard_status = _audio_guard_status(
            query_priority_ranges, applied_priority_ranges
        )
    else:
        if all_int8:
            raise SLAUnavailableError(
                "Experimental All-INT8 SLA is currently available only on SM75"
            )
        # Keep the full-sequence preprocessing peak bounded: after each source
        # is quantized, its FP16 storage is no longer needed by the Triton
        # attention kernel.
        q_int8, q_scale = _quantize(
            q, BLOCK_Q, multiplier=(HEAD_DIM ** -0.5) * _LOG2E,
        )
        del q
        k_int8, k_scale = _quantize(
            k, BLOCK_K, multiplier=1.0, mean=k_mean,
        )
        del k, k_mean
        if debug:
            _log_debug_tensor("q_scale", q_scale)
            _log_debug_tensor("k_scale", k_scale)
        output = torch.empty_like(v)
        compute_query_blocks = triton.cdiv(q_int8.shape[-2], 64)
        _sparse_qk_int8_pv_fp16_kernel[(compute_query_blocks, q_int8.shape[0] * q_int8.shape[1])](
            q_int8,
            k_int8,
            v,
            q_scale,
            k_scale,
            lut,
            output,
            q.shape[-2],
            query_blocks,
            compute_query_blocks,
            key_blocks,
            selected,
            HEAD_DIM,
            BLOCK_Q,
            64,
            BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        implementation = "triton-sm80plus-int8-dot-fp16-dot"
        dense_guard_status = _audio_guard_status(
            query_priority_ranges, applied_priority_ranges
        )
        if debug:
            _log_debug_tensor("raw SLA output", output)
    protected = {
        block_index
        for start, end in applied_priority_ranges
        for block_index in range(
            max(0, start // BLOCK_Q), min(query_blocks, (end + BLOCK_Q - 1) // BLOCK_Q)
        )
    }
    effective_sparsity = 1.0 - (
        selected * (query_blocks - len(protected)) + key_blocks * len(protected)
    ) / (query_blocks * key_blocks)
    return SLAResult(
        output, query_blocks, key_blocks, selected, effective_sparsity,
        implementation, len(protected), dense_guard_status,
    )


def _run_raw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float,
    query_priority_ranges: tuple[tuple[int, int], ...] = (),
    all_int8: bool = False,
    debug: bool = False,
) -> SLAResult:
    """Non-consuming entry point retained for self-tests and external callers."""
    return _run_raw_impl(
        [q, k, v], sparsity, query_priority_ranges, all_int8,
        release_inputs=False, debug=debug,
    )


def _reference_from_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Small-tensor golden reference used only by the one-time self-test."""
    output = torch.empty_like(v)
    scale = HEAD_DIM ** -0.5
    length = q.shape[-2]
    for query_block in range(lut.shape[-2]):
        q_start = query_block * BLOCK_Q
        q_end = min(q_start + BLOCK_Q, length)
        selected = lut[0, 0, query_block].tolist()
        indices = torch.cat([
            torch.arange(index * BLOCK_K, min((index + 1) * BLOCK_K, length), device=q.device)
            for index in selected
        ])
        q_part = q[:, :, q_start:q_end].float()
        k_part = k.index_select(-2, indices).float()
        v_part = v.index_select(-2, indices).float()
        probability = torch.softmax(q_part @ k_part.transpose(-1, -2) * scale, dim=-1)
        output[:, :, q_start:q_end] = (probability @ v_part).to(output.dtype)
    return output


_SELF_TESTED_DEVICES: set[tuple[str, int | None, bool]] = set()
_LONG_SELF_TESTED_DEVICES: set[tuple[str, int | None]] = set()
_SELF_TEST_LOCK = threading.Lock()


def ensure_self_test(device: torch.device, all_int8: bool = False) -> None:
    key = (device.type, device.index, all_int8)
    if key in _SELF_TESTED_DEVICES:
        return
    with _SELF_TEST_LOCK:
        if key in _SELF_TESTED_DEVICES:
            return
        generator = torch.Generator(device=device)
        generator.manual_seed(0x57A7)
        # 1025 exercises multiple selected K blocks plus both Q/K tail masks.
        q = torch.randn((1, 1, 1025, HEAD_DIM), generator=generator, device=device, dtype=torch.float16) * 0.25
        k = torch.randn(q.shape, generator=generator, device=device, dtype=torch.float16) * 0.25
        v = torch.randn(q.shape, generator=generator, device=device, dtype=torch.float16) * 0.25
        result = _run_raw(
            q.contiguous(), k.contiguous(), v.contiguous(), DEFAULT_SPARSITY,
            all_int8=all_int8,
        )
        lut, _mean, _qb, _kb, _selected = build_routing_lut(q, k, DEFAULT_SPARSITY)
        reference = _reference_from_lut(q, k, v, lut)
        torch.cuda.synchronize(device)
        if not torch.isfinite(result.output).all():
            raise RuntimeError(f"{STRICT_SLA_LABEL} self-test produced NaN/Inf")
        error = (result.output.float() - reference.float()).abs()
        mean_limit = 0.01 if all_int8 else 0.001
        max_limit = 0.08 if all_int8 else 0.02
        if error.mean().item() > mean_limit or error.max().item() > max_limit:
            raise RuntimeError(
                f"{STRICT_SLA_LABEL} self-test failed: mean_abs={error.mean().item():.6f}, "
                f"max_abs={error.max().item():.6f}"
            )
        _SELF_TESTED_DEVICES.add(key)
        capability = torch.cuda.get_device_capability(device)
        _LOG.info(
            "[Star7 H3 Chunk] SLA self-test passed | test=S1025/H1/D128 | "
            "SM%d%d | all-int8=%s | mean-abs=%.6g | max-abs=%.6g",
            capability[0], capability[1], all_int8,
            error.mean().item(), error.max().item(),
        )


def ensure_optional_long_shape_test(device: torch.device) -> None:
    """Opt-in SM100+ diagnostic matching the reported S=16206 route length."""
    if os.environ.get("STAR7_SLA_LONG_SELF_TEST", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return
    if torch.cuda.get_device_capability(device) < (10, 0):
        return
    key = (device.type, device.index)
    if key in _LONG_SELF_TESTED_DEVICES:
        return
    with _SELF_TEST_LOCK:
        if key in _LONG_SELF_TESTED_DEVICES:
            return
        generator = torch.Generator(device=device)
        generator.manual_seed(0x120294)
        q = torch.randn(
            (1, 1, 16206, HEAD_DIM), generator=generator,
            device=device, dtype=torch.float16,
        ) * 0.1
        k = torch.randn(q.shape, generator=generator, device=device, dtype=torch.float16) * 0.1
        v = torch.randn(q.shape, generator=generator, device=device, dtype=torch.float16) * 0.1
        result = _run_raw(q, k, v, DEFAULT_SPARSITY)
        lut, _mean, _qb, _kb, selected = build_routing_lut(q, k, DEFAULT_SPARSITY)
        reference = _reference_from_lut(q, k, v, lut)
        torch.cuda.synchronize(device)
        if not torch.isfinite(result.output).all():
            raise RuntimeError(f"{STRICT_SLA_LABEL} long-shape test produced NaN/Inf")
        error = (result.output.float() - reference.float()).abs()
        if error.mean().item() > 0.001 or error.max().item() > 0.02:
            raise RuntimeError(
                f"{STRICT_SLA_LABEL} long-shape test failed: "
                f"mean_abs={error.mean().item():.6f}, max_abs={error.max().item():.6f}"
            )
        _LONG_SELF_TESTED_DEVICES.add(key)
        capability = torch.cuda.get_device_capability(device)
        _LOG.info(
            "[Star7 H3 Chunk] SLA long-shape test passed | "
            "test=S16206/H1/D128 | SM%d%d | selected=%d | "
            "mean-abs=%.6g | max-abs=%.6g",
            capability[0], capability[1], selected,
            error.mean().item(), error.max().item(),
        )


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float = DEFAULT_SPARSITY,
    run_self_test: bool = True,
    query_priority_ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    all_int8: bool = False,
    debug: bool = False,
) -> SLAResult:
    """Execute strict SLA. No dense or CK fallback is performed."""
    _require_environment(q, k, v)
    if run_self_test:
        ensure_self_test(q.device, all_int8=all_int8)
        if not all_int8:
            ensure_optional_long_shape_test(q.device)
    result = _run_raw(
        q, k, v, float(sparsity), tuple(query_priority_ranges),
        all_int8=all_int8, debug=debug,
    )
    return result


def sparse_attention_consume(
    qkv: list[torch.Tensor],
    sparsity: float = DEFAULT_SPARSITY,
    run_self_test: bool = True,
    query_priority_ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    all_int8: bool = False,
    debug: bool = False,
) -> SLAResult:
    """Execute strict SLA while consuming Q/K/V to minimize the core peak.

    The caller must relinquish its individual tensor references before calling.
    The list is cleared as soon as this function transfers ownership.
    """
    if len(qkv) != 3:
        raise ValueError("SLA consume API requires [Q, K, V]")
    _require_environment(qkv[0], qkv[1], qkv[2])
    device = qkv[0].device
    if run_self_test:
        ensure_self_test(device, all_int8=all_int8)
        if not all_int8:
            ensure_optional_long_shape_test(device)
    return _run_raw_impl(
        qkv, float(sparsity), tuple(query_priority_ranges), all_int8,
        release_inputs=True, debug=debug,
    )
