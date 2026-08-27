"""MiniMax H3 Sol attention dispatch.

SM80+ recommended mode calls NVIDIA's public Sol-Attn interface unchanged.
SM75 uses Star7 Q64/K64 threshold routing, exact selected K/V blocks, and
centroid contributions for unselected blocks in one online softmax. FP16-PV
and experimental All-INT8 differ only in PV quantization, not Sol semantics.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


SOL_SM75_BACKEND_NAME = "sol_sm75_qk_int8_pv_fp16"
SOL_SM75_ALL_INT8_BACKEND_NAME = "sol_sm75_all_int8"
LEGACY_SOL_SM75_ALL_INT8_BACKEND_NAME = "sol_sm75_all_int8_experimental"
SOL_SM86PLUS_BACKEND_NAME = "sol_sm80+_bf16_official"
SOL_SM86PLUS_ALL_INT8_BACKEND_NAME = "sol_sm80+_all_int8"
LEGACY_SOL_SM86PLUS_ALL_INT8_BACKEND_NAME = "sol_sm80+_all_int8_experimental"
SOL_BLOCK_Q = 64
SOL_BLOCK_K = 64
HEAD_DIM = 128
DEFAULT_TAU = 1.0
DEFAULT_TOPK_BLOCKS = 32
ROUTE_CHUNK = 64
_LOG2E = 1.4426950408889634


class SolUnavailableError(RuntimeError):
    pass


@dataclass
class SolResult:
    output: torch.Tensor
    query_blocks: int
    key_blocks: int
    min_selected_blocks: int
    max_selected_blocks: int
    mean_density: float
    implementation: str
    routing_tau: float


def _load_sm75_backend():
    try:
        from . import sol_sm75_backend
        return sol_sm75_backend
    except ImportError:
        module_name = "star7_sol_sm75_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        path = Path(__file__).with_name("sol_sm75_backend.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Star7 SM75 Sol backend from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _official_module():
    search_paths = []
    extra_path = os.environ.get("STAR7_SOL_ATTN_PATH", "").strip()
    if extra_path:
        root = Path(extra_path).expanduser().resolve()
        search_paths.append(root)
        repo_backend = root / "techniques" / "sparse_backends"
        if repo_backend.is_dir():
            search_paths.insert(0, repo_backend)
        if root.name == "sol_attn":
            search_paths.insert(0, root.parent)
    bundled = Path(__file__).with_name("vendor")
    if (bundled / "sol_attn" / "interface.py").is_file():
        search_paths.append(bundled)
    for search_path in reversed(search_paths):
        value = str(search_path)
        if value not in sys.path:
            sys.path.insert(0, value)
    candidates = (
        "sol_attn.interface",
        "techniques.sparse_backends.sol_attn.interface",
    )
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise SolUnavailableError(
        "The bundled NVIDIA Sol-Attn module could not be imported. Reinstall "
        "the complete Star7 node package. STAR7_SOL_ATTN_PATH may optionally "
        "point to a newer NVlabs/Sana sol-engine checkout. "
        f"Import attempts: {'; '.join(errors)}"
    )


def check_runtime_support(
    requested_backend: str,
    device: torch.device | int | None = None,
) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise SolUnavailableError("Star7 H3 Sol requires NVIDIA CUDA")
    capability = torch.cuda.get_device_capability(device)
    sm75_names = {
        SOL_SM75_BACKEND_NAME,
        SOL_SM75_ALL_INT8_BACKEND_NAME,
        LEGACY_SOL_SM75_ALL_INT8_BACKEND_NAME,
    }
    sm86_names = {
        SOL_SM86PLUS_BACKEND_NAME,
        SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
        LEGACY_SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
    }
    if requested_backend not in sm75_names | sm86_names:
        raise SolUnavailableError(f"unknown Sol backend: {requested_backend}")
    if requested_backend in sm75_names:
        if capability != (7, 5):
            raise SolUnavailableError(
                f"{requested_backend} requires exactly SM75; got "
                f"SM{capability[0]}{capability[1]}. No fallback was attempted."
            )
        available, reason = _load_sm75_backend().availability()
        if not available:
            raise SolUnavailableError(reason)
    else:
        if capability < (8, 0):
            raise SolUnavailableError(
                f"{requested_backend} requires SM80 or newer; got "
                f"SM{capability[0]}{capability[1]}. No fallback was attempted."
            )
        if requested_backend == SOL_SM86PLUS_BACKEND_NAME:
            official = _official_module()
            official.get_sol_attn_backend(device)
        else:
            sla = _load_sla_backend()
            if sla.triton is None:
                raise SolUnavailableError(
                    "SM80+ experimental All-INT8 Sol requires Triton"
                )
    return capability


def run_official(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> SolResult:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("official Sol requires equal [B,T,H,128] Q/K/V")
    if q.dtype != torch.bfloat16:
        raise TypeError("official SM80+ Sol requires BF16 Q/K/V")
    if q.shape[-1] != HEAD_DIM:
        raise ValueError("official Sol requires head_dim=128")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("official Sol requires contiguous BTHD tensors")
    official = _official_module()
    output = official.sol_attn(
        q,
        k,
        v,
        scale=HEAD_DIM ** -0.5,
        tau=float(tau),
        thresh_type="diag",
        sink_tokens=int(sink_tokens),
        sink_start=sink_start,
    )
    blocks = (q.shape[1] + SOL_BLOCK_Q - 1) // SOL_BLOCK_Q
    return SolResult(
        output=output,
        query_blocks=blocks,
        key_blocks=blocks,
        min_selected_blocks=-1,
        max_selected_blocks=-1,
        mean_density=float("nan"),
        implementation=f"nvidia-official-{official.get_sol_attn_backend(q.device)}",
        routing_tau=float(tau),
    )


def _load_sla_backend():
    try:
        from . import sla_backend
        return sla_backend
    except ImportError:
        module_name = "star7_sla_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        path = Path(__file__).with_name("sla_backend.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Star7 SLA helpers from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _block_mean_fp32(x: torch.Tensor, block: int = SOL_BLOCK_Q) -> torch.Tensor:
    batch, heads, length, dim = x.shape
    blocks = (length + block - 1) // block
    padded = blocks * block
    values = x.float()
    if padded != length:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded - length))
    values = values.view(batch, heads, blocks, block, dim)
    sums = values.sum(dim=-2)
    counts = torch.full(
        (blocks,), block, dtype=torch.float32, device=x.device,
    )
    if length % block:
        counts[-1] = length % block
    return sums / counts.view(1, 1, blocks, 1)


def build_custom_routing(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
    topk_blocks: int = DEFAULT_TOPK_BLOCKS,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    neighbor: int = 1,
    return_aux: bool = False,
):
    """Build a bounded-memory variable-count Q64/K64 threshold LUT."""
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError("custom Sol routing requires equal [B,H,L,128] Q/K")
    batch, heads, length, _ = q.shape
    qc = _block_mean_fp32(q)
    kc = _block_mean_fp32(k)
    query_blocks, key_blocks = qc.shape[-2], kc.shape[-2]
    kc_mean = kc.mean(dim=-2)
    kc_var = (kc.square().mean(dim=-2) - kc_mean.square()).clamp_min_(0.0)
    scale_log2 = (HEAD_DIM ** -0.5) * _LOG2E
    key_indices = torch.arange(key_blocks, device=q.device)
    sink_first = key_blocks
    sink_last = key_blocks
    if sink_tokens > 0:
        sink_start = length - sink_tokens if sink_start is None else int(sink_start)
        sink_first = max(0, sink_start // SOL_BLOCK_K)
        sink_last = min(
            key_blocks,
            (sink_start + int(sink_tokens) + SOL_BLOCK_K - 1) // SOL_BLOCK_K,
        )

    counts_chunks = []
    packed_chunks = []
    exact_mask_chunks = []
    max_slots = 0
    for q_start in range(0, query_blocks, ROUTE_CHUNK):
        q_end = min(query_blocks, q_start + ROUTE_CHUNK)
        q_part = qc[:, :, q_start:q_end]
        mean = (q_part * kc_mean.unsqueeze(-2)).sum(dim=-1) * scale_log2
        variance = (
            q_part.square() * kc_var.unsqueeze(-2)
        ).sum(dim=-1) * (scale_log2 * scale_log2)
        threshold = mean + float(tau) * torch.sqrt(variance + 1.0e-6)
        scores = torch.matmul(q_part, kc.transpose(-1, -2)) * scale_log2
        selected = scores >= threshold.unsqueeze(-1)
        if topk_blocks > 0:
            floor = min(int(topk_blocks), key_blocks)
            topk = torch.topk(scores, floor, dim=-1, sorted=False).indices
            selected.scatter_(-1, topk, True)
        if neighbor > 0:
            q_indices = torch.arange(q_start, q_end, device=q.device)
            near = (q_indices[:, None] - key_indices[None, :]).abs() <= int(neighbor)
            selected |= near.view(1, 1, q_end - q_start, key_blocks)
        if sink_first < sink_last:
            selected[..., sink_first:sink_last] = True
        counts = selected.sum(dim=-1, dtype=torch.int32)
        slots = int(counts.max().item())
        keys = key_indices.view(1, 1, 1, key_blocks).expand_as(selected)
        packed = torch.where(selected, keys, key_blocks).sort(dim=-1).values
        packed = packed[..., :slots].to(torch.int32).contiguous()
        counts_chunks.append(counts)
        packed_chunks.append(packed)
        if return_aux:
            exact_mask_chunks.append(selected.to(torch.uint8).contiguous())
        max_slots = max(max_slots, slots)
        del scores, selected, keys

    padded_chunks = []
    for packed in packed_chunks:
        if packed.shape[-1] < max_slots:
            packed = torch.nn.functional.pad(
                packed, (0, max_slots - packed.shape[-1]), value=0,
            )
        padded_chunks.append(packed)
    row_count = torch.cat(counts_chunks, dim=2).contiguous()
    lut = torch.cat(padded_chunks, dim=2).contiguous()
    density = float(row_count.float().mean().item() / key_blocks)
    if return_aux:
        exact_mask = torch.cat(exact_mask_chunks, dim=2).contiguous()
        return row_count, lut, density, exact_mask, kc.to(dtype=q.dtype)
    return row_count, lut, density


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sol_qk_int8_pv_int8_kernel(
        Q, K, V, K_CENTROID, V_CENTROID,
        Q_SCALE, K_SCALE, V_SCALE, KC_SCALE, VC_SCALE,
        ROW_COUNT, LUT, EXACT_MASK, OUT,
        length: tl.constexpr, query_blocks: tl.constexpr,
        key_blocks: tl.constexpr, lut_stride: tl.constexpr,
        centroid_groups: tl.constexpr,
        head_dim: tl.constexpr, block_q: tl.constexpr, block_k: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        q_offsets = query_block * block_q + tl.arange(0, block_q)
        k_offsets = tl.arange(0, block_k)
        dims = tl.arange(0, head_dim)
        base = bh * length * head_dim
        q = tl.load(
            Q + base + q_offsets[:, None] * head_dim + dims[None, :],
            mask=q_offsets[:, None] < length,
            other=0,
        ).to(tl.int8)
        q_scale = tl.load(
            Q_SCALE + bh * query_blocks * 4 + q_offsets // 16,
        )
        route_row = bh * query_blocks + query_block
        selected_count = tl.load(ROW_COUNT + route_row)
        lut_base = route_row * lut_stride
        row_max = tl.full([block_q], -float("inf"), tl.float32)
        row_sum = tl.zeros([block_q], tl.float32)
        accumulator = tl.zeros([block_q, head_dim], tl.float32)
        for selected_index in tl.range(0, selected_count):
            key_block = tl.load(LUT + lut_base + selected_index)
            key_tokens = key_block * block_k + k_offsets
            valid = key_tokens < length
            k = tl.load(
                K + base + key_tokens[None, :] * head_dim + dims[:, None],
                mask=valid[None, :], other=0,
            ).to(tl.int8)
            k_scale = tl.load(K_SCALE + bh * key_blocks + key_block)
            score = tl.dot(q, k).to(tl.float32)
            score *= q_scale[:, None] * k_scale * (
                (head_dim ** -0.5) * 1.4426950408889634
            )
            score = tl.where(valid[None, :], score, -float("inf"))
            local_max = tl.max(score, axis=1)
            new_max = tl.maximum(row_max, local_max)
            probability = tl.math.exp2(score - new_max[:, None])
            correction = tl.math.exp2(row_max - new_max)
            accumulator *= correction[:, None]
            probability_scale = tl.maximum(tl.max(probability, axis=1) / 127.0, 1.0e-8)
            probability_int8 = tl.minimum(
                probability / probability_scale[:, None] + 0.5, 127.0,
            ).to(tl.int8)
            value = tl.load(
                V + base + key_tokens[:, None] * head_dim + dims[None, :],
                mask=valid[:, None], other=0,
            ).to(tl.int8)
            value_scale = tl.load(V_SCALE + bh * key_blocks + key_block)
            accumulator += (
                tl.dot(probability_int8, value).to(tl.float32)
                * probability_scale[:, None] * value_scale
            )
            row_sum = row_sum * correction + tl.sum(probability, axis=1)
            row_max = new_max

        # Exact tokens and unselected-block centroids share one online softmax.
        # All-INT8 changes PV arithmetic only; it does not remove Sol's
        # approximate contribution.
        centroid_base = bh * key_blocks * head_dim
        exact_mask_base = route_row * key_blocks
        for centroid_group in tl.range(0, centroid_groups):
            centroid_offsets = centroid_group * block_k + k_offsets
            centroid_valid = centroid_offsets < key_blocks
            is_exact = tl.load(
                EXACT_MASK + exact_mask_base + centroid_offsets,
                mask=centroid_valid, other=1,
            ).to(tl.int1)
            approximate = centroid_valid & ~is_exact
            centroid_k = tl.load(
                K_CENTROID + centroid_base
                + centroid_offsets[None, :] * head_dim + dims[:, None],
                mask=centroid_valid[None, :], other=0,
            ).to(tl.int8)
            centroid_k_scale = tl.load(
                KC_SCALE + bh * centroid_groups + centroid_group,
            )
            centroid_score = tl.dot(q, centroid_k).to(tl.float32)
            centroid_score *= (
                q_scale[:, None] * centroid_k_scale
                * ((head_dim ** -0.5) * 1.4426950408889634)
            )
            represented_tokens = tl.maximum(
                1, tl.minimum(block_k, length - centroid_offsets * block_k),
            ).to(tl.float32)
            centroid_score += tl.log2(represented_tokens)[None, :]
            centroid_score = tl.where(
                approximate[None, :], centroid_score, -float("inf"),
            )
            local_max = tl.max(centroid_score, axis=1)
            new_max = tl.maximum(row_max, local_max)
            probability = tl.math.exp2(centroid_score - new_max[:, None])
            correction = tl.math.exp2(row_max - new_max)
            accumulator *= correction[:, None]
            probability_scale = tl.maximum(
                tl.max(probability, axis=1) / 127.0, 1.0e-8,
            )
            probability_int8 = tl.minimum(
                probability / probability_scale[:, None] + 0.5, 127.0,
            ).to(tl.int8)
            centroid_v = tl.load(
                V_CENTROID + centroid_base
                + centroid_offsets[:, None] * head_dim + dims[None, :],
                mask=centroid_valid[:, None], other=0,
            ).to(tl.int8)
            centroid_v_scale = tl.load(
                VC_SCALE + bh * centroid_groups + centroid_group,
            )
            accumulator += (
                tl.dot(probability_int8, centroid_v).to(tl.float32)
                * probability_scale[:, None] * centroid_v_scale
            )
            row_sum = row_sum * correction + tl.sum(probability, axis=1)
            row_max = new_max
        accumulator /= row_sum[:, None]
        tl.store(
            OUT + base + q_offsets[:, None] * head_dim + dims[None, :],
            accumulator.to(OUT.type.element_ty),
            mask=q_offsets[:, None] < length,
        )


def run_custom_consume(
    owned_qkv: list[torch.Tensor],
    *,
    all_int8: bool,
    tau: float = DEFAULT_TAU,
    topk_blocks: int = DEFAULT_TOPK_BLOCKS,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> SolResult:
    if len(owned_qkv) != 3:
        raise ValueError("custom Sol requires Q/K/V")
    q, k, v = owned_qkv
    owned_qkv.clear()
    if not q.is_cuda or q.shape != k.shape or q.shape != v.shape:
        raise SolUnavailableError("custom Sol requires equal CUDA Q/K/V")
    if q.dtype != torch.float16 or q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise SolUnavailableError("custom Sol requires FP16 [B,H,L,128] Q/K/V")
    capability = torch.cuda.get_device_capability(q.device)
    sm75 = capability == (7, 5)
    sm75_exact_approx = sm75
    if sm75:
        native = _load_sm75_backend()
        routing = native.prepare(
            q, k, v, tau=tau,
            sink_tokens=sink_tokens, sink_start=sink_start,
        )
        row_count = routing["row_count"]
        lut = routing["lut"]
        density = routing["density"]
        exact_mask = routing["exact_mask"]
        k_centroid = routing["k_centroid"]
        v_centroid = routing["v_centroid"]
        minimum = routing["minimum"]
        maximum = routing["maximum"]
    else:
        routing = build_custom_routing(
            q, k, tau=tau, topk_blocks=topk_blocks,
            sink_tokens=sink_tokens, sink_start=sink_start,
            return_aux=all_int8,
        )
        if all_int8:
            row_count, lut, density, exact_mask, k_centroid = routing
            v_centroid = _block_mean_fp32(v).to(dtype=v.dtype)
        else:
            row_count, lut, density = routing
    query_blocks = row_count.shape[-1]
    key_blocks = (q.shape[-2] + SOL_BLOCK_K - 1) // SOL_BLOCK_K
    if not sm75:
        minimum = int(row_count.min().item())
        maximum = int(row_count.max().item())
    sla = _load_sla_backend()
    if sm75:
        q_int8, q_scale = native.quantize(q, 16)
    else:
        q_int8, q_scale = sla._quantize(q, 16, multiplier=1.0)
    del q
    required_q_scales = query_blocks * 4
    if q_scale.shape[-1] < required_q_scales:
        q_scale = torch.nn.functional.pad(
            q_scale, (0, required_q_scales - q_scale.shape[-1]), value=1.0,
        )
    if sm75:
        k_int8, k_scale = native.quantize(k, SOL_BLOCK_K)
    else:
        k_int8, k_scale = sla._quantize(k, SOL_BLOCK_K, multiplier=1.0)
    del k
    if capability == (7, 5):
        if all_int8:
            v_input: torch.Tensor | list[torch.Tensor] = [v]
            del v
        else:
            v_input = v
        approximation = None
        if sm75_exact_approx:
            k_centroid_int8, k_centroid_scale = native.quantize(
                k_centroid, SOL_BLOCK_K,
            )
            centroid_count = int(routing["centroid_count"])
            centroid_padded = int(routing["centroid_padded"])
            approximation = (
                k_centroid_int8, v_centroid, k_centroid_scale,
                exact_mask, centroid_count, centroid_padded,
            )
            del k_centroid
        output = _load_sm75_backend().run(
            q_int8, k_int8, v_input, q_scale, k_scale, row_count, lut,
            all_int8=all_int8, approximation=approximation,
        )
        implementation = (
            "star7-sm75-sol-exact-plus-centroid-q64k64-all-int8"
            if all_int8 else "star7-sm75-sol-exact-plus-centroid-q64k64-fp16-pv"
        )
    else:
        if not all_int8:
            raise SolUnavailableError(
                "SM80+ recommended Sol must use NVIDIA's official BF16 interface"
            )
        if capability < (8, 0) or triton is None:
            raise SolUnavailableError("SM80+ All-INT8 Sol requires SM80+ and Triton")
        v_int8, v_scale = sla._quantize(v, SOL_BLOCK_K, multiplier=1.0)
        del v
        k_centroid_int8, k_centroid_scale = sla._quantize(
            k_centroid, SOL_BLOCK_K, multiplier=1.0,
        )
        v_centroid_int8, v_centroid_scale = sla._quantize(
            v_centroid, SOL_BLOCK_K, multiplier=1.0,
        )
        centroid_groups = (key_blocks + SOL_BLOCK_K - 1) // SOL_BLOCK_K
        output = torch.empty_like(v_int8, dtype=torch.float16)
        grid = (query_blocks, q_int8.shape[0] * q_int8.shape[1])
        _sol_qk_int8_pv_int8_kernel[grid](
            q_int8, k_int8, v_int8, k_centroid_int8, v_centroid_int8,
            q_scale, k_scale, v_scale, k_centroid_scale, v_centroid_scale,
            row_count, lut, exact_mask, output,
            output.shape[-2], query_blocks, key_blocks, lut.shape[-1],
            centroid_groups, HEAD_DIM, SOL_BLOCK_Q, SOL_BLOCK_K,
            num_warps=4, num_stages=3,
        )
        implementation = "star7-sm80plus-sol-exact-plus-centroid-q64k64-all-int8"
    return SolResult(
        output, query_blocks, key_blocks, minimum, maximum, density,
        implementation, float(tau),
    )
