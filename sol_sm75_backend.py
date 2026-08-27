"""SM75 loader for the Star7 Q64/K64 threshold-routed Sol kernels."""

from __future__ import annotations

import ctypes

import torch


SOL_BLOCK_Q = 64
SOL_BLOCK_K = 64
HEAD_DIM = 128


def _load_library():
    try:
        from . import sm75_backend
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        module_name = "star7_sm75_backend"
        if module_name in sys.modules:
            sm75_backend = sys.modules[module_name]
        else:
            path = Path(__file__).with_name("sm75_backend.py")
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load SM75 backend from {path}")
            sm75_backend = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = sm75_backend
            spec.loader.exec_module(sm75_backend)
    library = sm75_backend._load()
    try:
        shared = library.star7_sol_sm75_shared_bytes
        prepare_routes = library.star7_sol_sm75_prepare_routes
        pack_lut = library.star7_sol_sm75_pack_lut
        quantize = library.star7_sol_sm75_quantize
        quantize_v_with_scale = library.star7_sol_sm75_quantize_v_with_scale
        launch = library.star7_sol_sm75_launch
        launch_all_int8 = library.star7_sol_sm75_launch_all_int8
        launch_all_int8_complete = library.star7_sol_sm75_launch_all_int8_complete
    except AttributeError as exc:
        raise RuntimeError(
            "The installed SM75 binary predates Star7 Sol support. "
            "Install the matching node release; no fallback was attempted."
        ) from exc
    shared.argtypes = []
    shared.restype = ctypes.c_int
    prepare_routes.argtypes = [
        *([ctypes.c_uint64] * 11),
        *([ctypes.c_int] * 4),
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    prepare_routes.restype = ctypes.c_int
    pack_lut.argtypes = [
        *([ctypes.c_uint64] * 3),
        *([ctypes.c_int] * 4), ctypes.c_uint64,
    ]
    pack_lut.restype = ctypes.c_int
    quantize.argtypes = [
        *([ctypes.c_uint64] * 3),
        *([ctypes.c_int] * 4), ctypes.c_uint64,
    ]
    quantize.restype = ctypes.c_int
    quantize_v_with_scale.argtypes = [
        *([ctypes.c_uint64] * 3),
        *([ctypes.c_int] * 4), ctypes.c_uint64,
    ]
    quantize_v_with_scale.restype = ctypes.c_int
    launch.argtypes = [
        *([ctypes.c_uint64] * 12),
        *([ctypes.c_int] * 6),
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    launch.restype = ctypes.c_int
    launch_all_int8.argtypes = [
        *([ctypes.c_uint64] * 9),
        *([ctypes.c_int] * 5),
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    launch_all_int8.restype = ctypes.c_int
    launch_all_int8_complete.argtypes = [
        *([ctypes.c_uint64] * 14),
        *([ctypes.c_int] * 7),
        ctypes.c_float, ctypes.c_uint64,
    ]
    launch_all_int8_complete.restype = ctypes.c_int
    return library


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    sink_tokens: int = 0,
    sink_start: int | None = None,
):
    """Run SM75 Sol centroids, threshold routing, and LUT packing in CUDA."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("SM75 Sol preprocessing requires equal [B,H,L,128] Q/K/V")
    if q.dtype != torch.float16 or any(x.dtype != q.dtype for x in (k, v)):
        raise TypeError("SM75 Sol preprocessing requires FP16 Q/K/V")
    if not all(x.is_cuda and x.is_contiguous() for x in (q, k, v)):
        raise ValueError("SM75 Sol preprocessing requires contiguous CUDA tensors")
    batch, heads, length, head_dim = q.shape
    if head_dim != HEAD_DIM:
        raise ValueError("SM75 Sol preprocessing requires head_dim=128")
    blocks = (length + SOL_BLOCK_K - 1) // SOL_BLOCK_K
    padded_blocks = ((blocks + SOL_BLOCK_K - 1) // SOL_BLOCK_K) * SOL_BLOCK_K
    summaries = (batch, heads, padded_blocks, head_dim)
    q_centroid = torch.empty(summaries, dtype=q.dtype, device=q.device)
    k_centroid = torch.empty_like(q_centroid)
    v_centroid = torch.empty_like(q_centroid)
    k_mean = torch.empty((batch, heads, head_dim), dtype=torch.float32, device=q.device)
    k_variance = torch.empty_like(k_mean)
    threshold = torch.empty((batch, heads, blocks), dtype=torch.float32, device=q.device)
    exact_mask = torch.empty(
        (batch, heads, blocks, blocks), dtype=torch.uint8, device=q.device,
    )
    row_count = torch.empty(
        (batch, heads, blocks), dtype=torch.int32, device=q.device,
    )
    if sink_tokens > 0:
        start = length - int(sink_tokens) if sink_start is None else int(sink_start)
        sink_first = max(0, start // SOL_BLOCK_K)
        sink_last = min(blocks, (start + int(sink_tokens) + SOL_BLOCK_K - 1) // SOL_BLOCK_K)
    else:
        sink_first = blocks
        sink_last = blocks
    library = _load_library()
    stream = torch.cuda.current_stream(q.device).cuda_stream
    code = int(library.star7_sol_sm75_prepare_routes(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        q_centroid.data_ptr(), k_centroid.data_ptr(), v_centroid.data_ptr(),
        k_mean.data_ptr(), k_variance.data_ptr(), threshold.data_ptr(),
        exact_mask.data_ptr(), row_count.data_ptr(),
        batch, heads, length, padded_blocks, float(tau), head_dim ** -0.5,
        sink_first, sink_last, stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol preprocessing failed with code={code}")
    minimum = int(row_count.min().item())
    maximum = int(row_count.max().item())
    if minimum <= 0 or maximum > blocks:
        raise RuntimeError(
            f"SM75 native Sol routing produced invalid counts {minimum}..{maximum}"
        )
    lut = torch.empty(
        (batch, heads, blocks, maximum), dtype=torch.int32, device=q.device,
    )
    rows = batch * heads * blocks
    code = int(library.star7_sol_sm75_pack_lut(
        exact_mask.data_ptr(), row_count.data_ptr(), lut.data_ptr(),
        rows, blocks, maximum, 0, stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol LUT packing failed with code={code}")
    density = float(row_count.float().mean().item() / blocks)
    return {
        "row_count": row_count,
        "lut": lut,
        "density": density,
        "minimum": minimum,
        "maximum": maximum,
        "exact_mask": exact_mask,
        "k_centroid": k_centroid,
        "v_centroid": v_centroid,
        "centroid_count": blocks,
        "centroid_padded": padded_blocks,
    }


def quantize(value: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_cuda:
        raise TypeError("SM75 native Sol quantization requires CUDA FP16 [B,H,L,D]")
    if value.shape[-1] != HEAD_DIM or not value.is_contiguous():
        raise ValueError("SM75 native Sol quantization requires contiguous head_dim=128")
    batch, heads, length, _ = value.shape
    groups = (length + block - 1) // block
    output = torch.empty_like(value, dtype=torch.int8)
    scale = torch.empty(
        (batch, heads, groups), dtype=torch.float32, device=value.device,
    )
    library = _load_library()
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(library.star7_sol_sm75_quantize(
        value.data_ptr(), output.data_ptr(), scale.data_ptr(),
        batch, heads, length, int(block), stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol quantization failed with code={code}")
    return output, scale


def quantize_v_with_scale(
    value: torch.Tensor,
    scale: torch.Tensor,
    padded_length: int,
) -> torch.Tensor:
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_contiguous():
        raise TypeError("SM75 Sol centroid V quantization requires contiguous FP16")
    batch, heads, length, head_dim = value.shape
    if head_dim != HEAD_DIM or scale.shape != (batch, heads, head_dim):
        raise ValueError("SM75 Sol centroid V scale shape is invalid")
    output = torch.empty(
        (batch, heads, head_dim, padded_length),
        dtype=torch.int8, device=value.device,
    )
    library = _load_library()
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(library.star7_sol_sm75_quantize_v_with_scale(
        value.data_ptr(), scale.data_ptr(), output.data_ptr(),
        batch, heads, length, padded_length, stream,
    ))
    if code:
        raise RuntimeError(
            f"SM75 native Sol centroid-V quantization failed with code={code}"
        )
    return output


def availability() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "NVIDIA CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        return False, f"Star7 Sol SM75 requires compute capability 7.5, got {capability}"
    try:
        library = _load_library()
        shared = int(library.star7_sol_sm75_shared_bytes())
    except Exception as exc:
        return False, str(exc)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    maximum = int(getattr(properties, "shared_memory_per_block_optin", 0) or 0)
    if maximum and shared > maximum:
        return False, f"Sol kernel needs {shared} shared bytes, GPU allows {maximum}"
    return True, f"native CUDA Q64/K64, 4 warps, shared={shared} bytes"


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | list[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    row_count: torch.Tensor,
    lut: torch.Tensor,
    *,
    all_int8: bool = False,
    approximation=None,
) -> torch.Tensor:
    consume_v = isinstance(v, list)
    if consume_v:
        if len(v) != 1:
            raise ValueError("SM75 Sol consuming V input requires one tensor")
        v = v.pop()
    tensors = (q, k, v, q_scale, k_scale, row_count, lut)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM75 Sol requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("SM75 Sol tensors must share one device")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float16:
        raise TypeError("SM75 Sol requires INT8 Q/K and FP16 V")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("SM75 Sol Q/K scales must be FP32")
    if row_count.dtype != torch.int32 or lut.dtype != torch.int32:
        raise TypeError("SM75 Sol row counts and LUT must be INT32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("SM75 Sol requires contiguous tensors")

    batch, heads, length, head_dim = q.shape
    q_blocks = (length + SOL_BLOCK_Q - 1) // SOL_BLOCK_Q
    key_blocks = (length + SOL_BLOCK_K - 1) // SOL_BLOCK_K
    if head_dim != HEAD_DIM or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("SM75 Sol requires equal [B,H,L,128] Q/K/V")
    if row_count.shape != (batch, heads, q_blocks):
        raise ValueError("SM75 Sol row_count shape does not match Q")
    if lut.ndim != 4 or lut.shape[:3] != row_count.shape:
        raise ValueError("SM75 Sol LUT shape does not match row_count")
    lut_stride = int(lut.shape[-1])
    if lut_stride <= 0 or lut_stride > key_blocks:
        raise ValueError("SM75 Sol LUT stride is invalid")
    if int(row_count.min().item()) <= 0 or int(row_count.max().item()) > lut_stride:
        raise ValueError("SM75 Sol row_count exceeds LUT stride")
    if q_scale.shape != (batch, heads, q_blocks * 4):
        raise ValueError("SM75 Sol Q scale must contain four 16-row scales per Q64 block")
    if k_scale.shape != (batch, heads, key_blocks):
        raise ValueError("SM75 Sol K scale shape does not match K64 blocks")

    library = _load_library()
    stream = torch.cuda.current_stream(q.device).cuda_stream
    if all_int8:
        if approximation is None or len(approximation) != 6:
            raise ValueError(
                "SM75 Sol All-INT8 requires exact-plus-centroid approximation inputs"
            )
        (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
            centroid_count, centroid_padded,
        ) = approximation
        centroid_groups = int(centroid_padded) // SOL_BLOCK_K
        if (
            k_centroid.dtype != torch.int8
            or v_centroid.dtype != torch.float16
            or k_centroid_scale.dtype != torch.float32
            or exact_mask.dtype != torch.uint8
        ):
            raise TypeError("SM75 Sol All-INT8 centroid tensor dtypes are invalid")
        if k_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol All-INT8 K-centroid shape is invalid")
        if v_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol All-INT8 V-centroid shape is invalid")
        if k_centroid_scale.shape != (batch, heads, centroid_groups):
            raise ValueError("SM75 Sol All-INT8 K-centroid scale shape is invalid")
        if exact_mask.shape != (batch, heads, q_blocks, centroid_count):
            raise ValueError("SM75 Sol All-INT8 exact-mask shape is invalid")
        padded_length = key_blocks * SOL_BLOCK_K
        v_int8 = torch.empty(
            (batch, heads, head_dim, padded_length),
            dtype=torch.int8,
            device=q.device,
        )
        v_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=q.device,
        )
        quant_code = int(library.star7_sla_sm75_quant_v_int8(
            v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(),
            batch, heads, length, head_dim, padded_length,
            v.stride(0), v.stride(1), v.stride(2), 1, stream,
        ))
        if quant_code:
            raise RuntimeError(f"SM75 Sol V quantization failed with code={quant_code}")
        v_centroid_int8 = torch.empty(
            (batch, heads, head_dim, centroid_padded),
            dtype=torch.int8, device=q.device,
        )
        v_centroid_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=q.device,
        )
        centroid_quant_code = int(library.star7_sla_sm75_quant_v_int8(
            v_centroid.data_ptr(), v_centroid_int8.data_ptr(),
            v_centroid_scale.data_ptr(), batch, heads, centroid_padded,
            head_dim, centroid_padded, v_centroid.stride(0),
            v_centroid.stride(1), v_centroid.stride(2), 1, stream,
        ))
        if centroid_quant_code:
            raise RuntimeError(
                "SM75 Sol centroid-V quantization failed with "
                f"code={centroid_quant_code}"
            )
        if consume_v:
            del tensors, v
            output = torch.empty(
                (batch, heads, length, head_dim),
                dtype=torch.float16,
                device=q.device,
            )
        else:
            output = torch.empty_like(v)
        code = int(library.star7_sol_sm75_launch_all_int8_complete(
            q.data_ptr(), k.data_ptr(), v_int8.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
            row_count.data_ptr(), lut.data_ptr(), k_centroid.data_ptr(),
            v_centroid_int8.data_ptr(), k_centroid_scale.data_ptr(),
            v_centroid_scale.data_ptr(), exact_mask.data_ptr(),
            output.data_ptr(), batch, heads, length,
            padded_length, lut_stride, centroid_count, centroid_padded,
            head_dim ** -0.5, stream,
        ))
    else:
        if approximation is None or len(approximation) != 6:
            raise ValueError("SM75 Sol FP16-PV requires centroid approximation inputs")
        (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
            centroid_count, centroid_padded,
        ) = approximation
        approximation_tensors = (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
        )
        if any(not tensor.is_cuda for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroids must be CUDA tensors")
        if any(tensor.device != q.device for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroids must share the Q device")
        if (
            k_centroid.dtype != torch.int8
            or v_centroid.dtype != torch.float16
            or k_centroid_scale.dtype != torch.float32
            or exact_mask.dtype != torch.uint8
        ):
            raise TypeError("SM75 Sol centroid tensor dtypes are invalid")
        if any(not tensor.is_contiguous() for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroid tensors must be contiguous")
        centroid_groups = int(centroid_padded) // SOL_BLOCK_K
        if k_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol K-centroid shape is invalid")
        if v_centroid.shape != k_centroid.shape:
            raise ValueError("SM75 Sol V-centroid shape is invalid")
        if k_centroid_scale.shape != (batch, heads, centroid_groups):
            raise ValueError("SM75 Sol K-centroid scale shape is invalid")
        if exact_mask.shape != (batch, heads, q_blocks, centroid_count):
            raise ValueError("SM75 Sol exact-mask shape is invalid")
        output = torch.empty_like(v)
        code = int(library.star7_sol_sm75_launch(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), row_count.data_ptr(),
            lut.data_ptr(), k_centroid.data_ptr(), v_centroid.data_ptr(),
            k_centroid_scale.data_ptr(), exact_mask.data_ptr(),
            output.data_ptr(), batch, heads, length, lut_stride,
            centroid_count, centroid_padded, head_dim ** -0.5, stream,
        ))
    if code:
        mode = "All-INT8" if all_int8 else "FP16-PV"
        raise RuntimeError(
            f"SM75 CUDA Sol {mode} launch failed with cudaError={code}; "
            "no fallback was attempted"
        )
    return output
