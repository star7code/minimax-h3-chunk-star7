"""ABI-stable loader for the precompiled Star7 SM75 CUDA attention core."""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
from pathlib import Path

import torch


ABI_VERSION = 7
_LIBRARY = None
_LOAD_ERROR: Exception | None = None


def _library_path() -> tuple[Path, dict]:
    system = platform.system()
    machine = platform.machine().lower()
    root = Path(__file__).resolve().parent
    manifest_path = root / "bin" / "sm75_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if system == "Windows" and machine in {"amd64", "x86_64"}:
        entry = manifest["windows_x64"]
        return root / "bin" / entry["file"], entry
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        entry = manifest["linux_x86_64"]
        return root / "bin" / entry["file"], entry
    raise RuntimeError(f"unsupported SM75 binary platform: {system}/{machine}")


def _load():
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_ERROR is not None:
        raise RuntimeError(f"SM75 CUDA binary failed to load: {_LOAD_ERROR}")
    try:
        path, manifest_entry = _library_path()
        if not path.is_file():
            raise FileNotFoundError(f"SM75 CUDA binary is missing: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest_entry["sha256"]:
            raise RuntimeError(
                f"SM75 CUDA binary checksum mismatch: expected "
                f"{manifest_entry['sha256']}, got {digest}"
            )
        library = ctypes.CDLL(str(path))
        library.star7_sla_sm75_abi_version.argtypes = []
        library.star7_sla_sm75_abi_version.restype = ctypes.c_int
        library.star7_sla_sm75_shared_bytes.argtypes = []
        library.star7_sla_sm75_shared_bytes.restype = ctypes.c_int
        launch = library.star7_sla_sm75_launch
        launch.argtypes = [
            *([ctypes.c_uint64] * 7),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_uint64,
        ]
        launch.restype = ctypes.c_int
        quant_v = library.star7_sla_sm75_quant_v_int8
        quant_v.argtypes = [
            *([ctypes.c_uint64] * 3),
            *([ctypes.c_int] * 5),
            *([ctypes.c_int64] * 3),
            ctypes.c_int, ctypes.c_uint64,
        ]
        quant_v.restype = ctypes.c_int
        launch_all_int8 = library.star7_sla_sm75_launch_all_int8
        launch_all_int8.argtypes = [
            *([ctypes.c_uint64] * 8),
            *([ctypes.c_int] * 7),
            ctypes.c_float, ctypes.c_uint64,
        ]
        launch_all_int8.restype = ctypes.c_int
        mean_pool = getattr(library, "star7_sla_sm75_mean_pool", None)
        if mean_pool is not None:
            mean_pool.argtypes = [
                *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 5),
                ctypes.c_uint64,
            ]
            mean_pool.restype = ctypes.c_int
        quantize = getattr(library, "star7_sla_sm75_quantize", None)
        if quantize is not None:
            quantize.argtypes = [
                *([ctypes.c_uint64] * 4), *([ctypes.c_int] * 4),
                ctypes.c_float, ctypes.c_int, ctypes.c_uint64,
            ]
            quantize.restype = ctypes.c_int
        found_abi = int(library.star7_sla_sm75_abi_version())
        if found_abi != ABI_VERSION:
            raise RuntimeError(
                f"SM75 CUDA ABI mismatch: expected {ABI_VERSION}, got {found_abi}"
            )
        _LIBRARY = library
        return library
    except Exception as exc:
        _LOAD_ERROR = exc
        raise


def availability() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "NVIDIA CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        return False, f"native SM75 backend requires compute capability 7.5, got {capability}"
    try:
        library = _load()
    except Exception as exc:
        return False, str(exc)
    shared_bytes = int(library.star7_sla_sm75_shared_bytes())
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    maximum = int(getattr(properties, "shared_memory_per_block_optin", 0) or 0)
    if maximum and shared_bytes > maximum:
        return False, f"kernel needs {shared_bytes} shared bytes, GPU allows {maximum}"
    return True, f"native CUDA ABI {ABI_VERSION}, shared={shared_bytes} bytes"


def preprocess_availability() -> tuple[bool, str]:
    available, reason = availability()
    if not available:
        return False, reason
    library = _load()
    if not hasattr(library, "star7_sla_sm75_mean_pool") or not hasattr(
        library, "star7_sla_sm75_quantize"
    ):
        return False, "installed SM75 binary predates native preprocessing"
    return True, "native CUDA routing/quantization preprocessing"


def mean_pool(
    x: torch.Tensor, block: int, mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous():
        raise ValueError("SM75 mean pooling requires contiguous CUDA FP16 input")
    if x.ndim != 4 or x.shape[-1] != 128 or block not in (64, 128):
        raise ValueError("SM75 mean pooling requires [B,H,L,128] and block 64/128")
    if mean is not None:
        if (
            not mean.is_cuda or mean.device != x.device or
            mean.dtype != torch.float16 or not mean.is_contiguous() or
            mean.numel() != x.shape[0] * x.shape[1] * 128
        ):
            raise ValueError("SM75 mean tensor must be contiguous CUDA FP16 [B,H,1,128]")
    available, reason = preprocess_availability()
    if not available:
        raise RuntimeError(reason)
    batch, heads, length, _ = x.shape
    groups = (length + block - 1) // block
    output = torch.empty(
        (batch, heads, groups, 128), dtype=torch.float16, device=x.device
    )
    code = int(_load().star7_sla_sm75_mean_pool(
        x.data_ptr(), 0 if mean is None else mean.data_ptr(), output.data_ptr(),
        batch, heads, length, block, int(mean is not None),
        torch.cuda.current_stream(x.device).cuda_stream,
    ))
    if code != 0:
        raise RuntimeError(f"SM75 native mean pooling failed with code={code}")
    return output


def quantize(
    x: torch.Tensor, block: int, multiplier: float,
    mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous():
        raise ValueError("SM75 quantization requires contiguous CUDA FP16 input")
    if x.ndim != 4 or x.shape[-1] != 128 or block not in (16, 64):
        raise ValueError("SM75 quantization requires [B,H,L,128] and block 16/64")
    if mean is not None:
        if (
            not mean.is_cuda or mean.device != x.device or
            mean.dtype != torch.float16 or not mean.is_contiguous() or
            mean.numel() != x.shape[0] * x.shape[1] * 128
        ):
            raise ValueError("SM75 mean tensor must be contiguous CUDA FP16 [B,H,1,128]")
    available, reason = preprocess_availability()
    if not available:
        raise RuntimeError(reason)
    batch, heads, length, _ = x.shape
    groups = (length + block - 1) // block
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((batch, heads, groups), dtype=torch.float32, device=x.device)
    code = int(_load().star7_sla_sm75_quantize(
        x.data_ptr(), 0 if mean is None else mean.data_ptr(), output.data_ptr(),
        scale.data_ptr(), batch, heads, length, block, float(multiplier),
        int(mean is not None), torch.cuda.current_stream(x.device).cuda_stream,
    ))
    if code != 0:
        raise RuntimeError(f"SM75 native quantization failed with code={code}")
    return output, scale


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | list[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    lut: torch.Tensor,
    dense_query_ranges: tuple[tuple[int, int], ...] = (),
    all_int8: bool = False,
) -> torch.Tensor:
    consume_v = isinstance(v, list)
    if consume_v:
        if len(v) != 1:
            raise ValueError("SM75 consuming V input requires one tensor")
        v = v.pop()
    tensors = (q, k, v, q_scale, k_scale, lut)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM75 CUDA backend requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("SM75 CUDA backend tensors must share one device")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float16:
        raise TypeError("SM75 requires INT8 Q/K and FP16 V")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("SM75 Q/K scales must be FP32")
    if lut.dtype != torch.int32:
        raise TypeError("SM75 routing LUT must be INT32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("SM75 CUDA backend requires contiguous tensors")

    batch, heads, length, head_dim = q.shape
    if head_dim != 128 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("SM75 CUDA backend requires equal [B,H,L,128] Q/K/V")
    key_blocks = (length + 63) // 64
    selected_blocks = lut.shape[-1]
    expected_query_blocks = (length + 127) // 128
    if selected_blocks <= 0 or selected_blocks > key_blocks:
        raise ValueError("SM75 routing LUT selected-block count is invalid")
    if lut.shape != (batch, heads, expected_query_blocks, selected_blocks):
        raise ValueError("SM75 routing LUT shape does not match Q/K/V")
    if q_scale.shape != (batch, heads, expected_query_blocks * 8):
        raise ValueError("SM75 Q scale shape does not match 16-row query warps")
    if k_scale.shape != (batch, heads, key_blocks):
        raise ValueError("SM75 K scale shape does not match key blocks")
    stream = torch.cuda.current_stream(q.device).cuda_stream
    library = _load()
    v_int8 = v_scale = None
    padded_length = key_blocks * 64
    if all_int8:
        v_int8 = torch.empty(
            (batch, heads, head_dim, padded_length),
            dtype=torch.int8, device=q.device,
        )
        v_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=q.device
        )
        quant_code = int(library.star7_sla_sm75_quant_v_int8(
            v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(),
            batch, heads, length, head_dim, padded_length,
            v.stride(0), v.stride(1), v.stride(2), 1, stream,
        ))
        if quant_code != 0:
            raise RuntimeError(
                f"SM75 CUDA SLA V quantization failed with code={quant_code}"
            )
        if consume_v:
            # The quantizer and attention launch share this CUDA stream. V is
            # no longer read after quantization, so its allocation can safely
            # be reused for the FP16 output queued later on the same stream.
            del tensors, v
            output = torch.empty(
                (batch, heads, length, head_dim),
                dtype=torch.float16, device=q.device,
            )
        else:
            output = torch.empty_like(v)
    else:
        output = torch.empty_like(v)

    def launch(active_lut: torch.Tensor, block_base: int, block_count: int) -> None:
        if all_int8:
            code = int(library.star7_sla_sm75_launch_all_int8(
                q.data_ptr(), k.data_ptr(), v_int8.data_ptr(),
                q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
                active_lut.data_ptr(), output.data_ptr(), batch, heads, length,
                padded_length, active_lut.shape[-1], block_base, block_count,
                head_dim ** -0.5, stream,
            ))
        else:
            code = int(library.star7_sla_sm75_launch(
                q.data_ptr(), k.data_ptr(), v.data_ptr(),
                q_scale.data_ptr(), k_scale.data_ptr(), active_lut.data_ptr(),
                output.data_ptr(), batch, heads, length,
                active_lut.shape[-1], block_base, block_count,
                head_dim ** -0.5, stream,
            ))
        if code != 0:
            mode = "All-INT8" if all_int8 else "FP16-PV"
            raise RuntimeError(
                f"SM75 CUDA SLA {mode} launch failed with cudaError={code}"
            )

    launch(lut, 0, expected_query_blocks)
    protected_blocks: set[int] = set()
    for start, end in dense_query_ranges:
        protected_blocks.update(
            range(max(0, start // 128), min(expected_query_blocks, (end + 127) // 128))
        )
    if protected_blocks:
        ordered = sorted(protected_blocks)
        groups: list[tuple[int, int]] = []
        group_start = previous = ordered[0]
        for block_index in ordered[1:]:
            if block_index != previous + 1:
                groups.append((group_start, previous + 1))
                group_start = block_index
            previous = block_index
        groups.append((group_start, previous + 1))
        all_keys = torch.arange(key_blocks, dtype=torch.int32, device=q.device)
        for block_start, block_end in groups:
            count = block_end - block_start
            dense_lut = all_keys.view(1, 1, 1, key_blocks).expand(
                batch, heads, count, key_blocks
            ).contiguous()
            launch(dense_lut, block_start, count)
    return output
