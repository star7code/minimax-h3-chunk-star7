from __future__ import annotations

# The native ABI and NVIDIA Optical Flow integration are adapted from the
# MIT-licensed ComfyUI-DLSS5-NR project. See vendor/dlss5nr/LICENSE and
# vendor/dlss5nr/THIRD_PARTY_NOTICES.md.

import ctypes
import hashlib
import math
import os
import platform
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # ComfyUI core does not require OpenCV in every distribution.
    cv2 = None

import comfy.model_management
import comfy.utils
import folder_paths


_LOG = "[Star7 DLSS Neural Enhance]"
_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "vendor" / "dlss5nr"
_BRIDGE = _VENDOR / "native" / "bin" / "dlss5nr_bridge_v2.dll"
_RUNTIME = _VENDOR / "runtime"
_MODEL = Path(folder_paths.models_dir) / "upscale_models" / "nvngx_dlssnr.dll"
_RUNTIME_MODEL = _RUNTIME / "nvngx_dlssnr.dll"

_KNOWN_MODEL_SHA256 = {
    # Bandukids HF build; verified working on the local SM75 test machine.
    "8270b350cd82de5ce89806872cdd6b6a9249b80836b91bbeb3573470744cc206",
    # RankFTW/ShortFuse SF-v2 cross-generation build.
    "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927",
}
_MODEL_SOURCES = (
    (
        "HF domestic mirror",
        "https://hf-mirror.com/Bandukids/DLSS-Runtimes/resolve/main/nvngx_dlssnr.dll?download=true",
        False,
    ),
    (
        "Hugging Face",
        "https://huggingface.co/Bandukids/DLSS-Runtimes/resolve/main/nvngx_dlssnr.dll?download=true",
        False,
    ),
    (
        "RankFTW GitHub SF-v2",
        "https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip",
        True,
    ),
)

_PRESETS = {
    # V2 presets deliberately avoid the aggressive cinematic style for real
    # footage. High structure/skin values make pores and wrinkles look older,
    # especially after enlargement and video compression.
    "真实风格": dict(style=1, strength=0.90, tone=0.75, structure=0.70, skin=-1.00, temporal=0.55, mask=True),
    "真实人像优化": dict(style=1, strength=0.85, tone=0.70, structure=0.60, skin=0.00, temporal=0.60, mask=True),
    "3D 动漫风格": dict(style=2, strength=1.20, tone=0.95, structure=1.25, skin=-1.00, temporal=0.50, mask=False),
    "2D 动漫风格": dict(style=0, strength=1.05, tone=0.70, structure=1.00, skin=-1.00, temporal=0.45, mask=False),
}
_PRESET_NAMES = tuple(_PRESETS) + ("自定义参数",)
_LEGACY_PRESET = "真实风格加强"

_LOCK = threading.RLock()
_LIB = None
_INITIALIZED_GPU = None
_DLL_DIRECTORY_HANDLES = []
_SWAP_CHANNELS = None
_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_THREAD = None
_DOWNLOAD_DONE = threading.Event()
_DOWNLOAD_ERROR = None


class Star7DLSSNRError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_model(path: Path, require_known_hash: bool) -> str:
    if not path.is_file():
        raise Star7DLSSNRError(f"模型文件不存在 / model file not found: {path}")
    size = path.stat().st_size
    if size < 150 * 1024 * 1024:
        raise Star7DLSSNRError(
            f"模型文件不完整 / incomplete model ({size} bytes): {path}. "
            "可能是下载中断或 Git LFS 指针。"
        )
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise Star7DLSSNRError(f"模型格式错误 / invalid Windows PE model: {path}")
    digest = _sha256(path)
    if require_known_hash and digest not in _KNOWN_MODEL_SHA256:
        raise Star7DLSSNRError(
            "模型校验失败 / model SHA-256 mismatch: "
            f"{digest}. 下载源可能返回了错误页面、损坏文件或未知版本。"
        )
    return digest


def _download_model_worker() -> None:
    global _DOWNLOAD_ERROR
    errors = []
    _MODEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        for source_name, url, is_zip in _MODEL_SOURCES:
            archive = _MODEL.with_name(f".{_MODEL.name}.{os.getpid()}.download")
            candidate = _MODEL.with_name(f".{_MODEL.name}.{os.getpid()}.candidate")
            try:
                print(f"[INFO] {_LOG} downloading model from {source_name}...")
                request = urllib.request.Request(url, headers={"User-Agent": "Star7-ComfyUI-DLSSNR/2"})
                with urllib.request.urlopen(request, timeout=45) as response, archive.open("wb") as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                if is_zip:
                    with zipfile.ZipFile(archive) as package:
                        members = [name for name in package.namelist() if Path(name).name.lower() == "nvngx_dlssnr.dll"]
                        if len(members) != 1:
                            raise Star7DLSSNRError("下载压缩包中未找到唯一的 nvngx_dlssnr.dll。")
                        with package.open(members[0]) as source, candidate.open("wb") as output:
                            while True:
                                block = source.read(8 * 1024 * 1024)
                                if not block:
                                    break
                                output.write(block)
                else:
                    os.replace(archive, candidate)
                digest = _validate_model(candidate, require_known_hash=True)
                os.replace(candidate, _MODEL)
                print(f"[INFO] {_LOG} model download complete | source={source_name} | SHA256={digest}")
                _DOWNLOAD_ERROR = None
                return
            except Exception as error:
                errors.append(f"{source_name}: {error}")
            finally:
                archive.unlink(missing_ok=True)
                candidate.unlink(missing_ok=True)
        _DOWNLOAD_ERROR = "\n".join(errors)
    finally:
        _DOWNLOAD_DONE.set()


def _start_model_download() -> None:
    global _DOWNLOAD_THREAD
    if _MODEL.is_file() or (_DOWNLOAD_THREAD is not None and _DOWNLOAD_THREAD.is_alive()):
        return
    with _DOWNLOAD_LOCK:
        if _MODEL.is_file() or (_DOWNLOAD_THREAD is not None and _DOWNLOAD_THREAD.is_alive()):
            return
        _DOWNLOAD_DONE.clear()
        print(
            f"[INFO] {_LOG} model missing; background download started. "
            "国内 HF 镜像优先，Hugging Face/GitHub 自动兜底。"
        )
        _DOWNLOAD_THREAD = threading.Thread(
            target=_download_model_worker, name="star7-dlssnr-download", daemon=True
        )
        _DOWNLOAD_THREAD.start()


def _ensure_model_available() -> None:
    if _MODEL.is_file():
        return
    _start_model_download()
    _DOWNLOAD_DONE.wait()
    if not _MODEL.is_file():
        detail = _DOWNLOAD_ERROR or "unknown download failure"
        raise Star7DLSSNRError(
            "缺少 DLSS Neural Rendering 模型，且所有自动下载源均失败。\n"
            "Missing nvngx_dlssnr.dll and all automatic download sources failed.\n"
            f"目标位置 / destination: {_MODEL}\n"
            f"失败详情 / details:\n{detail}\n"
            "请检查网络、代理、防火墙和磁盘空间，也可以手动将模型放到上述位置。"
        )


def _decode_error(buffer: ctypes.Array) -> str:
    try:
        return buffer.value.decode("utf-8", errors="replace")
    except Exception:
        return "Unknown native DLSS Neural Rendering error"


def _prepare_runtime_model() -> None:
    _ensure_model_available()
    _validate_model(_MODEL, require_known_hash=False)

    _RUNTIME.mkdir(parents=True, exist_ok=True)
    if _RUNTIME_MODEL.exists():
        try:
            if os.path.samefile(_MODEL, _RUNTIME_MODEL):
                return
        except OSError:
            pass
        _RUNTIME_MODEL.unlink()
    try:
        os.link(_MODEL, _RUNTIME_MODEL)
    except OSError as error:
        raise Star7DLSSNRError(
            "Could not create the private zero-copy model link used by the native bridge. "
            "Keep ComfyUI/models and custom_nodes on the same drive.\n"
            f"{error}"
        ) from error


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    if platform.system() != "Windows":
        raise Star7DLSSNRError("DLSS Neural Rendering currently requires Windows, D3D12 and NVIDIA NGX.")
    if not _BRIDGE.is_file():
        raise Star7DLSSNRError(f"Native DLSS bridge is missing: {_BRIDGE}")

    _prepare_runtime_model()
    for directory in (_BRIDGE.parent, _RUNTIME, _RUNTIME / "caller"):
        if directory.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))

    library = ctypes.WinDLL(str(_BRIDGE))
    library.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
    library.dlss5nr_init.restype = ctypes.c_int
    library.dlss5nr_process.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    library.dlss5nr_process.restype = ctypes.c_int
    library.dlss5nr_shutdown.argtypes = []
    library.dlss5nr_shutdown.restype = None
    library.dlss5nr_nvof_available.argtypes = []
    library.dlss5nr_nvof_available.restype = ctypes.c_int
    _LIB = library
    return library


def _ensure_initialized(gpu_index: int = 0):
    global _INITIALIZED_GPU
    library = _load_library()
    if _INITIALIZED_GPU == gpu_index:
        return library
    if _INITIALIZED_GPU is not None:
        library.dlss5nr_shutdown()
        _INITIALIZED_GPU = None
    error = ctypes.create_string_buffer(4096)
    if not library.dlss5nr_init(gpu_index, str(_RUNTIME), error, len(error)):
        raise Star7DLSSNRError(_decode_error(error))
    _INITIALIZED_GPU = gpu_index
    return library


def _even_up(value: float) -> int:
    return max(2, int(math.ceil(value / 2.0)) * 2)


def _target_size(width: int, height: int, target_megapixels: float) -> tuple[int, int, float, bool]:
    source_pixels = width * height
    requested_pixels = max(0.1, float(target_megapixels)) * 1_000_000.0
    preserved = requested_pixels <= source_pixels
    scale = 1.0 if preserved else math.sqrt(requested_pixels / source_pixels)
    width_out, height_out = _even_up(width * scale), _even_up(height * scale)
    long_edge, short_edge = max(width_out, height_out), min(width_out, height_out)
    cap = min(1.0, 7680.0 / long_edge, 4320.0 / short_edge)
    if cap < 1.0:
        width_out, height_out = _even_up(width_out * cap), _even_up(height_out * cap)
    actual_megapixels = (width_out * height_out) / 1_000_000.0
    return width_out, height_out, actual_megapixels, preserved


def _settings(
    preset: str,
    nr_intensity: float,
    local_structure: float,
    local_tone: float,
    skin_structure: float,
    temporal_stability: float,
    automatic_mask: bool,
) -> dict:
    # Seamlessly migrate V1 workflows that stored the removed aggressive preset.
    if preset == _LEGACY_PRESET:
        preset = "真实风格"
    if preset in _PRESETS:
        return dict(_PRESETS[preset])
    return {
        "style": 1, "strength": float(nr_intensity), "tone": float(local_tone),
        "structure": float(local_structure), "skin": float(skin_structure),
        "temporal": float(temporal_stability), "mask": bool(automatic_mask),
    }


def _process_frame(library, frame: np.ndarray, settings: dict, index: int, temporal: bool) -> np.ndarray:
    global _SWAP_CHANNELS
    frame_in = np.ascontiguousarray(frame, dtype=np.float32)
    frame_out = np.empty_like(frame_in, dtype=np.float32)
    height, width, _ = frame_in.shape
    error = ctypes.create_string_buffer(4096)
    ok = library.dlss5nr_process(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width, height, settings["style"], 0,
        # This runtime ignores DLSSNR.Intensity in deterministic A/B tests. Keep
        # its native value neutral and apply the visible strength after readback.
        ctypes.c_float(1.0), ctypes.c_float(settings["tone"]),
        ctypes.c_float(settings["structure"]), ctypes.c_float(settings["skin"]),
        1 if settings["mask"] else 0,
        1 if (not temporal or index == 0) else 0,
        1 if temporal else 0,
        error, len(error),
    )
    if not ok:
        raise Star7DLSSNRError(f"Frame {index}: {_decode_error(error)}")

    if _SWAP_CHANNELS is None:
        swapped = frame_out[..., [2, 1, 0]]
        step_y, step_x = max(1, height // 128), max(1, width // 128)
        reference = frame_in[::step_y, ::step_x]
        raw_sample = frame_out[::step_y, ::step_x]
        swap_sample = swapped[::step_y, ::step_x]
        raw_score = float(np.mean(np.abs(raw_sample - reference))) + float(np.mean(np.abs(raw_sample.mean((0, 1)) - reference.mean((0, 1)))))
        swap_score = float(np.mean(np.abs(swap_sample - reference))) + float(np.mean(np.abs(swap_sample.mean((0, 1)) - reference.mean((0, 1)))))
        _SWAP_CHANNELS = swap_score < raw_score
    return frame_out[..., [2, 1, 0]] if _SWAP_CHANNELS else frame_out


def _resize_lanczos(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one frame without ever materializing a second enlarged video batch."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return np.ascontiguousarray(frame, dtype=np.float32)
    if cv2 is None:
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]
        resized = F.interpolate(
            tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True,
        )[0].permute(1, 2, 0).numpy()
    else:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    if resized.ndim == 2:
        resized = resized[..., None]
    return np.ascontiguousarray(np.clip(resized, 0.0, 1.0), dtype=np.float32)


def _prepare_rgb_frame(cpu: torch.Tensor, index: int, channels: int, width: int, height: int) -> np.ndarray:
    frame = cpu[index, ..., :3]
    if channels == 1:
        frame = frame.repeat(1, 1, 3)
    return _resize_lanczos(frame.numpy(), width, height)


def _stabilize_residual(
    source: np.ndarray,
    processed: np.ndarray,
    previous_source: np.ndarray | None,
    previous_residual: np.ndarray | None,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    current_residual = processed - source
    if previous_source is None or previous_residual is None or strength <= 0.0:
        return processed, current_residual
    # Same-position motion gating deliberately avoids carrying detail across moving
    # edges. Static/compression-noise regions retain history; real movement quickly
    # drives the history weight to zero, limiting trails and double edges.
    motion = np.mean(np.abs(source - previous_source), axis=2, keepdims=True)
    stable = np.clip(1.0 - motion / 0.045, 0.0, 1.0)
    weight = (stable * stable) * min(max(float(strength), 0.0), 0.9)
    residual = current_residual * (1.0 - weight) + previous_residual * weight
    result = np.clip(source + residual, 0.0, 1.0)
    return np.ascontiguousarray(result, dtype=np.float32), np.ascontiguousarray(residual, dtype=np.float32)


class Star7DLSSNeuralEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                # Keep the removed V1 value server-valid solely so saved workflows
                # can migrate cleanly; the frontend removes it from the dropdown.
                "风格预设": (_PRESET_NAMES + (_LEGACY_PRESET,), {"default": "真实风格"}),
                "目标像素 (MP)": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "NR 强度": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 2.0, "step": 0.05}),
                "局部结构": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 2.0, "step": 0.05}),
                "局部色调": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "皮肤结构": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "时序稳定": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 0.9, "step": 0.05}),
                "自动蒙版": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("优化图像",)
    FUNCTION = "enhance"
    CATEGORY = "Star7/图像与视频"
    DESCRIPTION = (
        "对 IMAGE 或视频帧批次执行 GPU Neural Rendering。单个模型放在 "
        "models/upscale_models/nvngx_dlssnr.dll。目标百万像素高于输入时先做高质量放大，"
        "再在目标尺寸进行 NR；当前单模型路径不是原生 DLSS Super Resolution。"
    )

    def enhance(self, 图像: torch.Tensor, 风格预设: str, 目标像素_MP: float = 1.0, **kwargs):
        # ComfyUI normalizes spaces and parentheses in Python argument names on some
        # versions, while others pass the original widget label through kwargs.
        target_megapixels = kwargs.pop("目标像素 (MP)", 目标像素_MP)
        if not isinstance(图像, torch.Tensor) or 图像.ndim != 4 or 图像.shape[-1] not in (1, 3, 4):
            raise ValueError("IMAGE must be [batch, height, width, channels] with 1, 3, or 4 channels.")
        if not torch.cuda.is_available():
            raise Star7DLSSNRError("DLSS Neural Rendering requires a supported NVIDIA RTX GPU.")
        batch, height, width, channels = map(int, 图像.shape)
        if batch < 1 or width < 64 or height < 64:
            raise ValueError("DLSS Neural Rendering requires images of at least 64×64 pixels.")

        if 风格预设 == _LEGACY_PRESET:
            print(f"[INFO] {_LOG} migrated removed V1 preset '真实风格加强' to safer V2 preset '真实风格'.")
            风格预设 = "真实风格"
        settings = _settings(
            风格预设,
            kwargs.get("NR 强度", 0.90), kwargs.get("局部结构", 0.70),
            kwargs.get("局部色调", 0.75), kwargs.get("皮肤结构", -1.0),
            kwargs.get("时序稳定", 0.55), kwargs.get("自动蒙版", True),
        )
        output_width, output_height, actual_megapixels, preserved = _target_size(width, height, target_megapixels)
        if preserved:
            print(
                f"[INFO] {_LOG} target {float(target_megapixels):.2f} MP is not above the "
                f"{(width * height) / 1_000_000.0:.2f} MP input; preserving {width}×{height}."
            )
        elif actual_megapixels + 0.01 < float(target_megapixels):
            print(
                f"[WARNING] {_LOG} target capped at {output_width}×{output_height} "
                f"({actual_megapixels:.2f} MP, safe 8K boundary)."
            )

        cpu = 图像.detach().to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
        print(
            f"[INFO] {_LOG} preset={风格预设} | {width}×{height} -> "
            f"{output_width}×{output_height} ({actual_megapixels:.2f} MP) | "
            f"frames={batch} | in-process GPU NR"
        )
        output = torch.empty((batch, output_height, output_width, 3), dtype=torch.float32)
        alpha_output = torch.empty((batch, output_height, output_width, 1), dtype=torch.float32) if channels == 4 else None
        progress = comfy.utils.ProgressBar(batch)
        with _LOCK:
            compute_capability = torch.cuda.get_device_capability()
            library = _ensure_initialized(0)
            # The current NVOF bridge can report present and still fail during execution on
            # Turing/SM75 drivers. Independent-frame NR is deterministic and avoids that hang.
            temporal = (
                batch > 1
                and compute_capability >= (8, 0)
                and bool(library.dlss5nr_nvof_available())
            )
            if batch > 1 and not temporal:
                print(
                    f"[WARNING] {_LOG} NVIDIA Optical Flow is unavailable; processing the batch "
                    f"with motion-adaptive residual stabilization={settings['temporal']:.2f}."
                )
            previous_source = None
            previous_residual = None
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="star7-dlss-resize") as resize_worker:
                prepared = resize_worker.submit(
                    _prepare_rgb_frame, cpu, 0, channels, output_width, output_height
                )
                for index in range(batch):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    frame_np = prepared.result()
                    if index + 1 < batch:
                        prepared = resize_worker.submit(
                            _prepare_rgb_frame, cpu, index + 1, channels, output_width, output_height
                        )
                    processed = _process_frame(library, frame_np, settings, index, temporal)
                    strength = min(max(float(settings["strength"]), 0.0), 2.0)
                    processed = np.clip(frame_np + (processed - frame_np) * strength, 0.0, 1.0)
                    if batch > 1 and not temporal:
                        processed, previous_residual = _stabilize_residual(
                            frame_np, processed, previous_source, previous_residual, settings["temporal"]
                        )
                        previous_source = frame_np
                    output[index].copy_(torch.from_numpy(np.ascontiguousarray(processed)).clamp_(0.0, 1.0))
                    if alpha_output is not None:
                        alpha = _resize_lanczos(cpu[index, ..., 3:4].numpy(), output_width, output_height)
                        alpha_output[index].copy_(torch.from_numpy(alpha))
                    progress.update_absolute(index + 1, batch)

        result = torch.cat((output, alpha_output), dim=-1) if alpha_output is not None else output
        # Keep large video batches on CPU after processing. Returning a 2x batch to the
        # input GPU can needlessly consume 4x the original frame-storage VRAM.
        return (result.to(dtype=图像.dtype),)


NODE_CLASS_MAPPINGS = {"Star7DLSSNeuralEnhance": Star7DLSSNeuralEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"Star7DLSSNeuralEnhance": "DLSS 神经画质增强 V2 - Star7"}
