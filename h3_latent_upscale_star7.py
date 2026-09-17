from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import copy
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.utils
import folder_paths

from .refine_model_options import (
    INHERIT_FIRST_PASS,
    apply_selected_lora,
    lora_choices,
)


_LOG = logging.getLogger("Star7H3LatentUpscale")
_CONTEXT_TYPE = "STAR7_H3_REFINE_CONTEXT"
_MODEL_NAME = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
# Only checkpoints validated end-to-end by this node belong here. The public
# BF16/FP32 files are precision copies of the same training and this runtime
# deliberately computes the learned handoff in FP16, so listing them would add
# download size without offering another quality model.
_MODEL_VARIANTS = (_MODEL_NAME,)
_MODEL_SHA256 = "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2"
_MODEL_URLS = (
    "https://hf-mirror.com/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/"
    + _MODEL_NAME,
    "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/"
    + _MODEL_NAME,
)
_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.RLock()
_VERIFIED_MODELS: dict[str, tuple[int, int]] = {}
_PIXEL_STRIDE = 16
_PIXEL_ALIGN = 32
_TURBO_PRESETS = {
    # These are paired step/strength recipes, not claims that one sigma ladder is
    # universal. BasicScheduler derives the real ladder from the connected model's
    # sampling shift and the node reports that exact ladder in the UI.
    "平衡高清": {"target_megapixels": 1.0, "refine_steps": 2, "refine_strength": 0.25},
    "高质量": {"target_megapixels": 1.0, "refine_steps": 3, "refine_strength": 0.30},
    "远景小脸": {"target_megapixels": 1.0, "refine_steps": 3, "refine_strength": 0.36},
    "高速运动": {"target_megapixels": 1.0, "refine_steps": 2, "refine_strength": 0.18},
}
_BASE_PRESETS = {
    # Strength selects the native-flow starting point; steps only subdivide the
    # selected range. More quality steps therefore do not silently repaint from
    # a higher-noise point.
    "平衡高清": {"target_megapixels": 1.0, "refine_steps": 4, "refine_strength": 0.20},
    "高质量": {"target_megapixels": 1.0, "refine_steps": 6, "refine_strength": 0.25},
    "远景小脸": {"target_megapixels": 1.0, "refine_steps": 5, "refine_strength": 0.30},
    "高速运动": {"target_megapixels": 1.0, "refine_steps": 4, "refine_strength": 0.15},
}
_PDD_PRESET_STEPS = {
    "平衡高清": 2,
    "高质量": 3,
    "远景小脸": 4,
    "高速运动": 1,
}
# Kept as the public/internal compatibility name used by existing tests and
# callers. Runtime preset selection is profile-aware below.
_PRESETS = _TURBO_PRESETS

_LATENT_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
_LATENT_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def _unpack(value):
    if hasattr(value, "args"):
        return value.args
    if isinstance(value, (tuple, list)):
        return value
    raise RuntimeError(f"Unexpected ComfyUI node output: {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_directory() -> Path:
    try:
        paths = folder_paths.get_folder_paths("latent_upscale_models")
    except Exception:
        paths = []
    if not paths:
        paths = [str(Path(folder_paths.models_dir) / "latent_upscale_models")]
    target = Path(paths[0])
    target.mkdir(parents=True, exist_ok=True)
    return target


def _model_choices():
    _model_directory()
    # This folder is shared with LTX and other latent upscalers. Only expose
    # reviewed H3 3D variants so an unrelated checkpoint is never presented as
    # a valid choice merely because it has a supported file extension.
    return list(_MODEL_VARIANTS)


def _ensure_model(model_name: str = _MODEL_NAME) -> Path:
    selected = str(model_name or _MODEL_NAME).strip()
    if Path(selected).name != selected:
        raise RuntimeError(f"Invalid H3 latent-upscaler filename: {selected!r}")
    target = _model_directory() / selected
    with _MODEL_LOCK:
        if target.is_file():
            # Unknown local checkpoints are architecture-validated by
            # _load_model. The pinned default additionally gets a SHA-256 check.
            if selected != _MODEL_NAME:
                return target
            stat = target.stat()
            cache_key = str(target.resolve()).casefold()
            fingerprint = (int(stat.st_size), int(stat.st_mtime_ns))
            if _VERIFIED_MODELS.get(cache_key) == fingerprint:
                return target
            actual = _sha256(target)
            if actual.casefold() == _MODEL_SHA256:
                _VERIFIED_MODELS[cache_key] = fingerprint
                return target
            raise RuntimeError(
                f"H3 latent-upscaler model checksum mismatch: {target}. "
                "Delete the damaged or unsupported file and run again."
            )

        if selected != _MODEL_NAME:
            raise RuntimeError(
                f"Selected H3 latent-upscaler model is missing: '{target}'. "
                f"Place a compatible 3D checkpoint in '{target.parent}' or select '{_MODEL_NAME}'."
            )

        failures = []
        partial = target.with_suffix(target.suffix + ".part")
        for url in _MODEL_URLS:
            try:
                _LOG.info("Star7 H3 HD | downloading model | %s", url)
                request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Star7/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as out:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                actual = _sha256(partial)
                if actual.casefold() != _MODEL_SHA256:
                    raise RuntimeError(f"SHA-256 mismatch: {actual}")
                os.replace(partial, target)
                stat = target.stat()
                _VERIFIED_MODELS[str(target.resolve()).casefold()] = (
                    int(stat.st_size), int(stat.st_mtime_ns)
                )
                _LOG.info("Star7 H3 HD | model ready | %s", target)
                return target
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
        raise RuntimeError(
            "Star7 H3 HD requires minimax_h3_latent_upscaler_3d_fp16.safetensors. "
            f"Automatic download failed; place it in '{target.parent}'. Sources: "
            + " | ".join(failures)
        )


def _send_sigma_status(
    node_id, sigma_summary: str, steps: int, strength: float, shift, profile: str
) -> None:
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_sync("star7-h3-hd-sigmas", {
            "node_id": node_id,
            "sigmas": str(sigma_summary),
            "steps": int(steps),
            "strength": round(float(strength), 4),
            "shift": None if shift is None else round(float(shift), 4),
            "profile": str(profile),
        })
    except Exception as exc:
        _LOG.debug("Unable to update the Star7 H3 HD sigma widget: %s", exc)


def _prompt_ancestors(prompt, node_id):
    """Return prompt nodes feeding this node without trusting arbitrary document text."""
    if not isinstance(prompt, dict) or node_id is None:
        return []
    pending = [str(node_id)]
    visited = set()
    result = []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        node = prompt.get(current)
        if not isinstance(node, dict):
            continue
        result.append(node)
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if (
                isinstance(value, (list, tuple)) and len(value) == 2
                and str(value[0]) in prompt
            ):
                pending.append(str(value[0]))
    return result


def _pdd_partition(prompt, node_id, model=None):
    """Resolve the trained PDD fine-grid partition already applied upstream."""
    for node in _prompt_ancestors(prompt, node_id):
        class_type = str(node.get("class_type", "")).casefold()
        inputs = node.get("inputs", {})
        if "pdd" not in class_type or not isinstance(inputs, dict):
            continue
        if inputs.get("enabled", True) is False:
            continue
        raw_partition = str(inputs.get("partition", "") or "").strip()
        if raw_partition:
            try:
                sizes = [int(item.strip()) for item in raw_partition.split(",") if item.strip()]
            except ValueError as exc:
                raise RuntimeError(f"Invalid upstream PDD partition: {raw_partition}") from exc
        else:
            try:
                nfe = int(inputs.get("nfe", 8))
            except (TypeError, ValueError):
                nfe = 8
            defaults = {
                8: [4] * 8,
                4: [8] * 4,
                6: [8, 8, 4, 4, 4, 4],
            }
            sizes = defaults.get(nfe)
            if sizes is None:
                raise RuntimeError(
                    f"Unsupported upstream PDD NFE {nfe}; expected 4, 6, or 8"
                )
        if sum(sizes) != 32 or any(size not in (4, 8) for size in sizes):
            raise RuntimeError(
                "PDD partition must contain only trained 4/8-row blocks summing to 32"
            )
        return sizes

    # Current ComfyUI can also load a native stacked PDD head bank directly.
    # Its row count is visible from tensor shapes without loading the weights.
    if model is not None:
        try:
            final_layer = model.get_model_object("diffusion_model.final_layer")
            video_out = final_layer.video_out
            banks = int(video_out.weight.shape[0]) // int(video_out.out_features)
            if banks > 1:
                if banks != 32:
                    raise RuntimeError(
                        f"Unsupported native PDD head-bank size {banks}; expected 32 trained heads"
                    )
                return [4] * 8
        except RuntimeError:
            raise
        except Exception:
            pass
    return None


def _pdd_tail_sigmas(partition, keep_steps: int, shift: float, audio_shift: float = 3.0):
    if not partition:
        raise RuntimeError("PDD model has no recoverable trained partition")
    if not math.isclose(float(shift), 12.0, abs_tol=1e-6):
        raise RuntimeError(
            f"PDD Acc heads are trained for video Shift 12, but the connected model uses {shift}"
        )
    if not math.isclose(float(audio_shift), 3.0, abs_tol=1e-6):
        raise RuntimeError(
            f"PDD Acc heads are trained for audio Shift 3, but the connected model uses {audio_shift}"
        )
    keep = max(1, min(int(keep_steps), len(partition)))
    remaining = 32
    bounds = []
    for size in partition:
        t = remaining / 32.0
        bounds.append(float(shift) * t / (1.0 + (float(shift) - 1.0) * t))
        remaining -= int(size)
    bounds.append(0.0)
    return torch.tensor(bounds[-(keep + 1):], dtype=torch.float32)


def _pdd_tail_strength(partition, keep_steps: int) -> float:
    """Return the trained base-time fraction covered by the retained PDD tail."""
    if not partition:
        raise RuntimeError("PDD model has no recoverable trained partition")
    keep = max(1, min(int(keep_steps), len(partition)))
    return sum(int(size) for size in partition[-keep:]) / float(sum(partition))


def _native_flow_refine_sigmas(steps: int, strength: float, shift: float):
    """Subdivide a strength-selected tail of H3's shifted native-flow path."""
    if shift is None or not math.isfinite(float(shift)) or float(shift) <= 0.0:
        raise RuntimeError("Base H3 native-flow refinement requires a valid model Shift")
    count = max(1, int(steps))
    denoise = max(0.0, min(1.0, float(strength)))
    base = torch.linspace(denoise, 0.0, count + 1, dtype=torch.float32)
    shifted = float(shift) * base / (1.0 + (float(shift) - 1.0) * base)
    return shifted


def _crop_spatial(tensor, axis: int, start: int, end: int):
    if tensor is None:
        return None
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(start, end)
    return tensor[tuple(slices)].contiguous()


def _crop_tile(tensor, tile):
    if tensor is None:
        return None
    top, bottom, left, right = tile
    return tensor[..., top:bottom, left:right].contiguous()


def _normalize_tile_count(value: int):
    """Clamp API and legacy values without changing an explicitly chosen count."""
    return max(2, min(int(value), 64))


def _smart_tile_grid(height: int, width: int, requested_count: int):
    """Use long-edge strips while retaining full short-axis context."""
    count = _normalize_tile_count(requested_count)
    return (1, count) if width >= height else (count, 1)


def _axis_tile_regions(total: int, divisions: int, overlap_pixels: int):
    if divisions <= 1:
        return [(0, total)]
    divisions = min(int(divisions), max(1, total // 2))
    boundaries = [0]
    for index in range(1, divisions):
        boundary = int(round((total * index / divisions) / 2.0) * 2)
        boundary = max(boundaries[-1] + 2, min(total - 2, boundary))
        boundaries.append(boundary)
    boundaries.append(total)

    halo = max(0, int(math.ceil(int(overlap_pixels) / _PIXEL_STRIDE)))
    # The overlap is a halo on each side of a strip, so adjacent internal strips
    # share about twice this value. H3 crop edges stay 2x2-token aligned.
    halo = int(math.ceil(halo / 2.0) * 2) if halo else 0
    regions = []
    for index in range(divisions):
        start = boundaries[index] - (halo if index else 0)
        end = boundaries[index + 1] + (halo if index + 1 < divisions else 0)
        start = max(0, start - start % 2)
        end = min(total, end + (-end % 2))
        regions.append((start, end))
    return regions


def _spatial_tile_plan(video: torch.Tensor, tile_count: int, overlap_pixels: int):
    """Build even-aligned overlapping strips along the video's long edge."""
    height, width = int(video.shape[-2]), int(video.shape[-1])
    rows, columns = _smart_tile_grid(height, width, tile_count)
    rows = min(rows, max(1, height // 2))
    columns = min(columns, max(1, width // 2))
    y_regions = _axis_tile_regions(height, rows, overlap_pixels)
    x_regions = _axis_tile_regions(width, columns, overlap_pixels)
    tiles = [
        (top, bottom, left, right)
        for top, bottom in y_regions
        for left, right in x_regions
    ]
    return (len(y_regions), len(x_regions)), tiles


def _axis_blend_windows(regions, axis: int, device):
    windows = []
    for index, (start, end) in enumerate(regions):
        length = end - start
        weight = torch.ones(length, dtype=torch.float32, device=device)
        if index:
            overlap = max(0, regions[index - 1][1] - start)
            if overlap:
                phase = torch.arange(1, overlap + 1, device=device, dtype=torch.float32)
                phase = phase / float(overlap + 1)
                weight[:overlap] *= 0.5 - 0.5 * torch.cos(math.pi * phase)
        if index + 1 < len(regions):
            overlap = max(0, end - regions[index + 1][0])
            if overlap:
                phase = torch.arange(overlap, 0, -1, device=device, dtype=torch.float32)
                phase = phase / float(overlap + 1)
                weight[-overlap:] *= 0.5 - 0.5 * torch.cos(math.pi * phase)
        shape = [1, 1, 1, 1, 1]
        shape[axis] = length
        windows.append(weight.view(shape))
    return windows


def _tile_windows(tiles, height: int, width: int, device):
    y_regions = list(dict.fromkeys((top, bottom) for top, bottom, _, _ in tiles))
    x_regions = list(dict.fromkeys((left, right) for _, _, left, right in tiles))
    y_windows = _axis_blend_windows(y_regions, -2, device)
    x_windows = _axis_blend_windows(x_regions, -1, device)
    y_lookup = {region: value for region, value in zip(y_regions, y_windows)}
    x_lookup = {region: value for region, value in zip(x_regions, x_windows)}
    windows = []
    denominator = torch.zeros(
        (1, 1, 1, height, width), dtype=torch.float32, device=device
    )
    for top, bottom, left, right in tiles:
        window = y_lookup[(top, bottom)] * x_lookup[(left, right)]
        windows.append(window)
        denominator[..., top:bottom, left:right] += window
    return windows, denominator.clamp_min_(1e-8)


def _walk_inner(root):
    current = root
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "inner_model", None)


def _condition_groups(model):
    for item in _walk_inner(model):
        groups = getattr(item, "conds", None)
        if isinstance(groups, dict):
            return groups
    return {}


def _prepared_model(model):
    for item in _walk_inner(model):
        if hasattr(item, "latent_shapes"):
            return item
    return None


def _tile_keyframes(keyframes, tile):
    tiled = []
    for keyframe in keyframes or []:
        item = dict(keyframe)
        latent = item.get("latent")
        if latent is not None:
            if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
                raise RuntimeError("Star7 HD keyframe latent must be a 5D video tensor")
            item["latent"] = _crop_tile(latent, tile)
        tiled.append(item)
    return tiled


def _preserve_global_tile_positions(old_layout, tile_layout, tile):
    """Give a tile its coordinates within the full H3 canvas, not a new local origin."""
    old_segments = list(getattr(old_layout, "segments", ()) or ())
    tile_segments = list(getattr(tile_layout, "segments", ()) or ())
    old_positions = getattr(old_layout, "position_ids", None)
    signature = getattr(old_layout, "signature", None)
    if (
        old_positions is None or len(old_segments) != len(tile_segments)
        or not signature or len(signature) < 4
    ):
        return tile_layout

    full_h, full_w = int(signature[2]), int(signature[3])
    patch_h, patch_w = full_h // 2, full_w // 2
    top, bottom, left, right = (int(value) for value in tile)
    if patch_h <= 0 or patch_w <= 0 or any(value % 2 for value in tile):
        return tile_layout

    rebuilt = []
    for old_segment, tile_segment in zip(old_segments, tile_segments):
        old_start, old_stop, old_kind = old_segment
        tile_start, tile_stop, tile_kind = tile_segment
        if old_kind != tile_kind:
            return tile_layout
        positions = old_positions[int(old_start):int(old_stop)]
        if old_kind in {"cond", "video"}:
            frame_rows = patch_h * patch_w
            if frame_rows <= 0 or positions.shape[0] % frame_rows:
                return tile_layout
            latent_t = positions.shape[0] // frame_rows
            positions = positions.reshape(latent_t, patch_h, patch_w, 3)
            positions = positions[
                :, top // 2:bottom // 2, left // 2:right // 2, :
            ].reshape(-1, 3)
        if positions.shape[0] != int(tile_stop) - int(tile_start):
            return tile_layout
        rebuilt.append(positions)

    if rebuilt:
        tile_layout.position_ids = torch.cat(rebuilt, dim=0)
    return tile_layout


def _tile_payload(payload, tile_shapes, tile):
    old_layout = payload.get("layout")
    segments = getattr(old_layout, "segments", None)
    if not segments:
        return payload
    text_len = int(segments[0][1])
    video_shape = tile_shapes[0]
    audio_t = int(tile_shapes[1][-1]) if len(tile_shapes) > 1 else 0
    keyframes = _tile_keyframes(payload.get("keyframes"), tile)
    from comfy.ldm.minimax.model import PackedLayout

    args = (
        text_len,
        int(video_shape[2]),
        (int(video_shape[3]) + 1) // 2 * 2,
        (int(video_shape[4]) + 1) // 2 * 2,
        audio_t,
    )
    refs = list(payload.get("refs") or [])
    kwargs = {"keyframes": keyframes or None, "refs": refs or None}
    frame_count = payload.get("frame_count")
    try:
        layout = PackedLayout(*args, frame_count=frame_count, **kwargs)
    except TypeError:
        layout = PackedLayout(*args, **kwargs)
    layout = _preserve_global_tile_positions(old_layout, layout, tile)
    updated = dict(payload)
    updated["layout"] = layout
    if keyframes:
        updated["keyframes"] = keyframes
        updated["cond_video_latents"] = [
            item["latent"] for item in keyframes if item.get("latent") is not None
        ] + [item["latent"] for item in refs if item.get("latent") is not None]
        updated["cond_audio_latents"] = [
            item["audio_latent"] for item in keyframes
            if item.get("audio_latent") is not None
        ] + [
            item["audio_latent"] for item in refs
            if item.get("audio_latent") is not None
        ]
    return updated


class _SpatialTileModel:
    """Model-call proxy: preserves the chosen solver while tiling each H3 prediction."""

    def __init__(self, model, full_shapes, grid, tiles):
        self._model = model
        self._full_shapes = full_shapes
        self._grid = grid
        self._tiles = tiles

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, packed_x, sigma, **extra_args):
        streams = comfy.utils.unpack_latents(packed_x, self._full_shapes)
        video, audio = streams[0], streams[1] if len(streams) > 1 else None
        windows, denominator = _tile_windows(
            self._tiles, int(video.shape[-2]), int(video.shape[-1]), video.device
        )
        video_prediction = torch.zeros_like(video, dtype=torch.float32)
        audio_prediction = torch.zeros_like(audio, dtype=torch.float32) if audio is not None else None

        prepared = _prepared_model(self._model)
        saved_prepared_shapes = getattr(prepared, "latent_shapes", None) if prepared else None
        condition_restores = []
        payload_restores = []
        full_mask = extra_args.get("denoise_mask")
        mask_streams = (
            comfy.utils.unpack_latents(full_mask, self._full_shapes)
            if full_mask is not None else None
        )
        latent_streams = (
            comfy.utils.unpack_latents(self._model.latent_image, self._full_shapes)
            if getattr(self._model, "latent_image", None) is not None else None
        )
        noise_streams = (
            comfy.utils.unpack_latents(self._model.noise, self._full_shapes)
            if getattr(self._model, "noise", None) is not None else None
        )
        original_latent = getattr(self._model, "latent_image", None)
        original_noise = getattr(self._model, "noise", None)
        groups = _condition_groups(self._model)
        try:
            for tile_index, (tile, window) in enumerate(zip(self._tiles, windows)):
                top, bottom, left, right = tile
                tile_streams = [_crop_tile(video, tile)]
                if audio is not None:
                    tile_streams.append(audio)
                tile_x, tile_shapes = comfy.utils.pack_latents(tile_streams)
                if prepared is not None:
                    prepared.latent_shapes = tile_shapes

                for group in groups.values():
                    for cond in group or []:
                        model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                        shape_cond = model_conds.get("latent_shapes")
                        if shape_cond is not None and hasattr(shape_cond, "cond"):
                            condition_restores.append((shape_cond, shape_cond.cond))
                            shape_cond.cond = tile_shapes
                        payload_cond = model_conds.get("minimax_payload")
                        payload = getattr(payload_cond, "cond", None)
                        if isinstance(payload, dict):
                            payload_restores.append((payload_cond, payload_cond.cond))
                            payload_cond.cond = _tile_payload(
                                payload, tile_shapes, tile
                            )

                tile_args = dict(extra_args)
                tile_transformer_options = None
                tile_marker = "star7_spatial_tile_id"
                saved_tile_marker = None
                had_tile_marker = False
                model_options = tile_args.get("model_options")
                if isinstance(model_options, dict):
                    candidate = model_options.get("transformer_options")
                    if isinstance(candidate, dict):
                        tile_transformer_options = candidate
                        had_tile_marker = tile_marker in candidate
                        saved_tile_marker = candidate.get(tile_marker)
                        candidate[tile_marker] = tile_index
                if mask_streams is not None:
                    parts = [_crop_tile(mask_streams[0], tile)]
                    if len(mask_streams) > 1:
                        parts.append(mask_streams[1])
                    tile_args["denoise_mask"], _ = comfy.utils.pack_latents(parts)
                if latent_streams is not None:
                    parts = [_crop_tile(latent_streams[0], tile)]
                    if len(latent_streams) > 1:
                        parts.append(latent_streams[1])
                    self._model.latent_image, _ = comfy.utils.pack_latents(parts)
                if noise_streams is not None:
                    parts = [_crop_tile(noise_streams[0], tile)]
                    if len(noise_streams) > 1:
                        parts.append(noise_streams[1])
                    self._model.noise, _ = comfy.utils.pack_latents(parts)

                try:
                    tile_result = self._model(tile_x, sigma, **tile_args)
                finally:
                    if tile_transformer_options is not None:
                        if had_tile_marker:
                            tile_transformer_options[tile_marker] = saved_tile_marker
                        else:
                            tile_transformer_options.pop(tile_marker, None)
                predicted = comfy.utils.unpack_latents(tile_result, tile_shapes)
                slices = [slice(None)] * 5
                slices[-2] = slice(top, bottom)
                slices[-1] = slice(left, right)
                video_prediction[tuple(slices)] += predicted[0].float() * window
                if audio_prediction is not None:
                    audio_prediction += predicted[1].float()

                while condition_restores:
                    obj, value = condition_restores.pop()
                    obj.cond = value
                while payload_restores:
                    obj, value = payload_restores.pop()
                    obj.cond = value

            video_prediction /= denominator
            merged = [video_prediction.to(video.dtype)]
            if audio_prediction is not None:
                merged.append((audio_prediction / len(self._tiles)).to(audio.dtype))
            return comfy.utils.pack_latents(merged)[0]
        finally:
            if prepared is not None:
                prepared.latent_shapes = saved_prepared_shapes
            self._model.latent_image = original_latent
            self._model.noise = original_noise
            while condition_restores:
                obj, value = condition_restores.pop()
                obj.cond = value
            while payload_restores:
                obj, value = payload_restores.pop()
                obj.cond = value


def _wrap_sampler_for_spatial_tiles(sampler, full_shapes, grid, tiles):
    import comfy.samplers

    original_function = sampler.sampler_function

    def sample_with_tiles(model, x, sigmas, **kwargs):
        proxy = _SpatialTileModel(model, full_shapes, grid, tiles)
        return original_function(proxy, x, sigmas, **kwargs)

    sample_with_tiles.__name__ = f"star7_tiled_{getattr(original_function, '__name__', 'sampler')}"
    return comfy.samplers.KSAMPLER(
        sample_with_tiles,
        extra_options=dict(getattr(sampler, "extra_options", {}) or {}),
        inpaint_options=dict(getattr(sampler, "inpaint_options", {}) or {}),
    )


def _add_hd_endpoint_guides(positive, latent, context):
    """Re-encode original first/last frames against the actual HD latent canvas."""
    first_frame = context.get("first_frame")
    last_frame = context.get("last_frame")
    if first_frame is None and last_frame is None:
        return positive, "none"
    video_vae = context.get("video_vae")
    if video_vae is None:
        raise RuntimeError(
            "Star7 H3 HD received first/last-frame context without the Video VAE "
            "needed to re-encode the guides at the HD resolution."
        )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

    applied = []
    if first_frame is not None:
        positive = _unpack(MiniMaxH3AddGuide.execute(
            positive, latent, 0, vae=video_vae, image=first_frame
        ))[0]
        applied.append("first")
    if last_frame is not None:
        positive = _unpack(MiniMaxH3AddGuide.execute(
            positive, latent, -1, vae=video_vae, image=last_frame
        ))[0]
        applied.append("last")
    return positive, "+".join(applied)


def _sampling_profile(model, prompt, node_id, video_shift, selected_lora=INHERIT_FIRST_PASS):
    """Classify only schedule families we can distinguish without touching weights."""
    ancestors = _prompt_ancestors(prompt, node_id)
    searchable = []
    for node in ancestors:
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if (
            "pdd" in class_type.casefold()
            and isinstance(inputs, dict)
            and inputs.get("enabled", True) is False
        ):
            continue
        searchable.append(class_type)
        if isinstance(inputs, dict):
            searchable.extend(value for value in inputs.values() if isinstance(value, str))
    if str(selected_lora or INHERIT_FIRST_PASS) != INHERIT_FIRST_PASS:
        searchable.append(str(selected_lora))
    text = " ".join(searchable).casefold()
    partition = _pdd_partition(prompt, node_id, model)
    if partition:
        return "pdd", "PDD"
    if "pdd" in text or "parallel_decoding" in text:
        return "pdd_incomplete", "PDD incomplete"
    turbo_markers = ("turbo", "flashgen", "fasth3", "dmd2", "openvdn")
    if any(marker in text for marker in turbo_markers):
        return "turbo", "Turbo/Distilled"

    patches = getattr(model, "patches", {})
    patch_count = len(patches) if isinstance(patches, dict) else 0
    if patch_count == 0:
        return "base", "Base"
    # A renamed full-network Turbo LoRA usually patches hundreds of H3 weights.
    # Restrict this fallback to the native Turbo shift range; ordinary LoRAs at
    # shift 12 remain on the safer Base schedule.
    if video_shift is not None and float(video_shift) <= 8.0 and patch_count >= 200:
        return "turbo", "Turbo/Distilled (inferred)"
    return "base", "Base + standard LoRA"


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlock3D(nn.Module):
    def __init__(self, channels: int = 512, embed_dim: int = 64):
        super().__init__()
        self.in_layers = nn.Sequential(
            _group_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, channels * 2))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(0.1), nn.Conv3d(channels, channels, 3, padding=1)
        )

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(value)
        scale, shift = self.emb_layers(embedding).chunk(2, dim=1)
        hidden = self.out_norm(hidden) * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        return value + self.out_layers(hidden)


class _TemporalConv(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.norm = _group_norm(channels)
        # These attribute names are part of the published checkpoint contract.
        self.dwconv = nn.Conv3d(
            channels, channels, (5, 1, 1), padding=(2, 0, 0), groups=channels
        )
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pwconv(self.dwconv(F.silu(self.norm(value))))


class _H3Resizer3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(24, 512, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = self._blocks()
        self.out_blocks = self._blocks()
        self.norm_out = _group_norm(512)
        self.conv_out = nn.Conv3d(512, 24, 3, padding=1)

    @staticmethod
    def _blocks() -> nn.ModuleList:
        result = []
        for index in range(12):
            result.append(_ResBlock3D())
            if index % 2 == 0:
                result.append(_TemporalConv())
        return nn.ModuleList(result)

    @staticmethod
    def _apply_blocks(value, embedding, blocks):
        for block in blocks:
            value = block(value, embedding) if isinstance(block, _ResBlock3D) else block(value)
        return value

    def forward(self, latent, effective_scale: float, target_size):
        embedding = self.embed(latent.new_tensor([[float(effective_scale) - 1.0]])).expand(
            latent.shape[0], -1
        )
        hidden = self._apply_blocks(self.conv_in(latent), embedding, self.in_blocks)
        hidden = F.interpolate(hidden, size=target_size, mode="trilinear", align_corners=False)
        hidden = self._apply_blocks(hidden, embedding, self.out_blocks)
        return self.conv_out(F.silu(self.norm_out(hidden)))


@dataclass
class _CachedModel:
    patcher: comfy.model_patcher.ModelPatcher
    path: Path


_MODEL_CACHE: dict[str, _CachedModel] = {}


def _load_model(path: Path) -> _CachedModel:
    key = str(path.resolve()).casefold()
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        with torch.device("meta"):
            network = _H3Resizer3D()
        expected = network.state_dict()
        if set(state) != set(expected):
            raise RuntimeError(
                "The selected file is not the supported Star7 H3 3D latent-upscaler model"
            )
        mismatch = [name for name in state if tuple(state[name].shape) != tuple(expected[name].shape)]
        if mismatch:
            raise RuntimeError(f"H3 latent-upscaler tensor shape mismatch: {mismatch[:3]}")
        network.load_state_dict(state, strict=True, assign=True)
        del state, expected
        network.to(device=torch.device("cpu"), dtype=torch.float16)
        network.eval()
        patcher = comfy.model_patcher.CoreModelPatcher(
            network,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        cached = _CachedModel(patcher=patcher, path=path)
        _MODEL_CACHE[key] = cached
        return cached


def _target_geometry(video: torch.Tensor, target_megapixels: float):
    source_width = int(video.shape[-1]) * _PIXEL_STRIDE
    source_height = int(video.shape[-2]) * _PIXEL_STRIDE
    source_area = source_width * source_height
    requested_area = max(source_area, float(target_megapixels) * 1_000_000.0)
    aspect = source_width / source_height
    ideal_width = math.sqrt(requested_area * aspect)
    ideal_height = ideal_width / aspect
    width = max(source_width, int(round(ideal_width / _PIXEL_ALIGN)) * _PIXEL_ALIGN)
    height = max(source_height, int(round(ideal_height / _PIXEL_ALIGN)) * _PIXEL_ALIGN)
    scale_x, scale_y = width / source_width, height / source_height
    if max(scale_x, scale_y) > 4.0:
        raise RuntimeError(
            f"Requested H3 upscale exceeds the trained 4x range: {scale_x:.2f}x{scale_y:.2f}"
        )
    return source_width, source_height, width, height, math.sqrt(scale_x * scale_y)


def _temporal_weight(length: int, overlap: int, first: bool, last: bool, device, dtype):
    weight = torch.ones((1, 1, length, 1, 1), device=device, dtype=dtype)
    fade = min(overlap, length // 2)
    if fade and not first:
        weight[:, :, :fade] = torch.linspace(0.05, 1.0, fade, device=device, dtype=dtype)[None, None, :, None, None]
    if fade and not last:
        weight[:, :, -fade:] = torch.linspace(1.0, 0.05, fade, device=device, dtype=dtype)[None, None, :, None, None]
    return weight


def _run_resizer(network, value, scale: float, target_h: int, target_w: int):
    frames = int(value.shape[2])
    chunk, overlap = 40, 8
    if frames <= chunk:
        return network(value, scale, (frames, target_h, target_w))
    stride = chunk - overlap
    result = torch.zeros(
        (value.shape[0], 24, frames, target_h, target_w), device=value.device, dtype=value.dtype
    )
    weights = torch.zeros((1, 1, frames, 1, 1), device=value.device, dtype=value.dtype)
    starts = list(range(0, frames, stride))
    if starts[-1] + chunk < frames:
        starts.append(frames - chunk)
    for index, start in enumerate(starts):
        end = min(frames, start + chunk)
        piece = value[:, :, start:end]
        actual = int(piece.shape[2])
        if actual < chunk:
            piece = F.pad(piece, (0, 0, 0, 0, 0, chunk - actual), mode="replicate")
        output = network(piece, scale, (int(piece.shape[2]), target_h, target_w))[:, :, :actual]
        weight = _temporal_weight(
            actual, overlap, index == 0, index == len(starts) - 1,
            output.device, output.dtype,
        )
        result[:, :, start:end] += output * weight
        weights[:, :, start:end] += weight
    return result / weights.clamp_min(1e-6)


def _upscale_video(
    sampled_av_latent: dict, target_megapixels: float,
    upscale_model: str = _MODEL_NAME,
):
    samples = sampled_av_latent.get("samples")
    if samples is None or not getattr(samples, "is_nested", False):
        raise RuntimeError("Star7 H3 HD requires a sampled native H3 audio-video LATENT")
    members = list(samples.unbind())
    if len(members) < 2 or members[0].ndim != 5 or int(members[0].shape[1]) != 24:
        raise RuntimeError("Invalid H3 latent: expected video [B,24,T,H,W] plus audio latent")
    video, audio = members[0], members[1]
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise RuntimeError("Input H3 latent contains NaN or Inf")
    source_w, source_h, output_w, output_h, scale = _target_geometry(video, target_megapixels)
    if (source_w, source_h) == (output_w, output_h):
        output = dict(sampled_av_latent)
        output.pop("noise_mask", None)
        return output, video, audio, source_w, source_h, output_w, output_h, False

    path = _ensure_model(upscale_model)
    cached = _load_model(path)
    target_h, target_w = output_h // _PIXEL_STRIDE, output_w // _PIXEL_STRIDE
    device = cached.patcher.load_device
    activation_estimate = int(video.shape[0]) * 512 * min(int(video.shape[2]), 40) * target_h * target_w * 8
    model_management.load_models_gpu(
        [cached.patcher], memory_required=activation_estimate, force_full_load=True
    )
    work = video.to(device=device, dtype=torch.float16)
    mean = work.new_tensor(_LATENT_MEAN).view(1, 24, 1, 1, 1)
    std = work.new_tensor(_LATENT_STD).view(1, 24, 1, 1, 1)
    try:
        with torch.inference_mode():
            normalized = (work - mean) / std
            output_video = _run_resizer(cached.patcher.model, normalized, scale, target_h, target_w)
            output_video = output_video * std + mean
        if not torch.isfinite(output_video).all():
            raise RuntimeError("H3 latent upscaler produced NaN or Inf")
        output_video = output_video.to(
            device=model_management.intermediate_device(), dtype=video.dtype
        )
    finally:
        model_management.unload_model_and_clones(cached.patcher)
        del work, mean, std
        gc.collect()
        model_management.soft_empty_cache()
    output = dict(sampled_av_latent)
    output.pop("noise_mask", None)
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, audio))
    return output, output_video, audio, source_w, source_h, output_w, output_h, True


class MiniMaxH3OneClickHDStar7:
    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import _attention_backend_choices

        return {
            "required": {
                "enable_hd": ("BOOLEAN", {"default": True}),
                "sampled_av_latent": ("LATENT",),
                "h3_context": (_CONTEXT_TYPE,),
                "upscale_model": (
                    _model_choices(),
                    {
                        "default": _MODEL_NAME,
                        "tooltip": (
                            "3D H3 latent-upscaler checkpoint in models/latent_upscale_models. "
                            "The pinned FP16 default downloads automatically when missing."
                        ),
                    },
                ),
                "second_pass_lora": (
                    lora_choices(),
                    {
                        "default": INHERIT_FIRST_PASS,
                        "tooltip": (
                            "Inherit the first-pass model unchanged, or apply the selected "
                            "model-only LoRA for this HD refinement only."
                        ),
                    },
                ),
                "second_pass_lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                        "tooltip": "Model strength for the selected second-pass LoRA.",
                    },
                ),
                "second_pass_attention": (
                    [INHERIT_FIRST_PASS, *_attention_backend_choices()],
                    {"default": INHERIT_FIRST_PASS},
                ),
                "preset": (
                    ["平衡高清", "高质量", "远景小脸", "高速运动", "自定义"],
                    {"default": "平衡高清"},
                ),
                "target_megapixels": (
                    "FLOAT", {"default": 1.0, "min": 0.20, "max": 36.0, "step": 0.05}
                ),
                "refine_steps": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "refine_strength": (
                    "FLOAT",
                    {
                        "default": 0.25, "min": 0.0, "max": 0.50, "step": 0.01,
                        "tooltip": (
                            "Fraction of the denoising path replayed by refinement. Higher "
                            "values start from noisier latents and allow more repainting; 0 "
                            "disables refinement. PDD uses its trained tail boundaries."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "tile_count": ("INT", {"default": 2, "min": 2, "max": 64, "step": 1}),
                "tile_overlap": (
                    "INT", {"default": 128, "min": 32, "max": 512, "step": 32}
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("hd_av_latent", "report")
    FUNCTION = "upscale"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "One-node H3 learned latent upscale plus an optional short low-noise refinement. "
        "Optional long-edge prediction strips reduce peak refinement VRAM while retaining "
        "full short-axis context. "
        "Video is enhanced; native H3 audio is preserved exactly. Connect after the main "
        "sampler and before H3 VAE Decode or Star7 Face Repair."
    )

    def upscale(
        self, sampled_av_latent, h3_context, enable_hd=True,
        upscale_model=_MODEL_NAME, second_pass_lora=INHERIT_FIRST_PASS,
        second_pass_lora_strength=1.0,
        second_pass_attention=INHERIT_FIRST_PASS, preset="平衡高清",
        target_megapixels=1.0, refine_steps=2, refine_strength=0.25, seed=0,
        enable_tiling=False, tile_count=2, tile_overlap=128,
        unique_id=None, prompt=None,
    ):
        started = time.perf_counter()
        if not bool(enable_hd):
            _send_sigma_status(unique_id, "off", 0, 0.0, None, "disabled")
            report = "Star7 H3 HD bypassed | HD second pass disabled | latent unchanged"
            _LOG.info(report)
            return sampled_av_latent, report
        context = dict(h3_context or {})
        model, positive = context.get("model"), context.get("positive")
        if model is None or positive is None:
            raise RuntimeError(
                "Star7 H3 HD context is incomplete. Connect H3 context from "
                "MiniMax H3 All-in-one Conditioning - Star7."
            )
        get_model_object = getattr(model, "get_model_object", None)
        model_sampling = get_model_object("model_sampling") if callable(get_model_object) else None
        video_shift = getattr(model_sampling, "shift", None)
        audio_shift = getattr(model_sampling, "audio_shift", None)
        second_pass_lora_strength = float(second_pass_lora_strength)
        profile_lora = (
            second_pass_lora
            if second_pass_lora_strength != 0.0
            else INHERIT_FIRST_PASS
        )
        profile_key, profile_name = _sampling_profile(
            model, prompt, unique_id, video_shift, profile_lora
        )
        if profile_key == "pdd_incomplete":
            raise RuntimeError(
                "A PDD file/name was detected upstream, but the trained PDD final-layer head bank "
                "was not applied. Load it with MiniMax H3 PDD Acc LoRA (Apply), not a normal LoRA "
                "loader. Its separate Scheduler node is not required for Star7 one-click HD."
            )
        pdd_partition = _pdd_partition(prompt, unique_id, model) if profile_key == "pdd" else None
        active_presets = _TURBO_PRESETS if profile_key == "turbo" else _BASE_PRESETS
        if profile_key == "pdd" and preset in _PDD_PRESET_STEPS:
            target_megapixels = 1.0
            refine_steps = min(_PDD_PRESET_STEPS[preset], len(pdd_partition))
            refine_strength = _pdd_tail_strength(pdd_partition, refine_steps)
        elif preset in active_presets:
            values = active_presets[preset]
            target_megapixels = values["target_megapixels"]
            refine_steps = values["refine_steps"]
            refine_strength = values["refine_strength"]
        elif preset != "自定义":
            raise RuntimeError(f"Unknown Star7 H3 HD preset: {preset}")
        target_megapixels = float(target_megapixels)
        # Keep API submissions and legacy workflows inside the same range as
        # the widget.  Older graphs may still carry the former zero value.
        refine_steps = max(1, min(50, int(refine_steps)))
        refine_strength = max(0.0, min(0.50, float(refine_strength)))

        _LOG.info(
            "Star7 H3 HD | profile=%s preset=%s target=%.2fMP refine=%d strength=%.2f",
            profile_name, preset, target_megapixels, refine_steps, refine_strength,
        )
        upscale_started = time.perf_counter()
        output, video, original_audio, source_w, source_h, output_w, output_h, resized = (
            _upscale_video(sampled_av_latent, target_megapixels, upscale_model)
        )
        upscale_seconds = time.perf_counter() - upscale_started
        if resized:
            _LOG.info(
                "Star7 H3 HD | latent upscale completed | %dx%d -> %dx%d | %.2fs",
                source_w, source_h, output_w, output_h, upscale_seconds,
            )
        else:
            source_megapixels = source_w * source_h / 1_000_000.0
            sigma_summary = "off"
            _send_sigma_status(
                unique_id, sigma_summary, 0, 0.0, video_shift, profile_name
            )
            report = (
                f"Star7 H3 HD skipped | source={source_w}x{source_h} "
                f"({source_megapixels:.2f}MP) target={target_megapixels:.2f}MP | "
                "target <= source; first-pass latent returned unchanged"
            )
            _LOG.info(report)
            return sampled_av_latent, report

        refine_seconds = 0.0
        sigma_summary = "off"
        scheduler_name = "off"
        keyframe_summary = "none"
        if refine_steps > 0 and refine_strength > 0.0:
            from comfy_extras.nodes_custom_sampler import (
                BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
            )

            output = dict(output)
            output["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video), torch.zeros_like(original_audio))
            )
            if profile_key == "pdd":
                sigmas = _pdd_tail_sigmas(
                    pdd_partition, refine_steps, video_shift, audio_shift
                )
                refine_steps = len(sigmas) - 1
                refine_strength = _pdd_tail_strength(pdd_partition, refine_steps)
                scheduler_name = "PDD trained tail"
                sampler_name = "euler"
            elif profile_key == "base":
                sigmas = _native_flow_refine_sigmas(
                    refine_steps, refine_strength, video_shift
                )
                refine_steps = len(sigmas) - 1
                scheduler_name = (
                    f"native_flow refine (strength={refine_strength:.2f}, "
                    f"steps={refine_steps})"
                )
                sampler_name = "euler"
            else:
                sigmas = _unpack(BasicScheduler.execute(
                    model, "simple", refine_steps, refine_strength
                ))[0]
                scheduler_name = "simple"
                sampler_name = "res_multistep"
            sigma_values = [float(value) for value in sigmas.detach().float().cpu().reshape(-1)]
            sigma_summary = ",".join(f"{value:.4f}" for value in sigma_values)
            _LOG.info(
                "Star7 H3 HD | internal sigmas=[%s] | %s scheduler",
                sigma_summary, scheduler_name,
            )
            _send_sigma_status(
                unique_id, sigma_summary, refine_steps, refine_strength,
                video_shift, profile_name,
            )
            positive, keyframe_summary = _add_hd_endpoint_guides(
                positive, output, context
            )
            if keyframe_summary != "none":
                _LOG.info(
                    "Star7 H3 HD | endpoint guides re-encoded at %dx%d | %s",
                    output_w, output_h, keyframe_summary,
                )
            refine_model, lora_summary = apply_selected_lora(
                self, model, second_pass_lora, second_pass_lora_strength
            )
            attention_summary = "inherit first pass"
            attention_config_snapshot = None
            attention_runtime_config = None
            if str(second_pass_attention) not in {"", INHERIT_FIRST_PASS, "inherit", "inherit first pass"}:
                from . import nodes as chunk_nodes

                requested_attention = str(second_pass_attention)
                attention_config_snapshot = copy.deepcopy(chunk_nodes._CONFIG)
                try:
                    refine_model = chunk_nodes.install_model_patch(
                        refine_model,
                        int(chunk_nodes._CONFIG.get("chunk_tokens", 8192)),
                        bool(chunk_nodes._CONFIG.get("auto_halve_on_oom", True)),
                        bool(chunk_nodes._CONFIG.get("verbose", True)),
                        int(chunk_nodes._CONFIG.get("mlp_chunk_tokens", 8192)),
                        bool(chunk_nodes._CONFIG.get("out_proj_memory_protection", True)),
                        bool(chunk_nodes._CONFIG.get("reuse_mlp_weights", True)),
                        requested_attention,
                        unique_id,
                        qkv_chunk_tokens=int(chunk_nodes._CONFIG.get("qkv_chunk_tokens", 8192)),
                        out_proj_chunk_tokens=int(chunk_nodes._CONFIG.get("out_proj_chunk_tokens", 4096)),
                    )
                    attention_runtime_config = copy.deepcopy(chunk_nodes._CONFIG)
                finally:
                    chunk_nodes._CONFIG.clear()
                    chunk_nodes._CONFIG.update(attention_config_snapshot)
                attention_summary = requested_attention
                _LOG.info(
                    "Star7 H3 HD | second-pass attention override | %s",
                    attention_summary,
                )
            guider = _unpack(BasicGuider.execute(refine_model, positive))[0]
            sampler = _unpack(KSamplerSelect.execute(sampler_name))[0]
            noise = _unpack(RandomNoise.execute(int(seed) & 0xFFFFFFFFFFFFFFFF))[0]
            tiling_summary = "off"
            if bool(enable_tiling):
                full_members = list(output["samples"].unbind())
                grid, tiles = _spatial_tile_plan(
                    full_members[0], int(tile_count), int(tile_overlap)
                )
                if len(tiles) > 1:
                    sampler = _wrap_sampler_for_spatial_tiles(
                        sampler, [tuple(item.shape) for item in full_members], grid, tiles
                    )
                    tiling_summary = (
                        f"requested={int(tile_count)} actual={len(tiles)} "
                        f"strips={'width' if grid[1] > 1 else 'height'} "
                        f"grid={grid[0]}x{grid[1]} halo={int(tile_overlap)}px/side "
                        f"shared~{int(tile_overlap) * 2}px "
                        f"max_tile={max((bottom - top) * (right - left) for top, bottom, left, right in tiles) * (_PIXEL_STRIDE ** 2) / 1_000_000.0:.2f}MP"
                    )
                    _LOG.info("Star7 H3 HD | spatial tiling enabled | %s", tiling_summary)
            refine_started = time.perf_counter()
            if attention_runtime_config is not None:
                chunk_nodes._CONFIG.clear()
                chunk_nodes._CONFIG.update(attention_runtime_config)
            try:
                sampled = _unpack(SamplerCustomAdvanced.execute(
                    noise, guider, sampler, sigmas, output
                ))[0]
            finally:
                if attention_config_snapshot is not None:
                    # Attention kernels read a small amount of Star7 runtime
                    # state. Restore it even after a failed second pass so a
                    # cached first-pass chunk node cannot inherit this override
                    # on the next queued workflow.
                    chunk_nodes._CONFIG.clear()
                    chunk_nodes._CONFIG.update(attention_config_snapshot)
            refine_seconds = time.perf_counter() - refine_started
            sampled_members = list(sampled["samples"].unbind())
            if not torch.isfinite(sampled_members[0]).all():
                raise RuntimeError("Star7 H3 HD refinement produced NaN or Inf")
            output = dict(sampled)
            output.pop("noise_mask", None)
            output["samples"] = comfy.nested_tensor.NestedTensor(
                (sampled_members[0], original_audio)
            )
            _LOG.info(
                "Star7 H3 HD | high-resolution refine completed | steps=%d %.2fs | audio locked",
                refine_steps, refine_seconds,
            )
        else:
            tiling_summary = "ignored (refine disabled)" if enable_tiling else "off"
            lora_summary = "ignored (refine disabled)"
            attention_summary = "ignored (refine disabled)"
            if context.get("first_frame") is not None or context.get("last_frame") is not None:
                keyframe_summary = "ignored (refine disabled)"
            _send_sigma_status(
                unique_id, sigma_summary, 0, 0.0, video_shift, profile_name
            )

        total = time.perf_counter() - started
        report = (
            f"Star7 H3 HD completed | preset={preset} | {source_w}x{source_h} -> "
            f"{output_w}x{output_h} ({output_w * output_h / 1_000_000:.2f} MP) | "
            f"upscaler={upscale_model} | "
            f"refine={refine_steps} step(s) strength={refine_strength:.2f} | "
            f"profile={profile_name} | "
            f"HD keyframes={keyframe_summary} | "
            f"schedule={scheduler_name} | "
            f"sigmas=[{sigma_summary}] | "
            f"second-pass LoRA={lora_summary} | "
            f"second-pass attention={attention_summary} | "
            f"tiling={tiling_summary} | "
            f"upscale={upscale_seconds:.2f}s refine={refine_seconds:.2f}s total={total:.2f}s | "
            "audio preserved exactly"
        )
        _LOG.info(report)
        return output, report


NODE_CLASS_MAPPINGS = {"MiniMaxH3OneClickHDStar7": MiniMaxH3OneClickHDStar7}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3OneClickHDStar7": "MiniMax H3 一键高清放大 - Star7"
}
