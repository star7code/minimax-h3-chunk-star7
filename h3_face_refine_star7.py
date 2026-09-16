from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import threading
import time
import urllib.request
from typing import Any

import torch

import comfy.nested_tensor
import folder_paths
from comfy_api.latest import io

from .vendor.h3facerefine.core import (
    H3FaceStitch,
    H3FaceTrackCrop,
    H3InjectVideoLatent,
    H3PerFrameDenoise,
    _continuity_cost,
    _detector_list,
    _rank_boxes,
)


_LOG = logging.getLogger("Star7H3FaceRefine")
_CONTEXT_TYPE = "STAR7_H3_REFINE_CONTEXT"
_CONTEXT_IO = io.Custom(_CONTEXT_TYPE)
_FACE_DETECTOR_NAME = "face_yolov8m.pt"
_FACE_DETECTOR_SHA256 = "717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f"
_FACE_DETECTOR_URLS = (
    "https://hf-mirror.com/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
)
_FACE_DETECTOR_LOCK = threading.Lock()
_FPS = 24
_MAX_REFERENCE_IMAGES = 9
_MAX_REFERENCE_VIDEO_FRAMES = 15 * _FPS
_MIN_RECOMMENDED_REFERENCE_VIDEO_FRAMES = 2 * _FPS
_MEGAPIXEL = 1024 * 1024
_FACE_REPAIR_OUTPUT_FLOOR_MP = 0.98
_MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)
_AUDIO_ALIAS_RE = re.compile(r"<\s*Audio\s+(D)\s*>", re.IGNORECASE)

_TASK_OPTIONS = [
    "自动判断 / Auto",
    "文生视频 / T2VA",
    "首帧生视频 / I2VA",
    "首尾帧生视频 / FL2VA",
    "尾帧生视频 / L2VA",
    "参考素材生视频 / Ref2VA",
    "混合条件 / Hybrid",
]
_TASK_IDS = {
    "auto": "auto", "t2va": "T2VA", "i2va": "I2VA", "fl2va": "FL2VA",
    "l2va": "L2VA", "ref2va": "Ref2VA", "hybrid": "Hybrid",
}
_AUDIO_OPTIONS = [
    "锁定原音 / Lock Source",
    "重混原音 / Remix Source",
    "仅作音频参考 / Reference Only",
    "模型原生生成 / Native",
]
_AUDIO_IDS = {
    "lock_source": "lock_source", "remix_source": "remix_source",
    "reference_only": "reference_only", "native": "native",
}
_REFERENCE_SIZE_OPTIONS = [
    "匹配生成画布 / Match",
    "保留高分辨率参考 / Max（显存较高）",
]


def _unpack(value):
    if hasattr(value, "args"):
        return value.args
    if isinstance(value, (tuple, list)):
        return value
    raise RuntimeError(f"Unexpected ComfyUI node output: {type(value)!r}")


def _copy_conditioning_without_keyframes(conditioning):
    """Keep prompt/reference conditioning, but remove full-frame anchors from the crop pass."""
    output = []
    for tensor, metadata in conditioning:
        copied = dict(metadata)
        copied.pop("minimax_keyframes", None)
        output.append([tensor, copied])
    return output


def _fit_audio_latent(encoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    encoded = encoded.to(device=target.device, dtype=target.dtype)
    result = torch.zeros_like(target)
    common = [min(int(a), int(b)) for a, b in zip(encoded.shape, target.shape)]
    slices = tuple(slice(0, n) for n in common)
    result[slices] = encoded[slices]
    return result


def _replace_audio_latent(latent: dict, encoded: torch.Tensor, denoise: float) -> dict:
    """Replace only H3's audio member and preserve any existing video mask."""
    members = list(latent["samples"].unbind())
    members[1] = _fit_audio_latent(encoded, members[1])
    output = dict(latent)
    output["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
    old_mask = output.get("noise_mask")
    video_mask = torch.ones_like(members[0])
    if old_mask is not None and getattr(old_mask, "is_nested", False):
        old_members = list(old_mask.unbind())
        if old_members:
            video_mask = old_members[0]
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, torch.full_like(members[1], float(denoise)))
    )
    return output


def _round_canvas(value: int) -> int:
    return max(32, int(round(int(value) / 32.0)) * 32)


def _option_id(value: str, known: dict[str, str], kind: str) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    for key, result in known.items():
        if lower == key or lower.endswith(f"/ {key.replace('_', ' ')}"):
            return result
    # Bilingual labels always retain the stable identifier after the slash.
    tail = lower.rsplit("/", 1)[-1].strip().replace(" ", "_")
    if tail in known:
        return known[tail]
    raise ValueError(f"Unknown {kind}: {value}")


def _reference_size_id(value: str) -> str:
    lower = str(value or "").lower()
    return "max" if "max" in lower else "match"


def _aligned_reference_frames(frame_count: int, target_frames: int) -> int:
    capped = min(int(frame_count), int(target_frames), _MAX_REFERENCE_VIDEO_FRAMES)
    return capped - ((capped - 5) % 17)


def _crop_audio_seconds(audio: dict | None, seconds: float):
    if audio is None:
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise ValueError("Reference video audio is missing waveform or sample_rate")
    samples = max(1, min(int(waveform.shape[-1]), round(float(seconds) * sample_rate)))
    output = dict(audio)
    output["waveform"] = waveform[..., :samples]
    return output


def _prepare_reference_videos(ref_videos: dict, ref_video_audios: dict, target_frames: int):
    videos, audios, warnings = {}, {}, []
    for key, frames in ref_videos.items():
        if frames.ndim != 4 or int(frames.shape[0]) < 5:
            raise ValueError(f"{key} needs at least 5 IMAGE frames")
        source_count = int(frames.shape[0])
        aligned = _aligned_reference_frames(source_count, target_frames)
        if source_count < _MIN_RECOMMENDED_REFERENCE_VIDEO_FRAMES:
            warnings.append(
                f"{key} is only {source_count / _FPS:.2f}s; it is usable but shorter than the recommended 2s"
            )
        if source_count > _MAX_REFERENCE_VIDEO_FRAMES:
            warnings.append(f"{key} exceeded 15s and was trimmed automatically")
        if source_count > target_frames:
            warnings.append(f"{key} exceeded the target duration and was trimmed automatically")
        videos[key] = frames[:aligned]
        suffix = key.rsplit("_", 1)[-1]
        audio_key = f"ref_video_audio_{suffix}"
        soundtrack = ref_video_audios.get(audio_key)
        if soundtrack is not None:
            audios[audio_key] = _crop_audio_seconds(soundtrack, aligned / _FPS)
    orphan = sorted(set(ref_video_audios) - set(audios))
    if orphan:
        warnings.append("unpaired reference-video audio ignored: " + ", ".join(orphan))
    return videos, audios, warnings


def _prepare_prompt_tags(prompt: str, pictures: int, videos: int, audios: int,
                         primary_audio_ordinal: int = 0, audio_tag_map=None,
                         audio_alias_map=None):
    # audio_tag_map preserves the official visible reference order while
    # skipping the driving-audio slot. audio_alias_map gives that one special
    # source the readable <Audio D> alias.
    standalone_audio_count = len(audio_tag_map) if audio_tag_map else audios
    limits = {"picture": pictures, "video": videos, "audio": standalone_audio_count}
    warnings = []

    hidden_aliases = []

    def replace_audio_alias(match: re.Match):
        alias = re.sub(r"\s+", "", match.group(1)).upper()
        actual = (audio_alias_map or {}).get(alias)
        if actual is None:
            warnings.append(f"unconnected <Audio {alias}> was treated as plain text")
            return f"Audio {alias}"
        placeholder = f"__STAR7_AUDIO_ALIAS_{len(hidden_aliases)}__"
        hidden_aliases.append((placeholder, f"<Audio {actual}>"))
        return placeholder

    prompt = _AUDIO_ALIAS_RE.sub(replace_audio_alias, prompt or "")

    def replace(match: re.Match):
        media_type = (match.group(1) or match.group(3)).lower()
        ordinal = int(match.group(2) or match.group(4))
        official = "Picture" if media_type in {"image", "picture"} else media_type.title()
        limit = limits[official.lower()]
        if ordinal == 0 and limit:
            ordinal = 1
            warnings.append(f"{official} 0 was mapped to <{official} 1>")
        if official == "Audio" and audio_tag_map:
            if 1 <= ordinal <= limit:
                return f"<Audio {audio_tag_map[ordinal]}>"
            if limit == 1 and ordinal > 1:
                warnings.append(f"Audio {ordinal} was mapped to <Audio 1>")
                return f"<Audio {audio_tag_map[1]}>"
        else:
            if official == "Audio" and primary_audio_ordinal and ordinal == 1:
                ordinal = primary_audio_ordinal
            if 1 <= ordinal <= limit:
                return f"<{official} {ordinal}>"
        if not (official == "Audio" and audio_tag_map) and 1 <= ordinal <= limit:
            return f"<{official} {ordinal}>"
        if limit == 1 and ordinal > 1:
            warnings.append(f"{official} {ordinal} was mapped to <{official} 1>")
            return f"<{official} 1>"
        if match.group(1) is not None:
            warnings.append(f"unconnected <{official} {ordinal}> was treated as plain text")
            return f"{official} {ordinal}"
        return match.group(0)

    normalized = _MEDIA_TAG_RE.sub(replace, prompt)
    for placeholder, official_tag in hidden_aliases:
        normalized = normalized.replace(placeholder, official_tag)
    return normalized, list(dict.fromkeys(warnings))


def _collect_reference_images(ref_images, legacy_inputs=None):
    def slot(name):
        try:
            return int(str(name).rsplit("_", 1)[-1])
        except ValueError:
            return _MAX_REFERENCE_IMAGES

    values = dict(ref_images or {})
    values.update({
        name: image for name, image in (legacy_inputs or {}).items()
        if name.startswith("ref_image_") and name not in values
    })
    connected = [
        image for _, image in sorted(values.items(), key=lambda item: slot(item[0]))
        if image is not None
    ]
    return {
        f"ref_image_{index}": image
        for index, image in enumerate(connected)
    }


def _execute_reference_to_video(reference_node, clip, video_vae, audio_vae, prompt,
                                width, height, length, reference_quality,
                                ref_images, ref_videos, ref_video_audios, ref_audios):
    """Use stable parameter names across the H3 reference-node API reorder."""
    return reference_node.execute(
        clip=clip,
        vae=video_vae,
        audio_vae=audio_vae,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        ref_image_size=reference_quality,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
    )


def _decode_video_frames(vae, latent: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latent)
    if images.ndim == 5:
        if int(images.shape[0]) != 1:
            raise ValueError("Star7 H3 Face Repair currently supports one generated video at a time")
        images = images.reshape(-1, *images.shape[-3:])
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"Unexpected MiniMax H3 Video VAE output shape: {tuple(images.shape)}")
    return images[..., :3]


def _face_repair_output_size(
    width: int, height: int, preserve_detail: bool
) -> tuple[int, int]:
    """Choose a practical ~1 MP floor for the final composite; never shrink."""
    width, height = int(width), int(height)
    source_pixels = width * height
    target_pixels = int(_FACE_REPAIR_OUTPUT_FLOOR_MP * _MEGAPIXEL)
    if not preserve_detail or source_pixels >= target_pixels:
        return width, height

    scale = (target_pixels / max(source_pixels, 1)) ** 0.5
    ideal_width, ideal_height = width * scale, height * scale
    centre_width = max(width, int(round(ideal_width / 32.0)) * 32)
    centre_height = max(height, int(round(ideal_height / 32.0)) * 32)
    candidates = []
    for output_width in range(max(32, centre_width - 64), centre_width + 65, 32):
        for output_height in range(max(32, centre_height - 64), centre_height + 65, 32):
            if output_width < width or output_height < height:
                continue
            pixels = output_width * output_height
            if pixels < target_pixels:
                continue
            aspect_error = abs((output_width / output_height) / (width / height) - 1.0)
            overshoot = (pixels - target_pixels) / target_pixels
            candidates.append((aspect_error * 10.0 + overshoot, output_width, output_height))
    if candidates:
        _, output_width, output_height = min(candidates)
        return output_width, output_height

    # Defensive fallback for unusually narrow or wide inputs.
    output_width = max(width, int((ideal_width + 31) // 32) * 32)
    output_height = max(height, int((ideal_height + 31) // 32) * 32)
    return output_width, output_height


def _resize_image_batch(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize the final video on CPU in small chunks so sampling VRAM stays unchanged."""
    if (int(images.shape[2]), int(images.shape[1])) == (int(width), int(height)):
        return images
    import torch.nn.functional as F

    output = torch.empty(
        (int(images.shape[0]), int(height), int(width), 3),
        dtype=torch.float32, device="cpu",
    )
    for start in range(0, int(images.shape[0]), 8):
        source = images[start:start + 8, ..., :3].detach().to("cpu").movedim(-1, 1).float()
        resized = F.interpolate(
            source, size=(int(height), int(width)), mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp_(0.0, 1.0)
        output[start:start + int(resized.shape[0])].copy_(resized.movedim(1, -1))
    return output


def _scale_face_transform(transform: dict, width: int, height: int) -> dict:
    """Map source-space tracking coordinates onto the enlarged output canvas."""
    source_width, source_height = transform["src_size"]
    scale_x = int(width) / max(float(source_width), 1.0)
    scale_y = int(height) / max(float(source_height), 1.0)
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return transform
    scaled = dict(transform)
    scaled["src_size"] = (int(width), int(height))
    scaled["boxes"] = [
        (float(x) * scale_x, float(y) * scale_y,
         float(box_width) * scale_x, float(box_height) * scale_y)
        for x, y, box_width, box_height in transform["boxes"]
    ]
    return scaled


def _send_face_repair_resolution(node_id, width: int, height: int) -> None:
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_sync("star7-h3-face-repair-resolution", {
            "node_id": node_id,
            "width": int(width),
            "height": int(height),
            "megapixels": round(int(width) * int(height) / _MEGAPIXEL, 2),
        })
    except Exception as exc:
        _LOG.debug("Unable to update the Star7 face-repair resolution widget: %s", exc)


def _build_multiface_picks(
    tracking_cache: dict,
    maximum: int,
    min_visible_frames: int = 5,
    selection: str = "largest_face",
    reserved_indices=None,
):
    """Build exclusive shot-local tracks from one shared face-detection pass."""
    all_boxes = tracking_cache["boxes"]
    all_confs = tracking_cache["confs"]
    segments = tracking_cache["segments"]
    width, height = tracking_cache["src_size"]
    frame_count = int(tracking_cache["frames"])
    maximum = max(1, int(maximum))
    tracks = [[-1] * frame_count for _ in range(maximum)]
    if reserved_indices is not None:
        reserved_indices = [int(value) for value in reserved_indices]
        if len(reserved_indices) != frame_count:
            raise ValueError("reserved face track does not match the detected frame count")

    for start, stop in segments:
        start, stop = int(start), int(stop)
        reserve_this_shot = bool(
            reserved_indices is not None
            and any(reserved_indices[frame] >= 0 for frame in range(start, stop))
        )
        first_dynamic_lane = 1 if reserve_this_shot else 0
        if reserve_this_shot:
            for frame in range(start, stop):
                box = reserved_indices[frame]
                if 0 <= box < len(all_boxes[frame]):
                    tracks[0][frame] = box
        last = [None] * maximum
        started = [False] * maximum
        for frame in range(start, stop):
            boxes = all_boxes[frame]
            if not boxes:
                continue
            available_boxes = set(range(len(boxes)))
            if reserve_this_shot and tracks[0][frame] >= 0:
                available_boxes.discard(tracks[0][frame])
            active_lanes = [
                lane for lane in range(first_dynamic_lane, maximum)
                if started[lane] and last[lane] is not None
            ]
            # Global greedy matching keeps assignments exclusive. With the public
            # cap of four faces this is deterministic and avoids a SciPy dependency.
            candidates = sorted(
                (_continuity_cost(boxes[box], last[lane]), lane, box)
                for lane in active_lanes for box in available_boxes
            )
            assigned_lanes = set()
            for _cost, lane, box in candidates:
                if lane in assigned_lanes or box not in available_boxes:
                    continue
                tracks[lane][frame] = box
                q = boxes[box]
                last[lane] = ((q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0, q[3] - q[1])
                assigned_lanes.add(lane)
                available_boxes.remove(box)
            if available_boxes:
                ranked = _rank_boxes(
                    boxes, all_confs[frame], int(width), int(height), selection
                )
                for box in ranked:
                    if box not in available_boxes:
                        continue
                    lane = next(
                        (idx for idx in range(first_dynamic_lane, maximum) if not started[idx]),
                        None,
                    )
                    if lane is None:
                        break
                    tracks[lane][frame] = box
                    q = boxes[box]
                    last[lane] = ((q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0, q[3] - q[1])
                    started[lane] = True
                    available_boxes.remove(box)
            for lane in assigned_lanes:
                started[lane] = True

        # A one-frame false detection should not trigger an entire H3 sampling pass.
        for lane in range(maximum):
            visible = sum(tracks[lane][frame] >= 0 for frame in range(start, stop))
            if visible < int(min_visible_frames):
                for frame in range(start, stop):
                    tracks[lane][frame] = -1

    results = []
    for lane, indices in enumerate(tracks):
        if sum(index >= 0 for index in indices) < int(min_visible_frames):
            continue
        picks = []
        for start, stop in segments:
            lock = next((frame for frame in range(int(start), int(stop)) if indices[frame] >= 0), None)
            if lock is None:
                picks.append({"segment": [int(start), int(stop)], "absent": True})
            else:
                picks.append({
                    "segment": [int(start), int(stop)], "frame": int(lock),
                    "box": int(indices[lock]), "absent": False,
                })
        results.append({
            "frames": frame_count,
            "src_size": (int(width), int(height)),
            "boxes": all_boxes,
            "confs": all_confs,
            "segments": [(int(a), int(b)) for a, b in segments],
            "picks": picks,
            "track_indices": [int(index) for index in indices],
            "detector": tracking_cache.get("detector", "shared Star7 detection"),
            "confidence": float(tracking_cache.get("confidence", 0.35)),
        })
    return results


def _ensure_face_detector() -> str:
    """Use an installed face detector, or atomically fetch the small default model."""
    names = _detector_list()
    face_names = [name for name in names if "face" in name.lower()]
    if face_names and face_names[0] != _FACE_DETECTOR_NAME:
        return face_names[0]

    target_dir = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
    target = os.path.join(target_dir, _FACE_DETECTOR_NAME)
    legacy_target = os.path.join(folder_paths.models_dir, "ultralytics", _FACE_DETECTOR_NAME)
    if os.path.isfile(legacy_target) and os.path.getsize(legacy_target) > 1_000_000:
        return _FACE_DETECTOR_NAME
    if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
        return _FACE_DETECTOR_NAME

    with _FACE_DETECTOR_LOCK:
        if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
            return _FACE_DETECTOR_NAME
        os.makedirs(target_dir, exist_ok=True)
        failures = []
        for url in _FACE_DETECTOR_URLS:
            partial = f"{target}.part-{os.getpid()}"
            try:
                _LOG.info("Star7 H3 face repair downloading detector | source=%s", url)
                request = urllib.request.Request(url, headers={"User-Agent": "Star7-ComfyUI/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response, open(partial, "wb") as output:
                    digest = hashlib.sha256()
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest().lower() != _FACE_DETECTOR_SHA256:
                    raise RuntimeError("SHA-256 mismatch")
                os.replace(partial, target)
                _LOG.info("Star7 H3 face detector ready | %s", target)
                return _FACE_DETECTOR_NAME
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                try:
                    if os.path.exists(partial):
                        os.remove(partial)
                except OSError:
                    pass
        raise FileNotFoundError(
            "Star7 H3 Face Repair requires face_yolov8m.pt. Automatic download failed; "
            f"place it in '{target_dir}'. Sources: {' | '.join(failures)}"
        )


class MiniMaxH3MaterialPromptStar7(io.ComfyNode):
    """Unified H3 conditioning plus a compact context for the later face-refine pass."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaterialPromptStar7",
            display_name="MiniMax H3 All-in-one Conditioning - Star7",
            category="Star7/MiniMax H3",
            description="Unified H3 text/image/video/audio conditioning with one-line Star7 face-refine context reuse.",
            accept_all_inputs=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=243, min=5, max=3600, step=17),
                io.Combo.Input("task_type", options=_TASK_OPTIONS, default="自动判断 / Auto"),
                io.Combo.Input("audio_mode", options=_AUDIO_OPTIONS, default="锁定原音 / Lock Source"),
                io.Float.Input("audio_denoise_strength", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("reference_quality", options=_REFERENCE_SIZE_OPTIONS, default="匹配生成画布 / Match"),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    tooltip="Reference images. Connecting one slot automatically reveals the next.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=_MAX_REFERENCE_IMAGES,
                    ),
                ),
                io.Image.Input("ref_video_0", optional=True),
                io.Image.Input("ref_video_1", optional=True),
                io.Image.Input("ref_video_2", optional=True),
                io.Audio.Input("ref_video_audio_0", optional=True),
                io.Audio.Input("ref_video_audio_1", optional=True),
                io.Audio.Input("ref_video_audio_2", optional=True),
                io.Audio.Input("ref_audio_0", optional=True),
                io.Audio.Input("ref_audio_1", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(display_name="mux_audio"),
                _CONTEXT_IO.Output(display_name="refine_context"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls, model, clip, video_vae, audio_vae, prompt, width, height, length,
        task_type="自动判断 / Auto", audio_mode="锁定原音 / Lock Source",
        audio_denoise_strength=0.35, reference_quality="匹配生成画布 / Match",
        drive_audio=None, final_audio=None, first_frame=None, last_frame=None,
        ref_images=None,
        ref_video_0=None, ref_video_1=None, ref_video_2=None,
        ref_video_audio_0=None, ref_video_audio_1=None, ref_video_audio_2=None,
        ref_audio_0=None, ref_audio_1=None,
        **legacy_inputs,
    ):
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3AddGuide,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            _encode_ref_audio,
        )

        warnings = []
        width, height = int(width), int(height)
        aligned_width, aligned_height = _round_canvas(width), _round_canvas(height)
        if (aligned_width, aligned_height) != (width, height):
            warnings.append(
                f"canvas {width}x{height} was adjusted to {aligned_width}x{aligned_height} "
                "for the required 32-pixel grid"
            )
        width, height = aligned_width, aligned_height
        mode = _option_id(audio_mode, _AUDIO_IDS, "audio mode")
        requested_task = _option_id(task_type, _TASK_IDS, "task type")
        ref_size = _reference_size_id(reference_quality)
        if not 0.0 <= float(audio_denoise_strength) <= 1.0:
            raise ValueError("Audio denoise strength must be between 0 and 1")
        if mode != "native" and drive_audio is None:
            warnings.append(f"audio mode {mode} has no driving audio; using native audio generation")
            mode = "native"

        ref_images = _collect_reference_images(ref_images, legacy_inputs)
        ref_videos = {f"ref_video_{i}": item for i, item in enumerate(
                      (ref_video_0, ref_video_1, ref_video_2)) if item is not None}
        ref_video_audios = {f"ref_video_audio_{i}": item for i, item in enumerate(
                            (ref_video_audio_0, ref_video_audio_1, ref_video_audio_2)) if item is not None}
        standalone_audios = [item for item in (ref_audio_0, ref_audio_1) if item is not None]

        from comfy_extras.nodes_minimax_h3 import temporal_shape
        target_frames = temporal_shape(int(length))[0]
        ref_videos, ref_video_audios, video_warnings = _prepare_reference_videos(
            ref_videos, ref_video_audios, target_frames
        )
        warnings.extend(video_warnings)

        ordered_audios = []
        if drive_audio is not None and mode in ("lock_source", "remix_source", "reference_only"):
            ordered_audios.append(drive_audio)
        ordered_audios.extend(standalone_audios)
        ref_audios = {f"ref_audio_{i}": item for i, item in enumerate(ordered_audios)}

        has_refs = bool(ref_images or ref_videos or ref_audios)
        has_frames = first_frame is not None or last_frame is not None
        inferred = "T2VA"
        if has_refs and has_frames:
            inferred = "Hybrid"
        elif has_refs:
            inferred = "Ref2VA"
        elif first_frame is not None and last_frame is not None:
            inferred = "FL2VA"
        elif last_frame is not None:
            inferred = "L2VA"
        elif first_frame is not None:
            inferred = "I2VA"
        resolved = requested_task
        if resolved == "auto":
            resolved = inferred
        elif resolved != inferred:
            warnings.append(
                f"selected task {resolved} did not match connected inputs; routed as {inferred}"
            )
            resolved = inferred

        paired_audio_count = len(ref_video_audios)
        has_drive_reference = (
            drive_audio is not None and mode in ("lock_source", "remix_source", "reference_only")
        )
        # Keep normal audio references in their official visible order:
        # paired reference-video soundtracks first, standalone references next.
        # Driving audio is excluded from that public numeric sequence and uses D.
        audio_tag_map = {index: index for index in range(1, paired_audio_count + 1)}
        for standalone_index in range(len(standalone_audios)):
            friendly = paired_audio_count + standalone_index + 1
            actual = friendly + (1 if has_drive_reference else 0)
            audio_tag_map[friendly] = actual
        audio_alias_map = {}
        audio_tag_labels = []
        if has_drive_reference:
            audio_alias_map["D"] = paired_audio_count + 1
            audio_tag_labels.append("<Audio D>=driving audio")
        video_audio_slots = sorted(
            int(key.rsplit("_", 1)[-1]) + 1 for key in ref_video_audios
        )
        for friendly_ordinal, video_slot in enumerate(video_audio_slots, 1):
            audio_tag_labels.append(
                f"<Audio {friendly_ordinal}>=Reference Video {video_slot} soundtrack"
            )
        for standalone_index in range(len(standalone_audios)):
            friendly_ordinal = paired_audio_count + standalone_index + 1
            audio_tag_labels.append(
                f"<Audio {friendly_ordinal}>=standalone Reference Audio {standalone_index + 1}"
            )
        conditioned_prompt, prompt_warnings = _prepare_prompt_tags(
            str(prompt), len(ref_images), len(ref_videos),
            paired_audio_count + len(ref_audios), 0,
            audio_tag_map or None, audio_alias_map,
        )
        warnings.extend(prompt_warnings)

        if has_refs:
            positive, latent = _unpack(_execute_reference_to_video(
                MiniMaxH3ReferenceToVideo, clip, video_vae, audio_vae,
                conditioned_prompt, width, height, int(length), ref_size,
                ref_images, ref_videos, ref_video_audios, ref_audios,
            ))[:2]
            if first_frame is not None:
                positive = _unpack(MiniMaxH3AddGuide.execute(
                    positive, latent, 0, vae=video_vae, image=first_frame
                ))[0]
            if last_frame is not None:
                positive = _unpack(MiniMaxH3AddGuide.execute(
                    positive, latent, -1, vae=video_vae, image=last_frame
                ))[0]
        else:
            positive, latent = _unpack(MiniMaxH3ImageToVideo.execute(
                clip, video_vae, conditioned_prompt, width, height, int(length),
                first_frame=first_frame, last_frame=last_frame,
            ))[:2]

        if drive_audio is not None and mode in ("lock_source", "remix_source"):
            encoded, _ = _encode_ref_audio(audio_vae, drive_audio)
            latent = _replace_audio_latent(
                latent, encoded, 0.0 if mode == "lock_source" else float(audio_denoise_strength)
            )

        context = {
            "version": 2,
            "model": model,
            "positive": _copy_conditioning_without_keyframes(positive),
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "prompt": conditioned_prompt,
            "task_type": resolved,
            "identity_reference": ref_images.get("ref_image_0"),
            # Keep the original endpoint pixels outside CONDITIONING so local face
            # repair remains free of full-frame anchors, while the HD node can
            # re-encode them at its actual target resolution instead of injecting
            # stale low-resolution keyframe latents.
            "first_frame": first_frame,
            "last_frame": last_frame,
        }
        mux_audio = final_audio if final_audio is not None else (
            drive_audio if mode == "lock_source" else None
        )
        report = (
            f"Star7 H3 conditioning ready | task={resolved} | "
            f"audio={mode} | canvas={width}x{height} | "
            f"images={len(ref_images)} videos={len(ref_videos)} audios={len(ref_audios)} | "
            f"refine context attached"
        )
        if audio_tag_labels:
            report += "\nPrompt audio tags: " + "; ".join(audio_tag_labels)
        if warnings:
            report += "\n" + "\n".join(f"Warning: {item}" for item in warnings)
        _LOG.info(report.splitlines()[0])
        for warning in warnings:
            _LOG.warning("Star7 H3 conditioning | %s", warning)
        return io.NodeOutput(model, positive, latent, mux_audio, context, report)


_PRESETS = {
    "自动平衡": dict(denoise=0.30, small=1.0, large=0.30, crop=2.6, canvas="auto", blend=0.90, feather=20),
    "真人保真": dict(denoise=0.25, small=0.85, large=0.20, crop=2.8, canvas=512, blend=0.82, feather=24),
    "远景小脸": dict(denoise=0.48, small=1.0, large=0.35, crop=2.4, canvas=768, blend=0.95, feather=18),
    "动漫角色": dict(denoise=0.32, small=0.95, large=0.25, crop=2.7, canvas=512, blend=0.88, feather=20),
}


class MiniMaxH3FaceRefineStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_av_latent": ("LATENT",),
                "refine_context": (_CONTEXT_TYPE,),
                "enable_refine": ("BOOLEAN", {"default": True}),
                "face_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "preset": (["自动平衡", "真人保真", "远景小脸", "动漫角色", "自定义"], {"default": "自动平衡"}),
                "target_face": (["主人物", "画面中央", "参考图匹配"], {"default": "主人物"}),
                "refine_steps": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "custom_strength": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.80, "step": 0.01}),
                "custom_canvas": (["自动", "512", "768"], {"default": "自动"}),
                "custom_crop_context": ("FLOAT", {"default": 2.6, "min": 1.8, "max": 4.0, "step": 0.1}),
                "custom_blend": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_feather": ("INT", {"default": 20, "min": 0, "max": 64, "step": 2}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preserve_repair_detail": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("refined_images",)
    FUNCTION = "refine"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Decode, track, crop, H3-refine and GPU-stitch one face or up to four "
        "stable shot-local face tracks in one node."
    )

    def refine(
        self, sampled_av_latent, refine_context, enable_refine=True, face_count=1,
        preset="自动平衡", target_face="主人物", refine_steps=4, custom_strength=0.30, custom_canvas="自动",
        custom_crop_context=2.6, custom_blend=0.90, custom_feather=20, seed=0,
        preserve_repair_detail=True, unique_id=None,
        **legacy_options,
    ):
        context = dict(refine_context or {})
        vae = context.get("video_vae")
        model = context.get("model")
        positive = context.get("positive")
        if vae is None or model is None or positive is None:
            raise ValueError("Star7 face-repair context is incomplete. Use MiniMax H3 All-in-one Conditioning - Star7.")

        samples = sampled_av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("Star7 H3 Face Refine requires a sampled MiniMax H3 packed audio-video LATENT.")
        members = list(samples.unbind())
        if len(members) < 2 or members[0].ndim != 5 or members[0].shape[1] != 24:
            raise ValueError("Invalid H3 latent: expected video [B,24,T,H,W] plus audio latent.")

        refine_started = time.perf_counter()
        _LOG.debug("Star7 H3 face repair | phase=decode source video")
        decode_started = time.perf_counter()
        base_images = _decode_video_frames(vae, members[0])
        _LOG.debug(
            "Star7 H3 face repair | phase=decode source video completed | %.2fs | frames=%d %dx%d",
            time.perf_counter() - decode_started, int(base_images.shape[0]),
            int(base_images.shape[2]), int(base_images.shape[1]),
        )
        if not enable_refine:
            _send_face_repair_resolution(unique_id, int(base_images.shape[2]), int(base_images.shape[1]))
            return (base_images,)

        cfg = dict(_PRESETS.get(preset, {}))
        if preset == "自定义":
            cfg = dict(
                denoise=float(custom_strength), small=1.0, large=0.30,
                crop=float(custom_crop_context), canvas=custom_canvas,
                blend=float(custom_blend), feather=int(custom_feather),
            )

        canvas = cfg["canvas"]
        canvas_mode = "auto_capped_768" if canvas in ("auto", "自动") else "manual"
        canvas_size = 768 if canvas_mode != "manual" else int(canvas)
        requested_faces = max(1, min(4, int(face_count)))
        if legacy_options.get("multi_face") and requested_faces == 1:
            requested_faces = max(2, min(4, int(legacy_options.get("max_faces", 3))))
        multi_face = requested_faces > 1
        select = "centre_most" if target_face == "画面中央" else "largest_face"
        identity_reference = context.get("identity_reference") if target_face == "参考图匹配" else None
        identity_track = identity_reference is not None
        if target_face == "参考图匹配" and identity_reference is None:
            _LOG.warning("Reference-face matching requested without a reference; using main subject tracking")
            select = "largest_face"
        elif identity_track and importlib.util.find_spec("insightface") is None:
            _LOG.warning(
                "Reference-face matching is unavailable because insightface is not installed; "
                "using main subject tracking"
            )
            identity_reference = None
            identity_track = False
            select = "largest_face"

        detector = _ensure_face_detector()
        _LOG.debug("Star7 H3 face repair | phase=detect/track | preset=%s", preset)
        detect_started = time.perf_counter()
        try:
            tracked = H3FaceTrackCrop().run(
                base_images, detector, 0.35, float(cfg["crop"]), canvas_size, canvas_size,
                canvas_mode, 21, 51, "gaussian", "per_frame", select=select,
                identity_reference=identity_reference, identity_track=identity_track,
                cut_detection="auto (pyscenedetect)", cut_threshold=3.0,
                absent_shots="by_identity" if identity_track else "off",
                verbose=False,
            )
        except ValueError as exc:
            if "no face detected" not in str(exc).lower():
                raise
            _LOG.warning("Star7 H3 face repair found no face; returning original frames | %s", exc)
            _send_face_repair_resolution(unique_id, int(base_images.shape[2]), int(base_images.shape[1]))
            return (base_images,)
        crops, transform, _preview, report, canvas_w, canvas_h, _frame_count = tracked
        _LOG.debug(
            "Star7 H3 face repair | phase=detect/track completed | %.2fs | frames=%d target=%s",
            time.perf_counter() - detect_started, int(_frame_count),
            f"multi-face(requested={requested_faces})" if multi_face else target_face,
        )
        if not transform.get("boxes"):
            _LOG.warning("Star7 H3 face repair found no usable face; returning original frames")
            _send_face_repair_resolution(unique_id, int(base_images.shape[2]), int(base_images.shape[1]))
            return (base_images,)

        source_height, source_width = int(base_images.shape[1]), int(base_images.shape[2])
        output_width, output_height = _face_repair_output_size(
            source_width, source_height, bool(preserve_repair_detail)
        )
        images = _resize_image_batch(base_images, output_width, output_height)
        output_scale = ((output_width / source_width) + (output_height / source_height)) * 0.5
        if (output_width, output_height) != (source_width, source_height):
            _LOG.debug(
                "Star7 H3 face repair | detail-preserving output %dx%d (%.2f MP, %.2fx) from %dx%d",
                output_width, output_height, output_width * output_height / _MEGAPIXEL,
                output_scale, source_width, source_height,
            )
        else:
            _LOG.debug(
                "Star7 H3 face repair | output remains %dx%d (%.2f MP); never downscaled",
                output_width, output_height, output_width * output_height / _MEGAPIXEL,
            )

        from comfy_extras.nodes_custom_sampler import (
            BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
        )

        def sample_track(track_result, composite_images, pass_index):
            track_crops, track_transform, _track_preview, track_report, track_w, track_h, _ = track_result
            stitch_transform = _scale_face_transform(track_transform, output_width, output_height)
            target_video = torch.zeros(
                (members[0].shape[0], 24, members[0].shape[2], int(track_h) // 16, int(track_w) // 16),
                device=members[0].device, dtype=members[0].dtype,
            )
            refine_latent = dict(sampled_av_latent)
            refine_latent.pop("noise_mask", None)
            refine_latent["samples"] = comfy.nested_tensor.NestedTensor(
                (target_video, members[1].clone())
            )
            _LOG.debug("Star7 H3 face repair | phase=encode repair crops | face=%d", int(pass_index) + 1)
            encode_started = time.perf_counter()
            refine_latent = H3InjectVideoLatent().run(refine_latent, track_crops, vae)[0]
            _LOG.debug(
                "Star7 H3 face repair | phase=encode repair crops completed | face=%d | %.2fs",
                int(pass_index) + 1, time.perf_counter() - encode_started,
            )
            refine_latent, _mask_report, refine_model = H3PerFrameDenoise().run(
                model, refine_latent, track_transform, float(cfg["small"]), float(cfg["large"]),
                30.0, 120.0, 1.0, 9, "absolute_px", verbose=False,
            )
            sigmas = _unpack(BasicScheduler.execute(
                refine_model, "simple", int(refine_steps), float(cfg["denoise"])
            ))[0]
            guider = _unpack(BasicGuider.execute(refine_model, positive))[0]
            sampler = _unpack(KSamplerSelect.execute("res_multistep"))[0]
            pass_seed = (int(seed) + int(pass_index)) & 0xffffffffffffffff
            noise = _unpack(RandomNoise.execute(pass_seed))[0]
            _LOG.debug(
                "Star7 H3 face repair | phase=sample | face=%d | canvas=%sx%s denoise=%.2f steps=%d feather=%dpx",
                int(pass_index) + 1, track_w, track_h, cfg["denoise"], int(refine_steps), int(cfg["feather"]),
            )
            sample_started = time.perf_counter()
            sampled = _unpack(SamplerCustomAdvanced.execute(
                noise, guider, sampler, sigmas, refine_latent
            ))[0]
            _LOG.debug(
                "Star7 H3 face repair | phase=sample completed | face=%d | %.2fs | steps=%d",
                int(pass_index) + 1, time.perf_counter() - sample_started, int(refine_steps),
            )
            _LOG.debug("Star7 H3 face repair | phase=decode/stitch | face=%d", int(pass_index) + 1)
            composite_started = time.perf_counter()
            refined_video = list(sampled["samples"].unbind())[0]
            refined_crops = _decode_video_frames(vae, refined_video)
            stitched = H3FaceStitch().run(
                composite_images, refined_crops, stitch_transform, "face_only", 16,
                max(1, int(round(float(cfg["feather"]) * output_scale))),
                1.0, float(cfg["blend"]), "fade_out",
            )[0]
            _LOG.debug(
                "Star7 H3 face repair | phase=decode/stitch completed | face=%d | %.2fs",
                int(pass_index) + 1, time.perf_counter() - composite_started,
            )
            return stitched, track_report

        completed_faces = 0
        if multi_face:
            tracking_cache = transform.get("tracking_cache") or {}
            face_picks = _build_multiface_picks(
                tracking_cache,
                requested_faces,
                selection=select,
                reserved_indices=(
                    tracking_cache.get("selected_indices") if identity_track else None
                ),
            )
            detected_faces = len(face_picks)
            if detected_faces < requested_faces:
                _LOG.warning(
                    "Star7 H3 face repair | requested=%d faces, stable-detected=%d; "
                    "automatically using %d",
                    requested_faces, detected_faces, detected_faces,
                )
            else:
                _LOG.debug(
                    "Star7 H3 face repair | requested=%d faces, stable-detected=%d",
                    requested_faces, detected_faces,
                )
            for lane, face_pick in enumerate(face_picks):
                lane_tracked = H3FaceTrackCrop().run(
                    base_images, detector, 0.35, float(cfg["crop"]), canvas_size, canvas_size,
                    canvas_mode, 21, 51, "gaussian", "per_frame", select="largest_face",
                    identity_reference=None, identity_track=False,
                    cut_detection="auto (pyscenedetect)", cut_threshold=3.0,
                    absent_shots="off", face_pick=face_pick, verbose=False,
                )
                if not lane_tracked[1].get("boxes"):
                    continue
                images, _lane_report = sample_track(lane_tracked, images, lane)
                completed_faces += 1
            if not completed_faces:
                _LOG.warning("Star7 H3 multi-face repair found no stable 5-frame face track; returning original frames")
                _send_face_repair_resolution(unique_id, source_width, source_height)
                return (base_images,)
            report = f"multi-face tracks={completed_faces}"
            _LOG.debug(
                "Star7 H3 multi-face repair | completed=%d requested=%d | detection reused",
                completed_faces, requested_faces,
            )
        else:
            images, report = sample_track(tracked, images, 0)
            completed_faces = 1
        _LOG.info(
            "Star7 H3 face repair completed | %.1fs | faces=%d | output=%dx%d",
            time.perf_counter() - refine_started, completed_faces, output_width, output_height,
        )
        _send_face_repair_resolution(unique_id, output_width, output_height)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MaterialPromptStar7": MiniMaxH3MaterialPromptStar7,
    "MiniMaxH3FaceRefineStar7": MiniMaxH3FaceRefineStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MaterialPromptStar7": "MiniMax H3 All-in-one Conditioning - Star7",
    "MiniMaxH3FaceRefineStar7": "MiniMax H3 One-click Face Repair - Star7",
}
