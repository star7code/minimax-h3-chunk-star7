from __future__ import annotations

import logging
import time

import torch

from comfy_extras.nodes_audio import vae_decode_audio


_LOG = logging.getLogger("Star7H3Decode")


def _av_parts(av_latent):
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise RuntimeError("Star7 H3 decode requires an H3 audio-video LATENT")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise RuntimeError(
            "Star7 H3 decode expected a nested video/audio latent; a plain image latent is unsupported"
        )
    parts = list(samples.unbind())
    if len(parts) != 2 or parts[0].ndim != 5:
        raise RuntimeError(
            "Star7 H3 decode expected exactly video [B,C,T,H,W] and audio latent streams"
        )
    return parts


def _frames_are_finite(frames, chunk_size: int = 8):
    """Avoid allocating one UHD-sized boolean tensor for the safety check."""
    for start in range(0, int(frames.shape[0]), int(chunk_size)):
        if not torch.isfinite(frames[start:start + chunk_size]).all():
            return False
    return True


def _decode_video_frames(video_vae, latent: torch.Tensor) -> torch.Tensor:
    frames = video_vae.decode(latent)
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    return frames


def _stitch_deferred_faces(frames: torch.Tensor, av_latent: dict,
                           video_vae) -> tuple[torch.Tensor, int]:
    """Decode preserved repair crops and composite them once, in final RGB space."""
    overlays = list(av_latent.get("star7_face_latent_overlays", ()))
    if not overlays:
        return frames, 0

    from .h3_face_refine_star7 import _stitch_face_overlay

    output = frames
    applied = 0
    for overlay in overlays:
        latent = overlay.get("latent")
        if not isinstance(latent, torch.Tensor) or not overlay.get("boxes"):
            raise ValueError("Invalid deferred face repair data; rerun the face repair node")
        repaired = _decode_video_frames(video_vae, latent)
        if repaired.ndim != 4 or not _frames_are_finite(repaired):
            raise RuntimeError("Face VAE decode returned invalid repair crops")
        output = _stitch_face_overlay(output, repaired, overlay)
        applied += 1
    return output, applied


class MiniMaxH3ChunkedDecodeStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "decode"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Decodes a complete MiniMax H3 audio-video latent with the current H3 VAE's native "
        "temporal streaming and spatial tiling, then composites any preserved Star7 face "
        "repairs once in final RGB space. This is independent from HD sampling tiles."
    )

    def decode(self, av_latent, video_vae, audio_vae):
        started = time.perf_counter()
        video, audio = _av_parts(av_latent)
        width, height = int(video.shape[-1]) * 16, int(video.shape[-2]) * 16
        first_stage = getattr(video_vae, "first_stage_model", None)
        is_native_h3 = type(first_stage).__name__ == "MiniMaxH3VideoVAE"
        internal_tiling = bool(getattr(first_stage, "tiling", False)) if is_native_h3 else False
        tile_size = getattr(first_stage, "tile_size", None) if is_native_h3 else None
        route = (
            f"native H3 temporal streaming + spatial tiles ({tile_size}px)"
            if is_native_h3 and internal_tiling
            else "VAE-managed decode"
        )
        estimated_frames = (
            int(first_stage.decode_output_shape(video.shape)[2])
            if is_native_h3 and callable(getattr(first_stage, "decode_output_shape", None))
            else None
        )
        estimated_mib = (
            estimated_frames * width * height * 3 * 4 / (1024 ** 2)
            if estimated_frames is not None else None
        )
        _LOG.info(
            "Star7 H3 decode | %dx%d | route=%s%s",
            width, height, route,
            "" if estimated_mib is None else f" | final frames≈{estimated_mib:.0f} MiB RAM",
        )
        frames = _decode_video_frames(video_vae, video)
        if frames.ndim != 4 or not _frames_are_finite(frames):
            raise RuntimeError(
                f"Star7 H3 video VAE returned invalid frames with shape {tuple(frames.shape)}"
            )
        frames, face_stitches = _stitch_deferred_faces(frames, av_latent, video_vae)
        if face_stitches:
            _LOG.info(
                "Star7 H3 decode | final RGB face stitch completed | faces=%d",
                face_stitches,
            )
        height, width = int(frames.shape[1]), int(frames.shape[2])
        estimated_mib = frames.numel() * frames.element_size() / (1024 ** 2)
        decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
        elapsed = time.perf_counter() - started
        report = (
            f"Star7 H3 decode completed | {width}x{height} frames={int(frames.shape[0])} | "
            f"{route} | face stitches={face_stitches} | {elapsed:.2f}s"
        )
        if estimated_mib is not None:
            report += f" | final IMAGE tensor≈{estimated_mib:.0f} MiB RAM"
        _LOG.info(report)
        return frames, decoded_audio, report


NODE_CLASS_MAPPINGS = {"MiniMaxH3ChunkedDecodeStar7": MiniMaxH3ChunkedDecodeStar7}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChunkedDecodeStar7": "MiniMax H3 分块解码 - Star7"
}
