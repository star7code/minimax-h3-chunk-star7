import torch
import pytest

import comfy.nested_tensor

from .h3_stream_decode_star7 import (
    MiniMaxH3ChunkedDecodeStar7,
    _av_parts,
    _frames_are_finite,
    _stitch_deferred_faces,
)


def test_decode_node_has_independent_av_contract():
    schema = MiniMaxH3ChunkedDecodeStar7.INPUT_TYPES()["required"]
    assert list(schema) == ["av_latent", "video_vae", "audio_vae"]
    assert MiniMaxH3ChunkedDecodeStar7.RETURN_TYPES == ("IMAGE", "AUDIO", "STRING")


def test_av_parts_accepts_exact_h3_nested_pair():
    video = torch.zeros((1, 24, 2, 4, 6))
    audio = torch.zeros((1, 32, 2, 8))
    parts = _av_parts({"samples": comfy.nested_tensor.NestedTensor((video, audio))})
    assert parts[0] is video
    assert parts[1] is audio


def test_av_parts_rejects_plain_latent():
    try:
        _av_parts({"samples": torch.zeros((1, 4, 8, 8))})
    except RuntimeError as exc:
        assert "nested" in str(exc)
    else:
        raise AssertionError("plain latent must not be accepted as an H3 AV latent")


def test_frame_finite_check_is_chunked_and_detects_invalid_values():
    frames = torch.zeros((17, 4, 6, 3))
    assert _frames_are_finite(frames, chunk_size=4)
    frames[9, 1, 2, 0] = float("nan")
    assert not _frames_are_finite(frames, chunk_size=4)


def test_final_decode_stitches_preserved_face_in_rgb_space():
    import comfy.model_management as mm

    class VAE:
        def decode(self, latent):
            return torch.ones((5, 64, 64, 3), dtype=torch.float32)

    overlay = {
        "latent": torch.ones((1, 24, 5, 4, 4), dtype=torch.float32),
        "boxes": [(32.0, 32.0, 64.0, 64.0)] * 5,
        "face_rects": [(8.0, 8.0, 48.0, 48.0)] * 5,
        "weights": [1.0] * 5,
        "canvas": (64, 64),
        "src_size": (128, 128),
        "feather": 4,
        "blend": 1.0,
        "mask_dilation": 4,
        "preserve_repair_detail": False,
    }
    torch.manual_seed(11)
    frames = torch.rand((5, 128, 128, 3), dtype=torch.float32)
    old_device = mm.get_torch_device
    try:
        mm.get_torch_device = lambda: torch.device("cpu")
        output, count = _stitch_deferred_faces(
            frames, {"star7_face_latent_overlays": (overlay,)}, VAE()
        )
    finally:
        mm.get_torch_device = old_device
    assert count == 1
    assert not torch.equal(output[:, 64, 64], frames[:, 64, 64])
    assert torch.equal(output[:, 0, 0], frames[:, 0, 0])


@pytest.mark.parametrize("repair_frames,tracking_frames", [(4, 5), (5, 4)])
def test_deferred_decode_rejects_frame_mismatch_instead_of_shifting_faces(repair_frames, tracking_frames):
    class VAE:
        def decode(self, latent):
            return torch.ones(repair_frames, 32, 32, 3)
    overlay = {"latent": torch.zeros(1), "boxes": [(0, 0, 32, 32)] * tracking_frames}
    with pytest.raises(ValueError, match="timeline mismatch"):
        _stitch_deferred_faces(torch.zeros(5, 32, 32, 3), {"star7_face_latent_overlays": [overlay]}, VAE())


def test_invalid_deferred_data_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Invalid deferred"):
        _stitch_deferred_faces(torch.zeros(5, 32, 32, 3), {"star7_face_latent_overlays": [{}]}, None)


def test_decode_enlarges_canvas_before_face_paste_and_reports_final_size(monkeypatch):
    from . import h3_stream_decode_star7 as decode_module
    from . import h3_face_refine_star7 as face_module
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(face_module, "_FACE_REPAIR_OUTPUT_FLOOR_MP", 192 * 192 / (1024 * 1024))
    base = torch.full((3, 64, 64, 3), .4)
    y, x = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
    repaired = (.5 + .4 * torch.sin(x.float() * 1.2))[None, ..., None].repeat(3, 1, 1, 3)
    video, audio, crop = torch.zeros(1, 24, 1, 4, 4), torch.rand(1, 32, 1, 8), torch.ones(1, 24, 1, 6, 6)
    overlay = {"latent": crop, "boxes": [(24., 24., 16., 16.)] * 3,
               "face_rects": [(8., 8., 80., 80.)] * 3, "weights": [1.] * 3,
               "canvas": (96, 96), "src_size": (64, 64), "feather": 0,
               "mask_dilation": 0, "blend": 1., "colour_match": 0., "preserve_repair_detail": True}
    class VAE:
        def decode(self, latent):
            return base if latent is video else repaired
        def encode(self, frames):
            raise AssertionError("Final compositing must not encode RGB back to latent")
    sentinel = object()
    def decode_audio(vae, latent):
        assert latent["samples"] is audio
        return sentinel
    monkeypatch.setattr(decode_module, "vae_decode_audio", decode_audio)
    av = {"samples": comfy.nested_tensor.NestedTensor((video, audio)), "star7_face_latent_overlays": (overlay,)}
    output, sound, report = MiniMaxH3ChunkedDecodeStar7().decode(av, VAE(), object())
    assert sound is sentinel and output.shape == (3, 192, 192, 3)
    assert "192x192" in report and "face stitches=1" in report
    low = face_module._stitch_face_overlay(base, repaired, dict(overlay, preserve_repair_detail=False))
    enlarged_after_paste = face_module._resize_image_batch(low, 192, 192)
    def crossings(frames):
        profile = frames[0, 96, 88:104, 0] > .5
        return int((profile[1:] != profile[:-1]).sum())
    assert crossings(output) > crossings(enlarged_after_paste)
