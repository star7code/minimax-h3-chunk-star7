import torch

import comfy.nested_tensor

from .h3_stream_decode_star7 import (
    MiniMaxH3ChunkedDecodeStar7,
    _av_parts,
    _frames_are_finite,
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
