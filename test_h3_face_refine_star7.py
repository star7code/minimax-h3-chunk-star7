import torch

import comfy.nested_tensor
from comfy_api.latest._io import build_nested_inputs, get_finalized_class_inputs

from .h3_face_refine_star7 import (
    H3FaceStitch,
    MiniMaxH3FaceRefineStar7,
    MiniMaxH3MaterialPromptStar7,
    _aligned_reference_frames,
    _build_multiface_picks,
    _collect_reference_images,
    _copy_conditioning_without_keyframes,
    _decode_video_frames,
    _execute_reference_to_video,
    _face_repair_output_size,
    _fit_audio_latent,
    _option_id,
    _prepare_prompt_tags,
    _prepare_reference_videos,
    _scale_face_transform,
    _TASK_IDS,
)


def test_conditioning_copy_keeps_refs_and_removes_full_frame_keyframes():
    tensor = torch.zeros(1)
    source = [[tensor, {"minimax_refs": ["identity"], "minimax_keyframes": ["full-frame"]}]]
    result = _copy_conditioning_without_keyframes(source)
    assert result[0][0] is tensor
    assert result[0][1]["minimax_refs"] == ["identity"]
    assert "minimax_keyframes" not in result[0][1]
    assert "minimax_keyframes" in source[0][1]


def test_audio_latent_fit_does_not_mutate_source():
    source = torch.ones((1, 32, 2, 3), dtype=torch.float16)
    target = torch.zeros((1, 32, 2, 5), dtype=torch.float16)
    result = _fit_audio_latent(source, target)
    assert result.shape == target.shape
    assert torch.equal(result[..., :3], source)
    assert torch.count_nonzero(result[..., 3:]) == 0
    assert torch.count_nonzero(target) == 0


def test_face_refine_bypass_decodes_only_video_member():
    class VAE:
        def decode(self, latent):
            assert latent.shape == (1, 24, 2, 2, 2)
            return torch.full((5, 8, 8, 3), 0.25)

    av = {
        "samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, 2, 2, 2)),
            torch.zeros((1, 32, 2, 8)),
        ))
    }
    context = {"video_vae": VAE(), "model": object(), "positive": [[torch.zeros(1), {}]]}
    images = MiniMaxH3FaceRefineStar7().refine(
        av, context, enable_refine=False,
    )[0]
    assert images.shape == (5, 8, 8, 3)
    assert torch.all(images == 0.25)


def test_face_refine_flattens_real_h3_vae_batch_time_output():
    class VAE:
        def decode(self, latent):
            return torch.full((1, 5, 8, 8, 3), 0.5)

    result = _decode_video_frames(VAE(), torch.zeros((1, 24, 2, 2, 2)))
    assert result.shape == (5, 8, 8, 3)
    assert torch.all(result == 0.5)


def test_face_repair_output_grows_to_one_mp_without_shrinking():
    assert _face_repair_output_size(864, 480, True) == (1376, 768)
    assert _face_repair_output_size(480, 864, True) == (768, 1376)
    assert _face_repair_output_size(1280, 736, True) == (1344, 768)
    assert _face_repair_output_size(1344, 768, True) == (1344, 768)
    assert _face_repair_output_size(1504, 832, True) == (1504, 832)
    assert _face_repair_output_size(864, 480, False) == (864, 480)


def test_face_transform_scales_with_detail_preserving_output():
    transform = {"src_size": (864, 480), "boxes": [(10.0, 20.0, 100.0, 50.0)]}
    scaled = _scale_face_transform(transform, 1728, 960)
    assert scaled["src_size"] == (1728, 960)
    assert scaled["boxes"] == [(20.0, 40.0, 200.0, 100.0)]
    assert transform["src_size"] == (864, 480)


def test_scaled_face_transform_stitches_into_enlarged_output(monkeypatch):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
    base = torch.zeros((2, 96, 160, 3), dtype=torch.float32)
    crops = torch.ones((2, 64, 64, 3), dtype=torch.float32)
    transform = {
        "src_size": (80, 48), "canvas": (64, 64),
        "boxes": [(20.0, 8.0, 32.0, 32.0)] * 2,
        "source": [0, 1], "weights": [1.0, 1.0], "detected": [True, True],
        "face_rect": [(16.0, 16.0, 32.0, 32.0)] * 2,
    }
    scaled = _scale_face_transform(transform, 160, 96)
    output = H3FaceStitch().run(
        base, crops, scaled, "face_only", 0, 4, 0.0, 0.9, "fade_out"
    )[0]
    assert output.shape == base.shape
    assert torch.isfinite(output).all()
    assert output.max() > 0


def test_reference_video_is_bounded_and_audio_is_cropped_without_mutating_input():
    frames = torch.zeros((500, 16, 16, 3))
    waveform = torch.zeros((1, 2, 1_000_000))
    videos, audios, warnings = _prepare_reference_videos(
        {"ref_video_0": frames},
        {"ref_video_audio_0": {"waveform": waveform, "sample_rate": 48_000}},
        243,
    )
    expected = _aligned_reference_frames(500, 243)
    assert videos["ref_video_0"].shape[0] == expected
    assert audios["ref_video_audio_0"]["waveform"].shape[-1] == round(expected / 24 * 48_000)
    assert frames.shape[0] == 500
    assert any("15s" in warning for warning in warnings)


def test_prompt_media_tags_are_repaired_without_strict_failure():
    prompt, warnings = _prepare_prompt_tags(
        "Use Image 0, <Video 3>, and <Audio 1>.", 1, 1, 2, primary_audio_ordinal=2
    )
    assert "<Picture 1>" in prompt
    assert "<Video 1>" in prompt
    assert "<Audio 2>" in prompt
    assert warnings


def test_reference_images_support_nine_slots_and_compact_gaps():
    images = [object() for _ in range(9)]
    all_images = _collect_reference_images(
        {f"ref_image_{index}": images[index] for index in range(9)},
    )
    assert list(all_images) == [f"ref_image_{index}" for index in range(9)]
    assert list(all_images.values()) == images

    gapped = _collect_reference_images(
        {
            "ref_image_1": images[1],
            "ref_image_3": images[3],
            "ref_image_4": images[4],
            "ref_image_8": images[8],
        },
    )
    assert list(gapped) == [f"ref_image_{index}" for index in range(4)]
    assert list(gapped.values()) == [images[1], images[3], images[4], images[8]]


def test_legacy_reference_image_slots_survive_v3_input_normalization():
    first, fourth = object(), object()
    live_inputs = {"ref_image_0": first, "ref_image_3": fourth}
    _, _, v3_data = get_finalized_class_inputs(
        MiniMaxH3MaterialPromptStar7.INPUT_TYPES(), live_inputs
    )
    nested = build_nested_inputs(live_inputs, v3_data)
    assert MiniMaxH3MaterialPromptStar7.ACCEPT_ALL_INPUTS is True
    assert _collect_reference_images(nested["ref_images"], nested) == {
        "ref_image_0": first,
        "ref_image_1": fourth,
    }


def test_reference_to_video_call_uses_names_supported_by_old_and_new_comfyui():
    values = {
        "clip": object(), "vae": object(), "audio_vae": object(), "prompt": "test",
        "width": 1344, "height": 768, "length": 243, "ref_image_size": "match",
        "ref_images": {"ref_image_0": object()}, "ref_videos": {},
        "ref_video_audios": {}, "ref_audios": {},
    }

    class LegacyReferenceNode:
        @classmethod
        def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                    ref_image_size="match", ref_images=None, ref_videos=None,
                    ref_video_audios=None, ref_audios=None):
            return values | {"signature": "legacy"}

    class CurrentReferenceNode:
        @classmethod
        def execute(cls, clip, prompt, width, height, length, ref_image_size="match",
                    vae=None, audio_vae=None, ref_images=None, ref_videos=None,
                    ref_video_audios=None, ref_audios=None):
            return values | {"signature": "current"}

    for reference_node, signature in (
        (LegacyReferenceNode, "legacy"), (CurrentReferenceNode, "current"),
    ):
        result = _execute_reference_to_video(reference_node, **{
            "clip": values["clip"], "video_vae": values["vae"],
            "audio_vae": values["audio_vae"], "prompt": values["prompt"],
            "width": values["width"], "height": values["height"],
            "length": values["length"], "reference_quality": values["ref_image_size"],
            "ref_images": values["ref_images"], "ref_videos": values["ref_videos"],
            "ref_video_audios": values["ref_video_audios"], "ref_audios": values["ref_audios"],
        })
        assert result["signature"] == signature


def test_prompt_audio_tags_keep_official_order_and_use_drive_alias():
    prompt, warnings = _prepare_prompt_tags(
        "drive=<Audio D>; video=<Audio 1>; standalone=<Audio 2>",
        0, 1, 3, primary_audio_ordinal=2,
        audio_tag_map={1: 1, 2: 3}, audio_alias_map={"D": 2},
    )
    assert prompt == "drive=<Audio 2>; video=<Audio 1>; standalone=<Audio 3>"
    assert warnings == []


def test_disconnected_audio_alias_is_safe_plain_text():
    prompt, warnings = _prepare_prompt_tags("Use <Audio D>.", 0, 0, 0)
    assert prompt == "Use Audio D."
    assert warnings == ["unconnected <Audio D> was treated as plain text"]


def test_bilingual_task_option_keeps_stable_id():
    assert _option_id("自动判断 / Auto", _TASK_IDS, "task") == "auto"
    assert _option_id("参考素材生视频 / Ref2VA", _TASK_IDS, "task") == "Ref2VA"


def test_multiface_pick_uses_distinct_shot_local_rank_and_marks_absent_shots():
    cache = {
        "frames": 10,
        "src_size": (100, 100),
        "boxes": [
            [[0, 0, 40, 40], [60, 0, 80, 20]] for _ in range(5)
        ] + [
            [[10, 10, 50, 50]] for _ in range(5)
        ],
        "confs": [[0.9, 0.8] for _ in range(5)] + [[0.9] for _ in range(5)],
        "segments": [(0, 5), (5, 10)],
        "detector": "face.pt",
        "confidence": 0.35,
    }
    picks = _build_multiface_picks(cache, 2)
    pick = picks[1]
    assert pick["picks"][0]["box"] == 1
    assert pick["picks"][0]["absent"] is False
    assert pick["picks"][1]["absent"] is True
    for frame in range(5):
        used = [item["track_indices"][frame] for item in picks]
        assert len(used) == len(set(used))


def test_requested_two_faces_downgrades_to_one_stable_track():
    cache = {
        "frames": 8,
        "src_size": (100, 100),
        "boxes": [[[10, 10, 50, 50]] for _ in range(8)],
        "confs": [[0.9] for _ in range(8)],
        "segments": [(0, 8)],
    }
    picks = _build_multiface_picks(cache, 2)
    assert len(picks) == 1
    assert all(index == 0 for index in picks[0]["track_indices"])


def test_multiface_selection_can_prioritize_central_faces():
    cache = {
        "frames": 5,
        "src_size": (100, 100),
        "boxes": [[[0, 0, 50, 50], [42, 38, 62, 58]] for _ in range(5)],
        "confs": [[0.9, 0.9] for _ in range(5)],
        "segments": [(0, 5)],
    }
    picks = _build_multiface_picks(cache, 1, selection="centre_most")
    assert len(picks) == 1
    assert all(index == 1 for index in picks[0]["track_indices"])


def test_reference_matched_track_is_reserved_before_multiface_fill():
    cache = {
        "frames": 5,
        "src_size": (100, 100),
        "boxes": [[[0, 0, 50, 50], [60, 10, 80, 30]] for _ in range(5)],
        "confs": [[0.9, 0.9] for _ in range(5)],
        "segments": [(0, 5)],
    }
    picks = _build_multiface_picks(
        cache, 2, selection="largest_face", reserved_indices=[1] * 5
    )
    assert len(picks) == 2
    assert all(index == 1 for index in picks[0]["track_indices"])
    assert all(index == 0 for index in picks[1]["track_indices"])


def test_public_node_contract_is_two_wire_face_refine():
    material = MiniMaxH3MaterialPromptStar7.INPUT_TYPES()
    face = MiniMaxH3FaceRefineStar7.INPUT_TYPES()
    assert material["required"]["model"][0] == "MODEL"
    autogrow_type, autogrow_options = material["optional"]["ref_images"]
    assert autogrow_type == "COMFY_AUTOGROW_V3"
    assert autogrow_options["template"]["prefix"] == "ref_image_"
    assert autogrow_options["template"]["min"] == 0
    assert autogrow_options["template"]["max"] == 9
    assert MiniMaxH3MaterialPromptStar7.RETURN_NAMES[4] == "refine_context"
    assert face["required"]["sampled_av_latent"] == ("LATENT",)
    assert face["required"]["refine_context"] == ("STAR7_H3_REFINE_CONTEXT",)
    names = list(face["required"])
    assert names.index("face_count") == names.index("enable_refine") + 1
    assert face["required"]["face_count"][1]["default"] == 1
    assert face["required"]["face_count"][1]["max"] == 4
    assert face["required"]["target_face"][0] == ["主人物", "画面中央", "参考图匹配"]
    assert face["required"]["preserve_repair_detail"][1]["default"] is True
    assert face["hidden"]["unique_id"] == "UNIQUE_ID"
