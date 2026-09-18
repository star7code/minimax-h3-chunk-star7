import torch
import cv2
import numpy as np
import pytest
from types import SimpleNamespace

import comfy.nested_tensor
from comfy_api.latest._io import build_nested_inputs, get_finalized_class_inputs
from .vendor.h3facerefine import core as face_core

from .h3_face_refine_star7 import (
    H3FaceStitch,
    MiniMaxH3FaceRefineLatentStar7,
    MiniMaxH3FaceRefineStar7,
    MiniMaxH3MaterialPromptStar7,
    _aligned_reference_frames,
    _build_multiface_picks,
    _collect_reference_images,
    _copy_conditioning_without_keyframes,
    _decode_video_frames,
    _execute_reference_to_video,
    _face_repair_output_size,
    _face_detector_choices,
    _fit_audio_latent,
    _option_id,
    _prepare_prompt_tags,
    _prepare_reference_videos,
    _scale_face_transform,
    _PRESETS,
    _TASK_IDS,
)
from .vendor.h3facerefine.core import (
    H3FaceTrackCrop,
    H3PerFrameDenoise,
)


def test_face_mask_without_crop_edge_contact_is_pixel_identical():
    original = torch.zeros(1, 1, 128, 128)
    original[..., 40:88, 40:88] = 1
    expected = face_core._gaussian_blur_mask(original, 12).clamp(0, 1)
    actual = face_core._face_region_mask(
        128, 128, (44, 44, 40, 40), 4, 12, "rect", "cpu", torch.float32,
    )
    assert torch.equal(actual, expected)


def test_clipped_face_mask_fades_at_all_crop_edges_and_keeps_core():
    mask = face_core._face_region_mask(
        128, 128, (-20, -20, 168, 168), 16, 12, "rect", "cpu", torch.float32,
    )[0, 0]
    assert torch.count_nonzero(mask[0]) == 0
    assert torch.count_nonzero(mask[-1]) == 0
    assert torch.count_nonzero(mask[:, 0]) == 0
    assert torch.count_nonzero(mask[:, -1]) == 0
    assert torch.all(mask[12:-12, 12:-12] == 1)
    assert torch.all(torch.diff(mask[:13, 64]) >= 0)
    assert torch.diff(mask[:13, 64]).max() < 0.14


def test_closeup_stitch_has_no_hard_crop_seam(monkeypatch):
    monkeypatch.setattr(face_core.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    base = torch.full((1, 128, 128, 3), 0.25)
    refined = torch.full((1, 64, 64, 3), 0.75)
    transform = {
        "boxes": [(32, 32, 64, 64)], "canvas": (64, 64),
        "src_size": (128, 128), "face_rect": [(-10, -10, 84, 84)],
    }
    result = H3FaceStitch().run(base, refined, transform, "face_only", 16, 12, 0.0, 1.0)[0]
    assert torch.equal(result[:, :32], base[:, :32])
    assert torch.equal(result[:, 96:], base[:, 96:])
    assert torch.equal(result[:, 32, 64], base[:, 32, 64])
    assert torch.all(result[:, 64, 64] == 0.75)
    assert torch.diff(result[0, :, 64, 0]).abs().max() < 0.07


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


def test_latent_face_refine_bypass_is_an_exact_passthrough_without_vae_work():
    class VAE:
        def decode(self, _latent):
            raise AssertionError("latent bypass must not decode")

        def encode(self, _images):
            raise AssertionError("latent bypass must not encode")

    av = {
        "samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, 2, 2, 3)),
            torch.zeros((1, 32, 2, 8)),
        ))
    }
    context = {"video_vae": VAE(), "model": object(), "positive": [[torch.zeros(1), {}]]}
    output, report = MiniMaxH3FaceRefineLatentStar7().refine(
        av, context, enable_refine=False,
    )
    assert output is av
    assert "latent unchanged" in report


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


def test_face_crop_expands_and_contracts_with_subject_distance(monkeypatch):
    class Boxes:
        def __init__(self, box):
            self.xyxy = torch.tensor([box], dtype=torch.float32)
            self.conf = torch.tensor([0.95], dtype=torch.float32)

        def __len__(self):
            return 1

    class Detector:
        def __init__(self, heights):
            self.heights = iter(heights)

        def predict(self, _image, **_kwargs):
            height = float(next(self.heights))
            width = height * 0.8
            return [SimpleNamespace(boxes=Boxes((
                256.0 - width / 2.0, 256.0 - height / 2.0,
                256.0 + width / 2.0, 256.0 + height / 2.0,
            )))]

    def track(heights):
        monkeypatch.setattr(face_core, "_load_detector", lambda _name: Detector(heights))
        result = H3FaceTrackCrop().run(
            torch.zeros((len(heights), 512, 512, 3)),
            "fake.pt", 0.35, 3.0, 768, 768, "manual",
            21, 21, "gaussian", "per_frame",
            identity_track=False, cut_detection="none", verbose=False,
        )
        return [box[3] for box in result[1]["boxes"]]

    approaching = track([32, 48, 72, 104, 144, 184, 224])
    retreating = track([224, 184, 144, 104, 72, 48, 32])
    assert approaching == sorted(approaching)
    assert retreating == sorted(retreating, reverse=True)


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
    face = MiniMaxH3FaceRefineLatentStar7.INPUT_TYPES()
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
    assert names.index("face_detector") == names.index("enable_refine") + 1
    assert names.index("face_lora") == names.index("face_detector") + 1
    assert names.index("face_lora_strength") == names.index("face_lora") + 1
    assert names.index("face_attention") == names.index("face_lora_strength") + 1
    assert names.index("face_count") == names.index("face_attention") + 1
    assert face["required"]["face_detector"][1]["default"] == "face_yolov8m.pt"
    assert _face_detector_choices()[0] == "face_yolov8m.pt"
    assert face["required"]["face_lora"][1]["default"] == "继承一采"
    assert face["required"]["face_lora_strength"][1]["default"] == 1.0
    assert face["required"]["face_lora_strength"][1]["step"] == 0.01
    assert face["required"]["face_attention"][1]["default"] == "继承一采"
    assert face["required"]["custom_crop_context"][1]["default"] == 2.6
    assert face["required"]["custom_crop_context"][1]["min"] == 1.4
    assert face["required"]["face_count"][1]["default"] == 1
    assert face["required"]["face_count"][1]["max"] == 4
    assert face["required"]["target_face"][0] == ["主人物", "画面中央", "参考图匹配"]
    assert face["required"]["preserve_repair_detail"][1]["default"] is True
    assert face["hidden"]["unique_id"] == "UNIQUE_ID"
    assert MiniMaxH3FaceRefineLatentStar7.RETURN_TYPES == ("LATENT", "STRING")
    assert MiniMaxH3FaceRefineLatentStar7.RETURN_NAMES == (
        "refined_av_latent", "report"
    )
    assert MiniMaxH3FaceRefineLatentStar7.DEPRECATED is False
    assert MiniMaxH3FaceRefineStar7.RETURN_TYPES == ("IMAGE",)
    assert MiniMaxH3FaceRefineStar7.DEPRECATED is True


def test_face_presets_restore_legacy_sampling_baseline():
    assert _PRESETS["自动平衡"]["denoise"] == .30
    assert _PRESETS["远景小脸"]["denoise"] == .48
    assert _PRESETS["远景小脸"]["crop"] == 2.4


def _draw_test_face():
    image = np.full((128, 128, 3), .2, np.float32)
    cv2.ellipse(image, (64, 64), (20, 28), 0, 0, 360, (.65, .5, .4), -1)
    cv2.circle(image, (56, 58), 3, (.1, .1, .1), -1)
    cv2.circle(image, (72, 58), 3, (.1, .1, .1), -1)
    cv2.ellipse(image, (64, 75), (8, 3), 0, 0, 180, (.9, .15, .2), 2)
    return image


def test_absent_shots_keep_the_original_timeline_in_star7_repair():
    cache = {"frames": 39, "src_size": (32, 32),
             "boxes": [[[8, 8, 24, 24]]] * 5 + [[]] * 17 + [[[8, 8, 24, 24]]] * 17,
             "confs": [[.95]] * 5 + [[]] * 17 + [[.95]] * 17,
             "segments": [(0, 5), (5, 22), (22, 39)]}
    pick = _build_multiface_picks(cache, 1)[0]
    result = H3FaceTrackCrop().run(
        torch.zeros(39, 32, 32, 3), "unused.pt", .35, 2.7, 32, 32, "manual", 9, 15,
        "gaussian", "per_frame", identity_track=False, face_pick=pick,
        keep_all_frames=True, verbose=False,
    )
    assert result[0].shape[0] == 39
    assert result[1]["source"] == list(range(39))
    assert all(result[1]["absent"][5:22])


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("preserve_detail", [False, True])
def test_legacy_and_deferred_decode_match_pixels_without_full_frame_encode(monkeypatch, scale, preserve_detail):
    from . import h3_face_refine_star7 as node_module
    from .h3_stream_decode_star7 import _stitch_deferred_faces
    from comfy_extras import nodes_custom_sampler as sampler_nodes
    from comfy.model_sampling import ModelSamplingAV
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
    sampling = ModelSamplingAV()
    sampling.set_parameters(shift=10, audio_shift=3)

    class Model:
        model = None
        def clone(self):
            return self
        def get_model_object(self, _name):
            return sampling

    monkeypatch.setattr(node_module, "_FACE_REPAIR_OUTPUT_FLOOR_MP", 192 * 192 / (1024 * 1024))
    encodes = []

    class VAE:
        def decode(self, latent):
            face = torch.from_numpy(_draw_test_face())[None].movedim(-1, 1)
            frames = torch.nn.functional.interpolate(face, size=(latent.shape[-2] * 16, latent.shape[-1] * 16),
                                                     mode="bilinear", align_corners=False).movedim(1, -1).repeat(5, 1, 1, 1)
            if bool(latent.any()):
                frames[..., 0] = (frames[..., 0] + .12).clamp(0, 1)
            return frames * torch.linspace(.8, 1., 5)[:, None, None, None]
        def encode(self, images):
            encodes.append(tuple(images.shape))
            assert images.shape[1:3] == (64, 64), "Only face crops may be encoded"
            return torch.zeros(1, 24, 2, images.shape[1] // 16, images.shape[2] // 16)

    def track(_self, images, *args, **kwargs):
        assert kwargs["keep_all_frames"] is True
        width, height = images.shape[2], images.shape[1]
        transform = {"boxes": [(width / 4, height / 4, width / 2, height / 2)] * 5, "canvas": (64, 64), "src_size": (width, height),
                     "source": list(range(5)), "weights": [1.] * 5, "face_rect": [(21, 16, 22, 32)] * 5,
                     "crop_factor": 2.7, "segments": [(0, 5)]}
        centre = images[:, height // 4:height * 3 // 4, width // 4:width * 3 // 4]
        crops = torch.nn.functional.interpolate(centre.movedim(-1, 1), size=(64, 64), mode="bilinear", align_corners=False).movedim(1, -1)
        return crops, transform, None, "tracked", 64, 64, 5

    monkeypatch.setattr(node_module, "_ensure_face_detector", lambda _name: "installed.pt")
    monkeypatch.setattr(node_module.H3FaceTrackCrop, "run", track)
    monkeypatch.setattr(sampler_nodes.BasicGuider, "execute", lambda *a: (object(),))
    monkeypatch.setattr(sampler_nodes.KSamplerSelect, "execute", lambda *a: (object(),))
    monkeypatch.setattr(sampler_nodes.RandomNoise, "execute", lambda *a: (object(),))

    def sample(noise, guider, sampler, sigmas, latent):
        assert len(sigmas) == 5
        expected = node_module._unpack(sampler_nodes.BasicScheduler.execute(Model(), "simple", 4, .48))[0]
        assert torch.equal(sigmas, expected)
        assert torch.count_nonzero(list(latent["noise_mask"].unbind())[1]) == 0
        result = dict(latent)
        video, audio = latent["samples"].unbind()
        result["samples"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), audio))
        return (result,)

    monkeypatch.setattr(sampler_nodes.SamplerCustomAdvanced, "execute", sample)
    audio = torch.rand(1, 32, 2, 8)
    av = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 8 * scale, 8 * scale), audio))}
    vae = VAE()
    context = {"model": Model(), "video_vae": vae, "positive": [[torch.zeros(1), {}]]}
    result, report = MiniMaxH3FaceRefineLatentStar7().refine(
        av, context, preset="远景小脸", preserve_repair_detail=preserve_detail,
    )
    legacy = MiniMaxH3FaceRefineStar7().refine(
        av, context, preset="远景小脸", preserve_repair_detail=preserve_detail,
    )[0]
    assert result["samples"] is av["samples"]
    assert list(result["samples"].unbind())[1] is audio
    assert "no full-frame re-encode" in report
    frames = vae.decode(list(av["samples"].unbind())[0])
    output, count = _stitch_deferred_faces(frames, result, vae)
    size = max(128 * scale, 192) if preserve_detail else 128 * scale
    assert count == 1 and output.shape == (5, size, size, 3)
    assert torch.isfinite(output).all()
    assert torch.equal(output, legacy)
    assert len(encodes) == 2
    base = node_module._resize_image_batch(frames, size, size)
    assert torch.equal(output[:, 0, 0], base[:, 0, 0])
    assert not torch.equal(output[:, size // 2, size // 2], base[:, size // 2, size // 2])
