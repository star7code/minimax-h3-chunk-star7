import torch

import comfy.nested_tensor
import comfy.utils

from .h3_latent_upscale_star7 import (
    MiniMaxH3OneClickHDStar7,
    _H3Resizer3D,
    _add_hd_endpoint_guides,
    _sampling_profile,
    _pdd_partition,
    _pdd_tail_sigmas,
    _smart_tile_grid,
    _spatial_tile_plan,
    _tile_windows,
    _tile_payload,
    _SpatialTileModel,
    _target_geometry,
    _temporal_weight,
)


def test_target_geometry_reaches_requested_area_and_never_shrinks():
    video = torch.zeros((1, 24, 2, 30, 54))  # 864x480
    source_w, source_h, width, height, scale = _target_geometry(video, 1.0)
    assert (source_w, source_h) == (864, 480)
    assert width % 32 == 0 and height % 32 == 0
    assert width >= source_w and height >= source_h
    assert width * height >= 900_000
    assert scale > 1.0


def test_target_geometry_keeps_source_that_already_exceeds_target():
    video = torch.zeros((1, 24, 2, 64, 64))
    assert _target_geometry(video, 0.2)[:4] == (1024, 1024, 1024, 1024)


def test_temporal_overlap_weights_keep_every_frame_covered():
    first = _temporal_weight(40, 8, True, False, torch.device("cpu"), torch.float32)
    middle = _temporal_weight(40, 8, False, False, torch.device("cpu"), torch.float32)
    last = _temporal_weight(17, 8, False, True, torch.device("cpu"), torch.float32)
    assert torch.all(first > 0)
    assert torch.all(middle > 0)
    assert torch.all(last > 0)
    assert first[0, 0, 0, 0, 0] == 1.0
    assert last[0, 0, -1, 0, 0] == 1.0


def test_custom_zero_step_route_is_a_latent_passthrough_when_already_large_enough():
    video = torch.zeros((1, 24, 2, 64, 64))
    audio = torch.randn((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    context = {"model": object(), "positive": [[torch.zeros(1), {}]]}
    output, report = MiniMaxH3OneClickHDStar7().upscale(
        latent,
        context,
        preset="自定义",
        target_megapixels=0.2,
        refine_steps=0,
        refine_strength=0.0,
        seed=0,
    )
    output_video, output_audio = list(output["samples"].unbind())
    assert torch.equal(output_video, video)
    assert torch.equal(output_audio, audio)
    assert "1024x1024 -> 1024x1024" in report
    assert "audio preserved exactly" in report


def test_schema_has_scene_presets_and_latent_output():
    schema = MiniMaxH3OneClickHDStar7.INPUT_TYPES()
    assert schema["required"]["preset"][0] == [
        "平衡高清", "高质量", "远景小脸", "高速运动", "自定义",
    ]
    assert schema["hidden"] == {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}
    assert schema["required"]["refine_steps"][1]["default"] == 2
    assert schema["required"]["refine_strength"][1]["default"] == 0.25
    assert schema["required"]["enable_hd"][1]["default"] is True
    assert schema["required"]["enable_tiling"][1]["default"] is False
    assert schema["required"]["tile_count"][1]["default"] == 2
    assert schema["required"]["tile_count"][1]["max"] == 64
    assert schema["required"]["second_pass_attention"][1]["default"] == "继承一采"
    assert schema["required"]["upscale_model"][0][0] == (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    )
    assert MiniMaxH3OneClickHDStar7.RETURN_TYPES[0] == "LATENT"


def test_master_switch_bypasses_before_context_or_model_loading():
    marker = {"samples": object()}
    output, report = MiniMaxH3OneClickHDStar7().upscale(
        marker, None, enable_hd=False,
    )
    assert output is marker
    assert "disabled" in report


def test_spatial_tiles_are_even_aligned_and_weights_cover_the_canvas():
    video = torch.zeros((1, 24, 3, 40, 80))
    grid, tiles = _spatial_tile_plan(video, 3, 128)
    assert grid == (2, 2)
    assert len(tiles) == 4
    assert min(tile[0] for tile in tiles) == 0
    assert max(tile[1] for tile in tiles) == 40
    assert min(tile[2] for tile in tiles) == 0
    assert max(tile[3] for tile in tiles) == 80
    assert all(all(value % 2 == 0 for value in tile) for tile in tiles)
    windows, denominator = _tile_windows(tiles, 40, 80, video.device)
    assert len(windows) == 4
    assert torch.all(denominator > 0)


def test_smart_grid_avoids_bad_odd_strips_and_respects_orientation():
    assert _smart_tile_grid(40, 80, 2) == (1, 2)
    assert _smart_tile_grid(40, 80, 3) == (2, 2)
    assert _smart_tile_grid(40, 80, 4) == (2, 2)
    assert _smart_tile_grid(40, 80, 5) == (2, 3)
    assert _smart_tile_grid(40, 80, 7) == (2, 4)
    assert _smart_tile_grid(80, 40, 5) == (3, 2)
    assert _smart_tile_grid(64, 64, 7) == (3, 3)
    assert _smart_tile_grid(32, 96, 3) == (1, 3)


def test_smart_grid_handles_sixteen_tiles_without_extreme_tile_shapes():
    assert _smart_tile_grid(90, 160, 16) == (3, 6)
    assert _smart_tile_grid(160, 90, 16) == (6, 3)
    assert _smart_tile_grid(120, 160, 16) == (4, 4)
    assert _smart_tile_grid(160, 120, 16) == (4, 4)
    assert _smart_tile_grid(128, 128, 16) == (4, 4)


def test_tiled_model_proxy_merges_video_and_averages_audio_predictions():
    video = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 4, 8)
    audio = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 4)
    packed, shapes = comfy.utils.pack_latents([video, audio])

    class Model:
        latent_image = None
        noise = None

        def __call__(self, value, _sigma, **_kwargs):
            return value * 2

    grid, tiles = _spatial_tile_plan(video, 2, 32)
    merged = _SpatialTileModel(Model(), shapes, grid, tiles)(packed, torch.tensor([0.5]))
    merged_video, merged_audio = comfy.utils.unpack_latents(merged, shapes)
    assert torch.allclose(merged_video, video * 2)
    assert torch.allclose(merged_audio, audio * 2)


def test_hd_endpoint_guides_are_reencoded_at_target_resolution():
    video = torch.zeros((1, 24, 2, 32, 48))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    positive = [[torch.zeros((1, 2, 3)), {}]]

    class VAE:
        def encode(self, frames):
            assert tuple(frames.shape[1:3]) == (512, 768)
            return torch.ones((1, 24, 1, 32, 48))

    image = torch.zeros((1, 96, 128, 3))
    result, summary = _add_hd_endpoint_guides(
        positive, latent,
        {"video_vae": VAE(), "first_frame": image, "last_frame": image},
    )
    keyframes = result[0][1]["minimax_keyframes"]
    assert summary == "first+last"
    assert [item["resolved_frame_index"] for item in keyframes] == [0, 4]
    assert all(tuple(item["latent"].shape[-2:]) == (32, 48) for item in keyframes)


def test_tiled_payload_crops_hd_keyframes_and_rebuilds_condition_latents():
    from comfy.ldm.minimax.model import PackedLayout

    keyframe_latent = torch.ones((1, 24, 1, 32, 48))
    ref_latent = torch.ones((1, 24, 1, 8, 8))
    keyframe = {"resolved_frame_index": 0, "latent": keyframe_latent}
    ref = {"kind": "image", "latent": ref_latent, "latent_h": 8, "latent_w": 8}
    payload = {
        "layout": PackedLayout(4, 2, 32, 48, 8, keyframes=[keyframe], refs=[ref]),
        "keyframes": [keyframe], "refs": [ref],
        "cond_video_latents": [keyframe_latent, ref_latent],
    }
    updated = _tile_payload(
        payload, [(1, 24, 2, 20, 28), (1, 32, 2, 8)], (0, 20, 0, 28)
    )
    assert tuple(updated["keyframes"][0]["latent"].shape[-2:]) == (20, 28)
    assert updated["cond_video_latents"][0] is updated["keyframes"][0]["latent"]
    assert updated["cond_video_latents"][1] is ref_latent
    old_layout = payload["layout"]
    new_layout = updated["layout"]
    old_video_start, old_video_stop, _ = old_layout.segments[-1]
    new_video_start, new_video_stop, _ = new_layout.segments[-1]
    expected = old_layout.position_ids[old_video_start:old_video_stop]
    expected = expected.reshape(2, 16, 24, 3)[:, :10, :14, :].reshape(-1, 3)
    actual = new_layout.position_ids[new_video_start:new_video_stop]
    assert torch.equal(actual, expected)


def test_tiled_payload_preserves_full_canvas_audio_and_offset_video_positions():
    from comfy.ldm.minimax.model import PackedLayout

    payload = {"layout": PackedLayout(4, 2, 32, 48, 8)}
    tile = (12, 32, 20, 48)
    updated = _tile_payload(
        payload, [(1, 24, 2, 20, 28), (1, 32, 2, 8)], tile
    )
    old_layout = payload["layout"]
    new_layout = updated["layout"]
    old_audio = next(segment for segment in old_layout.segments if segment[2] == "audio")
    new_audio = next(segment for segment in new_layout.segments if segment[2] == "audio")
    assert torch.equal(
        old_layout.position_ids[old_audio[0]:old_audio[1]],
        new_layout.position_ids[new_audio[0]:new_audio[1]],
    )
    old_video = old_layout.segments[-1]
    new_video = new_layout.segments[-1]
    expected = old_layout.position_ids[old_video[0]:old_video[1]]
    expected = expected.reshape(2, 16, 24, 3)[:, 6:16, 10:24, :].reshape(-1, 3)
    actual = new_layout.position_ids[new_video[0]:new_video[1]]
    assert torch.equal(actual, expected)


def test_sampling_profile_distinguishes_base_turbo_and_pdd_prompt_paths():
    base = type("Model", (), {"patches": {}})()
    assert _sampling_profile(base, {}, "1", 12)[0] == "base"
    prompt = {
        "1": {"class_type": "MiniMaxH3OneClickHDStar7", "inputs": {"h3_context": ["2", 4]}},
        "2": {"class_type": "MiniMaxH3MaterialPromptStar7", "inputs": {"model": ["3", 0]}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "minimax_h3_turbo_v4_step600.safetensors"
        }},
    }
    assert _sampling_profile(base, prompt, "1", 6)[0] == "turbo"
    prompt["3"]["class_type"] = "MiniMaxH3PDDAccApply"
    assert _sampling_profile(base, prompt, "1", 12)[0] == "pdd"


def test_pdd_partition_and_trained_tail_sigmas_are_resolved_from_prompt():
    prompt = {
        "1": {"class_type": "MiniMaxH3OneClickHDStar7", "inputs": {"h3_context": ["2", 4]}},
        "2": {"class_type": "MiniMaxH3MaterialPromptStar7", "inputs": {"model": ["3", 0]}},
        "3": {"class_type": "MiniMaxH3PDDAccApply", "inputs": {"nfe": "8"}},
    }
    partition = _pdd_partition(prompt, "1")
    assert partition == [4] * 8
    assert torch.allclose(
        _pdd_tail_sigmas(partition, 3, 12),
        torch.tensor([0.8780488, 0.8, 0.6315789, 0.0]),
        atol=1e-6,
    )


def test_pdd_six_step_partition_and_shift_guard():
    prompt = {
        "1": {"class_type": "MiniMaxH3OneClickHDStar7", "inputs": {"h3_context": ["2", 0]}},
        "2": {"class_type": "MiniMaxH3PDDAccApply", "inputs": {"nfe": "6"}},
    }
    partition = _pdd_partition(prompt, "1")
    assert partition == [8, 8, 4, 4, 4, 4]
    assert torch.allclose(
        _pdd_tail_sigmas(partition, 2, 12),
        torch.tensor([0.8, 0.6315789, 0.0]),
        atol=1e-6,
    )
    try:
        _pdd_tail_sigmas(partition, 2, 6)
    except RuntimeError as exc:
        assert "Shift 12" in str(exc)
    else:
        raise AssertionError("PDD must reject an off-training video shift")
    try:
        _pdd_tail_sigmas(partition, 2, 12, 6)
    except RuntimeError as exc:
        assert "audio Shift 3" in str(exc)
    else:
        raise AssertionError("PDD must reject an off-training audio shift")


def test_disabled_pdd_apply_is_not_treated_as_an_active_head_bank():
    base = type("Model", (), {"patches": {}})()
    prompt = {
        "1": {"class_type": "MiniMaxH3OneClickHDStar7", "inputs": {"h3_context": ["2", 4]}},
        "2": {"class_type": "MiniMaxH3PDDAccApply", "inputs": {
            "nfe": "8", "enabled": False,
        }},
    }
    assert _pdd_partition(prompt, "1") is None
    assert _sampling_profile(base, prompt, "1", 12)[0] == "base"


def test_resizer_keeps_torch_module_apply_contract():
    # nn.Module.to() internally calls Module._apply(convert). A model helper must
    # never shadow that private PyTorch method (the original runtime regression).
    with torch.device("meta"):
        network = _H3Resizer3D()
    network.to(dtype=torch.float16)
    assert all(parameter.dtype == torch.float16 for parameter in network.parameters())
