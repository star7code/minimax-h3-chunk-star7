import copy
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parent
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CHUNK = _load(ROOT / "nodes.py", "star7_chunk_matrix")
FP16 = _load(
    ROOT.parent / "minimax-h3-fp16-exact-star7" / "nodes.py",
    "star7_fp16_matrix",
)
TURBO = _load(
    ROOT.parent / "ComfyUI-MiniMax-H3-Turbo" / "__init__.py",
    "official_minimax_h3_turbo_matrix",
)


class _FakePatcher:
    def __init__(self, diffusion, *, object_patches=None, patches=None):
        self.diffusion = diffusion
        self.model_options = {"transformer_options": {}}
        self.object_patches = dict(object_patches or {})
        self.patches = dict(patches or {})
        self.wrappers = {}
        self.compute_dtype = None
        self.force_cast_weights = False

    def clone(self):
        cloned = copy.copy(self)
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.object_patches = self.object_patches.copy()
        cloned.patches = self.patches.copy()
        cloned.wrappers = {
            kind: {key: list(values) for key, values in entries.items()}
            for kind, entries in self.wrappers.items()
        }
        return cloned

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def set_model_compute_dtype(self, dtype):
        self.compute_dtype = dtype
        self.force_cast_weights = dtype is not None
        self.add_object_patch("manual_cast_dtype", dtype)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def add_wrapper_with_key(self, wrapper_type, key, value):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(value)


def _make_patcher():
    from comfy.ldm.minimax import model as h3_model

    diffusion = h3_model.MiniMaxH3Model(
        hidden_size=16,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=24,
        latents_dim=4,
        audio_latents_dim=4,
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=2,
        dtype=torch.float16,
        device="cpu",
        operations=torch.nn,
    )
    return _FakePatcher(diffusion), diffusion


def _install_official_turbo_shape(patcher, diffusion):
    """Use the official loader helper without changing the official module."""
    base = diffusion.blocks[0].adaln_proj
    rank = 2
    a = torch.zeros((rank, base.linear.in_features))
    b = torch.zeros((base.linear.out_features, rank))
    external_forward = TURBO._make_adaln_forward(
        base, a, b, {"silu_temb": None}
    )
    patcher.add_object_patch(
        "diffusion_model.blocks.0.adaln_proj.forward", external_forward
    )
    TURBO._add_dbg_wrapper(patcher, diffusion, "matrix", "bypass")


def _install_stock_lora_shape(patcher, diffusion):
    """Represent the stock loader's model weight patch without custom wrappers."""
    marker = object()
    patcher.patches["diffusion_model.blocks.0.mlp.fc1.weight"] = [marker]
    linear = diffusion.blocks[0].mlp.fc1
    linear.weight_function = [lambda value: value]
    return marker


def _apply_fp16(patcher):
    node = FP16.MiniMaxH3FP16ExactFixStar7()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
    ):
        return node.patch(patcher, enabled=True)[0]


def _apply_chunk(patcher):
    return CHUNK.install_model_patch(
        patcher,
        chunk_tokens=8192,
        auto_halve_on_oom=True,
        verbose=False,
        mlp_chunk_tokens=4096,
        disable_dynamic_prefetch=True,
        reuse_mlp_weights=True,
        attention_backend="existing",
    )


def _run_case(loader, use_fp16):
    patcher, diffusion = _make_patcher()
    if loader == "official-turbo":
        _install_official_turbo_shape(patcher, diffusion)
    else:
        stock_marker = _install_stock_lora_shape(patcher, diffusion)

    if use_fp16:
        patcher = _apply_fp16(patcher)
    final = _apply_chunk(patcher)

    assert "diffusion_model.blocks.0.mlp.forward" in final.object_patches
    assert final.object_patches[
        "diffusion_model.blocks.0.mlp.forward"
    ].__func__._star7_wrapper_kind == "mlp-chunk-upstream"

    if loader == "official-turbo":
        assert "diffusion_model.blocks.0.adaln_proj.forward" in final.object_patches
        assert "h3turbo_dbg" in final.wrappers[next(iter(final.wrappers))]
    else:
        assert final.patches["diffusion_model.blocks.0.mlp.fc1.weight"] == [stock_marker]
        assert diffusion.blocks[0].mlp.fc1.weight_function

    if use_fp16:
        options = final.model_options["transformer_options"]
        assert options[FP16.PATCH_FLAG] == FP16.NODE_VERSION
        upstream = final.object_patches[
            "diffusion_model.blocks.0.mlp.forward"
        ].__func__._star7_original_forward
        assert getattr(getattr(upstream, "__func__", upstream), "_star7_wrapper_kind", None) is None

    diffusion_ref = weakref.ref(diffusion)
    del final, patcher, diffusion
    gc.collect()
    assert diffusion_ref() is None, f"{loader} fp16={use_fp16} retained MiniMaxH3"


def test_loader_chunk_compatibility_matrix():
    for loader in ("official-turbo", "comfy-lora"):
        for use_fp16 in (False, True):
            _run_case(loader, use_fp16)


if __name__ == "__main__":
    test_loader_chunk_compatibility_matrix()
    print("MiniMax H3 loader/FP16/Chunk compatibility matrix passed")
