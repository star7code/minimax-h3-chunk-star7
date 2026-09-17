import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "star7_adaptive_loader_tests", ROOT / "adaptive_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_loader_uses_the_fp16_loader_interface():
    module = _load_module()
    with mock.patch(
        "folder_paths.get_filename_list", return_value=["h3.safetensors"]
    ):
        required = module.MiniMaxH3FP16LoaderStar7.INPUT_TYPES()["required"]
    assert set(required) == {"unet_name"}
    assert module.NODE_VERSION == "2.0.14"


def test_chunk_project_registers_an_independent_enhanced_loader_id():
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    web_source = (ROOT / "web" / "adaptive_loader.js").read_text(encoding="utf-8")

    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3ChunkEnhancedLoaderStar7"]' in init_source
    assert 'NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ChunkEnhancedLoaderStar7"]' in init_source
    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3FP16LoaderStar7"]' not in init_source
    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3EnhancedLoaderStar7"]' not in init_source
    assert '"MiniMax H3 增强载入"' in init_source
    assert '"MiniMax H3 Enhanced Loader"' in init_source
    assert 'const NODE = "MiniMaxH3ChunkEnhancedLoaderStar7";' in web_source
    assert '"MiniMax H3 增强载入 - Star7"' in web_source
    assert '"MiniMax H3 Enhanced Loader - Star7"' in web_source
    assert "beforeRegisterNodeDef(_nodeType, nodeData)" in web_source
    assert "nodeData.display_name = title();" in web_source


def test_adaptive_loader_preserves_attention_override_under_fp16():
    module = _load_module()
    seen = {}

    class Identity(torch.nn.Module):
        def forward(self, value):
            return value

    class Block:
        norm1 = Identity()
        norm2 = Identity()

        def adaln_proj(self, _t_emb):
            return (torch.zeros(1),) * 6

        def attn(self, _value, **_kwargs):
            raise AssertionError("the supplied attention override must be used")

        def mlp(self, value):
            seen["mlp"] = value.dtype
            return value

    minimax = SimpleNamespace(
        _mod_scale_shift=lambda value, *_args: value,
        _mod_gate=lambda residual, _gate, update, _segments: residual + update,
    )
    forward = module._block_forward(
        lambda *_args, **_kwargs: None, minimax
    )

    def attention(value, **_kwargs):
        seen["attention"] = value.dtype
        return value

    output = forward(
        Block(), torch.ones(1, 2), torch.zeros(1), [], None, {},
        attention=attention,
    )
    assert output.dtype is torch.float32
    assert seen == {"attention": torch.float16, "mlp": torch.float16}
