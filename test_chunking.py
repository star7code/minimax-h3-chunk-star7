import importlib.util
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import torch


COMFY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

SPEC = importlib.util.spec_from_file_location("h3_chunk_nodes", pathlib.Path(__file__).with_name("nodes.py"))
chunk_nodes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(chunk_nodes)


def _rotation_table(seq_len: int, rot_dim: int, device: torch.device) -> torch.Tensor:
    angles = torch.randn(seq_len, rot_dim // 2, dtype=torch.float32, device=device)
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack((c, -s, s, c), dim=-1).reshape(1, seq_len, 1, rot_dim // 2, 2, 2)


def test_rope_matches_eager_partial_rotary(device=torch.device("cpu")):
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half_

    torch.manual_seed(123)
    shape = (1, 519, 3, 16)
    rot_dim = 12
    q = torch.randn(shape, dtype=torch.float32, device=device)
    k = torch.randn(shape, dtype=torch.float32, device=device)
    q_scale = torch.randn(shape[-1], dtype=torch.float32, device=device)
    k_scale = torch.randn(shape[-1], dtype=torch.float32, device=device)
    freqs = _rotation_table(shape[1], rot_dim, device)

    q_expected, k_expected = q.clone(), k.clone()
    rms_rope_split_half_(q_expected, k_expected, freqs, q_scale, k_scale, rot_dim=rot_dim)

    chunk_nodes._CONFIG.update(chunk_tokens=256, auto_halve_on_oom=False, verbose=False)
    q_actual, k_actual = q.clone(), k.clone()
    chunk_nodes._chunked_rms_rope_split_half_inplace(
        q_actual, k_actual, freqs, q_scale, k_scale, rot_dim=rot_dim
    )

    torch.testing.assert_close(q_actual, q_expected, rtol=0, atol=0)
    torch.testing.assert_close(k_actual, k_expected, rtol=0, atol=0)


def test_mlp_chunk_matches_full_forward(device=torch.device("cpu")):
    from comfy.ldm.minimax import model as h3_model

    torch.manual_seed(456)
    mlp = h3_model.MLP(hidden=16, ffn=24, operations=torch.nn).to(device)
    x = torch.randn(519, 16, dtype=torch.float32, device=device)
    expected = h3_model.MLP.forward(mlp, x)

    chunk_nodes._CONFIG.update(mlp_chunk_tokens=256, auto_halve_on_oom=False, verbose=False)
    actual = chunk_nodes._chunked_h3_mlp_forward(mlp, x)

    # Splitting a GEMM can select a different kernel tile, so float32 reduction
    # order may differ in the last bit even though every row's math is unchanged.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-7)


def test_star7_fp16_mlp_chunk_matches_full_formula(device=torch.device("cpu")):
    from comfy.ldm.minimax import model as h3_model

    torch.manual_seed(789)
    mlp = h3_model.MLP(hidden=16, ffn=24, operations=torch.nn).to(device=device, dtype=torch.float16)
    x = torch.randn(519, 16, dtype=torch.float16, device=device)
    projected = mlp.fc1(x)
    gate, up = projected.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate.float()).mul_(up.float())
    expected = mlp.fc2((activated / 256.0).half()).float().mul_(256.0)

    chunk_nodes._CONFIG.update(mlp_chunk_tokens=256, auto_halve_on_oom=False, verbose=False)
    actual = chunk_nodes._run_chunked_h3_mlp(mlp, x, star7_fp16=True)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_auto_fp16_out_proj_uses_exact_scaling(device=torch.device("cpu")):
    linear = torch.nn.Linear(4, 4, bias=False, device=device, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.fill_(4.0)
    value = torch.full((2, 4), 20000.0, device=device, dtype=torch.float16)
    expected = value.float().matmul(linear.weight.float().t())
    assert not torch.isfinite(linear(value)).all()
    actual = chunk_nodes._fp16_exact_out_proj(linear, value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_qkv_chunk_writes_backend_dtype_without_full_cast(device=torch.device("cpu")):
    from comfy.ldm.minimax import model as h3_model
    import comfy.model_management as mm
    import comfy.quant_ops

    torch.manual_seed(7891)
    attention = h3_model.Attention(
        hidden=16,
        heads=2,
        head_dim=8,
        eps=1e-6,
        operations=torch.nn,
    ).to(device=device, dtype=torch.float32)
    x = torch.randn(519, 16, device=device, dtype=torch.float32)
    qkv = attention.qkv_proj(x)
    q, k, v = qkv.split(16, dim=-1)
    expected_q = attention.q_norm(q.view(519, 2, 8)).permute(1, 0, 2).unsqueeze(0).half()
    expected_k = attention.k_norm(k.view(519, 2, 8)).permute(1, 0, 2).unsqueeze(0).half()
    expected_v = v.view(519, 2, 8).permute(1, 0, 2).unsqueeze(0).half()

    original_config = chunk_nodes._CONFIG.copy()
    try:
        chunk_nodes._CONFIG.update(
            qkv_chunk_tokens=256,
            effective_qkv_chunk_tokens=256,
            status_effective_qkv_chunk_tokens=256,
            auto_halve_on_oom=False,
            verbose=False,
        )
        actual_q, actual_k, actual_v = chunk_nodes._prepare_h3_qkv_chunked(
            attention, x, None, mm, comfy.quant_ops, output_dtype=torch.float16
        )
        assert actual_q.dtype == actual_k.dtype == actual_v.dtype == torch.float16
        torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=5e-7)
        torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=5e-7)
        torch.testing.assert_close(actual_v, expected_v, rtol=1e-3, atol=5e-7)
    finally:
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_star7_resident_weight_path_matches_formula(device=torch.device("cpu")):
    import comfy.ops

    torch.manual_seed(790)
    operations = comfy.ops.mixed_precision_ops(compute_dtype=torch.float16)

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = operations.Linear(16, 48, bias=False, device=device)
            self.fc2 = operations.Linear(24, 16, bias=False, device=device)
            self.fc1.weight = torch.nn.Parameter(
                torch.randn(48, 16, dtype=torch.float16, device=device) * 0.02,
                requires_grad=False,
            )
            self.fc2.weight = torch.nn.Parameter(
                torch.randn(16, 24, dtype=torch.float16, device=device) * 0.02,
                requires_grad=False,
            )

    mlp = TinyMLP()
    x = torch.randn(519, 16, dtype=torch.float16, device=device)
    projected = torch.nn.functional.linear(x, mlp.fc1.weight)
    gate, up = projected.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate.float()).mul_(up.float())
    expected = torch.nn.functional.linear(
        (activated / 256.0).half(), mlp.fc2.weight
    ).float().mul_(256.0)

    chunk_nodes._CONFIG.update(
        mlp_chunk_tokens=256,
        auto_halve_on_oom=False,
        verbose=False,
        reuse_mlp_weights=True,
    )
    actual = chunk_nodes._run_chunked_h3_mlp(mlp, x, star7_fp16=True)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_weight_only_quantized_resident_path_stays_quantized(device=torch.device("cpu")):
    import contextlib
    import comfy.ops

    class TinyQuantLinear(torch.nn.Module):
        comfy_force_cast_weights = False
        _full_precision_mm = False
        quant_format = "int8_tensorwise"
        layout_type = "TensorWiseINT8Layout"
        weight_function = []
        bias_function = []
        bias = None

        def __init__(self, out_features, in_features):
            super().__init__()
            params = comfy.ops.TensorWiseINT8Layout.Params(
                scale=torch.ones((), dtype=torch.float32, device=device),
                orig_dtype=torch.float16,
                orig_shape=(out_features, in_features),
                convrot=True,
                convrot_groupsize=256,
            )
            weight = comfy.ops.QuantizedTensor(
                torch.zeros(out_features, in_features, dtype=torch.int8, device=device),
                self.layout_type,
                params,
            )
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.out_features = out_features

        def _forward(self, x, weight, bias):
            assert isinstance(weight, comfy.ops.QuantizedTensor)
            return torch.nn.functional.linear(x, weight, bias)

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = TinyQuantLinear(512, 256)
            self.fc2 = TinyQuantLinear(256, 256)

    mlp = TinyMLP()
    x = torch.zeros(4, 256, dtype=torch.float16, device=device)
    assert chunk_nodes._linear_quantization_mode(mlp.fc1) == "weight-only"
    with contextlib.ExitStack() as stack:
        _fc1, _fc2, backend = chunk_nodes._resident_mlp_callers(mlp, x, stack)
        assert backend == "quantized"
    if device.type == "cuda":
        chunk_nodes._CONFIG.update(
            mlp_chunk_tokens=256,
            auto_halve_on_oom=False,
            verbose=False,
            reuse_mlp_weights=True,
        )
        actual = chunk_nodes._run_chunked_h3_mlp(
            mlp,
            torch.zeros(257, 256, dtype=torch.float16, device=device),
            star7_fp16=True,
        )
        assert actual.count_nonzero().item() == 0


def test_dynamic_vbar_linear_can_be_snapshotted_for_resident_reuse():
    dynamic = SimpleNamespace(
        _v=object(),
        weight=object(),
        weight_function=[],
        bias_function=[],
        _forward=lambda *args: None,
    )
    assert chunk_nodes._linear_can_reuse_weights(dynamic) is True

    low_vram = SimpleNamespace(
        weight=object(),
        weight_function=[],
        bias_function=[],
        weight_lowvram_function=object(),
        _forward=lambda *args: None,
    )
    assert chunk_nodes._linear_can_reuse_weights(low_vram) is True

    patched = SimpleNamespace(
        weight=object(),
        weight_function=[object()],
        bias_function=[],
        _forward=lambda *args: None,
    )
    assert chunk_nodes._linear_can_reuse_weights(patched) is True


def test_fused_residual_matches_materialized_mlp(device=torch.device("cpu")):
    from comfy.ldm.minimax import model as h3_model

    torch.manual_seed(791)
    mlp = h3_model.MLP(hidden=16, ffn=24, operations=torch.nn).to(
        device=device, dtype=torch.float16
    )
    x = torch.randn(519, 16, dtype=torch.float16, device=device)
    residual = torch.randn(519, 16, dtype=torch.float32, device=device)
    gate = torch.randn(3, 16, dtype=torch.float16, device=device)
    segments = [(0, 123, 0), (123, 400, 1), (400, 519, 2)]

    chunk_nodes._CONFIG.update(
        mlp_chunk_tokens=256,
        auto_halve_on_oom=False,
        verbose=False,
        reuse_mlp_weights=False,
    )
    materialized = chunk_nodes._run_chunked_h3_mlp(
        mlp, x, star7_fp16=True
    )
    expected = h3_model._mod_gate(residual.clone(), gate, materialized, segments)
    actual = chunk_nodes._run_chunked_h3_mlp(
        mlp,
        x,
        star7_fp16=True,
        residual=residual.clone(),
        gate=gate,
        segments=segments,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp16_block_patch_matches_materialized_formula(device=torch.device("cpu")):
    from comfy.ldm.minimax import model as h3_model

    torch.manual_seed(792)
    block = h3_model.DiTBlock(
        hidden=16,
        heads=2,
        head_dim=8,
        ffn=24,
        t_dim=8,
        eps=1e-6,
        qk_eps=1e-6,
        operations=torch.nn,
    ).to(device=device, dtype=torch.float16)

    class TinyAttention(torch.nn.Module):
        def forward(self, value, rope_freqs=None, transformer_options={}):
            return value.mul(0.5)

    block.attn = TinyAttention()
    x = torch.randn(519, 16, dtype=torch.float32, device=device)
    t_emb = torch.randn(1, 8, dtype=torch.float16, device=device)
    segments = [(0, 200, 0), (200, 400, 1), (400, 519, 2)]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h = h3_model._mod_scale_shift(
        block.norm1(x), shift_msa, scale_msa, segments
    ).half()
    expected = h3_model._mod_gate(
        x.clone(), gate_msa, block.attn(h, rope_freqs=None).float(), segments
    )
    h = h3_model._mod_scale_shift(
        block.norm2(expected), shift_mlp, scale_mlp, segments
    ).half()
    mlp = chunk_nodes._run_chunked_h3_mlp(block.mlp, h, star7_fp16=True)
    expected = h3_model._mod_gate(expected, gate_mlp, mlp, segments)

    patched_forward = chunk_nodes._make_chunked_h3_block_forward(True, h3_model)
    actual = patched_forward(
        block,
        x.clone(),
        t_emb,
        segments,
        None,
        transformer_options={},
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_install_preserves_upstream_block_patch():
    from comfy.ldm.minimax import model as h3_model

    diffusion_model = h3_model.MiniMaxH3Model(
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
    upstream_block_patch = object()

    class FakeModelPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {
                    "star7_minimax_h3_fp16_exact_fix": "test",
                }
            }
            self.object_patches = {
                "diffusion_model.blocks.0.forward": upstream_block_patch,
            }
            self.wrappers = {}

        def clone(self):
            return self

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return diffusion_model

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

        def add_wrapper_with_key(self, wrapper_type, key, value):
            self.wrappers[(wrapper_type, key)] = value

    patched = chunk_nodes.install_model_patch(
        FakeModelPatcher(),
        chunk_tokens=8192,
        auto_halve_on_oom=True,
        verbose=False,
        mlp_chunk_tokens=4096,
        disable_dynamic_prefetch=True,
        reuse_mlp_weights=True,
        attention_backend="existing",
    )

    assert patched.object_patches["diffusion_model.blocks.0.forward"] is upstream_block_patch
    assert "diffusion_model.blocks.1.forward" not in patched.object_patches
    for index in range(2):
        key = f"diffusion_model.blocks.{index}.mlp.forward"
        assert key in patched.object_patches
        assert patched.object_patches[key].__func__.__name__ == "forward"
    assert patched.wrappers == {}


def test_sm75_sla_auto_installs_fp16_exact_without_external_node():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    from comfy.ldm.minimax import model as h3_model

    diffusion_model = h3_model.MiniMaxH3Model(
        hidden_size=16,
        num_layers=1,
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

    class FakeModelPatcher:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.compute_dtype = None
            self.force_cast_weights = True

        def clone(self):
            return self

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return diffusion_model

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

        def set_model_compute_dtype(self, dtype):
            self.compute_dtype = dtype

    patched = chunk_nodes.install_model_patch(
        FakeModelPatcher(),
        chunk_tokens=8192,
        auto_halve_on_oom=True,
        verbose=False,
        mlp_chunk_tokens=4096,
        disable_dynamic_prefetch=True,
        reuse_mlp_weights=True,
        attention_backend="sla_sm75_qk_int8_pv_fp16",
    )
    assert patched.compute_dtype == torch.float16
    assert patched.model_options["transformer_options"][
        "star7_h3_sla_auto_fp16_exact"
    ] == chunk_nodes.NODE_VERSION
    assert "diffusion_model.condition_proj.forward" in patched.object_patches
    assert "diffusion_model.blocks.0.forward" in patched.object_patches
    assert diffusion_model.blocks[0].attn._star7_sla_auto_fp16_exact is True


def test_rope_oom_value_is_reused_for_k_and_later_calls():
    attempts = []
    original = chunk_nodes._rms_rope_one_chunk_inplace
    original_config = chunk_nodes._CONFIG.copy()

    def limited_chunk(tensor, *_args, **_kwargs):
        size = tensor.shape[1]
        attempts.append(size)
        if size > 512:
            raise RuntimeError("out of memory")

    try:
        chunk_nodes._rms_rope_one_chunk_inplace = limited_chunk
        chunk_nodes._CONFIG.update(
            chunk_tokens=1024,
            effective_chunk_tokens=1024,
            auto_halve_on_oom=True,
            verbose=False,
            node_id=None,
        )
        q = torch.zeros(1, 1024, 1, 128)
        k = torch.zeros_like(q)
        freqs = torch.zeros(1, 1024, 1, 48, 2, 2)
        scale = torch.ones(128)

        chunk_nodes._chunked_rms_rope_split_half_inplace(
            q, k, freqs, scale, rot_dim=96
        )
        first_call_attempts = attempts.copy()
        chunk_nodes._chunked_rms_rope_split_half_inplace(
            q, k, freqs, scale, rot_dim=96
        )

        assert first_call_attempts == [1024, 512, 512, 512, 512]
        assert attempts[len(first_call_attempts):] == [512, 512, 512, 512]
        assert chunk_nodes._CONFIG["effective_chunk_tokens"] == 512
    finally:
        chunk_nodes._rms_rope_one_chunk_inplace = original
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_mlp_oom_value_is_reused_for_later_blocks():
    attempts = []
    original_config = chunk_nodes._CONFIG.copy()

    class LimitedFC1:
        out_features = 4

        def __call__(self, tensor):
            attempts.append(tensor.shape[0])
            if tensor.shape[0] > 512:
                raise RuntimeError("out of memory")
            return torch.zeros(tensor.shape[0], 4, dtype=tensor.dtype)

    class FC2:
        def __call__(self, tensor):
            return torch.zeros(tensor.shape[0], 2, dtype=tensor.dtype)

    mlp = SimpleNamespace(fc1=LimitedFC1(), fc2=FC2())
    value = torch.zeros(1024, 2, dtype=torch.float16)

    try:
        chunk_nodes._CONFIG.update(
            mlp_chunk_tokens=1024,
            effective_mlp_chunk_tokens=1024,
            auto_halve_on_oom=True,
            verbose=False,
            reuse_mlp_weights=False,
            node_id=None,
        )
        chunk_nodes._run_chunked_h3_mlp(mlp, value, star7_fp16=True)
        first_call_attempts = attempts.copy()
        chunk_nodes._run_chunked_h3_mlp(mlp, value, star7_fp16=True)

        assert first_call_attempts == [1024, 512, 512]
        assert attempts[len(first_call_attempts):] == [512, 512]
        assert chunk_nodes._CONFIG["effective_mlp_chunk_tokens"] == 512
        assert chunk_nodes._runtime_status_payload("test") == {
            "node_id": None,
            "configured_rope": original_config["chunk_tokens"],
            "effective_rope": original_config["status_effective_chunk_tokens"],
            "configured_mlp": 1024,
            "effective_mlp": 512,
            "configured_qkv": original_config["qkv_chunk_tokens"],
            "effective_qkv": original_config["status_effective_qkv_chunk_tokens"],
            "reason": "test",
        }
    finally:
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_manual_settings_reset_learned_runtime_values():
    original_config = chunk_nodes._CONFIG.copy()
    try:
        chunk_nodes._CONFIG["effective_chunk_tokens"] = 2048
        chunk_nodes._CONFIG["effective_mlp_chunk_tokens"] = 1024
        chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] = 512
        chunk_nodes._configure_runtime(
            3072, 1536, True, False, True, node_id=None,
            qkv_chunk_tokens=2048,
        )
        assert chunk_nodes._CONFIG["chunk_tokens"] == 3072
        assert chunk_nodes._CONFIG["effective_chunk_tokens"] == 3072
        assert chunk_nodes._CONFIG["mlp_chunk_tokens"] == 1536
        assert chunk_nodes._CONFIG["effective_mlp_chunk_tokens"] == 1536
        assert chunk_nodes._CONFIG["qkv_chunk_tokens"] == 2048
        assert chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] == 2048
    finally:
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_qkv_oom_status_uses_qkv_sequence_length():
    original_config = chunk_nodes._CONFIG.copy()
    try:
        chunk_nodes._CONFIG.update(
            qkv_chunk_tokens=4096,
            effective_qkv_chunk_tokens=4096,
            status_effective_qkv_chunk_tokens=4096,
            status_sequence_qkv=3000,
            status_sequence_mlp=700,
            node_id=None,
        )
        chunk_nodes._remember_effective_chunk("QKV", 4096, 2048)
        assert chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] == 2048
        assert chunk_nodes._CONFIG["status_effective_qkv_chunk_tokens"] == 2048
    finally:
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_legacy_node_alias_is_deprecated():
    assert chunk_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ActivationChunkStar7"] is (
        chunk_nodes.MiniMaxH3ActivationChunkStar7
    )
    assert chunk_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3RoPEChunkPatch"] is (
        chunk_nodes.MiniMaxH3RoPEChunkPatch
    )
    assert chunk_nodes.MiniMaxH3RoPEChunkPatch.DEPRECATED is True


def test_comfy_kitchen_int8_attention_forward_cuda():
    if not torch.cuda.is_available():
        return

    import comfy_kitchen
    from comfy.ldm.minimax import model as h3_model

    if not comfy_kitchen.int8_attention_is_available():
        return

    attention = h3_model.Attention(
        hidden=128,
        heads=1,
        head_dim=128,
        eps=1e-6,
        operations=torch.nn,
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    consumable = [x]
    output = chunk_nodes._minimax_ck_int8_attention_forward(
        attention, consumable, rope_freqs=None, transformer_options={}
    )
    torch.cuda.synchronize()
    assert consumable == []
    assert output.shape == x.shape
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_sla_backend_is_strict_and_architecture_checked():
    backend = chunk_nodes._load_sla_backend()
    assert backend.SM75_BACKEND_NAME == "sla_sm75_qk_int8_pv_fp16"
    assert backend.SM75_ALL_INT8_BACKEND_NAME == "sla_sm75_all_int8_experimental"
    assert backend.SM80PLUS_BACKEND_NAME == "sla_sm80+_qk_int8_pv_fp16"
    assert backend.BLOCK_Q == 128
    assert backend.BLOCK_K == 64
    assert backend.DEFAULT_SPARSITY == 0.85
    choices = chunk_nodes.MiniMaxH3ActivationChunkStar7.INPUT_TYPES()["required"][
        "attention_backend"
    ][0]
    assert backend.SM75_BACKEND_NAME in choices
    assert backend.SM75_ALL_INT8_BACKEND_NAME in choices
    assert backend.SM80PLUS_BACKEND_NAME in choices

    original_available = backend.torch.cuda.is_available
    original_capability = backend.torch.cuda.get_device_capability
    original_triton = backend.triton
    original_native_loader = backend._load_sm75_backend
    try:
        backend.torch.cuda.is_available = lambda: True
        backend.torch.cuda.get_device_capability = lambda _device=None: (7, 0)
        try:
            backend.check_runtime_support(
                requested_backend=backend.SM75_BACKEND_NAME
            )
        except backend.SLAUnavailableError as exc:
            message = str(exc)
            assert "SM75" in message
            assert "No fallback" in message
        else:
            raise AssertionError("strict SLA incorrectly accepted SM70")
        backend.torch.cuda.get_device_capability = lambda _device=None: (7, 5)
        try:
            backend.check_runtime_support(
                requested_backend=backend.SM80PLUS_BACKEND_NAME
            )
        except backend.SLAUnavailableError as exc:
            assert "requires SM80" in str(exc)
        else:
            raise AssertionError("SM80+ SLA name incorrectly accepted SM75")

        backend.triton = None
        backend._load_sm75_backend = lambda: SimpleNamespace(
            availability=lambda: (True, "test SM75 native library")
        )
        assert backend.check_runtime_support(
            requested_backend=backend.SM75_BACKEND_NAME
        ) == (7, 5)
        try:
            backend.torch.cuda.get_device_capability = lambda _device=None: (8, 0)
            backend.check_runtime_support(
                requested_backend=backend.SM80PLUS_BACKEND_NAME
            )
        except backend.SLAUnavailableError as exc:
            assert "requires Triton" in str(exc)
        else:
            raise AssertionError("SM80+ SLA incorrectly accepted missing Triton")
    finally:
        backend.torch.cuda.is_available = original_available
        backend.torch.cuda.get_device_capability = original_capability
        backend.triton = original_triton
        backend._load_sm75_backend = original_native_loader


def test_sm75_binary_manifest_payloads():
    root = pathlib.Path(__file__).resolve().parent
    manifest = json.loads(
        (root / "bin" / "sm75_manifest.json").read_text(encoding="utf-8")
    )
    for platform_key in ("windows_x64", "linux_x86_64"):
        entry = manifest[platform_key]
        payload = root / "bin" / entry["file"]
        assert payload.is_file(), f"missing {platform_key} SM75 binary"
        data = payload.read_bytes()
        assert len(data) == entry["size"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_sm75_torch_preprocess_matches_triton():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    backend = chunk_nodes._load_sla_backend()
    if backend.triton is None:
        return
    original_mode = backend._SM75_TORCH_PREPROCESS
    try:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0x75)
        value = torch.randn(
            (1, 2, 257, backend.HEAD_DIM), generator=generator,
            device="cuda", dtype=torch.float16,
        )
        mean = value.mean(dim=-2, keepdim=True, dtype=torch.float32).to(value.dtype)
        backend._SM75_TORCH_PREPROCESS = False
        triton_pool = backend._mean_pool(value, backend.BLOCK_K, mean)
        triton_q, triton_scale = backend._quantize(
            value, backend.BLOCK_K, 1.0, mean
        )
        backend._SM75_TORCH_PREPROCESS = True
        torch_pool = backend._mean_pool(value, backend.BLOCK_K, mean)
        torch_q, torch_scale = backend._quantize(
            value, backend.BLOCK_K, 1.0, mean
        )
        assert torch.equal(triton_q, torch_q)
        assert torch.equal(triton_scale, torch_scale)
        assert torch.allclose(triton_pool, torch_pool, atol=1e-6, rtol=0.0)
    finally:
        backend._SM75_TORCH_PREPROCESS = original_mode


def test_sla_sm75_native_cuda_self_test():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    backend = chunk_nodes._load_sla_backend()
    native = backend._load_sm75_backend()
    available, reason = native.availability()
    assert available, reason
    backend.ensure_self_test(torch.device("cuda"))
    backend.ensure_self_test(torch.device("cuda"), all_int8=True)


def test_sla_sm75_consuming_inputs():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    backend = chunk_nodes._load_sla_backend()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0x7518)
    for all_int8 in (False, True):
        owned = [
            torch.randn(
                (1, 1, 1025, backend.HEAD_DIM), generator=generator,
                device="cuda", dtype=torch.float16,
            ) * 0.1
            for _ in range(3)
        ]
        result = backend.sparse_attention_consume(
            owned, run_self_test=False, all_int8=all_int8
        )
        torch.cuda.synchronize()
        assert owned == []
        assert torch.isfinite(result.output).all()


def test_sla_attention_forward_cuda():
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0) and capability != (7, 5):
        return
    backend = chunk_nodes._load_sla_backend()
    try:
        backend.check_runtime_support()
    except backend.SLAUnavailableError:
        return

    from comfy.ldm.minimax import model as h3_model

    attention = h3_model.Attention(
        hidden=128,
        heads=1,
        head_dim=128,
        eps=1e-6,
        operations=torch.nn,
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(1025, 128, device="cuda", dtype=torch.float16) * 0.1
    consumable = [x]
    output = chunk_nodes._minimax_sla_forward(
        attention, consumable, rope_freqs=None, transformer_options={}
    )
    torch.cuda.synchronize()
    assert consumable == []
    assert output.shape == x.shape
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for test_device in devices:
        test_rope_matches_eager_partial_rotary(test_device)
        test_mlp_chunk_matches_full_forward(test_device)
        test_star7_fp16_mlp_chunk_matches_full_formula(test_device)
        test_auto_fp16_out_proj_uses_exact_scaling(test_device)
        test_qkv_chunk_writes_backend_dtype_without_full_cast(test_device)
        test_star7_resident_weight_path_matches_formula(test_device)
        test_weight_only_quantized_resident_path_stays_quantized(test_device)
        test_fused_residual_matches_materialized_mlp(test_device)
        test_fp16_block_patch_matches_materialized_formula(test_device)
        print(f"MiniMax H3 RoPE/MLP chunk tests passed on {test_device}")
    test_rope_oom_value_is_reused_for_k_and_later_calls()
    test_mlp_oom_value_is_reused_for_later_blocks()
    test_manual_settings_reset_learned_runtime_values()
    test_qkv_oom_status_uses_qkv_sequence_length()
    test_legacy_node_alias_is_deprecated()
    test_dynamic_vbar_linear_can_be_snapshotted_for_resident_reuse()
    test_install_preserves_upstream_block_patch()
    test_sm75_sla_auto_installs_fp16_exact_without_external_node()
    test_comfy_kitchen_int8_attention_forward_cuda()
    test_sla_backend_is_strict_and_architecture_checked()
    test_sm75_binary_manifest_payloads()
    test_sm75_torch_preprocess_matches_triton()
    test_sla_sm75_native_cuda_self_test()
    test_sla_sm75_consuming_inputs()
    test_sla_attention_forward_cuda()
    print("MiniMax H3 prefetch removal compatibility test passed")
