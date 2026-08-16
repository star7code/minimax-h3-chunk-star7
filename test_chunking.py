import importlib.util
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


def test_prefetch_wrapper_forces_off():
    captured = {}

    def executor(*args, **kwargs):
        captured.update(kwargs["transformer_options"])
        return "ok"

    result = chunk_nodes._disable_h3_dynamic_prefetch_wrapper(
        executor, transformer_options={"prefetch_dynamic_vbars": True, "keep": 1}
    )
    assert result == "ok"
    assert captured == {"prefetch_dynamic_vbars": False, "keep": 1}


def test_adaptive_prefetch_retries_without_overlap():
    calls = []

    def executor(*args, **kwargs):
        calls.append(kwargs["transformer_options"].copy())
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return "recovered"

    result = chunk_nodes._adaptive_h3_dynamic_prefetch_wrapper(
        executor,
        transformer_options={"prefetch_dynamic_vbars": True, "keep": 1},
    )
    assert result == "recovered"
    assert calls == [
        {"prefetch_dynamic_vbars": True, "keep": 1},
        {"prefetch_dynamic_vbars": False, "keep": 1},
    ]


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
            "effective_rope": original_config["effective_chunk_tokens"],
            "configured_mlp": 1024,
            "effective_mlp": 512,
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
        chunk_nodes._configure_runtime(
            3072, 1536, True, False, True, node_id=None
        )
        assert chunk_nodes._CONFIG["chunk_tokens"] == 3072
        assert chunk_nodes._CONFIG["effective_chunk_tokens"] == 3072
        assert chunk_nodes._CONFIG["mlp_chunk_tokens"] == 1536
        assert chunk_nodes._CONFIG["effective_mlp_chunk_tokens"] == 1536
    finally:
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


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


if __name__ == "__main__":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for test_device in devices:
        test_rope_matches_eager_partial_rotary(test_device)
        test_mlp_chunk_matches_full_forward(test_device)
        test_star7_fp16_mlp_chunk_matches_full_formula(test_device)
        test_star7_resident_weight_path_matches_formula(test_device)
        test_weight_only_quantized_resident_path_stays_quantized(test_device)
        test_fused_residual_matches_materialized_mlp(test_device)
        test_fp16_block_patch_matches_materialized_formula(test_device)
        print(f"MiniMax H3 RoPE/MLP chunk tests passed on {test_device}")
    test_prefetch_wrapper_forces_off()
    test_adaptive_prefetch_retries_without_overlap()
    test_rope_oom_value_is_reused_for_k_and_later_calls()
    test_mlp_oom_value_is_reused_for_later_blocks()
    test_manual_settings_reset_learned_runtime_values()
    test_comfy_kitchen_int8_attention_forward_cuda()
    print("MiniMax H3 dynamic prefetch wrapper test passed")
