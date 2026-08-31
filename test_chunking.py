import importlib.util
import gc
import hashlib
import json
import pathlib
import sys
from types import MethodType, SimpleNamespace
from unittest import mock
import weakref

import torch


COMFY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

SPEC = importlib.util.spec_from_file_location("h3_chunk_nodes", pathlib.Path(__file__).with_name("nodes.py"))
chunk_nodes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(chunk_nodes)


def test_hybrid_scheduler_supports_arbitrary_step_counts():
    expected = {
        1: "CK",
        2: "CK CK",
        3: "CK SLA CK",
        4: "CK SLA SLA CK",
        5: "CK SLA SLA SLA CK",
        6: "CK SLA SLA SLA SLA CK",
        7: "CK CK SLA SLA SLA CK CK",
        8: "CK CK SLA SLA SLA SLA CK CK",
        9: "CK CK SLA SLA SLA SLA SLA CK CK",
        12: "CK CK SLA SLA SLA SLA SLA SLA SLA SLA CK CK",
    }
    for total_steps, sequence in expected.items():
        actual = " ".join(
            chunk_nodes.select_hybrid_backend(index, total_steps)
            for index in range(total_steps)
        )
        assert actual == sequence

    for total_steps in (20, 50):
        sequence = [
            chunk_nodes.select_hybrid_backend(index, total_steps)
            for index in range(total_steps)
        ]
        assert sequence[0] == sequence[-1] == "CK"
        assert sequence == list(reversed(sequence))
        assert "SLA" in sequence


def test_hybrid_sampler_context_is_step_stable_across_h3_blocks():
    sample_sigmas = torch.tensor([1.0, 0.7, 0.4, 0.0])
    transformer_options = {
        "sample_sigmas": sample_sigmas,
        "sigmas": torch.tensor([1.0]),
    }
    first = chunk_nodes._hybrid_sampling_context(transformer_options)
    assert first == (0, 3, "CK")
    for _ in range(50):
        assert chunk_nodes._hybrid_sampling_context(transformer_options) == first

    transformer_options["sigmas"] = torch.tensor([0.7])
    assert chunk_nodes._hybrid_sampling_context(transformer_options) == (1, 3, "SLA")


def test_hybrid_attention_dispatch_reuses_existing_paths():
    options = {
        "sample_sigmas": torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sigmas": torch.tensor([1.0]),
    }
    with (
        mock.patch.object(chunk_nodes, "_minimax_ck_int8_attention_forward", return_value="ck") as ck,
        mock.patch.object(chunk_nodes, "_minimax_sla_forward", return_value="sla") as sla,
    ):
        assert chunk_nodes._minimax_hybrid_attention_forward(object(), None, transformer_options=options) == "ck"
        ck.assert_called_once()
        sla.assert_not_called()

        options["sigmas"] = torch.tensor([0.7])
        assert chunk_nodes._minimax_hybrid_attention_forward(object(), None, transformer_options=options) == "sla"
        sla.assert_called_once()

    original = chunk_nodes._CONFIG.get("attention_backend")
    try:
        chunk_nodes._CONFIG["attention_backend"] = (
            chunk_nodes.HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME
        )
        options["sigmas"] = torch.tensor([0.7])
        with (
            mock.patch.object(chunk_nodes, "_minimax_sla_forward", return_value="sla") as sla,
            mock.patch.object(chunk_nodes, "_minimax_sol_forward", return_value="sol") as sol,
        ):
            assert chunk_nodes._minimax_hybrid_attention_forward(
                object(), None, transformer_options=options,
            ) == "sol"
            sol.assert_called_once()
            sla.assert_not_called()
    finally:
        chunk_nodes._CONFIG["attention_backend"] = original


def test_hybrid_sampler_context_requires_real_comfy_sigma_metadata():
    try:
        chunk_nodes._hybrid_sampling_context({"sigmas": torch.tensor([1.0])})
    except RuntimeError as exc:
        assert "sample_sigmas" in str(exc)
    else:
        raise AssertionError("Hybrid scheduling must not infer total steps")


def test_hybrid_all_int8_selects_existing_all_int8_sla_path():
    sequence = 129
    qkv = [
        torch.zeros((1, 1, sequence, 128), dtype=torch.float16)
        for _ in range(3)
    ]
    captured = []

    def sparse_attention_consume(owned, **kwargs):
        captured.append(kwargs["all_int8"])
        output = torch.zeros_like(owned[0])
        return SimpleNamespace(
            output=output,
            query_blocks=1,
            key_blocks=1,
            selected_key_blocks=1,
            protected_query_blocks=0,
            effective_sparsity=0.0,
            dense_guard_status="test",
            implementation="test",
        )

    fake_backend = SimpleNamespace(
        sparse_attention_consume=sparse_attention_consume,
    )
    attention = SimpleNamespace(
        heads=1,
        head_dim=128,
        out_proj=lambda value: value,
        _star7_block_index=0,
    )
    original_backend = chunk_nodes._CONFIG.get("attention_backend")
    original_verbose = chunk_nodes._CONFIG.get("verbose")
    try:
        chunk_nodes._CONFIG["verbose"] = False
        input_tokens = torch.zeros((sequence, 128), dtype=torch.float16)
        with (
            mock.patch.object(chunk_nodes, "_load_sla_backend", return_value=fake_backend),
            mock.patch.object(chunk_nodes, "_prepare_h3_qkv_chunked", return_value=tuple(qkv)),
        ):
            chunk_nodes._CONFIG["attention_backend"] = chunk_nodes.HYBRID_ALL_INT8_BACKEND_NAME
            chunk_nodes._minimax_sla_forward(attention, input_tokens)
            chunk_nodes._CONFIG["attention_backend"] = chunk_nodes.HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME
            chunk_nodes._minimax_sla_forward(attention, input_tokens)
            chunk_nodes._CONFIG["attention_backend"] = chunk_nodes.HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME
            chunk_nodes._minimax_sla_forward(attention, input_tokens)
            chunk_nodes._CONFIG["attention_backend"] = "sla_sm75_qk_int8_pv_fp16"
            chunk_nodes._minimax_sla_forward(attention, input_tokens)
        assert captured == [True, True, False, False]
    finally:
        chunk_nodes._CONFIG["attention_backend"] = original_backend
        chunk_nodes._CONFIG["verbose"] = original_verbose


def test_sla_restores_upstream_dtype_before_out_proj():
    sequence = 17
    qkv = tuple(
        torch.zeros((1, 1, sequence, 128), dtype=torch.float16)
        for _ in range(3)
    )
    projected_dtypes = []

    def sparse_attention_consume(_owned, **_kwargs):
        return SimpleNamespace(
            output=torch.full(qkv[0].shape, 500.0, dtype=torch.float16),
            query_blocks=1,
            key_blocks=1,
            selected_key_blocks=1,
            protected_query_blocks=0,
            effective_sparsity=0.0,
            dense_guard_status="test",
            implementation="test",
        )

    def out_proj(value):
        projected_dtypes.append(value.dtype)
        return value

    attention = SimpleNamespace(
        heads=1, head_dim=128, out_proj=out_proj, _star7_block_index=0,
    )
    original_backend = chunk_nodes._CONFIG.get("attention_backend")
    original_verbose = chunk_nodes._CONFIG.get("verbose")
    try:
        chunk_nodes._CONFIG["attention_backend"] = chunk_nodes.SM86PLUS_BACKEND_NAME
        chunk_nodes._CONFIG["verbose"] = False
        with (
            mock.patch.object(
                chunk_nodes, "_load_sla_backend",
                return_value=SimpleNamespace(
                    sparse_attention_consume=sparse_attention_consume,
                ),
            ),
            mock.patch.object(
                chunk_nodes, "_prepare_h3_qkv_chunked", return_value=qkv,
            ),
        ):
            output = chunk_nodes._minimax_sla_forward(
                attention,
                torch.zeros((sequence, 128), dtype=torch.bfloat16),
            )
        assert projected_dtypes == [torch.bfloat16]
        assert output.dtype == torch.bfloat16
    finally:
        chunk_nodes._CONFIG["attention_backend"] = original_backend
        chunk_nodes._CONFIG["verbose"] = original_verbose


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
    mlp._star7_reuse_mlp_input = True
    actual = chunk_nodes._chunked_h3_mlp_forward(mlp, x)

    # Splitting a GEMM can select a different kernel tile, so float32 reduction
    # order may differ in the last bit even though every row's math is unchanged.
    assert actual.data_ptr() == x.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-7)


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


def test_sm75_convrot_qkv_projection_is_bitwise_chunk_invariant():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return

    import comfy_kitchen as ck

    torch.manual_seed(0x75A0)
    hidden, projected, sequence = 256, 768, 1024
    weight = torch.randn(
        projected, hidden, device="cuda", dtype=torch.float16
    )
    qweight, scales = ck.quantize_convrot_w4a4_weight(
        weight, convrot_groupsize=256, stochastic_rounding=0
    )
    value = torch.randn(
        sequence, hidden, device="cuda", dtype=torch.float16
    )
    expected = ck.convrot_w4a4_linear(
        value, qweight, scales,
        convrot_groupsize=256, linear_dtype="int8",
    )
    actual = torch.cat([
        ck.convrot_w4a4_linear(
            value[start:start + 256], qweight, scales,
            convrot_groupsize=256, linear_dtype="int8",
        )
        for start in range(0, sequence, 256)
    ])

    assert torch.equal(actual, expected)


def test_patched_qkv_resident_snapshot_matches_streamed_chunks(
    device=torch.device("cpu"),
):
    import comfy.ops

    torch.manual_seed(0x4096)
    operations = comfy.ops.mixed_precision_ops(compute_dtype=torch.float16)
    linear = operations.Linear(16, 48, bias=False, device=device)
    linear.weight = torch.nn.Parameter(
        torch.randn(48, 16, dtype=torch.float16, device=device) * 0.02,
        requires_grad=False,
    )
    delta = torch.randn_like(linear.weight) * 0.001
    linear.weight_function = [lambda weight: weight + delta]
    linear.bias_function = []
    x = torch.randn(519, 16, dtype=torch.float16, device=device)

    expected = torch.cat([linear(x[start:start + 256]) for start in range(0, 519, 256)])
    resident_call, backend = chunk_nodes._resident_qkv_caller(linear, x)
    actual = torch.cat([
        resident_call(x[start:start + 256]) for start in range(0, 519, 256)
    ])
    assert backend == "dense"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weight_only_quantized_qkv_resident_path_stays_quantized(device=torch.device("cpu")):
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
    assert chunk_nodes._linear_quantization_mode(mlp.fc1) == "weight-only"
    if device.type == "cuda":
        qkv_input = torch.randn(257, 256, dtype=torch.float16, device=device)
        resident_call, qkv_backend = chunk_nodes._resident_qkv_caller(
            mlp.fc1, qkv_input,
        )
        quant_mode = chunk_nodes._linear_quantization_mode(mlp.fc1)
        direct_weight = mlp.fc1.weight.to(dtype=qkv_input.dtype)
        expected_qkv = torch.cat([
            chunk_nodes._resident_linear_forward(
                mlp.fc1, qkv_input[start:start + 128],
                direct_weight, None, quant_mode,
            )
            for start in range(0, 257, 128)
        ])
        actual_qkv = torch.cat([
            resident_call(qkv_input[start:start + 128])
            for start in range(0, 257, 128)
        ])
        assert qkv_backend == "quantized"
        assert torch.equal(actual_qkv, expected_qkv)


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

    class BypassHook:
        def forward(self, value):
            return value

    bypassed = SimpleNamespace(
        weight=object(),
        weight_function=[],
        bias_function=[],
        _forward=lambda *args: None,
        forward=BypassHook().forward,
    )
    assert chunk_nodes._linear_can_reuse_weights(bypassed) is False


def test_chunked_mlp_preserves_external_precision_callable(device=torch.device("cpu")):
    calls = []

    class TinyMLP:
        pass

    def external_fp16_exact(value):
        calls.append(value.shape[0])
        return value.to(torch.float32).mul(2.0)

    value = torch.randn(519, 16, dtype=torch.float16, device=device)
    chunk_nodes._CONFIG.update(
        mlp_chunk_tokens=256,
        effective_mlp_chunk_tokens=256,
        auto_halve_on_oom=False,
        verbose=False,
    )
    actual = chunk_nodes._run_chunked_h3_mlp(
        TinyMLP(), value, upstream_forward=external_fp16_exact
    )
    assert calls == [256, 256, 7]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, value.float().mul(2.0), rtol=0, atol=0)


def test_h3_output_guard_identifies_audio_before_vae():
    def original_forward(*_args, **_kwargs):
        return [torch.zeros(1), torch.tensor([float("nan")])]

    guarded = chunk_nodes._h3_output_finite_passthrough(original_forward)
    try:
        guarded(object())
    except RuntimeError as exc:
        message = str(exc)
        assert "audio model output" in message
        assert "before VAE decode" in message
    else:
        raise AssertionError("invalid H3 audio reached VAE decode")


def test_star7_finite_wrapper_is_idempotent_and_unwraps_legacy_shape():
    def original_forward(*_args, **_kwargs):
        return [torch.zeros(1), torch.zeros(1)]

    first = chunk_nodes._h3_output_finite_passthrough(original_forward)
    second = chunk_nodes._h3_output_finite_passthrough(
        chunk_nodes._star7_wrapper_original(first, "h3-output-finite")
    )
    assert second._star7_original_forward is original_forward

    # Simulate the untagged v2.9.3 closure retained by a cloned ModelPatcher.
    del first._star7_wrapper_kind
    del first._star7_original_forward
    assert chunk_nodes._star7_wrapper_original(
        first, "h3-output-finite"
    ) is original_forward


def test_weak_model_method_does_not_retain_owner():
    class Owner:
        def __init__(self):
            self.value = 3

        def forward(self, value):
            return self.value + value

    owner = Owner()
    method = chunk_nodes._weak_method(owner, Owner.forward)
    owner_ref = weakref.ref(owner)
    assert method(4) == 7

    del owner
    gc.collect()
    assert owner_ref() is None


def test_sla_failure_structure_reports_chunk_rows_and_heads():
    hidden = torch.zeros(10, 7)
    hidden[2:6] = float("nan")
    summary = chunk_nodes._bad_row_ranges(hidden, row_dim=0)
    assert summary["bad_rows"] == 4
    assert summary["first_bad_row"] == 2
    assert summary["last_bad_row"] == 5
    assert summary["ranges"] == ((2, 5),)

    headed = torch.zeros(1, 3, 10, 4)
    headed[:, 1, 7:9] = float("inf")
    summary = chunk_nodes._bad_row_ranges(headed, row_dim=2)
    assert summary["ranges"] == ((7, 8),)
    assert summary["bad_heads"] == (1,)


def test_sla_error_precision_text_is_architecture_specific():
    assert "SM75" in chunk_nodes._sla_architecture_note_for_capability((7, 5))
    sm120 = chunk_nodes._sla_architecture_note_for_capability((12, 0))
    assert "SM120" in sm120
    assert "SM75 FP16 Exact" not in sm120
    assert not chunk_nodes._auto_sla_probe_for_capability((7, 5))
    assert not chunk_nodes._auto_sla_probe_for_capability((8, 9))
    assert chunk_nodes._auto_sla_probe_for_capability((12, 0))


def test_new_architecture_auto_sla_probe_reports_first_bad_stage():
    value = torch.zeros(8, 4)
    value[3] = float("nan")
    original_probe = chunk_nodes._CONFIG["auto_sla_probe"]
    original_block = chunk_nodes._LAST_FAILED_SLA_BLOCK
    try:
        chunk_nodes._CONFIG["auto_sla_probe"] = False
        chunk_nodes._debug_sla_tensor("unarmed", value, block_index=44)

        chunk_nodes._CONFIG["auto_sla_probe"] = True
        try:
            chunk_nodes._debug_sla_tensor("raw SLA output", value, block_index=44)
        except RuntimeError as exc:
            message = str(exc)
            assert "after raw SLA output" in message
            assert "Automatic new-architecture stage diagnostics" in message
        else:
            raise AssertionError("automatic SLA probe did not report a non-finite stage")
    finally:
        chunk_nodes._CONFIG["auto_sla_probe"] = original_probe
        chunk_nodes._LAST_FAILED_SLA_BLOCK = original_block


def test_audio_guard_status_distinguishes_sm75_full_from_sm80_routing_only():
    backend = chunk_nodes._load_sla_backend()
    ranges = ((436, 1416),)
    assert backend._audio_guard_status((), ()) == "not-requested"
    assert backend._audio_guard_status(ranges, ranges) == "full-attention-applied"
    assert backend._audio_guard_status(ranges, ()) == "routing-priority-only"


def test_sm80_audio_query_override_matches_full_attention_for_both_layouts():
    torch.manual_seed(7)
    q = torch.randn(1, 2, 9, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ranges = [(2, 5), (4, 7)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, 2:7], k, v, dropout_p=0.0, is_causal=False,
        scale=8 ** -0.5,
    )

    overrides = chunk_nodes._dense_audio_query_overrides(
        q, k, v, ranges, layout="BHLD",
    )
    output = torch.zeros_like(q)
    chunk_nodes._apply_dense_audio_query_overrides(
        output, overrides, layout="BHLD",
    )
    assert torch.allclose(output[:, :, 2:7], expected, atol=1e-6)
    assert torch.count_nonzero(output[:, :, :2]) == 0

    q_bthd, k_bthd, v_bthd = (
        tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)
    )
    overrides = chunk_nodes._dense_audio_query_overrides(
        q_bthd, k_bthd, v_bthd, ranges, layout="BTHD",
    )
    output_bthd = torch.zeros_like(q_bthd)
    chunk_nodes._apply_dense_audio_query_overrides(
        output_bthd, overrides, layout="BTHD",
    )
    assert torch.allclose(
        output_bthd[:, 2:7].transpose(1, 2), expected, atol=1e-6,
    )


def test_h3_audio_ranges_include_reference_and_generated_audio():
    segments = [
        (0, 4, 0),
        (4, 7, 2),
        (7, 11, torch.tensor(1)),
        (11, 14, torch.tensor([0, 1])),
        (14, 20, 1),
    ]
    assert chunk_nodes._h3_audio_token_ranges(segments) == [(4, 7), (11, 14)]


def test_chunk_contains_no_fp16_exact_repair_implementation():
    assert not hasattr(chunk_nodes, "_fp16_exact_out_proj")
    assert not hasattr(chunk_nodes, "_condition_proj_fp32_forward")
    assert not hasattr(chunk_nodes, "_make_chunked_h3_block_forward")
    original = chunk_nodes._CONFIG.get("fp16_exact_present")
    try:
        chunk_nodes._CONFIG["fp16_exact_present"] = False
        assert "not detected" in chunk_nodes._sla_architecture_note_for_capability((7, 5))
        assert "FP16 Exact" not in chunk_nodes._sla_architecture_note_for_capability((12, 0))
    finally:
        chunk_nodes._CONFIG["fp16_exact_present"] = original


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

    def external_mlp_forward(self, value):
        return value.to(torch.float32)

    external_mlp_patch = MethodType(
        external_mlp_forward, diffusion_model.blocks[0].mlp
    )

    class FakeModelPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {
                    "star7_minimax_h3_fp16_exact_fix": "test",
                }
            }
            self.object_patches = {
                "diffusion_model.blocks.0.forward": upstream_block_patch,
                "diffusion_model.blocks.0.mlp.forward": external_mlp_patch,
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
    wrapped_mlp = patched.object_patches[
        "diffusion_model.blocks.0.mlp.forward"
    ].__func__
    # The upstream behavior is preserved through a weak callable. Keeping the
    # original bound method object here would retain the MLP and its H3 model.
    assert callable(wrapped_mlp._star7_original_forward)
    assert wrapped_mlp._star7_original_forward is not external_mlp_patch
    assert not isinstance(wrapped_mlp._star7_original_forward, MethodType)
    assert patched.wrappers == {}

    patched = chunk_nodes.install_model_patch(
        patched,
        chunk_tokens=8192,
        auto_halve_on_oom=True,
        verbose=False,
        mlp_chunk_tokens=4096,
        disable_dynamic_prefetch=True,
        reuse_mlp_weights=True,
        attention_backend="existing",
    )
    guarded = patched.object_patches["diffusion_model.forward"].__func__
    assert guarded._star7_wrapper_kind == "h3-output-finite"
    upstream = getattr(
        guarded._star7_original_forward, "__func__", guarded._star7_original_forward
    )
    assert getattr(upstream, "_star7_wrapper_kind", None) is None


def test_sm75_sla_does_not_install_fp16_exact_without_companion():
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
    assert patched.compute_dtype is None
    assert "star7_h3_sm75_auto_fp16_exact" not in patched.model_options[
        "transformer_options"
    ]
    assert patched.model_options["transformer_options"][
        "star7_h3_output_finite_guard"
    ] == chunk_nodes.NODE_VERSION
    assert "diffusion_model.condition_proj.forward" not in patched.object_patches
    assert "diffusion_model.forward" in patched.object_patches
    assert "diffusion_model.blocks.0.forward" in patched.object_patches
    assert not hasattr(diffusion_model.blocks[0].attn, "_star7_auto_fp16_exact")


def test_sm75_ck_keeps_precision_external_with_block_loop_cache():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    import comfy_kitchen
    from comfy.ldm.minimax import model as h3_model

    if not comfy_kitchen.int8_attention_is_available():
        return
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
    block_cache = object()

    class FakeModelPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {
                    "patches_replace": {"dit": {("block_loop", 0): block_cache}}
                }
            }
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
        attention_backend="comfy_kitchen_int8",
    )
    assert patched.compute_dtype is None
    assert patched.model_options["transformer_options"]["patches_replace"][
        "dit"
    ][("block_loop", 0)] is block_cache
    assert "diffusion_model.forward" in patched.object_patches
    assert "diffusion_model.blocks.0.forward" not in patched.object_patches
    assert "diffusion_model.blocks.0.attn.forward" in patched.object_patches
    assert not hasattr(diffusion_model.blocks[0].attn, "_star7_auto_fp16_exact")


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

    def upstream_forward(tensor):
        expanded = mlp.fc1(tensor)
        gate, up = expanded.chunk(2, dim=-1)
        return mlp.fc2(torch.nn.functional.silu(gate).mul(up))

    try:
        chunk_nodes._CONFIG.update(
            mlp_chunk_tokens=1024,
            effective_mlp_chunk_tokens=1024,
            auto_halve_on_oom=True,
            verbose=False,
            reuse_mlp_weights=False,
            node_id=None,
        )
        chunk_nodes._run_chunked_h3_mlp(
            mlp, value, upstream_forward=upstream_forward
        )
        first_call_attempts = attempts.copy()
        chunk_nodes._run_chunked_h3_mlp(
            mlp, value, upstream_forward=upstream_forward
        )

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


def test_qkv_zero_prefers_full_then_learns_oom_chunk():
    attempts = []
    original_config = chunk_nodes._CONFIG.copy()

    class LimitedQKV:
        weight = torch.empty(24, 8)

        def __call__(self, tensor):
            attempts.append(int(tensor.shape[0]))
            if tensor.shape[0] > 512:
                raise RuntimeError("out of memory")
            return torch.zeros(tensor.shape[0], 24, dtype=tensor.dtype)

    attention = SimpleNamespace(
        heads=2,
        head_dim=4,
        qkv_proj=LimitedQKV(),
        q_norm=lambda value: value,
        k_norm=lambda value: value,
    )
    x = torch.zeros(1024, 8)

    try:
        chunk_nodes._CONFIG.update(
            qkv_chunk_tokens=0,
            effective_qkv_chunk_tokens=0,
            status_effective_qkv_chunk_tokens=0,
            auto_halve_on_oom=True,
            verbose=False,
            node_id=None,
        )
        q, k, v = chunk_nodes._prepare_h3_qkv_chunked(
            attention,
            x,
            None,
            SimpleNamespace(),
            SimpleNamespace(ck=SimpleNamespace(rms_rope_split_half_=None)),
        )

        assert attempts == [1024, 512, 512]
        assert q.shape == k.shape == v.shape == (1, 2, 1024, 4)
        assert chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] == 512
        assert chunk_nodes._CONFIG["status_effective_qkv_chunk_tokens"] == 512
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


def test_new_activation_chunk_defaults_are_architecture_safe():
    required = chunk_nodes.MiniMaxH3ActivationChunkStar7.INPUT_TYPES()["required"]
    assert required["chunk_tokens"][1]["default"] == 8192
    assert required["mlp_chunk_tokens"][1]["default"] == 8192
    assert required["qkv_chunk_tokens"][1]["default"] == 8192


def test_sm75_qkv_resident_reuse_supports_configured_tiles():
    original_capability = torch.cuda.get_device_capability
    original_config = dict(chunk_nodes._CONFIG)
    fake_cuda_tensor = SimpleNamespace(device=torch.device("cuda"))
    try:
        torch.cuda.get_device_capability = lambda _device=None: (7, 5)
        for effective in (8192, 4096, 2048):
            chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] = effective
            assert chunk_nodes._sm75_qkv_reuse_path(fake_cuda_tensor) is True
        chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] = 0
        assert chunk_nodes._sm75_qkv_reuse_path(fake_cuda_tensor) is False
        torch.cuda.get_device_capability = lambda _device=None: (8, 0)
        chunk_nodes._CONFIG["effective_qkv_chunk_tokens"] = 4096
        assert chunk_nodes._sm75_qkv_reuse_path(fake_cuda_tensor) is False
    finally:
        torch.cuda.get_device_capability = original_capability
        chunk_nodes._CONFIG.clear()
        chunk_nodes._CONFIG.update(original_config)


def test_reference_video_loader_is_registered():
    assert chunk_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ReferenceVideoLoadStar7"] is (
        chunk_nodes.MiniMaxH3ReferenceVideoLoadStar7
    )


def test_reference_video_long_edge_limit_preserves_orientation():
    assert chunk_nodes._long_edge_reference_size(720, 1280, 1056) == (608, 1056)
    assert chunk_nodes._long_edge_reference_size(1920, 1080, 1056) == (1056, 608)


def test_reference_video_long_edge_upscale_is_explicit():
    assert chunk_nodes._long_edge_reference_size(360, 640, 1056, False) == (352, 640)
    assert chunk_nodes._long_edge_reference_size(360, 640, 1056, True) == (608, 1056)


def test_reference_image_uses_long_edge_controls():
    inputs = chunk_nodes.MiniMaxH3LoadImageScaleStar7.INPUT_TYPES()["required"]
    assert "最长边" in inputs
    assert "允许小图放大" in inputs
    assert "缩放算法" not in inputs
    assert "scale_by" not in inputs
    assert inputs["最长边"][1]["default"] == 1280
    assert inputs["最长边"][1]["min"] == 0
    assert inputs["允许小图放大"][1]["default"] is False
    assert chunk_nodes._normalize_reference_max_long_edge(0) == 0
    assert chunk_nodes._normalize_reference_max_long_edge(1) == 1024
    assert chunk_nodes._normalize_reference_max_long_edge(1024) == 1024


def test_reference_video_frame_count_is_h3_aligned():
    assert chunk_nodes._align_h3_reference_frame_count(360) == 345
    assert chunk_nodes._align_h3_reference_frame_count(480) == 464
    assert chunk_nodes._align_h3_reference_frame_count(205) == 192
    assert chunk_nodes._align_h3_reference_frame_count(5) == 5


def test_reference_video_trim_defaults_and_window_are_compact():
    inputs = chunk_nodes.MiniMaxH3ReferenceVideoLoadStar7.INPUT_TYPES()["required"]
    assert inputs["trim_enabled"][1]["default"] is False
    assert inputs["trim_start_seconds"][1]["default"] == 0.0
    assert inputs["trim_end_seconds"][1]["default"] == 0.0
    assert inputs["trim_end_seconds"][1]["max"] == 86400.0
    assert inputs["max_long_edge"][1]["default"] == 720
    assert inputs["max_long_edge"][1]["min"] == 0
    assert chunk_nodes._normalize_reference_trim(False, 7.0, 9.0, 20.2) == (0.0, 20.2)
    assert chunk_nodes._normalize_reference_trim(True, 7.0, 9.0, 20.2) == (7.0, 2.0)
    assert chunk_nodes._normalize_reference_trim(True, -2.0, 99.0, 8.0) == (0.0, 8.0)
    start, duration = chunk_nodes._normalize_reference_trim(True, 12.0, 20.2, 20.2)
    assert start == 12.0
    assert abs(duration - 8.2) < 1e-6


def test_reference_video_and_audio_share_the_same_trim_window():
    args = chunk_nodes._reference_media_input_args("reference.mp4", 4.2, 9.0)
    assert args == ["-ss", "4.2", "-i", "reference.mp4", "-t", "9"]
    command = chunk_nodes._reference_audio_decode_command(
        "ffmpeg", "reference.mp4", 44100, 4.2, 9.0,
    )
    assert command[command.index("-ss") + 1] == "4.2"
    assert command[command.index("-t") + 1] == "9"
    assert command.index("-ss") < command.index("-i")


def test_reference_audio_keeps_source_duration_instead_of_frame_grid_trim():
    command = chunk_nodes._reference_audio_decode_command(
        "ffmpeg", "reference.mp4", 44100,
    )
    assert "-t" not in command
    assert command[command.index("-ar") + 1] == "44100"
    assert "17n+5" not in " ".join(command)


def test_pruned_h3_lora_curve_preserves_full_width_adaln_delta():
    table = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    egrid = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 8.0],
        [4.0, 8.0, 12.0, 16.0],
    ])
    compressed = torch.tensor([[0.25, 0.5], [1.5, 3.0]])
    expected_rows = torch.tensor([
        [0.5, 1.0, 1.5, 2.0],
        [3.0, 6.0, 9.0, 12.0],
    ])
    rows = chunk_nodes._curve_silu_rows(compressed, table, egrid, {})
    assert torch.allclose(rows, expected_rows, atol=1e-5)

    up = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
    down = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    forward = chunk_nodes._make_pruned_adaln_lora_forward(
        lambda x: (torch.zeros(x.shape[0], 2), torch.zeros(x.shape[0], 2)),
        [(up, down, 0.7)], table, egrid, {}, 1, 2, 2,
    )
    actual = torch.cat(forward(None, compressed), dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(expected_rows, down), up,
    ) * 0.7
    assert torch.allclose(actual, expected, atol=1e-5)


def test_pruned_h3_lora_adapter_rejects_native_eight_wide_adaln_patch():
    import comfy.weight_adapter

    up = torch.zeros(24, 2)
    full_down = torch.zeros(2, 2688)
    native_down = torch.zeros(2, 8)

    def patch_for(down):
        adapter = comfy.weight_adapter.LoRAAdapter(
            set(), (up, down, None, None, None, None),
        )
        return (1.0, adapter, 1.0, None, None)

    assert chunk_nodes._pruned_adaln_lora_contribution(
        patch_for(full_down), full_input_features=2688, output_features=24,
    ) is not None
    assert chunk_nodes._pruned_adaln_lora_contribution(
        patch_for(native_down), full_input_features=2688, output_features=24,
    ) is None


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


def test_sm80_h3_rejects_upstream_fp16_compute():
    class FakeModel:
        def __init__(self, dtype):
            self.dtype = dtype
            self.model_options = {"transformer_options": {}}

        def get_model_object(self, name):
            assert name == "manual_cast_dtype"
            return self.dtype

    try:
        chunk_nodes._validate_sm80_h3_compute_dtype(
            FakeModel(torch.float16), (8, 9)
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "Upstream precision configuration error" in message
        assert "The GPU supports FP16" in message
        assert "分块节点尚未修改模型" in message
        assert "请自行检查启动器参数" in message
    else:
        raise AssertionError("SM80+ accepted unsupported FP16 H3 compute")

    chunk_nodes._validate_sm80_h3_compute_dtype(FakeModel(torch.bfloat16), (8, 9))
    chunk_nodes._validate_sm80_h3_compute_dtype(FakeModel(torch.float16), (7, 5))

    protected = FakeModel(torch.float16)
    protected.model_options["transformer_options"][
        chunk_nodes.FP16_EXACT_PATCH_FLAG
    ] = "test"
    chunk_nodes._validate_sm80_h3_compute_dtype(protected, (8, 9))


def test_sla_backend_is_strict_and_architecture_checked():
    backend = chunk_nodes._load_sla_backend()
    sol = chunk_nodes._load_sol_backend()
    assert backend.SM75_BACKEND_NAME == "sla_sm75_qk_int8_pv_fp16"
    assert backend.SM75_ALL_INT8_BACKEND_NAME == "sla_sm75_all_int8"
    assert backend.SM86PLUS_BACKEND_NAME == "sla_sm80+_qk_int8_pv_bf16"
    assert backend.SM86PLUS_ALL_INT8_BACKEND_NAME == "sla_sm80+_all_int8"
    assert backend.SM80PLUS_BACKEND_NAME == "sla_sm80+_qk_int8_pv_fp16"
    assert backend.BLOCK_Q == 128
    assert backend.BLOCK_K == 64
    assert backend.DEFAULT_SPARSITY == 0.85
    choices = chunk_nodes.MiniMaxH3ActivationChunkStar7.INPUT_TYPES()["required"][
        "attention_backend"
    ][0]
    assert chunk_nodes.HYBRID_ALL_INT8_BACKEND_NAME == "hybrid_sm75_ck_sla_all_int8"
    assert chunk_nodes.HYBRID_ALL_INT8_BACKEND_NAME in choices
    expected_sm75 = [
        backend.SM75_BACKEND_NAME,
        backend.SM75_ALL_INT8_BACKEND_NAME,
        sol.SOL_SM75_ALL_INT8_BACKEND_NAME,
        chunk_nodes.HYBRID_ALL_INT8_BACKEND_NAME,
        chunk_nodes.HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
    ]
    expected_sm80plus = [
        backend.SM86PLUS_BACKEND_NAME,
        backend.SM86PLUS_ALL_INT8_BACKEND_NAME,
        sol.SOL_SM86PLUS_BACKEND_NAME,
        sol.SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
        chunk_nodes.HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME,
        chunk_nodes.HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
    ]
    assert choices == [
        "existing", "comfy_kitchen_int8", *expected_sm75, *expected_sm80plus,
    ]
    assert backend.SM75_BACKEND_NAME in choices
    assert backend.SM86PLUS_BACKEND_NAME in choices
    assert sol.SOL_SM86PLUS_BACKEND_NAME in choices
    assert chunk_nodes.HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME not in choices
    assert chunk_nodes.HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME not in choices
    assert sol.SOL_SM75_BACKEND_NAME not in choices
    assert sol.SOL_SM75_ALL_INT8_BACKEND_NAME == "sol_sm75_all_int8"
    assert sol.SOL_SM86PLUS_ALL_INT8_BACKEND_NAME == "sol_sm80+_all_int8"

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

        backend.triton = object()
        backend.torch.cuda.get_device_capability = lambda _device=None: (8, 0)
        try:
            backend.check_runtime_support(
                requested_backend=backend.SM75_BACKEND_NAME
            )
        except backend.SLAUnavailableError as exc:
            assert "requires exactly SM75" in str(exc)
        else:
            raise AssertionError("SM75 Hybrid SLA incorrectly accepted SM80")
        assert backend.check_runtime_support(
            requested_backend=backend.SM86PLUS_BACKEND_NAME
        ) == (8, 0)
        for capability in ((8, 0), (8, 6), (8, 9), (12, 0)):
            backend.torch.cuda.get_device_capability = (
                lambda _device=None, cap=capability: cap
            )
            assert backend.check_runtime_support(
                requested_backend=backend.SM80PLUS_BACKEND_NAME
            ) == capability
            assert backend.backend_name_for_capability(capability) == backend.SM86PLUS_BACKEND_NAME
            assert backend.check_runtime_support(
                requested_backend=backend.SM86PLUS_BACKEND_NAME
            ) == capability
            assert backend.check_runtime_support(
                requested_backend=backend.SM86PLUS_ALL_INT8_BACKEND_NAME
            ) == capability
    finally:
        backend.torch.cuda.is_available = original_available
        backend.torch.cuda.get_device_capability = original_capability
        backend.triton = original_triton
        backend._load_sm75_backend = original_native_loader


def test_sol_q64k64_routing_has_variable_row_counts():
    sol = chunk_nodes._load_sol_backend()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x501)
    q = torch.randn((1, 2, 1025, 128), generator=generator, dtype=torch.float16)
    k = torch.randn(q.shape, generator=generator, dtype=torch.float16)
    row_count, lut, density = sol.build_custom_routing(
        q, k, tau=0.25, topk_blocks=4,
    )
    assert row_count.shape == (1, 2, 17)
    assert lut.shape[:3] == row_count.shape
    assert row_count.dtype == torch.int32
    assert lut.dtype == torch.int32
    assert int(row_count.min()) < int(row_count.max())
    assert 0.0 < density <= 1.0
    for row, packed in zip(row_count.flatten(), lut.reshape(-1, lut.shape[-1])):
        selected = packed[: int(row)]
        assert bool((selected[1:] >= selected[:-1]).all())
        assert int(selected.min()) >= 0
        assert int(selected.max()) < 17


def test_bundled_official_sol_dispatch_is_self_contained():
    sol = chunk_nodes._load_sol_backend()
    official = sol._official_module()
    assert "vendor" in pathlib.Path(official.__file__).parts
    assert official._backend_for_arch((8, 0), cute_available=False) == "triton"
    assert official._backend_for_arch((8, 6), cute_available=False) == "triton"
    assert official._backend_for_arch((8, 9), cute_available=False) == "triton"
    assert official._backend_for_arch((8, 9), cute_available=True) == "cute_sm89"
    assert official._backend_for_arch((9, 0), cute_available=True) == "cute_sm90"
    assert official._backend_for_arch((10, 0), cute_available=True) == "cute_sm100"
    assert official._backend_for_arch((12, 0), cute_available=True) == "cute_sm120"


def test_sol_sm75_native_cuda_matches_exact_plus_centroid_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    sol = chunk_nodes._load_sol_backend()
    available, _reason = sol._load_sm75_backend().availability()
    if not available:
        return
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0x507)
    q = torch.randn((1, 1, 1025, 128), generator=generator, device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn(q.shape, generator=generator, device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn(q.shape, generator=generator, device="cuda", dtype=torch.float16) * 0.2
    native_routing = sol._load_sm75_backend().prepare(q, k, v, tau=0.25)
    row_count = native_routing["row_count"]
    exact_mask = native_routing["exact_mask"]
    exact_approx_parts = []
    scale = 128 ** -0.5
    k_centroid = sol._block_mean_fp32(k)
    v_centroid = sol._block_mean_fp32(v)
    for query_block in range(row_count.shape[-1]):
        blocks = torch.nonzero(
            exact_mask[0, 0, query_block], as_tuple=False,
        ).flatten().tolist()
        indices = torch.cat([
            torch.arange(
                block * 64, min((block + 1) * 64, q.shape[-2]), device="cuda",
            )
            for block in blocks
        ])
        q_part = q[:, :, query_block * 64 : min((query_block + 1) * 64, q.shape[-2])].float()
        k_part = k.index_select(-2, indices).float()
        v_part = v.index_select(-2, indices).float()
        unselected = [block for block in range(row_count.shape[-1]) if block not in blocks]
        approximate_k = []
        approximate_v = []
        for block in unselected:
            block_length = min(64, q.shape[-2] - block * 64)
            approximate_k.append(k_centroid[:, :, block:block + 1].expand(-1, -1, block_length, -1))
            approximate_v.append(v_centroid[:, :, block:block + 1].expand(-1, -1, block_length, -1))
        if approximate_k:
            k_part = torch.cat([k_part, *approximate_k], dim=-2)
            v_part = torch.cat([v_part, *approximate_v], dim=-2)
        exact_approx_parts.append(
            (torch.softmax(q_part @ k_part.transpose(-1, -2) * scale, dim=-1) @ v_part).half()
        )
    exact_approx_reference = torch.cat(exact_approx_parts, dim=-2)
    fp16_pv = sol.run_custom_consume(
        [q.clone(), k.clone(), v.clone()], all_int8=False,
        tau=0.25,
    ).output
    all_int8 = sol.run_custom_consume(
        [q.clone(), k.clone(), v.clone()], all_int8=True,
        tau=0.25,
    ).output
    torch.cuda.synchronize()
    assert bool(torch.isfinite(fp16_pv).all())
    assert bool(torch.isfinite(all_int8).all())
    assert float((fp16_pv - exact_approx_reference).abs().float().mean()) < 2.0e-4
    assert float((all_int8 - exact_approx_reference).abs().float().mean()) < 1.0e-3


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
    test_hybrid_scheduler_supports_arbitrary_step_counts()
    test_hybrid_sampler_context_is_step_stable_across_h3_blocks()
    test_hybrid_attention_dispatch_reuses_existing_paths()
    test_hybrid_sampler_context_requires_real_comfy_sigma_metadata()
    test_hybrid_all_int8_selects_existing_all_int8_sla_path()
    test_sla_restores_upstream_dtype_before_out_proj()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for test_device in devices:
        test_rope_matches_eager_partial_rotary(test_device)
        test_mlp_chunk_matches_full_forward(test_device)
        test_qkv_chunk_writes_backend_dtype_without_full_cast(test_device)
        test_patched_qkv_resident_snapshot_matches_streamed_chunks(test_device)
        test_weight_only_quantized_qkv_resident_path_stays_quantized(test_device)
        test_chunked_mlp_preserves_external_precision_callable(test_device)
        print(f"MiniMax H3 RoPE/MLP chunk tests passed on {test_device}")
    test_h3_output_guard_identifies_audio_before_vae()
    test_star7_finite_wrapper_is_idempotent_and_unwraps_legacy_shape()
    test_sla_failure_structure_reports_chunk_rows_and_heads()
    test_sla_error_precision_text_is_architecture_specific()
    test_audio_guard_status_distinguishes_sm75_full_from_sm80_routing_only()
    test_chunk_contains_no_fp16_exact_repair_implementation()
    test_rope_oom_value_is_reused_for_k_and_later_calls()
    test_mlp_oom_value_is_reused_for_later_blocks()
    test_manual_settings_reset_learned_runtime_values()
    test_qkv_oom_status_uses_qkv_sequence_length()
    test_qkv_zero_prefers_full_then_learns_oom_chunk()
    test_legacy_node_alias_is_deprecated()
    test_new_activation_chunk_defaults_are_architecture_safe()
    test_sm75_qkv_resident_reuse_supports_configured_tiles()
    test_reference_video_loader_is_registered()
    test_reference_video_long_edge_limit_preserves_orientation()
    test_reference_video_long_edge_upscale_is_explicit()
    test_reference_image_uses_long_edge_controls()
    test_reference_video_frame_count_is_h3_aligned()
    test_reference_video_trim_defaults_and_window_are_compact()
    test_reference_video_and_audio_share_the_same_trim_window()
    test_reference_audio_keeps_source_duration_instead_of_frame_grid_trim()
    test_pruned_h3_lora_curve_preserves_full_width_adaln_delta()
    test_dynamic_vbar_linear_can_be_snapshotted_for_resident_reuse()
    test_install_preserves_upstream_block_patch()
    test_sm75_sla_does_not_install_fp16_exact_without_companion()
    test_sm75_ck_keeps_precision_external_with_block_loop_cache()
    test_sm75_convrot_qkv_projection_is_bitwise_chunk_invariant()
    test_comfy_kitchen_int8_attention_forward_cuda()
    test_sm80_h3_rejects_upstream_fp16_compute()
    test_sla_backend_is_strict_and_architecture_checked()
    test_sol_q64k64_routing_has_variable_row_counts()
    test_bundled_official_sol_dispatch_is_self_contained()
    test_sol_sm75_native_cuda_matches_exact_plus_centroid_reference()
    test_sm75_binary_manifest_payloads()
    test_sm75_torch_preprocess_matches_triton()
    test_sla_sm75_native_cuda_self_test()
    test_sla_sm75_consuming_inputs()
    test_sla_attention_forward_cuda()
    print("MiniMax H3 prefetch removal compatibility test passed")
