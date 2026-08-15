import gc
import contextlib
import logging
import time
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F

_LOG = logging.getLogger("MiniMaxH3ActivationChunkStar7")
NODE_VERSION = "2.1.1"

_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE = None
_PATCHED_CK = None
_LOGGED_ROPE_SHAPES = set()
_LOGGED_MLP_SHAPES = set()
_PROFILED_MLP_SHAPES = set()
_PROFILED_ATTENTION_SHAPES = set()
_CONFIG = {
    "chunk_tokens": 4096,
    "mlp_chunk_tokens": 4096,
    "auto_halve_on_oom": True,
    "verbose": True,
    "reuse_mlp_weights": True,
}


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "would exceed allowed memory" in msg


def _clear_cuda_after_oom(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _slice_freqs_for_tokens(freqs_cis: torch.Tensor, start: int, end: int, seq_len: int) -> torch.Tensor:
    """MiniMax H3 uses freqs shaped [1, S, 1, rot/2, 2, 2].

    We deliberately only patch when the sequence axis is unambiguous. This keeps
    the patch H3-specific and avoids silently changing unrelated models.
    """
    if freqs_cis.ndim < 2 or freqs_cis.shape[1] != seq_len:
        raise ValueError(
            f"unexpected H3 RoPE frequency shape {tuple(freqs_cis.shape)} for sequence length {seq_len}"
        )
    return freqs_cis[:, start:end, ...]


def _apply_split_half_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # Algebra intentionally mirrors comfy-kitchen eager apply_rope_split_half1.
    t = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    if t.dtype != freqs_cis.dtype:
        t = t.to(freqs_cis.dtype)
    out = freqs_cis[..., 0] * t[..., 0] + freqs_cis[..., 1] * t[..., 1]
    return out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def _rms_rope_one_chunk_inplace(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    rot_dim: int,
) -> None:
    # RMSNorm reduces only over head_dim, so token chunking is mathematically
    # independent: no temporal/spatial context is changed here.
    x_norm = F.rms_norm(x, (x.shape[-1],), weight=scale, eps=epsilon)

    effective_rot_dim = rot_dim if rot_dim else x.shape[-1]
    if effective_rot_dim < 0 or effective_rot_dim > x.shape[-1] or effective_rot_dim % 2:
        raise ValueError(
            f"invalid rot_dim={rot_dim} for head_dim={x.shape[-1]}"
        )

    if effective_rot_dim:
        prefix = x_norm[..., :effective_rot_dim]
        rotated = _apply_split_half_rope(prefix, freqs_cis)
        # Reuse x_norm as the chunk output buffer to avoid torch.cat / another
        # full chunk allocation for partial rotary.
        x_norm[..., :effective_rot_dim].copy_(rotated)
        del rotated, prefix

    x.copy_(x_norm)
    del x_norm


def _chunked_rms_rope_split_half_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
):
    global _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE

    if k_scale is None:
        k_scale = q_scale

    # This patch is intentionally narrow: MiniMax H3 passes q/k as [1,S,H,D]
    # with freqs [1,S,1,R/2,2,2]. Anything else goes back to the original op.
    compatible = (
        q.ndim == 4
        and k.ndim == 4
        and q.shape[0] == 1
        and k.shape[0] == 1
        and q.shape[1] == k.shape[1]
        and q.shape[-1] == k.shape[-1]
        and freqs_cis.ndim >= 2
        and freqs_cis.shape[1] == q.shape[1]
        and q.device == k.device
    )
    if not compatible:
        if _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE is None:
            raise RuntimeError("MiniMax H3 RoPE chunk patch has no original fallback")
        return _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE(
            q, k, freqs_cis, q_scale, k_scale, epsilon=epsilon, rot_dim=rot_dim
        )

    seq_len = q.shape[1]
    preferred = max(256, int(_CONFIG["chunk_tokens"]))
    chunk = min(preferred, seq_len)
    auto_halve = bool(_CONFIG["auto_halve_on_oom"])

    shape_key = (
        seq_len, q.shape[2], q.shape[3], rot_dim, chunk,
        q.dtype, q.device.type,
    )
    if _CONFIG["verbose"] and shape_key not in _LOGGED_ROPE_SHAPES:
        _LOGGED_ROPE_SHAPES.add(shape_key)
        _LOG.info(
            "[Star7 H3 Chunk] RoPE active | S=%d | heads=%d | head_dim=%d | "
            "rot_dim=%d | chunk=%d | dtype=%s",
            seq_len, q.shape[2], q.shape[3], rot_dim, chunk, q.dtype,
        )

    # q and k are disjoint views of the qkv projection buffer in ComfyUI H3.
    # Process one tensor and one token chunk at a time, writing back in place.
    for tensor, scale, label in ((q, q_scale, "q"), (k, k_scale, "k")):
        start = 0
        current_chunk = chunk
        while start < seq_len:
            end = min(start + current_chunk, seq_len)
            freq_chunk = _slice_freqs_for_tokens(freqs_cis, start, end, seq_len)
            try:
                _rms_rope_one_chunk_inplace(
                    tensor[:, start:end, ...],
                    freq_chunk,
                    scale,
                    epsilon,
                    rot_dim,
                )
                start = end
            except Exception as exc:
                if not (_is_cuda_oom(exc) and auto_halve and current_chunk > 256):
                    raise
                new_chunk = max(256, current_chunk // 2)
                _LOG.warning(
                    "H3 RoPE %s chunk OOM at %d tokens; retrying current slice with %d tokens",
                    label, current_chunk, new_chunk,
                )
                # Drop traceback-held chunk temporaries before empty_cache/retry.
                exc.__traceback__ = None
                _clear_cuda_after_oom(tensor.device)
                current_chunk = new_chunk

    return q, k


def _linear_can_reuse_weights(linear) -> bool:
    """Whether a ComfyUI Linear exposes the primitives needed for one cast per MLP."""
    return all(
        hasattr(linear, name)
        for name in ("weight", "weight_function", "bias_function", "_forward")
    )


def _linear_quantization_mode(linear) -> Optional[str]:
    """Return the active MixedPrecisionOps inference mode for one Linear."""
    import comfy.ops

    weight = linear.weight
    use_quantized = bool(
        getattr(linear, "layout_type", None) is not None
        and not getattr(linear, "_full_precision_mm", False)
        and not getattr(linear, "comfy_force_cast_weights", False)
        and len(linear.weight_function) == 0
        and len(linear.bias_function) == 0
        and isinstance(weight, comfy.ops.QuantizedTensor)
    )
    if not use_quantized:
        return None
    quantize_input = comfy.ops.QUANT_ALGOS.get(
        getattr(linear, "quant_format", None), {}
    ).get("quantize_input", True)
    return "input-and-weight" if quantize_input else "weight-only"


def _resident_linear_forward(linear, x, weight, bias, quant_mode):
    """Run one Linear with already prepared weights, matching ComfyUI inference."""
    import comfy.ops
    import comfy.model_management as model_management

    comfy.ops.run_every_op()

    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        pre_quant_scale = model_management.cast_to_device(
            pre_quant_scale, x.device, x.dtype
        )
        x = x * pre_quant_scale

    if quant_mode == "input-and-weight":
        scale = getattr(linear, "input_scale", None)
        if scale is not None:
            scale = model_management.cast_to_device(scale, x.device, None)
        x = comfy.ops.QuantizedTensor.from_float(x, linear.layout_type, scale=scale)

    return linear._forward(x, weight, bias)


def _resident_mlp_callers(self, x: torch.Tensor, stack: contextlib.ExitStack):
    """Prepare fc1/fc2 once and return callables that reuse those exact weights."""
    import comfy.ops

    if not (_linear_can_reuse_weights(self.fc1) and _linear_can_reuse_weights(self.fc2)):
        raise RuntimeError("MLP Linear implementation does not support resident weight reuse")

    fc1_quant = _linear_quantization_mode(self.fc1)
    fc2_quant = _linear_quantization_mode(self.fc2)

    # Weight-only INT8/ConvRot must be staged using the quantized weight's
    # logical dtype. Asking CastBiasWeightContext for x.dtype here dequantizes
    # the entire matrix before the first token chunk.
    fc1_stage_dtype = self.fc1.weight.dtype if fc1_quant == "weight-only" else x.dtype
    fc2_stage_dtype = self.fc2.weight.dtype if fc2_quant == "weight-only" else torch.float16
    fc1_weight, fc1_bias = stack.enter_context(
        comfy.ops.CastBiasWeightContext(
            self.fc1,
            input=None,
            dtype=fc1_stage_dtype,
            device=x.device,
            bias_dtype=x.dtype,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=fc1_quant is not None,
        )
    )
    fc2_weight, fc2_bias = stack.enter_context(
        comfy.ops.CastBiasWeightContext(
            self.fc2,
            input=None,
            dtype=fc2_stage_dtype,
            device=x.device,
            bias_dtype=torch.float16,
            offloadable=True,
            compute_dtype=torch.float16,
            want_requant=fc2_quant is not None,
        )
    )

    # This changes only QuantizedTensor.orig_dtype metadata and keeps its INT8
    # storage intact. It mirrors MixedPrecisionOps' weight-only forward path.
    if fc1_quant == "weight-only":
        fc1_weight = fc1_weight.to(dtype=x.dtype)
    if fc2_quant == "weight-only":
        fc2_weight = fc2_weight.to(dtype=torch.float16)

    def fc1_call(value):
        return _resident_linear_forward(
            self.fc1, value, fc1_weight, fc1_bias, fc1_quant
        )

    def fc2_call(value):
        return _resident_linear_forward(
            self.fc2, value, fc2_weight, fc2_bias, fc2_quant
        )

    quantized_cls = comfy.ops.QuantizedTensor
    prepared_backend = (
        "quantized"
        if isinstance(fc1_weight, quantized_cls) and isinstance(fc2_weight, quantized_cls)
        else "dense"
    )
    return fc1_call, fc2_call, prepared_backend


def _accumulate_gated_chunk(
    residual: torch.Tensor,
    result: torch.Tensor,
    start: int,
    end: int,
    gate: torch.Tensor,
    segments,
) -> None:
    """Apply H3's segment gate directly from one MLP result chunk."""
    for seg_start, seg_end, row in segments:
        a = max(start, seg_start)
        b = min(end, seg_end)
        if a < b:
            residual[a:b].addcmul_(
                result[a - start:b - start], gate[row].to(residual.dtype)
            )


def _mod_scale_shift_chunk(
    value: torch.Tensor,
    start: int,
    end: int,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments,
) -> torch.Tensor:
    """Apply H3 modulation to a token slice using the original segment order."""
    for seg_start, seg_end, row in segments:
        a = max(start, seg_start)
        b = min(end, seg_end)
        if a < b:
            local = value[a - start:b - start]
            local.mul_(1.0 + scale[row].to(value.dtype)).add_(
                shift[row].to(value.dtype)
            )
    return value


def _run_chunked_h3_mlp(
    self,
    x: torch.Tensor,
    star7_fp16: bool = False,
    residual: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    segments=None,
    input_factory=None,
    input_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run H3's row-independent SwiGLU MLP in token chunks.

    Each token row has independent fc1/SwiGLU/fc2 math. The installed block
    patch accumulates each result chunk directly into the residual stream, so
    neither the multi-GiB [S, 2*ffn] expansion nor a full [S, hidden] MLP output
    is needed. A different GEMM tile can change only the final float32 rounding
    bit; dtype, formula, weights, and token order remain unchanged.
    """
    chunk = max(256, int(_CONFIG["mlp_chunk_tokens"]))
    import comfy.ops

    if x.ndim != 2:
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    seq_len = x.shape[0]
    effective_input_dtype = input_dtype or x.dtype
    use_star7_fp16 = bool(star7_fp16 and effective_input_dtype == torch.float16)
    output_dtype = torch.float32 if use_star7_fp16 else x.dtype
    fuse_residual = residual is not None
    if fuse_residual and (gate is None or segments is None):
        raise ValueError("gate and segments are required for fused residual MLP")
    output = None if fuse_residual else torch.empty(
        (seq_len, x.shape[1]), dtype=output_dtype, device=x.device,
    )
    current_chunk = min(chunk, seq_len)
    auto_halve = bool(_CONFIG["auto_halve_on_oom"])
    mode = "star7-fp16" if use_star7_fp16 else "native"
    if fuse_residual:
        mode += "+residual"
    shape_key = (
        seq_len, x.shape[1], current_chunk, effective_input_dtype,
        x.device.type, mode, input_factory is not None,
    )
    if _CONFIG["verbose"] and shape_key not in _LOGGED_MLP_SHAPES:
        _LOGGED_MLP_SHAPES.add(shape_key)
        avoided_mib = (
            seq_len * x.shape[1] * torch.empty((), dtype=output_dtype).element_size()
            / (1024 ** 2)
            if fuse_residual else 0.0
        )
        _LOG.info(
            "[Star7 H3 Chunk] MLP active | S=%d | hidden=%d | ffn_x2=%d | "
            "chunk=%d | dtype=%s | mode=%s | avoided-output=%.1fMiB",
            seq_len, x.shape[1], self.fc1.out_features, current_chunk,
            effective_input_dtype,
            mode, avoided_mib,
        )

    staging_input = x if input_factory is None else torch.empty(
        (0, x.shape[1]), dtype=effective_input_dtype, device=x.device
    )

    def run_chunks(fc1_call, fc2_call):
        nonlocal current_chunk
        start = 0
        calls = 0
        while start < seq_len:
            end = min(start + current_chunk, seq_len)
            expanded = None
            result = None
            gate = up = activated = scaled = None
            chunk_input = None
            try:
                chunk_input = (
                    x[start:end]
                    if input_factory is None
                    else input_factory(start, end)
                )
                expanded = fc1_call(chunk_input)
                if use_star7_fp16:
                    # Match MiniMax H3 FP16 Exact Fix - Star7: FP16 activations,
                    # FP32 SwiGLU, and exact power-of-two protection before fc2.
                    gate, up = expanded.chunk(2, dim=-1)
                    activated = gate.to(torch.float32)
                    F.silu(activated, inplace=True)
                    activated.mul_(up)
                    activated.div_(256.0)
                    scaled = activated.to(torch.float16)
                    result = fc2_call(scaled).to(torch.float32).mul_(256.0)
                else:
                    # Native mode keeps ComfyUI's fused activation path. Resident
                    # reuse is deliberately limited to the Star7 FP16 formula.
                    result = comfy.ops.linear_input_act(self.fc2, expanded, "swiglu")
                expected = (end - start, x.shape[1])
                if result.dtype != output_dtype or result.shape != expected:
                    raise RuntimeError(
                        "unexpected H3 MLP chunk output: "
                        f"got shape={tuple(result.shape)} dtype={result.dtype}, "
                        f"expected shape={expected} dtype={output_dtype}"
                    )
                if fuse_residual:
                    _accumulate_gated_chunk(
                        residual, result, start, end, gate_residual, segments
                    )
                else:
                    output[start:end].copy_(result)
                del result, expanded, gate, up, activated, scaled, chunk_input
                start = end
                calls += 1
            except Exception as exc:
                if not (_is_cuda_oom(exc) and auto_halve and current_chunk > 256):
                    raise
                new_chunk = max(256, current_chunk // 2)
                _LOG.warning(
                    "H3 MLP chunk OOM at %d tokens; retrying current slice with %d tokens",
                    current_chunk, new_chunk,
                )
                del result, expanded, gate, up, activated, scaled, chunk_input
                # The handled exception traceback can otherwise retain fc1 output.
                exc.__traceback__ = None
                _clear_cuda_after_oom(x.device)
                current_chunk = new_chunk
        return calls

    profile_key = shape_key + (bool(_CONFIG["reuse_mlp_weights"]), fuse_residual)
    do_profile = bool(_CONFIG["verbose"] and profile_key not in _PROFILED_MLP_SHAPES)
    profile_start = time.perf_counter()
    cuda_start = cuda_end = None
    if do_profile and x.device.type == "cuda":
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()

    weight_mode = "streamed"
    calls = 0
    gate_residual = gate
    reuse_weights = bool(
        _CONFIG["reuse_mlp_weights"]
        and use_star7_fp16
        and _linear_can_reuse_weights(self.fc1)
        and _linear_can_reuse_weights(self.fc2)
    )
    if reuse_weights:
        try:
            with contextlib.ExitStack() as stack:
                fc1_call, fc2_call, prepared_backend = _resident_mlp_callers(
                    self, staging_input, stack
                )
                weight_mode = f"resident-{prepared_backend}"
                calls = run_chunks(fc1_call, fc2_call)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            exc.__traceback__ = None
            _clear_cuda_after_oom(x.device)
            current_chunk = min(chunk, seq_len)
            weight_mode = "streamed-fallback"
            _LOG.warning(
                "[Star7 H3 Chunk] Holding fc1/fc2 weights exceeded VRAM; "
                "falling back to per-chunk streaming"
            )
            calls = run_chunks(self.fc1, self.fc2)
    else:
        calls = run_chunks(self.fc1, self.fc2)

    if do_profile:
        _PROFILED_MLP_SHAPES.add(profile_key)
        if cuda_end is not None:
            cuda_end.record()
            cuda_end.synchronize()
            elapsed_ms = cuda_start.elapsed_time(cuda_end)
        else:
            elapsed_ms = (time.perf_counter() - profile_start) * 1000.0
        _LOG.info(
            "[Star7 H3 Chunk] MLP profile (one block) | %.1f ms | chunks=%d | "
            "final_chunk=%d | weights=%s",
            elapsed_ms, calls, current_chunk, weight_mode,
        )

    return residual if fuse_residual else output


def _chunked_h3_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Native-dtype entry point kept for direct tests and compatibility."""
    return _run_chunked_h3_mlp(self, x, star7_fp16=False)


def _make_chunked_h3_mlp_forward(star7_fp16: bool):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_chunked_h3_mlp(self, x, star7_fp16=star7_fp16)
    return forward


def _minimax_ck_int8_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    """Run H3 attention through Comfy Kitchen INT8 with native lifetimes."""
    if isinstance(x, list):
        x = x.pop()

    import comfy.model_management as mm
    import comfy.quant_ops
    from comfy.ldm.modules.attention import (
        AttentionTensorContainer,
        attention_comfy_kitchen_int8,
    )

    s = x.shape[0]
    qkv = self.qkv_proj(x)
    del x
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)

    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = mm.cast_to(self.q_norm.weight, device=q.device)
        kw = mm.cast_to(self.k_norm.weight, device=k.device)
        rot = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rot,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rot,
            )
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    # Stop V sharing the fused QKV storage. CK consumes Q/K after
    # pre-quantization, allowing the much larger QKV allocation to be freed.
    v = v.clone()
    del qkv
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = attention_comfy_kitchen_int8(
        q, k, v, self.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return self.out_proj(out.squeeze(0))


_minimax_ck_int8_attention_forward._star7_consumes_input = True


def _make_chunked_h3_block_forward(star7_fp16: bool, h3_model):
    """Fuse the chunked MLP result into H3's residual stream chunk by chunk."""
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if star7_fp16 and x.dtype != torch.float32:
            x = x.to(torch.float32)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

        h = h3_model._mod_scale_shift(
            self.norm1(x), shift_msa, scale_msa, mod_segments
        )
        if star7_fp16:
            h = h.to(torch.float16)
        attn_forward = self.attn.forward
        attn_func = getattr(attn_forward, "__func__", attn_forward)
        attn_name = getattr(attn_func, "__name__", type(self.attn).__name__)
        consumes_list = bool(
            getattr(attn_func, "_uses_optimized_attention", False)
            or getattr(attn_func, "_star7_consumes_input", False)
            or attn_name == "minimax_sageattn_forward"
        )
        attn_key = (h.shape[0], h.shape[1], h.dtype, h.device.type, attn_name)
        profile_attention = bool(
            _CONFIG["verbose"] and attn_key not in _PROFILED_ATTENTION_SHAPES
        )
        attn_start = attn_end = None
        if profile_attention and h.device.type == "cuda":
            attn_start = torch.cuda.Event(enable_timing=True)
            attn_end = torch.cuda.Event(enable_timing=True)
            attn_start.record()

        attention_input = [h] if consumes_list else h
        attention = self.attn(
            attention_input,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        if consumes_list:
            h = None
        if profile_attention:
            _PROFILED_ATTENTION_SHAPES.add(attn_key)
            if attn_end is not None:
                attn_end.record()
                attn_end.synchronize()
                elapsed_ms = attn_start.elapsed_time(attn_end)
            else:
                elapsed_ms = 0.0
            _LOG.info(
                "[Star7 H3 Chunk] Attention profile (one block) | %.1f ms | "
                "forward=%s | consumable-input=%s",
                elapsed_ms,
                attn_name,
                consumes_list,
            )
        if star7_fp16:
            attention = attention.to(torch.float32)
        x = h3_model._mod_gate(x, gate_msa, attention, mod_segments)

        mlp_input_dtype = torch.float16 if star7_fp16 else x.dtype

        def make_mlp_input(start, end):
            value = self.norm2(x[start:end])
            value = _mod_scale_shift_chunk(
                value,
                start,
                end,
                shift_mlp,
                scale_mlp,
                mod_segments,
            )
            return value.to(mlp_input_dtype)

        return _run_chunked_h3_mlp(
            self.mlp,
            x,
            star7_fp16=star7_fp16,
            residual=x,
            gate=gate_mlp,
            segments=mod_segments,
            input_factory=make_mlp_input,
            input_dtype=mlp_input_dtype,
        )

    return forward


def _disable_h3_dynamic_prefetch_wrapper(executor, *args, **kwargs):
    """Disable only H3's next-block async prefetch overlap for this MODEL branch."""
    args = list(args)
    if len(args) > 3 and isinstance(args[3], dict):
        options = args[3].copy()
        options["prefetch_dynamic_vbars"] = False
        args[3] = options
    else:
        options = kwargs.get("transformer_options", {}).copy()
        options["prefetch_dynamic_vbars"] = False
        kwargs["transformer_options"] = options
    return executor(*args, **kwargs)


def _adaptive_h3_dynamic_prefetch_wrapper(executor, *args, **kwargs):
    """Try overlapped block prefetch, then retry this forward without it on OOM."""
    try:
        return executor(*args, **kwargs)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        exc.__traceback__ = None
        import comfy.model_prefetch

        comfy.model_prefetch.cleanup_prefetch_queues()
        device = None
        for value in args:
            if torch.is_tensor(value):
                device = value.device
                break
        _clear_cuda_after_oom(device or torch.device("cuda"))
        _LOG.warning(
            "[Star7 H3 Chunk] Dynamic prefetch OOM; retrying current H3 forward "
            "with prefetch disabled"
        )

        retry_args = list(args)
        if len(retry_args) > 3 and isinstance(retry_args[3], dict):
            options = retry_args[3].copy()
            options["prefetch_dynamic_vbars"] = False
            retry_args[3] = options
        else:
            options = kwargs.get("transformer_options", {}).copy()
            options["prefetch_dynamic_vbars"] = False
            kwargs["transformer_options"] = options
        return executor(*retry_args, **kwargs)


def install_patch(
    chunk_tokens: int,
    auto_halve_on_oom: bool,
    verbose: bool,
    mlp_chunk_tokens: int = 4096,
    reuse_mlp_weights: bool = True,
):
    global _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE, _PATCHED_CK

    import comfy.quant_ops as quant_ops

    if not getattr(quant_ops, "_CK_AVAILABLE", False):
        raise RuntimeError("comfy-kitchen is unavailable; MiniMax H3 RoPE patch cannot be installed")

    ck = quant_ops.ck
    if _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE is None:
        _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE = ck.rms_rope_split_half_
        _PATCHED_CK = ck

    _CONFIG["chunk_tokens"] = int(chunk_tokens)
    _CONFIG["mlp_chunk_tokens"] = int(mlp_chunk_tokens)
    _CONFIG["auto_halve_on_oom"] = bool(auto_halve_on_oom)
    _CONFIG["verbose"] = bool(verbose)
    _CONFIG["reuse_mlp_weights"] = bool(reuse_mlp_weights)

    # Replace only the public in-place function used by ComfyUI MiniMax H3.
    # The separate model patch below controls the explicit optional attention
    # backend; RoPE dispatch itself does not alter backend priority.
    ck.rms_rope_split_half_ = _chunked_rms_rope_split_half_inplace

    if verbose:
        active_backends = []
        try:
            for name, info in ck.list_backends().items():
                if info.get("available") and not info.get("disabled"):
                    active_backends.append(name)
        except Exception:
            pass
        _LOG.info(
            "[Star7 H3 Chunk] Installed v%s | RoPE=%d | MLP=%d | auto-half=%s | "
            "reuse-weights=%s | backends=%s",
            NODE_VERSION, int(chunk_tokens), int(mlp_chunk_tokens),
            bool(auto_halve_on_oom),
            bool(reuse_mlp_weights),
            ",".join(active_backends) or "unknown",
        )


def install_model_patch(
    model,
    chunk_tokens: int,
    auto_halve_on_oom: bool,
    verbose: bool,
    mlp_chunk_tokens: int,
    disable_dynamic_prefetch: bool,
    reuse_mlp_weights: bool,
    attention_backend: str,
):
    install_patch(
        chunk_tokens, auto_halve_on_oom, verbose,
        mlp_chunk_tokens, reuse_mlp_weights,
    )

    from comfy.ldm.minimax import model as h3_model
    import comfy.patcher_extension

    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, h3_model.MiniMaxH3Model):
        _LOG.warning("[Star7 H3 Chunk] Non-H3 model received; only the guarded RoPE dispatch was installed")
        return patched

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    star7_fp16 = bool(transformer_options.get("star7_minimax_h3_fp16_exact_fix"))
    first_attn_patch = patched.object_patches.get(
        "diffusion_model.blocks.0.attn.forward"
    )
    first_attn_func = getattr(first_attn_patch, "__func__", first_attn_patch)
    attention_patch_name = getattr(first_attn_func, "__name__", "native")
    sage_attention = attention_patch_name == "minimax_sageattn_forward"
    if verbose and sage_attention:
        _LOG.warning(
            "[Star7 H3 Chunk] Sage Attention detected: Q/K INT8 attention is "
            "active and is not an exact-attention path"
        )

    ck_attention = False
    if attention_backend == "comfy_kitchen_int8":
        import comfy_kitchen
        if comfy_kitchen.int8_attention_is_available():
            for index, block in enumerate(diffusion_model.blocks):
                patched.add_object_patch(
                    f"diffusion_model.blocks.{index}.attn.forward",
                    MethodType(_minimax_ck_int8_attention_forward, block.attn),
                )
            attention_patch_name = "star7_comfy_kitchen_int8"
            sage_attention = False
            ck_attention = True
            if verbose:
                _LOG.info(
                    "[Star7 H3 Chunk] Comfy Kitchen INT8 attention selected; "
                    "any earlier MiniMax Sage attention patch is overridden"
                )
        else:
            _LOG.warning(
                "[Star7 H3 Chunk] Comfy Kitchen INT8 attention was requested "
                "but is unavailable; keeping the existing attention backend"
            )

    # Patch the whole block so each MLP result chunk can be gated directly into
    # the residual stream. This removes the full [S, hidden] MLP output buffer.
    block_forward = _make_chunked_h3_block_forward(star7_fp16, h3_model)
    for index, block in enumerate(diffusion_model.blocks):
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.forward",
            MethodType(block_forward, block),
        )

    prefetch_wrapper = (
        _disable_h3_dynamic_prefetch_wrapper
        if disable_dynamic_prefetch
        else _adaptive_h3_dynamic_prefetch_wrapper
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "h3_activation_chunk_star7_dynamic_prefetch_policy",
        prefetch_wrapper,
    )

    if verbose:
        _LOG.info(
            "[Star7 H3 Chunk] Model ready v%s | blocks=%d | FP16 Exact=%s | "
            "dynamic prefetch=%s | fused residual=True | attention=%s",
            NODE_VERSION, len(diffusion_model.blocks), star7_fp16,
            not disable_dynamic_prefetch,
            (
                "comfy-kitchen-int8" if ck_attention
                else "sage-qk-int8" if sage_attention
                else attention_patch_name
            ),
        )
    return patched


class MiniMaxH3ActivationChunkStar7:
    """Pass-through MODEL node that installs formula-preserving H3 activation chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "chunk_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 RoPE sequence tokens per chunk. 2080 Ti 22GB: use 8192 after a safe 4096 validation run.",
                    },
                ),
                "auto_halve_on_oom": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If a RoPE chunk OOMs before write-back, halve that chunk and retry down to 256 tokens.",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Print compact one-time configuration and shape summaries to the ComfyUI console.",
                    },
                ),
                "mlp_chunk_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 MLP tokens per chunk. v2 also streams norm/modulation and fuses each result into the residual. Start at 4096.",
                    },
                ),
                "disable_dynamic_prefetch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True is safest. False enables overlap for speed and v2 automatically retries the current H3 forward without prefetch after an OOM.",
                    },
                ),
                "reuse_mlp_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prepare fc1/fc2 once per block. v2 preserves INT8/ConvRot QuantizedTensor weights and falls back automatically if residency exceeds VRAM.",
                    },
                ),
                "attention_backend": (
                    ["existing", "comfy_kitchen_int8"],
                    {
                        "default": "existing",
                        "tooltip": (
                            "existing keeps the incoming attention patch (for example KJ Sage). "
                            "comfy_kitchen_int8 selects ComfyUI's native INT8 attention and "
                            "overrides an earlier MiniMax Sage patch. Both INT8 paths are approximate."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Low-VRAM MiniMax H3 patch. Chunks fused RMSNorm + split-half RoPE, streams "
        "MLP norm/modulation, preserves INT8/ConvRot weights, and accumulates fc2 "
        "chunks directly into the residual. Can safely retry without AIMDO prefetch. "
        "Compatible with FP16 Exact Fix - Star7. The default existing attention mode "
        "does not change attention math; the optional Comfy Kitchen INT8 mode is approximate."
    )

    def patch(
        self, model, chunk_tokens=4096, auto_halve_on_oom=True, verbose=True,
        mlp_chunk_tokens=4096, disable_dynamic_prefetch=True,
        reuse_mlp_weights=True, attention_backend="existing",
    ):
        return (install_model_patch(
            model, chunk_tokens, auto_halve_on_oom, verbose,
            mlp_chunk_tokens, disable_dynamic_prefetch, reuse_mlp_weights,
            attention_backend,
        ),)


MiniMaxH3RoPEChunkPatch = MiniMaxH3ActivationChunkStar7


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": MiniMaxH3ActivationChunkStar7,
    # Compatibility alias for workflows saved before the package was renamed.
    "MiniMaxH3RoPEChunkPatch": MiniMaxH3ActivationChunkStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": "MiniMax H3 Activation Chunk (RoPE + MLP) - Star7",
    "MiniMaxH3RoPEChunkPatch": "MiniMax H3 Activation Chunk (RoPE + MLP) - Star7",
}
