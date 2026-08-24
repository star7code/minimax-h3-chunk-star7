import gc
import contextlib
import logging
import math
import os
import shutil
import subprocess
import time
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F

_LOG = logging.getLogger("MiniMaxH3ActivationChunkStar7")
NODE_VERSION = "2.9.1"

_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE = None
_PATCHED_CK = None
_LOGGED_ROPE_SHAPES = set()
_LOGGED_MLP_SHAPES = set()
_PROFILED_MLP_SHAPES = set()
_PROFILED_ATTENTION_SHAPES = set()
_LOGGED_SLA_SHAPES = set()
_PROFILED_QKV_STAGES = set()
_LOGGED_REFERENCE_VIDEO_SHAPES = set()
_CONFIG = {
    "chunk_tokens": 8192,
    "mlp_chunk_tokens": 8192,
    "qkv_chunk_tokens": 8192,
    "effective_chunk_tokens": 8192,
    "effective_mlp_chunk_tokens": 8192,
    "effective_qkv_chunk_tokens": 8192,
    "status_effective_chunk_tokens": 8192,
    "status_effective_mlp_chunk_tokens": 8192,
    "status_effective_qkv_chunk_tokens": 8192,
    "status_sequence_rope": None,
    "status_sequence_mlp": None,
    "status_sequence_qkv": None,
    "auto_halve_on_oom": True,
    "verbose": True,
    "reuse_mlp_weights": True,
    "node_id": None,
}


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "would exceed allowed memory" in msg


def _should_enable_sm75_auto_fp16_exact(
    capability, star7_fp16: bool, attention_backend: str,
) -> bool:
    """Keep the Turing overflow fix isolated from Ampere and newer GPUs."""
    return bool(
        capability == (7, 5)
        and not star7_fp16
        and attention_backend in {
            "comfy_kitchen_int8",
            "sla_sm75_qk_int8_pv_fp16",
            "sla_sm75_all_int8_experimental",
        }
    )


def _clear_cuda_after_oom(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _runtime_status_payload(reason: str) -> dict:
    return {
        "node_id": _CONFIG.get("node_id"),
        "configured_rope": int(_CONFIG["chunk_tokens"]),
        "effective_rope": int(_CONFIG["status_effective_chunk_tokens"]),
        "configured_mlp": int(_CONFIG["mlp_chunk_tokens"]),
        "effective_mlp": int(_CONFIG["status_effective_mlp_chunk_tokens"]),
        "configured_qkv": int(_CONFIG["qkv_chunk_tokens"]),
        "effective_qkv": int(_CONFIG["status_effective_qkv_chunk_tokens"]),
        "reason": reason,
    }


def _send_runtime_status(reason: str) -> None:
    node_id = _CONFIG.get("node_id")
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "star7-h3-chunk-status", _runtime_status_payload(reason)
        )
    except Exception as exc:
        _LOG.debug("Unable to update the Star7 chunk status widget: %s", exc)


def _remember_effective_chunk(kind: str, failed: int, effective: int) -> None:
    if kind == "RoPE":
        configured_key = "chunk_tokens"
        effective_key = "effective_chunk_tokens"
    elif kind == "MLP":
        configured_key = "mlp_chunk_tokens"
        effective_key = "effective_mlp_chunk_tokens"
    elif kind == "QKV":
        configured_key = "qkv_chunk_tokens"
        effective_key = "effective_qkv_chunk_tokens"
    else:
        raise ValueError(f"unknown chunk kind: {kind}")

    configured = int(_CONFIG[configured_key])
    previous = int(_CONFIG[effective_key])
    effective = max(256, int(effective))
    # Zero means "prefer one full-sequence call", not "disable the OOM
    # handler". Once that full call has failed, remember the smaller working
    # value for every later block in the same model run.
    if previous > 0:
        effective = min(previous, effective)
    _CONFIG[effective_key] = effective
    status_key = (
        "status_effective_chunk_tokens" if kind == "RoPE"
        else "status_effective_mlp_chunk_tokens" if kind == "MLP"
        else "status_effective_qkv_chunk_tokens"
    )
    sequence_key = (
        "status_sequence_rope"
        if kind == "RoPE"
        else "status_sequence_mlp" if kind == "MLP"
        else "status_sequence_qkv"
    )
    sequence = _CONFIG.get(sequence_key)
    _CONFIG[status_key] = min(effective, int(sequence)) if sequence else effective
    _LOG.warning(
        "[Star7 H3 Chunk] %s VRAM fallback: configured=%d, failed-at=%d, "
        "now-using=%d for later blocks in this model session.",
        kind, configured, failed, effective,
    )
    _send_runtime_status(f"{kind.lower()}_oom")


def _configure_runtime(
    chunk_tokens: int,
    mlp_chunk_tokens: int,
    auto_halve_on_oom: bool,
    verbose: bool,
    reuse_mlp_weights: bool,
    node_id=None,
    qkv_chunk_tokens: int = 8192,
) -> None:
    """Start a fresh runtime budget whenever the node inputs are re-executed."""
    configured_rope = int(chunk_tokens)
    configured_mlp = int(mlp_chunk_tokens)
    configured_qkv = int(qkv_chunk_tokens)
    # Older workflows may contain the pre-MLP field's zero placeholder.
    if configured_rope < 0:
        configured_rope = 8192
    if configured_mlp < 0:
        configured_mlp = 4096
    if configured_qkv < 0:
        configured_qkv = 4096
    _CONFIG["chunk_tokens"] = configured_rope
    _CONFIG["mlp_chunk_tokens"] = configured_mlp
    _CONFIG["qkv_chunk_tokens"] = configured_qkv
    _CONFIG["effective_chunk_tokens"] = configured_rope
    _CONFIG["effective_mlp_chunk_tokens"] = configured_mlp
    _CONFIG["effective_qkv_chunk_tokens"] = configured_qkv
    _CONFIG["status_effective_chunk_tokens"] = configured_rope
    _CONFIG["status_effective_mlp_chunk_tokens"] = configured_mlp
    _CONFIG["status_effective_qkv_chunk_tokens"] = configured_qkv
    _CONFIG["status_sequence_rope"] = None
    _CONFIG["status_sequence_mlp"] = None
    _CONFIG["status_sequence_qkv"] = None
    _CONFIG["auto_halve_on_oom"] = bool(auto_halve_on_oom)
    _CONFIG["verbose"] = bool(verbose)
    _CONFIG["reuse_mlp_weights"] = bool(reuse_mlp_weights)
    _CONFIG["node_id"] = str(node_id) if node_id is not None else None
    _send_runtime_status("configured")


def _set_sequence_status(kind: str, sequence_length: int) -> None:
    """Expose the cap used by this forward without learning it as an OOM cap."""
    sequence_length = max(1, int(sequence_length))
    if kind == "RoPE":
        learned_key = "effective_chunk_tokens"
        status_key = "status_effective_chunk_tokens"
        sequence_key = "status_sequence_rope"
    elif kind == "MLP":
        learned_key = "effective_mlp_chunk_tokens"
        status_key = "status_effective_mlp_chunk_tokens"
        sequence_key = "status_sequence_mlp"
    elif kind == "QKV":
        learned_key = "effective_qkv_chunk_tokens"
        status_key = "status_effective_qkv_chunk_tokens"
        sequence_key = "status_sequence_qkv"
    else:
        raise ValueError(f"unknown chunk kind: {kind}")

    learned = int(_CONFIG[learned_key])
    actual = sequence_length if learned == 0 else min(learned, sequence_length)
    changed = (
        _CONFIG.get(status_key) != actual
        or _CONFIG.get(sequence_key) != sequence_length
    )
    _CONFIG[status_key] = actual
    _CONFIG[sequence_key] = sequence_length
    if changed:
        _send_runtime_status(
            "sequence_limit" if learned > 0 and actual < learned else "active"
        )


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

    if int(_CONFIG.get("chunk_tokens", 8192)) == 0:
        if _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE is None:
            raise RuntimeError("MiniMax H3 RoPE original fallback is unavailable")
        return _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE(
            q, k, freqs_cis, q_scale, k_scale, epsilon=epsilon, rot_dim=rot_dim
        )

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
    _set_sequence_status("RoPE", seq_len)
    configured = int(_CONFIG["effective_chunk_tokens"])
    preferred = seq_len if configured == 0 else max(256, configured)
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
        current_chunk = min(
            chunk, max(256, int(_CONFIG["effective_chunk_tokens"]))
        )
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
                _remember_effective_chunk("RoPE", current_chunk, new_chunk)
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
    """Prepare private fc1/fc2 snapshots and reuse them across token chunks.

    AIMDO/VBAR may return views into a shared cast buffer.  Preparing one layer
    at a time, cloning its fully patched result, and then releasing the context
    prevents the next layer from overwriting a weight still used by this MLP.
    """
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
    def snapshot(linear, stage_dtype, bias_dtype, compute_dtype, quant_mode):
        with comfy.ops.CastBiasWeightContext(
            linear,
            input=None,
            dtype=stage_dtype,
            device=x.device,
            bias_dtype=bias_dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=quant_mode is not None,
        ) as (weight, bias):
            # clone() preserves QuantizedTensor layout/metadata while giving
            # the resident caller storage independent of AIMDO's cast buffer.
            private_weight = weight.detach().clone() if weight is not None else None
            private_bias = bias.detach().clone() if bias is not None else None
        return private_weight, private_bias

    fc1_weight, fc1_bias = snapshot(
        self.fc1, fc1_stage_dtype, x.dtype, x.dtype, fc1_quant
    )
    fc2_weight, fc2_bias = snapshot(
        self.fc2, fc2_stage_dtype, torch.float16, torch.float16, fc2_quant
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
    configured_chunk = int(_CONFIG["effective_mlp_chunk_tokens"])
    import comfy.ops

    if x.ndim != 2:
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    seq_len = x.shape[0]
    chunk = seq_len if configured_chunk == 0 else max(256, configured_chunk)
    _set_sequence_status("MLP", seq_len)
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
                _remember_effective_chunk("MLP", current_chunk, new_chunk)
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
    if (
        _CONFIG["reuse_mlp_weights"]
        and use_star7_fp16
        and not reuse_weights
        and _CONFIG["verbose"]
        and "dynamic-residency" not in _LOGGED_MLP_SHAPES
    ):
        _LOGGED_MLP_SHAPES.add("dynamic-residency")
        _LOG.info(
            "[Star7 H3 Chunk] Resident MLP reuse disabled automatically: "
            "dynamic/low-VRAM weight residency detected; using streamed-safe path"
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
        expansion_mib = (
            current_chunk * self.fc1.out_features
            * torch.empty((), dtype=effective_input_dtype).element_size()
            / (1024 ** 2)
        )
        _LOG.info(
            "[Star7 H3 Chunk] First-block MLP | S=%d | chunk=%d x %d | "
            "mode=%s | weights=%s | temp=%.1fMiB | %.1fms",
            seq_len, current_chunk, calls, mode, weight_mode,
            expansion_mib, elapsed_ms,
        )

    return residual if fuse_residual else output


def _chunked_h3_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Native-dtype entry point kept for direct tests and compatibility."""
    return _run_chunked_h3_mlp(self, x, star7_fp16=False)


def _make_chunked_h3_mlp_forward(star7_fp16: bool):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_chunked_h3_mlp(self, x, star7_fp16=star7_fp16)
    return forward


def _require_strict_sla_finite(value: torch.Tensor, stage: str) -> None:
    """Stop strict SLA jobs at the first observable non-finite model stage."""
    if not str(_CONFIG.get("attention_backend", "")).startswith("sla_"):
        return
    if bool(torch.isfinite(value).all().item()):
        return
    nan_count = int(torch.isnan(value).sum().item())
    inf_count = int(torch.isinf(value).sum().item())
    raise RuntimeError(
        f"Star7 strict SLA detected NaN/Inf after {stage} "
        f"(nan={nan_count}, inf={inf_count}). The task was stopped before VAE "
        "decode to prevent checkerboard/flicker output. SM75 FP16 Exact is "
        "enabled automatically for Star7 SLA; if this still occurs, report "
        "the Star7 version, failing block and model/LoRA/sampler combination. "
        "No CK/Sage fallback was attempted."
    )


def _h3_output_finite_passthrough(original_forward):
    """Reject invalid H3 video/audio velocities before the sampler or VAE."""
    def forward(self, *args, **kwargs):
        result = original_forward(*args, **kwargs)
        labels = ("video model output", "audio model output")
        for label, value in zip(labels, result):
            if bool(torch.isfinite(value).all().item()):
                continue
            nan_count = int(torch.isnan(value).sum().item())
            inf_count = int(torch.isinf(value).sum().item())
            raise RuntimeError(
                f"Star7 H3 detected NaN/Inf in {label} "
                f"(nan={nan_count}, inf={inf_count}, "
                f"attention={_CONFIG.get('attention_backend', 'unknown')}). "
                "The task was stopped during sampling, before VAE decode and "
                "video/audio muxing; replacing invalid samples would not "
                "recover the generated content."
            )
        return result

    return forward


def _condition_proj_fp32_forward(original_forward):
    """Keep the SM75 SLA text-conditioning projection in its FP32 island."""
    def forward(self, tensor):
        return original_forward(tensor.to(torch.float32))
    return forward


def _fp16_exact_out_proj(linear, tensor: torch.Tensor) -> torch.Tensor:
    """Power-of-two protected FP16 attention down-projection, consuming input."""
    if tensor.dtype == torch.float16:
        scaled = tensor.div_(64.0)
    else:
        scaled = tensor.div(64.0).to(torch.float16)
    return linear(scaled).to(torch.float32).mul_(64.0)


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
    q, k, v = _prepare_h3_qkv_chunked(
        self, x, rope_freqs, mm, comfy.quant_ops, output_dtype=x.dtype
    )
    del x

    # Stop V sharing the fused QKV storage. CK consumes Q/K after
    # pre-quantization, allowing the much larger QKV allocation to be freed.
    q = AttentionTensorContainer(q)
    k = AttentionTensorContainer(k)
    v = AttentionTensorContainer(v)
    out = attention_comfy_kitchen_int8(
        q, k, v, self.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    out = out.squeeze(0)
    if getattr(self, "_star7_auto_fp16_exact", False):
        return _fp16_exact_out_proj(self.out_proj, out)
    return self.out_proj(out)


_minimax_ck_int8_attention_forward._star7_consumes_input = True


def _load_sla_backend():
    try:
        from . import sla_backend
    except ImportError:
        # Direct-file loading is used by the standalone test suite.
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).with_name("sla_backend.py")
        module_name = "star7_sla_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Star7 SLA backend from {path}")
        sla_backend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = sla_backend
        spec.loader.exec_module(sla_backend)
    return sla_backend


def _minimax_sla_forward(
    self, x, rope_freqs=None, transformer_options={},
    star7_sla_mod_segments=(),
):
    """Run strict LightX2V-style sparse attention without any fallback."""
    if isinstance(x, list):
        x = x.pop()

    import comfy.model_management as mm
    import comfy.quant_ops

    sla_backend = _load_sla_backend()
    sequence = x.shape[0]
    q, k, v = _prepare_h3_qkv_chunked(
        self, x, rope_freqs, mm, comfy.quant_ops, output_dtype=torch.float16
    )
    del x

    # SLA kernels take strict FP16 inputs. Both architecture paths quantize Q/K
    # for tensor-core QK while keeping V and the PV multiplication in FP16.
    # The chunk helper writes FP16 directly, avoiding a second full Q/K/V copy
    # when the upstream H3 projection computes in FP32.

    sla_segments = star7_sla_mod_segments or getattr(
        self, "_star7_sla_mod_segments", ()
    )
    priority_ranges = []
    for start, end, mod_row in sla_segments:
        if isinstance(mod_row, int):
            modality_tag = mod_row % 3
        elif torch.is_tensor(mod_row) and mod_row.numel() == 1:
            modality_tag = int(mod_row.item()) % 3
        else:
            modality_tag = -1
        if modality_tag == 2:
            priority_ranges.append((int(start), int(end)))
    # MiniMax H3 guarantees that the target audio and video streams are the
    # final two packed segments. Use that contract even when a masked segment
    # carries a per-row tensor instead of a scalar modality tag.
    if len(sla_segments) >= 2:
        audio_segment = sla_segments[-2]
        priority_ranges.append((int(audio_segment[0]), int(audio_segment[1])))
    priority_ranges = sorted(set(priority_ranges))
    device_index = q.device.index
    owned_qkv = [q, k, v]
    del q, k, v
    result = sla_backend.sparse_attention_consume(
        owned_qkv, query_priority_ranges=priority_ranges,
        all_int8=(
            _CONFIG.get("attention_backend")
            == "sla_sm75_all_int8_experimental"
        ),
    )
    shape_key = (sequence, self.heads, self.head_dim, device_index)
    if _CONFIG["verbose"] and shape_key not in _LOGGED_SLA_SHAPES:
        _LOGGED_SLA_SHAPES.add(shape_key)
        _LOG.info(
            "[Star7 H3 Chunk] SLA verified | Q-blocks=%d | K-blocks=%d | "
            "selected=%d | audio-guard-blocks=%d | effective-sparsity=%.2f%% | "
            "segments=%d | audio-ranges=%s | backend=%s | implementation=%s",
            result.query_blocks, result.key_blocks, result.selected_key_blocks,
            result.protected_query_blocks,
            result.effective_sparsity * 100.0,
            len(sla_segments), priority_ranges,
            _CONFIG.get("attention_backend"),
            result.implementation,
        )
    out = result.output.transpose(1, 2).reshape(
        1, sequence, self.heads * self.head_dim
    )
    if getattr(self, "_star7_auto_fp16_exact", False):
        # Match the standalone Native FP16 fix. Scaling by an exact power of
        # two prevents the FP16 attention down-projection from overflowing
        # without changing the represented result.
        return _fp16_exact_out_proj(self.out_proj, out.squeeze(0))
    return self.out_proj(out.squeeze(0))


_minimax_sla_forward._star7_consumes_input = True


def _sla_segment_passthrough(original_forward, block_index=None):
    """Expose H3 packed segments to SLA while preserving the upstream block."""
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        old_segments = getattr(self.attn, "_star7_sla_mod_segments", None)
        self.attn._star7_sla_mod_segments = mod_segments
        try:
            result = original_forward(
                x, t_emb, mod_segments, rope_freqs,
                transformer_options=transformer_options,
            )
            # One check after the complete block catches attention projection,
            # residual/gating and MLP failures without adding extra host
            # synchronizations beyond the previous SLA output guard.
            stage = (
                f"transformer block {block_index} output"
                if block_index is not None
                else "transformer block output"
            )
            _require_strict_sla_finite(result, stage)
            return result
        finally:
            if old_segments is None:
                delattr(self.attn, "_star7_sla_mod_segments")
            else:
                self.attn._star7_sla_mod_segments = old_segments

    return forward


def _prepare_h3_qkv_chunked(
    self, x, rope_freqs, mm, quant_ops, output_dtype: Optional[torch.dtype] = None,
):
    """Prepare contiguous backend-layout Q/K/V in token chunks.

    Buffers are allocated directly as [1, H, S, D], so CK/SLA do not create a
    second full-size transpose/contiguous copy after projection.
    """
    sequence = int(x.shape[0])
    heads, head_dim = self.heads, self.head_dim
    configured_chunk = int(_CONFIG["effective_qkv_chunk_tokens"])
    chunk = sequence if configured_chunk == 0 else min(
        sequence, max(256, configured_chunk)
    )
    output_dtype = output_dtype or x.dtype
    qkv_weight = getattr(self.qkv_proj, "weight", None)
    qkv_quantized = (
        getattr(self.qkv_proj, "layout_type", None) is not None
        or type(qkv_weight).__name__ == "QuantizedTensor"
    )
    total_profile_key = (
        sequence, chunk, str(x.dtype), str(output_dtype),
        x.device.type, x.device.index, qkv_quantized,
    )
    profile_total = bool(
        _CONFIG["verbose"]
        and ("summary", total_profile_key) not in _PROFILED_QKV_STAGES
    )
    total_profile_start = time.perf_counter() if profile_total else None
    first_projection_ms = None
    first_rope_ms = None
    _set_sequence_status("QKV", sequence)
    # These complete Q/K/V tensors are required by CK and SLA regardless of
    # projection chunk size. Retry their allocation once after releasing only
    # unused allocator cache, but do not pretend that lowering a local chunk can
    # solve a full-buffer OOM.
    qkv_buffers = []
    for allocation_attempt in range(2):
        try:
            qkv_buffers = [
                torch.empty(
                    (1, heads, sequence, head_dim),
                    dtype=output_dtype,
                    device=x.device,
                )
            ]
            qkv_buffers.append(torch.empty_like(qkv_buffers[0]))
            qkv_buffers.append(torch.empty_like(qkv_buffers[0]))
            break
        except Exception as exc:
            qkv_buffers.clear()
            if not _is_cuda_oom(exc) or allocation_attempt:
                if _is_cuda_oom(exc):
                    required_gib = (
                        3 * heads * sequence * head_dim
                        * torch.empty((), dtype=output_dtype).element_size()
                        / 1024**3
                    )
                    _LOG.error(
                        "[Star7 H3 Chunk] Full Q/K/V buffer OOM | required=%.2fGiB "
                        "| S=%d | dtype=%s. QKV/MLP/RoPE chunk reduction cannot "
                        "lower this fixed attention input; reduce reference tokens "
                        "or canvas size.",
                        required_gib, sequence, output_dtype,
                    )
                raise
            exc.__traceback__ = None
            _clear_cuda_after_oom(x.device)
    q_out, k_out, v_out = qkv_buffers
    del qkv_buffers
    rope_fn = _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE or quant_ops.ck.rms_rope_split_half_
    start = 0
    while start < sequence:
        end = min(start + chunk, sequence)
        try:
            profile_qkv = bool(profile_total and start == 0)
            qkv_profile_start = time.perf_counter() if profile_qkv else None
            qkv_chunk = self.qkv_proj(x[start:end])
            if profile_qkv:
                if x.device.type == "cuda":
                    torch.cuda.synchronize(x.device)
                first_projection_ms = (
                    time.perf_counter() - qkv_profile_start
                ) * 1000.0
            q, k, v = qkv_chunk.split(heads * head_dim, dim=-1)
            v = v.view(end - start, heads, head_dim)
            if rope_freqs is not None:
                q = q.view(1, end - start, heads, head_dim)
                k = k.view(1, end - start, heads, head_dim)
                qw = mm.cast_to(self.q_norm.weight, device=q.device)
                kw = mm.cast_to(self.k_norm.weight, device=k.device)
                freq_chunk = rope_freqs[:, start:end, ...]
                rot = rope_freqs.shape[-3] * 2
                profile_rope = bool(profile_total and start == 0)
                rope_profile_start = time.perf_counter() if profile_rope else None
                rope_fn(q, k, freq_chunk, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                if profile_rope:
                    if x.device.type == "cuda":
                        torch.cuda.synchronize(x.device)
                    first_rope_ms = (
                        time.perf_counter() - rope_profile_start
                    ) * 1000.0
                q_out[:, :, start:end, :].copy_(q.permute(0, 2, 1, 3))
                k_out[:, :, start:end, :].copy_(k.permute(0, 2, 1, 3))
            else:
                q_norm = self.q_norm(q.view(end - start, heads, head_dim))
                k_norm = self.k_norm(k.view(end - start, heads, head_dim))
                q_out[:, :, start:end, :].copy_(q_norm.permute(1, 0, 2).unsqueeze(0))
                k_out[:, :, start:end, :].copy_(k_norm.permute(1, 0, 2).unsqueeze(0))
            v_out[:, :, start:end, :].copy_(v.permute(1, 0, 2).unsqueeze(0))
            start = end
            del qkv_chunk, q, k, v
        except Exception as exc:
            if not (
                _is_cuda_oom(exc)
                and _CONFIG["auto_halve_on_oom"]
                and chunk > 256
            ):
                raise
            new_chunk = max(256, chunk // 2)
            _remember_effective_chunk("QKV", chunk, new_chunk)
            exc.__traceback__ = None
            _clear_cuda_after_oom(x.device)
            chunk = new_chunk
    if profile_total:
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        # A first-call OOM may have reduced ``chunk`` after the profile key was
        # created. Record the value that actually completed so block 2 does not
        # print the same summary again.
        completed_profile_key = (
            sequence, chunk, str(x.dtype), str(output_dtype),
            x.device.type, x.device.index, qkv_quantized,
        )
        _PROFILED_QKV_STAGES.add(("summary", completed_profile_key))
        allocated = reserved = 0.0
        if x.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(x.device) / 1024**3
            reserved = torch.cuda.memory_reserved(x.device) / 1024**3
        _LOG.info(
            "[Star7 H3 Chunk] First-block QKV | S=%d | chunk=%d x %d | "
            "quantized=%s | %s->%s | first projection=%.1fms | "
            "first norm+RoPE=%.1fms | total=%.1fms | VRAM=%.2f/%.2fGiB",
            sequence, chunk, (sequence + chunk - 1) // chunk,
            qkv_quantized, x.dtype, output_dtype,
            first_projection_ms or 0.0, first_rope_ms or 0.0,
            (time.perf_counter() - total_profile_start) * 1000.0,
            allocated, reserved,
        )
    return q_out, k_out, v_out


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
        attention_kwargs = {
            "rope_freqs": rope_freqs,
            "transformer_options": transformer_options,
        }
        if _CONFIG.get("attention_backend", "existing").startswith("sla_"):
            attention_kwargs["star7_sla_mod_segments"] = mod_segments
        old_attn_segments = getattr(self.attn, "_star7_sla_mod_segments", None)
        if _CONFIG.get("attention_backend", "existing").startswith("sla_"):
            self.attn._star7_sla_mod_segments = mod_segments
        try:
            attention = self.attn(attention_input, **attention_kwargs)
        finally:
            if _CONFIG.get("attention_backend", "existing").startswith("sla_"):
                if old_attn_segments is None:
                    delattr(self.attn, "_star7_sla_mod_segments")
                else:
                    self.attn._star7_sla_mod_segments = old_attn_segments
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

        result = _run_chunked_h3_mlp(
            self.mlp,
            x,
            star7_fp16=star7_fp16,
            residual=x,
            gate=gate_mlp,
            segments=mod_segments,
            input_factory=make_mlp_input,
            input_dtype=mlp_input_dtype,
        )
        block_index = getattr(self, "_star7_block_index", None)
        stage = (
            f"transformer block {block_index} output"
            if block_index is not None
            else "transformer block output"
        )
        _require_strict_sla_finite(result, stage)
        return result

    return forward


def install_patch(
    chunk_tokens: int,
    auto_halve_on_oom: bool,
    verbose: bool,
    mlp_chunk_tokens: int = 8192,
    qkv_chunk_tokens: int = 8192,
    reuse_mlp_weights: bool = True,
    node_id=None,
):
    global _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE, _PATCHED_CK

    import comfy.quant_ops as quant_ops

    if not getattr(quant_ops, "_CK_AVAILABLE", False):
        raise RuntimeError("comfy-kitchen is unavailable; MiniMax H3 RoPE patch cannot be installed")

    ck = quant_ops.ck
    if _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE is None:
        _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE = ck.rms_rope_split_half_
        _PATCHED_CK = ck

    _configure_runtime(
        chunk_tokens, mlp_chunk_tokens, auto_halve_on_oom, verbose,
        reuse_mlp_weights, node_id, qkv_chunk_tokens=qkv_chunk_tokens,
    )

    # Keep the zero setting as a true control group: restore the exact public
    # operator so unrelated encoders never even enter the Star7 dispatcher.
    # Otherwise replace only the public in-place function used by H3.
    ck.rms_rope_split_half_ = (
        _ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE
        if int(chunk_tokens) == 0 and not auto_halve_on_oom
        else _chunked_rms_rope_split_half_inplace
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
    node_id=None,
    qkv_chunk_tokens: int = 8192,
):
    install_patch(
        chunk_tokens=chunk_tokens,
        auto_halve_on_oom=auto_halve_on_oom,
        verbose=verbose,
        mlp_chunk_tokens=mlp_chunk_tokens,
        qkv_chunk_tokens=qkv_chunk_tokens,
        reuse_mlp_weights=reuse_mlp_weights,
        node_id=node_id,
    )
    _CONFIG["attention_backend"] = attention_backend
    from comfy.ldm.minimax import model as h3_model
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, h3_model.MiniMaxH3Model):
        _LOG.warning("[Star7 H3 Chunk] Non-H3 model received; only the guarded RoPE dispatch was installed")
        return patched

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    star7_fp16 = bool(transformer_options.get("star7_minimax_h3_fp16_exact_fix"))
    dit_replacements = transformer_options.get("patches_replace", {}).get("dit", {})
    block_loop_cache = ("block_loop", 0) in dit_replacements
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
    sla_attention = False
    auto_sm75_fp16_exact = False
    strict_sla_backends = {
        "sla_sm75_qk_int8_pv_fp16",
        "sla_sm75_all_int8_experimental",
        "sla_sm80+_qk_int8_pv_fp16",
    }
    sla_backend = None
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    if attention_backend in strict_sla_backends:
        sla_backend = _load_sla_backend()
        # Strict preflight: selecting SLA must never leave an earlier Sage/CK
        # patch active while the UI claims SLA was selected.
        capability = sla_backend.check_runtime_support(
            requested_backend=attention_backend
        )

    # Both Star7 SLA and CK execute the same H3 residual/MLP stack. Turing has
    # no native BF16 arithmetic, so either attention backend needs the same
    # overflow-safe FP16 branches and FP32 residual/SwiGLU formula. Keep unknown
    # incoming `existing` attention patches untouched.
    auto_sm75_fp16_exact = _should_enable_sm75_auto_fp16_exact(
        capability, star7_fp16, attention_backend,
    )
    if auto_sm75_fp16_exact:
        patched.set_model_compute_dtype(torch.float16)
        is_quantized = any(
            getattr(module, "layout_type", None) is not None
            for module in diffusion_model.modules()
        )
        if is_quantized:
            patched.force_cast_weights = False
        condition_proj = diffusion_model.condition_proj
        patched.add_object_patch(
            "diffusion_model.condition_proj.forward",
            MethodType(
                _condition_proj_fp32_forward(condition_proj.forward),
                condition_proj,
            ),
        )
        transformer_options["star7_h3_sm75_auto_fp16_exact"] = NODE_VERSION

    # This guard covers CK, SLA and preserved attention alike. It is deliberately
    # placed at the joint model output rather than in VideoHelperSuite: FFmpeg can
    # only report corrupt PCM, whereas here the failing video/audio stream and
    # selected attention backend are still known.
    if transformer_options.get("star7_h3_output_finite_guard") != NODE_VERSION:
        model_forward_path = "diffusion_model.forward"
        upstream_model_forward = patched.object_patches.get(
            model_forward_path, diffusion_model.forward
        )
        patched.add_object_patch(
            model_forward_path,
            MethodType(
                _h3_output_finite_passthrough(upstream_model_forward),
                diffusion_model,
            ),
        )
        transformer_options["star7_h3_output_finite_guard"] = NODE_VERSION

    if attention_backend in strict_sla_backends:
        for index, block in enumerate(diffusion_model.blocks):
            block._star7_block_index = index
            block.attn._star7_auto_fp16_exact = auto_sm75_fp16_exact
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                MethodType(_minimax_sla_forward, block.attn),
            )
            block_path = f"diffusion_model.blocks.{index}.forward"
            upstream_block_forward = patched.object_patches.get(
                block_path, block.forward
            )
            if auto_sm75_fp16_exact:
                patched.add_object_patch(
                    block_path,
                    MethodType(
                        _make_chunked_h3_block_forward(True, h3_model), block
                    ),
                )
            else:
                patched.add_object_patch(
                    block_path,
                    MethodType(
                        _sla_segment_passthrough(
                            upstream_block_forward, block_index=index
                        ), block
                    ),
                )
        attention_patch_name = attention_backend
        sage_attention = False
        sla_attention = True
        if verbose and attention_backend == "sla_sm75_all_int8_experimental":
            _LOG.warning(
                "[Star7 H3 Chunk] All-INT8 SLA also approximates audio; use "
                "sla_sm75_qk_int8_pv_fp16 for the quality-control path"
            )
    elif attention_backend == "comfy_kitchen_int8":
        try:
            import comfy_kitchen
            ck_available = comfy_kitchen.int8_attention_is_available()
        except (ImportError, AttributeError, RuntimeError) as exc:
            ck_available = False
            if verbose:
                _LOG.warning(
                    "[Star7 H3 Chunk] Comfy Kitchen INT8 is unavailable (%s); "
                    "keeping the existing attention backend",
                    exc,
                )
        if ck_available:
            for index, block in enumerate(diffusion_model.blocks):
                block.attn._star7_auto_fp16_exact = auto_sm75_fp16_exact
                patched.add_object_patch(
                    f"diffusion_model.blocks.{index}.attn.forward",
                    MethodType(_minimax_ck_int8_attention_forward, block.attn),
                )
                if auto_sm75_fp16_exact:
                    patched.add_object_patch(
                        f"diffusion_model.blocks.{index}.forward",
                        MethodType(
                            _make_chunked_h3_block_forward(True, h3_model),
                            block,
                        ),
                    )
            attention_patch_name = "star7_comfy_kitchen_int8"
            sage_attention = False
            ck_attention = True
        else:
            _LOG.warning(
                "[Star7 H3 Chunk] Comfy Kitchen INT8 attention was requested "
                "but is unavailable; keeping the existing attention backend"
            )
    elif attention_backend != "existing":
        raise ValueError(f"unknown attention backend: {attention_backend}")

    # Outside the self-contained SM75 FP16 Exact path, patch only the MLP so an
    # upstream loader or third-party block patch keeps residual ownership. The
    # auto SM75 path intentionally owns the block because FP32 residuals and
    # FP16 branches are part of its overflow-safety contract.
    fp16_exact_active = star7_fp16 or auto_sm75_fp16_exact
    if int(mlp_chunk_tokens) != 0:
        mlp_forward = _make_chunked_h3_mlp_forward(fp16_exact_active)
        for index, block in enumerate(diffusion_model.blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                MethodType(mlp_forward, block.mlp),
            )

    # Dynamic VBAR prefetch is intentionally disabled for H3.  The upstream
    # queue can increase reserved VRAM without improving the supported low-VRAM
    # path. Keep the legacy function argument so old workflows remain loadable.
    transformer_options["prefetch_dynamic_vbars"] = False

    if verbose:
        architecture = (
            f"SM{capability[0]}{capability[1]}" if capability else "unknown-GPU"
        )
        precision = (
            "SM75 FP16 Exact (automatic)" if auto_sm75_fp16_exact
            else "FP16 Exact (upstream)" if star7_fp16
            else "upstream"
        )
        selected_attention = (
            attention_patch_name if sla_attention
            else "comfy-kitchen-int8" if ck_attention
            else "sage-qk-int8" if sage_attention
            else attention_patch_name
        )
        _LOG.info(
            "[Star7 H3 Chunk] Ready v%s | %s | attention=%s | precision=%s | "
            "chunks(RoPE/MLP/QKV)=%d/%d/%d | MLP-weight-reuse=%s | "
            "block-cache=%s | finite-guard=model-output%s",
            NODE_VERSION, architecture, selected_attention, precision,
            int(chunk_tokens), int(mlp_chunk_tokens), int(qkv_chunk_tokens),
            bool(reuse_mlp_weights), "external" if block_loop_cache else "none",
            "+SLA-block" if sla_attention else "",
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
                        "default": 8192,
                        "min": 0,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 RoPE sequence tokens per chunk. 2080 Ti 22GB: use 8192 after a safe 4096 validation run.",
                    },
                ),
                "auto_halve_on_oom": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If a reducible RoPE, MLP, or QKV temporary chunk OOMs, halve only that stage and retry down to 256 tokens. Zero still tries one full sequence first, then may auto-reduce.",
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
                        "default": 8192,
                        "min": 0,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 MLP tokens per chunk. Keeps the upstream block path while streaming the large expansion activation. The default 8192 is validated on the 22GB reference workflow.",
                    },
                ),
                "disable_dynamic_prefetch": (
                    "STRING",
                    {
                        "default": "实验功能已移除",
                        "multiline": False,
                        "tooltip": "提前加载下一层（实验功能已移除）。该字段仅用于兼容旧工作流，不再参与计算。",
                    },
                ),
                "qkv_chunk_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 QKV 投影临时显存分块。设为 0 会优先整段投影；若自动降档开启且整段 OOM，只降低 QKV 后重试。数值越小只会缩小投影临时张量，不会消除注意力所需的完整 Q/K/V。SLA 会直接保存 FP16 Q/K/V。默认 8192。",
                    },
                ),
                "reuse_mlp_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatic MLP weight strategy: reuse only when weights are static and unpatched; otherwise switch to streamed-safe mode.",
                    },
                ),
                "attention_backend": (
                    [
                        "existing",
                        "comfy_kitchen_int8",
                        "sla_sm75_qk_int8_pv_fp16",
                        "sla_sm75_all_int8_experimental",
                        "sla_sm80+_qk_int8_pv_fp16",
                    ],
                    {
                        "default": "comfy_kitchen_int8",
                        "tooltip": (
                            "existing keeps the incoming attention patch (for example KJ Sage). "
                            "comfy_kitchen_int8 selects ComfyUI's native INT8 attention and "
                            "overrides an earlier MiniMax Sage patch. Strict SLA targets 85% "
                            "dynamic video-block sparsity, protects target-audio queries, and "
                            "never falls back after failure. Choose "
                            "sla_sm75_qk_int8_pv_fp16 for the recommended SM75 path, "
                            "sla_sm75_all_int8_experimental for the faster but lower-precision "
                            "SM75 experiment, or "
                            "sla_sm80+_qk_int8_pv_fp16 for SM80+."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Low-VRAM MiniMax H3 patch. Chunks fused split-half RoPE and the large MLP "
        "expansion activation, preserves INT8/ConvRot weights, and keeps the upstream "
        "DiT block path for FP16/BF16, Sage, LoRA, and third-party compatibility. "
        "Can safely retry without AIMDO prefetch. "
        "Compatible with FP16 Exact Fix - Star7. The default Comfy Kitchen INT8 "
        "attention mode is approximate; select existing to preserve upstream attention math. "
        "Strict SLA is dependency-free from Sage, includes a native target-audio guard, "
        "and errors instead of silently falling back."
    )

    def patch(
        self, model, chunk_tokens=8192, auto_halve_on_oom=True, verbose=True,
        mlp_chunk_tokens=8192, disable_dynamic_prefetch=True,
        qkv_chunk_tokens=8192,
        reuse_mlp_weights=True, attention_backend="comfy_kitchen_int8", unique_id=None,
    ):
        return (install_model_patch(
            model, chunk_tokens, auto_halve_on_oom, verbose,
            mlp_chunk_tokens, disable_dynamic_prefetch, reuse_mlp_weights,
            attention_backend, unique_id, qkv_chunk_tokens=qkv_chunk_tokens,
        ),)


def _matched_reference_size(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    multiple: int = 32,
) -> tuple[int, int]:
    """Cap reference-video area to the output area without changing its aspect."""
    values = (source_width, source_height, target_width, target_height)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Reference and target dimensions must be positive")
    multiple = max(1, int(multiple))
    source_area = int(source_width) * int(source_height)
    target_area = int(target_width) * int(target_height)
    if source_area <= target_area:
        return int(source_width), int(source_height)

    scale = math.sqrt(target_area / source_area)
    width = max(multiple, round(source_width * scale / multiple) * multiple)
    height = max(multiple, round(source_height * scale / multiple) * multiple)
    while width * height > target_area and (width > multiple or height > multiple):
        width_loss = abs((width - multiple) / max(height, 1) - source_width / source_height)
        height_loss = abs(width / max(height - multiple, 1) - source_width / source_height)
        if width > multiple and (height <= multiple or width_loss <= height_loss):
            width -= multiple
        else:
            height -= multiple
    return width, height


def _long_edge_reference_size(
    source_width: int,
    source_height: int,
    max_long_edge: int,
    allow_upscale: bool = False,
    multiple: int = 32,
) -> tuple[int, int]:
    """Fit a reference video to a long-edge limit and return an H3-ready canvas."""
    values = (source_width, source_height, max_long_edge)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Reference dimensions and max_long_edge must be positive")
    multiple = max(1, int(multiple))
    limit = max(multiple, int(max_long_edge) // multiple * multiple)
    source_width = int(source_width)
    source_height = int(source_height)
    source_long_edge = max(source_width, source_height)
    should_resize = source_long_edge > limit or (
        bool(allow_upscale) and source_long_edge < limit
    )
    scale = limit / source_long_edge if should_resize else 1.0
    raw_width = source_width * scale
    raw_height = source_height * scale
    width = max(multiple, round(raw_width / multiple) * multiple)
    height = max(multiple, round(raw_height / multiple) * multiple)

    if not allow_upscale:
        # Keep the checkbox strict relative to the decoded source. T8 would
        # otherwise round an odd input dimension upward on its own.
        source_floor_width = max(multiple, source_width // multiple * multiple)
        source_floor_height = max(multiple, source_height // multiple * multiple)
        width = min(width, source_floor_width)
        height = min(height, source_floor_height)
    if width >= height and width > limit:
        width = limit
    elif height > width and height > limit:
        height = limit
    return width, height


def _align_h3_reference_frame_count(frame_count: int) -> int:
    """Trim a decoded 24fps reference to MiniMax H3's 17n+5 grid."""
    frame_count = min(360, int(frame_count))
    if frame_count < 5:
        return frame_count
    return frame_count - ((frame_count - 5) % 17)


def _star7_ffmpeg_path() -> str:
    """Find FFmpeg without depending on VideoHelperSuite internals."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        candidate = get_ffmpeg_exe()
        if candidate and os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    candidate = shutil.which("ffmpeg")
    if candidate:
        return candidate
    raise RuntimeError(
        "FFmpeg was not found. Install imageio-ffmpeg or make ffmpeg available on PATH."
    )


def _hidden_subprocess_kwargs() -> dict:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _video_stream_info(video_path: str) -> tuple[int, int, bool]:
    """Read the displayed frame dimensions and whether an audio stream exists."""
    import av

    with av.open(video_path, mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        source_width, source_height = int(stream.width), int(stream.height)
        try:
            first_frame = next(container.decode(stream))
            rotation = int(round(float(getattr(first_frame, "rotation", 0)))) % 360
            if rotation in {90, 270}:
                source_width, source_height = source_height, source_width
        except StopIteration:
            raise ValueError(f"No video frames found in {video_path}")
        return source_width, source_height, bool(container.streams.audio)


class MiniMaxH3ReferenceVideoLoadStar7:
    """Minimal 24fps H3 reference loader with a target-aware long-edge limit."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "max_long_edge": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 32,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "The reference video keeps its aspect ratio and is fitted to this H3-aligned long edge.",
                    },
                ),
                "allow_upscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Disabled avoids spending H3 reference tokens on interpolated detail. Enable only for structure/motion A/B tests.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("reference_video", "reference_audio", "frame_count", "report")
    FUNCTION = "load"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Loads a MiniMax H3 reference video at the mandatory 24fps, limits it to 15 seconds, "
        "fits its long edge without changing orientation, aligns frames to 17n+5, and extracts "
        "the matching soundtrack. FFmpeg is resolved independently from VideoHelperSuite."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, video, **_kwargs):
        import folder_paths

        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    @classmethod
    def IS_CHANGED(cls, video, **_kwargs):
        import folder_paths

        path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(path)

    def load(self, video, max_long_edge=1344, allow_upscale=False):
        import folder_paths

        video_path = folder_paths.get_annotated_filepath(video)
        source_width, source_height, has_audio = _video_stream_info(video_path)
        width, height = _long_edge_reference_size(
            source_width, source_height, int(max_long_edge), bool(allow_upscale), 32,
        )
        ffmpeg = _star7_ffmpeg_path()
        command = [
            ffmpeg,
            "-v", "error",
            "-i", video_path,
            "-t", "15",
            "-an",
            "-vf", f"fps=24,scale={width}:{height}:flags=lanczos",
            "-frames:v", "360",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        result = subprocess.run(
            command, capture_output=True, check=False, **_hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"FFmpeg reference-video decode failed: {detail}")
        frame_bytes = width * height * 3
        decoded_count = len(result.stdout) // frame_bytes
        aligned_count = _align_h3_reference_frame_count(decoded_count)
        if aligned_count < 5:
            raise ValueError("MiniMax H3 reference video must contain at least 5 frames at 24fps")
        usable_bytes = aligned_count * frame_bytes
        raw = torch.frombuffer(bytearray(result.stdout[:usable_bytes]), dtype=torch.uint8)
        frames = raw.reshape(aligned_count, height, width, 3).to(torch.float32).div_(255.0)

        audio = None
        audio_status = "none"
        if has_audio:
            audio_rate = 44100
            duration = aligned_count / 24.0
            audio_command = [
                ffmpeg,
                "-v", "error",
                "-i", video_path,
                "-t", f"{duration:.9f}",
                "-vn",
                "-ac", "2",
                "-ar", str(audio_rate),
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "pipe:1",
            ]
            audio_result = subprocess.run(
                audio_command, capture_output=True, check=False, **_hidden_subprocess_kwargs(),
            )
            if audio_result.returncode != 0:
                detail = audio_result.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"FFmpeg reference-audio decode failed: {detail}")
            sample_values = torch.frombuffer(
                bytearray(audio_result.stdout), dtype=torch.float32,
            )
            complete_values = sample_values.numel() // 2 * 2
            if complete_values:
                waveform = sample_values[:complete_values].reshape(-1, 2).transpose(0, 1)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": audio_rate}
                audio_status = "stereo-44100Hz"

        scale_direction = "same"
        if width * height < source_width * source_height:
            scale_direction = "down"
        elif width * height > source_width * source_height:
            scale_direction = "up"
        report = (
            f"24fps | frames={aligned_count} | {source_width}x{source_height} -> {width}x{height} "
            f"({scale_direction}) | max_long_edge={int(max_long_edge)} | "
            f"allow_upscale={bool(allow_upscale)} | audio={audio_status}"
        )
        _LOG.info("[Star7 H3 Ref Load] %s", report)
        return frames, audio, aligned_count, report


class MiniMaxH3ReferenceVideoOptimizeStar7:
    """Resize reference frames before the H3 VAE so they do not exceed the output area."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_video": ("IMAGE",),
                "target_width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 16384, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384, "step": 32},
                ),
                "resize_policy": (
                    ["match_output_area", "match_output_size", "keep_original"],
                    {
                        "default": "match_output_area",
                        "tooltip": (
                            "match_output_area preserves the reference aspect ratio and only "
                            "downscales when its pixel area exceeds the output canvas."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("reference_video", "report")
    FUNCTION = "optimize"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Pre-resizes MiniMax H3 reference-video frames before Video VAE encoding. "
        "This reduces both conditioning time and reference tokens while preserving every "
        "frame and leaving audio untouched. Pure PyTorch; compatible with RTX 20-50 series."
    )

    def optimize(
        self, reference_video, target_width=1344, target_height=768,
        resize_policy="match_output_area",
    ):
        if not isinstance(reference_video, torch.Tensor) or reference_video.ndim != 4:
            raise ValueError("reference_video must be IMAGE [frames, height, width, channels]")
        frames, source_height, source_width = map(int, reference_video.shape[:3])
        if frames < 1:
            raise ValueError("reference_video contains no frames")

        policy = str(resize_policy)
        if policy == "keep_original":
            width, height = source_width, source_height
        elif policy == "match_output_size":
            width = max(32, round(int(target_width) / 32) * 32)
            height = max(32, round(int(target_height) / 32) * 32)
        elif policy == "match_output_area":
            width, height = _matched_reference_size(
                source_width, source_height, int(target_width), int(target_height), 32,
            )
        else:
            raise ValueError(f"Unknown reference resize policy: {resize_policy}")

        if (width, height) == (source_width, source_height):
            output = reference_video
        else:
            import comfy.utils

            samples = reference_video[..., :3].movedim(-1, 1)
            output = comfy.utils.common_upscale(
                samples, width, height, "lanczos", "disabled",
            ).movedim(1, -1)

        source_area = source_width * source_height
        output_area = width * height
        reduction = max(0.0, 100.0 * (1.0 - output_area / source_area))
        signature = (frames, source_width, source_height, width, height, policy)
        if signature not in _LOGGED_REFERENCE_VIDEO_SHAPES:
            _LOGGED_REFERENCE_VIDEO_SHAPES.add(signature)
            _LOG.info(
                "[Star7 H3 Ref] %d frames | %dx%d -> %dx%d | pixels -%.1f%% | policy=%s",
                frames, source_width, source_height, width, height, reduction, policy,
            )
        report = (
            f"frames={frames} | {source_width}x{source_height} -> {width}x{height} | "
            f"pixels -{reduction:.1f}% | policy={policy} | audio=unchanged"
        )
        return output, report


class MiniMaxH3RoPEChunkPatch(MiniMaxH3ActivationChunkStar7):
    """Legacy class ID retained only so existing workflows keep loading."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": MiniMaxH3ActivationChunkStar7,
    "MiniMaxH3ReferenceVideoLoadStar7": MiniMaxH3ReferenceVideoLoadStar7,
    "MiniMaxH3ReferenceVideoOptimizeStar7": MiniMaxH3ReferenceVideoOptimizeStar7,
    # Compatibility alias for workflows saved before the package was renamed.
    "MiniMaxH3RoPEChunkPatch": MiniMaxH3RoPEChunkPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": "MiniMax H3 Activation Chunk (RoPE + MLP) - Star7",
    "MiniMaxH3ReferenceVideoLoadStar7": "MiniMax H3 Reference Video Load - Star7",
    "MiniMaxH3ReferenceVideoOptimizeStar7": "MiniMax H3 Reference Video Optimize - Star7",
    "MiniMaxH3RoPEChunkPatch": "MiniMax H3 RoPE Chunk Patch (Legacy) - Star7",
}
