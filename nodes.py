import gc
import logging
import os
import shutil
import subprocess
import time
import weakref
from fractions import Fraction
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F

_LOG = logging.getLogger("MiniMaxH3ActivationChunkStar7")
NODE_VERSION = "2.12.0"
SM75_QKV_QUALITY_CHUNK = 4096
HYBRID_ALL_INT8_BACKEND_NAME = "hybrid_sm75_ck_sla_all_int8"
SM86PLUS_BACKEND_NAME = "sla_sm80+_qk_int8_pv_bf16"
LEGACY_SM86PLUS_FP16_BACKEND_NAME = "sla_sm80+_qk_int8_pv_fp16"
SM75_ALL_INT8_BACKEND_NAME = "sla_sm75_all_int8"
LEGACY_SM75_ALL_INT8_BACKEND_NAME = "sla_sm75_all_int8_experimental"
SM86PLUS_ALL_INT8_BACKEND_NAME = "sla_sm80+_all_int8"
LEGACY_SM86PLUS_ALL_INT8_BACKEND_NAME = "sla_sm80+_all_int8_experimental"
HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME = "hybrid_sm80+_ck_sla_all_int8"
HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME = "hybrid_sm80+_ck_sla_qk_int8_pv_bf16"
LEGACY_HYBRID_SM86PLUS_CK_SLA_FP16_BACKEND_NAME = "hybrid_sm80+_ck_sla_qk_int8_pv_fp16"
SOL_SM75_BACKEND_NAME = "sol_sm75_qk_int8_pv_fp16"
SOL_SM75_ALL_INT8_BACKEND_NAME = "sol_sm75_all_int8"
LEGACY_SOL_SM75_ALL_INT8_BACKEND_NAME = "sol_sm75_all_int8_experimental"
SOL_SM86PLUS_BACKEND_NAME = "sol_sm80+_bf16_official"
SOL_SM86PLUS_ALL_INT8_BACKEND_NAME = "sol_sm80+_all_int8"
LEGACY_SOL_SM86PLUS_ALL_INT8_BACKEND_NAME = "sol_sm80+_all_int8_experimental"
HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME = "hybrid_sm75_ck_sol_all_int8"
HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME = "hybrid_sm80+_ck_sol_all_int8"
HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME = "hybrid_sm80+_ck_sol_bf16_official"
LEGACY_SM80PLUS_BACKEND_NAME = "sla_sm80+_qk_int8_pv_fp16"
HYBRID_BACKEND_NAMES = {
    HYBRID_ALL_INT8_BACKEND_NAME,
    HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME,
    HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME,
    LEGACY_HYBRID_SM86PLUS_CK_SLA_FP16_BACKEND_NAME,
    HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
    HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
    HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
}
HYBRID_GUARD_RATIO = Fraction(1, 6)


def _attention_backend_choices():
    common = ["existing", "comfy_kitchen_int8"]
    sm75 = [
        "sla_sm75_qk_int8_pv_fp16",
        SM75_ALL_INT8_BACKEND_NAME,
        SOL_SM75_ALL_INT8_BACKEND_NAME,
        HYBRID_ALL_INT8_BACKEND_NAME,
        HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
    ]
    sm80plus = [
        SM86PLUS_BACKEND_NAME,
        SM86PLUS_ALL_INT8_BACKEND_NAME,
        SOL_SM86PLUS_BACKEND_NAME,
        SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
    ]
    return common + sm75 + sm80plus

_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE = None
_PATCHED_CK = None
_LOGGED_ROPE_SHAPES = set()
_LOGGED_MLP_SHAPES = set()
_PROFILED_MLP_SHAPES = set()
_LOGGED_SLA_SHAPES = set()
_LOGGED_SOL_SHAPES = set()
_PROFILED_QKV_STAGES = set()
_LOGGED_SLA_ENVIRONMENTS = set()
_LAST_FAILED_SLA_BLOCK = None
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
    "auto_sla_probe": False,
    "verbose": True,
    "reuse_mlp_weights": True,
    "node_id": None,
    "fp16_exact_present": False,
    "hybrid_sla_backend": None,
}


def _h3_memory_debug_enabled() -> bool:
    return os.environ.get("STAR7_H3_MEMORY_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _log_h3_cuda_memory(stage: str, device, block_index=None) -> None:
    if (
        not _h3_memory_debug_enabled()
        or device.type != "cuda"
        or (block_index is not None and int(block_index) != 0)
    ):
        return
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    _LOG.info(
        "[Star7 H3 Memory] %s | block=%s | allocated=%.2fGiB | reserved=%.2fGiB",
        stage, block_index if block_index is not None else "n/a",
        allocated, reserved,
    )


def _weak_callable(value):
    """Keep an upstream bound model method callable without retaining its owner."""
    if value is None:
        return None
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    if owner is None or isinstance(owner, weakref.ProxyTypes):
        return value

    owner_ref = weakref.ref(owner)

    def call(*args, **kwargs):
        current = owner_ref()
        if current is None:
            raise ReferenceError("Star7 H3 Chunk wrapper owner was released")
        return function(current, *args, **kwargs)

    return call


def _weak_method(owner, function):
    """Bind a model patch through a weak proxy instead of the model module."""
    return MethodType(function, weakref.proxy(owner))


def _current_cuda_capability():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def _default_qkv_chunk_tokens() -> int:
    return SM75_QKV_QUALITY_CHUNK if _current_cuda_capability() == (7, 5) else 8192


def _quality_limited_qkv_chunk(requested: int, capability) -> tuple[int, bool]:
    requested = int(requested)
    if capability != (7, 5) or (requested != 0 and requested <= SM75_QKV_QUALITY_CHUNK):
        return requested, False
    return SM75_QKV_QUALITY_CHUNK, True


def _sm75_qkv_reuse_path(x: torch.Tensor) -> bool:
    """Use the validated resident-weight path for any SM75 QKV tile <= 4096."""
    if x.device.type != "cuda":
        return False
    try:
        capability = torch.cuda.get_device_capability(x.device)
    except Exception:
        return False
    effective = int(_CONFIG["effective_qkv_chunk_tokens"])
    return capability == (7, 5) and 0 < effective <= SM75_QKV_QUALITY_CHUNK


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


def _set_sequence_status(
    kind: str, sequence_length: int, reason_override: Optional[str] = None,
) -> None:
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
            reason_override
            or ("sequence_limit" if learned > 0 and actual < learned else "active")
        )


def _limit_sm75_qkv_chunk_for_quality(x: torch.Tensor) -> bool:
    """Keep Turing QKV projection on the validated speech-stable tile size.

    A same-seed four-step H3 run changes decoded PCM between 4,096 and 8,192
    even though isolated INT8 projection and RoPE tests are bitwise chunk
    invariant. Keep the validated 4,096 upper bound on Turing; smaller
    OOM-learned values remain valid, while SM80+ keeps the requested value.
    """
    if x.device.type != "cuda":
        return False
    # Direct SM75 INT8 projection and RoPE tests are bitwise chunk invariant,
    # including the irregular tail. The complete H3 workflow is not: changing
    # this allocation/call schedule changes downstream long-sequence results,
    # and four diffusion steps amplify that difference in reference speech.
    # Apply the validated cap regardless of how ComfyUI materialized LoRA.
    try:
        capability = torch.cuda.get_device_capability(x.device)
    except Exception:
        return False
    configured = int(_CONFIG["qkv_chunk_tokens"])
    _configured_limit, applies = _quality_limited_qkv_chunk(configured, capability)
    if not applies:
        return False
    current = int(_CONFIG["effective_qkv_chunk_tokens"])
    if current == 0 or current > SM75_QKV_QUALITY_CHUNK:
        _CONFIG["effective_qkv_chunk_tokens"] = SM75_QKV_QUALITY_CHUNK
        current = SM75_QKV_QUALITY_CHUNK
    # If OOM fallback already learned a smaller value, report that as a
    # reduction rather than mislabelling it as the 4096 quality cap.
    return current == SM75_QKV_QUALITY_CHUNK


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
    """Whether a ComfyUI Linear supports one cast per chunked operation."""
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


def _resident_qkv_caller(linear, x: torch.Tensor):
    """Prepare one private QKV weight snapshot for all token chunks."""
    import comfy.ops

    if not _linear_can_reuse_weights(linear):
        raise RuntimeError("QKV Linear implementation does not support resident weight reuse")
    quant_mode = _linear_quantization_mode(linear)
    stage_dtype = linear.weight.dtype if quant_mode == "weight-only" else x.dtype
    with comfy.ops.CastBiasWeightContext(
        linear,
        input=None,
        dtype=stage_dtype,
        device=x.device,
        bias_dtype=x.dtype,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=quant_mode is not None,
    ) as (weight, bias):
        private_weight = weight.detach().clone() if weight is not None else None
        private_bias = bias.detach().clone() if bias is not None else None
    if quant_mode == "weight-only":
        private_weight = private_weight.to(dtype=x.dtype)

    def call(value):
        return _resident_linear_forward(
            linear, value, private_weight, private_bias, quant_mode
        )

    prepared_backend = (
        "quantized"
        if isinstance(private_weight, comfy.ops.QuantizedTensor)
        else "dense"
    )
    return call, prepared_backend


def _run_chunked_h3_mlp(
    self,
    x: torch.Tensor,
    upstream_forward=None,
) -> torch.Tensor:
    """Run the native or upstream-patched row-independent MLP in token chunks."""
    configured_chunk = int(_CONFIG["effective_mlp_chunk_tokens"])
    import comfy.ops

    if x.ndim != 2:
        if upstream_forward is not None:
            return upstream_forward(x)
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    seq_len = x.shape[0]
    chunk = seq_len if configured_chunk == 0 else max(256, configured_chunk)
    _set_sequence_status("MLP", seq_len)
    # H3 MLP is row-independent and the block does not need the normalized
    # input after each row has been transformed. Reuse ``x`` as the output
    # buffer when the dtype is unchanged instead of retaining a second full-S
    # hidden-state allocation.
    output = x if bool(getattr(self, "_star7_reuse_mlp_input", False)) else None
    current_chunk = min(chunk, seq_len)
    auto_halve = bool(_CONFIG["auto_halve_on_oom"])
    mode = "upstream-preserved" if upstream_forward is not None else "native"
    shape_key = (
        seq_len, x.shape[1], current_chunk, x.dtype, x.device.type, mode,
    )
    block_index = getattr(self, "_star7_block_index", None)

    start = 0
    calls = 0
    while start < seq_len:
        end = min(start + current_chunk, seq_len)
        expanded = result = None
        try:
            chunk_input = x[start:end]
            _debug_sla_tensor(
                f"MLP input chunk [{start}:{end}]", chunk_input,
                block_index, row_dim=0, check_fp16_range=True,
            )
            if upstream_forward is not None:
                result = upstream_forward(chunk_input)
            else:
                expanded = self.fc1(chunk_input)
                _debug_sla_tensor(
                    f"MLP fc1 chunk [{start}:{end}]", expanded,
                    block_index, row_dim=0, check_fp16_range=True,
                )
                result = comfy.ops.linear_input_act(self.fc2, expanded, "swiglu")
            _debug_sla_tensor(
                f"MLP output chunk [{start}:{end}]", result,
                block_index, row_dim=0,
            )
            expected = (end - start, x.shape[1])
            if result.shape != expected:
                raise RuntimeError(
                    "unexpected H3 MLP chunk output: "
                    f"got shape={tuple(result.shape)}, expected shape={expected}"
                )
            if output is x and result.dtype != x.dtype:
                # Preserve the previous safe behavior for an upstream MLP
                # that deliberately changes the block compute dtype.
                output = None
            if output is None:
                output = torch.empty(
                    (seq_len, x.shape[1]), dtype=result.dtype, device=result.device
                )
            elif result.dtype != output.dtype:
                raise RuntimeError(
                    f"H3 MLP output dtype changed between chunks: "
                    f"{output.dtype} -> {result.dtype}"
                )
            output[start:end].copy_(result)
            del result, expanded, chunk_input
            start = end
            calls += 1
        except Exception as exc:
            if not (_is_cuda_oom(exc) and auto_halve and current_chunk > 256):
                raise
            new_chunk = max(256, current_chunk // 2)
            _remember_effective_chunk("MLP", current_chunk, new_chunk)
            del result, expanded
            exc.__traceback__ = None
            _clear_cuda_after_oom(x.device)
            current_chunk = new_chunk

    profile_key = shape_key
    do_profile = bool(_CONFIG["verbose"] and profile_key not in _PROFILED_MLP_SHAPES)
    if do_profile:
        _PROFILED_MLP_SHAPES.add(profile_key)
        _LOG.info(
            "[Star7 H3 Chunk] First-block MLP | S=%d | chunk=%d x %d | mode=%s",
            seq_len, current_chunk, calls, mode,
        )

    _log_h3_cuda_memory("after-mlp", x.device, block_index=block_index)

    return output


def _chunked_h3_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Native-dtype entry point kept for direct tests and compatibility."""
    return _run_chunked_h3_mlp(self, x)


def _make_chunked_h3_mlp_forward(upstream_forward=None):
    upstream_forward = _weak_callable(upstream_forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_chunked_h3_mlp(self, x, upstream_forward=upstream_forward)
    forward._star7_wrapper_kind = "mlp-chunk-upstream"
    forward._star7_original_forward = upstream_forward
    return forward


def _sla_debug_block() -> Optional[int]:
    requested = os.environ.get("STAR7_SLA_DEBUG_BLOCK", "").strip()
    if requested:
        try:
            return int(requested)
        except ValueError:
            _LOG.warning(
                "[Star7 H3 Chunk] Ignoring invalid STAR7_SLA_DEBUG_BLOCK=%r",
                requested,
            )
    return _LAST_FAILED_SLA_BLOCK


def _is_sla_debug_block(block_index) -> bool:
    selected = _sla_debug_block()
    return selected is not None and block_index is not None and int(block_index) == selected


def _auto_sla_probe_enabled() -> bool:
    """Probe every SLA stage only on newer architectures during first-run triage."""
    return bool(_CONFIG.get("auto_sla_probe"))


def _auto_sla_probe_for_capability(capability) -> bool:
    return bool(capability and capability >= (10, 0))


def select_hybrid_backend(
    step_index: int,
    total_steps: int,
    guard_ratio=HYBRID_GUARD_RATIO,
) -> str:
    """Select CK or SLA from normalized sampler progress.

    The default ratio is represented as an exact fraction so the boundary
    decisions do not depend on floating-point rounding.
    """
    total_steps = int(total_steps)
    step_index = int(step_index)
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if step_index < 0 or step_index >= total_steps:
        raise ValueError(
            f"step_index must be in [0, {total_steps}), got {step_index}"
        )
    ratio = guard_ratio
    if not isinstance(ratio, Fraction):
        ratio = Fraction(str(ratio)).limit_denominator(1_000_000)
    if ratio <= 0 or ratio >= Fraction(1, 2):
        raise ValueError(f"guard_ratio must be between 0 and 1/2, got {guard_ratio}")
    if total_steps <= 1:
        return "CK"

    span = total_steps - 1
    numerator, denominator = ratio.numerator, ratio.denominator
    use_ck = (
        denominator * step_index <= numerator * span
        or denominator * (span - step_index) <= numerator * span
    )
    return "CK" if use_ck else "SLA"


def _hybrid_sampling_context(transformer_options: dict) -> tuple[int, int, str]:
    """Resolve the real sampler step from ComfyUI's sigma context."""
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigma = transformer_options.get("sigmas")
    if not torch.is_tensor(sample_sigmas) or not torch.is_tensor(current_sigma):
        raise RuntimeError(
            "Star7 Hybrid Attention requires ComfyUI sampler context: "
            "transformer_options.sample_sigmas and transformer_options.sigmas "
            "were not provided. The step cannot be inferred from block calls."
        )

    schedule = sample_sigmas.detach().flatten()
    current = current_sigma.detach().flatten()
    if schedule.numel() < 2 or current.numel() < 1:
        raise RuntimeError(
            "Star7 Hybrid Attention received an invalid sampler sigma schedule; "
            "at least one sampling sigma and a terminal sigma are required."
        )
    current_value = current[0].to(schedule.device, dtype=schedule.dtype)
    if not bool(torch.isclose(current, current[0], rtol=1e-5, atol=1e-6).all().item()):
        raise RuntimeError(
            "Star7 Hybrid Attention received different sigmas in one H3 model "
            "call; refusing to guess a shared backend."
        )

    total_steps = int(schedule.numel() - 1)
    active_schedule = schedule[:total_steps]
    matches = torch.isclose(active_schedule, current_value, rtol=1e-4, atol=1e-6)
    matching_steps = torch.nonzero(matches, as_tuple=False).flatten()
    if matching_steps.numel() == 0:
        raise RuntimeError(
            "Star7 Hybrid Attention could not map the current sigma to "
            "ComfyUI's sample_sigmas schedule; refusing to guess the step."
        )

    cached = transformer_options.get("_star7_hybrid_selection")
    step_index = int(matching_steps[0].item())
    if cached is not None:
        cached_sigma, cached_total, cached_step = cached
        if (
            cached_total == total_steps
            and abs(float(cached_sigma) - float(current_value.item())) <= 1e-5
        ):
            step_index = int(cached_step)
        elif cached_total == total_steps:
            later = matching_steps[matching_steps > int(cached_step)]
            if later.numel():
                step_index = int(later[0].item())

    backend = select_hybrid_backend(step_index, total_steps)
    transformer_options["_star7_hybrid_selection"] = (
        float(current_value.item()), total_steps, step_index
    )
    return step_index, total_steps, backend


def _sampling_step_context(transformer_options: dict):
    """Resolve a sampler step without requiring Hybrid attention."""
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigma = transformer_options.get("sigmas")
    if not torch.is_tensor(sample_sigmas) or not torch.is_tensor(current_sigma):
        return None

    schedule = sample_sigmas.detach().flatten()
    current = current_sigma.detach().flatten()
    if schedule.numel() < 2 or current.numel() < 1:
        return None

    current_value = current[0].to(schedule.device, dtype=schedule.dtype)
    active_schedule = schedule[:-1]
    matches = torch.isclose(active_schedule, current_value, rtol=1e-4, atol=1e-6)
    matching_steps = torch.nonzero(matches, as_tuple=False).flatten()
    if matching_steps.numel() == 0:
        return None
    return int(matching_steps[0].item()), int(schedule.numel() - 1)


def _step_backend_label(transformer_options: dict):
    configured = _CONFIG.get("attention_backend", "existing")
    if configured in HYBRID_BACKEND_NAMES:
        try:
            _, _, backend = _hybrid_sampling_context(transformer_options)
            if backend == "SLA" and configured in {
                HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
                HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
                HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
            }:
                backend = "Sol"
        except (RuntimeError, ValueError):
            backend = "Hybrid"
    elif configured == "comfy_kitchen_int8":
        backend = "CK"
    elif configured.startswith("sla_"):
        backend = "SLA"
    else:
        backend = configured
    return configured, backend


def _step_timing_start(transformer_options: dict, x, step_index: int, total_steps: int):
    if x is None or not torch.is_tensor(x):
        return
    if x.device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(x.device))
        transformer_options["_star7_step_timing"] = {
            "step": step_index,
            "total": total_steps,
            "start_event": event,
            "start_time": None,
        }
    else:
        transformer_options["_star7_step_timing"] = {
            "step": step_index,
            "total": total_steps,
            "start_event": None,
            "start_time": time.perf_counter(),
        }


def _step_timing_finish(
    transformer_options: dict,
    device,
    step_index: int,
    total_steps: int,
    model,
) -> None:
    timing = transformer_options.pop("_star7_step_timing", None)
    if not isinstance(timing, dict) or timing.get("step") != step_index:
        return

    start_event = timing.get("start_event")
    if start_event is not None and device.type == "cuda":
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(torch.cuda.current_stream(device))
        end_event.synchronize()
        elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = timing.get("start_time")
        if start_time is None:
            return
        elapsed_seconds = time.perf_counter() - start_time

    configured, backend = _step_backend_label(transformer_options)
    extra = f" | mode={configured}"
    if configured in HYBRID_BACKEND_NAMES:
        extra += f" | guard_ratio={float(HYBRID_GUARD_RATIO):.4f}"
    _LOG.info(
        "[Star7 H3] step %d/%d -> %-8s | %7.2fs/it | blocks=%d%s",
        step_index + 1, total_steps, backend, elapsed_seconds,
        len(getattr(model, "blocks", ())), extra,
    )


def _complete_automatic_sla_debug(block_index) -> None:
    global _LAST_FAILED_SLA_BLOCK
    if os.environ.get("STAR7_SLA_DEBUG_BLOCK", "").strip():
        return
    if _LAST_FAILED_SLA_BLOCK is not None and int(block_index) == _LAST_FAILED_SLA_BLOCK:
        _LOG.info(
            "[Star7 H3 Chunk] Targeted SLA diagnostics completed cleanly for block %d",
            block_index,
        )
        _LAST_FAILED_SLA_BLOCK = None


def _bad_row_ranges(value: torch.Tensor, row_dim: int = 0, limit: int = 6) -> dict:
    """Compact spatial summary for a failed tensor without dumping its values."""
    invalid = ~torch.isfinite(value)
    row_dim %= max(1, value.ndim)
    reduce_dims = tuple(index for index in range(value.ndim) if index != row_dim)
    bad_rows_mask = invalid.any(dim=reduce_dims) if reduce_dims else invalid
    rows = bad_rows_mask.nonzero(as_tuple=False).flatten().tolist()
    ranges = []
    for row in rows:
        if ranges and row == ranges[-1][1] + 1:
            ranges[-1][1] = row
        else:
            ranges.append([row, row])
    heads = []
    if value.ndim == 4:
        # SLA tensors use [B,H,S,D].
        heads = invalid.any(dim=(0, 2, 3)).nonzero(as_tuple=False).flatten().tolist()
    return {
        "bad_rows": len(rows),
        "first_bad_row": rows[0] if rows else None,
        "last_bad_row": rows[-1] if rows else None,
        "ranges": tuple(tuple(item) for item in ranges[:limit]),
        "bad_heads": tuple(heads[:16]),
    }


def _sla_architecture_note_for_capability(capability) -> str:
    selected = str(_CONFIG.get("attention_backend", ""))
    if "sol" in selected:
        if capability == (7, 5):
            return "SM75 used the Star7 Q64/K64 Sol exact-plus-centroid CUDA path"
        if capability:
            mode = (
                "NVIDIA official BF16 exact+approx Sol"
                if selected == SOL_SM86PLUS_BACKEND_NAME
                else "Star7 Q64/K64 exact-plus-centroid All-INT8 Sol"
            )
            return f"SM{capability[0]}{capability[1]} used {mode}"
    if capability == (7, 5):
        companion = (
            "detected" if _CONFIG.get("fp16_exact_present") else "not detected"
        )
        return (
            f"SM75 used the upstream precision path; the separate FP16 Exact "
            f"companion was {companion}"
        )
    if capability:
        return (
            f"SM{capability[0]}{capability[1]} used Triton SLA with FP16 "
            "Q/K/V buffers and upstream model precision"
        )
    return "the active SLA precision path is unknown"


def _sla_architecture_note(value: torch.Tensor) -> str:
    capability = None
    if value.device.type == "cuda":
        try:
            capability = torch.cuda.get_device_capability(value.device)
        except Exception:
            pass
    return _sla_architecture_note_for_capability(capability)


def _raise_strict_sla_failure(
    value: torch.Tensor,
    stage: str,
    block_index=None,
    row_dim: int = 0,
) -> None:
    global _LAST_FAILED_SLA_BLOCK
    nan_count = int(torch.isnan(value).sum().item())
    inf_count = int(torch.isinf(value).sum().item())
    structure = _bad_row_ranges(value, row_dim=row_dim)
    if block_index is not None:
        _LAST_FAILED_SLA_BLOCK = int(block_index)
    heads = (
        f", bad-heads={structure['bad_heads']}" if structure["bad_heads"] else ""
    )
    if _auto_sla_probe_enabled():
        rerun = " Automatic new-architecture stage diagnostics were active for this run."
    else:
        rerun = (
            f" Detailed diagnostics are armed for block {block_index} on the next run."
            if block_index is not None else ""
        )
    raise RuntimeError(
        f"Star7 strict sparse attention detected NaN/Inf after {stage} "
        f"(nan={nan_count}, inf={inf_count}, bad-rows={structure['bad_rows']}, "
        f"first={structure['first_bad_row']}, last={structure['last_bad_row']}, "
        f"ranges={structure['ranges']}{heads}). The task was stopped before VAE "
        f"decode; {_sla_architecture_note(value)}.{rerun} Report the Star7 version, "
        "failing stage and model/LoRA/sampler combination. No CK/Sage fallback "
        "was attempted."
    )


def _require_strict_sla_finite(
    value: torch.Tensor, stage: str, block_index=None, row_dim: int = 0,
    transformer_options: Optional[dict] = None,
) -> None:
    """Stop strict SLA jobs at the first observable non-finite model stage."""
    selected_backend = str(_CONFIG.get("attention_backend", ""))
    if selected_backend in HYBRID_BACKEND_NAMES:
        hybrid_selection = (
            transformer_options.get("_star7_hybrid_selection")
            if transformer_options else None
        )
        if not hybrid_selection or select_hybrid_backend(
            hybrid_selection[2], hybrid_selection[1]
        ) != "SLA":
            return
    elif not (
        selected_backend.startswith("sla_")
        or selected_backend.startswith("sol_")
    ):
        return
    if bool(torch.isfinite(value).all().item()):
        return
    _raise_strict_sla_failure(
        value, stage, block_index=block_index, row_dim=row_dim
    )


def _debug_sla_tensor(
    stage: str,
    value: torch.Tensor,
    block_index: int,
    row_dim: int = 0,
    check_fp16_range: bool = False,
) -> None:
    """Check an explicitly armed block, or probe all stages on newer GPUs."""
    targeted = _is_sla_debug_block(block_index)
    automatic = _auto_sla_probe_enabled()
    if not targeted and not automatic:
        return
    finite = torch.isfinite(value)
    is_finite = bool(finite.all().item())
    if automatic and not targeted and not is_finite:
        _raise_strict_sla_failure(
            value, stage, block_index=block_index, row_dim=row_dim
        )
    if automatic and not targeted:
        return
    max_abs = float(value.abs().max().item()) if is_finite and value.numel() else float("nan")
    overflow = (
        int((value.abs() > torch.finfo(torch.float16).max).sum().item())
        if check_fp16_range else 0
    )
    nan_count = int(torch.isnan(value).sum().item())
    inf_count = int(torch.isinf(value).sum().item())
    _LOG.info(
        "[Star7 H3 Chunk] SLA debug block %d | %s | dtype=%s shape=%s "
        "max-abs=%.6g fp16-overflow=%d nan=%d inf=%d",
        block_index, stage, value.dtype, tuple(value.shape), max_abs,
        overflow, nan_count, inf_count,
    )
    if not is_finite:
        _raise_strict_sla_failure(
            value, stage, block_index=block_index, row_dim=row_dim
        )


def _star7_wrapper_original(value, kind: str):
    """Return the upstream callable after removing any same-kind Star7 layer."""
    current = value
    legacy_qualnames = {
        "h3-output-finite": "_h3_output_finite_passthrough.<locals>.forward",
        "sla-segment-block": "_sla_segment_passthrough.<locals>.forward",
        "mlp-chunk-upstream": "_make_chunked_h3_mlp_forward.<locals>.forward",
    }
    while True:
        function = getattr(current, "__func__", current)
        if getattr(function, "_star7_wrapper_kind", None) == kind:
            current = getattr(function, "_star7_original_forward")
            continue
        # v2.9.3 wrappers predate the explicit marker. Their exact local
        # qualname and named closure let upgrades collapse existing layers
        # without confusing unrelated third-party forwards.
        if getattr(function, "__qualname__", "") == legacy_qualnames.get(kind):
            closure = dict(zip(
                getattr(function.__code__, "co_freevars", ()),
                function.__closure__ or (),
            ))
            original_cell = closure.get("original_forward")
            if original_cell is not None:
                current = original_cell.cell_contents
                continue
        return current


def _h3_output_finite_passthrough(original_forward):
    """Reject invalid H3 video/audio velocities before the sampler or VAE."""
    original_forward = _weak_callable(original_forward)

    def forward(self, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict) and len(args) > 3:
            candidate = args[3]
            transformer_options = candidate if isinstance(candidate, dict) else None

        step_context = (
            _sampling_step_context(transformer_options)
            if transformer_options is not None else None
        )
        timing_device = None
        if step_context is not None:
            input_value = args[0] if args else kwargs.get("x")
            if isinstance(input_value, (list, tuple)):
                input_value = next(
                    (value for value in input_value if torch.is_tensor(value)), None
                )
            if torch.is_tensor(input_value):
                timing_device = input_value.device
                step_index, total_steps = step_context
                _step_timing_start(
                    transformer_options, input_value, step_index, total_steps,
                )

        try:
            result = original_forward(*args, **kwargs)
        except Exception:
            if transformer_options is not None:
                transformer_options.pop("_star7_step_timing", None)
            raise

        if step_context is not None and timing_device is not None:
            _step_timing_finish(
                transformer_options, timing_device,
                step_context[0], step_context[1], self,
            )
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

    forward._star7_wrapper_kind = "h3-output-finite"
    forward._star7_original_forward = original_forward
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
    q, k, v = _prepare_h3_qkv_chunked(
        self, x, rope_freqs, mm, comfy.quant_ops, output_dtype=x.dtype
    )
    _log_h3_cuda_memory(
        "after-attention-qkv", x.device,
        block_index=getattr(self, "_star7_block_index", None),
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
    projected = self.out_proj(out)
    _log_h3_cuda_memory(
        "after-attention", projected.device,
        block_index=getattr(self, "_star7_block_index", None),
    )
    return projected


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


def _load_sol_backend():
    try:
        from . import sol_backend
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).with_name("sol_backend.py")
        module_name = "star7_sol_backend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Star7 Sol backend from {path}")
        sol_backend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = sol_backend
        spec.loader.exec_module(sol_backend)
    return sol_backend


def _minimax_sla_forward(
    self, x, rope_freqs=None, transformer_options={},
    star7_sla_mod_segments=(),
):
    """Run strict LightX2V-style sparse attention without any fallback."""
    if isinstance(x, list):
        x = x.pop()

    # SM75 and legacy newer-GPU paths use FP16 working buffers. The visible
    # SM80+ SLA path stays BF16 through QKV and PV, then restores the upstream
    # interface dtype before out_proj when necessary.
    upstream_dtype = x.dtype

    import comfy.model_management as mm
    import comfy.quant_ops

    sla_backend = _load_sla_backend()
    sequence = x.shape[0]
    block_index = getattr(self, "_star7_block_index", None)
    debug_block = _is_sla_debug_block(block_index)
    configured = str(_CONFIG.get("attention_backend", ""))
    sla_bf16 = configured in {
        SM86PLUS_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME,
    }
    q, k, v = _prepare_h3_qkv_chunked(
        self, x, rope_freqs, mm, comfy.quant_ops,
        output_dtype=torch.bfloat16 if sla_bf16 else torch.float16,
    )
    _log_h3_cuda_memory(
        "after-attention-qkv", x.device,
        block_index=getattr(self, "_star7_block_index", None),
    )
    del x
    if debug_block:
        _debug_sla_tensor("Q after QKV/RoPE", q, block_index, row_dim=2)
        _debug_sla_tensor("K after QKV/RoPE", k, block_index, row_dim=2)
        _debug_sla_tensor("V after QKV projection", v, block_index, row_dim=2)

    # Both paths quantize Q/K for tensor-core QK. SM75 keeps V/PV in FP16;
    # visible SM80+ keeps V/PV in BF16. The chunk helper writes the selected
    # dtype directly, avoiding a second full Q/K/V conversion.

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
        all_int8=_CONFIG.get("attention_backend") in {
            SM75_ALL_INT8_BACKEND_NAME,
            HYBRID_ALL_INT8_BACKEND_NAME,
            SM86PLUS_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME,
        },
        debug=debug_block,
    )
    shape_key = (sequence, self.heads, self.head_dim, device_index)
    if _CONFIG["verbose"] and shape_key not in _LOGGED_SLA_SHAPES:
        _LOGGED_SLA_SHAPES.add(shape_key)
        _LOG.info(
            "[Star7 H3 Chunk] SLA runtime | Q-blocks=%d | K-blocks=%d | "
            "selected=%d | audio-guard-blocks=%d | effective-sparsity=%.2f%% | "
            "dense-audio-guard=%s | segments=%d | audio-ranges=%s | "
            "backend=%s | implementation=%s",
            result.query_blocks, result.key_blocks, result.selected_key_blocks,
            result.protected_query_blocks,
            result.effective_sparsity * 100.0,
            result.dense_guard_status,
            len(sla_segments), priority_ranges,
            _CONFIG.get("attention_backend"),
            result.implementation,
        )
    if debug_block:
        _debug_sla_tensor(
            "raw SLA output before out_proj", result.output,
            block_index, row_dim=2,
        )
    out = result.output.transpose(1, 2).reshape(
        1, sequence, self.heads * self.head_dim
    ).squeeze(0)
    if out.dtype != upstream_dtype:
        out = out.to(dtype=upstream_dtype)
    projected = self.out_proj(out)
    _log_h3_cuda_memory(
        "after-attention", projected.device,
        block_index=block_index,
    )
    if debug_block:
        _debug_sla_tensor(
            "attention out_proj output", projected, block_index, row_dim=0
        )
    return projected


_minimax_sla_forward._star7_consumes_input = True


def _minimax_sol_forward(
    self, x, rope_freqs=None, transformer_options={},
    star7_sla_mod_segments=(),
):
    """Run architecture-specific Sol without a silent fallback."""
    if isinstance(x, list):
        x = x.pop()
    upstream_dtype = x.dtype
    sequence = x.shape[0]
    block_index = getattr(self, "_star7_block_index", None)
    configured = str(_CONFIG.get("attention_backend", ""))
    official = configured in {
        SOL_SM86PLUS_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
    }
    all_int8 = configured in {
        SOL_SM75_ALL_INT8_BACKEND_NAME,
        LEGACY_SOL_SM75_ALL_INT8_BACKEND_NAME,
        SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
        HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
    }

    import comfy.model_management as mm
    import comfy.quant_ops

    sol_backend = _load_sol_backend()
    q, k, v = _prepare_h3_qkv_chunked(
        self,
        x,
        rope_freqs,
        mm,
        comfy.quant_ops,
        output_dtype=torch.bfloat16 if official else torch.float16,
        output_layout="BTHD" if official else "BHLD",
    )
    del x
    segments = star7_sla_mod_segments or getattr(
        self, "_star7_sla_mod_segments", ()
    )
    sink_start = None
    sink_tokens = 0
    if len(segments) >= 2:
        audio_segment = segments[-2]
        sink_start = int(audio_segment[0])
        sink_tokens = max(0, int(audio_segment[1]) - sink_start)

    if official:
        result = sol_backend.run_official(
            q, k, v,
            tau=sol_backend.DEFAULT_TAU,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
        )
        out = result.output.reshape(
            1, sequence, self.heads * self.head_dim
        ).squeeze(0)
    else:
        owned_qkv = [q, k, v]
        del q, k, v
        result = sol_backend.run_custom_consume(
            owned_qkv,
            all_int8=all_int8,
            tau=sol_backend.DEFAULT_TAU,
            topk_blocks=sol_backend.DEFAULT_TOPK_BLOCKS,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
        )
        out = result.output.transpose(1, 2).reshape(
            1, sequence, self.heads * self.head_dim
        ).squeeze(0)
    if out.dtype != upstream_dtype:
        out = out.to(dtype=upstream_dtype)
    projected = self.out_proj(out)

    shape_key = (configured, sequence, self.heads, self.head_dim, projected.device.index)
    if _CONFIG["verbose"] and shape_key not in _LOGGED_SOL_SHAPES:
        _LOGGED_SOL_SHAPES.add(shape_key)
        density = (
            "official-runtime"
            if result.mean_density != result.mean_density
            else f"{result.mean_density * 100.0:.2f}%"
        )
        _LOG.info(
            "[Star7 H3 Chunk] Sol runtime | backend=%s | Q64/K64 | "
            "Q-blocks=%d | K-blocks=%d | selected=%d..%d | density=%s | "
            "tau=%.2f | audio-sink=%s:%d | implementation=%s",
            configured, result.query_blocks, result.key_blocks,
            result.min_selected_blocks, result.max_selected_blocks, density,
            result.routing_tau, sink_start, sink_tokens, result.implementation,
        )
    return projected


_minimax_sol_forward._star7_consumes_input = True


def _minimax_hybrid_attention_forward(
    self, x, rope_freqs=None, transformer_options={},
):
    """Dispatch whole H3 attention calls between the existing CK and SLA paths."""
    step_index, total_steps, backend = _hybrid_sampling_context(transformer_options)

    if backend == "CK":
        return _minimax_ck_int8_attention_forward(
            self, x, rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
    sparse_forward = (
        _minimax_sol_forward
        if _CONFIG.get("attention_backend") in {
            HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
        }
        else _minimax_sla_forward
    )
    return sparse_forward(
        self, x, rope_freqs=rope_freqs,
        transformer_options=transformer_options,
    )


_minimax_hybrid_attention_forward._star7_consumes_input = True


def _sla_segment_passthrough(
    original_forward, block_index=None,
):
    """Expose H3 packed segments to SLA while preserving the upstream block."""
    original_forward = _weak_callable(original_forward)

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
            _require_strict_sla_finite(
                result, stage, block_index=block_index, row_dim=0,
                transformer_options=transformer_options,
            )
            if _is_sla_debug_block(block_index):
                _complete_automatic_sla_debug(block_index)
            return result
        finally:
            if old_segments is None:
                delattr(self.attn, "_star7_sla_mod_segments")
            else:
                self.attn._star7_sla_mod_segments = old_segments

    forward._star7_wrapper_kind = "sla-segment-block"
    forward._star7_original_forward = original_forward
    return forward


def _prepare_h3_qkv_chunked(
    self, x, rope_freqs, mm, quant_ops, output_dtype: Optional[torch.dtype] = None,
    output_layout: str = "BHLD",
):
    """Prepare contiguous backend-layout Q/K/V in token chunks.

    Buffers are allocated directly in the consuming backend's layout. CK/SLA
    use [1,H,S,D]; NVIDIA Sol uses contiguous [1,S,H,D].
    """
    sequence = int(x.shape[0])
    heads, head_dim = self.heads, self.head_dim
    quality_limited = _limit_sm75_qkv_chunk_for_quality(x)
    configured_chunk = int(_CONFIG["effective_qkv_chunk_tokens"])
    chunk = sequence if configured_chunk == 0 else min(
        sequence, max(256, configured_chunk)
    )
    output_dtype = output_dtype or x.dtype
    if output_layout not in {"BHLD", "BTHD"}:
        raise ValueError(f"unsupported QKV output layout: {output_layout}")
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
    qkv_call = self.qkv_proj
    qkv_weight_mode = "streamed"
    if (
        _sm75_qkv_reuse_path(x)
        and _CONFIG["reuse_mlp_weights"]
        and _linear_can_reuse_weights(self.qkv_proj)
    ):
        try:
            qkv_call, prepared_backend = _resident_qkv_caller(self.qkv_proj, x)
            qkv_weight_mode = f"resident-{prepared_backend}"
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            exc.__traceback__ = None
            _clear_cuda_after_oom(x.device)
            qkv_weight_mode = "streamed-fallback"
            _LOG.warning(
                "[Star7 H3 Chunk] Holding the patched QKV weight exceeded VRAM; "
                "using speech-stable per-chunk streaming"
            )
    _set_sequence_status(
        "QKV", sequence,
        reason_override="qkv_quality_cap" if quality_limited else None,
    )
    # These complete Q/K/V tensors are required by CK and SLA regardless of
    # projection chunk size. Retry their allocation once after releasing only
    # unused allocator cache, but do not pretend that lowering a local chunk can
    # solve a full-buffer OOM.
    qkv_buffers = []
    for allocation_attempt in range(2):
        try:
            qkv_buffers = [
                torch.empty(
                    (1, heads, sequence, head_dim)
                    if output_layout == "BHLD"
                    else (1, sequence, heads, head_dim),
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
    block_index = getattr(self, "_star7_block_index", None)
    while start < sequence:
        end = min(start + chunk, sequence)
        try:
            profile_qkv = bool(profile_total and start == 0)
            qkv_profile_start = time.perf_counter() if profile_qkv else None
            qkv_chunk = qkv_call(x[start:end])
            if profile_qkv:
                if x.device.type == "cuda":
                    torch.cuda.synchronize(x.device)
                first_projection_ms = (
                    time.perf_counter() - qkv_profile_start
                ) * 1000.0
            q, k, v = qkv_chunk.split(heads * head_dim, dim=-1)
            _debug_sla_tensor(
                f"Q projection chunk [{start}:{end}] before norm/cast",
                q, block_index, row_dim=0, check_fp16_range=True,
            )
            _debug_sla_tensor(
                f"K projection chunk [{start}:{end}] before norm/cast",
                k, block_index, row_dim=0, check_fp16_range=True,
            )
            _debug_sla_tensor(
                f"V projection chunk [{start}:{end}] before FP16 cast",
                v, block_index, row_dim=0, check_fp16_range=True,
            )
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
                _debug_sla_tensor(
                    f"Q norm+RoPE chunk [{start}:{end}]",
                    q, block_index, row_dim=1, check_fp16_range=True,
                )
                _debug_sla_tensor(
                    f"K norm+RoPE chunk [{start}:{end}]",
                    k, block_index, row_dim=1, check_fp16_range=True,
                )
                if profile_rope:
                    if x.device.type == "cuda":
                        torch.cuda.synchronize(x.device)
                    first_rope_ms = (
                        time.perf_counter() - rope_profile_start
                    ) * 1000.0
                if output_layout == "BHLD":
                    q_out[:, :, start:end, :].copy_(q.permute(0, 2, 1, 3))
                    k_out[:, :, start:end, :].copy_(k.permute(0, 2, 1, 3))
                else:
                    q_out[:, start:end, :, :].copy_(q)
                    k_out[:, start:end, :, :].copy_(k)
            else:
                q_norm = self.q_norm(q.view(end - start, heads, head_dim))
                k_norm = self.k_norm(k.view(end - start, heads, head_dim))
                _debug_sla_tensor(
                    f"Q norm chunk [{start}:{end}]", q_norm,
                    block_index, row_dim=0, check_fp16_range=True,
                )
                _debug_sla_tensor(
                    f"K norm chunk [{start}:{end}]", k_norm,
                    block_index, row_dim=0, check_fp16_range=True,
                )
                if output_layout == "BHLD":
                    q_out[:, :, start:end, :].copy_(q_norm.permute(1, 0, 2).unsqueeze(0))
                    k_out[:, :, start:end, :].copy_(k_norm.permute(1, 0, 2).unsqueeze(0))
                else:
                    q_out[:, start:end, :, :].copy_(q_norm.unsqueeze(0))
                    k_out[:, start:end, :, :].copy_(k_norm.unsqueeze(0))
            if output_layout == "BHLD":
                v_out[:, :, start:end, :].copy_(v.permute(1, 0, 2).unsqueeze(0))
            else:
                v_out[:, start:end, :, :].copy_(v.unsqueeze(0))
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
        fixed_qkv_gib = (
            3 * sequence * heads * head_dim * torch.empty(
                (), dtype=output_dtype
            ).element_size() / 1024**3
        )
        hidden_gib = (
            sequence * x.shape[1] * x.element_size() / 1024**3
        )
        _LOG.info(
            "[Star7 H3 Chunk] First-block QKV | S=%d | chunk=%d x %d | "
            "quantized=%s | weights=%s | %s->%s | first projection=%.1fms | "
            "first norm+RoPE=%.1fms | total=%.1fms | fixed-QKV=%.2fGiB | "
            "hidden=%.2fGiB | VRAM=%.2f/%.2fGiB",
            sequence, chunk, (sequence + chunk - 1) // chunk,
            qkv_quantized, qkv_weight_mode, x.dtype, output_dtype,
            first_projection_ms or 0.0, first_rope_ms or 0.0,
            (time.perf_counter() - total_profile_start) * 1000.0,
            fixed_qkv_gib, hidden_gib,
            allocated, reserved,
        )
    return q_out, k_out, v_out


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
    attention_backend = {
        LEGACY_SM75_ALL_INT8_BACKEND_NAME: SM75_ALL_INT8_BACKEND_NAME,
        LEGACY_SOL_SM75_ALL_INT8_BACKEND_NAME: SOL_SM75_ALL_INT8_BACKEND_NAME,
        LEGACY_SM86PLUS_ALL_INT8_BACKEND_NAME: SM86PLUS_ALL_INT8_BACKEND_NAME,
        LEGACY_SOL_SM86PLUS_ALL_INT8_BACKEND_NAME: SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
    }.get(attention_backend, attention_backend)
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
    _CONFIG["auto_sla_probe"] = False
    from comfy.ldm.minimax import model as h3_model
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, h3_model.MiniMaxH3Model):
        _LOG.warning("[Star7 H3 Chunk] Non-H3 model received; only the guarded RoPE dispatch was installed")
        return patched

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    star7_fp16 = bool(transformer_options.get("star7_minimax_h3_fp16_exact_fix"))
    _CONFIG["fp16_exact_present"] = star7_fp16
    transformer_options.pop("star7_h3_sm75_auto_fp16_exact", None)
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
    sol_attention = False
    strict_sla_backends = {
        "sla_sm75_qk_int8_pv_fp16",
        SM75_ALL_INT8_BACKEND_NAME,
        SM86PLUS_BACKEND_NAME,
        LEGACY_SM86PLUS_FP16_BACKEND_NAME,
        SM86PLUS_ALL_INT8_BACKEND_NAME,
        LEGACY_SM80PLUS_BACKEND_NAME,
    }
    strict_sol_backends = {
        SOL_SM75_BACKEND_NAME,
        SOL_SM75_ALL_INT8_BACKEND_NAME,
        SOL_SM86PLUS_BACKEND_NAME,
        SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
    }
    hybrid_attention = attention_backend in HYBRID_BACKEND_NAMES
    hybrid_sol_attention = attention_backend in {
        HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
        HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
    }
    sparse_backends = strict_sla_backends | strict_sol_backends | HYBRID_BACKEND_NAMES
    sla_backend = None
    sol_backend = None
    _CONFIG["hybrid_sla_backend"] = None
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    if attention_backend in strict_sla_backends or (
        hybrid_attention and not hybrid_sol_attention
    ):
        sla_backend = _load_sla_backend()
        if attention_backend in {
            SM75_ALL_INT8_BACKEND_NAME,
            HYBRID_ALL_INT8_BACKEND_NAME,
        }:
            requested_sla_backend = sla_backend.SM75_ALL_INT8_BACKEND_NAME
        elif attention_backend in {
            SM86PLUS_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME,
        }:
            requested_sla_backend = sla_backend.SM86PLUS_ALL_INT8_BACKEND_NAME
        elif attention_backend == HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME:
            requested_sla_backend = sla_backend.SM86PLUS_BACKEND_NAME
        elif attention_backend == LEGACY_HYBRID_SM86PLUS_CK_SLA_FP16_BACKEND_NAME:
            requested_sla_backend = sla_backend.LEGACY_SM86PLUS_FP16_BACKEND_NAME
        else:
            requested_sla_backend = attention_backend
        # Strict preflight: every Hybrid resolves to its architecture-specific
        # SLA backend. A selected SLA step never silently falls back to CK.
        capability = sla_backend.check_runtime_support(
            requested_backend=requested_sla_backend
        )
        if hybrid_attention:
            _CONFIG["hybrid_sla_backend"] = requested_sla_backend

    if attention_backend in strict_sol_backends or hybrid_sol_attention:
        sol_backend = _load_sol_backend()
        if attention_backend == HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME:
            requested_sol_backend = sol_backend.SOL_SM75_ALL_INT8_BACKEND_NAME
        elif attention_backend == HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME:
            requested_sol_backend = sol_backend.SOL_SM86PLUS_ALL_INT8_BACKEND_NAME
        elif attention_backend == HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME:
            requested_sol_backend = sol_backend.SOL_SM86PLUS_BACKEND_NAME
        else:
            requested_sol_backend = attention_backend
        capability = sol_backend.check_runtime_support(requested_sol_backend)

    if hybrid_attention:
        try:
            import comfy_kitchen
            ck_available = comfy_kitchen.int8_attention_is_available()
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "Star7 Hybrid Attention requires the existing "
                f"comfy_kitchen_int8 path; it is unavailable: {exc}"
            ) from exc
        if not ck_available:
            raise RuntimeError(
                "Star7 Hybrid Attention requires the existing "
                "comfy_kitchen_int8 path, but it is unavailable."
            )

    if capability == (7, 5) and not star7_fp16:
        _LOG.warning(
            "[Star7 H3 Chunk] SM75: separate MiniMax H3 FP16 Exact companion "
            "was not detected. Chunk keeps upstream precision unchanged and "
            "will only report numerical failures; pairing the two nodes is recommended."
        )

    # This guard covers CK, SLA and preserved attention alike. It is deliberately
    # placed at the joint model output rather than in VideoHelperSuite: FFmpeg can
    # only report corrupt PCM, whereas here the failing video/audio stream and
    # selected attention backend are still known.
    model_forward_path = "diffusion_model.forward"
    current_model_forward = patched.object_patches.get(
        model_forward_path, diffusion_model.forward
    )
    upstream_model_forward = _star7_wrapper_original(
        current_model_forward, "h3-output-finite"
    )
    patched.add_object_patch(
        model_forward_path,
        _weak_method(
            diffusion_model,
            _h3_output_finite_passthrough(upstream_model_forward),
        ),
    )
    transformer_options["star7_h3_output_finite_guard"] = NODE_VERSION

    if attention_backend in sparse_backends:
        for index, block in enumerate(diffusion_model.blocks):
            block._star7_block_index = index
            block.attn._star7_block_index = index
            block.mlp._star7_block_index = index
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                _weak_method(
                    block.attn,
                    _minimax_hybrid_attention_forward
                    if hybrid_attention else
                    _minimax_sol_forward if attention_backend in strict_sol_backends
                    else _minimax_sla_forward,
                ),
            )
            block_path = f"diffusion_model.blocks.{index}.forward"
            upstream_block_forward = patched.object_patches.get(
                block_path, block.forward
            )
            upstream_block_forward = _star7_wrapper_original(
                upstream_block_forward, "sla-segment-block"
            )
            patched.add_object_patch(
                block_path,
                _weak_method(
                    block,
                _sla_segment_passthrough(
                    upstream_block_forward,
                    block_index=index,
                ),
                ),
            )
        attention_patch_name = attention_backend
        sage_attention = False
        sla_attention = True
        sol_attention = attention_backend in strict_sol_backends or hybrid_sol_attention
        _CONFIG["auto_sla_probe"] = _auto_sla_probe_for_capability(capability)
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
                patched.add_object_patch(
                    f"diffusion_model.blocks.{index}.attn.forward",
                    _weak_method(block.attn, _minimax_ck_int8_attention_forward),
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

    # Chunk only controls token tiling. If another node already patched the MLP
    # (including FP16 Exact), invoke that exact upstream callable per tile rather
    # than copying its precision formula into this project.
    if int(mlp_chunk_tokens) != 0:
        for index, block in enumerate(diffusion_model.blocks):
            mlp_path = f"diffusion_model.blocks.{index}.mlp.forward"
            block.mlp._star7_reuse_mlp_input = True
            upstream_mlp = patched.object_patches.get(mlp_path)
            if upstream_mlp is not None:
                upstream_mlp = _star7_wrapper_original(
                    upstream_mlp, "mlp-chunk-upstream"
                )
            mlp_forward = _make_chunked_h3_mlp_forward(upstream_mlp)
            patched.add_object_patch(
                mlp_path,
                _weak_method(block.mlp, mlp_forward),
            )

    # Dynamic VBAR prefetch is intentionally disabled for H3.  The upstream
    # queue can increase reserved VRAM without improving the supported low-VRAM
    # path. Keep the legacy function argument so old workflows remain loadable.
    transformer_options["prefetch_dynamic_vbars"] = False

    if verbose:
        architecture = (
            f"SM{capability[0]}{capability[1]}" if capability else "unknown-GPU"
        )
        model_precision = (
            "FP16 Exact (external companion)" if star7_fp16
            else "upstream"
        )
        selected_attention = (
            attention_patch_name if sla_attention
            else "comfy-kitchen-int8" if ck_attention
            else "sage-qk-int8" if sage_attention
            else attention_patch_name
        )
        all_int8_attention = attention_backend in {
            SM75_ALL_INT8_BACKEND_NAME,
            HYBRID_ALL_INT8_BACKEND_NAME,
            SM86PLUS_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_ALL_INT8_BACKEND_NAME,
            SOL_SM75_ALL_INT8_BACKEND_NAME,
            SOL_SM86PLUS_ALL_INT8_BACKEND_NAME,
            HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME,
            HYBRID_SM86PLUS_CK_SOL_ALL_INT8_BACKEND_NAME,
        }
        bf16_sla_attention = attention_backend in {
            SM86PLUS_BACKEND_NAME,
            HYBRID_SM86PLUS_CK_SLA_BF16_BACKEND_NAME,
        }
        precision_suffix = (
            " | sol-input=bf16 | exact+approx=nvidia-official"
            if attention_backend in {
                SOL_SM86PLUS_BACKEND_NAME,
                HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
            } else
            " | sol-input=fp16 | exact+centroid | qk=int8 | pv=int8 | softmax/accum=fp32"
            if sol_attention and all_int8_attention else
            " | sol-input=fp16 | exact+centroid | qk=int8 | pv=fp16 | softmax/accum=fp32"
            if sol_attention else
            " | sla-input=fp16 | qk=int8 | pv=int8 | softmax/accum=fp32"
            if sla_attention and all_int8_attention else
            " | sla-input=bf16 | qk=int8 | pv=bf16 | softmax/accum=fp32"
            if sla_attention and bf16_sla_attention else
            " | sla-input=fp16 | qk=int8 | pv=fp16 | softmax/accum=fp32"
            if sla_attention else ""
        )
        _LOG.info(
            "[Star7 H3 Chunk] Ready v%s | %s | attention=%s | "
            "model-precision=%s%s | "
            "chunks(RoPE/MLP/QKV)=%d/%d/%d | chunk-weight-reuse=%s | "
            "block-cache=%s | finite-guard=model-output%s",
            NODE_VERSION, architecture, selected_attention, model_precision,
            precision_suffix,
            int(chunk_tokens), int(mlp_chunk_tokens), int(qkv_chunk_tokens),
            bool(reuse_mlp_weights), "external" if block_loop_cache else "none",
            "+sparse-block" if sla_attention else "",
        )
        if sla_attention and capability:
            environment_key = (capability, torch.__version__, torch.version.cuda)
            if environment_key not in _LOGGED_SLA_ENVIRONMENTS:
                _LOGGED_SLA_ENVIRONMENTS.add(environment_key)
                runtime_backend_module = sol_backend if sol_attention else sla_backend
                triton_version = getattr(
                    getattr(runtime_backend_module, "triton", None),
                    "__version__", "unavailable",
                )
                try:
                    driver_version = torch._C._cuda_getDriverVersion()
                except Exception:
                    driver_version = "unknown"
                validation = (
                    "SM75-native-CUDA-path"
                    if capability == (7, 5) else
                    "newer-architecture-needs-device-validation"
                    if capability >= (10, 0) else
                    "SM80+-path"
                )
                _LOG.info(
                    "[Star7 H3 Chunk] Sparse attention environment | gpu=%s | SM%d%d | "
                    "torch=%s | cuda-runtime=%s | triton=%s | driver=%s | %s",
                    torch.cuda.get_device_name(), capability[0], capability[1],
                    torch.__version__, torch.version.cuda, triton_version,
                    driver_version, validation,
                )
                if capability >= (10, 0):
                    implementation_family = (
                        "the NVIDIA official Sol dispatcher"
                        if attention_backend in {
                            SOL_SM86PLUS_BACKEND_NAME,
                            HYBRID_SM86PLUS_CK_SOL_BF16_BACKEND_NAME,
                        }
                        else "the SM80+ Triton path"
                    )
                    _LOG.warning(
                        "[Star7 H3 Chunk] SM%d%d uses %s and "
                        "requires real-device validation; strict finite checks "
                        "remain enabled and no fallback is allowed.",
                        capability[0], capability[1], implementation_family,
                    )
        selected_debug_block = _sla_debug_block() if sla_attention else None
        if _CONFIG.get("auto_sla_probe"):
            _LOG.info(
                "[Star7 H3 Chunk] Automatic sparse-attention diagnostics enabled for "
                "SM%d%d; the first non-finite QKV/attention/out_proj/MLP stage will "
                "be reported in this run.",
                capability[0], capability[1],
            )
        if selected_debug_block is not None:
            _LOG.info(
                "[Star7 H3 Chunk] Targeted SLA diagnostics armed for block %d",
                selected_debug_block,
            )
    return patched


class MiniMaxH3ActivationChunkStar7:
    """Pass-through MODEL node that installs formula-preserving H3 activation chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        qkv_default = _default_qkv_chunk_tokens()
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
                        "default": qkv_default,
                        "min": 0,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "H3 QKV 投影临时显存分块。SM75 为保护参考语音稳定性，0 或高于 4096 的设定会按 4096 运行；SM80+ 不设此质量上限。若自动降档开启，QKV OOM 时只降低 QKV 后重试。",
                    },
                ),
                "reuse_mlp_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse prepared QKV/MLP weight snapshots across token chunks when safe. Falls back to streamed preparation on VRAM pressure.",
                    },
                ),
                "attention_backend": (
                    _attention_backend_choices(),
                    {
                        "default": "comfy_kitchen_int8",
                        "tooltip": (
                            "existing keeps the incoming attention patch (for example KJ Sage). "
                            "comfy_kitchen_int8 selects ComfyUI's native INT8 attention and "
                            "overrides an earlier MiniMax Sage patch. SLA uses fixed Top-K "
                            "Q128/K64 routing. Sol uses Q64/K64 threshold routing; the SM80+ "
                            "recommended Sol mode directly calls NVIDIA official BF16 "
                            "exact+approx Sol-Attn. SM75 exposes the native All-INT8 Sol path; "
                            "it keeps exact selected blocks and centroid contributions for "
                            "unselected blocks while quantizing PV. Hybrid modes schedule "
                            "CK/SLA/CK or CK/Sol/CK by complete sampling step. Strict sparse "
                            "modes never silently fall back."
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
        "MiniMax H3 model patch with independent QKV, split-half RoPE, and MLP activation "
        "chunking. It preserves INT8/ConvRot weights and the upstream DiT block structure "
        "for FP16/BF16, Sage, LoRA, and third-party compatibility. Attention can preserve "
        "the incoming backend or select CK INT8, architecture-specific SLA/Sol, and "
        "step-level CK/Sparse/CK Hybrid paths. Strict sparse modes report failures without "
        "substituting another backend. Compatible with the separate FP16 Exact Fix - Star7."
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


def _normalize_reference_max_long_edge(value, default: int = 1024) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return int(default)
    if value == 0:
        return 0
    return int(default) if value < 32 else min(8192, value)


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


def _reference_audio_decode_command(
    ffmpeg: str, video_path: str, audio_rate: int = 44100,
) -> list[str]:
    """Keep source audio up to 15 seconds instead of H3 frame-grid duration."""
    return [
        ffmpeg,
        "-v", "error",
        "-i", video_path,
        "-t", "15",
        "-vn",
        "-ac", "2",
        "-ar", str(audio_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]


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
            audio_command = _reference_audio_decode_command(
                ffmpeg, video_path, audio_rate,
            )
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
                audio_status = "stereo-44100Hz-source-up-to-15s"

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


class MiniMaxH3RoPEChunkPatch(MiniMaxH3ActivationChunkStar7):
    """Legacy class ID retained only so existing workflows keep loading."""

    DEPRECATED = True


class MiniMaxH3LoadImageScaleStar7:
    """Core image upload plus an H3-aligned long-edge limit in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = [
            name for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "最长边": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "保持宽高比，将图片最长边限制在这个 H3 对齐尺寸内；设为 0 保持原尺寸。",
                    },
                ),
                "允许小图放大": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "默认关闭，不放大小图；需要放大参考图时再开启。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_and_scale"
    CATEGORY = "Star7/image"

    def load_and_scale(self, image, **kwargs):
        import comfy.utils
        import folder_paths
        import numpy as np
        from PIL import Image, ImageOps

        max_long_edge = _normalize_reference_max_long_edge(kwargs.get("最长边", 1024))
        allow_upscale = bool(kwargs.get("允许小图放大", False))
        image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(image_path) as source:
            source = ImageOps.exif_transpose(source)
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            loaded = torch.from_numpy(rgb).unsqueeze(0)
            if "A" in source.getbands():
                alpha = np.asarray(source.getchannel("A"), dtype=np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha).unsqueeze(0)
            else:
                mask = torch.zeros((1, rgb.shape[0], rgb.shape[1]), dtype=loaded.dtype)
        source_height, source_width = map(int, loaded.shape[1:3])
        max_long_edge = _normalize_reference_max_long_edge(max_long_edge)
        if max_long_edge == 0:
            width, height = source_width, source_height
        else:
            width, height = _long_edge_reference_size(
                source_width, source_height, max_long_edge,
                bool(allow_upscale), 32,
            )
        if (width, height) == (source_width, source_height):
            return loaded, mask

        # Keep the UI focused on geometry; use a fixed quality-safe resampler.
        scaled = comfy.utils.common_upscale(
            loaded.movedim(-1, 1), width, height, "area", "disabled",
        ).movedim(1, -1)
        if tuple(mask.shape[-2:]) == (source_height, source_width):
            mask = F.interpolate(
                mask.unsqueeze(1), size=(height, width), mode="nearest-exact",
            ).squeeze(1)
        else:
            mask = torch.zeros(
                (scaled.shape[0], height, width), dtype=scaled.dtype, device=scaled.device,
            )
        _LOG.info(
            "[Star7 H3 Ref Image] %dx%d -> %dx%d | max_long_edge=%d | allow_upscale=%s",
            source_width, source_height, width, height,
            max_long_edge, bool(allow_upscale),
        )
        return scaled, mask

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        import folder_paths
        import hashlib

        image_path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(image_path, "rb") as handle:
            digest.update(handle.read())
        max_long_edge = _normalize_reference_max_long_edge(kwargs.get("最长边", 1024))
        allow_upscale = bool(kwargs.get("允许小图放大", False))
        return f"{digest.hexdigest()}:{max_long_edge}:{allow_upscale}"

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        import folder_paths

        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": MiniMaxH3ActivationChunkStar7,
    "MiniMaxH3ReferenceVideoLoadStar7": MiniMaxH3ReferenceVideoLoadStar7,
    "MiniMaxH3LoadImageScaleStar7": MiniMaxH3LoadImageScaleStar7,
    # Compatibility alias for workflows saved before the package was renamed.
    "MiniMaxH3RoPEChunkPatch": MiniMaxH3RoPEChunkPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ActivationChunkStar7": "MiniMax H3 Activation Chunk - Star7",
    "MiniMaxH3ReferenceVideoLoadStar7": "Reference Video Load - Star7",
    "MiniMaxH3LoadImageScaleStar7": "Reference Image Load - Star7",
    "MiniMaxH3RoPEChunkPatch": "MiniMax H3 RoPE Chunk Patch (Legacy) - Star7",
}
