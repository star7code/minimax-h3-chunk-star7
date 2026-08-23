"""Numerically validate the bundled SM75 sparse kernel against PyTorch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


NODE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("star7_sm75_validation", NODE_ROOT / "sm75_backend.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sla_spec = importlib.util.spec_from_file_location("star7_sla_validation", NODE_ROOT / "sla_backend.py")
sla = importlib.util.module_from_spec(sla_spec)
sys.modules[sla_spec.name] = sla
sla_spec.loader.exec_module(sla)


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for q_block in range(lut.shape[2]):
                q_start = q_block * 128
                q_end = min(q_start + 128, q.shape[2])
                indices = []
                for block in lut[batch, head, q_block].tolist():
                    indices.extend(range(block * 64, min(block * 64 + 64, q.shape[2])))
                index = torch.tensor(indices, device=q.device)
                query = q[batch, head, q_start:q_end].float()
                key = k[batch, head, index].float()
                value = v[batch, head, index].float()
                probability = torch.softmax(query @ key.T * scale, dim=-1)
                result[batch, head, q_start:q_end] = (probability @ value).half()
    return result


def quantize(q: torch.Tensor, k: torch.Tensor):
    mean = k.mean(dim=-2, keepdim=True, dtype=torch.float32).to(k.dtype)
    qi, qs = sla._quantize(q, 16, multiplier=1.0)
    ki, ks = sla._quantize(k, 64, multiplier=1.0, mean=mean)
    return qi, qs, ki, ks


def quantized_reference(qi, qs, ki, ks, v, lut):
    rows = torch.arange(qi.shape[2], device=qi.device)
    qd = qi.float() * qs[:, :, rows // 16, None]
    kd = ki.float() * ks[:, :, rows // 64, None]
    return reference(qd, kd, v, lut)


torch.manual_seed(5717)
q = torch.randn((1, 2, 384, 128), device="cuda", dtype=torch.float16) * 0.35
k = torch.randn_like(q) * 0.35
v = torch.randn_like(q) * 0.35
lut = torch.tensor(
    [[[[0, 2, 5], [1, 3, 5], [0, 4, 5]], [[1, 2, 4], [0, 3, 5], [2, 4, 5]]]],
    device="cuda",
    dtype=torch.int32,
)
for all_int8 in (False, True):
    mode = "all-int8" if all_int8 else "fp16-pv"
    for label, query, key in (("random", q, k), ("uniform", torch.zeros_like(q), torch.zeros_like(k))):
        qi, qs, ki, ks = quantize(query, key)
        actual = module.run(qi, ki, v, qs, ks, lut, all_int8=all_int8)
        expected = reference(query, key, v, lut)
        torch.cuda.synchronize()
        delta = (actual.float() - expected.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        print(
            f"{mode}/{label}: finite={bool(torch.isfinite(actual).all())} "
            f"mae={delta.mean().item():.8f} max={delta.max().item():.8f} "
            f"cosine={cosine.item():.8f} "
            f"norm_ratio={(actual.float().norm() / expected.float().norm()).item():.6f}"
        )

one_lut = lut[..., :1].contiguous()
qi, qs, ki, ks = quantize(q, k)
one_actual = module.run(qi, ki, v, qs, ks, one_lut)
one_expected = quantized_reference(qi, qs, ki, ks, v, one_lut)
one_delta = (one_actual.float() - one_expected.float()).abs()
one_cosine = torch.nn.functional.cosine_similarity(one_actual.float().flatten(), one_expected.float().flatten(), dim=0)
print(f"one-block: mae={one_delta.mean().item():.8f} max={one_delta.max().item():.8f} cosine={one_cosine.item():.8f}")

guarded = module.run(qi, ki, v, qs, ks, lut, dense_query_ranges=((130, 250),))
guarded_int8 = module.run(
    qi, ki, v, qs, ks, lut, dense_query_ranges=((130, 250),), all_int8=True
)
full_lut = torch.arange(6, device=q.device, dtype=torch.int32).view(1, 1, 1, 6).expand(1, 2, 3, 6).contiguous()
full_expected = reference(q, k, v, full_lut)
guard_delta = (guarded[:, :, 128:256].float() - full_expected[:, :, 128:256].float()).abs()
guard_cosine = torch.nn.functional.cosine_similarity(
    guarded[:, :, 128:256].float().flatten(), full_expected[:, :, 128:256].float().flatten(), dim=0
)
print(
    f"dense-guard: mae={guard_delta.mean().item():.8f} "
    f"max={guard_delta.max().item():.8f} cosine={guard_cosine.item():.8f}"
)
guard_delta_int8 = (
    guarded_int8[:, :, 128:256].float() - full_expected[:, :, 128:256].float()
).abs()
guard_cosine_int8 = torch.nn.functional.cosine_similarity(
    guarded_int8[:, :, 128:256].float().flatten(),
    full_expected[:, :, 128:256].float().flatten(), dim=0,
)
print(
    f"dense-guard/all-int8: mae={guard_delta_int8.mean().item():.8f} "
    f"max={guard_delta_int8.max().item():.8f} "
    f"cosine={guard_cosine_int8.item():.8f}"
)

diag_q = q[:, :1, :128].contiguous()
diag_k = k[:, :1, :128].contiguous()
diag_v = torch.zeros_like(diag_q)
diag_v[0, 0, :64, :64] = torch.eye(64, device="cuda", dtype=torch.float16)
diag_lut = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
dqi, dqs, dki, dks = quantize(diag_q, diag_k)
diag_actual = module.run(dqi, dki, diag_v, dqs, dks, diag_lut)
diag_expected = quantized_reference(dqi, dqs, dki, dks, diag_v, diag_lut)
for row in (0, 1, 8, 9):
    print(
        f"row={row} actual_top={torch.topk(diag_actual[0,0,row,:64], 4).indices.tolist()} "
        f"expected_top={torch.topk(diag_expected[0,0,row,:64], 4).indices.tolist()}"
    )
