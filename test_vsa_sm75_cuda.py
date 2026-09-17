import ctypes
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
COMFY_ROOT = ROOT.parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bind_test_library(library):
    quantize = library.star7_sla_sm75_quantize
    quantize.argtypes = [
        *([ctypes.c_uint64] * 4), *([ctypes.c_int] * 4),
        ctypes.c_float, ctypes.c_int, ctypes.c_uint64,
    ]
    quantize.restype = ctypes.c_int
    quant_v = library.star7_sla_sm75_quant_v_int8
    quant_v.argtypes = [
        *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 5),
        *([ctypes.c_int64] * 3), ctypes.c_int, ctypes.c_uint64,
    ]
    quant_v.restype = ctypes.c_int


def test_vsa_topk_routing_forces_prefix_and_neighbors():
    backend = _module("star7_vsa_sm75_routing_test", "vsa_sm75_backend.py")
    q_mean = torch.randn((1, 2, 6, 128), dtype=torch.float32)
    k_mean = torch.randn_like(q_mean)
    row_count, lut, density = backend.build_topk_routing(
        q_mean, k_mean, topk_ratio=0.2, prefix_blocks=2,
    )
    assert row_count.shape == (1, 2, 6)
    assert 0.0 < density <= 1.0
    for head in range(2):
        for query in range(6):
            selected = set(lut[0, head, query, :row_count[0, head, query]].tolist())
            assert {0, 1}.issubset(selected)
            assert set(range(max(0, query - 1), min(6, query + 2))).issubset(selected)
            if query < 2:
                assert selected == set(range(6))


def test_vsa_topk_routing_forces_true_3d_video_neighbors():
    backend = _module("star7_vsa_sm75_3d_routing_test", "vsa_sm75_backend.py")
    q_mean = torch.randn((1, 1, 9, 128), dtype=torch.float32)
    k_mean = torch.randn_like(q_mean)
    row_count, lut, _density = backend.build_topk_routing(
        q_mean, k_mean, topk_ratio=0.01, prefix_blocks=1,
        video_grid=(2, 2, 2),
    )
    selected = set(lut[0, 0, 1, :row_count[0, 0, 1]].tolist())
    # Prefix plus the origin cube and its +W, +H and +T neighbours.
    assert {0, 1, 2, 3, 5}.issubset(selected)

    selected = set(lut[0, 0, 8, :row_count[0, 0, 8]].tolist())
    # Opposite corner and its -W, -H and -T neighbours.
    assert {0, 8, 7, 6, 4}.issubset(selected)


def test_vsa_topk_routing_rejects_a_mismatched_video_grid():
    backend = _module("star7_vsa_sm75_bad_grid_test", "vsa_sm75_backend.py")
    q_mean = torch.randn((1, 1, 9, 128), dtype=torch.float32)
    try:
        backend.build_topk_routing(
            q_mean, torch.randn_like(q_mean), topk_ratio=0.1,
            prefix_blocks=1, video_grid=(1, 2, 2),
        )
    except ValueError as exc:
        assert "video_grid" in str(exc)
    else:
        raise AssertionError("mismatched VSA video_grid must be rejected")


def test_vsa_3d_locality_mask_is_reused_for_every_transformer_block():
    backend = _module("star7_vsa_sm75_3d_cache_test", "vsa_sm75_backend.py")
    first = backend._local_3d_mask((2, 3, 4), torch.device("cpu"))
    second = backend._local_3d_mask((2, 3, 4), torch.device("cpu"))
    assert first is second
    assert len(backend._LOCAL_3D_CACHE) == 1


def test_vsa_sm75_fine_kernel_matches_selected_token_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    backend = _module("star7_vsa_sm75_cuda_test", "vsa_sm75_backend.py")
    native = backend._load_sm75_backend()
    test_binary = Path(os.environ.get(
        "STAR7_VSA_TEST_LIBRARY",
        ROOT / "bin" / "win_amd64" / "star7_sla_sm75_v7_sol_nativepreprocess.dll",
    ))
    library = ctypes.CDLL(str(test_binary))
    _bind_test_library(library)
    native._load = lambda: library
    native.availability = lambda: (True, "test binary")

    generator = torch.Generator(device="cuda").manual_seed(0x75A)
    shape = (1, 2, 256, 128)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.2
    block_len = torch.tensor([64, 37, 64, 13], device="cuda", dtype=torch.int32)
    for block, live in enumerate(block_len.tolist()):
        start = block * 64 + live
        end = (block + 1) * 64
        q[:, :, start:end] = 0
        k[:, :, start:end] = 0
        v[:, :, start:end] = 0

    q_mean = backend.block_means(q, block_len)
    k_mean = backend.block_means(k, block_len)
    row_count, lut, _density = backend.build_topk_routing(
        q_mean, k_mean, topk_ratio=0.5, prefix_blocks=1,
    )
    q_int8, k_int8, q_scale, k_scale = backend.quantize_qk(q, k, k_mean)
    output = backend.run_fine(
        q_int8, k_int8, v, q_scale, k_scale, row_count, lut, block_len,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()

    references = []
    actual = []
    scale = 128 ** -0.5
    for head in range(shape[1]):
        for query_block, query_live in enumerate(block_len.tolist()):
            selected = lut[0, head, query_block, :row_count[0, head, query_block]].tolist()
            key_rows = torch.cat([
                torch.arange(block * 64, block * 64 + int(block_len[block]), device="cuda")
                for block in selected
            ])
            query_rows = torch.arange(
                query_block * 64, query_block * 64 + query_live, device="cuda",
            )
            scores = q[0, head, query_rows].float() @ k[0, head, key_rows].float().T * scale
            references.append(torch.softmax(scores, dim=-1) @ v[0, head, key_rows].float())
            actual.append(output[0, head, query_rows].float())
    reference = torch.cat(references).flatten()
    measured = torch.cat(actual).flatten()
    cosine = torch.nn.functional.cosine_similarity(reference, measured, dim=0)
    relative = (reference - measured).norm() / reference.norm()
    assert float(cosine) > 0.985
    assert float(relative) < 0.20

    gate = torch.randn(
        (1, shape[2], shape[1], shape[3]), generator=generator,
        device="cuda", dtype=torch.float16,
    )
    v_mean = backend.block_means(v, block_len)
    combined = output.clone()
    backend.add_coarse_(combined, q_mean, k_mean, v_mean, gate)
    coarse_scores = torch.bmm(
        q_mean.reshape(-1, 4, 128),
        k_mean.reshape(-1, 4, 128).transpose(1, 2),
    ) * 128 ** -0.5
    coarse = torch.bmm(
        torch.softmax(coarse_scores, dim=-1),
        v_mean.reshape(-1, 4, 128),
    ).view(1, 2, 4, 128).permute(0, 2, 1, 3)
    expected = output.transpose(1, 2).view(1, 4, 64, 2, 128).float()
    expected = expected + gate.view(1, 4, 64, 2, 128).float() * coarse[:, :, None]
    assert torch.allclose(
        combined.transpose(1, 2).view_as(expected).float(),
        expected, rtol=2e-3, atol=2e-3,
    )


def test_vsa_sm75_h3_attention_integration():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    nodes = _module("star7_vsa_sm75_nodes_test", "nodes.py")
    device = torch.device("cuda")
    heads, head_dim, hidden = 2, 128, 256
    source_tokens, padded_tokens = 133, 192

    class Attention:
        def __init__(self):
            self.heads = heads
            self.head_dim = head_dim
            self.qkv_proj = torch.nn.Linear(
                hidden, heads * head_dim * 3, bias=False,
                device=device, dtype=torch.float16,
            )
            self.out_proj = torch.nn.Linear(
                heads * head_dim, hidden, bias=False,
                device=device, dtype=torch.float16,
            )
            self.to_gate_compress = torch.nn.Linear(
                hidden, heads * head_dim, bias=False,
                device=device, dtype=torch.float16,
            )
            self.qkv_proj.requires_grad_(False)
            self.out_proj.requires_grad_(False)
            self.to_gate_compress.requires_grad_(False)
            self.q_norm = SimpleNamespace(
                weight=torch.ones(head_dim, device=device, dtype=torch.float16),
                eps=1.0e-6,
            )
            self.k_norm = SimpleNamespace(
                weight=torch.ones(head_dim, device=device, dtype=torch.float16),
                eps=1.0e-6,
            )

    inv = torch.cat([
        torch.arange(5, device=device),
        torch.arange(64, padded_tokens, device=device),
    ])
    block_len = torch.tensor([5, 64, 64], device=device, dtype=torch.int32)
    plan = {
        "n": padded_tokens,
        "n_prefix": 1,
        "inv": inv,
        "block_len": block_len,
    }

    class Patch:
        topk_ratio = 0.10
        _logged = set()
        messages = []

        @staticmethod
        def vsa_plan(_layout, _device):
            return plan

        @staticmethod
        def vsa_rope_freqs(rope, _plan):
            padded = rope.new_zeros((1, padded_tokens) + tuple(rope.shape[2:]))
            padded[0, inv] = rope[0]
            return padded

        @classmethod
        def log_once(cls, key, message):
            if key not in cls._logged:
                cls._logged.add(key)
                cls.messages.append(message)

    rope = torch.zeros(
        (1, source_tokens, 1, 48, 2, 2), device=device, dtype=torch.float16,
    )
    rope[..., 0, 0] = 1
    rope[..., 1, 1] = 1
    value = torch.randn(
        (source_tokens, hidden), device=device, dtype=torch.float16,
    ) * 0.2
    layout = SimpleNamespace(seq_len=source_tokens)
    reuse = nodes._CONFIG["reuse_mlp_weights"]
    nodes._CONFIG["reuse_mlp_weights"] = False
    attention = Attention()
    patch = Patch()
    try:
        output = nodes._minimax_vsa_sm75_forward(
            attention, value, rope, {"minimax_h3_layout": layout}, patch, 0,
        )
        nodes._minimax_vsa_sm75_forward(
            attention, value, rope, {"minimax_h3_layout": layout}, patch, 49,
        )
    finally:
        nodes._CONFIG["reuse_mlp_weights"] = reuse
    torch.cuda.synchronize()
    assert output.shape == value.shape
    assert output.dtype == value.dtype
    assert torch.isfinite(output).all()
    assert len(patch.messages) == 1
    assert "first-block fine density" in patch.messages[0]


def benchmark_vsa_sm75_representative():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    backend = _module("star7_vsa_sm75_benchmark", "vsa_sm75_backend.py")
    batch, heads, length, head_dim = 1, 56, 30720, 128
    shape = (batch, heads, length, head_dim)
    q = torch.empty(shape, device="cuda", dtype=torch.float16).normal_(std=0.2)
    k = torch.empty_like(q).normal_(std=0.2)
    v = torch.empty_like(q).normal_(std=0.2)
    block_len = torch.full(
        (length // 64,), 64, device="cuda", dtype=torch.int32,
    )

    def timed(call):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    q_mean = backend.block_means(q, block_len)
    k_mean = backend.block_means(k, block_len)
    v_mean = backend.block_means(v, block_len)
    (row_count, lut, density), route_seconds = timed(lambda: backend.build_topk_routing(
        q_mean, k_mean, topk_ratio=0.10, prefix_blocks=4,
    ))
    (q_int8, k_int8, q_scale, k_scale), quant_seconds = timed(
        lambda: backend.quantize_qk(q, k, k_mean)
    )
    del q, k
    output, fine_seconds = timed(lambda: backend.run_fine(
        q_int8, k_int8, [v], q_scale, k_scale, row_count, lut, block_len,
    ))
    gate = torch.empty(
        (batch, length, heads, head_dim), device="cuda", dtype=torch.float16,
    ).normal_(std=0.2)
    _, coarse_seconds = timed(
        lambda: backend.add_coarse_(output, q_mean, k_mean, v_mean, gate)
    )
    print(
        f"SM75 VSA benchmark T={length} H={heads} density={density * 100:.2f}% "
        f"route={route_seconds:.3f}s quant_qk={quant_seconds:.3f}s "
        f"fine={fine_seconds:.3f}s coarse={coarse_seconds:.3f}s"
    )


if __name__ == "__main__":
    test_vsa_topk_routing_forces_prefix_and_neighbors()
    test_vsa_topk_routing_forces_true_3d_video_neighbors()
    test_vsa_topk_routing_rejects_a_mismatched_video_grid()
    test_vsa_3d_locality_mask_is_reused_for_every_transformer_block()
    test_vsa_sm75_fine_kernel_matches_selected_token_reference()
    test_vsa_sm75_h3_attention_integration()
    if os.environ.get("STAR7_VSA_BENCHMARK") == "1":
        benchmark_vsa_sm75_representative()
    print("Star7 SM75 VSA tests passed")
