"""Benchmark Star7 SLA stages against Comfy Kitchen on the active GPU.

Run with ComfyUI's Python, for example::

    python benchmark_attention.py --lengths 4096 8192 16384 --heads 56

The benchmark intentionally includes routing and quantization allocations.  It
is a development tool, not imported by the custom node at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import statistics
import sys
from pathlib import Path

import torch


NODE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = NODE_ROOT.parents[2]
PYTHON_ROOT = COMFY_ROOT.parent
for path in (str(NODE_ROOT), str(COMFY_ROOT), str(PYTHON_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_backend():
    path = NODE_ROOT / "sla_backend.py"
    spec = importlib.util.spec_from_file_location("star7_bench_sla_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SLA backend from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _events(count: int) -> list[torch.cuda.Event]:
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def _sla_once(backend, q, k, v, sparsity: float, all_int8: bool) -> dict[str, float]:
    marks = _events(6)
    marks[0].record()
    lut, k_mean, _qb, _kb, _selected = backend.build_routing_lut(q, k, sparsity)
    marks[1].record()
    native_sm75 = torch.cuda.get_device_capability(q.device) == (7, 5)
    q_int8, q_scale = backend._quantize(
        q,
        16 if native_sm75 else backend.BLOCK_Q,
        multiplier=1.0 if native_sm75 else (backend.HEAD_DIM ** -0.5) * backend._LOG2E,
    )
    marks[2].record()
    if native_sm75:
        required_q_scales = ((q.shape[-2] + 127) // 128) * 8
        if q_scale.shape[-1] < required_q_scales:
            q_scale = torch.nn.functional.pad(
                q_scale, (0, required_q_scales - q_scale.shape[-1]), value=1.0
            )
    k_int8, k_scale = backend._quantize(
        k, backend.BLOCK_K, multiplier=1.0, mean=k_mean
    )
    marks[3].record()
    if native_sm75:
        output = backend._load_sm75_backend().run(
            q_int8, k_int8, v, q_scale, k_scale, lut, all_int8=all_int8
        )
    else:
        output = backend._load_sm75_backend().run(
            q_int8, k_int8, v, q_scale, k_scale, lut
        )
    marks[4].record()
    output.sum().item()
    marks[5].record()
    return {
        "route": marks[0].elapsed_time(marks[1]),
        "quant_q": marks[1].elapsed_time(marks[2]),
        "quant_k": marks[2].elapsed_time(marks[3]),
        "core": marks[3].elapsed_time(marks[4]),
        "sla_total": marks[0].elapsed_time(marks[4]),
    }


def _ck_once(q, k, v) -> float:
    import comfy_kitchen

    start, end = _events(2)
    start.record()
    output = comfy_kitchen.int8_attention(q, k, v)
    end.record()
    output.sum().item()
    return start.elapsed_time(end)


def _median(rows: list[dict[str, float]], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def benchmark(backend, length: int, heads: int, repeats: int, sparsity: float) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0x57A7 + length)
    shape = (1, heads, length, backend.HEAD_DIM)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.25

    _sla_once(backend, q, k, v, sparsity, False)
    _sla_once(backend, q, k, v, sparsity, True)
    _ck_once(q, k, v)
    torch.cuda.synchronize()

    ck = [_ck_once(q, k, v) for _ in range(repeats)]
    ck_median = statistics.median(ck)
    for all_int8 in (False, True):
        rows = [
            _sla_once(backend, q, k, v, sparsity, all_int8)
            for _ in range(repeats)
        ]
        torch.cuda.synchronize()
        keys = ("route", "quant_q", "quant_k", "core", "sla_total")
        values = " ".join(f"{key}={_median(rows, key):.3f}ms" for key in keys)
        ratio = _median(rows, "sla_total") / ck_median
        mode = "all-int8" if all_int8 else "fp16-pv"
        print(
            f"L={length} H={heads} mode={mode} {values} "
            f"ck={ck_median:.3f}ms ratio={ratio:.3f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sparsity", type=float, default=0.85)
    parser.add_argument(
        "--sm75-library", type=Path,
        help="Development-only SM75 DLL/.so override for A/B kernel benchmarks.",
    )
    parser.add_argument(
        "--torch-sm75-preprocess", action="store_true",
        help="Force the bounded-memory PyTorch SM75 routing/quantization fallback.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("this benchmark requires an SM75 CUDA GPU")
    backend = _load_backend()
    if args.torch_sm75_preprocess:
        backend._SM75_TORCH_PREPROCESS = True
    if args.sm75_library is not None:
        library_path = args.sm75_library.resolve()
        digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
        native = backend._load_sm75_backend()
        native._LIBRARY = None
        native._LOAD_ERROR = None
        native._library_path = lambda: (library_path, {"sha256": digest})
    print(
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} sparsity={args.sparsity:.3f}"
    )
    for length in args.lengths:
        benchmark(backend, length, args.heads, args.repeats, args.sparsity)


if __name__ == "__main__":
    main()
