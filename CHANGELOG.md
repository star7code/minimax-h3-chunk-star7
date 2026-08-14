# Changelog

## 2.1.0 - 2026-08-15

- Added chunked, in-place eager split-half RoPE handling for MiniMax H3 Q/K tensors.
- Added streaming token-chunked H3 MLP execution and residual accumulation.
- Preserved INT8 Tensorwise + ConvRot weights across MLP chunks.
- Added optional Comfy Kitchen INT8 attention selection for Turing-class benchmarking.
- Added compact one-block attention and MLP diagnostics.
- Added RTX 2080 Ti 22GB case data and sanitized general/RTX20 example workflows.

## 2.0.0

- Added MLP activation chunking and quantized-weight reuse.

## 1.0.0

- Initial experimental eager RoPE token chunk patch.
