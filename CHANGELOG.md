# Changelog

## Unreleased

- Fixed legacy ComfyUI workflow loading so display-only runtime status rows no
  longer shift saved widget values or reset `attention_backend` to `existing`.
- Sanitize workflows already saved with status rows in `widgets_values`, and
  serialize only the seven real node inputs on subsequent saves.

- Added visible per-parameter runtime status rows for the effective RoPE and
  MLP chunk values.
- Added clear VRAM fallback diagnostics and remembered successful chunk caps
  across later H3 blocks, Q/K processing, and repeated forwards in the same
  model session.
- Reset remembered effective values whenever the node inputs are re-executed,
  so manual parameter changes always take effect.
- Added per-VRAM and per-OOM-location parameter guidance.
- Added confirmed C/D cases: 0.4MP/10s at 180s and 0.4MP/5s at 85s.
- Clarified that chunking controls memory residency rather than FLOPs, and documented mixed linear/quadratic sequence scaling.
- Clarified the RTX 20-series CK INT8 comparison against the custom SM75 SageAttention 2 path.
- Linked the T8 single-reference conditioning node used by the example workflows.

## 2.1.2 - 2026-08-15

- Replaced the external Jjk-Nodes prompt box with ComfyUI's built-in multiline text node in both packaged workflows.
- Preserved the prompt node ID, position, size, content, and downstream link so the published layouts remain unchanged.

## 2.1.1 - 2026-08-15

- Removed NVIDIA RTX Video Super Resolution from every packaged example workflow.
- Connected VAE decode directly to video output so the examples no longer require RTX upscaler nodes or models.

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
