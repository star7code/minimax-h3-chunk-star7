# Changelog

## Unreleased

- Changed `disable_dynamic_prefetch` to default off for faster next-block
  preloading, including both packaged example workflows. Workflows that already
  saved the option as `true` keep that explicit choice; users can enable it if
  preloading or block switching causes a VRAM error.

- Added automatic Chinese UI localization. Chinese ComfyUI environments now
  show plain-language node titles, parameter labels, tooltips, boolean labels,
  and runtime chunk status; other locales remain English. Internal input names,
  attention backend values, and workflow serialization are unchanged.
- Preserve the last effective RoPE/MLP values and OOM reduction state when the
  interface language changes, while discarding stale status after the user
  changes either configured chunk size.

- Added isolated resident MLP snapshots in v2.2.8 and updated tuning guidance:
  keep RoPE at a larger value by default and lower MLP first when VRAM is
  tight. The documented local measurements show substantially more memory
  sensitivity in MLP chunks than in RoPE chunks.

- Fixed model-compatibility regression when the activation chunk node was used
  without LoRA or without the Star7 FP16 loader. The node now patches only the
  H3 MLP token path and preserves the upstream DiT block, FP16 Exact, Sage,
  low-VRAM attention, and third-party model patches.
- Removed the whole-block residual replacement from the installed runtime path;
  this also avoids extra per-chunk norm/modulation launches and restores the
  native BF16 path when the FP16 node is absent. The MLP output buffer remains
  owned by the upstream block, so the runtime trades some output residency for
  compatibility; diagnostics now report the actual expansion chunk size.
- Fixed runtime status rows so they show the actual `min(configured, sequence)`
  chunk for short packed sequences and distinguish sequence limits from OOM
  degradation without changing saved workflow values.
- Added isolated MLP resident snapshots. fc1/fc2 are prepared one at a time,
  including active LoRA/weight patches, cloned out of AIMDO/VBAR staging
  storage, and then reused across chunks. Snapshot failure or OOM falls back
  to streamed execution; the input remains compatible with old workflows.

- Reduced both example workflows from 35 Mbps to 15 Mbps H.264 output for a
  better quality/size balance at the packaged 1.0 MP, 24 fps target.
- Moved the guider and custom sampler down to clear the taller runtime-status
  layout of the activation chunk node.

- Hid the legacy `MiniMaxH3RoPEChunkPatch` alias from new-node search while
  retaining its class ID for old workflow compatibility.
- Migrated both packaged workflows to `MiniMaxH3ActivationChunkStar7` so only
  the canonical activation chunk node is used in new examples.

- Fixed legacy ComfyUI workflow loading so display-only runtime status rows no
  longer shift saved widget values or reset `attention_backend` to `existing`.
- Sanitize workflows already saved with status rows in `widgets_values`, and
  serialize only the seven real node inputs on subsequent saves.
- Render effective RoPE and MLP values in the always-visible status label for
  ComfyUI themes that hide disabled text-widget values.

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
