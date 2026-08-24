# Changelog

## Unreleased

- Fixed the reported reference-speech regression with a validated SM75 QKV
  quality cap of 4,096 tokens. In a same-seed four-step Turbo-LoRA A/B run,
  QKV 8,192 changed decoded PCM versus 4,096 (20.18 dB difference), while
  changing only MLP from 4,096 to 8,192 remained PCM hash-identical. SM80+
  keeps the requested QKV size; OOM fallback may still reduce either path.
- SM75 QKV now prepares one private weight snapshot and reuses it across any
  speech-stable tile at or below 4,096 instead of repeating dynamic weight
  staging for every token chunk. Snapshot OOM falls back to the same numerical
  path with per-chunk streaming.
- At `S=103,546`, QKV snapshot reuse reduced the first-block QKV stage from
  565.5 ms to 391.3 ms. The complete four-step run averaged 196.27 seconds per
  step, and its decoded PCM was bitwise identical to both automatic-cap and
  explicit-QKV-4,096 validation runs.
- The Star7 reference loader now preserves source audio up to 15 seconds
  instead of trimming it to the shorter `17n+5`-aligned video duration, which
  could remove as much as 16/24 seconds and cut the final spoken syllable.
- Clarified that `ref_audio_N` creates the standalone `<Audio N>` speech/voice
  reference used by existing workflows, while `ref_video_audio_N` deliberately
  creates a different fused video-audio conditioning block.

- Fixed automatic VRAM fallback for zero-valued RoPE/MLP/QKV controls. Zero
  tries one full-sequence operation first except for the SM75 QKV speech cap;
  after a reducible OOM, only the failing stage is halved and reused by later
  H3 blocks. Previously QKV zero bypassed fallback entirely.
- Full Q/K/V buffer allocation now retries once after allocator-cache cleanup
  and reports a distinct non-chunkable OOM instead of suggesting unrelated
  MLP/RoPE reductions.
- New Activation Chunk nodes open with RoPE/MLP at 8,192 and QKV at 4,096 on
  SM75; SM80+ opens QKV at 8,192. Saved workflow inputs remain readable, while
  SM75 values above the validated quality cap run at 4,096 and report it.
- Added the standalone `Reference Video Load - Star7` node with only a video
  selector, H3 long-edge limit, and explicit small-video upscale switch. It
  decodes directly to the selected 32-pixel-aligned canvas, fixes output to
  24fps, caps references at 15 seconds, trims to the H3 `17n+5` frame grid,
  and extracts the source stereo soundtrack up to 15 seconds without VHS.
- Kept small-reference upscaling disabled by default because interpolation
  adds no source detail while increasing H3 reference tokens. The explicit
  switch remains available for structure/motion A/B tests.
- RTX 2080 Ti 22GB validation on the supplied 0.6MP/9s reference workflow cut
  one-step diagnostic runtime from 311.60s to 232.50s (-25.4%), sequence length
  from 103,546 to 87,101, and sampler time from 98.14s to 83.02s.
- Documented that MLP/QKV chunk sizes control temporary activations, not model
  weight residency. On the resized reference case, 59,904 caused allocator
  failures and an automatic MLP reduction, while 8,192 completed faster.

- Extended the self-contained SM75 FP16 Exact overflow formula from strict SLA
  to Comfy Kitchen INT8. This also applies inside full and cached-prefix block
  passes when an external H3 block-loop cache such as TE-Speed precedes Star7.
- Added a joint H3 video/audio model-output finite guard for every attention
  selection. It identifies the corrupt stream during sampling, before Audio VAE
  decode or FFmpeg muxing, and never disguises invalid generated content by
  silently replacing NaN/Inf samples with zero.
- Clarified that finite guards are the last line of defense after automatic
  SM75 FP16 Exact prevention, and include the first failing transformer-block
  index in strict-SLA diagnostics.
- Kept the SM75 precision rewrite strictly isolated to compute capability 7.5;
  architecture regression tests now cover SM75, SM80, SM86, SM89, and SM120.
- Consolidated startup and first-block diagnostics into compact configuration,
  QKV, attention, and MLP summaries. Expected dense/quantized QKV differences
  are reported as profiles instead of a misleading warning.

- SM75 strict SLA now installs its validated FP16 Exact formula automatically
  when the standalone Star7 loader is absent: FP32 residual/SwiGLU, FP16
  branches, and power-of-two protected attention `out_proj`/MLP `fc2`. This is
  restricted to SM75; SM80+ keeps its native BF16-capable path.
- Added a consuming SLA input path. Once routing and Q/K quantization finish,
  the full FP16 Q/K sources are released; All-INT8 also recycles FP16 V storage
  after V quantization. At `S=59,904`, 56 heads on RTX 2080 Ti, the measured
  attention allocation above resident Q/K/V fell from 2.017 GiB to 0.815 GiB
  for All-INT8, matching FP16-PV's new peak.

- Added a bundled Linux x86_64 SM75 ABI-v7 library built with CUDA 12.6 on
  Ubuntu 20.04/glibc 2.31. It uses the CUDA static runtime and has no PyTorch
  C++ ABI or SageAttention dependency. Windows and Linux payload hashes are
  verified from the package manifest.
- SM75 no longer requires a working Turing Triton installation for routing and
  quantization. A bounded-memory PyTorch preprocessing path activates only when
  Triton is missing or cannot compile; the native CUDA SLA attention core still
  executes and strict failure semantics remain unchanged.
- QKV projection chunks now write strict SLA inputs directly into their final
  FP16 backend layout, avoiding an extra full Q/K/V conversion copy. Setting
  `qkv_chunk_tokens=0` disables projection chunking without changing the chosen
  attention backend.
- Added one strict NaN/Inf check after each complete transformer block so invalid
  attention, residual/gating, or MLP states stop before VAE decoding instead of
  becoming checkerboard/flicker output. It replaces the previous SLA-core-only
  check rather than adding more per-block host synchronizations.

- Split strict SLA selection into architecture-explicit names:
  `sla_sm75_qk_int8_pv_fp16`, `sla_sm75_all_int8_experimental`, and
  `sla_sm80+_qk_int8_pv_fp16`. The unreleased
  generic SLA value was removed rather than retained as a compatibility alias.

- Added ABI v7 with a separate SM75 All-INT8 experimental entry point. It uses
  per-channel INT8 V and U8 softmax probabilities for PV while retaining FP32
  softmax state/accumulation, corrected routing, per-16-row Q scaling, and the
  native target-audio full-attention guard. It never silently substitutes the
  recommended FP16-PV kernel or CK.
- Full H3 validation completed at 60.83 seconds/step and 325.51 seconds total.
  Tail audio remained balanced at -72.49/-72.57 dB RMS; random numerical error
  against FP32 reference measured 0.000155 mean and 0.000830 maximum. The mode
  remains explicitly experimental rather than becoming the default.

- Kept the recommended FP16-PV implementation in the ABI v7 SM75 library:
  Q/K are INT8, PV is FP16, and online softmax/PV accumulation remain FP32.
  Fixed the Turing `m16n8k8` output-row mapping and changed Q quantization to
  per-16-row scaling to reduce accumulated long-sequence error.
- Added an in-kernel full-attention guard for the target-audio query blocks while
  keeping video queries dynamically sparse. On the validated H3 sequence, video
  sparsity is 85.08% and overall effective sparsity is 83.93%. This is deliberate
  SLA quality protection, not a CK/Sage failure fallback.
- RTX 2080 Ti validation at sequence length 75,872 and 56 heads completed four
  steps at 96.68 seconds/step, versus the recorded approximately 118 seconds/step
  CK baseline. The earlier 59.18-second pure-INT8 result was retired because of
  unacceptable late-video visual/audio error.
- Added a persistent writable Triton cache location for common SLA routing and
  quantization kernels. Cold-start compilation remains separate from steady-state
  sampling time. SLA errors still stop the task; no silent fallback was added.

- Added strict architecture-specific MiniMax H3 SLA attention backends. They follow
  LightX2V dynamic block routing (Q128/K64, 85% target video sparsity); both SM75
  and SM80+ use INT8 QK/FP16 PV with FP32 online softmax and accumulation.
  It never silently falls back to CK or Sage.
- Added strict SLA preflight and one-time numerical self-test diagnostics.
  SM75/RTX 20-series now uses a precompiled native CUDA core;
  SM80+ continues to use Triton. Missing binaries and self-test failures stop
  explicitly without changing attention backends.
- The Windows x64 SM75 core has an ABI-stable raw-pointer/stream interface with
  no PyTorch C++ or SageAttention runtime dependency. Its SASS contains native
  Turing IMMA instructions for QK and native FP16 tensor-core instructions for PV.
- Added the SLA choice without changing the node input count or the serialized
  positions of existing workflow fields.

- Removed the H3 dynamic next-block prefetch experiment from execution. The legacy
  workflow field remains as a non-functional compatibility placeholder so saved
  workflows keep later widget positions intact.

- Set the `RandomNoise` seed mode to `randomize` in both final example
  workflows. Positional and named widget serialization now agree, including on
  older ComfyUI frontends that restore the seed mode by widget position.

- The old prefetch toggle is now shown as a removed experimental feature; its
  serialized field remains only as a compatibility placeholder and runtime
  execution always keeps prefetch disabled.

- Removed all `<Picture 1>` references from the built-in prompts in both
  example workflows. The connected multiline prompt and the conditioning
  node's positional/named fallback values now use the same no-reference prompt,
  so bypassing or disconnecting the reference-image branch does not leave an
  invalid image reference in the prompt.

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
