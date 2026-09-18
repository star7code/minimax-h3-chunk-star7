# Changelog

## 2.16.2 - 2026-09-18

- Fade face masks at crop boundaries to prevent hard seams on extreme close-ups; preserve masks whose feathered support stays clear of those edges.
- Replace Chinese and English examples with the 0919 workflow: optional HD, optional Face Repair, then independent chunked RGB decoding.

## 2.16.1 - 2026-09-18

- Fix face-repair detail loss by decoding preserved repair crops and compositing them in RGB in Star7 Chunked Decode, without re-encoding the full video.
- Restore established face-repair crop presets, including 2.4x context for distant faces, while retaining the independent decoder and legacy IMAGE-output node.
- Apply the ~1 MP detail-preserving output floor before final face compositing; keep larger HD frames and original audio unchanged.
- Preserve missing-face frames on the original timeline, reject mismatched repair frame counts, and report final RGB dimensions.
- Use optional HD before Face Repair, then Star7 Chunked Decode. Ordinary latent decoders do not composite the attached repairs.

## 2.16.0 - 2026-09-18

- Added the independent `MiniMax H3 Enhanced Loader - Star7` directly to this
  project. It provides the same protected FP16/BF16 hardware policy as the
  standalone FP16 loader under a distinct class ID, so this project does not
  depend on or conflict with the standalone loader installation.
- Tightened every face-repair crop preset so the detected face occupies more of
  the 512/768 repair canvas; the distant-small-face preset now uses a 1.8x
  face-height crop instead of 2.4x.
- Corrected native SM75 VSA locality protection to follow the real 3D video
  cube grid. Each query cube now retains its temporal, vertical, and horizontal
  neighbours instead of protecting only adjacent flattened tile numbers.
- Added per-node LoRA and attention selection to One-click HD and One-click
  Face Repair. Both inherit the first-pass model by default; a selected LoRA is
  applied only for that node's internal refinement, with an independent model
  strength control directly below each LoRA selector.
- Added an explicit face-detector model selector and aligned both refinement
  node layouts as Enable, Model, LoRA, LoRA Strength, Attention, then task
  parameters. Existing positional workflow values are migrated without shifting
  saved settings.
- Localized refinement node titles, inputs, and outputs in registration metadata
  so current Vue and legacy LiteGraph renderers show the same labels.
- Added architecture-specific `hybrid_sm75_ck_vsa` and
  `hybrid_sm80+_ck_vsa` modes. They use CK on the protected first and last
  sampling steps and VSA on the middle steps.
- Preserve direct and Hybrid VSA selections when workflows are saved and
  reopened instead of replacing them with the CK default.
- Apply each installed Comfy Kitchen version's eager RoPE primitive inside the
  token chunks, preserving chunked VRAM use while matching its CUDA result bit
  for bit across both old and new operation orders.
- Added a latent-output One-click Face Repair node for chaining with HD upscale
  and one final external chunked VAE decode. It re-encodes only the repaired
  video result and preserves the original audio latent exactly; disabled or
  skipped repairs pass the input latent through unchanged.
- Retained the former IMAGE-output face-repair class ID as a deprecated,
  search-hidden Legacy Workflow Compatibility node for existing workflows.
- Keep all One-click HD controls visible and editable when HD refinement or
  tiling is disabled; the switches now affect execution only.
- Allow direct and Hybrid VSA modes to run ordinary H3 checkpoints through the
  fine sparse branch. Gated FastH3/VSA checkpoints retain their learned coarse
  correction, and fine-only operation is reported explicitly instead of being
  rejected.
- Give Base-H3 refine strength and refine steps independent meanings: strength
  now selects the native-flow starting range, while steps subdivide that range.
  Extra quality steps no longer raise the starting sigma, and zero strength now
  genuinely skips second-pass refinement.

## 2.15.0 - 2026-09-17

- Aligned base-H3 second-pass refinement with the tail of its native eight-step
  shifted-flow schedule while keeping the existing compact frontend controls.
- Switched spatial refinement to exact-count long-edge strips with a per-side
  overlap halo, full short-axis context, and support for every count from 2–64.
- Added direct `vsa_sm75` and `vsa_sm80+` attention paths above their
  architecture-specific Hybrid choices.
- Added a precompiled SM75 VSA Q64/K64 All-INT8 fine-attention kernel with
  per-tile `block_len` masking; Top-K routing and the learned FP32 coarse gate
  now run without depending on Comfy Kitchen Sol-Attn or falling back to dense.
- Upgraded ComfyUI H3 VSA block replacements inside Star7 so FP16 Exact blocks
  accept the attention override and every transformer block remains eligible.
- Isolated VSA routing state per spatial second-pass strip so tiled refinement
  cannot reuse statistics from a neighboring strip.
- Collapsed native SM75 VSA activation statistics from one line per transformer
  block to one representative first-block line per sequence shape.
- Run direct `vsa_sm75` and `vsa_sm80+` for the full sampling interval instead
  of spending the first 20% of steps in prohibitively slow dense attention;
  explicit CK/Sparse Hybrid modes retain their independent step schedule.

## 2.14.9 - 2026-09-16

- Keep ComfyUI graph link target-slot indices synchronized when dynamic reference-image inputs are inserted and repositioned.
- Compact legacy or mismatched reference-image gaps so connected inputs remain contiguous before exposing the next empty slot.
- Replace the four overlapping example workflows with one Chinese and one fully translated English general workflow; optional second-pass refinement remains disabled by default.

## 2.14.8 - 2026-09-16

- Continue the reference-image connect-to-grow sequence beyond migrated four-image workflows, up to the H3 limit of 9.

## 2.14.7 - 2026-09-16

- Support both the ComfyUI 0.34 H3 reference-node argument order and the newer optional-VAE order.
- Remove empty legacy reference-image ports, keeping only connected ports plus the next empty port below the last-frame input and above reference videos.

## 2.14.6 - 2026-09-16

- Restore connect-to-grow reference-image inputs instead of displaying every slot.
- Raise the autogrow reference-image capacity from 4 to the upstream H3 limit of 9 while preserving prompt order.

## 2.14.5 - 2026-09-16

- Expand all-in-one conditioning from 4 to 16 reference-image inputs.
- Keep prompt picture numbering contiguous when optional input slots are skipped.

## 2.14.4 - 2026-09-16

- Preserve HD settings when second-pass refinement is disabled and reopened.
- Keep selectable tile counts exact and choose compact grids by frame aspect.

## 2.14.3 - 2026-09-16

- Constrain one-click HD refinement to 1–50 steps in the UI and backend.
- Repair legacy workflows that saved a zero refinement-step value.

## 2.14.2 - 2026-09-16

- Raise the one-click HD target control ceiling to 36.0 MP.
- Use the concise `分格数量` / `Tile count` label consistently across frontend renderers and examples.

## 2.14.1 - 2026-09-15

- Skip HD upscaling and second-pass sampling when the first-pass resolution
  already meets or exceeds the target, preserving the original latent unchanged.

## 2.14.0 - 2026-09-15

- Added one-click MiniMax H3 latent HD upscaling with scene presets, optional
  spatial tiling, automatic model installation, and exact Sigma reporting.
- Added independent chunked H3 audio-video VAE decoding and bilingual HD
  example workflows with face repair disabled by default.
- Fixed tiled H3 global position coordinates, simplified face-repair logging,
  and corrected the disabled repair-status display.

## 2.13.2 - 2026-09-10

- Added optional detail-preserving face-repair output sizing with a practical
  ~1 MP floor and a never-downscale rule.
- Added a read-only repaired-resolution display and clear phase timing logs.
- Fixed single-face compositing after automatic output enlargement.

## 2.13.1 - 2026-09-08

- Fixed the SM80+ Sol All-INT8 Triton kernel across supported GPU architectures.
- Corrected compact-centroid V scaling.

## 2.13.0 - 2026-09-08

- Added All-in-one Conditioning for H3 text, images, videos, keyframes, and audio.
- Added One-click Face Repair with 1–4 face tracking, presets, reference matching,
  local second-pass sampling, and automatic detector-model installation.
- Added bilingual controls, connected-media prompt tags, and bounded media alignment.
- Added DLSS Neural Image Enhance V2 for images and video-frame batches, with
  target-pixel sizing, visual presets, custom controls, and verified model download.

## 2.12.22 - 2026-09-06

- Exposed the SM80+ All-INT8 Hybrid choices that were already implemented:
  `hybrid_sm80+_ck_sla_all_int8` and
  `hybrid_sm80+_ck_sol_all_int8`.
- Preserved the existing SM80+ BF16 Hybrid choices:
  `hybrid_sm80+_ck_sla_qk_int8_pv_bf16` and
  `hybrid_sm80+_ck_sol_bf16_official`. The All-INT8 Hybrid choices are separate
  opt-in modes; serialized BF16 Hybrid workflows are not silently changed.
- SM80+ official BF16 Sol now prefers Comfy Kitchen's compiled `sol_attn`
  dispatcher when available; the bundled NVIDIA Triton implementation remains
  the compatibility fallback.

## 2.12.20 - 2026-09-06

- Reduced redundant work in the SM80+ Star7 All-INT8 Sol path by packing
  unselected K blocks into a compact centroid LUT. Selected blocks are no
  longer recomputed as centroids and then discarded by a mask.
- The runtime implementation label now includes `compact-centroid` so that
  benchmark logs can distinguish this kernel from the previous path.

## 2.12.19 - 2026-09-06

- Added a ComfyUI 0.34 / `comfy_aimdo` compatibility layer for Star7 H3.
  When the Star7 chunk runtime is active, only the H3 block malloc-graph
  replay is bypassed; normal VBAR prefetch, `cudaMallocAsync`, and AIMDO for
  unrelated models remain enabled.
- Added runtime detection and an explicit `[Star7 AIMDO compat]` log entry;
  no startup flag, TE path, FP16 path, or H3 math path is changed.

## 2.12.18 - 2026-09-04

- 精简 H3 Live Preview 日志：预览继续逐步更新，但不再让每一步重复的
  TAEHV/H3 显存驻留信息及成功编码提示打断四条采样速度日志。
- 仅屏蔽由当前预览解码调用产生的已知常规 INFO；首次解码器识别信息、
  下载状态以及任何 warning/error 仍正常显示。
- Turing 第三方插件冲突隔离日志改为纯英文，避免 Windows 非 UTF-8
  控制台乱码，并明确说明需停用插件后重启而非只绕过工作流节点。

## 2.12.17 - 2026-09-03

- Fixed Live Preview widget-value migration. The serialized order remains the
  original `frames / resolution / first-step-only` sequence, with the new
  enable switch appended after it, so opening v1 workflows cannot shift every
  setting into the wrong widget.
- Kept `Show preview` visually at the top through a non-serialized mirror
  switch. Both original workflows and workflows briefly saved by 2.12.16 are
  repaired before LiteGraph assigns positional widget values; the mirror is
  temporarily excluded from that assignment and restored afterwards.

## 2.12.16 - 2026-09-03

- Renamed the compact UI control to `输出显存保护（可能降速）` /
  `Output VRAM guard (may slow down)` while keeping its serialized input name
  compatible with existing workflows.
- Added a top-level `显示预览 / Show preview` switch. Disabling it returns the
  incoming MODEL unchanged and installs no sampler callback, decoder load,
  background download, encoding worker, or frontend transport.
- Cached only the latest encoded preview per node and added foreground/page
  recovery. Returning from another browser tab redraws retained frames or
  fetches the most recent WebP if its websocket event arrived while the page
  was suspended.
- Added one concise per-run log after the first preview is decoded, encoded and
  cached, making backend success distinguishable from a frontend display issue.
- Made explicit `Auto` protection proactive: when the full contraction plus
  output and safety reserve do not fit current free VRAM, it selects a
  budget-derived 256–4096-token tile before the first risky full allocation.

## 2.12.15 - 2026-09-03

- Made `Attention output memory protection = Off` a true zero-overhead control
  path. The node no longer installs an `out_proj` Python wrapper on any H3
  block when protection is disabled, so ordinary CK/SLA/Sol execution retains
  the exact incoming projection hot path.
- Changed new nodes to default to `Off`. Legacy values from the retired boolean
  prefetch field also migrate to `Off`; protection is installed only after an
  explicit `Auto` selection. The earlier live-preview/H3 residency fix remains
  active.

## 2.12.14 - 2026-09-03

- SM75 SLA 的均值池化与分块 INT8 量化改用随节点分发的原生 CUDA 预处理，
  不再依赖 Triton/Python 扩展兼容性；原生入口不可用时仍自动回退到有界显存
  PyTorch 实现，不影响任务继续运行。
- 检测并中和 `comfyui-minimax-h3-turing` 对 MiniMax H3 核心类和模型实例安装的
  全局补丁，随后继续使用 Star7 路线；日志会明确建议停用冲突插件并重启。
- 增加原生预处理数值一致性、二进制校验以及 Turing 类级/实例级补丁恢复测试。

## 2.12.13 - 2026-09-03

- 修复正常显存余量下仍强制 SM75 fused `out_proj` 的性能回退：1.0MP 等常规任务
  现在完整保留上游投影路径，仅在显存风险或真实 OOM 时启用 fused/分块保护。
- 修复实时预览可能触发 H3 权重反复卸载、重载的问题：TAEH3 与当前采样模型共同驻留，
  并以单帧微批直接解码，避免 25 帧预览挤占大量临时显存。
- 显存不足时仍会自动尝试无完整 INT32 工作区的 SM75 fused 投影；不可用或 OOM
  才进入预分配输出的分块兜底，不改变最终采样结果。

## 2.12.12 - 2026-09-02

- 修复新版 Vue 节点界面仍显示内部 `out_proj` 兜底分块控件的问题：同时使用
  新版 `hidden/type/options` 标记与旧版 LiteGraph `computeSize` 隐藏方式。
- 内部值仍正常保存到工作流并传给后端，只统一界面显示，不改变显存保护逻辑。

## 2.12.11 - 2026-09-02

- 实时预览同时维护 `taeh3.safetensors` 与 `taeh3_decoder.safetensors`：
  国内 HF 镜像和固定版本官方 GitHub 源并行准备，两个文件可共存，加载时选择
  任意一个通过 SHA-256 且可识别为 24 通道 TAEHV 的副本。
- 已有文件会在首次使用时快速校验；损坏副本只覆盖自身，不删除另一文件。
  下载失败允许后续任务重试，且不会阻断视频生成。
- 日志明确区分文件缺失、SHA-256 不匹配、下载源失败，以及文件正确但
  ComfyUI 核心过旧、无法识别 H3 TAEHV 的情况。

## 2.12.10 - 2026-09-01

- Replaced the architecture-specific example set with one universal Chinese
  workflow and one fully translated English workflow. Both retain identical
  node IDs, links, backend values, and precision-protection behavior.
- Updated workflow notes for SM75/SM80+ precision selection, current reference
  trimming, TAEH3 mirror fallback, LoRA compatibility, and exact benchmark
  conditions.
- Fixed H3 Live Preview localization so English environments no longer receive
  a hard-coded Chinese title or widget labels.

## 2.12.9 - 2026-09-01

- Added compatibility with both `taeh3.safetensors` and
  `taeh3_decoder.safetensors` decoder filenames, validating the required
  24-channel TAEHV layout before use.
- The automatic preview-decoder download now races the HF domestic mirror and
  the pinned official source using isolated temporary files, then falls back
  to the remaining mirrors with SHA-256 verification. Sampling remains
  non-blocking and only one canonical decoder file is kept.

## 2.12.8 - 2026-09-01

- Added a compact hover-only timeline to H3 Live Preview. Dragging pauses the
  loop and scrubs across all decoded temporal samples; releasing resumes
  playback from the selected position.
- Reused the existing animated WebP payload through browser-side frame decoding,
  adding no backend encoding, transport, or sampling work. Older frontends that
  cannot decode animation frames continue using the original animated preview.
- Added stale-decode cancellation and explicit timer/frame cleanup when a newer
  sampling-step preview arrives or the node is removed.

## 2.12.7 - 2026-09-01

- Added an optional aspect-ratio crop to Reference Image Load. It is disabled
  on new nodes and defaults to 16:9 landscape when enabled.
- Included square, landscape, and portrait presets from 1:1 through 21:9/9:21.
  The node automatically takes the largest centered crop, preserving as much
  source area as possible before applying the existing long-edge limit.
- Kept the selected ratio in normal workflow serialization, synchronized alpha
  masks with the image crop, and added compact bilingual conditional controls.

## 2.12.6 - 2026-09-01

- Replaced the retired prefetch placeholder with automatic attention-output
  memory protection. Normal workloads keep full `out_proj`; risky long
  sequences select an internal 4096/2048-token tile from live CUDA headroom.
- Cached each `out_proj` policy by device and tensor shape, preventing every H3
  block and sampling step from repeating the same full-projection OOM probe.
  Protection logs are emitted once per shape and chunk output is preallocated.
- Added a shape-scoped SM75 Comfy Kitchen TensorWise INT8 dispatch that prefers
  the existing fused CUTLASS contraction for H3 `N=5376, K=7168`, with bounded
  chunks only when full execution is unsafe or unavailable.
- Added Auto/Off and 自动/关闭 workflow compatibility, renamed the visible
  backend control to “Attention acceleration method / 注意力加速方式”, and kept
  the internal fallback tile out of the UI.
- Made per-step backend/timing summaries use progress-aware, permanent lines so
  they remain vertically aligned without corrupting ComfyUI's live progress bar.
- Added compatibility probing for Comfy Kitchen releases that expose INT8
  attention without the optional availability helper.

## 2.12.5 - 2026-08-31

- 所有注意力模式在 SM80+ 上都会于采样前检查 H3 的实际 compute dtype；由
  Star7 载入节点安装 Exact 保护的 FP16 会正常放行，只有普通未保护 FP16 才
  提示检查启动器参数并提前终止任务。
- Chunk 仅负责诊断，不修改模型精度；BF16 固定由对应载入节点完成。
- 未保护 FP16 的错误现在明确标记为上游载入/启动参数配置问题，并说明分块、
  注意力和采样尚未执行，避免被误判为 Chunk 或注意力后端故障。
- 新增轻量 H3 实时预览；缺少 `taeh3.safetensors` 时采用校验哈希的后台下载，
  不阻塞采样，并在下载完成后的下一个可用步骤即时启用预览。
- 实时预览默认调整为 25 个时间位置和 512 长边；新增默认关闭的“只显示
  第一步预览”，启用后跳过 Step 2 及后续预览解码。

## 2.12.4 - 2026-08-31

- Added lossless full-model LoRA AdaLN adaptation for compressed/pruned H3
  checkpoints. Standard ComfyUI LoRA patches with the original 2688-wide time
  input are evaluated through the H3 curve grid, while native 8-wide T8 LoRA
  patches remain on ComfyUI's normal path. Both layouts can be stacked without
  shape errors or dropping compatible backbone patches.
- Preserved external bypass-LoRA forwards when the SM75 QKV chunk path considers
  resident weight reuse, preventing an upstream Turbo LoRA injection from being
  skipped.
- Added full-attention replacement for reference- and generated-audio query
  ranges in every SM80+ SLA and Sol path, including their Hybrid sparse steps.
  Sparse video attention remains unchanged; Sol audio KV sinks and SLA boundary
  routing are retained.

## 2.12.3 - 2026-08-31

- Refreshed the directly connected VHS Video Combine preview after workflow
  metadata export replaces the completed video, and added a scoped playback
  recovery for stale preview URLs. VHS source files and unrelated VHS nodes are
  not modified.
- Added Copy and Paste actions to Prompt Load, arranged with Import File and
  Candidates in one compact, width-responsive row.

## 2.12.2 - 2026-08-30

- Added FastH3 VSA compatibility for upstream enhanced-loader models. Use
  `attention_backend=existing` to preserve VSA while QKV, RoPE, and MLP
  chunking continue to manage activation-memory peaks.
- Suppressed the generic chunk step-timing line when the upstream VSA runtime
  already reports complete step timing.
- Exposed the original Comfy Kitchen RoPE callable on the Chunk dispatcher so
  an independent upstream loader can remove a stale process-wide wrapper when
  no Chunk node is connected. A connected Chunk node installs it again normally.
- Extended workflow-video metadata replacement polling from roughly 6 seconds
  to as long as 120 seconds for long-video encoding.

## 2.12.1 - 2026-08-29

- Added automatic Chinese/English UI labels for the chunk, reference-media,
  prompt-loading, and workflow-export nodes while preserving stable workflow
  field values and attention backend IDs.
- Added a complete English README and an English general example workflow.
- Synchronized the recorded CK baseline to `119.50 seconds/step` and
  `620.32 seconds` across the README and benchmark document.

- Removed the former SM75 QKV `4096` cap. QKV now follows the configured value
  for every architecture and generation type, and only decreases after an
  actual QKV projection OOM when automatic fallback is enabled.
- Clarified the QKV control as the reference-memory adjustment in the node UI.

- Added an optional compact time-range control to `参考视频载入 - Star7`.
  It reads the selected file's real duration, refreshes when the video changes,
  and applies the same uncapped time window to picture and source audio.

- Reorganized the release documentation around the current feature set:
  independent QKV/RoPE/MLP chunking followed by selectable CK, SLA, Sol, and
  Hybrid attention. Removed obsolete prefetch guidance and corrected SM80+
  Hybrid, audio-protection, QKV tuning, distribution, and licensing details.
- Added the compact `提示词载入 - Star7` helper. It imports prompt metadata from
  selected or dropped image, video, and workflow files, prioritizes the longest
  viable positive-text candidate, and provides a candidate picker for review.
- Added the combined reference-image upload and long-edge scaling helper.

- Reorganized the attention selector with explicit SM75 and SM80+ names. Both
  architecture families remain visible so workflows can be prepared or shared
  without the menu changing according to the machine that opened them.
- Restored the SM75 SLA QK-INT8/PV-FP16 quality path to the visible SM75 list.
- Bundled the NVlabs/Sana Sol-Attn source used by the SM80+ official BF16 mode.
  SM80/SM86 and systems without CuTe use its official Triton backend; supported
  SM89/SM90/SM100/SM120 systems use the matching CuTe backend when available.
- The selector now always shows both architecture families. SM80+ exposes
  BF16-PV SLA, official BF16 Sol, and the two All-INT8 comparison paths.
  Its visible hybrids now pair CK with BF16-PV SLA or official BF16 Sol rather
  than forcing All-INT8 in the sparse middle steps.
- Added a real SM80+ SLA BF16 path: Q/K remain INT8, V and PV use BF16 Tensor
  Core operands, and online softmax/final accumulation remain FP32. The former
  FP16-PV names stay loadable for existing workflows but are hidden.
- Promoted the SM75 visible sparse names to `sla_sm75_all_int8` and
  `sol_sm75_all_int8`, while retaining former `*_experimental` values as
  load-time aliases. SM80+ aliases are also preserved for saved workflows.

- Added Q64/K64 Sol attention modes for SM75 and SM80+. The SM80+ recommended
  mode calls NVIDIA's official contiguous-BTHD BF16 `sol_attn()` interface;
  SM75 FP16-PV combines exact selected blocks with K/V-centroid contributions
  for unselected blocks in the same online softmax. Both Star7 All-INT8 modes
  now preserve those complete Sol semantics and change only PV quantization.
- Moved SM75 Sol centroid reduction, diagonal-threshold routing, LUT packing,
  and Q/K quantization to native CUDA. The normal and All-INT8 kernels now both
  execute exact blocks plus centroid approximations without a PyTorch hot path.
- Kept official Sol routing at `tau=1.0`. Full H3 traces showed that increasing
  the threshold only traded routing precision for enough speed to approach CK;
  the SM75 FP16-PV mode is therefore hidden from new-node menus while its
  implementation remains readable by existing workflows. SM75 All-INT8 is the
  visible standard Sol mode.
- Added variable per-query-block Sol routing counts and compact LUTs. The SM75
  native FP16-PV and All-INT8 kernels use Q64/K64, four warps, and skip LUT
  padding rather than computing fake blocks.
- Added architecture-specific CK/Sol/CK Hybrid modes. Attention choices are
  ordered as SLA FP16, SLA All-INT8, Sol recommended, Sol All-INT8, CK-SLA,
  and CK-Sol for SM75 first and SM80+ second.
- Bumped the package version to `2.12.0`.

- Unified the newer-architecture menu names at SM80+, matching the actual
  lower bound of both the SLA Triton path and NVIDIA's official Sol interface:
  `sla_sm80+_*`, `sol_sm80+_*`, and their CK Hybrid modes.
- Fixed newer-GPU SLA and Hybrid overflow by restoring the upstream H3 dtype
  (normally BF16) after the FP16 SLA kernel and before `out_proj`. QKV routing,
  sparse attention, and the existing FP16-PV Triton kernel are unchanged.
- Added a separate SM80+ experimental Triton All-INT8 PV kernel. Q/K/V and the
  per-tile softmax probabilities use INT8 tensor-core products, while online
  softmax state and final accumulation remain FP32. It never silently falls
  back to FP16-PV or CK.
- Bumped the package version to `2.11.0`.

- Added the SM75-only `hybrid_sm75_ck_sla_all_int8` attention scheduling mode.
  It keeps CK in the first and
  last guard regions and reuses the existing experimental SM75 All-INT8 SLA
  only in the middle region. The mode is explicitly approximate and has not
  been presented as a quality-validated replacement for FP16-PV. The previous
  FP16-PV Hybrid option was removed; only the tested All-INT8 Hybrid remains.
- Bumped the package version to `2.10.3`.

- Removed the duplicated automatic SM75 FP16 Exact implementation. Chunk now
  preserves an external FP16 Exact MLP callable per tile, never changes model
  compute dtype, and only warns on SM75 when the companion is absent.
- Corrected SM80+ audio reporting to `routing-priority-only`; only the SM75
  native path currently reports full-attention audio protection. No Triton/SLA
  mathematical path was changed for this diagnostic release.
- Added targeted strict-SLA diagnostics. A failing block is armed for the next
  run, where QKV conversion, routing/quantization, raw SLA output, `out_proj`,
  and each MLP chunk are checked without adding normal-path synchronizations.
  `STAR7_SLA_DEBUG_BLOCK` is supported for remote SM80+/SM120 reproduction.
- SM100/SM120 strict SLA now probes those internal stages during the first run
  automatically and reports the first non-finite stage directly; SM75 and
  established SM80+ paths keep the previous opt-in diagnostics behavior.
- NaN/Inf errors now report bad token rows/ranges and heads, and describe the
  real architecture precision path instead of mentioning SM75 FP16 Exact on
  SM80+/SM120. Repeated model/block finite wrappers are collapsed to one layer.
- Renamed the small startup result to `SLA self-test passed` and separated it
  from real-shape `SLA runtime` routing data. Startup diagnostics now distinguish
  model precision from FP16 SLA buffers and identify newer architectures that
  still require real-device validation.

- Added a private SM75 QKV weight snapshot so projection chunks can reuse the
  prepared weight instead of repeating dynamic staging. Snapshot OOM falls
  back to per-chunk streaming. At `S=103,546`, the measured first-block QKV
  stage decreased from 565.5 ms to 391.3 ms in the recorded workflow.
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
