# MiniMax H3 Activation Chunk & Attention Acceleration - Star7

[中文](README.md) · [Benchmarks](BENCHMARKS.md) · [Example workflows](examples/workflows)

Run high-quality, long-duration MiniMax H3 video generation on GPUs with limited VRAM. The core node combines independent QKV, RoPE, and MLP activation chunking with a selectable attention backend. It does not alter the sampler, sigma schedule, latent layout, VAE, duration, frame count, or output resolution.

## Main features

| Feature | Description |
|---|---|
| Independent QKV / RoPE / MLP chunking | Controls the temporary VRAM peak of each stage while preserving the original H3 block, weights, LoRA, and conditioning layout |
| Targeted OOM reduction | Halves only the chunk size of the stage that actually ran out of memory |
| Attention selection | Supports `existing`, Comfy Kitchen INT8, SLA, Sol, and CK/Sparse/CK Hybrid paths |
| Architecture-specific sparse kernels | Bundled native CUDA kernels for SM75; Triton or the official NVIDIA Sol-Attn path for SM80+ |
| Numerical diagnostics | Detects NaN/Inf and can identify the first failing QKV, attention, `out_proj`, or MLP stage |
| Lightweight utilities | Reference-image, reference-video, prompt-loading, and workflow-export helpers |

Chunking reduces temporary activation residency; it does not remove tokens or reduce theoretical FLOPs. It is most useful when an unchunked workload would OOM, spill into shared memory, or page frequently.

## Attention backends

The dropdown intentionally uses stable backend IDs so workflows remain portable across languages and machines.

### General

| ID | Purpose |
|---|---|
| `existing` | Keep the incoming model's current attention implementation, including an upstream Sage or other patch |
| `comfy_kitchen_int8` | Use ComfyUI / Comfy Kitchen INT8 attention |

When an enhanced upstream loader has already installed VSA, select `existing` to add only QKV, RoPE, and MLP chunking. Alternatively, select the direct `vsa_sm75` or `vsa_sm80+` path: SM75 uses Star7's precompiled CUDA producer, while SM80+ uses Comfy Kitchen Sol-Attn. Ordinary H3 checkpoints run the fine sparse branch; gated FastH3/VSA checkpoints additionally run the learned coarse correction, which generally preserves quality better. Ordinary H3 emits an explicit warning instead of being rejected. Both paths still stop before sampling if their producer is unavailable instead of silently running dense attention.

### SM75 / RTX 20 series

| ID | Computation path |
|---|---|
| `sla_sm75_qk_int8_pv_fp16` | SLA with INT8 QK, FP16 PV, and FP32 softmax/accumulation |
| `sla_sm75_all_int8` | SLA with INT8 QK/PV and protected full attention for target-audio queries |
| `sol_sm75_all_int8` | Sol Q64/K64 exact selected blocks plus centroid approximation, with INT8 PV |
| `vsa_sm75` | H3 VSA with 10% keep over the full 0%–100% interval; ordinary H3 is fine-only, while gated FastH3 adds coarse correction |
| `hybrid_sm75_ck_sla_all_int8` | CK / SLA All-INT8 / CK across sampling steps |
| `hybrid_sm75_ck_sol_all_int8` | CK / Sol All-INT8 / CK across sampling steps |
| `hybrid_sm75_ck_vsa` | CK / SM75 VSA / CK across sampling steps |

### SM80+ / RTX 30–50 series and newer

| ID | Computation path |
|---|---|
| `sla_sm80+_qk_int8_pv_bf16` | SLA with INT8 QK, BF16 PV, FP32 softmax/accumulation, and full-attention audio queries |
| `sla_sm80+_all_int8` | SLA INT8 QK/PV comparison mode with full-attention audio queries |
| `sol_sm80+_bf16_official` | Official NVIDIA BF16 exact+approx Sol-Attn with audio KV sinks and full-attention audio queries |
| `sol_sm80+_all_int8` | Star7 exact+centroid Sol with INT8 PV, audio KV sinks, and full-attention audio queries |
| `vsa_sm80+` | H3 VSA with 10% keep over the full 0%–100% interval; ordinary H3 is fine-only, while gated FastH3 adds coarse correction |
| `hybrid_sm80+_ck_sla_qk_int8_pv_bf16` | CK / SLA BF16-PV / CK |
| `hybrid_sm80+_ck_sol_bf16_official` | CK / official NVIDIA BF16 Sol / CK |
| `hybrid_sm80+_ck_sla_all_int8` | CK / SLA All-INT8 / CK |
| `hybrid_sm80+_ck_sol_all_int8` | CK / Star7 Sol All-INT8 / CK |
| `hybrid_sm80+_ck_vsa` | CK / SM80+ VSA / CK |

SLA uses dynamic Top-K block routing. Sol combines exact selected-block contributions with centroid approximations for non-selected blocks. Hybrid switches the backend between complete denoising steps; it does not mix two kernels inside one attention call. CK/VSA Hybrid uses CK for the protected first and last steps and VSA for the middle steps. The BF16 Hybrid IDs remain available for existing workflows; the All-INT8 Hybrid IDs are separate opt-in modes and are not silent migrations.

On SM80+, every SLA and Sol mode replaces sparse results for reference- and generated-audio query ranges with full attention computed from the pre-quantization Q/K/V tensors. Video queries remain sparse. Hybrid inherits the same protection during its sparse steps.

Sparse attention is not guaranteed to outperform CK at every resolution, duration, or GPU. Compare sampling time under the same model, seed, frame count, step count, and offload policy.

## Nodes

| Node | Purpose |
|---|---|
| `MiniMax H3 Enhanced Loader - Star7` | Independent H3 model loader bundled with this project; selects protected FP16 or native BF16 by GPU architecture, preserves quantized dispatch, and uses a distinct class ID to avoid conflicts with the standalone FP16 project |
| `MiniMax H3 VRAM Chunk Acceleration - Star7` | QKV/RoPE/MLP chunking, targeted OOM reduction, and attention selection |
| `Reference Video Load - Star7` | Drag-and-drop video loading, time-range trimming, long-edge limiting, synchronized video/audio output |
| `Reference Image Load - Star7` | Drag-and-drop loading, long-edge limiting, optional upscale, and maximum-area centered cropping for common landscape/portrait ratios |
| `Prompt Load - Star7` | Extract prompts from dropped image, video, or workflow JSON files and retain alternative candidates |
| `Video and Workflow Export - Star7` | Export video alone or with embedded/separate workflow metadata |
| `DLSS Neural Image Enhance V2 - Star7` | Adjustable Neural Rendering for an image or video-frame batch, with a target megapixel count and realistic/portrait/anime presets |
| `MiniMax H3 All-in-one Conditioning - Star7` | Builds text, keyframe, reference image/video, and audio conditioning in one node and emits reusable sampling context |
| `MiniMax H3 One-click HD Upscale - Star7` | Upscales the sampled H3 latent to a target megapixel count with optional short refinement; VAE decoding remains external |
| `MiniMax H3 Chunked Decode - Star7` | Independently decodes a complete H3 audio-video latent using the current H3 VAE's native temporal streaming and spatial tiling |
| `MiniMax H3 One-click Face Repair - Star7` | Performs face detection, tracking, local second-pass sampling, and compositing, then returns a standard packed H3 latent for chaining with HD upscale and one final external decode |

Chinese ComfyUI environments display Chinese node and control labels; other locales display English. Attention backend IDs remain unchanged.

### H3 One-click Face Repair

Replace the original conditioning node with All-in-one Conditioning, then connect Sampled result and Sampling context to One-click Face Repair. It reuses the existing model chain, so no second loader, LoRA, or chunk chain is required. Disabling repair or detecting no face passes the input latent through unchanged without an extra VAE encode.

The top rows are Enable repair, Face detector model, Face-repair LoRA, Face-repair LoRA strength, and Face-repair attention. The detector selector uses an installed model from `models/ultralytics/bbox`. LoRA and attention inherit the first pass by default. Selecting a LoRA adds it to the incoming first-pass model at the independent model strength for this face-repair run only; selecting an attention backend overrides only the internal face sampling and restores the first-pass configuration afterward. When Inherit first pass is selected, the face-repair LoRA strength does not reapply an upstream LoRA.

Repair crops remain separate until Star7 Chunked Decode composites them in RGB; neither setting of Preserve repair detail re-encodes the full video. Enabling it restores the legacy ~1 MP minimum output canvas (480×864 becomes 768×1376), before pasting decoded repair crops. Disabling it keeps the source resolution. Larger HD outputs are never shrunk. Use the same video VAE for conditioning and final decode. Place optional HD before Face Repair, then connect directly to Star7 Chunked Decode. Ordinary latent decoders do not read the deferred crops. The hidden legacy IMAGE-output class remains available and shares the same RGB compositor.

The established presets are restored: Balanced 0.30 / crop 2.6, Realistic 0.25 / 2.8, Distant Face 0.48 / 2.4, Anime 0.32 / 2.7. Tracking, per-frame denoise weighting and RGB compositing retain the original FaceRefine-based behavior. Existing Custom settings are preserved; select a preset to adopt its restored parameters.

Missing-subject frames stay on the original audio/video timeline. Incompatible repair frame counts fail explicitly. Higher repair strengths can still change identity or head pose; the decoder integration does not correct content already distorted during generation.

The node supports 1–4 faces, Main, Center, and Reference Match selection, plus Balanced, Realistic, Distant Face, Anime, and Custom presets. If fewer faces are found, it processes the available count. Reference Match optionally uses InsightFace; other modes require no reference image.

On first use, `face_yolov8m.pt` is downloaded, verified, and stored under `ComfyUI/models/ultralytics/bbox`. Connected sockets display prompt tags: `<Picture N>`, `<Video N>`, `<Audio N>`, and the dedicated driving-audio tag `<Audio D>`.

```text
Sampler Sampled result -> One-click HD -> One-click Face Repair -> H3 Chunked Decode
All-in-one Sampling context --------------------^              Video/audio VAEs --^
```

### H3 One-click HD Upscale

The top-level Enable HD second pass switch returns the latent unchanged without loading the HD model when disabled. The next rows are Latent upscaler model, Second-pass LoRA, Second-pass LoRA strength, and Second-pass attention, followed by the preset and refinement controls. LoRA and attention inherit the first pass by default. A selected LoRA is added to the incoming model at the independent model strength for this HD refinement only; when Inherit first pass is selected, that strength does not reapply an upstream LoRA. Attention can independently select any CK/SLA/Sol/Hybrid path exposed by the chunk node. Every setting remains visible and editable; the switches only decide whether their features run. Enable tiling is a separate VRAM control that builds an overlapping, aspect-aware 2D grid. Target tiles accepts 2–64 and is treated as a lower bound: the grid is chosen to keep tiles reasonably shaped and may round up slightly. For example, 16 requested tiles use 3×6 (18 actual) at 16:9, but 4×4 at 4:3, 3:4, and 1:1. Every boundary follows H3's 2×2 latent-patch alignment, and the report shows requested count, actual count, and grid. Tiling lowers second-pass peak VRAM but increases model calls and runtime; it does not change sigmas, sampler, or audio.

The HD node does not decode either VAE internally. Connect its output to the independent MiniMax H3 Chunked Decode node, then to Video Combine. The decoder reuses the H3 VAE's native temporal streaming and spatial tiling and is independent from the second-pass tile count. A normal IMAGE output still retains all decoded frames in system memory, so long 4K clips remain RAM-heavy.

The latent upscaler model is `minimax_h3_latent_upscaler_3d_fp16.safetensors`. On the first run that actually needs enlargement, a missing default is downloaded from the HF mirror first and Hugging Face second, verified against the pinned SHA-256, and installed in `ComfyUI/models/latent_upscale_models`. Failure messages include the exact target directory and each source error. Only this end-to-end validated FP16 checkpoint is currently exposed; unrelated LTX files and BF16/FP32 precision copies of the same training are not presented as distinct quality models.

The second pass keeps the original prompt and reference conditioning. For first- or last-frame generation, All-in-one Conditioning retains the source endpoint pixels and the HD node VAE-encodes them again at the actual target resolution before refinement. Tiled refinement crops those rebuilt HD keyframes with each tile instead of reusing stale low-resolution keyframe latents.

Each preset pairs its step count and denoise range. Refine strength only selects where refinement starts on the native denoising path: higher values start from a noisier latent and permit more repainting, while `0` disables refinement. Refine steps only subdivide that selected range down to `0`, so adding quality steps no longer silently raises the starting Sigma. The node walks the current prompt ancestry for Turbo/PDD model names and combines that evidence with conservative model-patch inspection: Turbo uses the two/three-step scene recipes, while Base and ordinary LoRAs use four to six steps. An already-applied PDD model must follow its trained discrete tail boundaries, so its effective strength is derived from the selected PDD tail. The node reads the upstream 4/6/8-NFE or custom trained partition and switches to Euler internally, with no separate PDD Scheduler connection required. Distant Small Face expands the repaint range, Fast Motion narrows it, and Custom remains adjustable. A read-only row shows the exact Sigma sequence, video Shift, and detected model profile produced by the backend; the number of model evaluations is one fewer than the number of Sigma values.

```text
Sampler Sampled result -> One-click HD Sampled result -> H3 Chunked Decode -> Video Combine
All-in-one Sampling context ----------------^             Video/audio VAEs --^
```

### DLSS Neural Image Enhance V2

Place the single model file directly at:

```text
ComfyUI/models/upscale_models/nvngx_dlssnr.dll
```

The node accepts a single image or a video-frame batch and provides Realistic, Portrait, 3D Anime, 2D Anime, and Custom presets. Custom values are stored in the workflow, and Reset restores the preset defaults.

`Target pixels (MP)` preserves the source aspect ratio and never downsizes when the target is below the input resolution. If the model is missing, the node tries the HF mirror, Hugging Face, and GitHub, then installs it only after verification.

`nvngx_dlssnr.dll` provides Neural Rendering. Enlargement combines high-quality resizing with NR at the target size; it is not the complete in-game DLSS Super Resolution rendering pipeline.

## Core parameters

| Parameter | Purpose | Suggested start |
|---|---|---:|
| `chunk_tokens` | RoPE token chunk limit | `8192` |
| `mlp_chunk_tokens` | MLP expanded-activation chunk limit; usually the primary VRAM control | `8192`, then `4096` when memory is tight |
| `qkv_chunk_tokens` | QKV projection workspace chunk limit | `8192`, then lower when needed |
| `auto_halve_on_oom` | Retry only the failed chunk stage at half size | `true` |
| Attention output memory protection | Off installs no `out_proj` wrapper; select Auto explicitly to protect risky long sequences | `Off` |
| `reuse_mlp_weights` | Reuse prepared QKV/MLP weight snapshots when safe | `true` |
| Attention acceleration method | Select existing, CK, SLA, Sol, or Hybrid attention | `comfy_kitchen_int8` |

Automatic downshift and attention-output protection are independent. The first
handles recoverable QKV, RoPE, and MLP chunk OOMs; the second handles only the
attention `out_proj` peak. Retired prefetch values in older workflows migrate
to Auto, and the internal fallback tile is not exposed in the UI.

For RTX 20-series GPUs, pair this project with [MiniMax H3 FP16 Exact Fix - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7). It adds FP16 numerical protection without converting CK/SLA/Sol INT8 attention calculations to FP16.

Compressed/T8 H3 checkpoints can stack native 8-wide LoRAs with converted full-model 2688-wide LoRAs loaded through ComfyUI's standard LoRA node. The chunk node maps only incompatible full-width AdaLN contributions through the compressed time curve and leaves native T8 and compatible backbone patches unchanged. An unconverted original Turbo LoRA still requires its dedicated loader; FastH3 VSA `adapter_model.safetensors` is conversion input rather than a runtime LoRA.

## Installation

Comfy CLI:

```bash
comfy node install minimax-h3-chunk-star7
```

Manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-chunk-star7.git
```

Restart ComfyUI after installing or updating.

## Example workflow

- [General workflow - English](examples/workflows/MiniMax-H3-Activation-Chunk-Star7-English.json): fully translated canvas and notes with all-in-one conditioning, chunk acceleration, live preview, independent H3 chunked decode, and optional second-pass refinement disabled by default. It is ready to use as a normal workflow immediately after import.
- [通用工作流（中文）](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json): Chinese version with the same features and defaults.

## Recorded 1.0MP / 10-second result

Test conditions: 1.0MP, 10 seconds, 24fps, MiniMax H3 INT8 Tensorwise + ConvRot, 768p Turbo 4-step LoRA, Euler/simple, RTX 2080 Ti 22GB.

| Attention path | Average sampling | Complete task | Relative CK step throughput |
|---|---:|---:|---:|
| KJNodes SM75 SageAttention 2 | `190.50 s/step` | `863.41 s` | about `0.63×` |
| Comfy Kitchen INT8 | `119.50 s/step` | `620.32 s` | `1.00×` |
| SLA SM75 QK-INT8/PV-FP16 | `96.68 s/step` | `471.12 s` | about `1.24×` |
| SLA SM75 All-INT8 | `60.83 s/step` | `325.51 s` | about `1.96×` |
| Sol SM75 All-INT8 | `88.71 s/step` | `442.57 s` | about `1.35×` |
| Standard CK + Sol Hybrid | `106.12 s/step` | `498.27 s` | about `1.13×` |
| Standard CK + SLA Hybrid | `94.67 s/step` | `454.76 s` | about `1.26×` |

These are observations from one local configuration, not cross-GPU performance guarantees. See [BENCHMARKS.md](BENCHMARKS.md) for methodology and additional details.

## Compatibility notes

- SM75 Windows x64 ships with a CUDA 13 static-runtime DLL and requires an NVIDIA 580+ driver.
- SM75 Linux x86_64 ships with a CUDA 12.6 static-runtime `.so`, targets Ubuntu 20.04 / glibc 2.31 or newer, and requires driver 525.60.13+.
- SM80+ SLA paths use Triton and compile/cache kernels on first use.
- Official BF16 Sol first uses ComfyUI 0.34's compiled `comfy_kitchen.sol_attn`
  dispatcher when available, then falls back to the bundled NVIDIA Triton path.
- BF16 remains the default on SM80+. If the launcher explicitly enables `--fp16-unet`, the latest Star7 loader installs FP16 Exact protection so CK, SLA, Sol, and Hybrid can continue. Only ordinary unprotected FP16 is rejected before sampling with a clear loader/launcher diagnostic.
- The official SM80+ Sol mode bundles the relevant NVlabs/Sana `sol-engine` source.
- Strict SLA/Sol/Hybrid modes stop on architecture, environment, self-test, or computation failures; they do not silently fall back to CK or Sage.
- NaN/Inf guards detect and locate invalid output. They do not replace invalid values with zero and are not an FP16 repair mechanism.

## H3 Live Preview

`MiniMax H3 Live Preview - Star7` has a top-level Show preview switch. Off returns the incoming MODEL unchanged and installs no sampling callback, decoder load/download, decode, encoding worker, or transport. When enabled, it decodes uniformly sampled temporal positions after each eligible H3 sampling step with `taeh3.safetensors`; the installed alias `taeh3_decoder.safetensors` is also accepted. Hovering over the preview reveals a compact timeline: dragging scrubs across the sampled positions, and releasing resumes looping from the selected frame. Returning from another browser tab redraws retained frames or recovers the latest WebP from the backend if the page slept through its websocket event. The animation still updates at each configured step, but repetitive model-residency and successful-encode INFO messages from preview housekeeping are suppressed; initial decoder detection, download state, and all warnings/errors remain visible so sampling-speed lines stay easy to compare. If neither decoder filename is present in `models/vae_approx`, the node races the HF mirror and the pinned madebyollin/taehv source first, then tries the remaining fallbacks, verifying SHA-256 without blocking sampling. Preview starts at the next sampling callback after the download completes. If it becomes ready only at the final callback and no earlier preview was shown, one final preview is emitted. Download or preview failure never stops the main generation.

The default preview uses 25 uniformly distributed temporal positions at a 512-pixel long edge. `First step only` is disabled by default; when enabled, only Step 1 is decoded and all later preview work is skipped.

## License

Star7 code is distributed under the [MIT License](LICENSE). Bundled NVIDIA Sol-Attn source is distributed under its [Apache 2.0 license and third-party notices](vendor/sol_attn/THIRD_PARTY_NOTICES.md). The H3 AdaLN curve grid and adaptation provenance are documented in the [Larryvrh H3 Turbo notice](vendor/LARRYVRH-H3-TURBO-NOTICE.md).
