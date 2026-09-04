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

For a FastH3 VSA model, select `existing`. The upstream enhanced loader owns VSA acceleration; this node adds only QKV, RoPE, and MLP chunking without replacing VSA attention. VSA remains independently usable when this node is absent.

### SM75 / RTX 20 series

| ID | Computation path |
|---|---|
| `sla_sm75_qk_int8_pv_fp16` | SLA with INT8 QK, FP16 PV, and FP32 softmax/accumulation |
| `sla_sm75_all_int8` | SLA with INT8 QK/PV and protected full attention for target-audio queries |
| `sol_sm75_all_int8` | Sol Q64/K64 exact selected blocks plus centroid approximation, with INT8 PV |
| `hybrid_sm75_ck_sla_all_int8` | CK / SLA All-INT8 / CK across sampling steps |
| `hybrid_sm75_ck_sol_all_int8` | CK / Sol All-INT8 / CK across sampling steps |

### SM80+ / RTX 30–50 series and newer

| ID | Computation path |
|---|---|
| `sla_sm80+_qk_int8_pv_bf16` | SLA with INT8 QK, BF16 PV, FP32 softmax/accumulation, and full-attention audio queries |
| `sla_sm80+_all_int8` | SLA INT8 QK/PV comparison mode with full-attention audio queries |
| `sol_sm80+_bf16_official` | Official NVIDIA BF16 exact+approx Sol-Attn with audio KV sinks and full-attention audio queries |
| `sol_sm80+_all_int8` | Star7 exact+centroid Sol with INT8 PV, audio KV sinks, and full-attention audio queries |
| `hybrid_sm80+_ck_sla_qk_int8_pv_bf16` | CK / SLA BF16-PV / CK |
| `hybrid_sm80+_ck_sol_bf16_official` | CK / official NVIDIA BF16 Sol / CK |

SLA uses dynamic Top-K block routing. Sol combines exact selected-block contributions with centroid approximations for non-selected blocks. Hybrid switches the backend between complete denoising steps; it does not mix two kernels inside one attention call.

On SM80+, every SLA and Sol mode replaces sparse results for reference- and generated-audio query ranges with full attention computed from the pre-quantization Q/K/V tensors. Video queries remain sparse. Hybrid inherits the same protection during its sparse steps.

Sparse attention is not guaranteed to outperform CK at every resolution, duration, or GPU. Compare sampling time under the same model, seed, frame count, step count, and offload policy.

## Nodes

| Node | Purpose |
|---|---|
| `MiniMax H3 VRAM Chunk Acceleration - Star7` | QKV/RoPE/MLP chunking, targeted OOM reduction, and attention selection |
| `Reference Video Load - Star7` | Drag-and-drop video loading, time-range trimming, long-edge limiting, synchronized video/audio output |
| `Reference Image Load - Star7` | Drag-and-drop loading, long-edge limiting, optional upscale, and maximum-area centered cropping for common landscape/portrait ratios |
| `Prompt Load - Star7` | Extract prompts from dropped image, video, or workflow JSON files and retain alternative candidates |
| `Video and Workflow Export - Star7` | Export video alone or with embedded/separate workflow metadata |

Chinese ComfyUI environments display Chinese node and control labels; other locales display English. Attention backend IDs remain unchanged.

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

- [General workflow — English](examples/workflows/MiniMax-H3-Activation-Chunk-Star7-English.json): fully translated canvas labels and notes; one graph covers SM75, SM80+, CK, sparse, precision-protection, reference-loading, and live-preview paths.
- [通用工作流（中文）](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json)

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
- BF16 remains the default on SM80+. If the launcher explicitly enables `--fp16-unet`, the latest Star7 loader installs FP16 Exact protection so CK, SLA, Sol, and Hybrid can continue. Only ordinary unprotected FP16 is rejected before sampling with a clear loader/launcher diagnostic.
- The official SM80+ Sol mode bundles the relevant NVlabs/Sana `sol-engine` source.
- Strict SLA/Sol/Hybrid modes stop on architecture, environment, self-test, or computation failures; they do not silently fall back to CK or Sage.
- NaN/Inf guards detect and locate invalid output. They do not replace invalid values with zero and are not an FP16 repair mechanism.

## H3 Live Preview

`MiniMax H3 Live Preview - Star7` has a top-level Show preview switch. Off returns the incoming MODEL unchanged and installs no sampling callback, decoder load/download, decode, encoding worker, or transport. When enabled, it decodes uniformly sampled temporal positions after each eligible H3 sampling step with `taeh3.safetensors`; the installed alias `taeh3_decoder.safetensors` is also accepted. Hovering over the preview reveals a compact timeline: dragging scrubs across the sampled positions, and releasing resumes looping from the selected frame. Returning from another browser tab redraws retained frames or recovers the latest WebP from the backend if the page slept through its websocket event. The animation still updates at each configured step, but repetitive model-residency and successful-encode INFO messages from preview housekeeping are suppressed; initial decoder detection, download state, and all warnings/errors remain visible so sampling-speed lines stay easy to compare. If neither decoder filename is present in `models/vae_approx`, the node races the HF mirror and the pinned madebyollin/taehv source first, then tries the remaining fallbacks, verifying SHA-256 without blocking sampling. Preview starts at the next sampling callback after the download completes. If it becomes ready only at the final callback and no earlier preview was shown, one final preview is emitted. Download or preview failure never stops the main generation.

The default preview uses 25 uniformly distributed temporal positions at a 512-pixel long edge. `First step only` is disabled by default; when enabled, only Step 1 is decoded and all later preview work is skipped.

## License

Star7 code is distributed under the [MIT License](LICENSE). Bundled NVIDIA Sol-Attn source is distributed under its [Apache 2.0 license and third-party notices](vendor/sol_attn/THIRD_PARTY_NOTICES.md). The H3 AdaLN curve grid and adaptation provenance are documented in the [Larryvrh H3 Turbo notice](vendor/LARRYVRH-H3-TURBO-NOTICE.md).
