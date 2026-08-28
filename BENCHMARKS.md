# RTX 2080 Ti 22GB Case Study

This document records one local Windows/ComfyUI case. It is evidence for configuration planning, not a universal benchmark.

## Reproducible configuration

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti, 22GB modified-memory board |
| Launcher memory policy | Keep models resident in VRAM enabled |
| Other launcher options | Stable computation off; Channels-Last off; CUDA built-in async allocator selected |
| Task | MiniMax H3 single-reference mode; one reference image connected |
| Case A | 1.0MP portrait, 10 seconds, 24fps, approximately 243 frames |
| Case B | 0.6MP portrait, 15 seconds, 24fps, approximately 362 frames |
| Case C | 0.4MP portrait, 10 seconds, 24fps, approximately 243 frames |
| Case D | 0.4MP portrait, 5 seconds, 24fps, approximately 124 frames |
| Reference conditioning | [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8), one reference image |
| DiT | INT8 Tensorwise + ConvRot MiniMax H3 FL2VA |
| Text encoder | Qwen3-VL 32B MiniMax H3 INT4 ConvRot |
| Video VAE | MiniMax H3 video VAE INT8 ConvRot |
| Audio VAE | MiniMax H3 audio VAE FP32 |
| LoRA | MiniMax H3 FL2V Turbo 4-step v1.0 768p, strength 1.0 |
| Sampling | Euler, simple scheduler, 4 steps, denoise 1.0 |
| Chunking | RoPE 8192; MLP 4096; QKV 4096; automatic QKV/MLP weight strategy; prefetch removed |
| Attention | Comfy Kitchen INT8 |
| Output | RTX Video Super Resolution 2× Ultra; NVENC H.264 at 35Mbps |

## Timing

| Attention path | Observed sampling time |
|---|---:|
| KJNodes custom SM75 SageAttention 2 path | **190.50 seconds/step** |
| Comfy Kitchen INT8 | **119.50 seconds/step** |
| Star7 SLA SM75 QK INT8 / PV FP16 + audio guard | **96.68 seconds/step** |
| Star7 SLA SM75 All-INT8 + audio guard | **60.83 seconds/step** |

Compared with the observed [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) custom SM75 SageAttention 2 path, CK INT8 reduced step time by approximately `50s`, or `29.4%`, and raised step throughput by approximately `1.42×`. This is a Turing/SM75 result for the recorded local software stack, not a claim that CK is faster than Sage on every GPU architecture.

The current bundled SM75 kernel was validated in the attached 1.0MP, 10-second,
four-step workflow. Its real packed sequence was 75,872 tokens with 56 heads.
Video queries used 85.08% dynamic block sparsity; eight target-audio query blocks
used full attention in the same native kernel, giving 83.93% overall effective
sparsity. The four steps took 97.23, 96.98, 95.88, and 96.88 seconds (96.68
seconds/step average), and the complete prompt finished in 471.12 seconds. Against
the separately recorded 119.50-second CK run, sampling throughput was
approximately 1.24×. No CK or Sage failure fallback was used.

The earlier 59.18-second result belonged to the first pure-INT8 PV kernel and was
removed because it had accumulated visual/audio error. ABI v7 reintroduces a
separate All-INT8 path while retaining per-16-row Q scaling, corrected
mapping, and the native dense audio-query guard. It completed the same workflow
at 60.83 seconds/step and 325.51 seconds total. The 8.5–10.1 second tail measured
-72.49/-72.57 dB RMS across the two channels, without the old right-channel burst.
One successful seed cannot establish general quality equivalence to FP16-PV,
so the mode still requires prompt-level visual and speech validation.

### SM75 attention microbenchmark

The development benchmark includes SLA routing, Q/K/V quantization, the sparse
attention core, and output allocation. Medians on the same RTX 2080 Ti were:

| Sequence length | FP16-PV | All-INT8 | CK INT8 |
|---:|---:|---:|---:|
| 4,096 | 4.735 ms | 2.627 ms | 7.407 ms |
| 16,384 | 56.303 ms | 17.158 ms | 80.859 ms |
| 75,872 | 944.200 ms | 289.938 ms | 1,535.828 ms |

Both modes use INT8 Q/K and FP32 online-softmax state/accumulation. FP16-PV keeps
probability/value multiplication in FP16; All-INT8 quantizes V per channel and
softmax probabilities for U8×S8 tensor-core PV. The table includes routing and
Q/K/V quantization but no audio guard; the full-workflow results above are
authoritative for the protected H3 path.

The four denoising steps of the 1.0MP/10-second case account for approximately 480 seconds. The currently confirmed end-to-end results are:

| Case | Confirmed complete task time |
|---|---:|
| 1.0MP, 10 seconds, 4-step Turbo LoRA | **620.32 seconds** |
| 0.6MP, 15 seconds, 4-step Turbo LoRA | approximately 530 seconds |
| 0.4MP, 10 seconds, 4-step Turbo LoRA | approximately 180 seconds |
| 0.4MP, 5 seconds, 4-step Turbo LoRA | approximately 85 seconds |

These totals include the current workflow's setup, decoding, super-resolution, audio work, and video output. Post-processing is still being tuned. Because the four cases use different spatial and temporal budgets, their total times are reported as observations rather than a linear speed comparison.

## Scaling model

Let `S` be the packed H3 sequence length. A useful engineering approximation is:

```text
S ≈ S_condition + k × spatial_tokens × temporal_tokens
T ≈ T_fixed + aS + bS² + T_post
```

RoPE, normalization, projections, and MLP work are primarily linear in `S`. Full self-attention includes QK/AV work that is approximately quadratic in `S`; memory-efficient kernels reduce residency but do not remove all attention arithmetic. VAE, super-resolution, and encoding add a separate component that is approximately proportional to pixels times frames plus fixed overhead.

The A/C spatial-duration budget ratio is approximately `2.5×`, while the observed end-to-end time ratio is `620.32/180 ≈ 3.45×`. The C/D budget ratio is approximately `2×`, while the time ratio is `180/85 ≈ 2.12×`. These observations are consistent with a mixture of fixed, linear, quadratic-attention, and post-processing costs rather than one constant linear multiplier.

Activation chunking does not reduce theoretical FLOPs. If a workload already fits dedicated VRAM without allocator pressure or host-memory migration, chunking alone should not be expected to improve speed and can add small-GEMM/kernel-launch overhead. The recorded 190.50-to-119.50 seconds/step improvement came primarily from selecting CK INT8 instead of the custom SM75 SageAttention 2 path.

## VRAM tuning decision guide

| Failure point | Primary control |
|---|---|
| eager split-half RoPE | reduce `chunk_tokens` |
| `fc1`/SwiGLU/`fc2` activation | reduce `mlp_chunk_tokens` |
| QKV projection temporary tensors | reduce `qkv_chunk_tokens` |
| attention kernel | change the attention backend or reduce sequence/reference tokens; activation chunks do not shrink the attention core |
| model loading | loader, quantization, or offload policy; chunk values do not apply yet |

| Dedicated VRAM | RoPE start | MLP start | QKV start |
|---:|---:|---:|---:|
| 20–24GB | 8192 | 4096 | 4096 |
| 16–20GB | 8192 | 2048–4096 | 2048–4096 |
| 12–16GB | 8192 | 1024–2048 | 1024–2048 |
| below 12GB | 4096–8192 | 512–1024 | 512–1024 |

These are initial tuning ranges rather than capacity guarantees. QKV follows the
configured value on every architecture and only decreases after an actual QKV
projection OOM when automatic fallback is enabled. Next-block prefetch has been
removed and is always disabled, so it is no longer a tuning control.

## Memory observation

The captured single-reference run showed approximately `16.7 / 22.0GB` dedicated GPU memory, about `13.0 / 15.8GB` shared GPU memory mapped by Windows, approximately 95% GPU utilization, and near-zero Copy-engine activity at the instant of capture.

![RTX 2080 Ti runtime memory](docs/assets/rtx2080ti-runtime-memory.png)

A separate no-reference text-to-video run was reported at approximately 15.6GB dedicated VRAM. These values are snapshots, not peak-memory traces. The data supports “no sustained copy-engine bottleneck was observed”; it does not support a claim that Windows never mapped or accessed shared memory.

![Launcher memory settings](docs/assets/launcher-memory-settings.png)

The “keep models resident in VRAM” option reduces model swapping. It does not disable WDDM shared-memory mapping and should not be described as a zero-shared-memory guarantee.

## Node settings

![RTX 20-series node settings](docs/assets/node-settings-rtx20.png)

```text
chunk_tokens = 8192
mlp_chunk_tokens = 4096
qkv_chunk_tokens = 4096
auto_halve_on_oom = true
disable_dynamic_prefetch = "实验功能已移除"  # Legacy compatibility field; runtime always disabled
reuse_mlp_weights = true  # auto-detects resident vs streamed safely
attention_backend = sla_sm75_qk_int8_pv_fp16  # strict SM75 SLA quality path
```

## Fixed-token-budget duration estimate

For rough planning only, if `r` is the spatial pixel/token budget relative to the 1.0MP case, the recorded approximation was:

```text
T_r ≈ 10 × (7168 + 80) / (7168r + 80)
```

| Relative spatial budget `r` | Estimated equivalent duration | Relative to 1.0MP/10s |
|---:|---:|---:|
| 1.0 | 10.0s | 1.00× |
| 0.9 | 11.1s | 1.11× |
| 0.8 | 12.5s | 1.25× |
| 0.7 | 14.2s | 1.42× |
| 0.6 | 16.5s | 1.65× |
| 0.5 | 19.8s | 1.98× |
| 0.4 | 24.6s | 2.46× |
| 0.3 | 32.5s | 3.25× |
| 0.25 | 38.7s | 3.87× |
| 0.2 | 47.9s | 4.79× |
| 0.15 | 62.7s | 6.27× |
| 0.1 | 91.0s | 9.10× |

![Fixed token-budget duration estimate](docs/assets/token-budget-duration-estimate.png)

This table estimates sequence-budget equivalence. It does not guarantee that the model, VAE, conditioning implementation, or available VRAM supports the listed duration, and it does not predict wall-clock time linearly.

## Interpretation limits

- Tests were local observations rather than a multi-run statistical benchmark.
- The 22GB board is not representative of a stock 11GB RTX 2080 Ti.
- CK INT8 and Sage are approximate attention paths and can produce different pixels.
- Chunking reduces peak activation residency; it does not reduce theoretical GEMM FLOPs.
- Performance comparisons require identical model, LoRA, seed, resolution, frame count, steps, backend, and offload state.
