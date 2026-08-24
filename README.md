# MiniMax H3 Activation Chunk (RoPE + MLP) - Star7

[中文说明](#中文说明) · [Benchmark](BENCHMARKS.md) · [Example workflows](examples/workflows)

Run high-resolution, long-duration MiniMax H3 videos efficiently on GPUs with limited VRAM: this ComfyUI node chunks the two largest RoPE and MLP activation peaks so those operations fit in dedicated VRAM instead of spilling into much slower shared system memory. For workloads that would otherwise OOM or page through shared memory, this can greatly improve the practical video size and runtime; when a workload already fits entirely in VRAM, chunking alone is not a speedup. The default path uses Comfy Kitchen INT8 attention and does not change the sampler, latent, VAE, video duration, or spatial resolution; choose `existing` to preserve an upstream Sage or environment-selected attention backend.

**New in v2.8.0:** the optional `Reference Video Optimize - Star7` node caps reference-video pixel area to the requested output canvas before H3 Video VAE encoding. It preserves every frame, the reference aspect ratio, and audio, while reducing both conditioning time and the reference tokens later processed by attention. On the local RTX 2080 Ti 22GB 0.6MP/9s reference case, total one-step diagnostics fell from 311.60s to 232.50s and H3 sequence length from 103,546 to 87,101. The implementation is pure PyTorch and is not restricted to SM75.

> This is an independent community project. MiniMax, ComfyUI, Comfy Kitchen, KJNodes, and NVIDIA are trademarks or projects of their respective owners.

## 中文说明

让高画质、长时长 MiniMax H3 视频在有限显存的显卡上高效运行：本节点把最容易爆显存的 RoPE 与 MLP 激活按 token 分块，使这两段关键计算适配专用显存，避免溢出到速度远低于显存的共享系统内存。对于原本会 OOM 或发生共享显存换页的任务，这能显著提升可运行规模与实际生成效率；如果任务本来就能完整装入显存，分块本身不会凭空加速。

**v2.8.0 新增可选的“MiniMax H3 参考视频优化 - Star7”前置节点。** 它会在 H3 Video VAE 编码前，把参考视频像素面积限制到目标输出画布；不删帧、不改音频，并保持参考画面比例。这样既缩短参考编码，也减少后续注意力需要处理的参考 token。RTX 2080 Ti 22GB 的本机 0.6MP/9 秒参考案例中，一步诊断总耗时由 311.60 秒降至 232.50 秒，H3 序列由 103,546 降至 87,101。该节点只使用通用 PyTorch 图像缩放，适用于 RTX 20–50 系，不会向新架构注入 SM75 专用计算。

模型的视频/音频输出会在采样阶段统一检查；若仍有 NaN/Inf，会明确报告损坏的是哪一路，不会等 VAE 解码和 FFmpeg 合成后才报错，也不会把无效样本静默替换为 0。SM75 SLA 与 CK 均不再要求外接 FP16 修复节点。SLA 建议配合
[MiniMax H3 Turbo SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA)
使用。SM75 使用随节点分发的预编译 CUDA 内核，SM80+ 使用 Triton。

节点针对 MiniMax H3 长序列推理的两个显存峰值：

- `RMSNorm -> split-half RoPE (Q/K)`；
- `fc1 -> SwiGLU -> fc2` MLP 扩展激活。

RoPE 沿 sequence/token 维分块并原位写回 Q/K。MLP 按 token 分块计算，避免同时保留完整的扩展激活；MLP 输出仍交给上游 block 处理，以保留 FP16 Exact、Sage、低显存 attention 和第三方模型补丁的兼容性。对于 `int8_tensorwise + ConvRot` 权重，节点保持 `QuantizedTensor` 路径，并允许各 token chunk 复用已经准备好的权重。

### 注意力模式

| `attention_backend` | 架构 | 计算路径 |
|---|---|---|
| `existing` | 通用 | 保留上游 Sage、原生或第三方注意力 |
| `comfy_kitchen_int8` | 由 CK 决定 | Comfy Kitchen INT8 |
| `sla_sm75_qk_int8_pv_fp16` | SM75 | QK INT8、PV FP16；推荐模式 |
| `sla_sm75_all_int8_experimental` | SM75 | QK INT8、PV INT8；实验模式 |
| `sla_sm80+_qk_int8_pv_fp16` | SM80+ | QK INT8、PV FP16；Triton |

SLA 按 LightX2V 契约使用 `Q=128`、`K=64` 动态块路由，视频查询约保留 15% 的 K 块；
softmax 状态和最终累积保持 FP32。目标音频查询使用完整注意力保护。

注意事项：

- SM75 Windows x64：预编译 CUDA 13 静态运行时内核，要求支持 CUDA 13 的 580+ NVIDIA 驱动。
- SM75 Linux x86_64：预编译 CUDA 12.6 静态运行时内核，兼容 Ubuntu 20.04 / glibc 2.31 及更新系统，要求 NVIDIA 驱动 525.60.13+；不依赖 PyTorch C++ ABI 或 SageAttention。若 Turing Triton 不可用，路由与量化会自动改用有界显存 PyTorch 预处理，SLA 核心仍由 `.so` 执行。
- SM75 SLA 自动使用 FP32 残差与 SwiGLU、FP16 分支计算及 `out_proj/fc2` 二次幂防溢出；外部 Native FP16 Loader 可省略。SM80+ 不启用这条 SM75 专用修复。
- SM80+ 使用 Triton，首次运行会编译并缓存内核。
- SLA 不会静默回退；环境、自检或计算失败会直接中止，需手动改选 CK 或 `existing`。
- SM75 SLA/CK 自动采用 FP16 分支、FP32 残差/SwiGLU，以及幂次缩放保护的 `out_proj`/`fc2`，无需外接 FP16 修复节点；该精度改写严格限定于 SM75，不会作用于 SM80/SM86/SM89/SM120。
- NaN/Inf 检查是上述修复之后的最后防线，不是用报错代替修复。严格 SLA 会在每个完整 Transformer block 后检查并报告首个故障 block；所有注意力模式还会在 H3 的视频/音频模型输出处统一检查。只有保护后仍产生异常才会停止，避免输出棋格闪烁或把无效音频拖到 FFmpeg 合成时才报错。
- SM75 使用 CK 时也会自动采用 FP16 分支、FP32 残差/SwiGLU 和幂次缩放保护的 `out_proj`/`fc2`。外部 TE-Speed 等 block-loop 缓存可以接在本节点之前；其完整步与缓存前缀仍会经过 Star7 block/attention 补丁。
- All-INT8 量化误差高于 FP16-PV，因此保留为实验选项。

## 安装

ComfyUI Manager / Comfy Registry：

```text
搜索：MiniMax H3 Activation Chunk - Star7
包名：minimax-h3-chunk-star7
```

Comfy CLI：

```bash
comfy node install minimax-h3-chunk-star7
```

也可以从 GitHub 手动安装：

```text
ComfyUI/
└─ custom_nodes/
   └─ minimax-h3-chunk-star7/
      ├─ __init__.py
      ├─ nodes.py
      ├─ sla_backend.py
      ├─ sm75_backend.py
      ├─ csrc/sla_sm75_sparse.cu
      ├─ csrc/third_party/comfy_kitchen_sage/
      ├─ bin/win_amd64/star7_sla_sm75_v7.dll
      └─ bin/linux_x86_64/star7_sla_sm75_v7.so
```

或者：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-chunk-star7.git
```

重启 ComfyUI，搜索：

```text
MiniMax H3 Activation Chunk (RoPE + MLP) - Star7
```

节点界面会跟随 ComfyUI 语言：中文环境自动显示中文标题、参数名、提示和运行状态，
其他语言环境显示英文。汉化只改变界面文字，不改变工作流保存的参数名或实际值；旧工作流
以及 `existing`、`comfy_kitchen_int8` 旧注意力选项均保持兼容；新版本增加
两个架构明确的新名称；开发阶段的旧 SLA 值不再读取。

## 推荐连接顺序

通用链路：

```text
UNET Loader -> LoRA -> Attention patch (optional) -> Activation Chunk - Star7
            -> Guider / Scheduler / Sampler
```

带参考视频时，增加一条前置媒体链路：

```text
Load Video (frames) -> Reference Video Optimize - Star7 -> H3 Conditioning ref_video
Resolution Selector width/height -------------------------> optimizer target width/height
Load Video (audio) ---------------------------------------> H3 Conditioning ref_audio
```

`match_output_area` 是推荐默认值：仅在参考视频像素面积高于目标画布时缩小，保持纵横比且不放大低分辨率素材。音频仍从视频加载节点直接连接到 Conditioning，本节点不修改音频。

RTX 20 系及其他不适合原生 BF16 计算的显卡：

```text
MiniMax H3 Native FP16 Loader - Star7 -> LoRA
    -> Activation Chunk - Star7 -> Guider / Scheduler / Sampler
```

以上外部 Loader 仍适用于 CK 或 `existing` 注意力；选择任一 SM75 SLA 模式时，
本节点会自动安装同一套 FP16 Exact 防溢出公式，可以直接使用普通 H3 Loader。

20 系示例依赖另一个项目：

- [MiniMax H3 Native FP16 Loader - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7)

Native FP16 Loader 已包含精确防溢出处理，不要再串接旧的后置 `FP16 Exact Fix` 节点。

## 参数

| 参数 | 作用 | RTX 2080 Ti 22GB 实测值 |
|---|---|---:|
| `chunk_tokens` | RoPE 的目标 token 分块上限；RoPE 工作集相对较小，优先保持较大值 | `8192` |
| `mlp_chunk_tokens` | MLP 的目标 token 分块上限；节点下方会显示本次实际生效值 | `4096` |
| `qkv_chunk_tokens` | QKV 投影临时工作集；SLA 直接写入 FP16 后端布局，`0` 表示整段投影但不改变注意力后端 | `4096` |
| `auto_halve_on_oom` | 当前 chunk OOM 时自动减半重试 | `true` |
| `提前加载下一层（实验功能已移除）` | 仅为兼容旧工作流保留，不再参与计算，始终关闭 | 兼容字段 |
| `reuse_mlp_weights` | 自动策略：将已准备权重复制到独立快照后复用；无法快照或 OOM 时改用 streamed | `true` |
| `attention_backend` | 保留上游后端、采用 CK INT8，或选择严格 SLA | `comfy_kitchen_int8` |
| `verbose` | 输出首个同形状 block 的紧凑诊断 | `true` |

为兼容旧工作流，节点保留内部字段名 `disable_dynamic_prefetch`，但该字段现在仅作为显示占位，
不再安装动态预取 wrapper，实际运行始终关闭预取。这样旧工作流不会因字段位置变化而错位，
同时避免预取在慢速显存/PCIe 路径上增加等待或引发 CUDA 同步问题。

参数越大不等于必然更快。较大的 chunk 减少 kernel 启动次数，但会增加瞬时激活和权重预取竞争。比较参数时必须固定模型、seed、分辨率、帧数、步数和注意力后端。

节点会在三个数值输入的正下方分别显示 `RoPE 当前使用`、`MLP 当前使用` 和 `QKV 当前使用`。正常时会显示
`当前使用 N（设定值）`；发生显存不足后会显示 `已降级为 N（设定 M）`。这里的 `M` 是工作流
里的目标上限，`N` 是本次模型会话实际采用的上限。输入框本身不会被偷偷改写，也不会把临时
降级值保存进工作流。

如果当前 packed sequence 本身短于设定值，状态会显示
`当前使用 N（设定 M，受序列长度限制）`。这是本次 forward 的真实 token 上限，不是 OOM
降级，也不会把较短序列的值记忆到后续较长序列。

`auto_halve_on_oom=true` 时，RoPE、MLP 或 QKV 当前分块 OOM 会按当前值整数减半，最低到 `256`。
找到可用值后，后续 H3 block 和同一次模型会话中的后续 forward 会直接沿用该值，避免每个
block 反复以失败的大块重试。用户手动修改任一分块输入并重新执行节点后，记忆值会用新的
设定值重置；例如旧值曾自动降到 `2048`，手动改成 `3072` 后不会继续沿用 `2048`。

### 优先调整 MLP，不要先动 RoPE

本机真实 ConvRot 权重的单 MLP 工作集测试显示：`MLP 8192` 约 `1970 MiB`、`4096` 约
`1228 MiB`、`2048` 约 `858 MiB`、`1024` 约 `672 MiB`。同条件 RoPE 约为：`8192`
`820 MiB`、`4096` `638 MiB`、`2048` `550 MiB`、`1024` `501 MiB`。因此显存紧张时，
先降低 `mlp_chunk_tokens` 通常更有效；RoPE 从 `8192` 降低到 `4096` 的收益相对有限，
除非日志明确指出 RoPE OOM，否则建议保持 `8192`。

### 按 OOM 位置调整，而不是盲目同时降低两个值

| 报错位置或现象 | 应优先调整 | 建议动作 |
|---|---|---|
| `fc1` / SwiGLU / `fc2` MLP 激活 OOM | `mlp_chunk_tokens` | `4096 -> 2048 -> 1024 -> 512` |
| `rms_rope_split_half_` / `apply_rope_split_half1` | `chunk_tokens` | 只有明确 RoPE OOM 时才 `8192 -> 4096 -> 2048` |
| 旧工作流含下一 block 预取字段 | `提前加载下一层（实验功能已移除）` | 无需处理，字段仅作兼容占位，运行时始终关闭 |
| QKV 投影临时张量 OOM | `qkv_chunk_tokens` | `4096 -> 2048 -> 1024`；只降低投影峰值，完整 Q/K/V 仍需驻留 |
| attention kernel OOM | 注意力后端 | 改用 CK INT8，或保留上游 Low VRAM/Sage；激活分块不能降低注意力核心工作集 |
| 模型加载阶段已经 OOM | 加载器/量化/卸载策略 | 分块节点尚未执行，调 chunk 无效 |

显存档位只能作为起点，因为同容量显卡还会受到模型格式、LoRA 反量化、驱动和常驻策略影响：

| 专用显存 | RoPE 起点 | MLP 起点 | QKV 起点 |
|---:|---:|---:|---:|
| 20–24GB | `8192` | `4096–8192` | `4096–8192` |
| 16–20GB | `8192` | `2048–4096` | `2048–4096` |
| 12–16GB | `8192` | `1024–2048` | `1024–2048` |
| 12GB 以下 | `4096–8192` | `512–1024` | `512–1024` |

22GB 卡处理参考视频长序列时，MLP/QKV 建议从 `8192` 开始。它们只调节临时激活，不决定模型权重驻留；设为 `0` 是整段计算，不是“自动值”。本机 `S=87,101` 热态测试中，`8192` 为 77.46 秒/步；`59904` 因显存申请失败自动降档，反而为 86.92 秒/步。

保持 `auto_halve_on_oom=true` 可以让 RoPE chunk 自动降档，但 MLP、attention 或模型加载 OOM 仍需根据 traceback 手动选择正确参数。

## RTX 2080 Ti 22GB 实测案例

测试卡是 **22GB 显存改装版 RTX 2080 Ti**，不是标准11GB版本。

| 项目 | 配置 |
|---|---|
| 任务 | MiniMax H3 单参考图模式，连接1张参考图 |
| 实测 A | 1.0MP，10秒，24fps，约243帧 |
| 实测 B | 0.6MP，15秒，24fps，约362帧 |
| 实测 C | 0.4MP，10秒，24fps，约243帧 |
| 实测 D | 0.4MP，5秒，24fps，约124帧 |
| 主模型 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| LoRA | 768p Turbo 4-step v1.0，强度1.0 |
| 采样 | Euler / simple / 4 steps / denoise 1.0 |
| 分块 | RoPE `8192`，MLP `4096` |
| 注意力 | `comfy_kitchen_int8` |
| 后处理 | RTX Video Super Resolution 2× Ultra，NVENC H.264 |

本机观察值（1.0MP / 10秒 / 4-step LoRA）：

| 注意力路径 | 采样耗时 | 完整任务 | 相对 CK 单步吞吐 |
|---|---:|---:|---:|
| KJNodes 自定义 SM75 SageAttention 2 | 约170秒/步 | — | 约0.71× |
| KJNodes Low VRAM Attention (`head_chunks=4`) | 约177秒/步 | — | 约0.68× |
| Comfy Kitchen INT8 | 约120秒/步 | 约620秒 | 基准 |
| SLA SM75 QK-INT8/PV-FP16 | **96.68秒/步** | **约470秒** | **约1.24×** |
| SLA SM75 All-INT8 实验模式 | **60.83秒/步** | **约325秒** | **约1.97×** |

建议 SLA 配合 [MiniMax H3 Turbo SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) 使用。
完整任务包含模型调度、VAE 解码、超分、音频和视频封装；结果仅代表本机配置。

其他 CK INT8 完整任务：

| 空间/时长 | 完整任务实测 |
|---|---:|
| 0.6MP / 15秒 / 4-step LoRA | 约530秒 |
| 0.4MP / 10秒 / 4-step LoRA | 约180秒 |
| 0.4MP / 5秒 / 4-step LoRA | 约85秒 |

### 耗时与显存原理

令 `S` 为 H3 实际 packed sequence token 数。固定模型结构下，可以粗略写成：

```text
S ≈ S_condition + k × spatial_tokens × temporal_tokens
T ≈ T_fixed + aS + bS² + T_post
```

- RoPE、RMSNorm、投影和 MLP 主要随 `S` 线性增长；全局注意力含近似 `S²` 项。
- RoPE/MLP 分块降低激活峰值，不减少理论 FLOPs；无需分块时可能增加少量 kernel 启动开销。
- SLA 通过减少参与 QK/PV 的 K 块降低注意力计算量。
- 完整任务还包含 VAE、超分、音频和编码，因此不会与采样步耗时线性对应。

完整记录和截图见 [BENCHMARKS.md](BENCHMARKS.md)。以上是本机观察案例，不是跨平台性能保证。

## 30 系、40 系及可用 Sage 的显卡

可以保留自己的 Sage 节点。连接顺序：

```text
Loader -> LoRA -> Sage Attention Patch -> Activation Chunk - Star7
```

并把本节点的：

```text
attention_backend = existing
```

这样本节点保留上游 Sage，只负责 RoPE/MLP 激活分块。若前面没有注意力节点，`existing` 就使用当前 ComfyUI 环境为该模型选择的原生后端；它不会自动切换成 CK 或 XFormers。

若选择 `comfy_kitchen_int8`，本节点会覆盖更早安装的 Sage、XFormers 或其他 MiniMax H3 注意力 patch，只对本节点输出的模型生效。CK 组件不可用时会记录警告并自动保留原有后端，不会因导入失败中断工作流。50 系显卡同样可以使用 CK，但最终速度和显存表现取决于已安装的 Comfy Kitchen/CUDA 支持；`existing` 则交给当前环境的原生后端选择。

## 小显存补充说明

`reuse_mlp_weights=true` 会在权重静态且显存允许时使用独立 resident 快照；不满足条件时自动切换 streamed-safe 路径。分块只降低 RoPE/MLP 激活峰值，模型权重、LoRA、注意力工作集和 VAE 仍需占用显存。

## 示例工作流

- [通用工作流](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json)：普通 `UNETLoader`，适合自行选择原生、Sage或CK注意力的环境。
- [RTX 20系工作流](examples/workflows/MiniMax-H3-Activation-Chunk-RTX20-Star7.json)：使用 Native FP16 Loader，并默认采用 CK INT8。

两份工作流的参考条件由 [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) 提供，并保留一个 `ref_image_0` 接口。本次 A/B/C/D 案例均采用该 T8 参考节点的单参考图模式。提示词输入已改用 ComfyUI 自带的 `Text (Multiline)`，不再要求安装 ComfyUI-Jjk-Nodes。仓库不附带可能存在版权或隐私问题的原始参考素材；导入后请把 `replace-with-your-reference-image.png` 替换为自己的图片。为保证导入后可以直接生成，发布版已移除 NVIDIA RTX Video Super Resolution 节点，VAE 解码结果直接交给 VideoHelperSuite 封装；需要超分的用户可自行在解码后添加。

## 数值与兼容性

- RoPE eager 分块路径可以逐元素一致；
- MLP 不改变公式、权重、dtype或token顺序，但大 GEMM 拆成小 GEMM 后，底层归约顺序可能产生约 `1.19e-7` 的 float32 末位差，因此是数值等价而非 bitwise identical；
- `comfy_kitchen_int8` 会量化注意力计算，是显式的近似模式；
- 只有检测到 MiniMax H3 典型 Q/K shape 和匹配的 RoPE frequency 序列维时才安装补丁；其他 shape 回退到原实现。

## 日志

节点只对首个同形状 block 输出紧凑信息，例如：

```text
[Star7 H3 Chunk] v2.2.8 active | blocks=... | fp16=...
[Star7 H3 Chunk] Attention profile (one block) | ...
[Star7 H3 Chunk] MLP active | ... | chunk-expansion=...
[Star7 H3 Chunk] MLP profile (one block) | chunks=... | weights=resident-snapshot
```

`weights=resident-*` 表示独立权重快照已启用；`weights=streamed` 表示快照失败、OOM 或手动关闭后使用安全流式路径。

## License

[MIT](LICENSE)
