# MiniMax H3 Activation Chunk & Attention Acceleration - Star7

[中文说明](#中文说明) · [Benchmark](BENCHMARKS.md) · [Example workflows](examples/workflows)

This ComfyUI project helps MiniMax H3 run high-quality, long-duration video generation on GPUs with limited VRAM. It provides independent QKV/RoPE/MLP activation chunking and selectable attention acceleration while preserving the original sampling process, latent layout, VAE, duration, and output resolution. Attention can use Comfy Kitchen INT8, preserve an existing upstream backend, or select an architecture-specific SLA mode.

An optional reference-video loader can limit conditioning resolution before H3 Video VAE encoding.

> This is an independent community project. MiniMax, ComfyUI, Comfy Kitchen, KJNodes, and NVIDIA are trademarks or projects of their respective owners.

## 中文说明

让高画质、长时长MiniMax H3视频在有限显存的显卡上高效运行。

本节点用于 MiniMax H3 的显存分块与可选注意力加速：QKV、RoPE、MLP 可分别分块；注意力可选择 Comfy Kitchen INT8、保留上游已有后端，或使用对应架构的 SLA 模式。采样器、latent、VAE、时长和输出分辨率均不改变。

SLA 是动态稀疏注意力，通过块级路由只计算选中的 K 块，减少长序列中的注意力计算量，因此中段采样通常更快。当前已经支持 SM75 的 CK + SLA 混合注意力：采样前段和后段使用 Comfy Kitchen INT8，中段使用 SM75 All-INT8 SLA，在速度与画质之间取得更好的平衡；该模式仍属于实验功能，实际收益取决于显卡、分辨率、帧数和采样步数。

SLA 建议配合 [MiniMax H3 Turbo SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) 使用；另附参考视频载入节点，可在 H3 Video VAE 编码前限制参考分辨率。

对于不支持原生 BF16 的 RTX 20 系等显卡，请先使用独立的
[MiniMax H3 Native FP16 Loader - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7)，
再连接本分块节点；支持原生 BF16 的显卡通常只需使用本分块节点。

节点控制 MiniMax H3 长序列推理的三个临时显存峰值：

- `QKV projection`；
- `RMSNorm -> split-half RoPE (Q/K)`；
- `fc1 -> SwiGLU -> fc2` MLP 扩展激活。

三项分块只改变行独立投影/激活的临时工作集，不裁帧、不减少注意力 token。RoPE 原位写回 Q/K，MLP 输出仍交给原 block；`int8_tensorwise + ConvRot`、FP16 Exact、LoRA 与第三方 block 补丁保持兼容。分块主要用于避免 OOM 或共享显存换页，任务原本完全驻留显存时不保证提速。

### 注意力模式

| `attention_backend` | 架构 | 计算路径 |
|---|---|---|
| `existing` | 通用 | 保留上游 Sage、原生或第三方注意力 |
| `comfy_kitchen_int8` | 由 CK 决定 | Comfy Kitchen INT8 |
| `sla_sm75_qk_int8_pv_fp16` | SM75 | QK INT8、PV FP16；推荐模式 |
| `sla_sm75_all_int8_experimental` | SM75 | QK INT8、PV INT8；实验模式 |
| `hybrid_sm75_ck_sla_all_int8` | SM75 | 采样步级 CK / SM75 All-INT8 SLA / CK；实验模式 |
| `sla_sm80+_qk_int8_pv_fp16` | SM80+ | QK INT8、PV FP16；Triton |

SLA 按 LightX2V 契约使用 `Q=128`、`K=64` 动态块路由，视频查询约保留 15% 的 K 块；
softmax 状态和最终累积保持 FP32。SM75 原生内核对目标音频查询执行完整注意力；
SM80+ 当前仅使用音频优先路由，日志明确显示 `routing-priority-only`。

注意事项：

- SM75 Windows x64：预编译 CUDA 13 静态运行时内核，要求支持 CUDA 13 的 580+ NVIDIA 驱动。
- SM75 Linux x86_64：预编译 CUDA 12.6 静态运行时内核，兼容 Ubuntu 20.04 / glibc 2.31 及更新系统，要求 NVIDIA 驱动 525.60.13+；不依赖 PyTorch C++ ABI 或 SageAttention。若 Turing Triton 不可用，路由与量化会自动改用有界显存 PyTorch 预处理，SLA 核心仍由 `.so` 执行。
- Chunk 不注入 FP16 Exact 或改变模型 compute dtype。SM75 未检测到独立 FP16 Exact 节点时只提示并继续运行；SM80+ 无需该修复且不会显示提示。
- SM80+ 使用 Triton，首次运行会编译并缓存内核。
- SM100/SM120 继续使用 SM80+ Triton 路径，但启动日志会明确标记为需要对应实机验证的新架构。
- SLA 不会静默回退；环境、自检或计算失败会直接中止，需手动改选 CK 或 `existing`。
- NaN/Inf 检查只负责检测，不替代 FP16 修复。严格 SLA 会在每个完整 Transformer block 后检查并报告首个故障 block；下次运行只对该 block 启用 QKV、SLA、`out_proj` 和 MLP 分段诊断。所有注意力模式还会在 H3 的视频/音频模型输出处统一检查。
- 外部 TE-Speed 等 block-loop 缓存可以接在本节点之前；完整步与缓存前缀仍会经过 Star7 block/attention 补丁。
- All-INT8 量化误差高于 FP16-PV，因此保留为实验选项。
- `hybrid_sm75_ck_sla_all_int8` 是 SM75 专用采样步级调度：默认前约 `1/6` 和后约
  `1/6` 使用现有 CK，中间约 `2/3` 使用现有 SM75 All-INT8 SLA。SM80+ 尚未适配，
  选择该模式会直接报不支持，不会自动改用 SM80+ SLA。它是在完整采样 step 之间
  切换 backend，不是在一次 Attention 内混合两个 kernel；Hybrid 的 SLA step 仍保持
  严格 SLA 语义，失败不会偷偷切回 CK。中间
  SLA 区域调用现有的 SM75 All-INT8 实验内核。All-INT8 只改变中间区域，仍可能带来
  比 FP16-PV 更大的近似误差，不代表已完成画质验证。

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
MiniMax H3 Activation Chunk - Star7
```

节点界面会跟随 ComfyUI 语言：中文环境自动显示中文标题、参数名、提示和运行状态，
其他语言环境显示英文。汉化只改变界面文字，不改变工作流保存的参数名或实际值；旧工作流
以及 `existing`、`comfy_kitchen_int8` 旧注意力选项均保持兼容；新版本增加
架构/精度明确的 SLA 名称和可选的 Hybrid 名称；开发阶段的旧 SLA 值不再读取。

## 推荐连接顺序

通用链路：

```text
UNET Loader -> LoRA -> Attention patch (optional) -> Activation Chunk - Star7
            -> Guider / Scheduler / Sampler
```

带参考视频时，增加一条前置媒体链路：

```text
Reference Video Load - Star7 (video) -> H3 Conditioning ref_video
Reference Video Load - Star7 (audio) -> H3 Conditioning ref_audio
```

语音/音色独立参考沿用常见工作流：音频接 `ref_audio_0`，提示词明确引用
`<Audio 1>`。只有确实需要把画面与声音打包成同一个联合参考块时，才将
`ref_video_0` 与 `ref_video_audio_0` 配对；它会改变条件布局，不是
`ref_audio_0` 的无差别替换。

`audio_mode=native` 会让 H3 根据参考重新生成声音，并不保证逐字复制原音轨；
要求最终视频保留原声时，应把载入节点的音频接到 `drive_audio`，选择
`audio_mode=lock_source`，最终合成使用 T8 的 `mux_audio`。

精简加载节点内部固定输出 H3 所需的 24fps，最长读取 15 秒并裁齐到 `17n+5`
帧网格。`最长边限制` 保持参考视频的横竖方向；`允许小视频放大` 默认关闭，
避免为插值画面增加参考 token。音频保留源文件最多 15 秒，不随 `17n+5`
画面裁齐而截断尾字；节点不依赖 VHS。

RTX 20 系及其他不适合原生 BF16 计算的显卡：

```text
MiniMax H3 Native FP16 Loader - Star7 -> LoRA
    -> Activation Chunk - Star7 -> Guider / Scheduler / Sampler
```

FP16 Exact 与 Chunk 是两个独立节点；SM75 建议按上图组合，SM80+ 可仅使用 Chunk。

20 系示例依赖另一个项目：

- [MiniMax H3 Native FP16 Loader - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7)

Native FP16 Loader 已包含精确防溢出处理，不要再串接旧的后置 `FP16 Exact Fix` 节点。

## 参数

| 参数 | 作用 | RTX 2080 Ti 22GB 实测值 |
|---|---|---:|
| `chunk_tokens` | RoPE 的目标 token 分块上限；RoPE 工作集相对较小，优先保持较大值 | `8192` |
| `mlp_chunk_tokens` | MLP 的目标 token 分块上限；节点下方会显示本次实际生效值 | `8192` |
| `qkv_chunk_tokens` | QKV 投影临时工作集；SM75 质量上限 `4096`，SM80+ 不限 | `4096`（SM75） |
| `auto_halve_on_oom` | 当前 chunk OOM 时自动减半重试 | `true` |
| `提前加载下一层（实验功能已移除）` | 仅为兼容旧工作流保留，不再参与计算，始终关闭 | 兼容字段 |
| `reuse_mlp_weights` | 在 QKV/MLP token 块间复用已准备权重；无法快照或 OOM 时改用 streamed | `true` |
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

SM75 上 `qkv_chunk_tokens=0` 或高于 `4096` 时，运行状态会明确显示
`QKV 质量保护：4096`。这是参考语音稳定上限，不是 OOM 降档；若 `4096`
仍发生 QKV OOM，自动降档仍可只把 QKV 继续降低。SM80、SM86、SM89、SM120
等更新架构不应用此限制。

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
| 20–24GB | `8192` | `4096–8192` | `4096`（SM75）/ `4096–8192`（SM80+） |
| 16–20GB | `8192` | `2048–4096` | `2048–4096` |
| 12–16GB | `8192` | `1024–2048` | `1024–2048` |
| 12GB 以下 | `4096–8192` | `512–1024` | `512–1024` |

22GB SM75 卡处理参考视频长序列时建议 MLP `8192`、QKV `4096`。它们只调节临时激活，不决定模型权重驻留。本机 `S=87,101` 热态测试中，MLP `8192` 为 77.46 秒/步；`59904` 因显存申请失败自动降档，反而为 86.92 秒/步。

保持 `auto_halve_on_oom=true` 会按实际失败位置只降低 RoPE、MLP 或 QKV 中的一项。完整 Q/K/V 缓冲、attention kernel 或模型加载本身的 OOM 不属于可缩小的局部分块，日志会明确指出，需减少参考 token/画布或更换注意力与加载策略。

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
| QKV | 原路径（该组历史数据采集时未启用 QKV 分块） |
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
| CK + SLA Hybrid（SM75，CK 2步 + SLA 2步） | — | **约463秒** | — |

建议 SLA 配合 [MiniMax H3 Turbo SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) 使用。
完整任务包含模型调度、VAE 解码、超分、音频和视频封装；结果仅代表本机配置。

参考视频预处理实测（0.6MP / 9秒，单步诊断）：

| 路径 | H3 sequence | 总耗时 |
|---|---:|---:|
| 原始参考尺寸 | `103,546` | `311.60秒` |
| 限制参考画布后 | `87,101` | `232.50秒` |

参考语音 QKV 回归实测（`S=103,546`，同 seed，4 steps，MLP `8192`）：

| QKV 路径 | 平均采样 | 解码音频 |
|---|---:|---|
| 旧版实际 `8192` | 约 `199.5秒/步` | 相对 `4096` 发生可测漂移 |
| 当前 SM75 质量保护 `4096` + 权重复用 | `196.27秒/步` | 与手动 `4096` PCM 逐位一致 |

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
- QKV/RoPE/MLP 分块降低临时激活峰值，不减少理论 FLOPs；无需分块时可能增加少量 kernel 启动开销。
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

这样本节点保留上游 Sage，只负责 QKV/RoPE/MLP 激活分块。若前面没有注意力节点，`existing` 就使用当前 ComfyUI 环境为该模型选择的原生后端；它不会自动切换成 CK 或 XFormers。

若选择 `comfy_kitchen_int8`，本节点会覆盖更早安装的 Sage、XFormers 或其他 MiniMax H3 注意力 patch，只对本节点输出的模型生效。CK 组件不可用时会记录警告并自动保留原有后端，不会因导入失败中断工作流。50 系显卡同样可以使用 CK，但最终速度和显存表现取决于已安装的 Comfy Kitchen/CUDA 支持；`existing` 则交给当前环境的原生后端选择。

## 小显存补充说明

`reuse_mlp_weights=true` 会在显存允许时让 SM75 QKV 与 MLP 使用独立 resident 快照，避免每个 token 块重复准备同一权重；不满足条件时自动切换 streamed-safe 路径。分块只降低 QKV/RoPE/MLP 临时激活峰值，模型权重、LoRA、注意力工作集和 VAE 仍需占用显存。

## 示例工作流

- [通用工作流](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json)：普通 `UNETLoader`，适合自行选择原生、Sage或CK注意力的环境。
- [RTX 20系工作流](examples/workflows/MiniMax-H3-Activation-Chunk-RTX20-Star7.json)：使用 Native FP16 Loader，并默认采用 CK INT8。

两份工作流的参考条件由 [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) 提供，并保留一个 `ref_image_0` 接口。本次 A/B/C/D 案例均采用该 T8 参考节点的单参考图模式。提示词输入已改用 ComfyUI 自带的 `Text (Multiline)`，不再要求安装 ComfyUI-Jjk-Nodes。仓库不附带可能存在版权或隐私问题的原始参考素材；导入后请把 `replace-with-your-reference-image.png` 替换为自己的图片。为保证导入后可以直接生成，发布版已移除 NVIDIA RTX Video Super Resolution 节点，VAE 解码结果直接交给 VideoHelperSuite 封装；需要超分的用户可自行在解码后添加。

## 数值与兼容性

- RoPE eager 分块路径可以逐元素一致；
- QKV 裸 INT8/ConvRot 投影与 RoPE（含不规则尾块）在本机分块逐位一致；但同 seed 的完整四步 H3 实测可稳定复现 `8192` 与 `4096` 的解码语音差异，说明差异发生在完整模型的 QKV 权重准备、调用与后续迭代组合路径，而不是已知的单个投影或 RoPE 算术错误。SM75 因此固定采用验证通过的 `4096` 质量上限；
- MLP 不改变公式、权重、dtype或token顺序，但大 GEMM 拆成小 GEMM 后可能出现 float32 末位差，因此是数值等价而非 bitwise identical；
- `comfy_kitchen_int8` 会量化注意力计算，是显式的近似模式；
- 只有检测到 MiniMax H3 典型 Q/K shape 和匹配的 RoPE frequency 序列维时才安装补丁；其他 shape 回退到原实现。

## 日志

节点只对首个同形状 block 输出紧凑信息，例如：

```text
[Star7 H3 Chunk] Ready v2.9.4 | ... | chunks(RoPE/MLP/QKV)=8192/8192/4096 | ...
[Star7 H3 Chunk] First-block QKV | ... | weights=resident-quantized | ...
[Star7 H3 Chunk] First-block MLP | ... | mode=upstream-preserved | ...
```

QKV 日志中的 `weights=resident-*` 表示独立权重快照已启用；`weights=streamed` 表示使用安全流式路径。
严格 SLA 若首次在 block N 失败，进程内下一次运行会自动只诊断该 block；远程复现可设置
`STAR7_SLA_DEBUG_BLOCK=N`。SM100/SM120 还可按需设置 `STAR7_SLA_LONG_SELF_TEST=1`
执行一次 `S=16206/H=1` 长序列内核检查，默认不运行。

## License

[MIT](LICENSE)
