# MiniMax H3 Activation Chunk (RoPE + MLP) - Star7

[中文说明](#中文说明) · [Benchmark](BENCHMARKS.md) · [Example workflows](examples/workflows)

Run high-resolution, long-duration MiniMax H3 videos efficiently on GPUs with limited VRAM: this ComfyUI node chunks the two largest RoPE and MLP activation peaks so those operations fit in dedicated VRAM instead of spilling into much slower shared system memory. For workloads that would otherwise OOM or page through shared memory, this can greatly improve the practical video size and runtime; when a workload already fits entirely in VRAM, chunking alone is not a speedup. The default path uses Comfy Kitchen INT8 attention and does not change the sampler, latent, VAE, video duration, or spatial resolution; choose `existing` to preserve an upstream Sage or environment-selected attention backend.

> This is an independent community project. MiniMax, ComfyUI, Comfy Kitchen, KJNodes, and NVIDIA are trademarks or projects of their respective owners.

## 中文说明

让高画质、长时长 MiniMax H3 视频在有限显存的显卡上高效运行：本节点把最容易爆显存的 RoPE 与 MLP 激活按 token 分块，使这两段关键计算适配专用显存，避免溢出到速度远低于显存的共享系统内存。对于原本会 OOM 或发生共享显存换页的任务，这能显著提升可运行规模与实际生成效率；如果任务本来就能完整装入显存，分块本身不会凭空加速。

节点针对 MiniMax H3 长序列推理的两个显存峰值：

- `RMSNorm -> split-half RoPE (Q/K)`；
- `fc1 -> SwiGLU -> fc2` MLP 扩展激活。

RoPE 沿 sequence/token 维分块并原位写回 Q/K。MLP 按 token 分块计算，避免同时保留完整的扩展激活；MLP 输出仍交给上游 block 处理，以保留 FP16 Exact、Sage、低显存 attention 和第三方模型补丁的兼容性。对于 `int8_tensorwise + ConvRot` 权重，节点保持 `QuantizedTensor` 路径，并允许各 token chunk 复用已经准备好的权重。

默认 `attention_backend = comfy_kitchen_int8` 时，本项目会使用 Comfy Kitchen INT8 注意力；它是近似注意力路径。若要完全保留前置节点或当前环境的注意力算法，请选择 `existing`。两种模式都不会改变采样器、sigma、seed、VAE、latent、帧数或画面分辨率。

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
      └─ nodes.py
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
以及 `existing`、`comfy_kitchen_int8` 两种注意力选项均保持兼容。

## 推荐连接顺序

通用链路：

```text
UNET Loader -> LoRA -> Attention patch (optional) -> Activation Chunk - Star7
            -> Guider / Scheduler / Sampler
```

RTX 20 系及其他不适合原生 BF16 计算的显卡：

```text
MiniMax H3 Native FP16 Loader - Star7 -> LoRA
    -> Activation Chunk - Star7 -> Guider / Scheduler / Sampler
```

20 系示例依赖另一个项目：

- [MiniMax H3 Native FP16 Loader - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7)

Native FP16 Loader 已包含精确防溢出处理，不要再串接旧的后置 `FP16 Exact Fix` 节点。

## 参数

| 参数 | 作用 | RTX 2080 Ti 22GB 实测值 |
|---|---|---:|
| `chunk_tokens` | RoPE 的目标 token 分块上限；RoPE 工作集相对较小，优先保持较大值 | `8192` |
| `mlp_chunk_tokens` | MLP 的目标 token 分块上限；节点下方会显示本次实际生效值 | `4096` |
| `auto_halve_on_oom` | 当前 chunk OOM 时自动减半重试 | `true` |
| `提前加载下一层（提速）` | 开启时提前加载下一 block 以提速；预取或切换 block OOM 时关闭 | `true` |
| `reuse_mlp_weights` | 自动策略：将已准备权重复制到独立快照后复用；无法快照或 OOM 时改用 streamed | `true` |
| `attention_backend` | 保留上游后端或显式采用 CK INT8 | `comfy_kitchen_int8` |
| `verbose` | 输出首个同形状 block 的紧凑诊断 | `true` |

为兼容旧工作流，该开关保存时仍沿用内部字段名 `disable_dynamic_prefetch`；从 v2.2.12
开始其布尔值采用界面上的正向含义：`true` 就是提前加载，`false` 就是不提前加载。

参数越大不等于必然更快。较大的 chunk 减少 kernel 启动次数，但会增加瞬时激活和权重预取竞争。比较参数时必须固定模型、seed、分辨率、帧数、步数和注意力后端。

节点会在两个数值输入的正下方分别显示 `RoPE 当前使用` 和 `MLP 当前使用`。正常时会显示
`当前使用 N（设定值）`；发生显存不足后会显示 `已降级为 N（设定 M）`。这里的 `M` 是工作流
里的目标上限，`N` 是本次模型会话实际采用的上限。输入框本身不会被偷偷改写，也不会把临时
降级值保存进工作流。

如果当前 packed sequence 本身短于设定值，状态会显示
`当前使用 N（设定 M，受序列长度限制）`。这是本次 forward 的真实 token 上限，不是 OOM
降级，也不会把较短序列的值记忆到后续较长序列。

`auto_halve_on_oom=true` 时，RoPE 或 MLP 当前分块 OOM 会按当前值整数减半，最低到 `256`。
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
| 下一 block 预取或 block 切换时 OOM | `提前加载下一层` | 改为 `false`，关闭预取以换取显存余量 |
| QKV 或 attention kernel OOM | 注意力后端 | 改用 CK INT8，或保留上游 Low VRAM/Sage；两个 chunk 值不是主要控制项 |
| 模型加载阶段已经 OOM | 加载器/量化/卸载策略 | 分块节点尚未执行，调 chunk 无效 |

显存档位只能作为起点，因为同容量显卡还会受到模型格式、LoRA反量化、驱动和常驻策略影响：

| 专用显存 | `chunk_tokens` 起点 | `mlp_chunk_tokens` 起点 | 预取建议 |
|---:|---:|---:|---|
| 20–24GB | `8192` | `4096` | `true`；稳定后可测试 `false` |
| 16–20GB | `8192` | `2048–4096` | `true`；优先降低 MLP |
| 12–16GB | `8192` | `1024–2048` | `true`；RoPE OOM 再降 RoPE |
| 12GB以下 | `4096–8192` | `512–1024` | `true`；长视频成功率取决于模型卸载 |

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

本机观察值：

| 路径 | 采样耗时 | 相对 CK INT8 |
|---|---:|---:|
| KJNodes 自定义 SM75 SageAttention 2 路径 | 约170秒/步 | CK 单步缩短约29.4%，吞吐约1.42× |
| KJNodes Low VRAM Attention (`head_chunks=4`) | 约177秒/步 | CK 单步缩短约32.2%，吞吐约1.48× |
| Comfy Kitchen INT8 | 约120秒/步 | 基准 |

本机 RTX 20 系/SM75 结果表明，CK INT8 比 [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) 的 MiniMax H3 自定义 SM75 SageAttention 2 路径更快。这个结论限定于当前2080 Ti、模型和软件版本，不代表 CK 在30/40系上必然优于 Sage。

| 案例 | 空间/时长 | 完整任务实测 |
|---|---|---:|
| A | 1.0MP / 10秒 / 4-step LoRA | **620秒** |
| B | 0.6MP / 15秒 / 4-step LoRA | **530秒** |
| C | 0.4MP / 10秒 / 4-step LoRA | **180秒** |
| D | 0.4MP / 5秒 / 4-step LoRA | **85秒** |

A 的四步采样约480秒。以上完整时间还包含当前工作流的模型调度、VAE解码、超分、音频和视频封装；后处理仍在继续调优，因此只作为同机案例，不把120秒/步误写成完整视频耗时。

### 为什么分辨率和时长增加后不是线性耗时

令 `S` 为 H3 实际 packed sequence token 数。固定模型结构下，可以粗略写成：

```text
S ≈ S_condition + k × spatial_tokens × temporal_tokens
T ≈ T_fixed + aS + bS² + T_post
```

- RoPE、RMSNorm、QKV投影和MLP的主要工作量近似随 `S` 线性增长；
- 全局 self-attention 的 QK/AV 计算近似含 `S²` 项；memory-efficient 或分块实现主要降低中间张量驻留，并不会消除全部注意力算术；
- VAE、超分和编码又大致随“像素数 × 帧数”增长，并带有固定启动和封装开销。

因此总耗时是固定项、线性项、二次注意力项和后处理项的混合。分辨率与时长同时增加时，token 预算近似相乘，attention 占比上升后会表现为超线性增长。实测 A/C 的空间×时长预算约为 `2.5×`，完整耗时却为 `3.44×`；C/D 的预算约为 `2×`，耗时约为 `2.12×`，符合“并非固定线性倍数”的现象。

RoPE/MLP 分块本身不减少理论 FLOPs。显存没有 OOM、没有触发系统内存换入换出、且原 kernel 已能高效运行时，单纯启用或缩小 chunk 通常不会提速，甚至可能因更多小 GEMM 和 kernel 启动而变慢。本机从约170秒/步降到约120秒/步的主要加速来源是 **CK INT8 替换 SM75 Sage2 注意力路径**，不是激活分块凭空减少了计算量。

显存观察：

- 单参考图案例截图：专用显存约 `16.7 / 22.0GB`；
- 无参考文生视频案例：专用显存约 `15.6GB`；
- 采样截图时 GPU 约95%，Copy引擎接近0%。

本次 Aki 启动器中开启了“尽量将模型保留在显存”，关闭“稳定计算”和 Channels-Last，并选择“CUDA 内置异步分配器”。模型常驻选项用于减少模型在显存与系统内存之间的换入换出，不等于禁用 Windows WDDM 共享 GPU 内存。任务管理器仍显示约13GB共享 GPU 内存映射，因此本项目不宣称“零共享内存”。实测未观察到持续 Copy/PCIe 搬运造成的明显采样降速，但单张任务管理器截图不能证明整个运行过程从未访问系统内存。

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

`reuse_mlp_weights` 是自动策略请求，而不是强制命令。节点会先按 ComfyUI 当前路径准备并应用 LoRA/权重补丁，再把每层结果复制到独立 resident 快照，避免 AIMDO/VBAR 共享 staging buffer 被后续层覆盖。快照不支持、复制失败或显存不足时才改用 streamed-safe 路径。旧工作流里的 `true` 仍会按这个规则自动判断。小显存卡能否完成目标还取决于模型权重、LoRA 是否触发反量化、注意力工作集、ComfyUI 卸载策略和操作系统。分块降低的是扩展激活峰值，不会让全部模型权重凭空消失。

该字段保留在节点界面，方便高级用户手动关闭 resident 策略进行排查或兼容性验证。开启时会使用独立权重快照，失败或 OOM 才会自动切换 streamed。字段位置和工作流序列化保持不变，因此旧工作流不会错位。

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
