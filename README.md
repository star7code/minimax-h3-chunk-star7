# MiniMax H3 Activation Chunk (RoPE + MLP) - Star7

[中文说明](#中文说明) · [Benchmark](BENCHMARKS.md) · [Example workflows](examples/workflows)

Activation-memory control for local MiniMax H3 inference in ComfyUI. The default path chunks eager RoPE and H3 MLP activations without changing the sampler, latent, VAE, video duration, or spatial resolution. An explicit Comfy Kitchen INT8 attention option is available for GPUs where the existing Sage path is not competitive.

> This is an independent community project. MiniMax, ComfyUI, Comfy Kitchen, KJNodes, and NVIDIA are trademarks or projects of their respective owners.

## 中文说明

这个节点针对 MiniMax H3 长序列推理的两个显存峰值：

- `RMSNorm -> split-half RoPE (Q/K)`；
- `fc1 -> SwiGLU -> fc2` MLP 扩展激活。

RoPE 沿 sequence/token 维分块并原位写回 Q/K。MLP 按 token 分块计算，把各段 `fc2` 结果直接累加回 residual，避免同时保留完整扩展激活和完整 MLP 输出。对于 `int8_tensorwise + ConvRot` 权重，节点保持 `QuantizedTensor` 路径，并允许各 token chunk 复用已经准备好的权重。

默认 `attention_backend = existing` 时，本项目不会修改注意力算法，也不会改变采样器、sigma、seed、VAE、latent、帧数或画面分辨率。`comfy_kitchen_int8` 是用户显式选择的近似注意力后端，不属于无损默认路径。

## 安装

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
| `chunk_tokens` | RoPE 的 token 分块大小 | `8192` |
| `mlp_chunk_tokens` | MLP 的 token 分块大小 | `4096` |
| `auto_halve_on_oom` | 当前 chunk OOM 时自动减半重试 | `true` |
| `disable_dynamic_prefetch` | 禁止下一 block 权重预取，省显存但可能变慢 | `false` |
| `reuse_mlp_weights` | token chunks 之间复用已准备权重 | `true` |
| `attention_backend` | 保留上游后端或显式采用 CK INT8 | `comfy_kitchen_int8` |
| `verbose` | 输出首个同形状 block 的紧凑诊断 | `true` |

参数越大不等于必然更快。较大的 chunk 减少 kernel 启动次数，但会增加瞬时激活和权重预取竞争。比较参数时必须固定模型、seed、分辨率、帧数、步数和注意力后端。

### 按 OOM 位置调整，而不是盲目同时降低两个值

| 报错位置或现象 | 应优先调整 | 建议动作 |
|---|---|---|
| `rms_rope_split_half_` / `apply_rope_split_half1` | `chunk_tokens` | `8192 -> 4096 -> 2048 -> 1024` |
| `fc1` / SwiGLU / `fc2` MLP 激活 OOM | `mlp_chunk_tokens` | `4096 -> 2048 -> 1024 -> 512` |
| 下一 block 预取或 block 切换时 OOM | `disable_dynamic_prefetch` | 改为 `true`，牺牲部分重叠速度换显存余量 |
| QKV 或 attention kernel OOM | 注意力后端 | 改用 CK INT8，或保留上游 Low VRAM/Sage；两个 chunk 值不是主要控制项 |
| 模型加载阶段已经 OOM | 加载器/量化/卸载策略 | 分块节点尚未执行，调 chunk 无效 |

显存档位只能作为起点，因为同容量显卡还会受到模型格式、LoRA反量化、驱动和常驻策略影响：

| 专用显存 | `chunk_tokens` 起点 | `mlp_chunk_tokens` 起点 | 预取建议 |
|---:|---:|---:|---|
| 20–24GB | `8192` | `4096` | 先 `false`，OOM 再改 `true` |
| 16–20GB | `4096` | `2048–4096` | 建议先 `true` 验证 |
| 12–16GB | `2048–4096` | `1024–2048` | `true` |
| 12GB以下 | `1024–2048` | `512–1024` | `true`；可能仍需模型卸载，不保证长视频可行 |

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

这样本节点保留上游 Sage，只负责 RoPE/MLP 激活分块。若仍设为 `comfy_kitchen_int8`，本节点会覆盖更早安装的 Sage patch。Sage 与 CK INT8 都属于近似注意力，速度和画面质量应在同 seed、同分辨率下比较。

## 小显存补充说明

建议始终保持 `reuse_mlp_weights=true`，并从前面的显存档位表开始。小显存卡能否完成目标还取决于模型权重、LoRA是否触发反量化、注意力工作集、ComfyUI卸载策略和操作系统。分块降低的是激活峰值，不会让全部模型权重凭空消失。

## 示例工作流

- [通用工作流](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json)：普通 `UNETLoader`，适合自行选择原生、Sage或CK注意力的环境。
- [RTX 20系工作流](examples/workflows/MiniMax-H3-Activation-Chunk-RTX20-Star7.json)：使用 Native FP16 Loader，并默认采用 CK INT8。

两份工作流的参考条件由 [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) 提供，并保留一个 `ref_image_0` 接口。本次 A/B/C/D 案例均采用该 T8 参考节点的单参考图模式。仓库不附带可能存在版权或隐私问题的原始参考素材；导入后请把 `replace-with-your-reference-image.png` 替换为自己的图片。工作流还使用了 VideoHelperSuite、NVIDIA RTX Nodes 等可选节点；缺失节点需按 ComfyUI 提示安装。

## 数值与兼容性

- RoPE eager 分块路径可以逐元素一致；
- MLP 不改变公式、权重、dtype或token顺序，但大 GEMM 拆成小 GEMM 后，底层归约顺序可能产生约 `1.19e-7` 的 float32 末位差，因此是数值等价而非 bitwise identical；
- `comfy_kitchen_int8` 会量化注意力计算，是显式的近似模式；
- 只有检测到 MiniMax H3 典型 Q/K shape 和匹配的 RoPE frequency 序列维时才安装补丁；其他 shape 回退到原实现。

## 日志

节点只对首个同形状 block 输出紧凑信息，例如：

```text
[Star7 H3 Chunk] v2.1.0 active | blocks=... | fp16=...
[Star7 H3 Chunk] Attention profile (one block) | ...
[Star7 H3 Chunk] MLP profile (one block) | chunks=... | weights=resident-quantized
```

`weights=resident-quantized` 表示 INT8/ConvRot 权重仍保持量化并在 chunks 间复用；`streamed-fallback` 表示显存不足后进入流式回退。

## License

[MIT](LICENSE)
