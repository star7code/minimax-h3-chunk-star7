# MiniMax H3 Turbo curve-grid notice

`assets/h3_silu_temb_grid.safetensors` and the pruned-H3 AdaLN adaptation
mechanism are derived from Larryvrh's `ComfyUI-MiniMax-H3-Turbo` project.

- Upstream: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
- License: Apache License 2.0
- Purpose here: preserve full-model LoRA time-conditioning contributions when
  the receiving MiniMax H3 checkpoint uses the compressed AdaLN curve format.

The remaining Star7 implementation integrates this mechanism with ComfyUI
ModelPatcher cloning, existing object patches, and chunk-node diagnostics.
