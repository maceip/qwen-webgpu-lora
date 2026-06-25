---
title: VibeThinker WebGPU Train + Hot-Swap LoRA
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
  - WeiboAI/VibeThinker-3B
short_description: In-browser WebGPU LoRA training + hot-swap (VibeThinker-3B)
---

# VibeThinker-3B · in-browser WebGPU training + hot-swap LoRA

A static Space that runs **VibeThinker-3B** entirely in your browser tab on a
custom **WebGPU int4 runtime** — no server, no upload, nothing leaves the page.

What it shows off:

- **WebGPU kernels** — int4 GEMV/GEMM, fused RMSNorm+QKV+RoPE, paged attention,
  f32 accumulation in f16 paths. The whole forward + backward stack is WGSL.
- **In-browser training** — fine-tune a LoRA adapter in the tab with a *real*
  backward pass + AdamW over the frozen int4 base (gradient checkpointing,
  f32 master weights). The guided demo teaches made-up facts in ~30s.
- **Hot-swap LoRA** — every trained adapter is saved locally (IndexedDB) and
  hot-swaps into inference instantly; export to `.safetensors` or re-load later.

The model weights are streamed once from
[`WeiboAI/VibeThinker-3B`](https://huggingface.co/WeiboAI/VibeThinker-3B) on the
Hub and cached in your browser. A Space does **not** have to be tied to a model;
the model above is linked for discovery only.

> Requires a WebGPU-capable browser (recent Chrome/Edge; Safari Technology
> Preview). ~6 GB weights, one-time download.

Source: <https://github.com/maceip/vibethinker-webgpu-lora>
