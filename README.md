<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>Optimized WebGPU inference for VibeThinker-3B — in your browser tab. No server, no upload.</em></p>

<p align="center">
<b>≥20 tok/s decode floor · live LoRA hot-swap · bit-exact reference checks · 100% client-side WebGPU</b>
</p>

<p align="center"><a href="https://maceip.github.io/qwen-webgpu-lora/"><b>▶ Live demo</b></a> · <a href="https://github.com/maceip/emberglass-tune">Training docs</a> · <a href="https://github.com/maceip/vibebounty">VibeBounty demo</a></p>

---

## Three repos

| Repo | Role | Train? | Run inference? |
|---|---|---|---|
| **[emberglass](https://github.com/maceip/qwen-webgpu-lora)** (this) | Custom **WebGPU** runtime — int4, fused kernels, LoRA hot-swap | **No** | **Yes** (browser) |
| **[emberglass-tune](https://github.com/maceip/emberglass-tune)** | LoRA **training** — MLX + CUDA, Anthropic trace pipeline | **Yes** | No |
| **[vibebounty](https://github.com/maceip/vibebounty)** | Bug-bounty **demo** — tuned adapter, HackerOne UI, CPU/GPU serve | Uses emberglass-tune | Yes (server) |

**How the weights are made:** labeled reports → Anthropic teacher traces → LoRA SFT → `adapter_model.safetensors`. Full pipeline: **[emberglass-tune README](https://github.com/maceip/emberglass-tune)**.

**How to run them here:** load base weights + optional adapter into WebGPU; forward pass only. No backward pass, no optimizer, no dataset code in this repo.

---

## What this repo is

Emberglass is an **inference-only** engine for Qwen2.5-class models (VibeThinker-3B):

- Custom **WGSL kernels** (GEMV/GEMM, attention, RoPE, sampling)
- **int4** layer weights on GPU, GPU-resident KV cache
- **Runtime LoRA hot-swap** — load PEFT/MLX `adapter_model.safetensors` without re-quantizing base (`src/lora_gpu.js`)
- Playwright correctness and throughput harnesses (`npm run test:*`)

| In emberglass | Elsewhere |
|---|---|
| WebGPU forward pass | Training → **emberglass-tune** |
| LoRA apply / swap / clear | Data + Anthropic traces → **emberglass-tune** + **vibebounty** |
| int4 load from `./model` or HF | HackerOne demo UI → **vibebounty** |
| | CPU/GPU serve for demos → **vibebounty** |

---

## Run it

```bash
cd ~/emberglass
npm install
npm run build
npm run serve    # http://localhost:8013
```

Open in Chrome/Edge with **WebGPU + `subgroups`**. Load base weights from `./model`, Hugging Face, or a directory picker. Optional LoRA adapter URL for hot-swap.

**Base model:** [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)  
**Example adapter:** [macmacmacmac/vibebounty](https://huggingface.co/macmacmacmac/vibebounty) (train with emberglass-tune)

---

## Using a trained adapter

1. Train (or download) a PEFT adapter — see [emberglass-tune](https://github.com/maceip/emberglass-tune).
2. Serve adapter files same-origin (e.g. under `/adapters/my-run/`).
3. Load in the Emberglass UI or via VibeBounty's Emberglass bridge.

Tests: `npm run test:lora`, `npm run test:lora-path`.

---

## Verification

```bash
npm run test:correctness   # argmax / generation vs reference
npm run test:lora          # adapter parse, hot-swap, restore
npm run test:app           # full streaming UI path
npm run bench:wgpu         # structured throughput JSON
```

Requires port **8013**, WebGPU **`subgroups`**, and weights in `./model` (not bundled in repo).

---

## Performance

Throughput is hardware-dependent. Target: **≥20 tok/s** greedy decode on Intel Arc class; **~35 tok/s** on Apple M5 Max (Metal).

| Platform | Greedy decode (typical) |
|---|---:|
| Apple M5 Max + Metal | ~33–35 tok/s @ long ctx |
| Intel Arc 140V + D3D11 | ~22–24 tok/s @ short ctx |
| LoRA active (180 modules) | ~23 tok/s (M5 reference) |

Fused decode path: `fuseQKV` / `fuseRoPE` / `fuseMLP` / `fuseResidual`.

---

## Requirements

- Browser WebGPU with **`subgroups`** (no fallback kernel set)
- GPU memory for chosen context window
- Bring your own weights — repo does not ship model files

---

## Layout

```
src/qwgpu/           WGSL kernels, runtime, int4 quantize
src/lora_gpu.js      PEFT/MLX adapter → GPU buffers
src/services/        App session, generation, adapter registry
test/                Browser harnesses
docs/                GitHub Pages demo + architecture notes
model/               BYO base weights (gitignored)
```

---

## Related docs

- **Training (MLX, CUDA, Anthropic traces):** [emberglass-tune README](https://github.com/maceip/emberglass-tune)
- **Bug-bounty demo:** [vibebounty](https://github.com/maceip/vibebounty)
- **Architecture map:** [`docs/REPO_ARCHITECTURE.md`](docs/REPO_ARCHITECTURE.md)

---

## Kernel work, prefil speed, decode tok/sec

The benchmark (`npm run bench:wgpu`) run inside a browser reports:

- Kernel category timings (timestamp queries): embed, rmsNormQkvRope, attnP, attnC, g4add, rms, gu, gemv, etc.
- Prefil latency (ms) across context lengths.
- Decode / sampling throughput (tok/sec).

Example from verification run:

{"type":"sampling-topk","topK":40,"tokens":8,"seconds":2.78,"tokPerSec":2.87}

Run the benchmark in Google Chrome Canary on real hardware to capture accurate kernel work, prefil speed, and decode tok/sec.

---

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
