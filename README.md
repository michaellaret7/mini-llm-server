# mini-llm-server

<p align="center">
  <img src="banner.svg" alt="Pixel-art night skyline with a tiny GPU server tower streaming tokens into the sky" width="100%">
</p>

vLLM OpenAI-compatible server for NVIDIA Nemotron 3 Nano 4B FP8.

The whole repo is a thin wrapper that picks the right vLLM flags for this model and renders them into a `vllm serve ...` invocation.

## Requirements

- Python 3.12
- NVIDIA GPU with at least 16 GB VRAM (24 GB recommended)
- CUDA-capable driver
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
git clone https://github.com/michaellaret7/mini-llm-server.git
cd mini-llm-server
uv sync
cp .env.example .env   # add HF_TOKEN if needed
```

## Download the model

```bash
uv run python scripts/download_model.py
```

Weights land in `./models` (~5 GB for the FP8 variant). Run once per machine.

## Run the server

```bash
uv run python -m server.serve
```

Once it logs `Application startup complete`, the OpenAI-compatible API is at:

```
http://localhost:8000/v1
```

Test it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron3-nano-4b-fp8",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

## Config

Edit `server/config.py`. Defaults are tuned for a 24 GB GPU at 16k context with 32 concurrent sequences. Notable knobs:

| Field | Notes |
|---|---|
| `max_model_len` | Context window. NVIDIA recommends 262144; needs >24 GB. |
| `max_num_seqs` | Concurrent sequences. Raise if you have spare VRAM. |
| `attention_backend` | `FLASHINFER` (faster, needs nvcc) or `TRITON_ATTN` (works without). |
| `kv_cache_dtype` | `fp8` halves KV cache memory. |
| `gpu_memory_utilization` | Fraction of VRAM vLLM may allocate. |

## Deploy on RunPod

1. Spin up a GPU pod (RTX 4090 recommended) with the **PyTorch 2.x** template
2. Mount `/workspace` as a persistent volume; expose HTTP port 8000
3. SSH in:
   ```bash
   cd /workspace
   git clone https://github.com/michaellaret7/mini-llm-server.git
   cd mini-llm-server
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   uv run python scripts/download_model.py
   uv run python -m server.serve
   ```
4. The proxy URL is `https://<pod-id>-8000.proxy.runpod.net/v1`

## Troubleshooting

- **`Could not find nvcc`** — switch `attention_backend` to `TRITON_ATTN`, or install the CUDA toolkit
- **OOM on startup** — lower `max_model_len` or `max_num_seqs`, or drop `gpu_memory_utilization` to `0.85`
- **`no snapshot under .../models`** — run `scripts/download_model.py` first
- **Slow on WSL** — `pin_memory=False` warning is unavoidable; native Linux is faster
