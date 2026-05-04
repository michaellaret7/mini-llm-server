"""
Configuration for the vLLM server.

Defaults follow NVIDIA's recommended launch command for
NVIDIA-Nemotron-3-Nano-4B-FP8 (see model README), scaled for a 24 GB GPU.
Override any field by editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ServerConfig:
    # --- Model ---
    model: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
    served_model_name: str = "nemotron3-nano-4b-fp8"

    # --- Network ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Capacity / memory ---
    max_model_len: int = 16384
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True

    # --- Cache dtypes (memory savings) ---
    kv_cache_dtype: str = "fp8"
    mamba_ssm_cache_dtype: str = "float32"

    # --- Reasoning + tool calling ---
    # Nemotron 3 emits <think>...</think> reasoning blocks parsed by a custom
    # plugin shipped with the model snapshot.
    reasoning_parser: str = "nano_v3"
    reasoning_parser_plugin: Path | None = None  # set at runtime by serve.py

    # The model was post-trained to emit tool calls in Qwen3-Coder's format.
    tool_call_parser: str = "qwen3_coder"
    enable_auto_tool_choice: bool = True

    # --- Misc ---
    trust_remote_code: bool = True
    enforce_eager: bool = False
    async_scheduling: bool = True
    download_dir: Path = field(default_factory=lambda: MODELS_DIR)

    # FlashInfer is faster but its prefill module JIT-compiles via nvcc at
    # startup. Use FLASHINFER on hosts with the CUDA toolkit installed
    # (RunPod's PyTorch templates have it). TRITON_ATTN works without nvcc.
    attention_backend: str = "FLASHINFER"

    def to_cli_args(self) -> list[str]:
        """Render this config as a list of `vllm serve ...` CLI arguments."""
        args: list[str] = [
            "serve", self.model,
            "--served-model-name", self.served_model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--mamba_ssm_cache_dtype", self.mamba_ssm_cache_dtype,
            "--reasoning-parser", self.reasoning_parser,
            "--tool-call-parser", self.tool_call_parser,
            "--download-dir", str(self.download_dir),
            "--attention-backend", self.attention_backend,
        ]
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.async_scheduling:
            args.append("--async-scheduling")
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enable_auto_tool_choice:
            args.append("--enable-auto-tool-choice")
        if self.reasoning_parser_plugin is not None:
            args += ["--reasoning-parser-plugin", str(self.reasoning_parser_plugin)]
        return args
