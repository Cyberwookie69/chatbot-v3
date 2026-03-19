"""
gpu_utils.py — Multi-GPU utilities for OpenShift AI / CUDA clusters.

Version : 4.1.1
Modified: 2026-03-19
Changes : v4.1.1 — Remove prefer_gpu, add max_gpus param + CPU-only mode (--gpus 0)
          v4.1.0 — Add max_gpus parameter to setup_device for CLI --gpus support
          v4.0.0 — Version bump for multi-corpus project
          v3.0.0 — Initial version: auto-detect, DataParallel, A100 batch scaling

Supports 1-4 A100 GPUs (or any CUDA device) with automatic detection and
batch size scaling. Designed for environments where GPU count and type are
not known in advance.

Usage:
    from gpu_utils import setup_device, auto_scale_config, wrap_model, unwrap_model

    device, gpu_info = setup_device()
    config = auto_scale_config(config, gpu_info)
    model = wrap_model(model, gpu_info)
    raw_model = unwrap_model(model)  # access .encoder, .decoder, etc.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GPUInfo:
    """Detected GPU environment."""
    device: torch.device
    num_gpus: int = 0
    gpu_names: List[str] = field(default_factory=list)
    total_memory_gb: List[float] = field(default_factory=list)
    is_a100: bool = False
    supports_bf16: bool = False

    def summary(self) -> str:
        if self.num_gpus == 0:
            return "CPU only"
        gpus = ", ".join(
            f"{name} ({mem:.0f}GB)" for name, mem in zip(self.gpu_names, self.total_memory_gb)
        )
        return f"{self.num_gpus}x GPU: {gpus}"


def setup_device(max_gpus: Optional[int] = None) -> Tuple[torch.device, "GPUInfo"]:
    """
    Detect available GPUs and return (device, GPUInfo).

    Args:
        max_gpus: Cap the number of GPUs to use. 0 = force CPU. None = no cap.

    Returns:
        (device, gpu_info) — device is always cuda:0 when GPUs are available,
        or cpu when not. gpu_info contains detection results for config scaling.
    """
    info = GPUInfo(device=torch.device("cpu"))

    if max_gpus == 0:
        print("[gpu_utils] GPU disabled via --gpus 0 — using CPU")
        return info.device, info

    if not torch.cuda.is_available():
        print("[gpu_utils] CUDA not available — using CPU")
        return info.device, info

    num_gpus = torch.cuda.device_count()

    if max_gpus is not None and num_gpus > max_gpus:
        print(f"[gpu_utils] Capping GPUs: {num_gpus} available → {max_gpus} (--gpus)")
        num_gpus = max_gpus

    info.num_gpus = num_gpus
    info.device = torch.device("cuda:0")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        info.gpu_names.append(props.name)
        mem_gb = props.total_memory / (1024 ** 3)
        info.total_memory_gb.append(mem_gb)

    info.is_a100 = any("A100" in name for name in info.gpu_names)
    info.supports_bf16 = all(
        torch.cuda.get_device_properties(i).major >= 8
        for i in range(num_gpus)
    )

    print(f"[gpu_utils] {info.summary()}")
    print(f"[gpu_utils] bf16 support: {info.supports_bf16}")

    return info.device, info


def auto_scale_config(config: dict, gpu_info: GPUInfo) -> dict:
    """
    Scale config for the detected GPU environment.

    Adjustments:
    - Batch size scales linearly with GPU count (DataParallel splits across GPUs).
    - num_workers increased for multi-GPU to avoid data loading bottleneck.
    - A100-specific: larger batch size (80GB VRAM), TF32 enabled.
    - amp_dtype set based on hardware capability.

    Returns a new config dict (does not mutate the input).
    """
    cfg = dict(config)

    if gpu_info.num_gpus == 0:
        cfg["amp_dtype"] = "float32"
        return cfg

    # A100-specific optimisations.
    if gpu_info.is_a100:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[gpu_utils] TF32 enabled (A100)")

    # Scale batch size with GPU count.
    base_batch = cfg.get("batch_size", 256)

    # A100 80GB can handle much larger batches than consumer GPUs.
    # Model is ~44M params LSTM — VRAM usage is dominated by batch size.
    min_vram = min(gpu_info.total_memory_gb) if gpu_info.total_memory_gb else 0
    if min_vram >= 70:       # A100 80GB
        per_gpu_batch = 1024
    elif min_vram >= 35:     # A100 40GB / A6000
        per_gpu_batch = 512
    elif min_vram >= 20:     # RTX 3090 / 4090
        per_gpu_batch = 256
    else:
        per_gpu_batch = base_batch

    # DataParallel: effective batch = per_gpu_batch * num_gpus.
    # Keep grad_accum_steps constant so effective batch scales with GPU count.
    cfg["batch_size"] = per_gpu_batch * gpu_info.num_gpus
    print(f"[gpu_utils] batch_size: {base_batch} → {cfg['batch_size']} "
          f"({per_gpu_batch}/GPU × {gpu_info.num_gpus} GPUs)")

    # Scale data loading workers.
    if gpu_info.num_gpus >= 2:
        cfg["num_workers"] = min(gpu_info.num_gpus * 4, 16)
    elif gpu_info.num_gpus == 1:
        cfg["num_workers"] = 8
    print(f"[gpu_utils] num_workers: {config.get('num_workers', 0)} → {cfg['num_workers']}")

    # AMP dtype: bf16 if supported (A100, H100, RTX 30xx+), else fp16.
    if gpu_info.supports_bf16:
        cfg["amp_dtype"] = "bfloat16"
    else:
        cfg["amp_dtype"] = "float16"

    return cfg


def wrap_model(model: nn.Module, gpu_info: GPUInfo) -> nn.Module:
    """
    Wrap model in DataParallel if multiple GPUs are available.

    DataParallel is used (not DistributedDataParallel) because:
    - Simpler setup: no process spawning or NCCL init needed.
    - Works in notebooks and interactive environments (OpenShift AI JupyterLab).
    - Sufficient for 1–4 GPU LSTM models (DDP overhead matters more for
      large transformer models across many nodes).

    Returns the (possibly wrapped) model.
    """
    if gpu_info.num_gpus <= 1:
        return model

    device_ids = list(range(gpu_info.num_gpus))
    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"[gpu_utils] Model wrapped in DataParallel (GPUs: {device_ids})")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Get the underlying model from a DataParallel wrapper.

    Safe to call on non-wrapped models (returns as-is).
    Use this whenever you need to access model internals (encoder, decoder, etc.)
    for inference, checkpoint saving, or attention visualization.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model
