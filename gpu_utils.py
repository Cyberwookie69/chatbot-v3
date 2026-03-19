"""
gpu_utils.py — Multi-GPU utilities for OpenShift AI / CUDA clusters.

Version : 4.2.1
Modified: 2026-03-19
Changes : v4.2.1 — Remove DataParallel completely, DDP-only for multi-GPU
          v4.1.1 — Remove prefer_gpu, add max_gpus param + CPU-only mode (--gpus 0)
          v4.1.0 — Add max_gpus parameter to setup_device for CLI --gpus support
          v4.0.0 — Version bump for multi-corpus project
          v3.0.0 — Initial version: auto-detect, DataParallel, A100 batch scaling

Supports 1-4 A100 GPUs (or any CUDA device) with automatic detection and
batch size scaling. Designed for environments where GPU count and type are
not known in advance.

Usage (single GPU or CPU):
    from gpu_utils import setup_device, auto_scale_config, wrap_model, unwrap_model

    device, gpu_info = setup_device()
    config = auto_scale_config(config, gpu_info)
    model = wrap_model(model, gpu_info)
    raw_model = unwrap_model(model)

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=2 train.py

    # In train.py, setup_device() auto-detects torchrun and initializes DDP.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class GPUInfo:
    """Detected GPU environment."""
    device: torch.device
    num_gpus: int = 0
    gpu_names: List[str] = field(default_factory=list)
    total_memory_gb: List[float] = field(default_factory=list)
    is_a100: bool = False
    supports_bf16: bool = False
    # DDP fields
    use_ddp: bool = False
    rank: int = 0             # global rank (0 = main process)
    local_rank: int = 0       # GPU index on this node
    world_size: int = 1       # total number of processes

    def summary(self) -> str:
        if self.num_gpus == 0:
            return "CPU only"
        gpus = ", ".join(
            f"{name} ({mem:.0f}GB)" for name, mem in zip(self.gpu_names, self.total_memory_gb)
        )
        if self.use_ddp:
            return f"DDP rank {self.rank}/{self.world_size} — {gpus}"
        return f"{self.num_gpus}x GPU: {gpus}"


def is_main_process(gpu_info: GPUInfo) -> bool:
    """True if this is rank 0 (or non-DDP). Use to gate printing/saving."""
    return gpu_info.rank == 0


def setup_device(max_gpus: Optional[int] = None) -> Tuple[torch.device, GPUInfo]:
    """
    Detect available GPUs and return (device, GPUInfo).

    Auto-detects torchrun (DDP) via RANK/LOCAL_RANK/WORLD_SIZE env vars.
    Falls back to single-GPU mode otherwise.

    Args:
        max_gpus: Cap GPUs in non-DDP mode. 0 = force CPU. Ignored in DDP mode.

    Returns:
        (device, gpu_info)
    """
    info = GPUInfo(device=torch.device("cpu"))

    # ── DDP mode: launched via torchrun ───────────────────────────────────
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        device = torch.device(f"cuda:{local_rank}")
        info.device = device
        info.num_gpus = 1  # each DDP process sees 1 GPU
        info.use_ddp = True
        info.rank = rank
        info.local_rank = local_rank
        info.world_size = world_size

        # Fill GPU info for local device
        props = torch.cuda.get_device_properties(local_rank)
        info.gpu_names.append(props.name)
        info.total_memory_gb.append(props.total_memory / (1024 ** 3))
        info.is_a100 = "A100" in props.name
        info.supports_bf16 = props.major >= 8

        if rank == 0:
            print(f"[gpu_utils] DDP initialized: {world_size} processes")
            for i in range(world_size):
                name = torch.cuda.get_device_properties(i).name if i < torch.cuda.device_count() else "remote"
                print(f"  rank {i}: {name}")
            print(f"[gpu_utils] bf16 support: {info.supports_bf16}")

        return device, info

    # ── Non-DDP mode ─────────────────────────────────────────────────────
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

    In DDP mode, batch_size is per-GPU (each process sees num_gpus=1).
    In non-DDP mode, batch_size = per_gpu_batch * num_gpus.

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
        if is_main_process(gpu_info):
            print("[gpu_utils] TF32 enabled (A100)")

    # Scale batch size based on VRAM.
    base_batch = cfg.get("batch_size", 256)
    min_vram = min(gpu_info.total_memory_gb) if gpu_info.total_memory_gb else 0
    if min_vram >= 70:       # A100 80GB
        per_gpu_batch = 1024
    elif min_vram >= 35:     # A100 40GB / A6000
        per_gpu_batch = 512
    elif min_vram >= 20:     # RTX 3090 / 4090
        per_gpu_batch = 256
    else:
        per_gpu_batch = base_batch

    if gpu_info.use_ddp:
        # DDP: batch_size is per-GPU, each process handles one GPU
        cfg["batch_size"] = per_gpu_batch
        if is_main_process(gpu_info):
            print(f"[gpu_utils] batch_size: {base_batch} → {per_gpu_batch}/GPU "
                  f"(DDP, {gpu_info.world_size} processes, "
                  f"effective={per_gpu_batch * gpu_info.world_size})")
    else:
        # Non-DDP: effective batch = per_gpu_batch * num_gpus
        cfg["batch_size"] = per_gpu_batch * gpu_info.num_gpus
        if is_main_process(gpu_info):
            print(f"[gpu_utils] batch_size: {base_batch} → {cfg['batch_size']} "
                  f"({per_gpu_batch}/GPU × {gpu_info.num_gpus} GPUs)")

    # Scale data loading workers.
    if gpu_info.num_gpus >= 1:
        cfg["num_workers"] = 8
    if is_main_process(gpu_info):
        print(f"[gpu_utils] num_workers: {config.get('num_workers', 0)} → {cfg['num_workers']}")

    # AMP dtype: bf16 if supported (A100, H100, RTX 30xx+), else fp16.
    if gpu_info.supports_bf16:
        cfg["amp_dtype"] = "bfloat16"
    else:
        cfg["amp_dtype"] = "float16"

    return cfg


def wrap_model(model: nn.Module, gpu_info: GPUInfo) -> nn.Module:
    """
    Wrap model in DDP for multi-GPU training.

    DDP wrapping requires the model to already be on the correct device.
    torch.compile should be applied BEFORE calling wrap_model.

    Returns the (possibly wrapped) model.
    """
    if gpu_info.use_ddp:
        model = DDP(model, device_ids=[gpu_info.local_rank])
        if is_main_process(gpu_info):
            print(f"[gpu_utils] Model wrapped in DDP (rank {gpu_info.rank}, "
                  f"device cuda:{gpu_info.local_rank})")
        return model

    # Single GPU or CPU — no wrapping needed
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Get the underlying model from a DDP wrapper.

    Safe to call on non-wrapped models (returns as-is).
    Use this whenever you need to access model internals (encoder, decoder, etc.)
    for inference, checkpoint saving, or attention visualization.
    """
    if isinstance(model, DDP):
        return model.module
    return model


def cleanup_ddp() -> None:
    """Tear down DDP process group if it was initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
