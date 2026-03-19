"""
train.py — Training loop for clean-from-scratch Seq2Seq chatbot.

Version : 4.2.6
Modified: 2026-03-19
Changes : v4.2.6 — Suppress PyTorch scheduler epoch deprecation warning
          v4.2.5 — Show decoded samples every epoch instead of every 2
          v4.1.1 — Replace --gpu-id with --cpus for CPU core/thread control
          v4.1.0 — Add CLI args (--gpus, --cpus, --workers, --batch-size, --epochs)
          v4.0.3 — Fix DataParallel validation: no_grad instead of inference_mode
          v4.0.2 — Skip torch.compile with DataParallel (multi-GPU incompatible)
          v4.0.1 — Fix torch.compile + DataParallel crash during validation
          v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Initial version: 3-phase TF schedule, cosine LR, label smoothing
          v2.0.0 — Initial clean-from-scratch rewrite

Trains both "baseline" (no attention) and "attention" (Bahdanau) models.
Both models are trained with identical hyperparameters — the only difference
is the attention mechanism — providing a controlled apples-to-apples ablation.

  TEACHER FORCING STRATEGY (3-phase schedule):
    Epochs  1– 5:  TF = 1.0        — Foundation: burn in basic token representations.
                                      Loss drops from ~5.5 → ~4.25 here.
    Epochs  6–12:  TF 0.9 → 0.5    — Annealing: linear decay; decoder begins
                                      practicing self-feeding while representations
                                      are still being refined.
    Epochs 13–20:  TF = 0.5        — Maturation: hold at floor; both models
                                      fully adapt to semi-autoregressive generation.

  TF floor = 0.5 for both models. The baseline decoder has no attention to
  recover from compounding errors; TF < 0.5 causes collapse. Floor is kept
  identical for a fair comparison.

  Rationale for change from run 1:
    Run 1 kept TF=1.0 for 15 epochs, deepening exposure bias with diminishing
    returns (only ~0.13 loss reduction in epochs 6–15). The baseline's dramatic
    val_loss drop (7.89→6.00) when TF finally annealed proved the learned
    representations were valuable but locked behind exposure bias. Annealing
    from epoch 6 unlocks this 7-epoch window that was previously wasted.

  LR STRATEGY: cosine annealing with linear warmup.
    500-step warmup (LR: 0 → 3e-4), then cosine decay to 1e-5.
    Replaces ReduceLROnPlateau, which was prematurely reducing LR during the
    flat-val-loss TF=1.0 phase and wasting the LR budget.
    scheduler.step() is called per OPTIMIZER STEP (not per epoch).

  EARLY STOPPING:
    Patience = 4 epochs, monitored only from Phase 2 onward (epoch > 5).
    Val loss under autoregressive evaluation is not meaningful during Phase 1
    because the model isn't yet training for that objective.

  GRADIENT CLIPPING:
    clip = 1.0. Allows valid large gradients during TF=1.0 phase.
"""

import argparse
import os
import json
import math
import time
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, get_tf_ratio, set_seed
from dataset import build_dataloaders
from models import build_model
from gpu_utils import (setup_device, auto_scale_config, wrap_model, unwrap_model,
                       is_main_process, cleanup_ddp, GPUInfo)

# bf16 does not underflow like fp16 — GradScaler is not needed.
# torch.amp.autocast with dtype=torch.bfloat16 is sufficient.

warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler", category=UserWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_checkpoint(data: dict, path: str) -> None:
    """Atomically save a checkpoint (write .tmp then rename)."""
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _make_checkpoint(epoch, global_step, model, model_type, optimizer, scheduler,
                     val_loss, config, history, **extra) -> dict:
    """Build checkpoint dict with common fields. Extra keys via kwargs."""
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_type": model_type,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "config": dict(config),
        "history": history,
    }
    ckpt.update(extra)
    return ckpt


def _decode_samples(
    model: nn.Module,
    loader,
    device: torch.device,
    tokenizer,
    config: dict,
    model_type: str,
    epoch: int,
    n_samples: int = 8,
) -> None:
    """Decode a few validation samples and print Src / Ref / Hyp."""
    model.eval()
    sos_idx = config.get("sos_idx", 2)
    eos_idx = config.get("eos_idx", 3)
    pad_idx = config.get("pad_idx", 0)

    def _ids_to_text(ids):
        """Convert token IDs to text, stripping special tokens."""
        clean = [i for i in ids if i not in (sos_idx, eos_idx, pad_idx)]
        return tokenizer.decode(clean)

    print(f"\n  [{model_type}] Decoded samples — epoch {epoch}")

    with torch.no_grad():
        batch = next(iter(loader))
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        trg = batch["trg"].to(device)

        amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))
        _dev = device.type if hasattr(device, "type") else "cpu"
        with torch.amp.autocast(device_type=_dev, dtype=amp_dtype, enabled=_dev == "cuda"):
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

        preds = output.argmax(dim=-1)  # [B, trg_len-1]

        for i in range(min(n_samples, src.size(0))):
            src_text = _ids_to_text(src[i].tolist())
            ref_text = _ids_to_text(trg[i].tolist())
            hyp_text = _ids_to_text(preds[i].tolist())
            print(f"\n  Src : {src_text[:80]}")
            print(f"  Ref : {ref_text[:80]}")
            print(f"  Hyp : {hyp_text[:80]}")

    print()
    model.train()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: dict,
    device: torch.device,
    epoch: int,
    writer,
    global_step: int,
    scheduler,
    gpu_info: GPUInfo = None,
    train_sampler=None,
) -> Tuple[float, float, int]:
    """
    One training epoch with bf16 AMP and gradient accumulation.

    Returns:
        (avg_train_loss, avg_grad_norm, updated_global_step)

    Notes:
        - global_step = optimizer steps, NOT forward passes.
          It increments only when an optimizer step is taken (after accumulation).
        - bf16 does not underflow like fp16 — GradScaler is not needed.
        - NaN loss: batch is skipped (nan_count incremented, no backward).
        - NaN grad norm: optimizer.step() is skipped (nan_count incremented).
        - Periodic checkpoint written every 2000 optimizer steps (atomic).
    """
    model.train()
    _is_main = gpu_info is None or is_main_process(gpu_info)

    # DDP: set sampler epoch for proper shuffling
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    vocab_size: int = config["vocab_size"]
    grad_accum_steps: int = config["grad_accum_steps"]
    max_grad_norm: float = config["max_grad_norm"]
    checkpoint_dir: str = config["checkpoint_dir"]
    periodic_ckpt_steps: int = 2000   # save a step-level checkpoint every N optimizer steps
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))   # torch.bfloat16

    # Compute tf_ratio once — it is constant within an epoch.
    tf_ratio: float = get_tf_ratio(epoch, config)

    total_loss = 0.0
    total_grad_norm = 0.0
    n_updates = 0      # number of successful optimizer steps this epoch
    nan_count = 0      # batches skipped due to NaN loss or NaN grad norm
    num_batches = len(loader)

    pbar = tqdm(
        enumerate(loader),
        total=num_batches,
        desc=f"  Epoch {epoch:3d} train",
        dynamic_ncols=True,
        unit="batch",
        disable=not _is_main,
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in pbar:
        src: torch.Tensor = batch["src"].to(device)                   # [B, src_len]
        src_lengths: torch.Tensor = batch["src_lengths"].to(device)   # [B]
        trg: torch.Tensor = batch["trg"].to(device)                   # [B, trg_len]

        # ── Forward pass under bf16 autocast ───────────────────────────────
        # bf16 does not underflow like fp16 — GradScaler is not needed.
        # device.type is used dynamically so CPU debug runs don't warn/fail.
        _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        with torch.amp.autocast(device_type=_device_type, dtype=_amp_dtype,
                                enabled=_device_type == "cuda"):
            # output: [B, trg_len-1, vocab_size]
            output = model(src, src_lengths, trg, teacher_forcing_ratio=tf_ratio)

            # trg[:, 1:] excludes <sos>; output already aligns with trg[1:] steps.
            loss = criterion(
                output.reshape(-1, vocab_size),
                trg[:, 1:].reshape(-1),
            )
            # Scale loss for gradient accumulation so each micro-step contributes
            # an equal 1/grad_accum_steps share to the final gradient.
            scaled_loss = loss / grad_accum_steps

        # ── NaN loss guard ──────────────────────────────────────────────────
        if not torch.isfinite(loss):
            nan_count += 1
            # Zero accumulated gradients so this bad batch has no residual effect.
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss="NaN", nan_skip=nan_count)
            continue

        scaled_loss.backward()

        # ── Optimizer step guard (gradient accumulation) ────────────────────
        is_last_batch = (batch_idx + 1) == num_batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        if should_step:
            # Compute grad norm before clipping for logging.
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            ).item()

            # ── NaN grad norm guard ─────────────────────────────────────────
            if not math.isfinite(grad_norm):
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm="NaN", nan_skip=nan_count)
                continue

            optimizer.step()
            scheduler.step()   # cosine/warmup scheduler steps per optimizer step, not per epoch
            optimizer.zero_grad(set_to_none=True)

            # global_step = optimizer steps, NOT forward passes.
            global_step += 1

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_updates += 1

            # ── TensorBoard step-level logging (rank 0 only) ────────────────
            if _is_main and writer is not None:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/grad_norm_step", grad_norm, global_step)
                writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

            # ── Periodic checkpoint (rank 0 only, atomic) ─────────────────
            if _is_main and global_step % periodic_ckpt_steps == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{config.get('_model_type', 'model')}_step_{global_step}.pt",
                )
                _save_checkpoint({
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "train_loss_so_far": total_loss / max(n_updates, 1),
                    "tf_ratio": tf_ratio,
                }, ckpt_path)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.3f}",
                nan_skip=nan_count,
            )

    avg_train_loss = total_loss / max(n_updates, 1)
    avg_grad_norm = total_grad_norm / max(n_updates, 1)
    return avg_train_loss, avg_grad_norm, global_step


def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    gpu_info: GPUInfo = None,
) -> Tuple[float, float]:
    """
    Validation pass — teacher forcing is DISABLED (ratio=0.0).

    trg is passed only to set the decoder step count (trg.size(1)-1 steps).
    TF=0.0 means the model never sees gold tokens during the forward pass.
    This is standard practice — true autoregressive evaluation is in evaluate.py.

    Loss is accumulated only on non-padding positions because criterion has
    ignore_index=pad_idx; padding tokens contribute zero to the mean.

    Returns:
        (avg_val_loss, val_ppl) where val_ppl = exp(min(avg_val_loss, 20)).
    """
    model.eval()
    _is_main = gpu_info is None or is_main_process(gpu_info)
    _use_ddp = gpu_info is not None and gpu_info.use_ddp

    total_loss = 0.0
    n_batches = 0
    nan_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  val", unit="batch", dynamic_ncols=True,
                          leave=False, disable=not _is_main):
            src: torch.Tensor = batch["src"].to(device)
            src_lengths: torch.Tensor = batch["src_lengths"].to(device)
            trg: torch.Tensor = batch["trg"].to(device)

            _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
            with torch.amp.autocast(device_type=_device_type, dtype=amp_dtype,
                                    enabled=_device_type == "cuda"):
                output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

            vocab_size: int = output.size(-1)
            loss = criterion(
                output.reshape(-1, vocab_size),
                trg[:, 1:].reshape(-1),
            )

            if torch.isfinite(loss):
                total_loss += loss.item()
                n_batches += 1
            else:
                nan_batches += 1

    # DDP: aggregate val loss across all ranks
    if _use_ddp:
        stats = torch.tensor([total_loss, float(n_batches), float(nan_batches)], device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_loss = stats[0].item()
        n_batches = int(stats[1].item())
        nan_batches = int(stats[2].item())

    if nan_batches > 0 and _is_main:
        print(f"  ⚠ val: {nan_batches} NaN/Inf batches out of {n_batches + nan_batches}")

    avg_val_loss = total_loss / max(n_batches, 1)
    val_ppl = math.exp(min(avg_val_loss, 20))

    model.train()
    return avg_val_loss, val_ppl


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.SequentialLR]:
    """
    Build AdamW optimizer and a cosine-annealing scheduler with linear warmup.

    Schedule (both phases are chained via SequentialLR):
      1. Warmup  — LR ramps linearly from ~0 → peak (learning_rate) over
                   ``lr_warmup_steps`` optimizer steps.
      2. Cosine  — LR decays from peak → ``lr_min`` via cosine annealing
                   over the remaining (total_steps - lr_warmup_steps) steps.

    IMPORTANT: scheduler.step() must be called once per optimizer step
    (inside train_epoch), NOT once per epoch on validation loss.

    Replaces ReduceLROnPlateau, which was prematurely reducing LR during the
    flat-val-loss TF=1.0 phase and wasting the LR budget before TF annealing.

    Args:
        model:        The model whose parameters to optimise.
        config:       CONFIG dict — must contain ``learning_rate``,
                      ``weight_decay``, ``lr_warmup_steps``, ``lr_min``.
        total_steps:  Total optimizer steps for the full training run.
                      Compute as: num_epochs * (len(train_loader) // grad_accum_steps).
                      Used to size the cosine tail correctly.

    Returns:
        (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    warmup_steps: int = config.get("lr_warmup_steps", 500)
    lr_min: float = config.get("lr_min", 1e-5)
    cosine_steps: int = max(total_steps - warmup_steps, 1)

    # Linear warmup: multiply base LR by a factor that grows from
    # 1/warmup_steps → 1.0 over warmup_steps steps (≈ 0 → peak LR).
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Cosine annealing: decay from peak LR → lr_min over the remaining steps.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=lr_min,
    )

    # Chain: first warmup_steps use LinearLR, then switch to CosineAnnealingLR.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def train_model(model_type: str, config: dict, device: torch.device, gpu_info=None) -> Dict[str, List]:
    """
    Full training run for one model_type ("baseline" or "attention").

    Saves:
      {checkpoint_dir}/{model_type}_best.pt      — best val-loss checkpoint (atomic)
      {checkpoint_dir}/{model_type}_step_{n}.pt  — periodic checkpoints (atomic)
      {checkpoint_dir}/{model_type}_history.json — training history

    Args:
        gpu_info: GPUInfo from gpu_utils.setup_device(). If provided and use_ddp,
                  the model is wrapped in DDP for multi-GPU training.

    Returns:
        List of per-epoch history dicts.
    """
    config = dict(config)
    config["_model_type"] = model_type

    _is_main = gpu_info is None or is_main_process(gpu_info)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _, train_sampler = build_dataloaders(
        artifact_dir=config["artifact_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_ctx_len=config["max_ctx_tokens"],
        max_resp_len=config["max_resp_tokens"] + 2,   # +2 for <sos> and <eos>
        pad_idx=config["pad_idx"],
        max_train_samples=config.get("max_train_samples", 0),
        gpu_info=gpu_info,
    )

    # ── 1b. Tokenizer for decoded sample display ────────────────────────────
    tokenizer = None
    if _is_main:
        try:
            from tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer(config["artifact_dir"])
        except Exception as e:
            print(f"[{model_type}] Could not load tokenizer for sample decoding: {e}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model = build_model(model_type, config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _is_main:
        print(f"\n[{model_type}] Trainable parameters: {num_params:,}")

    # ── 2b. torch.compile (before DDP wrapping — PyTorch recommended order) ──
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            if _is_main:
                print(f"[{model_type}] torch.compile enabled")
        except Exception as e:
            if _is_main:
                print(f"[{model_type}] torch.compile skipped: {e}")

    # ── 2c. DDP wrapping ─────────────────────────────────────────────────────
    if gpu_info is not None:
        model = wrap_model(model, gpu_info)

    # ── 3. Optimizer + scheduler ──────────────────────────────────────────────
    # total_steps must be computed AFTER building the dataloader so we know
    # len(train_loader). Cosine annealing needs this to size the decay tail.
    total_steps = config["num_epochs"] * (len(train_loader) // config["grad_accum_steps"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    # label_smoothing=0.0 during TF=1.0 phase — smoothing fights sharp token
    # predictions that the TF schedule is designed to burn in.
    criterion = nn.CrossEntropyLoss(
        ignore_index=config["pad_idx"],
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))   # read from config

    # ── 5. TensorBoard (rank 0 only) ────────────────────────────────────────
    writer = None
    if _is_main:
        tb_dir = os.path.join(config["tensorboard_dir"], model_type)
        writer = SummaryWriter(log_dir=tb_dir)

    # ── 6. Resume — prefer last-epoch checkpoint to avoid re-running epochs ──
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_best.pt")
    last_ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_last.pt")

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0
    _patience: int = config.get("patience", 0)      # 0 = disabled; set >0 in train_mini
    _no_improve: int = 0
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "tf_ratios": [],
        "lrs": [],
    }

    # Prefer last checkpoint (has most recent epoch) over best checkpoint
    # to avoid silently discarding epochs on Colab disconnect (QA2-M1).
    resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else (
        best_ckpt_path if os.path.exists(best_ckpt_path) else None
    )
    if resume_path:
        print(f"[{model_type}] Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"[{model_type}] Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_val={best_val_loss:.4f}")

    num_epochs: int = config["num_epochs"]
    print(f"[{model_type}] Training epochs {start_epoch}–{num_epochs}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # 7a. Teacher forcing ratio for this epoch (constant within epoch).
        tf_ratio = get_tf_ratio(epoch, config)

        # 7b. Train.
        train_loss, avg_gnorm, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            scheduler=scheduler,
            gpu_info=gpu_info,
            train_sampler=train_sampler,
        )

        # 7c. Validate.
        val_loss, val_ppl = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=_amp_dtype,
            gpu_info=gpu_info,
        )

        # 7d. LR is now stepped per optimizer step inside train_epoch();
        #     no epoch-level scheduler.step() needed here.
        elapsed = time.time() - epoch_start
        lr: float = optimizer.param_groups[0]["lr"]

        # 7e. TensorBoard epoch-level logging (rank 0 only).
        if _is_main and writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/ppl", val_ppl, epoch)
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("learning_rate", lr, epoch)
            writer.add_scalar("tf_ratio", tf_ratio, epoch)

        # 7f. Save best checkpoint (rank 0 only, atomic).
        _improved = val_loss < best_val_loss
        if _improved:
            best_val_loss = val_loss
            if _is_main:
                try:
                    _git = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                    ).decode().strip()
                except Exception:
                    _git = "unknown"
                ckpt = _make_checkpoint(
                    epoch, global_step, model, model_type, optimizer, scheduler,
                    val_loss, config, history,
                    val_ppl=val_ppl, train_loss=train_loss, tf_ratio=tf_ratio,
                    git_hash=_git, torch_version=torch.__version__,
                    run_timestamp=time.strftime("%Y%m%d_%H%M%S"),
                )
                _save_checkpoint(ckpt, best_ckpt_path)

        # 7g. Update history.
        for key, val in [("train_loss", train_loss), ("val_loss", val_loss),
                         ("tf_ratios", tf_ratio), ("lrs", lr)]:
            history[key].append(val)

        # Early stopping — only monitored from Phase 2 onward.
        # During Phase 1 (TF=1.0), autoregressive val loss is not meaningful
        # because the model isn't yet training for that objective; counting
        # non-improvements there would trigger a premature stop.
        if _patience > 0 and epoch > config["tf_schedule"]["phase1_end"]:
            if _improved:
                _no_improve = 0
            else:
                _no_improve += 1
                if _no_improve >= _patience:
                    if _is_main:
                        print(f"[{model_type}] Early stopping at epoch {epoch} "
                              f"(no improvement for {_patience} epochs)")
                        _save_checkpoint(
                            _make_checkpoint(epoch, global_step, model, model_type,
                                             optimizer, scheduler, val_loss, config, history),
                            last_ckpt_path)
                    break

        # 7g2. Save last-epoch checkpoint (rank 0 only).
        if _is_main:
            _save_checkpoint(
                _make_checkpoint(epoch, global_step, model, model_type,
                                 optimizer, scheduler, val_loss, config, history),
                last_ckpt_path)

        # 7h. Epoch summary (rank 0 only).
        if _is_main:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"PPL: {val_ppl:.2f} | "
                f"LR: {lr:.2e} | "
                f"TF: {tf_ratio:.2f} | "
                f"Grad: {avg_gnorm:.3f} | "
                f"{elapsed:.1f}s"
            )

        # 7i. Decoded samples (rank 0, every epoch).
        if _is_main and tokenizer is not None:
            _decode_samples(model, val_loader, device, tokenizer, config,
                            model_type, epoch)

    if _is_main and writer is not None:
        writer.close()

    # ── 8. Save history JSON (rank 0 only, atomic write) ─────────────────
    if _is_main:
        history_path = os.path.join(checkpoint_dir, f"{model_type}_history.json")
        _tmp_hist = history_path + ".tmp"
        with open(_tmp_hist, "w") as fh:
            json.dump(history, fh, indent=2)
        os.replace(_tmp_hist, history_path)
        print(f"[{model_type}] History saved to {history_path}")

    return history


def main(cfg: dict = None, script_name: str = "train",
         cli_args: argparse.Namespace | None = None) -> None:
    """Train baseline model, then attention model.

    Args:
        cfg:         Config dict overrides. Defaults to CONFIG from config.py.
        script_name: Used for the run log filename (e.g. "train_mini").
        cli_args:    Parsed command-line arguments (--gpus, --gpu-id, etc.).
    """
    active_cfg = cfg if cfg is not None else CONFIG

    # AC2-C1: set all random seeds for full reproducibility.
    set_seed(active_cfg.get("seed", 42))

    # CLI overrides for epochs
    if cli_args and cli_args.epochs is not None:
        active_cfg = dict(active_cfg)
        active_cfg["num_epochs"] = cli_args.epochs

    # CPU core limit — caps torch threads and DataLoader workers
    if cli_args and cli_args.cpus is not None:
        torch.set_num_threads(cli_args.cpus)
        torch.set_num_interop_threads(max(1, cli_args.cpus // 2))

    device, gpu_info = setup_device(
        max_gpus=cli_args.gpus if cli_args else None,
    )
    _is_main = is_main_process(gpu_info)
    active_cfg = auto_scale_config(active_cfg, gpu_info)

    # CLI overrides for batch size and workers (after auto_scale so they take priority)
    if cli_args and cli_args.batch_size is not None:
        active_cfg["batch_size"] = cli_args.batch_size
        if _is_main:
            print(f"[cli] batch_size overridden → {cli_args.batch_size}")
    if cli_args and cli_args.workers is not None:
        active_cfg["num_workers"] = cli_args.workers
        if _is_main:
            print(f"[cli] num_workers overridden → {cli_args.workers}")
    elif cli_args and cli_args.cpus is not None:
        max_workers = min(active_cfg.get("num_workers", 8), cli_args.cpus)
        active_cfg["num_workers"] = max_workers
        if _is_main:
            print(f"[cli] CPU cores={cli_args.cpus}, num_workers={max_workers}")
    if _is_main:
        print(f"Device: {device}")

    # ── Logging and clean (rank 0 only) ──────────────────────────────────────
    ckpt_dir = active_cfg["checkpoint_dir"]
    tb_dir_root = active_cfg["tensorboard_dir"]

    if _is_main:
        from logging_utils import setup_run_logging
        setup_run_logging(script_name, log_dir=active_cfg.get("log_dir", "new/logs"))

        if cli_args and cli_args.clean:
            import glob as _glob
            removed = 0
            for pattern in [
                os.path.join(ckpt_dir, "*.pt"),
                os.path.join(ckpt_dir, "*.pt.tmp"),
                os.path.join(ckpt_dir, "*_history.json"),
                os.path.join(ckpt_dir, "run_info.json"),
            ]:
                for f in _glob.glob(pattern):
                    os.remove(f)
                    removed += 1
            for tb_sub in ["baseline", "attention"]:
                tb_path = os.path.join(tb_dir_root, tb_sub)
                if os.path.isdir(tb_path):
                    import shutil
                    shutil.rmtree(tb_path)
                    removed += 1
            print(f"[clean] Removed {removed} checkpoint/log files — fresh start")

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir_root, exist_ok=True)

        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = "unknown"
        run_info = {
            "git_hash":       git_hash,
            "python_version": sys.version,
            "torch_version":  torch.__version__,
            "cuda_version":   torch.version.cuda or "cpu",
            "run_timestamp":  time.strftime("%Y%m%d_%H%M%S"),
            "seed":           active_cfg.get("seed", 42),
            "device":         str(device),
            "ddp":            gpu_info.use_ddp,
            "world_size":     gpu_info.world_size,
        }
        run_info_path = os.path.join(ckpt_dir, "run_info.json")
        with open(run_info_path, "w") as fh:
            json.dump(run_info, fh, indent=2)
        print(f"Run info saved → {run_info_path}")
        print(f"  git: {git_hash}  |  torch: {torch.__version__}  |  seed: {run_info['seed']}")

    # DDP barrier: ensure rank 0 has created dirs before other ranks proceed
    if gpu_info.use_ddp:
        torch.distributed.barrier()

    # ── Baseline ─────────────────────────────────────────────────────────────
    if _is_main:
        print("\n" + "=" * 70)
        print("  TRAINING: baseline (no attention)")
        print("=" * 70)
    baseline_history = train_model("baseline", active_cfg, device, gpu_info)

    # ── Attention ─────────────────────────────────────────────────────────────
    if _is_main:
        print("\n" + "=" * 70)
        print("  TRAINING: attention (Bahdanau)")
        print("=" * 70)
    attention_history = train_model("attention", active_cfg, device, gpu_info)

    if _is_main:
        print("\nTraining complete.")
        print(f"  Baseline  best val loss : {min(baseline_history['val_loss']):.4f}")
        print(f"  Attention best val loss : {min(attention_history['val_loss']):.4f}")

    # ── DDP cleanup ──────────────────────────────────────────────────────────
    cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Seq2Seq chatbot models")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs. 0=CPU, 1=single GPU, 2+=DDP multi-GPU.")
    parser.add_argument("--cpus", type=int, default=None,
                        help="Number of CPU cores to use. Caps workers and torch threads.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of DataLoader workers (overrides auto-detect).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override per-GPU batch size.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing checkpoints before training (fresh start).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-launch DDP via torchrun when --gpus > 1
    _in_ddp = "RANK" in os.environ
    if not _in_ddp and args.gpus is not None and args.gpus > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={args.gpus}"] + sys.argv
        sys.exit(subprocess.call(cmd))

    main(cli_args=args)
