"""
dataset.py — PyTorch Dataset and DataLoader for BPE-tokenised Ubuntu pairs.

Version : 4.1.0
Modified: 2026-03-19
Changes : v4.1.0 — Add DDP support: DistributedSampler, return train_sampler
          v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — JSONL format, dynamic padding, src_lengths support
          v2.0.0 — Initial clean-from-scratch rewrite

Reads the JSONL files produced by phase1.py Stage 6.
Each line: {"ctx": [int, ...], "resp": [int, ...]}

Key design decisions:
  - All splits loaded fully into memory on __init__ (1.5M pairs ≈ 400 MB total,
    well within typical RAM budgets; avoids per-epoch file I/O overhead)
  - collate_fn returns a plain dict (not a tuple) for readability in train.py
  - num_workers=0 default for macOS safety; set to 4 on Linux/Colab
"""

import json
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class UbuntuPairDataset(Dataset):
    """
    Loads BPE-tokenised Ubuntu dialogue pairs from a JSONL file produced by
    phase1.py Stage 6.  All pairs are held in memory; at ~50 IDs per pair and
    8 bytes per int, 1.5 M pairs occupy roughly 400 MB — comfortably feasible
    on any modern training machine or Colab instance.
    """

    def __init__(self, jsonl_path: str, max_ctx_len: int, max_resp_len: int) -> None:
        """
        Args:
            jsonl_path:   Path to stage6_{split}_ids.jsonl.
            max_ctx_len:  Hard truncation length for context sequences.
            max_resp_len: Hard truncation length for response sequences
                          (should include <sos> and <eos>, i.e. max_resp_tokens + 2).
        """
        self.pairs: List[Dict[str, List[int]]] = []
        self.skipped_count: int = 0

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Keep the LAST max_ctx_len tokens (most recent dialogue turns),
                    # matching the right-end truncation applied in phase1 stage6.
                    raw_ctx = record["ctx"]
                    ctx  = raw_ctx[-max_ctx_len:] if len(raw_ctx) > max_ctx_len else raw_ctx
                    resp = record["resp"][:max_resp_len]
                except (json.JSONDecodeError, KeyError):
                    self.skipped_count += 1
                    warnings.warn(f"Malformed line {lineno} in {jsonl_path} — skipping.")
                    continue

                if not ctx or not resp:
                    self.skipped_count += 1
                    warnings.warn(
                        f"Empty ctx or resp after truncation at line {lineno} "
                        f"in {jsonl_path} — skipping."
                    )
                    continue

                self.pairs.append({"ctx": ctx, "resp": resp})

        print(f"Loaded {len(self.pairs)} pairs from {jsonl_path} (skipped {self.skipped_count})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Return {"ctx": List[int], "resp": List[int]}."""
        return self.pairs[idx]


def collate_fn(batch: List[Dict[str, List[int]]], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of variable-length pairs to the longest sequence in that batch.

    No sort needed — Encoder uses enforce_sorted=False.

    Returns:
        src         — LongTensor [batch, max_ctx_len_in_batch]   padded context IDs
        src_lengths — LongTensor [batch]                         actual ctx lengths
        trg         — LongTensor [batch, max_resp_len_in_batch]  padded response IDs
        trg_lengths — LongTensor [batch]                         actual resp lengths
    """
    ctxs = [torch.tensor(item["ctx"], dtype=torch.long) for item in batch]
    resps = [torch.tensor(item["resp"], dtype=torch.long) for item in batch]

    src_lengths = torch.tensor([len(c) for c in ctxs], dtype=torch.long)
    trg_lengths = torch.tensor([len(r) for r in resps], dtype=torch.long)

    src = pad_sequence(ctxs, batch_first=True, padding_value=pad_idx)
    trg = pad_sequence(resps, batch_first=True, padding_value=pad_idx)

    return {
        "src":         src,
        "src_lengths": src_lengths,
        "trg":         trg,
        "trg_lengths": trg_lengths,
    }


def build_dataloaders(
    artifact_dir: str,
    batch_size: int = 256,
    num_workers: int = 0,
    max_ctx_len: int = 100,
    max_resp_len: int = 42,    # 40 tokens + <sos> + <eos>
    pad_idx: int = 0,
    max_train_samples: int = 0,  # 0 = use all; set >0 to subsample train split
    gpu_info=None,               # GPUInfo for DDP sampler
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Build train / val / test DataLoaders from Phase 1 Stage 6 artifacts.

    All three splits are loaded fully into memory via UbuntuPairDataset.
    Files expected in artifact_dir:
        stage6_train_ids.jsonl, stage6_val_ids.jsonl, stage6_test_ids.jsonl

    When gpu_info.use_ddp is True, a DistributedSampler is created for the
    train and val splits. The train sampler is returned so the caller can
    call sampler.set_epoch(epoch) each epoch for proper shuffling.

    Args:
        artifact_dir:       Directory containing stage6_*_ids.jsonl files.
        batch_size:         Samples per batch (per-GPU in DDP mode).
        num_workers:        DataLoader worker processes.
        max_ctx_len:        Hard truncation for context sequences.
        max_resp_len:       Hard truncation for response sequences (incl. <sos>/<eos>).
        pad_idx:            Padding token ID (must match config pad_idx, typically 0).
        max_train_samples:  If > 0, randomly subsample train split.
        gpu_info:           GPUInfo from gpu_utils for DDP sampler creation.

    Returns:
        (train_loader, val_loader, test_loader, train_sampler)
        train_sampler is None when not using DDP.
    """
    import random as _random
    artifact_dir = Path(artifact_dir)
    _collate = partial(collate_fn, pad_idx=pad_idx)

    train_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_train_ids.jsonl"), max_ctx_len, max_resp_len
    )

    # Subsample train if requested (mini runs for fast iteration).
    if max_train_samples > 0 and len(train_ds) > max_train_samples:
        indices = _random.Random(42).sample(range(len(train_ds)), max_train_samples)
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, indices)
        print(f"  [mini] Train subsampled: {max_train_samples:,} of {len(train_ds.dataset):,} pairs (seed=42)")

    val_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_val_ids.jsonl"), max_ctx_len, max_resp_len
    )
    test_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_test_ids.jsonl"), max_ctx_len, max_resp_len
    )

    pin = torch.cuda.is_available()
    persist = num_workers > 0

    # ── DDP samplers ─────────────────────────────────────────────────────
    train_sampler = None
    val_sampler = None
    use_ddp = gpu_info is not None and getattr(gpu_info, "use_ddp", False)

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=gpu_info.world_size,
            rank=gpu_info.rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=gpu_info.world_size,
            rank=gpu_info.rank,
            shuffle=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # sampler handles shuffling in DDP
        drop_last=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        sampler=val_sampler,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
    )

    return train_loader, val_loader, test_loader, train_sampler
