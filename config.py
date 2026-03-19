"""
config.py — Single source of truth for all hyperparameters and paths.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — 3-phase TF schedule, cosine LR with warmup, label smoothing
          v3.0.0 — Jupyter notebook paths, multi-GPU batch scaling
          v2.0.0 — Initial clean-from-scratch rewrite

Every other file imports from here. No magic numbers anywhere else.

Implementation notes:
- CONFIG is a plain dict (not a dataclass) for easy JSON serialisation
  and Colab cell override via CONFIG.update({...}).
- All paths are computed relative to this file's location (new/) so the
  code works correctly regardless of which directory you run from —
  project root, new/, or any other working directory.
- Notebooks may still override paths with absolute Drive paths, but it
  is no longer required.
"""

import json
import os
from pathlib import Path

# ── Tokenization ──────────────────────────────────────────────────────────────
_TOKENIZATION = {
    "vocab_size":            16000,   # SentencePiece BPE vocabulary size
    "spm_vocab_size":        16000,   # alias used by SPM training scripts
    "embed_dim":             300,     # token embedding dimensionality
    # Special token IDs — must match SentencePieceTrainer pad_id/bos_id/eos_id
    "pad_idx":               0,       # <pad>
    "unk_idx":               1,       # <unk>
    "sos_idx":               2,       # <sos> / bos
    "eos_idx":               3,       # <eos>
}

# ── Architecture ──────────────────────────────────────────────────────────────
_ARCHITECTURE = {
    "enc_hidden_dim":        512,     # per-direction; effective = 1024 (bidirectional)
    "dec_hidden_dim":        1024,    # matches bidirectional encoder output
    "projection_dim":        512,     # encoder → decoder state projection size
    "attn_dim":              256,     # Bahdanau attention hidden size
    "num_layers":            2,       # stacked LSTM layers in encoder and decoder
    "shared_embeddings":     True,    # encoder and decoder share the embedding matrix
    # Dropout — kept together so a single config change tunes all three.
    "dropout_embed":         0.3,     # embedding dropout (light: preserve FastText signal)
    "dropout_lstm":          0.5,     # inter-layer LSTM dropout (aggressive regularisation)
    "dropout_out":           0.4,     # output/projection dropout
}

# ── Training ──────────────────────────────────────────────────────────────────
_TRAINING = {
    "learning_rate":         3e-4,    # peak LR (reached after warmup; cosine decays from here)
    "weight_decay":          1e-5,    # L2 regularisation
    "max_grad_norm":         1.0,     # gradient clipping threshold (gentler than 0.5)
    "batch_size":            1024,    # per-step batch size (A100 80GB has ample VRAM)
    "grad_accum_steps":      1,       # effective batch = batch_size × 1 = 1024
    "num_epochs":            20,      # total training epochs
    "amp_dtype":             "bfloat16",  # automatic mixed precision dtype
    "patience":              4,       # early stopping patience (0 = disabled); monitoring
                                      # begins only after Phase 1 ends (see get_tf_ratio)
}

# ── Teacher-Forcing Schedule ───────────────────────────────────────────────────
# 3-phase schedule with gradual annealing to address exposure bias.
#
# Insight from run 1: epochs 1–5 under TF=1.0 produce large loss drops (~5.5→4.25).
# Epochs 6–15 under TF=1.0 yield only ~0.13 additional loss reduction while
# deepening exposure bias. The dramatic val_loss improvement at TF=0.8/0.5 in
# the baseline (7.89→6.00) confirms the representations are good — the decoder
# just needs to practice self-feeding earlier.
#
# Phase 1 — Foundation  (epochs  1– 5): TF = 1.0        (burn in representations)
# Phase 2 — Annealing   (epochs  6–12): TF 0.9 → 0.5    (linear decay, per-epoch)
# Phase 3 — Maturation  (epochs 13–20): TF = 0.5         (floor; both models stabilise)
#
# TF floor is 0.5 for both models. The baseline decoder has no attention to recover
# from compounding errors; TF < 0.5 causes collapse. Floor is kept identical for
# a fair apples-to-apples ablation.
_TF_SCHEDULE = {
    "tf_schedule": {
        "phase1_end":       5,    # last epoch (inclusive) at TF = phase1_tf
        "phase1_tf":        1.0,  # TF ratio for epochs 1 → phase1_end
        "phase2_end":       12,   # last epoch (inclusive) of linear annealing
        "phase2_start_tf":  0.9,  # TF at the first epoch of phase 2 (phase1_end + 1)
        "phase2_end_tf":    0.5,  # TF at the last epoch of phase 2 (phase2_end)
        "phase3_tf":        0.5,  # floor TF held for all remaining epochs
    },
}

# ── LR Scheduler ──────────────────────────────────────────────────────────────
# Cosine annealing with linear warmup (replaces ReduceLROnPlateau).
#
# Rationale: ReduceLROnPlateau was premature-triggering during the old TF=1.0 phase
# (flat/climbing val loss) and wasting the LR budget before TF annealing where
# the model most needs to adapt. A cosine schedule is deterministic and
# well-matched to a fixed epoch budget; warmup avoids large initial gradients.
#
# scheduler.step() is called once per OPTIMIZER STEP (not per epoch) in train_epoch().
_LR_SCHEDULER = {
    "lr_warmup_steps": 500,   # linear warmup: LR ramps from ~0 → peak over this many steps
    "lr_min":          1e-5,  # cosine annealing floor (tail of decay)
}

# ── Loss ──────────────────────────────────────────────────────────────────────
# Label smoothing is ON at 0.1.
# With the new TF schedule, TF=1.0 only lasts 5 epochs (not 15), so the model
# spends most of training in the annealing/maturation phases where preventing
# overconfident token predictions directly improves autoregressive generation.
_LOSS = {
    "label_smoothing":       0.1,
}

# ── Data ──────────────────────────────────────────────────────────────────────
_DATA = {
    "max_ctx_tokens":        100,  # max tokens per context (encoder input, no <sos>)
    "max_ctx_turns":         8,    # max dialogue turns retained in context
    "max_resp_tokens":       40,   # max response tokens (excl. <sos>/<eos>);
                                   # padded length = 42 (includes <sos> + <eos>)
    "num_workers":           0,    # DataLoader workers; Colab can override to 4
}

# ── Path root (always resolves to the new/ directory, wherever you run from) ──
_NEW_DIR = Path(__file__).resolve().parent   # .../NLP_Final_Project_v2/new/

# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute paths computed from this file's location — work from any cwd.
# Notebooks may still override with Drive paths via CONFIG.update({...}).
_PATHS = {
    "artifact_dir":           str(_NEW_DIR / "artifacts"),
    "checkpoint_dir":         str(_NEW_DIR / "checkpoints"),
    "tensorboard_dir":        str(_NEW_DIR / "tb_logs"),
    "log_dir":                str(_NEW_DIR / "logs"),
    "spm_model_path":         str(_NEW_DIR / "artifacts" / "stage5_spm.model"),
    "embedding_matrix_path":  str(_NEW_DIR / "artifacts" / "stage8_embedding_matrix.npy"),
}

# ── Chat Inference ────────────────────────────────────────────────────────────
_CHAT = {
    "chat_max_history_turns": 10,   # turns kept in sliding context window
    "top_p":                  0.9,  # nucleus sampling cumulative probability
    "temperature":            0.8,  # softmax temperature for sampling
    "ngram_block":            3,    # block repeated n-grams during decoding
    "max_decode_len":         40,   # maximum tokens generated per response
}

# ── Reproducibility ───────────────────────────────────────────────────────────
_REPRODUCIBILITY = {
    "seed": 42,   # global random seed for torch, numpy, python random, DataLoader
}

# ── Master CONFIG ─────────────────────────────────────────────────────────────
CONFIG: dict = {
    **_TOKENIZATION,
    **_ARCHITECTURE,
    **_TRAINING,
    **_TF_SCHEDULE,
    **_LR_SCHEDULER,
    **_LOSS,
    **_DATA,
    **_PATHS,
    **_CHAT,
    **_REPRODUCIBILITY,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility (AC2-C1).

    Must be called at the top of train.py main() and evaluate.py main()
    to ensure results are reproducible across runs.

    Note: DataLoader worker seeds are controlled separately via
    ``worker_init_fn`` when ``num_workers > 0``.
    """
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = False
    _torch.backends.cudnn.benchmark = True   # auto-tune LSTM/conv kernels for ~10% speedup

def get_tf_ratio(epoch: int, config: dict) -> float:
    """Return the teacher-forcing ratio for a given 1-indexed epoch.

    Three phases:
      Phase 1 (epochs 1 → phase1_end):         TF = phase1_tf  (constant, full TF)
      Phase 2 (epochs phase1_end+1 → phase2_end): TF decays linearly from
                                                  phase2_start_tf → phase2_end_tf
      Phase 3 (epochs phase2_end+1 → end):     TF = phase3_tf  (constant floor)

    The phase-2 decay uses the formula from the training spec:
        tf = phase2_start_tf
             - (epoch - phase2_start_epoch)
             * (phase2_start_tf - phase2_end_tf) / (phase2_end - phase2_start_epoch)

    Examples::

        >>> get_tf_ratio(1,  CONFIG)   # → 1.0   (phase 1)
        >>> get_tf_ratio(5,  CONFIG)   # → 1.0   (phase 1, last epoch)
        >>> get_tf_ratio(6,  CONFIG)   # → 0.9   (phase 2, first epoch)
        >>> get_tf_ratio(9,  CONFIG)   # → 0.7   (phase 2, midpoint)
        >>> get_tf_ratio(12, CONFIG)   # → 0.5   (phase 2, last epoch)
        >>> get_tf_ratio(13, CONFIG)   # → 0.5   (phase 3)
        >>> get_tf_ratio(20, CONFIG)   # → 0.5   (phase 3)

    Args:
        epoch:  Current training epoch, **1-indexed**.
        config: Config dict containing a ``"tf_schedule"`` sub-dict with keys
                ``phase1_end``, ``phase1_tf``, ``phase2_end``,
                ``phase2_start_tf``, ``phase2_end_tf``, and ``phase3_tf``.

    Returns:
        Teacher-forcing ratio as a float in [0, 1].
    """
    schedule = config["tf_schedule"]
    if epoch <= schedule["phase1_end"]:
        return schedule["phase1_tf"]
    if epoch <= schedule["phase2_end"]:
        # Linear interpolation: phase2_start_tf → phase2_end_tf over
        # the epochs in [phase1_end+1, phase2_end].
        phase2_start_epoch = schedule["phase1_end"] + 1
        tf_range = schedule["phase2_start_tf"] - schedule["phase2_end_tf"]
        step_range = schedule["phase2_end"] - phase2_start_epoch   # denominator
        tf = schedule["phase2_start_tf"] - (epoch - phase2_start_epoch) * (
            tf_range / step_range
        )
        return tf
    return schedule["phase3_tf"]


if __name__ == "__main__":
    print(json.dumps(CONFIG, indent=2))
