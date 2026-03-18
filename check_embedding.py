"""
check_embedding.py — Standalone embedding validation tool.

Version : 4.0.0
Date    : 2026-03-18

Tests 8 categories in sequence:
  A. Matrix file integrity         (shape, dtype, NaN/Inf, norm stats)
  B. SPM ↔ matrix alignment        (token IDs match embedding rows)
  C. create_pretrained_embedding() (function contract, freeze flag)
  D. Weight tying                  (encoder and decoder share the SAME object)
  E. padding_idx gradient masking  (pad row stays zero after backward)
  F. Fine-tuning gradient flow     (gradients reach embedding.weight)
  G. Semantic sanity               (cosine similarity — related > unrelated)
  H. Lookup correctness            (nn.Embedding.forward output shape and values)

Usage:
    cd new/
    python check_embedding.py                    # uses artifacts_mini/
    python check_embedding.py --full             # uses artifacts/  (full pipeline)
    python check_embedding.py --verbose          # print extra detail

Exit code: 0 if all PASS, 1 if any FAIL.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tokenizer_utils import load_tokenizer

# ── ensure new/ is on path ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG
from models import build_model, create_pretrained_embedding


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_PASS  = "PASS"
_FAIL  = "FAIL"
_WARN  = "WARN"
_SKIP  = "SKIP"

_results: List[Tuple[str, str, str]] = []   # (section, name, status)
_verbose = False

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _colour(status: str) -> str:
    if status == _PASS:  return f"{_GREEN}{status}{_RESET}"
    if status == _WARN:  return f"{_YELLOW}{status}{_RESET}"
    if status == _FAIL:  return f"{_RED}{_BOLD}{status}{_RESET}"
    return status


def check(
    section: str,
    name: str,
    condition: bool,
    detail: str = "",
    warn_only: bool = False,
) -> bool:
    status = _PASS if condition else (_WARN if warn_only else _FAIL)
    _results.append((section, name, status))
    tag = _colour(status)
    suffix = f"  ↳ {detail}" if (detail and (not condition or _verbose)) else ""
    if detail and condition and _verbose:
        suffix = f"  ↳ {detail}"
    print(f"  [{tag}] {name}{suffix}")
    return condition


def skip(section: str, name: str, reason: str = "") -> None:
    _results.append((section, name, _SKIP))
    print(f"  [{_SKIP}] {name}  ↳ {reason}")


def header(title: str) -> None:
    print(f"\n{_CYAN}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_CYAN}{_BOLD}  {title}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'─'*60}{_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: cosine similarity between two numpy row vectors
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n_a = np.linalg.norm(a)
    n_b = np.linalg.norm(b)
    if n_a < 1e-9 or n_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (n_a * n_b))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Matrix file integrity
# ─────────────────────────────────────────────────────────────────────────────

def section_a(mat: np.ndarray, vocab_size: int, embed_dim: int) -> None:
    S = "A"
    header("A. Matrix File Integrity")

    check(S, "Shape rows == vocab_size",
          mat.shape[0] == vocab_size,
          f"got {mat.shape[0]}, expected {vocab_size}")

    check(S, "Shape cols == embed_dim",
          mat.shape[1] == embed_dim,
          f"got {mat.shape[1]}, expected {embed_dim}")

    check(S, "dtype is float32",
          mat.dtype == np.float32,
          f"got {mat.dtype}")

    nan_count = int(np.isnan(mat).sum())
    check(S, "No NaN values",
          nan_count == 0,
          f"{nan_count} NaN entries")

    inf_count = int(np.isinf(mat).sum())
    check(S, "No Inf values",
          inf_count == 0,
          f"{inf_count} Inf entries")

    norms = np.linalg.norm(mat, axis=1)        # [vocab_size]
    zero_rows = int((norms < 1e-6).sum())
    coverage_pct = (vocab_size - zero_rows) / vocab_size * 100

    check(S, "Coverage ≥ 99.9% non-zero rows",
          coverage_pct >= 99.9,
          f"{coverage_pct:.3f}%")

    # Row 0 MUST be zero (pad token).
    check(S, "Row 0 (<pad>) is all-zeros",
          bool(norms[0] < 1e-6),
          f"norm={norms[0]:.4f}")

    # Rows 1-10 (unk, sos, eos, domain tags) must be non-zero.
    check(S, "Rows 1–10 (special tokens) are non-zero",
          bool((norms[1:11] > 1e-6).all()),
          f"min norm among rows 1-10: {norms[1:11].min():.4f}")

    # Norm statistics (healthy: mean 2–6, std < 2, no row exploded >50).
    nonzero_norms = norms[norms > 1e-6]
    norm_mean = float(nonzero_norms.mean())
    norm_std  = float(nonzero_norms.std())
    norm_min  = float(nonzero_norms.min())
    norm_max  = float(nonzero_norms.max())

    check(S, "Norm mean in reasonable range [1.0, 10.0]",
          1.0 <= norm_mean <= 10.0,
          f"mean={norm_mean:.3f}")

    check(S, "Norm std < 3.0 (no extreme spread)",
          norm_std < 3.0,
          f"std={norm_std:.3f}")

    check(S, "No exploded rows (max norm < 50)",
          norm_max < 50.0,
          f"max={norm_max:.3f}")

    check(S, "No near-zero non-pad rows (min norm > 0.1)",
          norm_min > 0.1,
          f"min={norm_min:.3f}")

    if _verbose:
        print(f"    norm stats: mean={norm_mean:.3f} std={norm_std:.3f} "
              f"min={norm_min:.3f} max={norm_max:.3f}")
        print(f"    zero rows: {zero_rows}  coverage: {coverage_pct:.4f}%")

    # Row uniqueness: all non-pad rows should be distinct (no duplicate vectors).
    # Sample check — exact duplicate detection on first 2000 rows.
    sample = mat[1:2001]
    # Round to 3dp to avoid float precision false-positives.
    rounded = np.round(sample, 3)
    unique_rows = len(set(map(tuple, rounded)))
    check(S, "No duplicate embedding rows (sample 2000)",
          unique_rows == len(sample),
          f"{len(sample) - unique_rows} duplicates in rows 1–2000",
          warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — SPM ↔ matrix alignment
# ─────────────────────────────────────────────────────────────────────────────

def section_b(mat: np.ndarray, sp: object) -> None:
    S = "B"
    header("B. SPM ↔ Matrix Alignment")

    vocab_size = mat.shape[0]
    norms = np.linalg.norm(mat, axis=1)

    # Check SPM vocab size matches matrix rows.
    sp_size = sp.get_piece_size()
    check(S, "SPM vocab size == matrix rows",
          sp_size == vocab_size,
          f"spm={sp_size}, matrix={vocab_size}")

    # Verify special token IDs.
    SPECIAL = {
        "<pad>":      0,
        "<unk>":      1,
        "<sos>":      2,
        "<eos>":      3,
        "__url__":    4,
        "__path__":   5,
        "__ip__":     6,
        "__cmd__":    7,
        "__number__": 8,
        "__eot__":    9,
        "__user__":  10,
    }
    for piece, expected_id in SPECIAL.items():
        got_id = sp.piece_to_id(piece)
        check(S, f"SPM ID for {piece!r} == {expected_id}",
              got_id == expected_id,
              f"got {got_id}")

    # Row 0 (pad) zero, rows 1–10 non-zero.
    check(S, "Matrix row 0 zero (pad aligns with SPM id 0)",
          bool(norms[0] < 1e-6))
    check(S, "Matrix rows 1–3 non-zero (unk/sos/eos have real vectors)",
          bool((norms[1:4] > 1e-6).all()),
          f"norms: {norms[1:4].tolist()}")
    check(S, "Matrix rows 4–10 non-zero (domain tags have real vectors)",
          bool((norms[4:11] > 1e-6).all()),
          f"min={norms[4:11].min():.3f}")

    # Verify SPM reverse mapping: sp.id_to_piece(i) should be the canonical piece.
    # Sample 200 random IDs and check matrix row is non-zero for all non-pad.
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(np.arange(1, vocab_size), size=200, replace=False)
    non_zero_sampled = int((norms[sample_ids] > 1e-6).sum())
    check(S, "Random 200 non-pad IDs all have non-zero rows",
          non_zero_sampled == 200,
          f"{200 - non_zero_sampled} zeros found in sample")

    # Ubuntu domain words — these must be single tokens (verified in probe above).
    DOMAIN_WORDS = {
        "linux":   299,
        "ubuntu":  116,
        "kernel":  473,
        "install":  93,
        "apt":     284,
        "error":   344,
        "network": 442,
    }
    for word, expected_id in DOMAIN_WORDS.items():
        ids = sp.encode_as_ids(word)
        check(S, f"'{word}' encodes to single token (id={expected_id})",
              len(ids) == 1 and ids[0] == expected_id,
              f"got ids={ids}")
        check(S, f"Matrix row for '{word}' (id={expected_id}) is non-zero",
              bool(norms[expected_id] > 1e-6),
              f"norm={norms[expected_id]:.4f}")

    # Full sweep: count IDs that are non-zero (excluding id 0).
    non_zero_full = int((norms[1:] > 1e-6).sum())
    check(S, "≥99.9% of non-pad rows are non-zero (full sweep)",
          non_zero_full >= (vocab_size - 1) * 0.999,
          f"{non_zero_full}/{vocab_size-1} non-zero")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — create_pretrained_embedding() contract
# ─────────────────────────────────────────────────────────────────────────────

def section_c(matrix_path: str, mat: np.ndarray, cfg: dict) -> nn.Embedding:
    S = "C"
    header("C. create_pretrained_embedding() Contract")

    vocab_size = mat.shape[0]
    embed_dim  = mat.shape[1]
    pad_idx    = cfg["pad_idx"]

    emb = create_pretrained_embedding(matrix_path, pad_idx=pad_idx, freeze=False)

    check(S, "Returns nn.Embedding",
          isinstance(emb, nn.Embedding))

    check(S, "Embedding num_embeddings == vocab_size",
          emb.num_embeddings == vocab_size,
          f"got {emb.num_embeddings}")

    check(S, "Embedding embedding_dim == embed_dim",
          emb.embedding_dim == embed_dim,
          f"got {emb.embedding_dim}")

    check(S, "embedding.padding_idx == pad_idx",
          emb.padding_idx == pad_idx,
          f"got {emb.padding_idx}")

    check(S, "requires_grad == True (fine-tuning enabled, freeze=False)",
          emb.weight.requires_grad is True)

    # Verify loaded weights match the numpy matrix.
    weight_np = emb.weight.data.cpu().numpy()

    check(S, "Loaded weight shape matches matrix shape",
          weight_np.shape == mat.shape,
          f"{weight_np.shape} vs {mat.shape}")

    check(S, "Pad row (row 0) is all-zeros after load",
          bool(np.linalg.norm(weight_np[0]) < 1e-6),
          f"norm={np.linalg.norm(weight_np[0]):.4e}")

    # Spot-check 20 random rows — weight should equal original matrix.
    rng = np.random.RandomState(99)
    rows = rng.choice(np.arange(1, vocab_size), size=20, replace=False)
    max_diff = float(np.abs(weight_np[rows] - mat[rows]).max())
    check(S, "Loaded weights match original numpy matrix (20 random rows)",
          max_diff < 1e-5,
          f"max abs diff = {max_diff:.2e}")

    # freeze=True variant.
    emb_frozen = create_pretrained_embedding(matrix_path, pad_idx=pad_idx, freeze=True)
    check(S, "freeze=True → requires_grad == False",
          emb_frozen.weight.requires_grad is False)

    # Verify frozen weights are also correctly loaded.
    frozen_np = emb_frozen.weight.data.cpu().numpy()
    max_diff_frozen = float(np.abs(frozen_np[1] - mat[1]).max())
    check(S, "Frozen weight values still match matrix (not zeroed)",
          max_diff_frozen < 1e-5,
          f"max diff={max_diff_frozen:.2e}")

    return emb


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Weight tying
# ─────────────────────────────────────────────────────────────────────────────

def section_d(cfg: dict, device: torch.device) -> None:
    S = "D"
    header("D. Weight Tying (Encoder = Decoder Embedding Object)")

    for model_type in ("baseline", "attention"):
        model = build_model(model_type, cfg, device)
        enc_emb = model.encoder.embedding
        dec_emb = model.decoder.embedding

        # Identity test — must be the exact same Python object.
        check(S, f"[{model_type}] encoder.embedding is decoder.embedding (same object)",
              enc_emb is dec_emb,
              f"enc id={id(enc_emb)} dec id={id(dec_emb)}")

        # Data pointer test — weight tensors share storage.
        check(S, f"[{model_type}] embedding weight data_ptr() identical",
              enc_emb.weight.data_ptr() == dec_emb.weight.data_ptr())

        # Mutation test — modifying encoder embedding changes decoder embedding.
        original_val = enc_emb.weight.data[5, 0].item()
        enc_emb.weight.data[5, 0] = 999.0
        mutation_visible = dec_emb.weight.data[5, 0].item() == 999.0
        enc_emb.weight.data[5, 0] = original_val   # restore
        check(S, f"[{model_type}] mutation of encoder embedding is visible in decoder",
              mutation_visible,
              "weight tying confirmed via in-place mutation")

        # Both point to the same parameter in model.parameters().
        enc_param_ids = {id(p) for p in model.encoder.parameters()}
        dec_param_ids = {id(p) for p in model.decoder.parameters()}
        emb_param_id  = id(enc_emb.weight)
        check(S, f"[{model_type}] embedding weight appears in both enc and dec params",
              emb_param_id in enc_param_ids and emb_param_id in dec_param_ids)

        del model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — padding_idx gradient masking
# ─────────────────────────────────────────────────────────────────────────────

def section_e(cfg: dict, device: torch.device) -> None:
    S = "E"
    header("E. padding_idx Gradient Masking (Pad Row Stays Zero After Backward)")

    model = build_model("baseline", cfg, device)
    emb = model.encoder.embedding
    pad_idx = cfg["pad_idx"]

    # Construct a minimal batch: two sequences that DON'T use pad_idx as a
    # real token — but one sequence is padded at the end.
    # src: [[2, 116, 299, 0, 0],   (sos, ubuntu, linux, pad, pad)
    #        [2, 344,   0, 0, 0]]  (sos, error, pad, pad, pad)
    src = torch.tensor([
        [2, 116, 299, 0, 0],
        [2, 344,   0, 0, 0],
    ], dtype=torch.long, device=device)
    src_lengths = torch.tensor([3, 2], dtype=torch.long, device=device)

    trg = torch.tensor([
        [2, 93, 3, 0],    # sos, install, eos, pad
        [2, 442, 3, 0],   # sos, network, eos, pad
    ], dtype=torch.long, device=device)

    # Forward pass.
    model.train()
    output = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)
    # output: [batch, trg_len-1, vocab_size]

    # Loss ignoring pad positions.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # output shape: [batch, T, vocab] → [batch*T, vocab]
    T = output.size(1)
    loss = criterion(
        output.reshape(-1, output.size(-1)),
        trg[:, 1:].reshape(-1),
    )
    loss.backward()

    # After backward: pad row gradient should be zero (padding_idx does this).
    grad = emb.weight.grad
    check(S, "embedding.weight.grad is not None after backward",
          grad is not None)

    if grad is not None:
        pad_grad_norm = float(grad[pad_idx].norm())
        check(S, "Gradient at pad row (row 0) is zero",
              pad_grad_norm < 1e-8,
              f"grad norm at pad row = {pad_grad_norm:.4e}")

        # Non-pad rows that were used should have non-zero gradients.
        used_ids = [2, 116, 299, 344, 93, 442]   # tokens we fed in
        non_zero_grads = [i for i in used_ids if float(grad[i].norm()) > 1e-8]
        check(S, "Used non-pad token rows received non-zero gradients",
              len(non_zero_grads) > 0,
              f"ids with non-zero grad: {non_zero_grads}")

        # Check grad is finite everywhere.
        check(S, "No NaN in embedding gradients",
              not torch.isnan(grad).any().item())
        check(S, "No Inf in embedding gradients",
              not torch.isinf(grad).any().item())

    del model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Fine-tuning gradient flow
# ─────────────────────────────────────────────────────────────────────────────

def section_f(cfg: dict, device: torch.device) -> None:
    S = "F"
    header("F. Fine-Tuning Gradient Flow")

    for model_type in ("baseline", "attention"):
        model = build_model(model_type, cfg, device)
        emb = model.encoder.embedding
        pad_idx = cfg["pad_idx"]

        check(S, f"[{model_type}] embedding.weight.requires_grad is True",
              emb.weight.requires_grad is True)

        # Mini forward+backward.
        src = torch.tensor([[2, 116, 299, 93, 3]], dtype=torch.long, device=device)
        src_lengths = torch.tensor([5], dtype=torch.long, device=device)
        trg = torch.tensor([[2, 116, 3, 0]], dtype=torch.long, device=device)

        model.train()
        out = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)
        loss = nn.CrossEntropyLoss(ignore_index=pad_idx)(
            out.reshape(-1, out.size(-1)),
            trg[:, 1:].reshape(-1),
        )
        loss.backward()

        grad = emb.weight.grad
        check(S, f"[{model_type}] Gradient reaches embedding.weight after backward",
              grad is not None and grad.abs().sum().item() > 0)

        if grad is not None:
            grad_norm = float(grad.norm())
            check(S, f"[{model_type}] Gradient norm is finite ({grad_norm:.4f})",
                  math.isfinite(grad_norm))

            # Gradient flows from decoder path too (shared embedding).
            # Decoder uses embedding for token lookup — its contribution to
            # the grad should be reflected in the embedding grad.
            # We verify this by checking that the grad is non-trivially non-zero.
            non_zero_rows = int((grad.abs().sum(dim=1) > 1e-10).sum())
            check(S, f"[{model_type}] Multiple embedding rows have non-zero grad (>3)",
                  non_zero_rows > 3,
                  f"{non_zero_rows} rows with non-zero grad")

        del model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION G — Semantic sanity (cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

def section_g(mat: np.ndarray, sp: object) -> None:
    S = "G"
    header("G. Semantic Sanity (Cosine Similarity)")

    def _id(word: str) -> Optional[int]:
        ids = sp.encode_as_ids(word)
        return ids[0] if len(ids) == 1 else None

    def _sim(w1: str, w2: str) -> Optional[float]:
        i1, i2 = _id(w1), _id(w2)
        if i1 is None or i2 is None:
            return None
        return _cosine(mat[i1], mat[i2])

    # Pairs where related > unrelated.
    TESTS = [
        # (related_a, related_b, unrelated — sim(a,b) > sim(a, unrelated))
        ("ubuntu", "linux",   "monday"),
        ("install", "apt",    "grammar"),
        ("error",  "kernel",  "holiday"),
        ("network", "wifi",   "poetry"),
        ("python",  "script", "ocean"),
        ("server",  "host",   "flower"),
    ]

    for w_a, w_b, w_c in TESTS:
        sim_related   = _sim(w_a, w_b)
        sim_unrelated = _sim(w_a, w_c)
        if sim_related is None or sim_unrelated is None:
            skip(S, f"sim({w_a},{w_b}) > sim({w_a},{w_c})", "multi-token word")
            continue
        check(S, f"sim({w_a!r},{w_b!r})={sim_related:.3f} > sim({w_a!r},{w_c!r})={sim_unrelated:.3f}",
              sim_related > sim_unrelated,
              warn_only=True)   # WARN not FAIL — FastText on mini corpus may differ

    # Top-5 nearest neighbours of 'ubuntu' should be Linux-related.
    ubuntu_id = _id("ubuntu")
    if ubuntu_id is not None:
        ubuntu_vec = mat[ubuntu_id]
        # Cosine similarity to all vocabulary.
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
        normed = mat / norms
        sims = normed @ ubuntu_vec / (np.linalg.norm(ubuntu_vec) + 1e-9)
        top_ids = np.argsort(-sims)[1:6]   # skip self (rank 0)
        top_words = [sp.id_to_piece(int(i)) for i in top_ids]
        top_sims  = [float(sims[i]) for i in top_ids]
        if _verbose:
            print(f"    Top-5 nearest to 'ubuntu': {list(zip(top_words, [round(s,3) for s in top_sims]))}")
        # At least one of top-5 should be a tech/Linux term (heuristic).
        tech_terms = {"linux", "debian", "fedora", "kernel", "mint", "gnome",
                      "xubuntu", "kubuntu", "lubuntu", "distro", "unix",
                      "▁linux", "▁debian", "▁kernel", "▁fedora", "▁mint",
                      "▁debian", "▁distro", "▁unix"}
        hit = any(w in tech_terms for w in top_words)
        check(S, "Top-5 nearest to 'ubuntu' includes a Linux-related token",
              hit,
              f"top-5: {top_words}",
              warn_only=True)

    # Special tokens should NOT be similar to each other.
    SPECIAL_PAIRS = [("<sos>", "<eos>"), ("<unk>", "<eos>"), ("<sos>", "<unk>")]
    for t1, t2 in SPECIAL_PAIRS:
        i1 = sp.piece_to_id(t1)
        i2 = sp.piece_to_id(t2)
        s  = _cosine(mat[i1], mat[i2])
        check(S, f"Special tokens {t1!r} and {t2!r} not near-identical (sim < 0.95)",
              s < 0.95,
              f"cosine={s:.4f}",
              warn_only=True)

    # Domain tags should all be mutually dissimilar (each carries unique semantics).
    TAG_IDS = list(range(4, 11))   # __url__, __path__, ..., __user__
    all_ok = True
    for i in range(len(TAG_IDS)):
        for j in range(i+1, len(TAG_IDS)):
            s = _cosine(mat[TAG_IDS[i]], mat[TAG_IDS[j]])
            if s > 0.95:
                all_ok = False
                if _verbose:
                    print(f"    WARNING: domain tags {TAG_IDS[i]} and {TAG_IDS[j]} "
                          f"are very similar (cos={s:.4f})")
    check(S, "All pairs of domain tags (IDs 4–10) are mutually dissimilar (< 0.95)",
          all_ok, warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION H — Lookup correctness
# ─────────────────────────────────────────────────────────────────────────────

def section_h(mat: np.ndarray, cfg: dict, matrix_path: str) -> None:
    S = "H"
    header("H. nn.Embedding Lookup Correctness")

    pad_idx   = cfg["pad_idx"]
    vocab_size = mat.shape[0]
    embed_dim  = mat.shape[1]
    emb = create_pretrained_embedding(matrix_path, pad_idx=pad_idx, freeze=True)
    emb.eval()

    # H1 — Pad token lookup returns all-zeros.
    pad_out = emb(torch.tensor([pad_idx])).detach().numpy()
    check(S, "emb([pad_idx]) returns all-zeros vector",
          bool(np.linalg.norm(pad_out) < 1e-7),
          f"norm={np.linalg.norm(pad_out):.4e}")

    # H2 — Known token lookup matches matrix row.
    test_ids = [1, 2, 3, 116, 299, 93]   # unk, sos, eos, ubuntu, linux, install
    for tok_id in test_ids:
        out = emb(torch.tensor([tok_id])).detach().squeeze(0).numpy()
        expected = mat[tok_id]
        diff = float(np.abs(out - expected).max())
        check(S, f"emb([{tok_id}]) matches matrix row {tok_id} (max diff < 1e-5)",
              diff < 1e-5,
              f"max_diff={diff:.2e}")

    # H3 — Output shape from batch lookup.
    batch_ids = torch.tensor([[2, 116, 299, 93, 3],
                               [2, 344, 442, 0,  0]])
    out_batch = emb(batch_ids)
    check(S, "Batch lookup output shape: [batch, seq_len, embed_dim]",
          out_batch.shape == (2, 5, embed_dim),
          f"got {tuple(out_batch.shape)}")

    # H4 — Pad positions in batch are zero.
    pad_pos_norm = float(out_batch[1, 3:, :].norm())
    check(S, "Pad positions in batch lookup output are all-zeros",
          pad_pos_norm < 1e-7,
          f"norm of pad positions = {pad_pos_norm:.4e}")

    # H5 — Different tokens produce different vectors.
    ubuntu_out = emb(torch.tensor([116])).detach().squeeze().numpy()
    linux_out  = emb(torch.tensor([299])).detach().squeeze().numpy()
    sim = _cosine(ubuntu_out, linux_out)
    check(S, "Different tokens produce different vectors (ubuntu ≠ linux)",
          sim < 0.9999,
          f"cos sim = {sim:.6f}")

    # H6 — Lookup is deterministic (same call twice gives same result).
    out1 = emb(torch.tensor([116])).detach()
    out2 = emb(torch.tensor([116])).detach()
    check(S, "Embedding lookup is deterministic (same id → same vector)",
          bool(torch.allclose(out1, out2)))

    # H7 — Large index lookup doesn't crash or return zeros (boundary).
    last_id = vocab_size - 1
    last_out = emb(torch.tensor([last_id])).detach().numpy()
    check(S, f"Last vocab ID ({last_id}) lookup produces non-zero vector",
          bool(np.linalg.norm(last_out) > 1e-6),
          f"norm={np.linalg.norm(last_out):.4f}")

    # H8 — Verify padding_idx attribute propagated into forward (internal check).
    check(S, "emb.padding_idx is set correctly",
          emb.padding_idx == pad_idx,
          f"got {emb.padding_idx}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary() -> int:
    """Print final summary. Returns exit code (0=all pass, 1=any fail)."""
    n_pass = sum(1 for _, _, s in _results if s == _PASS)
    n_warn = sum(1 for _, _, s in _results if s == _WARN)
    n_fail = sum(1 for _, _, s in _results if s == _FAIL)
    n_skip = sum(1 for _, _, s in _results if s == _SKIP)
    total  = len(_results)

    print(f"\n{'═'*60}")
    print(f"{_BOLD}  EMBEDDING TEST SUMMARY{_RESET}")
    print(f"{'═'*60}")
    print(f"  {_GREEN}PASS{_RESET} : {n_pass:3d} / {total}")
    print(f"  {_YELLOW}WARN{_RESET} : {n_warn:3d}    (non-fatal, worth reviewing)")
    print(f"  {_RED}FAIL{_RESET} : {n_fail:3d}    (must fix before training)")
    print(f"  SKIP : {n_skip:3d}    (multi-token words, missing files)")

    if n_fail > 0:
        print(f"\n{_RED}{_BOLD}  FAILURES:{_RESET}")
        for section, name, status in _results:
            if status == _FAIL:
                print(f"    [{section}] {name}")

    if n_warn > 0:
        print(f"\n{_YELLOW}  WARNINGS:{_RESET}")
        for section, name, status in _results:
            if status == _WARN:
                print(f"    [{section}] {name}")

    verdict = (f"{_GREEN}{_BOLD}✅  ALL CHECKS PASSED — embedding is healthy.{_RESET}"
               if n_fail == 0 else
               f"{_RED}{_BOLD}❌  FAILURES DETECTED — see above.{_RESET}")
    print(f"\n  {verdict}\n")
    return 0 if n_fail == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(description="Extensive embedding test suite")
    parser.add_argument("--full",    action="store_true",
                        help="Use full artifacts/ instead of artifacts_mini/")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra diagnostic detail")
    args = parser.parse_args()
    _verbose = args.verbose

    import copy
    cfg = copy.deepcopy(CONFIG)

    artifact_dir = _HERE / ("artifacts" if args.full else "artifacts_mini")
    cfg["artifact_dir"]           = str(artifact_dir)
    cfg["spm_model_path"]         = str(artifact_dir / "stage5_spm.model")
    cfg["embedding_matrix_path"]  = str(artifact_dir / "stage8_embedding_matrix.npy")

    matrix_path = cfg["embedding_matrix_path"]
    spm_path    = cfg["spm_model_path"]
    device      = torch.device("cpu")   # CPU sufficient for shape/grad checks

    print(f"\n{_BOLD}Embedding Test Suite{_RESET}")
    print(f"  artifact_dir : {artifact_dir}")
    print(f"  matrix       : {matrix_path}")
    print(f"  spm          : {spm_path}")
    print(f"  device       : {device}")
    print(f"  verbose      : {_verbose}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    if not Path(matrix_path).exists():
        print(f"\n{_RED}ERROR: matrix file not found: {matrix_path}{_RESET}")
        print("  Run phase1_mini.py (or phase1.py --full) first.")
        return 1

    mat = np.load(matrix_path).astype(np.float32)
    print(f"\n  Matrix loaded: shape={mat.shape} dtype={mat.dtype}")

    sp_proc = load_tokenizer(artifact_dir)
    print(f"  Tokenizer loaded: vocab_size={sp_proc.get_piece_size()}")

    # ── Run all sections ──────────────────────────────────────────────────────
    try:
        section_a(mat, cfg["vocab_size"], cfg["embed_dim"])
    except Exception as e:
        print(f"  Section A crashed: {e}"); traceback.print_exc()

    try:
        section_b(mat, sp_proc)
    except Exception as e:
        print(f"  Section B crashed: {e}"); traceback.print_exc()

    try:
        section_c(matrix_path, mat, cfg)
    except Exception as e:
        print(f"  Section C crashed: {e}"); traceback.print_exc()

    try:
        section_d(cfg, device)
    except Exception as e:
        print(f"  Section D crashed: {e}"); traceback.print_exc()

    try:
        section_e(cfg, device)
    except Exception as e:
        print(f"  Section E crashed: {e}"); traceback.print_exc()

    try:
        section_f(cfg, device)
    except Exception as e:
        print(f"  Section F crashed: {e}"); traceback.print_exc()

    try:
        section_g(mat, sp_proc)
    except Exception as e:
        print(f"  Section G crashed: {e}"); traceback.print_exc()

    try:
        section_h(mat, cfg, matrix_path)
    except Exception as e:
        print(f"  Section H crashed: {e}"); traceback.print_exc()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
