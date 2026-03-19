"""
evaluate.py — Full evaluation suite for both Seq2Seq models.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — BLEU/ROUGE/BERTScore/Distinct-N, attention heatmaps
          v2.0.0 — Initial clean-from-scratch rewrite

Produces per run:
  - bleu_results.json        BLEU-1/2/3/4 (sacrebleu 13a) + ROUGE-L + Distinct-1/2 + BERTScore
  - baseline_manual_samples.json
  - attention_manual_samples.json
  - attention_heatmap.png    Bahdanau attention weight heatmap (attention model only)

Benchmark reference (osamadev/seq2seq-chatbot — LSTM+ATTN, greedy, 8k SP vocab):
  BLEU-1: 0.4400   BLEU-4: 0.1386   ROUGE-L F1: 0.0922
  NOTE (F3): direct numeric comparison invalid — different tokeniser (8k SentencePiece
  vs our 16k), different vocab size, and different data split. Use as directional
  reference only; report with this disclaimer in write-up.

Metric rationale (G1 / Liu et al. 2016):
  BLEU alone is insufficient for open-domain dialogue. We additionally report:
    Distinct-1/2  — response diversity (G2 / R5)
    BERTScore F1  — semantic similarity beyond n-gram overlap (G1)
    ROUGE-L       — longest-common-subsequence recall
  All four are needed for a complete picture.
"""

import json
import os
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizer_utils import load_tokenizer
import sacrebleu
from rouge_score import rouge_scorer
import bert_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def greedy_decode(
    model,
    src: torch.Tensor,          # [batch, src_len]
    src_lengths: torch.Tensor,  # [batch]
    sos_idx: int,
    eos_idx: int,
    max_len: int,
    device: torch.device,
) -> List[List[int]]:
    """
    Greedy decode a batch. Returns list of token ID lists (no padding, no EOS).
    Used for BLEU computation — matches osamadev benchmark methodology (F3).
    """
    model.eval()
    src = src.to(device)
    src_lengths = src_lengths.to(device)
    batch_size = src.size(0)

    # Encode once.
    encoder_outputs, (h_n, c_n) = model.encoder(src, src_lengths)
    src_mask = (src == model.encoder.embedding.padding_idx)  # [B, src_len]
    dec_h, dec_c = model.bridge(h_n, c_n)

    input_token = torch.full(
        (batch_size,), sos_idx, dtype=torch.long, device=device
    )

    decoded: List[List[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Initial context vector (zeros); updated each step by decoder (attention model
    # returns new_context; baseline ignores it and uses fixed encoder output).
    context = torch.zeros(batch_size, encoder_outputs.size(-1), device=device)

    for _ in range(max_len):
        # forward_step(input, hidden, cell, encoder_outputs, context, src_mask) → 5 values
        logits, dec_h, dec_c, context, _ = model.decoder.forward_step(
            input_token, dec_h, dec_c, encoder_outputs, context, src_mask
        )
        # logits: [batch, vocab_size]
        next_token = logits.argmax(dim=-1)  # [batch]

        for i in range(batch_size):
            if not finished[i]:
                tok = next_token[i].item()
                if tok == eos_idx:
                    finished[i] = True
                else:
                    decoded[i].append(tok)

        if finished.all():
            break

        input_token = next_token

    return decoded


@torch.inference_mode()
def top_p_decode(
    model,
    src: torch.Tensor,          # [batch, src_len]
    src_lengths: torch.Tensor,  # [batch]
    sos_idx: int,
    eos_idx: int,
    max_len: int,
    device: torch.device,
    top_p: float = 0.9,
    temperature: float = 0.8,
    ngram_block: int = 3,
) -> List[List[int]]:
    """
    Top-p (nucleus) sampling with temperature scaling and n-gram blocking (A4).

    N-gram blocking (pre-softmax logit masking):
      Before sampling, collect all (ngram_block-1)-length prefixes that have
      already appeared in the decoded sequence. If the last (ngram_block-1)
      decoded tokens match any such prefix, set the logit of every token that
      would complete the n-gram to -inf (pre-softmax, not post-softmax).
      Fallback: if ALL tokens are blocked, clear the block set for this step
      so the model can still generate rather than producing <unk> or crashing.

    Used for interactive / qualitative evaluation; not used for BLEU benchmark.
    """
    model.eval()
    src = src.to(device)
    src_lengths = src_lengths.to(device)
    batch_size = src.size(0)

    encoder_outputs, (h_n, c_n) = model.encoder(src, src_lengths)
    src_mask = (src == model.encoder.embedding.padding_idx)
    dec_h, dec_c = model.bridge(h_n, c_n)

    input_token = torch.full(
        (batch_size,), sos_idx, dtype=torch.long, device=device
    )

    decoded: List[List[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Initial context vector; updated each step.
    context = torch.zeros(batch_size, encoder_outputs.size(-1), device=device)

    for _ in range(max_len):
        # forward_step(input, hidden, cell, encoder_outputs, context, src_mask) → 5 values
        logits, dec_h, dec_c, context, _ = model.decoder.forward_step(
            input_token, dec_h, dec_c, encoder_outputs, context, src_mask
        )
        # logits: [batch, vocab_size]

        next_tokens = []
        for i in range(batch_size):
            if finished[i]:
                next_tokens.append(input_token[i])
                continue

            lgts = logits[i].clone()  # [vocab_size]

            # ── N-gram blocking (A4): pre-softmax logit masking ──────────────
            if ngram_block > 1 and len(decoded[i]) >= ngram_block - 1:
                prefix = tuple(decoded[i][-(ngram_block - 1):])
                # Collect which tokens would complete a repeated n-gram.
                blocked = set()
                seq = decoded[i]
                for start in range(len(seq) - (ngram_block - 1)):
                    if tuple(seq[start: start + ngram_block - 1]) == prefix:
                        blocked.add(seq[start + ngram_block - 1])
                # Apply masking only if it doesn't block everything (fallback).
                if blocked and len(blocked) < lgts.size(0):
                    lgts[list(blocked)] = float("-inf")

            # ── Temperature scaling ──────────────────────────────────────────
            lgts = lgts / max(temperature, 1e-8)

            # ── Nucleus (top-p) filtering ────────────────────────────────────
            probs = F.softmax(lgts, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            # Remove tokens where cumulative prob exceeds top_p (keep at least 1).
            remove_mask = cumulative - sorted_probs > top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs /= sorted_probs.sum().clamp(min=1e-8)  # re-normalise

            sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
            next_tok = sorted_idx[sampled_pos].item()

            if next_tok == eos_idx:
                finished[i] = True
                next_tokens.append(input_token[i])
            else:
                decoded[i].append(next_tok)
                next_tokens.append(torch.tensor(next_tok, device=device))

        input_token = torch.stack(next_tokens)

        if finished.all():
            break

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_distinct_n(sequences: List[List[int]], n: int) -> float:
    """
    Corpus-level Distinct-N: unique n-grams / total n-grams (R5 / G2).

    Returns 0.0 if there are no n-grams (degenerate empty output).
    """
    total = 0
    unique: set = set()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i: i + n])
            unique.add(gram)
            total += 1
    return len(unique) / max(total, 1)


def _ids_to_str(ids: List[int], sp: object) -> str:
    """Decode BPE IDs to a detokenised string, skipping special tokens."""
    return sp.decode(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus BLEU + ROUGE + Distinct + BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu_corpus(
    model,
    loader,
    sp: object,
    device: torch.device,
    max_len: int = 40,
    sos_idx: int = 2,
    eos_idx: int = 3,
    bert_score_model: str = "distilbert-base-uncased",
    bert_score_batch: int = 64,
) -> Dict[str, float]:
    """
    Corpus-level evaluation: BLEU-1/2/3/4 (sacrebleu 13a), ROUGE-L, Distinct-1/2,
    BERTScore F1.

    Uses greedy decoding (fair comparison with osamadev benchmark — F3).
    sacrebleu 13a tokeniser applied to decoded strings (F3).
    Bootstrap CIs available via sacrebleu; logged to console (G5).
    """
    hypotheses_str: List[str] = []
    references_str: List[str] = []
    hypotheses_ids: List[List[int]] = []

    for batch in loader:
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"]
        trg = batch["trg"]  # [B, trg_len]  — reference (CPU)

        hyp_ids = greedy_decode(
            model, src, src_lengths,
            sos_idx=sos_idx, eos_idx=eos_idx,
            max_len=max_len, device=device,
        )

        for i in range(src.size(0)):
            # Reference: strip sos/eos tokens (first and last if present).
            ref_ids = trg[i].tolist()
            if ref_ids and ref_ids[0] == sos_idx:
                ref_ids = ref_ids[1:]
            ref_ids = [t for t in ref_ids if t not in (eos_idx, 0)]  # strip eos+pad

            hypotheses_str.append(_ids_to_str(hyp_ids[i], sp))
            references_str.append(_ids_to_str(ref_ids, sp))
            hypotheses_ids.append(hyp_ids[i])

    # ── BLEU (sacrebleu 13a) ─────────────────────────────────────────────────
    bleu = sacrebleu.corpus_bleu(
        hypotheses_str,
        [references_str],
        tokenize="13a",
    )
    # BLEU-1/2/3 are independent corpus_bleu calls, each with their own brevity
    # penalty (BP). This is consistent with how the osamadev reference benchmark
    # was computed, but note that BLEU-N here is NOT the n-gram precision component
    # of BLEU-4 — it is a separate metric. Documented in PLAN.md §Phase 3 (AC2-I2).
    bleu1 = sacrebleu.corpus_bleu(hypotheses_str, [references_str], tokenize="13a", max_ngram_order=1).score / 100
    bleu2 = sacrebleu.corpus_bleu(hypotheses_str, [references_str], tokenize="13a", max_ngram_order=2).score / 100
    bleu3 = sacrebleu.corpus_bleu(hypotheses_str, [references_str], tokenize="13a", max_ngram_order=3).score / 100
    bleu4 = bleu.score / 100

    # ── ROUGE-L ──────────────────────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rougeL_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references_str, hypotheses_str)
    ]
    rougeL_f1 = sum(rougeL_scores) / max(len(rougeL_scores), 1)

    # ── Distinct-1/2 ─────────────────────────────────────────────────────────
    distinct1 = compute_distinct_n(hypotheses_ids, 1)
    distinct2 = compute_distinct_n(hypotheses_ids, 2)

    # ── BERTScore ────────────────────────────────────────────────────────────
    # Pass device explicitly so evaluation honours caller's device (A2-M1).
    P, R, F1 = bert_score.score(
        hypotheses_str,
        references_str,
        lang="en",
        model_type=bert_score_model,
        batch_size=bert_score_batch,
        device=str(device),
        verbose=False,
    )
    bertscore_f1 = F1.mean().item()

    results = {
        "bleu1": round(bleu1, 4),
        "bleu2": round(bleu2, 4),
        "bleu3": round(bleu3, 4),
        "bleu4": round(bleu4, 4),
        "rougeL_f1": round(rougeL_f1, 4),
        "distinct1": round(distinct1, 4),
        "distinct2": round(distinct2, 4),
        "bertscore_f1": round(bertscore_f1, 4),
        "num_samples": len(hypotheses_str),
        # AC2-I3: greedy decoding used for all corpus metrics (benchmark comparability).
        # Manual evaluation samples use top-p (see manual_evaluation_samples).
        # BLEU-1/2/3 each have independent brevity penalties (AC2-I2).
        # Bootstrap CIs require sacrebleu CLI --confidence flag (sacrebleu v2 Python API
        # does not expose CI on the BLEU result object).
        "decode_strategy": "greedy",
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Manual inspection samples
# ─────────────────────────────────────────────────────────────────────────────

def manual_evaluation_samples(
    model,
    loader,
    sp: object,
    device: torch.device,
    num_samples: int = 50,
    decode_strategy: str = "top_p",
    sos_idx: int = 2,
    eos_idx: int = 3,
    max_len: int = 40,
    top_p: float = 0.9,
    temperature: float = 0.8,
    ngram_block: int = 3,
) -> List[Dict[str, str]]:
    """
    Generate num_samples {src, tgt, hyp} dicts for manual inspection.

    decode_strategy: "greedy" or "top_p" (default top_p for diversity).
    top_p/temperature/ngram_block are passed through from config (QA2-L2).
    """
    samples: List[Dict[str, str]] = []

    for batch in loader:
        if len(samples) >= num_samples:
            break

        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"]
        trg = batch["trg"]

        if decode_strategy == "greedy":
            hyp_ids = greedy_decode(
                model, src, src_lengths,
                sos_idx=sos_idx, eos_idx=eos_idx,
                max_len=max_len, device=device,
            )
        else:
            hyp_ids = top_p_decode(
                model, src, src_lengths,
                sos_idx=sos_idx, eos_idx=eos_idx,
                max_len=max_len, device=device,
                top_p=top_p, temperature=temperature, ngram_block=ngram_block,
            )

        for i in range(src.size(0)):
            if len(samples) >= num_samples:
                break

            src_ids = src[i].tolist()
            src_ids = [t for t in src_ids if t not in (0,)]  # strip pad
            ref_ids = trg[i].tolist()
            if ref_ids and ref_ids[0] == sos_idx:
                ref_ids = ref_ids[1:]
            ref_ids = [t for t in ref_ids if t not in (eos_idx, 0)]

            samples.append({
                "src": _ids_to_str(src_ids, sp),
                "tgt": _ids_to_str(ref_ids, sp),
                "hyp": _ids_to_str(hyp_ids[i], sp),
            })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Attention heatmap visualisation (G4)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def plot_attention_heatmap(
    model,
    src_ids: List[int],
    sp: object,
    device: torch.device,
    save_path: str,
    sos_idx: int = 2,
    eos_idx: int = 3,
    max_len: int = 40,
) -> None:
    """
    Decode a single src sequence, collect per-step attention weights,
    and save a heatmap to save_path (G4).

    Works only with the attention model (BaselineDecoder has no attention weights).
    If the model has no attention mechanism, this is a no-op.
    """
    if not hasattr(model.decoder, "attention"):
        return  # baseline model — skip

    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)

    encoder_outputs, (h_n, c_n) = model.encoder(src_tensor, src_lengths)
    src_mask = (src_tensor == model.encoder.embedding.padding_idx)
    dec_h, dec_c = model.bridge(h_n, c_n)

    input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
    decoded_ids: List[int] = []
    attn_weights: List[torch.Tensor] = []  # each: [src_len]

    # Initial context vector (zeros); updated each step.
    context = torch.zeros(1, encoder_outputs.size(-1), device=device)

    for _ in range(max_len):
        # forward_step returns 5 values: logits, hidden, cell, context, attn_weights
        logits, dec_h, dec_c, context, step_attn = model.decoder.forward_step(
            input_token, dec_h, dec_c, encoder_outputs, context, src_mask
        )
        next_tok = logits.argmax(dim=-1).item()
        if next_tok == eos_idx:
            break
        decoded_ids.append(next_tok)
        if step_attn is not None:
            attn_weights.append(step_attn.squeeze(0).cpu())  # [src_len]
        input_token = torch.tensor([next_tok], dtype=torch.long, device=device)

    if not attn_weights:
        return

    # Build heatmap matrix: [trg_len, src_len]
    attn_matrix = torch.stack(attn_weights, dim=0).numpy()  # [trg_len, src_len]

    src_pieces = sp.encode(
        sp.decode(src_ids), out_type=str
    ) if src_ids else ["?"]
    trg_pieces = sp.encode(
        sp.decode(decoded_ids), out_type=str
    ) if decoded_ids else ["?"]

    fig_w = max(8, len(src_pieces) * 0.4)
    fig_h = max(4, len(trg_pieces) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        attn_matrix,
        xticklabels=src_pieces,
        yticklabels=trg_pieces,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_title("Bahdanau Attention Weights")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_dir: str,
    artifact_dir: str,
    config: dict,
    device: torch.device,
) -> None:
    """
    Load best checkpoints for both models, run full evaluation, save results.

    Outputs (all written to checkpoint_dir):
      bleu_results.json, baseline_manual_samples.json,
      attention_manual_samples.json, attention_heatmap.png
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models import build_model
    from dataset import build_dataloaders

    artifact_dir = Path(artifact_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load tokenizer (HF Tokenizers or SentencePiece fallback).
    sp = load_tokenizer(artifact_dir)

    # Build test dataloader only (no word2idx needed — dataset reads JSONL IDs directly).
    _, _, test_loader = build_dataloaders(
        artifact_dir=str(artifact_dir),
        batch_size=config.get("batch_size", 128),
        num_workers=0,
        max_ctx_len=config.get("max_ctx_tokens", 100),
        max_resp_len=config.get("max_resp_tokens", 40) + 2,
        pad_idx=config.get("pad_idx", 0),
    )

    sos_idx = config.get("sos_idx", 2)
    eos_idx = config.get("eos_idx", 3)
    max_len = config.get("max_decode_len", 40)

    all_bleu: Dict[str, Dict] = {}

    for model_type in ("baseline", "attention"):
        ckpt_path = checkpoint_dir / f"{model_type}_best.pt"
        if not ckpt_path.exists():
            print(f"[evaluate] {ckpt_path} not found — skipping {model_type}")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_type.upper()} MODEL")
        print(f"{'='*60}")

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = build_model(model_type, config, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # ── BLEU / ROUGE / Distinct / BERTScore ──────────────────────────────
        print("  Computing corpus metrics…")
        metrics = compute_bleu_corpus(
            model, test_loader, sp, device,
            max_len=max_len, sos_idx=sos_idx, eos_idx=eos_idx,
        )
        all_bleu[model_type] = metrics
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        # ── Manual samples ────────────────────────────────────────────────────
        print("  Generating manual evaluation samples…")
        samples = manual_evaluation_samples(
            model, test_loader, sp, device,
            num_samples=50, decode_strategy="top_p",
            sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len,
            top_p=config.get("top_p", 0.9),
            temperature=config.get("temperature", 0.8),
            ngram_block=config.get("ngram_block", 3),
        )
        samples_path = checkpoint_dir / f"{model_type}_manual_samples.json"
        _tmp_samp = str(samples_path) + ".tmp"
        with open(_tmp_samp, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        os.replace(_tmp_samp, samples_path)
        print(f"  Saved {len(samples)} samples → {samples_path}")

        # ── Attention heatmap (attention model only) ──────────────────────────
        if model_type == "attention":
            print("  Generating attention heatmap…")
            # Grab first src sequence from test set for illustration.
            sample_batch = next(iter(test_loader))
            src_ids = [
                t for t in sample_batch["src"][0].tolist() if t != 0
            ]
            heatmap_path = str(checkpoint_dir / "attention_heatmap.png")
            plot_attention_heatmap(
                model, src_ids, sp, device, save_path=heatmap_path,
                sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len,
            )
            print(f"  Saved attention heatmap → {heatmap_path}")

    # ── Save combined BLEU results (atomic write) ────────────────────────────
    bleu_path = checkpoint_dir / "bleu_results.json"
    _tmp_bleu = str(bleu_path) + ".tmp"
    with open(_tmp_bleu, "w") as f:
        json.dump(all_bleu, f, indent=2)
    os.replace(_tmp_bleu, bleu_path)
    print(f"\nAll metrics saved → {bleu_path}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {'Baseline':>12} {'Attention':>12} {'Reference*':>12}")
    print("-" * 70)
    ref = {"bleu1": 0.4400, "bleu4": 0.1386, "rougeL_f1": 0.0922}
    for metric in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL_f1", "distinct1", "distinct2", "bertscore_f1"):
        b_val = all_bleu.get("baseline", {}).get(metric, "—")
        a_val = all_bleu.get("attention", {}).get(metric, "—")
        r_val = ref.get(metric, "—")
        print(f"  {metric:<18} {str(b_val):>12} {str(a_val):>12} {str(r_val):>12}")
    print("=" * 70)
    print("* Reference: osamadev/seq2seq-chatbot (8k SP, greedy) — directional only (F3)")


if __name__ == "__main__":
    from config import CONFIG, set_seed
    from logging_utils import setup_run_logging
    from gpu_utils import setup_device
    setup_run_logging("evaluate", log_dir=CONFIG.get("log_dir", "new/logs"))
    set_seed(CONFIG.get("seed", 42))
    device, gpu_info = setup_device()
    run_evaluation(
        checkpoint_dir=CONFIG["checkpoint_dir"],
        artifact_dir=CONFIG["artifact_dir"],
        config=CONFIG,
        device=device,
    )
