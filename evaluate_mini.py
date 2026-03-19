"""
evaluate_mini.py — Lightweight evaluation of mini-trained Seq2Seq models.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Lightweight 3-layer evaluation for mini models
          v2.0.0 — Initial version

Designed to run automatically after train_mini.py, but can also be called
directly. Produces a 3-layer analysis:

  Layer 1 — Automatic metrics  : BLEU-1/2, Distinct-1/2, generic-response rate,
                                  UNK rate in outputs, avg/median response length
  Layer 2 — Structural checks  : top-20 most-generated responses (collapse?),
                                  response length histogram, novel-word ratio
  Layer 3 — Sample review      : 30 ctx → [BASELINE|ATTENTION] vs GOLD pairs
                                  printed to console AND saved to report dir

Final verdict printed to console + written to report:
  ✅ HEALTHY        — metrics positive, no collapse, samples coherent
  ⚠️  COLLAPSE       — top response accounts for > 20% of all outputs
  ⚠️  NOT LEARNING   — val loss still near initial (~ln(vocab_size)); BLEU-1 < 0.05
  ⚠️  OVERFIT        — train loss << val loss (gap > 0.5)

Usage:
    python evaluate_mini.py                           # uses default mini paths
    Called automatically by train_mini.py after training.
"""

import copy
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── Ensure new/ is on path ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG
from dataset import build_dataloaders
from models import build_model
from evaluate import greedy_decode, compute_distinct_n


# ─────────────────────────────────────────────────────────────────────────────
# Generic response detection
# ─────────────────────────────────────────────────────────────────────────────

# Common generic chatbot fillers — any decoded response that matches one of
# these (after lower-casing and stripping) is flagged as "generic".
_GENERIC_RESPONSES = {
    "i don't know",
    "i don t know",
    "i'm not sure",
    "i m not sure",
    "yes",
    "no",
    "ok",
    "okay",
    "sure",
    "thanks",
    "thank you",
    "you're welcome",
    "you re welcome",
    "you're right",
    "you re right",
    "i see",
    "i agree",
    "maybe",
    "perhaps",
    "of course",
    "that's right",
    "that s right",
    "good",
    "great",
    "nice",
    "cool",
    "sorry",
    "i'm sorry",
    "i m sorry",
    "no problem",
    "that's a good question",
    "that s a good question",
    "i have no idea",
    "i don't understand",
    "i don t understand",
}


def _is_generic(text: str) -> bool:
    return text.strip().lower() in _GENERIC_RESPONSES


# ─────────────────────────────────────────────────────────────────────────────
# BLEU-1/2 (sacrebleu-style simple n-gram overlap — no dependency on sacrebleu)
# ─────────────────────────────────────────────────────────────────────────────

def _ngram_counts(seq: List[int], n: int) -> Counter:
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1))


def _corpus_bleu_n(hypotheses: List[List[int]], references: List[List[int]], n: int) -> float:
    """Compute corpus-level BLEU precision for n-grams (clipped)."""
    clip_total = 0
    hyp_total = 0
    for hyp, ref in zip(hypotheses, references):
        if len(hyp) < n:
            continue
        hyp_counts = _ngram_counts(hyp, n)
        ref_counts = _ngram_counts(ref, n)
        clip_total += sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        hyp_total += max(len(hyp) - n + 1, 0)
    if hyp_total == 0:
        return 0.0
    return clip_total / hyp_total


def compute_bleu_mini(
    hypotheses: List[List[int]],
    references: List[List[int]],
) -> Dict[str, float]:
    """Compute BLEU-1 and BLEU-2 with brevity penalty."""
    bp_hyp = sum(len(h) for h in hypotheses)
    bp_ref = sum(len(r) for r in references)
    bp = 1.0 if bp_hyp >= bp_ref else math.exp(1 - bp_ref / max(bp_hyp, 1))

    p1 = _corpus_bleu_n(hypotheses, references, 1)
    p2 = _corpus_bleu_n(hypotheses, references, 2)
    p3 = _corpus_bleu_n(hypotheses, references, 3)
    p4 = _corpus_bleu_n(hypotheses, references, 4)

    bleu1 = bp * p1
    bleu2 = bp * math.exp((math.log(max(p1, 1e-9)) + math.log(max(p2, 1e-9))) / 2)
    bleu4 = bp * math.exp(
        (math.log(max(p1, 1e-9)) + math.log(max(p2, 1e-9)) +
         math.log(max(p3, 1e-9)) + math.log(max(p4, 1e-9))) / 4
    )
    return {"bleu1": round(bleu1, 4), "bleu2": round(bleu2, 4), "bleu4": round(bleu4, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# SPM decoder helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_idx2word(artifact_dir: str) -> Dict[int, str]:
    """Build idx→word from stage6_idx2word.json (saved by phase1 stage 6)."""
    path = Path(artifact_dir) / "stage6_idx2word.json"
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def _ids_to_tokens(ids: List[int], idx2word: Dict[int, str]) -> List[str]:
    return [idx2word.get(i, f"<{i}>") for i in ids]


def _ids_to_text(ids: List[int], sp) -> str:
    """Use SentencePiece to detokenise; fall back to space-joined tokens."""
    try:
        return sp.decode(ids)
    except Exception:
        return " ".join(str(i) for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
# Collect decoded outputs and references over the test loader (max N batches)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_greedy_on_loader(
    model,
    loader,
    device: torch.device,
    max_batches: int = 20,
    sos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 0,
    max_len: int = 40,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Returns (hypotheses, references, contexts) as lists of token ID lists.
    Stops after max_batches to keep evaluation fast.
    """
    hyps, refs, ctxs = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        trg = batch["trg"]  # CPU

        decoded = greedy_decode(
            model, src, src_lengths,
            sos_idx=sos_idx, eos_idx=eos_idx,
            max_len=max_len, device=device,
        )

        for b in range(src.size(0)):
            hyps.append(decoded[b])
            # Strip <sos>, <eos>, <pad> from reference.
            ref = trg[b].tolist()
            ref = [t for t in ref if t not in (sos_idx, eos_idx, pad_idx)]
            refs.append(ref)
            # Store context for sample display.
            ctx = src[b].tolist()
            ctx = [t for t in ctx if t != pad_idx]
            ctxs.append(ctx)

    return hyps, refs, ctxs


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Automatic metrics
# ─────────────────────────────────────────────────────────────────────────────

def layer1_auto_metrics(
    hyps: List[List[int]],
    refs: List[List[int]],
    vocab_size: int,
    unk_idx: int = 1,
) -> Dict[str, float]:
    """Compute all Layer 1 metrics and return as a flat dict."""
    bleu = compute_bleu_mini(hyps, refs)
    d1 = compute_distinct_n(hyps, 1)
    d2 = compute_distinct_n(hyps, 2)

    lengths = [len(h) for h in hyps]
    avg_len = sum(lengths) / max(len(lengths), 1)
    sorted_lens = sorted(lengths)
    median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0

    generic_rate = sum(1 for h in hyps if len(h) == 0) / max(len(hyps), 1)

    unk_count = sum(h.count(unk_idx) for h in hyps)
    total_tokens = sum(len(h) for h in hyps)
    unk_rate = unk_count / max(total_tokens, 1)

    empty_rate = sum(1 for h in hyps if len(h) == 0) / max(len(hyps), 1)

    return {
        **bleu,
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "avg_resp_len": round(avg_len, 2),
        "median_resp_len": float(median_len),
        "unk_rate": round(unk_rate, 4),
        "empty_rate": round(empty_rate, 4),
        "n_samples": len(hyps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Structural checks
# ─────────────────────────────────────────────────────────────────────────────

def layer2_structural(
    hyps: List[List[int]],
    sp,
    top_n: int = 20,
) -> Dict:
    """Return top-N most frequent responses and length histogram."""
    decoded_strs = [_ids_to_text(h, sp) if h else "<EMPTY>" for h in hyps]
    counts = Counter(decoded_strs)
    top = counts.most_common(top_n)

    # Length histogram (buckets: 0, 1-5, 6-10, 11-20, 21+)
    buckets = {"0": 0, "1-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
    for h in hyps:
        n = len(h)
        if n == 0:         buckets["0"] += 1
        elif n <= 5:       buckets["1-5"] += 1
        elif n <= 10:      buckets["6-10"] += 1
        elif n <= 20:      buckets["11-20"] += 1
        else:              buckets["21+"] += 1

    # Novel-word ratio: unique output tokens / total output tokens
    all_toks = [t for h in hyps for t in h]
    novel_ratio = len(set(all_toks)) / max(len(all_toks), 1)

    # Collapse check: top-1 response accounts for what fraction?
    top1_frac = top[0][1] / max(len(hyps), 1) if top else 0.0

    return {
        "top_responses": top,
        "length_buckets": buckets,
        "novel_word_ratio": round(novel_ratio, 4),
        "top1_response_fraction": round(top1_frac, 4),
        "collapse_flag": top1_frac > 0.20,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Sample display
# ─────────────────────────────────────────────────────────────────────────────

def layer3_sample_pairs(
    ctxs: List[List[int]],
    hyps_baseline: List[List[int]],
    hyps_attention: List[List[int]],
    refs: List[List[int]],
    sp,
    n_samples: int = 30,
) -> List[Dict]:
    """Build a list of sample dicts for review + report writing."""
    import random as _random
    rng = _random.Random(42)
    indices = rng.sample(range(len(ctxs)), min(n_samples, len(ctxs)))

    samples = []
    for idx in indices:
        ctx_text = _ids_to_text(ctxs[idx], sp)
        hyp_b = _ids_to_text(hyps_baseline[idx], sp) if hyps_baseline[idx] else "<EMPTY>"
        hyp_a = _ids_to_text(hyps_attention[idx], sp) if hyps_attention[idx] else "<EMPTY>"
        ref_text = _ids_to_text(refs[idx], sp) if refs[idx] else "<EMPTY>"
        samples.append({
            "idx": idx,
            "context": ctx_text,
            "baseline_response": hyp_b,
            "attention_response": hyp_a,
            "gold_response": ref_text,
        })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(
    metrics_b: Dict,
    metrics_a: Dict,
    struct_b: Dict,
    struct_a: Dict,
    history_b: Optional[Dict],
    history_a: Optional[Dict],
    vocab_size: int = 16000,
) -> Tuple[str, List[str]]:
    """Return (verdict_string, list_of_reasons)."""
    reasons = []
    flags = set()

    initial_loss_threshold = math.log(vocab_size)  # ~9.68 for 16k vocab

    for name, metrics, struct, history in [
        ("baseline", metrics_b, struct_b, history_b),
        ("attention", metrics_a, struct_a, history_a),
    ]:
        if metrics["bleu1"] < 0.05:
            flags.add("NOT_LEARNING")
            reasons.append(f"{name}: BLEU-1 {metrics['bleu1']:.4f} < 0.05 threshold")

        if struct["collapse_flag"]:
            flags.add("COLLAPSE")
            reasons.append(
                f"{name}: top response covers {struct['top1_response_fraction']*100:.1f}% "
                f"of outputs (>20% threshold) — '{struct['top_responses'][0][0][:80]}'"
            )

        if metrics["empty_rate"] > 0.30:
            flags.add("COLLAPSE")
            reasons.append(f"{name}: {metrics['empty_rate']*100:.1f}% empty responses")

        if history:
            train_losses = history.get("train_loss", [])
            val_losses = history.get("val_loss", [])
            if train_losses and val_losses:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                if final_val > initial_loss_threshold * 0.95:
                    flags.add("NOT_LEARNING")
                    reasons.append(
                        f"{name}: val_loss={final_val:.3f} still near random "
                        f"(ln(vocab)={initial_loss_threshold:.2f})"
                    )
                if final_val - final_train > 0.5:
                    flags.add("OVERFIT")
                    reasons.append(
                        f"{name}: val_loss={final_val:.3f} >> train_loss={final_train:.3f} "
                        f"(gap {final_val-final_train:.3f} > 0.5)"
                    )

    if not flags:
        return "✅ HEALTHY", ["All checks passed — model appears to be learning coherent responses."]

    verdict_parts = []
    if "COLLAPSE" in flags:
        verdict_parts.append("⚠️  COLLAPSE")
    if "NOT_LEARNING" in flags:
        verdict_parts.append("⚠️  NOT LEARNING")
    if "OVERFIT" in flags:
        verdict_parts.append("⚠️  OVERFIT")

    return " | ".join(verdict_parts), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    cfg: Dict,
    metrics_b: Dict,
    metrics_a: Dict,
    struct_b: Dict,
    struct_a: Dict,
    samples: List[Dict],
    verdict: str,
    reasons: List[str],
    report_dir: Path,
) -> Path:
    """Write mini_training_analysis.md report and return the path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "mini_training_analysis.md"

    lines = [
        "# Mini Training Analysis",
        "",
        f"> **Verdict: {verdict}**",
        "",
        "## Configuration",
        "```",
        f"artifact_dir      : {cfg.get('artifact_dir')}",
        f"checkpoint_dir    : {cfg.get('checkpoint_dir')}",
        f"max_train_samples : {cfg.get('max_train_samples', 'all')}",
        f"num_epochs        : {cfg.get('num_epochs')}",
        f"batch_size        : {cfg.get('batch_size')}",
        f"patience          : {cfg.get('patience', 'disabled')}",
        "```",
        "",
        "## Verdict Reasons",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines += [
        "## Layer 1 — Automatic Metrics",
        "",
        "| Metric | Baseline | Attention |",
        "|---|---|---|",
    ]
    all_keys = sorted(set(list(metrics_b.keys()) + list(metrics_a.keys())))
    for k in all_keys:
        vb = metrics_b.get(k, "—")
        va = metrics_a.get(k, "—")
        lines.append(f"| {k} | {vb} | {va} |")
    lines.append("")

    lines += [
        "## Layer 2 — Structural Analysis",
        "",
        "### Response Length Distribution",
        "",
        "| Bucket | Baseline | Attention |",
        "|---|---|---|",
    ]
    for bucket in ["0", "1-5", "6-10", "11-20", "21+"]:
        vb = struct_b["length_buckets"].get(bucket, 0)
        va = struct_a["length_buckets"].get(bucket, 0)
        lines.append(f"| {bucket} tokens | {vb} | {va} |")
    lines.append("")

    lines += [
        "### Top-10 Most Frequent Responses",
        "",
        "**Baseline:**",
        "",
    ]
    for resp, cnt in struct_b["top_responses"][:10]:
        pct = 100 * cnt / max(metrics_b["n_samples"], 1)
        lines.append(f"- ({pct:.1f}%) `{resp[:100]}`")
    lines.append("")

    lines += ["**Attention:**", ""]
    for resp, cnt in struct_a["top_responses"][:10]:
        pct = 100 * cnt / max(metrics_a["n_samples"], 1)
        lines.append(f"- ({pct:.1f}%) `{resp[:100]}`")
    lines.append("")

    lines += [
        f"- Baseline novel-word ratio: {struct_b['novel_word_ratio']}",
        f"- Attention novel-word ratio: {struct_a['novel_word_ratio']}",
        f"- Baseline collapse flag: {struct_b['collapse_flag']}",
        f"- Attention collapse flag: {struct_a['collapse_flag']}",
        "",
        "## Layer 3 — Sample Review (30 pairs)",
        "",
        "Format: **CTX** → **BASELINE** | **ATTENTION** | **GOLD**",
        "",
    ]
    for s in samples:
        lines += [
            f"**[{s['idx']}]**",
            f"- **CTX**: {s['context'][:200]}",
            f"- **BASELINE**: {s['baseline_response'][:150]}",
            f"- **ATTENTION**: {s['attention_response'][:150]}",
            f"- **GOLD**: {s['gold_response'][:150]}",
            "",
        ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return path


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: Dict = None) -> None:
    """
    Run the full evaluate_mini pipeline.

    Args:
        cfg: Config dict. If None, loads from config.py and applies mini defaults.
    """
    if cfg is None:
        import copy
        cfg = copy.deepcopy(CONFIG)
        cfg["artifact_dir"]    = str(_HERE / "artifacts_mini")
        cfg["checkpoint_dir"]  = str(_HERE / "checkpoints_mini")
        cfg["num_workers"]     = 0
        cfg["batch_size"]      = 64

    from gpu_utils import setup_device
    device, gpu_info = setup_device()
    print(f"\n[evaluate_mini] Device: {device}")
    print(f"[evaluate_mini] artifact_dir  : {cfg['artifact_dir']}")
    print(f"[evaluate_mini] checkpoint_dir: {cfg['checkpoint_dir']}")

    # ── Load tokenizer ────────────────────────────────────────────────────
    from tokenizer_utils import load_tokenizer
    sp = load_tokenizer(cfg["artifact_dir"])
    print(f"[evaluate_mini] Tokenizer loaded from {cfg['artifact_dir']}")

    # ── Build test loader ────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(
        artifact_dir=cfg["artifact_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        max_ctx_len=cfg["max_ctx_tokens"],
        max_resp_len=cfg["max_resp_tokens"] + 2,
        pad_idx=cfg["pad_idx"],
    )

    # ── Load best checkpoints ────────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"])
    models_loaded = {}
    histories = {}

    for model_type in ("baseline", "attention"):
        ckpt_path = ckpt_dir / f"{model_type}_best.pt"
        if not ckpt_path.exists():
            # Fall back to last checkpoint if best not available.
            ckpt_path = ckpt_dir / f"{model_type}_last.pt"
        if not ckpt_path.exists():
            print(f"[evaluate_mini] WARNING: no checkpoint found for {model_type} — skipping")
            models_loaded[model_type] = None
            histories[model_type] = None
            continue

        print(f"[evaluate_mini] Loading {model_type} from {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(model_type, cfg, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models_loaded[model_type] = model
        histories[model_type] = ckpt.get("history", None)
        best_epoch = ckpt.get("epoch", "?")
        best_val = ckpt.get("val_loss", float("nan"))
        print(f"  → epoch={best_epoch}, best_val_loss={best_val:.4f}")

    # Bail if both models missing (e.g., training hasn't run yet).
    if all(m is None for m in models_loaded.values()):
        print("[evaluate_mini] No checkpoints found — run train_mini.py first.")
        return

    # ── Collect decoded outputs (max 20 batches ≈ 1280 samples @ bs=64) ────
    MAX_BATCHES = 20
    print(f"\n[evaluate_mini] Decoding test set (max {MAX_BATCHES} batches)…")

    hyps_b, refs_b, ctxs_b = ([], [], [])
    hyps_a, refs_a, ctxs_a = ([], [], [])

    if models_loaded.get("baseline"):
        hyps_b, refs_b, ctxs_b = run_greedy_on_loader(
            models_loaded["baseline"], test_loader, device,
            max_batches=MAX_BATCHES,
            sos_idx=cfg["sos_idx"], eos_idx=cfg["eos_idx"],
            pad_idx=cfg["pad_idx"], max_len=cfg["max_decode_len"],
        )
        print(f"  baseline: {len(hyps_b)} samples decoded")

    if models_loaded.get("attention"):
        hyps_a, refs_a, ctxs_a = run_greedy_on_loader(
            models_loaded["attention"], test_loader, device,
            max_batches=MAX_BATCHES,
            sos_idx=cfg["sos_idx"], eos_idx=cfg["eos_idx"],
            pad_idx=cfg["pad_idx"], max_len=cfg["max_decode_len"],
        )
        print(f"  attention: {len(hyps_a)} samples decoded")

    # Use whichever model produced outputs for shared refs/ctxs.
    refs = refs_b if refs_b else refs_a
    ctxs = ctxs_b if ctxs_b else ctxs_a

    # ── Layer 1 — Automatic metrics ──────────────────────────────────────────
    print("\n[evaluate_mini] Layer 1 — automatic metrics")
    metrics_b = layer1_auto_metrics(hyps_b, refs, cfg["vocab_size"], cfg["unk_idx"]) \
        if hyps_b else {}
    metrics_a = layer1_auto_metrics(hyps_a, refs, cfg["vocab_size"], cfg["unk_idx"]) \
        if hyps_a else {}

    def _print_metrics(name: str, m: Dict) -> None:
        if not m:
            return
        print(f"\n  {name.upper()}:")
        for k, v in m.items():
            print(f"    {k:25s}: {v}")

    _print_metrics("baseline", metrics_b)
    _print_metrics("attention", metrics_a)

    # ── Layer 2 — Structural analysis ────────────────────────────────────────
    print("\n[evaluate_mini] Layer 2 — structural analysis")
    struct_b = layer2_structural(hyps_b, sp) if hyps_b else {
        "top_responses": [], "length_buckets": {}, "novel_word_ratio": 0,
        "top1_response_fraction": 0, "collapse_flag": False,
    }
    struct_a = layer2_structural(hyps_a, sp) if hyps_a else {
        "top_responses": [], "length_buckets": {}, "novel_word_ratio": 0,
        "top1_response_fraction": 0, "collapse_flag": False,
    }

    for name, struct in [("baseline", struct_b), ("attention", struct_a)]:
        if not struct["top_responses"]:
            continue
        print(f"\n  {name.upper()} — top-5 responses:")
        for resp, cnt in struct["top_responses"][:5]:
            pct = 100 * cnt / max(metrics_b["n_samples"] if name == "baseline"
                                   else metrics_a["n_samples"], 1)
            print(f"    ({pct:5.1f}%) {resp[:100]}")
        print(f"  collapse_flag       : {struct['collapse_flag']}")
        print(f"  novel_word_ratio    : {struct['novel_word_ratio']}")
        print(f"  length distribution : {struct['length_buckets']}")

    # ── Layer 3 — Sample review ──────────────────────────────────────────────
    print("\n[evaluate_mini] Layer 3 — 30 sample pairs")
    if ctxs and (hyps_b or hyps_a):
        # Pad whichever list is shorter with empty lists.
        n = min(len(ctxs), max(len(hyps_b), len(hyps_a)))
        hyps_b_pad = hyps_b + [[]] * max(0, n - len(hyps_b))
        hyps_a_pad = hyps_a + [[]] * max(0, n - len(hyps_a))
        refs_pad   = refs   + [[]] * max(0, n - len(refs))
        ctxs_pad   = ctxs  [:n]

        samples = layer3_sample_pairs(ctxs_pad, hyps_b_pad, hyps_a_pad, refs_pad, sp, n_samples=30)

        for s in samples[:10]:  # Print first 10 to console; all 30 go to report.
            print(f"\n  [{s['idx']}] CTX:      {s['context'][:120]}")
            print(f"       BASELINE: {s['baseline_response'][:100]}")
            print(f"       ATTENT. : {s['attention_response'][:100]}")
            print(f"       GOLD    : {s['gold_response'][:100]}")
    else:
        samples = []

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict, reasons = compute_verdict(
        metrics_b, metrics_a, struct_b, struct_a,
        histories.get("baseline"), histories.get("attention"),
        vocab_size=cfg["vocab_size"],
    )

    print("\n" + "=" * 60)
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    for r in reasons:
        print(f"  • {r}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_dir = _HERE / "report"
    report_path = write_report(
        cfg, metrics_b, metrics_a, struct_b, struct_a,
        samples, verdict, reasons, report_dir,
    )
    print(f"\n[evaluate_mini] Report saved → {report_path}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics_baseline": metrics_b,
        "metrics_attention": metrics_a,
    }
    results_path = ckpt_dir / "mini_eval_results.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[evaluate_mini] JSON results → {results_path}")


if __name__ == "__main__":
    main()
