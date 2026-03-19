"""
chatv2.py — CLI dual-model comparison chat with greedy / top-p / beam decoding.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Dual-model comparison, top-p/beam decoding, confidence display
          v2.0.0 — Initial clean-from-scratch rewrite

Loads baseline and attention models simultaneously and shows both responses
side by side.  If checkpoints are not specified via CLI, presents an
interactive numbered list of available .pt files from the checkpoint dir.

Usage:
    python chatv2.py                                  # auto-discover both models
    python chatv2.py --decoding beam --beam-width 5   # beam search
    python chatv2.py --baseline checkpoints/baseline_best.pt \
                     --attention checkpoints/attention_best.pt
    python chatv2.py --checkpoint-dir checkpoints_mini --decoding greedy

Runtime commands (type in chat):
    :mode greedy         switch to greedy decoding
    :mode topp           switch to top-p sampling
    :mode beam [N]       switch to beam search (optional width, default 5)
    :clear               clear conversation history
    :help                show this list
    quit / exit / q      exit
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizer_utils import load_tokenizer

from config import CONFIG
from models import build_model
from phase1 import _clean_text


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _greedy_decode(model, src, src_lengths, sos_idx, eos_idx, max_len, device):
    from evaluate import greedy_decode
    return greedy_decode(model, src, src_lengths, sos_idx, eos_idx, max_len, device)


@torch.inference_mode()
def _topp_decode(model, src, src_lengths, sos_idx, eos_idx, max_len, device,
                 top_p, temperature, ngram_block):
    from evaluate import top_p_decode
    return top_p_decode(model, src, src_lengths, sos_idx, eos_idx, max_len,
                        device, top_p, temperature, ngram_block)


@torch.inference_mode()
def beam_decode(
    model,
    src: torch.Tensor,          # [1, src_len]
    src_lengths: torch.Tensor,  # [1]
    sos_idx: int,
    eos_idx: int,
    max_len: int,
    device: torch.device,
    beam_width: int = 5,
    length_penalty: float = 0.7,
) -> List[List[int]]:
    """
    Beam search decoding for a single input (batch_size=1).

    At each step, each active beam is expanded into beam_width candidates.
    The overall top beam_width candidates (by length-penalised score) survive.
    When a beam emits EOS it is moved to completed hypotheses.

    length_penalty: score /= len(tokens)^alpha  (0=no penalty, 1=linear)
    Returns [[token_ids]] for the best completed hypothesis.
    """
    model.eval()
    src = src.to(device)
    src_lengths = src_lengths.to(device)

    # Encode once.
    enc_out, (h_n, c_n) = model.encoder(src, src_lengths)  # enc_out: [1, L, H]
    src_mask = (src == model.encoder.embedding.padding_idx)  # [1, L]
    dec_h, dec_c = model.bridge(h_n, c_n)                   # [layers, 1, dec_dim]
    enc_dim = enc_out.size(-1)
    context = torch.zeros(1, enc_dim, device=device)

    # Each beam: dict with keys score, tokens, h, c, ctx
    # Initialise by expanding the first SOS step.
    sos_tok = torch.tensor([sos_idx], dtype=torch.long, device=device)
    logits, h1, c1, ctx1, _ = model.decoder.forward_step(
        sos_tok, dec_h, dec_c, enc_out, context, src_mask
    )
    log_probs = F.log_softmax(logits[0], dim=-1)
    topk_lp, topk_ids = log_probs.topk(min(beam_width, log_probs.size(-1)))

    active: List[dict] = []
    completed: List[dict] = []

    for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
        if tid == eos_idx:
            completed.append({"score": lp, "tokens": []})
        else:
            active.append({"score": lp, "tokens": [tid],
                           "h": h1, "c": c1, "ctx": ctx1})

    for _ in range(max_len - 1):
        if not active:
            break

        candidates: List[dict] = []
        for beam in active:
            inp = torch.tensor([beam["tokens"][-1]], dtype=torch.long, device=device)
            logits, new_h, new_c, new_ctx, _ = model.decoder.forward_step(
                inp, beam["h"], beam["c"], enc_out, beam["ctx"], src_mask
            )
            lp_all = F.log_softmax(logits[0], dim=-1)
            topk_lp, topk_ids = lp_all.topk(min(beam_width, lp_all.size(-1)))

            for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                new_score = beam["score"] + lp
                new_tokens = beam["tokens"] + [tid]
                if tid == eos_idx:
                    norm = len(new_tokens) ** length_penalty
                    completed.append({"score": new_score / max(norm, 1e-6),
                                      "tokens": beam["tokens"]})
                else:
                    candidates.append({"score": new_score, "tokens": new_tokens,
                                       "h": new_h, "c": new_c, "ctx": new_ctx})

        # Prune to top beam_width by length-normalised score.
        candidates.sort(
            key=lambda b: b["score"] / max(len(b["tokens"]) ** length_penalty, 1e-6),
            reverse=True,
        )
        active = candidates[:beam_width]

    # Collect remaining active beams as completed.
    for b in active:
        norm = len(b["tokens"]) ** length_penalty
        completed.append({"score": b["score"] / max(norm, 1e-6),
                          "tokens": b["tokens"]})

    if not completed:
        return [[]]
    completed.sort(key=lambda b: b["score"], reverse=True)
    return [completed[0]["tokens"]]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def list_checkpoints(checkpoint_dir: Path, model_type: str) -> List[Path]:
    """Return sorted list of .pt checkpoints matching model_type."""
    patterns = [f"{model_type}_best.pt", f"{model_type}_last.pt",
                f"{model_type}_step_*.pt"]
    found = []
    for pat in patterns:
        found.extend(checkpoint_dir.glob(pat))
    # Deduplicate and sort: best first, last second, then step files.
    seen = set()
    ordered = []
    for priority in [f"{model_type}_best.pt", f"{model_type}_last.pt"]:
        p = checkpoint_dir / priority
        if p.exists() and p not in seen:
            ordered.append(p)
            seen.add(p)
    for p in sorted(checkpoint_dir.glob(f"{model_type}_step_*.pt")):
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def _ckpt_meta(path: Path) -> str:
    """Read epoch and val_loss from checkpoint without loading model weights."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        vl = ckpt.get("val_loss", float("nan"))
        return f"epoch={epoch}, val_loss={vl:.4f}"
    except Exception:
        return "unreadable"


def pick_checkpoint(
    checkpoint_dir: Path,
    model_type: str,
    cli_path: Optional[str],
) -> Optional[Path]:
    """
    Return the checkpoint path for model_type.

    If cli_path is given → use it directly.
    If only one checkpoint exists → use it automatically.
    If multiple exist → present numbered list for user to pick.
    If none exist → return None (model will be skipped).
    """
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    ckpts = list_checkpoints(checkpoint_dir, model_type)
    if not ckpts:
        print(f"  ⚠  No {model_type} checkpoints found in {checkpoint_dir}")
        return None
    if len(ckpts) == 1:
        print(f"  {model_type}: auto-selected {ckpts[0].name}")
        return ckpts[0]

    print(f"\n  Available {model_type} checkpoints:")
    for i, p in enumerate(ckpts, 1):
        meta = _ckpt_meta(p)
        print(f"    [{i}] {p.name:<40}  {meta}")
    while True:
        raw = input(f"  Select {model_type} [1–{len(ckpts)}] (default=1): ").strip()
        if raw == "":
            return ckpts[0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(ckpts):
                return ckpts[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(ckpts)}.")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    ckpt_path: Path,
    model_type: str,
    config: dict,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    model = build_model(model_type, config, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Context building (mirrors phase1 stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    history: List[str],
    sp_processor,
    max_ctx_tokens: int,
    max_turns: int,
) -> torch.Tensor:
    """Encode conversation history as a [1, ctx_len] LongTensor."""
    turns = history[-max_turns:] if history else []
    cleaned = [_clean_text(t) for t in turns]
    cleaned = [t for t in cleaned if t]
    joined = " __eot__ ".join(cleaned) if cleaned else ""
    tokens = sp_processor.encode(joined, out_type=int) if joined else []
    tokens = tokens[-max_ctx_tokens:]
    if not tokens:
        tokens = [CONFIG.get("unk_idx", 1)]
    return torch.tensor([tokens], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Decode dispatch
# ─────────────────────────────────────────────────────────────────────────────

def decode(
    model,
    context_tensor: torch.Tensor,
    sp_processor,
    config: dict,
    device: torch.device,
    mode: str,
    beam_width: int,
) -> str:
    src = context_tensor.to(device)
    src_lengths = torch.tensor([src.size(1)], dtype=torch.long)

    sos = config.get("sos_idx", 2)
    eos = config.get("eos_idx", 3)
    max_len = config.get("max_decode_len", 40)

    if mode == "greedy":
        ids = _greedy_decode(model, src, src_lengths, sos, eos, max_len, device)
    elif mode == "beam":
        ids = beam_decode(model, src, src_lengths, sos, eos, max_len, device,
                          beam_width=beam_width)
    else:  # topp (default)
        ids = _topp_decode(model, src, src_lengths, sos, eos, max_len, device,
                           top_p=config.get("top_p", 0.9),
                           temperature=config.get("temperature", 0.8),
                           ngram_block=config.get("ngram_block", 3))

    toks = ids[0] if ids else []
    return sp_processor.decode(toks) if toks else "…"


# ─────────────────────────────────────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────────────────────────────────────

_HELP = """
Runtime commands:
  :mode greedy         switch to greedy (argmax) decoding
  :mode topp           switch to top-p nucleus sampling
  :mode beam [N]       switch to beam search (optional width, default 5)
  :context clear       each question answered independently (no history)
  :context multi       keep rolling conversation history
  :clear               manually clear conversation history now
  :help                show this message
  quit / exit / q      exit
"""


def chat_loop(
    models: Dict[str, torch.nn.Module],   # {"baseline": model, "attention": model}
    sp_processor,
    config: dict,
    device: torch.device,
    decoding_mode: str,
    beam_width: int,
    context_mode: str = "clear",          # "clear" = each Q independent; "multi" = rolling history
):
    mode = decoding_mode
    ctx_mode = context_mode
    history: List[str] = []

    label_w = max(len(k) for k in models) + 2  # column width for labels

    def _mode_str():
        if mode == "beam":
            return f"beam (width={beam_width})"
        return mode

    print()
    print("=" * 65)
    print("  Seq2Seq Ubuntu Chatbot — dual-model comparison")
    print(f"  Models   : {', '.join(models)}")
    print(f"  Decoding : {_mode_str()}")
    print(f"  Context  : {ctx_mode}  (each question {'independent' if ctx_mode == 'clear' else 'uses rolling history'})")
    print("  Type :help for commands, quit to exit")
    print("=" * 65)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Runtime commands ──────────────────────────────────────────────
        if user_input.startswith(":"):
            parts = user_input[1:].split()
            cmd = parts[0].lower() if parts else ""
            if cmd == "help":
                print(_HELP)
            elif cmd == "clear":
                history.clear()
                print("  [history cleared]")
            elif cmd == "context" and len(parts) >= 2:
                new_ctx = parts[1].lower()
                if new_ctx in ("clear", "multi"):
                    ctx_mode = new_ctx
                    label = "independent questions" if ctx_mode == "clear" else "rolling history"
                    print(f"  [context → {ctx_mode} ({label})]")
                else:
                    print("  Unknown context mode. Use: clear | multi")
            elif cmd == "mode" and len(parts) >= 2:
                new_mode = parts[1].lower()
                if new_mode in ("greedy", "topp", "beam"):
                    mode = new_mode
                    if new_mode == "beam" and len(parts) >= 3:
                        try:
                            beam_width = int(parts[2])
                        except ValueError:
                            pass
                    print(f"  [decoding → {_mode_str()}]")
                else:
                    print("  Unknown mode. Use: greedy | topp | beam [N]")
            else:
                print("  Unknown command. Type :help for options.")
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # ── Build context ─────────────────────────────────────────────────
        history.append(user_input)
        context = build_context(
            history, sp_processor,
            config["max_ctx_tokens"],
            config["max_ctx_turns"],
        ).to(device)

        # ── Decode with each model ────────────────────────────────────────
        print()
        history_response = None
        for name, model in models.items():
            response = decode(model, context, sp_processor, config, device,
                              mode, beam_width)
            label = f"[{name}]"
            print(f"  {label:<{label_w}}  {response}")
            if history_response is None:
                history_response = response   # use first model's response for history
        print()

        # Append first model's response to history for next-turn context.
        history.append(history_response or "…")

        # clear mode: discard history after each exchange so the next
        # question is answered without any prior context.
        if ctx_mode == "clear":
            history.clear()
            return_after_clear = False  # flag only for code clarity
        else:
            # Rolling window — keep last max_history_turns pairs.
            max_h = config.get("chat_max_history_turns", 10)
            if len(history) > max_h * 2:
                history = history[-(max_h * 2):]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    default_ckpt_dir = Path(CONFIG.get("checkpoint_dir", "checkpoints"))

    parser = argparse.ArgumentParser(
        description="Dual-model comparison chat with greedy / top-p / beam search."
    )
    parser.add_argument("--baseline",  default=None,
                        metavar="CKPT",
                        help="Path to baseline checkpoint (interactive selection if omitted)")
    parser.add_argument("--attention", default=None,
                        metavar="CKPT",
                        help="Path to attention checkpoint (interactive selection if omitted)")
    parser.add_argument("--checkpoint-dir", default=str(default_ckpt_dir),
                        metavar="DIR",
                        help=f"Directory to scan for checkpoints (default: {default_ckpt_dir})")
    parser.add_argument("--artifact-dir",   default=str(CONFIG.get("artifact_dir", "artifacts")),
                        metavar="DIR",
                        help="Directory containing phase1 artifacts (stage5_spm.model, …)")
    parser.add_argument("--decoding", default="topp",
                        choices=["greedy", "topp", "beam"],
                        help="Decoding strategy (default: topp)")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for beam search (default: 5)")
    parser.add_argument("--context-mode", default="clear",
                        choices=["clear", "multi"],
                        help="clear=each question independent (default); multi=rolling history")
    args = parser.parse_args()

    from gpu_utils import setup_device
    ckpt_dir  = Path(args.checkpoint_dir)
    art_dir   = Path(args.artifact_dir)
    device, _ = setup_device()

    print(f"\nDevice         : {device}")
    print(f"Checkpoint dir : {ckpt_dir}")
    print(f"Artifact dir   : {art_dir}")
    print()

    # ── Select checkpoints ────────────────────────────────────────────────
    print("Selecting checkpoints …")
    baseline_ckpt  = pick_checkpoint(ckpt_dir, "baseline",  args.baseline)
    attention_ckpt = pick_checkpoint(ckpt_dir, "attention", args.attention)

    if baseline_ckpt is None and attention_ckpt is None:
        print("\nNo checkpoints found. Train the models first with: python train.py")
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────────
    sp_processor = load_tokenizer(art_dir)
    print(f"\nTokenizer loaded from: {art_dir}")

    # ── Load models ───────────────────────────────────────────────────────
    models: Dict[str, torch.nn.Module] = {}
    print()
    for model_type, ckpt_path in [("baseline", baseline_ckpt),
                                   ("attention", attention_ckpt)]:
        if ckpt_path is None:
            continue
        model, ckpt = load_model_from_checkpoint(ckpt_path, model_type, CONFIG, device)
        epoch = ckpt.get("epoch", "?")
        vl    = ckpt.get("val_loss", float("nan"))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Loaded {model_type:<10} | {ckpt_path.name}  "
              f"(epoch={epoch}, val_loss={vl:.4f}, params={n_params:,})")
        models[model_type] = model

    # ── Chat ──────────────────────────────────────────────────────────────
    chat_loop(
        models=models,
        sp_processor=sp_processor,
        config=CONFIG,
        device=device,
        decoding_mode=args.decoding,
        beam_width=args.beam_width,
        context_mode=args.context_mode,
    )


if __name__ == "__main__":
    main()
