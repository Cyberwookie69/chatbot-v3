"""
chat.py — Interactive inference for trained Seq2Seq chatbot.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v2.0.0 — Initial version

Loads a checkpoint and runs an interactive terminal chat loop.
Maintains a rolling context window (last N turns) for multi-turn conversation.

Usage:
    python chat.py --model-type attention
    python chat.py --checkpoint checkpoints/baseline_best.pt --model-type baseline
    python chat.py --model-type attention --greedy
"""

import argparse
from pathlib import Path
from typing import List

import torch
import sentencepiece as spm

from config import CONFIG
from models import build_model

# Lazy import of _clean_text from phase1 so chat.py stays lightweight.
# The import executes phase1's module-level code (regex compilation, constants)
# but starts no processes or file I/O.
from phase1 import _clean_text


def build_context(
    history: List[str],
    sp_processor,
    max_ctx_tokens: int,
    max_turns: int,
) -> torch.Tensor:
    """
    Build context tensor from conversation history.

    Mirrors phase1 stage 4 context construction exactly:
      - Each turn is passed through _clean_text (same normalisation as training)
      - Turns are joined with ' __eot__ ' (Lowe et al. 2015 Ubuntu convention)
      - The joined string is tokenised as a single sequence and right-truncated
        to max_ctx_tokens

    Returns a [1, ctx_len] LongTensor.
    """
    turns = history[-max_turns:] if history else []
    if turns:
        cleaned = [_clean_text(t) for t in turns]
        # Drop any turns that cleaned to empty string.
        cleaned = [t for t in cleaned if t]
        joined = " __eot__ ".join(cleaned) if cleaned else ""
    else:
        joined = ""

    tokens: List[int] = sp_processor.encode(joined, out_type=int) if joined else []

    # Right-truncate to max_ctx_tokens.
    tokens = tokens[-max_ctx_tokens:]

    if not tokens:
        # Empty history fallback: single UNK token so encoder never sees empty input.
        tokens = [CONFIG.get("unk_idx", 1)]

    return torch.tensor([tokens], dtype=torch.long)  # [1, ctx_len]


def respond(
    model,
    context_tensor: torch.Tensor,
    sp_processor,
    config: dict,
    device: torch.device,
    greedy: bool = False,
) -> str:
    """
    Generate one response string from context_tensor.

    greedy=False: top-p nucleus sampling with temperature and n-gram blocking
    greedy=True:  argmax at each step (for reproducible / comparable output)
    """
    from evaluate import greedy_decode, top_p_decode

    src = context_tensor.to(device)
    src_lengths = torch.tensor([src.size(1)], dtype=torch.long)

    if greedy:
        decoded_ids = greedy_decode(
            model, src, src_lengths,
            sos_idx=config.get("sos_idx", 2),
            eos_idx=config.get("eos_idx", 3),
            max_len=config.get("max_decode_len", 40),
            device=device,
        )
    else:
        decoded_ids = top_p_decode(
            model, src, src_lengths,
            sos_idx=config.get("sos_idx", 2),
            eos_idx=config.get("eos_idx", 3),
            max_len=config.get("max_decode_len", 40),
            device=device,
            top_p=config.get("top_p", 0.9),
            temperature=config.get("temperature", 0.8),
            ngram_block=config.get("ngram_block", 3),
        )

    ids = decoded_ids[0] if decoded_ids else []
    return sp_processor.decode(ids) if ids else "…"


def chat_loop(model, sp_processor, config: dict, device: torch.device, greedy: bool):
    """Run the interactive chat REPL."""
    print()
    print("=" * 60)
    print("  Ubuntu Dialogue Chatbot")
    print("  Decoding:", "greedy" if greedy else "top-p (p=0.9, t=0.8, ngram_block=3)")
    print("  Type 'quit' or Ctrl+C to exit")
    print("=" * 60)
    print()

    # history stores raw strings; _clean_text is applied inside build_context.
    history: List[str] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        history.append(user_input)

        context = build_context(
            history,
            sp_processor,
            config["max_ctx_tokens"],
            config["max_ctx_turns"],
        ).to(device)

        response = respond(model, context, sp_processor, config, device, greedy)
        print(f"Bot: {response}")

        # Add bot response to history so next turn has full conversation context.
        history.append(response)

        # Rolling window: keep at most max_history_turns *pairs* (user + bot each).
        max_history = config.get("chat_max_history_turns", 10)
        if len(history) > max_history * 2:
            history = history[-(max_history * 2):]


def main():
    default_ckpt_dir = Path(CONFIG.get("checkpoint_dir", "checkpoints"))

    parser = argparse.ArgumentParser(
        description="Interactive chat with a trained Seq2Seq Ubuntu chatbot."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to .pt checkpoint. "
            "Defaults to {checkpoint_dir}/{model_type}_best.pt"
        ),
    )
    parser.add_argument(
        "--model-type",
        default="attention",
        choices=["attention", "baseline"],
        help="Model architecture to load (default: attention)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(CONFIG.get("artifact_dir", "artifacts")),
        help="Directory containing phase1 artifacts (stage5_spm.model, etc.)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy (argmax) decoding instead of top-p sampling",
    )
    args = parser.parse_args()

    # Resolve checkpoint path.
    if args.checkpoint is None:
        ckpt_path = default_ckpt_dir / f"{args.model_type}_best.pt"
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first with: python train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    # Load SentencePiece model.
    sp_path = Path(args.artifact_dir) / "stage5_spm.model"
    if not sp_path.exists():
        raise FileNotFoundError(
            f"SPM model not found: {sp_path}\n"
            f"Run phase1.py first to generate artifacts."
        )
    sp_processor = spm.SentencePieceProcessor(model_file=str(sp_path))
    print(f"SPM loaded  : {sp_path}")

    # Load model from checkpoint.
    model = build_model(args.model_type, CONFIG, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    trained_epoch = ckpt.get("epoch", "?")
    best_val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Model       : {args.model_type}  (epoch {trained_epoch}, val_loss={best_val_loss:.4f})")
    print(f"Checkpoint  : {ckpt_path}")

    chat_loop(model, sp_processor, CONFIG, device, greedy=args.greedy)


if __name__ == "__main__":
    main()
