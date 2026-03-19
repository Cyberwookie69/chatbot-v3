"""
tokenizer_utils.py — Unified tokenizer interface for BPE encoding/decoding.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Unified HF Tokenizers + SentencePiece interface
          v3.0.0 — Initial version

Supports both HuggingFace Tokenizers (fast, Rust-based) and SentencePiece
(legacy fallback). All consumers (phase1 stages 6-7, chatv2, evaluate) use
load_tokenizer() to get a tokenizer with a consistent API.

Token ID contract: pad=0, unk=1, sos=2, eos=3
"""

from pathlib import Path
from typing import List, Optional, Union


class HFTokenizerWrapper:
    """Wraps a HuggingFace tokenizers.Tokenizer to expose the SentencePiece API.

    This allows stages 6, 7, chatv2, and evaluate to work unchanged —
    they call .encode(), .decode(), .piece_to_id(), .id_to_piece(), etc.
    """

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._vocab = self._tok.get_vocab()
        self._id2piece = {v: k for k, v in self._vocab.items()}

    def encode(self, input, out_type=int):
        """Encode text(s) to token IDs or piece strings.

        Supports both single string and list-of-strings (batch) input,
        matching the SentencePiece API.
        """
        if isinstance(input, str):
            enc = self._tok.encode(input)
            if out_type == int:
                return enc.ids
            return enc.tokens
        # Batch mode
        encoded = self._tok.encode_batch(input)
        if out_type == int:
            return [e.ids for e in encoded]
        return [e.tokens for e in encoded]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self._tok.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        """Return the integer ID for a piece string."""
        return self._vocab.get(piece, self._vocab.get("<unk>", 1))

    def id_to_piece(self, id: int) -> str:
        """Return the piece string for an integer ID."""
        return self._id2piece.get(id, "<unk>")

    def get_piece_size(self) -> int:
        """Return vocabulary size."""
        return self._tok.get_vocab_size()


def train_bpe_tokenizer(
    sentence_iterator,
    model_prefix: str,
    vocab_size: int = 16000,
    special_tokens: Optional[List[str]] = None,
    user_defined_symbols: Optional[List[str]] = None,
):
    """Train a BPE tokenizer using HuggingFace Tokenizers (Rust, multithreaded).

    Produces a .json tokenizer file compatible with HFTokenizerWrapper.
    Guarantees the token ID contract: pad=0, unk=1, sos=2, eos=3.

    Args:
        sentence_iterator: Iterator yielding text lines.
        model_prefix: Output path prefix (produces {model_prefix}.json).
        vocab_size: Target vocabulary size.
        special_tokens: Override special tokens. Default: [<pad>, <unk>, <sos>, <eos>].
        user_defined_symbols: Additional tokens to guarantee as single pieces.

    Returns:
        Path to the saved .json tokenizer file.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    # Build special tokens list with guaranteed order: pad=0, unk=1, sos=2, eos=3
    base_specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
    extra = user_defined_symbols or []
    all_specials = base_specials + [s for s in extra if s not in base_specials]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=all_specials,
        show_progress=True,
    )

    # Collect sentences into a list (HF trainer needs iterable)
    sentences = list(sentence_iterator)
    tokenizer.train_from_iterator(sentences, trainer=trainer)

    # Verify token ID contract
    assert tokenizer.token_to_id("<pad>") == 0, f"<pad> ID mismatch: {tokenizer.token_to_id('<pad>')}"
    assert tokenizer.token_to_id("<unk>") == 1, f"<unk> ID mismatch: {tokenizer.token_to_id('<unk>')}"
    assert tokenizer.token_to_id("<sos>") == 2, f"<sos> ID mismatch: {tokenizer.token_to_id('<sos>')}"
    assert tokenizer.token_to_id("<eos>") == 3, f"<eos> ID mismatch: {tokenizer.token_to_id('<eos>')}"

    out_path = model_prefix + ".json"
    tokenizer.save(out_path)
    return out_path


def load_tokenizer(artifact_dir: Union[str, Path]):
    """Load the best available tokenizer from artifact_dir.

    Prefers HF Tokenizer (.json) for speed; falls back to SentencePiece (.model).

    Returns an object with .encode(), .decode(), .piece_to_id(),
    .id_to_piece(), .get_piece_size() methods.
    """
    artifact_dir = Path(artifact_dir)

    hf_path = artifact_dir / "stage5_spm.json"
    spm_path = artifact_dir / "stage5_spm.model"

    if hf_path.exists():
        try:
            return HFTokenizerWrapper(str(hf_path))
        except ImportError:
            pass  # tokenizers not installed, fall through to SPM

    if spm_path.exists():
        import sentencepiece as spm
        return spm.SentencePieceProcessor(model_file=str(spm_path))

    raise FileNotFoundError(
        f"No tokenizer found in {artifact_dir}. "
        f"Expected stage5_spm.json or stage5_spm.model"
    )
