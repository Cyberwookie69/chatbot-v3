"""
corpus_loaders.py — Load and filter additional training corpora into (ctx, resp) pairs.

Each loader converts its corpus into the same {"ctx": str, "resp": str} format
that phase1.py Stage 4 produces, so pairs can be merged before Stage 5 (BPE training).

Supported corpora:
  - WikiQA (Microsoft Research) — TSV, label-filtered
  - Question-Answer Dataset (CMU/rtatman) — TSV, difficulty-filtered
  - CoQA (Stanford) — JSON, conversational Q&A
  - English Movie Subtitles (OpenSubtitles) — XML, consecutive-line pairing
  - Movie Dialog Corpus (Cornell) — custom delimited, conversation-structured

All loaders apply corpus-specific filters plus shared quality filters:
  - Minimum context/response length
  - Echo pair removal
  - Non-English filtering
  - Response diversity capping
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Shared filter settings ───────────────────────────────────────────────────

_DEFAULT_CFG = {
    "min_ctx_words": 3,
    "min_resp_words": 3,
    "max_resp_words": 60,
    "max_ctx_words": 150,
    "max_response_occurrences": 500,
    "non_english_max_ratio": 0.3,     # max fraction of non-ASCII chars
}

# ── Shared regex patterns ────────────────────────────────────────────────────

_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_BRACKETS = re.compile(r"\[.*?\]")           # [Music], [Applause], etc.
_RE_PARENS = re.compile(r"\(.*?\)")             # (sighs), (laughs), etc.
_RE_MUSIC = re.compile(r"♪[^♪]*♪?|♫[^♫]*♫?")
_RE_TIMESTAMP = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?")
_RE_SUBTITLE_NUMBERING = re.compile(r"^\d+\s*$")


# ── Shared filters ───────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Basic text cleanup shared across all corpora."""
    text = _RE_HTML_TAGS.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def _is_english(text: str, max_ratio: float = 0.3) -> bool:
    """Check if text is mostly ASCII (proxy for English)."""
    if not text:
        return False
    non_ascii = len(_RE_NON_ASCII.findall(text))
    return (non_ascii / len(text)) <= max_ratio


def _is_echo(ctx: str, resp: str) -> bool:
    """Check if response is a verbatim copy of context."""
    return ctx.strip().lower() == resp.strip().lower()


def _word_count(text: str) -> int:
    return len(text.split())


def _apply_shared_filters(
    pairs: List[Dict[str, str]],
    cfg: dict,
    corpus_name: str,
) -> List[Dict[str, str]]:
    """Apply shared quality filters to a list of (ctx, resp) pairs."""
    min_ctx = cfg.get("min_ctx_words", _DEFAULT_CFG["min_ctx_words"])
    min_resp = cfg.get("min_resp_words", _DEFAULT_CFG["min_resp_words"])
    max_resp = cfg.get("max_resp_words", _DEFAULT_CFG["max_resp_words"])
    max_ctx = cfg.get("max_ctx_words", _DEFAULT_CFG["max_ctx_words"])
    max_occ = cfg.get("max_response_occurrences", _DEFAULT_CFG["max_response_occurrences"])
    non_eng_ratio = cfg.get("non_english_max_ratio", _DEFAULT_CFG["non_english_max_ratio"])

    filtered = []
    resp_counts: Counter = Counter()
    stats = {"total": len(pairs), "too_short_ctx": 0, "too_short_resp": 0,
             "too_long_ctx": 0, "too_long_resp": 0, "echo": 0,
             "non_english": 0, "diversity_cap": 0, "kept": 0}

    for p in pairs:
        ctx, resp = p["ctx"], p["resp"]

        if _word_count(ctx) < min_ctx:
            stats["too_short_ctx"] += 1
            continue
        if _word_count(resp) < min_resp:
            stats["too_short_resp"] += 1
            continue
        if _word_count(ctx) > max_ctx:
            stats["too_long_ctx"] += 1
            continue
        if _word_count(resp) > max_resp:
            stats["too_long_resp"] += 1
            continue
        if _is_echo(ctx, resp):
            stats["echo"] += 1
            continue
        if not _is_english(resp, non_eng_ratio):
            stats["non_english"] += 1
            continue

        resp_lower = resp.strip().lower()
        if resp_counts[resp_lower] >= max_occ:
            stats["diversity_cap"] += 1
            continue
        resp_counts[resp_lower] += 1

        filtered.append(p)
        stats["kept"] += 1

    print(f"  [{corpus_name}] {stats['total']:,} raw → {stats['kept']:,} kept")
    for k, v in stats.items():
        if k not in ("total", "kept") and v > 0:
            print(f"    {k}: {v:,}")

    return filtered


# ── WikiQA loader ────────────────────────────────────────────────────────────

def load_wikiqa(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load WikiQA corpus: question → answer pairs (label=1 only).

    Format: TSV with columns QuestionID, Question, DocumentID,
    DocumentTitle, SentenceID, Sentence, Label.
    """
    corpus_dir = data_dir / "wikiqa-corpus"
    if not corpus_dir.exists():
        print(f"  WikiQA not found at {corpus_dir} — skipping")
        return []

    # Find TSV files
    tsv_files = list(corpus_dir.rglob("*.tsv"))
    if not tsv_files:
        # Try txt files (some versions use .txt)
        tsv_files = list(corpus_dir.rglob("WikiQA*.txt"))
    if not tsv_files:
        print(f"  No WikiQA TSV files found in {corpus_dir} — skipping")
        return []

    pairs = []
    for tsv_path in tsv_files:
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            # Find column indices
            try:
                q_idx = header.index("Question")
                s_idx = header.index("Sentence")
                l_idx = header.index("Label")
            except ValueError:
                print(f"  Unexpected WikiQA header: {header} — skipping {tsv_path.name}")
                continue

            for line in f:
                cols = line.strip().split("\t")
                if len(cols) <= max(q_idx, s_idx, l_idx):
                    continue
                label = cols[l_idx].strip()
                if label != "1":
                    continue  # Only keep correct answers

                question = _clean(cols[q_idx])
                answer = _clean(cols[s_idx])
                if question and answer:
                    pairs.append({"ctx": question, "resp": answer})

    # Deduplicate: same question may have multiple correct answers — keep all
    # but apply shared filters
    return _apply_shared_filters(pairs, cfg, "WikiQA")


# ── Question-Answer Dataset loader ──────────────────────────────────────────

def load_question_answer(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load CMU/rtatman Question-Answer dataset.

    Structure: subdirectories per article, each with question_answer_pairs.txt.
    Format: TSV with ArticleTitle, Question, Answer, DifficultyFromQuestioner,
    DifficultyFromAnswerer, ArticleFile.
    """
    corpus_dir = data_dir / "question-answer-dataset"
    if not corpus_dir.exists():
        print(f"  Question-Answer dataset not found at {corpus_dir} — skipping")
        return []

    qa_files = list(corpus_dir.rglob("question_answer_pairs.txt"))
    if not qa_files:
        # Try alternative structure — single file
        qa_files = list(corpus_dir.rglob("*.txt"))
        qa_files = [f for f in qa_files if "article" not in f.name.lower()
                     and "readme" not in f.name.lower()]
    if not qa_files:
        print(f"  No Q&A files found in {corpus_dir} — skipping")
        return []

    pairs = []
    min_difficulty = cfg.get("qa_min_difficulty", 1)  # Filter trivial questions

    for qa_path in qa_files:
        try:
            with open(qa_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline()
                for line in f:
                    cols = line.strip().split("\t")
                    if len(cols) < 3:
                        continue
                    question = _clean(cols[1])
                    answer = _clean(cols[2])

                    # Filter by difficulty if available
                    if len(cols) >= 5:
                        try:
                            diff_q = float(cols[3]) if cols[3].strip() else 0
                            diff_a = float(cols[4]) if cols[4].strip() else 0
                            avg_diff = (diff_q + diff_a) / 2 if diff_q and diff_a else max(diff_q, diff_a)
                            if avg_diff < min_difficulty:
                                continue
                        except ValueError:
                            pass  # Non-numeric difficulty — keep the pair

                    if question and answer and answer.lower() not in ("", "?", "unknown"):
                        pairs.append({"ctx": question, "resp": answer})
        except Exception as e:
            print(f"  Warning: could not read {qa_path.name}: {e}")
            continue

    return _apply_shared_filters(pairs, cfg, "Question-Answer")


# ── CoQA loader ──────────────────────────────────────────────────────────────

_TRIVIAL_ANSWERS = frozenset({
    "yes", "no", "yes.", "no.", "unknown", "unknown.", "none", "none.",
})


def load_coqa(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load CoQA conversational Q&A dataset.

    Format: JSON with story + multi-turn questions/answers.
    Generates pairs using conversational context (previous Q&A turns).
    """
    corpus_dir = data_dir / "coqa"
    if not corpus_dir.exists():
        print(f"  CoQA not found at {corpus_dir} — skipping")
        return []

    json_files = list(corpus_dir.rglob("*.json"))
    if not json_files:
        print(f"  No CoQA JSON files found in {corpus_dir} — skipping")
        return []

    filter_trivial = cfg.get("coqa_filter_trivial", True)
    max_ctx_turns = cfg.get("coqa_max_ctx_turns", 5)

    pairs = []
    trivial_skipped = 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: could not parse {json_path.name}: {e}")
            continue

        entries = data.get("data", [])
        for entry in entries:
            questions = entry.get("questions", [])
            answers = entry.get("answers", [])

            # Build conversational context from previous turns
            history: List[str] = []
            for q, a in zip(questions, answers):
                q_text = _clean(q.get("input_text", ""))
                a_text = _clean(a.get("input_text", ""))

                if not q_text or not a_text:
                    continue

                # Filter trivial answers
                if filter_trivial and a_text.strip().lower() in _TRIVIAL_ANSWERS:
                    trivial_skipped += 1
                    history.append(q_text)
                    history.append(a_text)
                    continue

                # Build context from recent history + current question
                ctx_turns = history[-max_ctx_turns * 2:]  # pairs of Q+A
                ctx_turns.append(q_text)
                ctx = " __eot__ ".join(ctx_turns)

                pairs.append({"ctx": ctx, "resp": a_text})

                # Add to history for next turn
                history.append(q_text)
                history.append(a_text)

    if trivial_skipped:
        print(f"  [CoQA] Skipped {trivial_skipped:,} trivial answers (yes/no/unknown)")

    return _apply_shared_filters(pairs, cfg, "CoQA")


# ── English Movie Subtitles loader ───────────────────────────────────────────

def _clean_subtitle_line(text: str) -> str:
    """Clean a single subtitle line: remove tags, music symbols, timestamps."""
    text = _RE_HTML_TAGS.sub("", text)
    text = _RE_BRACKETS.sub("", text)       # [Music], [Applause]
    text = _RE_PARENS.sub("", text)         # (sighs), (laughs)
    text = _RE_MUSIC.sub("", text)          # ♪...♪
    text = _RE_TIMESTAMP.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def _is_non_dialogue(text: str) -> bool:
    """Detect non-dialogue subtitle lines."""
    lower = text.lower().strip()
    if not lower:
        return True
    # Scene descriptions, credits, etc.
    if lower.startswith(("subtitle", "caption", "translated by", "synced by",
                         "corrected by", "encoded by", "ripped by")):
        return True
    if _RE_SUBTITLE_NUMBERING.match(lower):
        return True
    # All-caps lines are often titles or scene headers
    if text.isupper() and len(text.split()) <= 3:
        return True
    return False


def load_movie_subtitles(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load English Movie Subtitle dataset (OpenSubtitles XML format).

    Pairs consecutive subtitle lines as context → response.
    Groups nearby lines (within 4 seconds) as multi-line context.
    """
    corpus_dir = data_dir / "english-movie-subtitles"
    if not corpus_dir.exists():
        print(f"  Movie Subtitles not found at {corpus_dir} — skipping")
        return []

    xml_files = list(corpus_dir.rglob("*.xml"))
    if not xml_files:
        print(f"  No XML files found in {corpus_dir} — skipping")
        return []

    print(f"  [Subtitles] Processing {len(xml_files)} XML files …")
    pairs = []
    context_window = cfg.get("subtitle_context_lines", 3)

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Extract all subtitle lines
        lines = []
        for s_elem in root.iter("s"):
            text = "".join(s_elem.itertext()).strip()
            text = _clean_subtitle_line(text)
            if text and not _is_non_dialogue(text):
                lines.append(text)

        # If no XML structure, try plain text lines
        if not lines:
            try:
                with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
                    for raw_line in f:
                        text = _clean_subtitle_line(raw_line)
                        if text and not _is_non_dialogue(text):
                            lines.append(text)
            except Exception:
                continue

        # Generate pairs from consecutive lines using sliding window
        for i in range(context_window, len(lines)):
            ctx_lines = lines[max(0, i - context_window):i]
            ctx = " __eot__ ".join(ctx_lines)
            resp = lines[i]
            pairs.append({"ctx": ctx, "resp": resp})

    return _apply_shared_filters(pairs, cfg, "Movie Subtitles")


# ── Cornell Movie Dialog Corpus loader ───────────────────────────────────────

_CORNELL_SEP = " +++$+++ "


def load_movie_dialog(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load Cornell Movie Dialog Corpus.

    Uses movie_lines.txt + movie_conversations.txt to reconstruct
    dialogues and generate context-response pairs.
    """
    corpus_dir = data_dir / "movie-dialog-corpus"
    if not corpus_dir.exists():
        print(f"  Movie Dialog Corpus not found at {corpus_dir} — skipping")
        return []

    lines_path = None
    conv_path = None
    for candidate in corpus_dir.rglob("movie_lines.txt"):
        lines_path = candidate
    for candidate in corpus_dir.rglob("movie_conversations.txt"):
        conv_path = candidate

    if not lines_path or not conv_path:
        print(f"  movie_lines.txt or movie_conversations.txt not found in {corpus_dir} — skipping")
        return []

    # Load all lines into a dict: lineID → text
    line_texts: Dict[str, str] = {}
    with open(lines_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.strip().split(_CORNELL_SEP)
            if len(parts) >= 5:
                line_id = parts[0].strip()
                text = _clean(parts[4].strip())
                if text:
                    line_texts[line_id] = text

    print(f"  [Cornell] Loaded {len(line_texts):,} lines")

    # Load conversations and generate pairs
    max_ctx_turns = cfg.get("cornell_max_ctx_turns", 4)
    pairs = []

    with open(conv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.strip().split(_CORNELL_SEP)
            if len(parts) < 4:
                continue

            # Parse line ID list: ['L1', 'L2', 'L3']
            line_ids_str = parts[3].strip()
            try:
                line_ids = [lid.strip().strip("'\"") for lid in
                           line_ids_str.strip("[]").split(",")]
            except Exception:
                continue

            # Reconstruct conversation text
            conv_lines = []
            for lid in line_ids:
                lid = lid.strip()
                if lid in line_texts:
                    conv_lines.append(line_texts[lid])

            if len(conv_lines) < 2:
                continue

            # Generate (context, response) pairs with sliding window
            for i in range(1, len(conv_lines)):
                ctx_start = max(0, i - max_ctx_turns)
                ctx = " __eot__ ".join(conv_lines[ctx_start:i])
                resp = conv_lines[i]
                pairs.append({"ctx": ctx, "resp": resp})

    return _apply_shared_filters(pairs, cfg, "Cornell Movie Dialog")


# ── Main entry point ─────────────────────────────────────────────────────────

def load_all_extra_corpora(data_dir: Path, cfg: dict) -> List[Dict[str, str]]:
    """Load all additional corpora and return merged pairs.

    Args:
        data_dir: Parent data directory (e.g., .../data/).
        cfg:      Config dict with filter settings.

    Returns:
        Combined list of {"ctx": str, "resp": str} pairs from all corpora.
    """
    print("=" * 60)
    print("STAGE 4.6 — Load additional corpora")
    print("=" * 60)

    all_pairs: List[Dict[str, str]] = []

    loaders = [
        ("WikiQA",              load_wikiqa),
        ("Question-Answer",     load_question_answer),
        ("CoQA",                load_coqa),
        ("Movie Subtitles",     load_movie_subtitles),
        ("Cornell Movie Dialog", load_movie_dialog),
    ]

    for name, loader_fn in loaders:
        try:
            pairs = loader_fn(data_dir, cfg)
            if pairs:
                all_pairs.extend(pairs)
                print(f"  ✓ {name}: {len(pairs):,} pairs")
            else:
                print(f"  – {name}: 0 pairs (not found or empty)")
        except Exception as e:
            print(f"  ✗ {name}: failed ({e})")

    print(f"\n  Total extra corpus pairs: {len(all_pairs):,}\n")
    return all_pairs
