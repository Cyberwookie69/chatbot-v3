"""
phase1.py — Data pipeline for clean-from-scratch Ubuntu Dialogue Corpus processing.

Version : 3.2.0
Modified: 2026-03-18
Changes : v3.2.0 — Word2Vec replaces FastText, HF Tokenizers for Stage 5,
                    lru_cache on date parsing, precompiled regex, orjson support,
                    auto-cleanup of intermediate files, progress bars
          v3.1.0 — DuckDB-accelerated Stage 1, fork/spawn fix for OpenShift AI
          v3.0.0 — Jupyter notebook conversion, multi-GPU support, OpenShift AI compat
          v2.0.0 — Initial clean-from-scratch rewrite

Stages (run once, produces artifacts in ARTIFACT_DIR):
  Stage 1 — Load raw Ubuntu Dialogue Corpus CSV files into structured dialogues
  Stage 2 — Clean text + apply quality filters (parallel, chunked)
  Stage 3 — Temporal split (train / val / test) by THREAD first-turn date (G6 fix)
  Stage 4 — Generate context-response pairs + response diversity filter
  Stage 4.5 — Domain-focused filtering: retain command-line OR question pairs
  Stage 5 — Train SentencePiece BPE tokenizer on raw training text (16k vocab)
  Stage 6 — Encode all pairs to BPE token IDs + save vocab JSON
  Stage 7 — Train Word2Vec 300d skip-gram on BPE-tokenised corpus
  Stage 8 — Build embedding matrix aligned to BPE vocab → .npy matrix

Key decisions vs old phase1.py:
  - SentencePiece BPE replaces word-level vocab (smaller softmax, zero OOV)
  - Word2Vec on BPE tokens (subword n-grams already handled by BPE, FastText overhead unnecessary)
  - Response diversity filter caps repeat responses (reduces "averaging signal")
  - Max context: 100 tokens / 8 turns (tighter = more specific)
  - Max response: 40 tokens
  - Target ~1.5M train pairs (quality > quantity vs prior 4.57M)
  - Temporal split on THREAD (dialogue) first-turn date — no thread boundary leakage

Quality filters:
  - filter_paste:           drop pasted terminal output blocks
  - filter_irc_actions:     drop /me emote turns
  - filter_repetitive:      drop "help help help" style turns
  - filter_temporal:        drop dialogues with large timestamp gaps
  - dyadic_only:            only 2-speaker conversations
  - filter_echo_pairs:      skip verbatim context-response copies
  - filter_placeholder:     skip placeholder-only responses
  - filter_non_english:     drop mostly non-ASCII responses
  - filter_bot_responses:   drop known boilerplate canned responses
  - filter_response_diversity: cap max occurrences of any identical response

Token ID contract (must match config.py):
  pad=0, unk=1, sos=2, eos=3  (enforced via explicit SPM trainer parameters)
"""

from __future__ import annotations

import csv
import functools
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Resolve paths relative to this file (works from any working directory) ───
_NEW_DIR = Path(__file__).resolve().parent          # .../NLP_Final_Project_v2/new/
_PROJECT_DIR = _NEW_DIR.parent                      # .../NLP_Final_Project_v2/

# ── CONFIG ────────────────────────────────────────────────────────────────────

PHASE1_CONFIG = {
    # Corpus source — data/ lives inside new/ alongside this file
    "corpus_dir":                    str(_NEW_DIR / "data" / "Ubuntu-dialogue-corpus"),

    # Dialogue-level quality filters (set False to disable individually)
    "min_turns":                     2,
    "dyadic_only":                   True,
    "filter_paste":                  True,
    "filter_irc_actions":            True,
    "filter_repetitive":             True,
    "filter_temporal":               True,
    "max_turn_gap_seconds":          3600,   # hard ceiling per gap; 0 = disabled
    "large_gap_threshold":           600,    # soft threshold (seconds) for ratio check
    "max_large_gap_ratio":           0.3,    # fraction of gaps allowed to exceed threshold

    # Pair-level quality filters
    "filter_echo_pairs":             True,
    "filter_placeholder":            True,
    "filter_non_english":            True,
    "filter_bot_responses":          True,

    # Response diversity cap (train set only)
    "filter_response_diversity":     True,
    "max_response_occurrences":      500,

    # Pair generation dimensions
    "max_ctx_tokens":                100,
    "max_ctx_turns":                 8,
    "max_resp_tokens":               40,
    "min_resp_tokens":               5,    # raised from 3 — eliminates 3-4 token fragments
    "min_ctx_tokens":                3,    # discard pairs where full ctx < 3 words (degenerate: "yes", "__url__", "hello")
    "max_train_pairs":               1_500_000,   # 0 = no cap

    # Temporal split boundaries — split by THREAD first-turn date.
    # Calibrated to the Ubuntu Dialogue Corpus (2004-08 → 2012-11).
    # These dates give ~80% train / ~10% val / ~10% test.
    "train_cutoff_date":             "2012-04-27",
    "val_cutoff_date":               "2012-08-07",

    # SentencePiece BPE — special token IDs guaranteed via explicit params
    "spm_vocab_size":                16000,
    "spm_model_type":                "bpe",
    "spm_character_coverage":        0.9999,
    "spm_input_sentence_size":       2_000_000,   # cap SPM training lines (avoids "too many" warning)
    # NOTE: do NOT add spm_user_defined_symbols — use explicit pad_id/bos_id/eos_id
    # Dialogue-level speaker quality filters
    "max_single_speaker_ratio":      0.80,  # drop monologue-style dialogues (1 speaker > 80%)
    "min_alternation_ratio":         0.15,  # drop non-alternating floods (A,A,A,B,B,B style)

    # FastText
    "fasttext_dim":                  300,
    "fasttext_epochs":               10,    # 10 matches original; 5 was too few
    "fasttext_min_count":            3,     # 3 filters hapax noise; 1 was too inclusive
    "fasttext_window":               5,     # explicit context window
    "fasttext_sg":                   1,     # skip-gram > CBOW for rare BPE pieces
    "fasttext_workers":              8,

    # Pipeline internals
    "artifact_dir":                  str(_NEW_DIR / "artifacts"),
    "log_dir":                       str(_NEW_DIR / "logs"),
    "num_workers":                   4,
    "chunk_size":                    50_000,

    # Mini-mode subsample — fraction of stage 1 dialogues to keep.
    # Set to 0.10 in phase1_mini.py for a ~10x faster pipeline run.
    # 1.0 = use all dialogues (default full run).
    "stage1_subsample_frac":         1.0,

    # ── Stage 4.5: Domain-focused filtering ──────────────────────────────────
    # Filters (ctx, resp) pairs to retain only domain-relevant content.
    # Applied in-memory after Stage 4 generates pairs, before Stage 5 trains
    # the SentencePiece model, so vocabulary reflects the filtered domain.
    #
    # Strategies:
    #   "command"      — retain pairs where ctx OR resp contains a Linux command
    #                    (regex on command names) or a __path__ token
    #   "question"     — retain pairs where the last substantive ctx turn
    #                    contains a question pattern (how/what/why/etc.)
    #   "union"        — retain pairs matching EITHER strategy (recommended)
    #   "intersection" — retain pairs matching BOTH strategies (highest precision,
    #                    smallest yield ~300k — too small for 44M params)
    #   "none"         — disable filtering entirely (default — preserves existing
    #                    behaviour for backwards compatibility)
    "domain_filter":            True,         # Stage 4.5: retain command/question pairs only
    "domain_filter_strategy":   "union",      # "command" | "question" | "union" | "intersection"

    # ── Stage 4.6: Additional corpora ──────────────────────────────────────
    "extra_corpora":            True,         # Stage 4.6: load WikiQA, CoQA, subtitles, etc.
    "qa_min_difficulty":        1,            # Question-Answer: min difficulty filter
    "coqa_filter_trivial":      True,         # CoQA: filter yes/no/unknown answers
    "coqa_max_ctx_turns":       5,            # CoQA: max previous Q&A turns in context
    "subtitle_context_lines":   3,            # Subtitles: lines of context before response
    "cornell_max_ctx_turns":    4,            # Cornell: max previous turns in context
}

# ── Stage expected artifacts ──────────────────────────────────────────────────

STAGE_ARTIFACTS = {
    1: ["stage1_dialogues.pkl"],
    2: ["stage2_clean_dialogues.pkl", "stage2_stats.json"],
    3: ["stage3_train.pkl", "stage3_val.pkl", "stage3_test.pkl", "stage3_stats.json"],
    4: ["stage4_train_pairs.pkl", "stage4_val_pairs.pkl",
        "stage4_test_pairs.pkl", "stage4_stats.json"],
    4.5: ["stage4_5_train_pairs.pkl", "stage4_5_val_pairs.pkl",
          "stage4_5_test_pairs.pkl", "stage4_5_filter_stats.json"],
    # Stage 5 produces either .json (HF Tokenizers) or .model+.vocab (SentencePiece).
    # _stage_done(5) is overridden below to check for either format.
    5: [],
    6: ["stage6_train_ids.jsonl", "stage6_val_ids.jsonl",
        "stage6_test_ids.jsonl", "stage6_vocab.json", "stage6_idx2word.json", "stage6_stats.json"],
    7: ["stage7_fasttext.model"],
    8: ["stage8_embedding_matrix.npy", "stage8_stats.json"],
}

# ── Known IRC bots (turns from these speakers are dropped in stage 2) ─────────

_BOTS = frozenset({
    "ubottu", "ubotu", "ubot5", "ubot3", "ubot4", "ubot2",
    "chanserv", "nickserv", "memoserv", "operserv",
    "logbot", "meetingology", "supybot",
})

# ── Bot/boilerplate response blacklist (matched against cleaned, lowercased text)

_BOT_RESPONSE_BLACKLIST = frozenset({
    # Flood / paste warnings (ubottu)
    "please do not flood use __url__ to paste do not use enter as punctuation",
    "please do not flood the channel use __url__ to paste",
    "please do not flood",
    "i am a bot please do not message me",
    "you can use __url__ to paste",
    # Moderation / op scripted responses
    "watch your language",
    "please keep the channel family friendly",
    "please keep this channel on-topic",
    "this is not the right channel for that",
    "this is ubuntu support only",
    "this channel is for ubuntu support",
    "please take offtopic chat",
    "please use #ubuntu-offtopic",
    "try #ubuntu-offtopic",
    "not ubuntu related",
    "that is not ubuntu related",
    # Join / leave / mode announcements (occasionally leak through)
    "has joined",
    "has quit",
    "has left",
    "was kicked",
    "changed the topic",
    # ubottu factoid pattern prefix
    "ubuntu is",
    "please see",
})

# ── Compiled regex patterns ───────────────────────────────────────────────────

_RE_URL      = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(com|org|net|io|edu|gov)\S*", re.IGNORECASE)
_RE_PATH     = re.compile(r"(?:/[a-zA-Z0-9._~-]+){2,}")
_RE_IP       = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b")  # IPv4 + optional port
_RE_IRC_NICK = re.compile(r"^<[^>]+>\s*")
_RE_NONALPHA = re.compile(r"[^a-z0-9 '\-_.]+")
_RE_MULTI_SP = re.compile(r"\s{2,}")
_RE_ACTION   = re.compile(r"^\*\s*\S+\s+")   # IRC /me emotes: "* nick does thing"
_RE_IRC_ADDR = re.compile(r"^[a-z][a-z0-9_\-\[\]\\^{}|`]{1,25}\s*[:,]\s*")  # "nick: msg" → "msg"
_RE_SPEAKER_SPECIAL = re.compile(r"[\d_\-\[\]\\^{}|`]")  # speaker names needing masking
_RE_DOTS     = re.compile(r'\.{2,}')         # collapse ".." → "."
_RE_SENT_END = re.compile(r'([a-z0-9])\.((?:\s|$))')  # "word." → "word ."
_RE_PLACEHOLDER_ONLY = re.compile(
    r"^(__url__|__path__|__ip__|__cmd__|__number__|__user__|\s)+$"
)
# Matches known IRC bot names (including possessive 's) in message text.
# Stage 2 drops bot *turns*, but humans still reference bot names (e.g.
# "follow ubottu's message"), so we mask these in stage 4 text as __user__.
_RE_BOT_NAMES = re.compile(
    r"\b(?:" + "|".join(re.escape(b) for b in sorted(_BOTS, key=len, reverse=True)) + r")(?:'s)?\b",
    re.IGNORECASE,
)

# ── Domain-filter patterns (Stage 4.5) ───────────────────────────────────────
#
# IMPORTANT — these patterns operate on Stage 4 TEXT (post _clean_text):
#   • Commands survive cleaning: sudo/apt-get/chmod etc. are pure [a-z0-9-]
#   • __path__ is the correct signal for filesystem paths (/etc/ etc. are gone)
#   • __cmd__ is NOT generated by the pipeline (reserved but unimplemented)
#   • '?' is stripped by _RE_NONALPHA — all ?-based patterns are dead
#   • "can't" → "cannot" via contraction map — only use "cannot"
#
# Strategy A — command-related pairs
_DOMAIN_CMD_RE = re.compile(
    r"\b("
    # Package management
    r"sudo|apt-get|apt|dpkg|dpkg-reconfigure|snap|add-apt-repository|apt-key|"
    # File operations
    r"chmod|chown|ls|ll|mkdir|rm|cp|mv|find|locate|ln|touch|"
    # Text processing
    r"grep|cat|sed|awk|sort|uniq|wc|head|tail|less|more|cut|tee|xargs|"
    # Archives & downloads
    r"tar|gzip|gunzip|unzip|wget|curl|"
    # Editors
    r"nano|vim|vi|emacs|"
    # Process management
    r"kill|ps|top|htop|service|systemctl|pkill|"
    # Network
    r"ping|ifconfig|ip|netstat|ss|ufw|iptables|ssh|nc|dig|nslookup|"
    # Disk / mounting
    r"df|du|mount|umount|fdisk|parted|mkfs|fsck|lsblk|"
    # User / environment
    r"export|echo|source|adduser|useradd|usermod|passwd|su|chroot|"
    # Build / misc
    r"make|gcc|crontab|screen|tmux|lsof|strace|update-grub|pip|"
    # Scripting
    r"bash|sh|chmod\s|\./"
    r")\b",
    re.IGNORECASE,
)

# Strategy B — question-pattern pairs (scan last substantive context turn)
# No ?-based patterns — ? stripped by _RE_NONALPHA in _clean_text.
# No can't — expanded to cannot by contraction map.
# Combined into a single compiled regex for ~15x fewer .search() calls per pair.
_DOMAIN_Q_RE = re.compile(
    r"\bhow (do|can|to|would|should|did) (i|you|we|one)\b"
    r"|\bhow to\b"
    r"|\bwhat (is|are|does|do|was|were|should|the)\b"
    r"|\bwhere (is|are|can|do|should|to find)\b"
    r"|\bwhy (is|does|do|will|would|cannot|wont|didnt|isnt|doesnt)\b"
    r"|\bwhich (command|file|package|version|driver|tool|way|one)\b"
    r"|\bi cannot\b"
    r"|\b(problem|error|fail|failed|broken|issue|not working)\b"
    r"|\b(i need help|help me|need to know)\b"
    r"|\bi (am |)trying to\b"
    r"|\bi (need|want) to\b"
    r"|\b(anyone|anybody) know\b"
    r"|\bshould i\b"
    r"|\bis (there|it) (a |any |)(way|possible|correct|normal)\b"
    r"|^(is|can|do|will|does|has|have|should|would|are)\b"
)

# ── Coherence filter stopwords ────────────────────────────────────────────────
# Used in stage 4 to discard pairs where the last ctx turn and response share
# no content words — a signal of dialogue-window misalignment.
_COHERENCE_STOPWORDS = frozenset({
    "that", "this", "with", "have", "from", "they", "will", "been", "would",
    "could", "should", "there", "their", "what", "when", "which", "your",
    "also", "into", "more", "some", "such", "about", "just", "then", "than",
    "here", "does", "like", "very", "even", "most", "over", "only", "same",
    "each", "make", "much", "come", "want", "need", "know", "think", "look",
    "time", "back", "after", "where", "while", "through", "however", "because",
    "these", "those", "other", "them", "been", "were", "have", "will", "shall",
    "still", "again", "being", "going", "doing", "using", "something", "anything",
})

# ── Contraction map ───────────────────────────────────────────────────────────

_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'll": "he will",
    "he'd": "he would", "she's": "she is", "she'll": "she will",
    "she'd": "she would", "it's": "it is", "it'll": "it will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "we'd": "we would", "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would",
    "that's": "that is", "that'll": "that will",
    "who's": "who is", "who'll": "who will", "who'd": "who would",
    "what's": "what is", "what'll": "what will",
    "where's": "where is", "where'd": "where did",
    "when's": "when is", "when'd": "when did",
    "why's": "why is", "why'd": "why did",
    "how's": "how is", "how'd": "how did",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not",
    "let's": "let us", "there's": "there is", "here's": "here is",
    "ain't": "is not",
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def _elapsed(t0: float) -> str:
    """Return human-readable elapsed time since t0."""
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def _save_json(obj: object, path: Path) -> None:
    """Atomically write obj as JSON (tmp-then-rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)
    print(f"  saved {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def _load_json(path: Path) -> object:
    """Load JSON from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_pickle(obj: object, path: Path) -> None:
    """Atomically write obj as pickle (tmp-then-rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"  saved {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def _load_pickle(path: Path) -> Any:
    """Load pickle from path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _stage_done(stage: int, artifact_dir: Path) -> bool:
    """Return True if all artifacts for stage already exist."""
    return all((artifact_dir / f).exists() for f in STAGE_ARTIFACTS[stage])


def _cleanup_intermediate(artifact_dir: Path, patterns: list) -> None:
    """Delete intermediate files that are no longer needed to free disk space.

    Only deletes files that actually exist; silently skips missing files.
    """
    freed = 0
    for pat in patterns:
        for p in artifact_dir.glob(pat):
            sz = p.stat().st_size
            p.unlink()
            freed += sz
    if freed > 0:
        print(f"  🧹 Cleaned up {freed / (1024**2):.0f} MB of intermediate files")


# ── Text cleaning helpers ─────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Clean a single utterance string.

    Steps: URL/path normalisation, IRC nick stripping, lowercase,
    contraction expansion, non-alphanumeric removal, whitespace collapse.
    """
    if not text:
        return ""
    text = _RE_URL.sub(" __url__ ", text)
    text = _RE_PATH.sub(" __path__ ", text)
    text = _RE_IP.sub(" __ip__ ", text)      # mask IPv4 addresses (e.g. 173.224.120.70)
    text = _RE_IRC_NICK.sub("", text)
    text = text.lower()
    text = _RE_IRC_ADDR.sub("", text)
    words = text.split()
    words = [_CONTRACTIONS.get(w, w) for w in words]
    text = " ".join(words)
    text = _RE_NONALPHA.sub(" ", text)
    text = _RE_DOTS.sub('.', text)
    text = _RE_SENT_END.sub(r'\1 .\2', text)
    text = " ".join(w.lstrip("'") for w in text.split())
    return _RE_MULTI_SP.sub(" ", text).strip()


_PASTE_SPECIAL = set(r'[]{}()=<>|;:@#$%^&*\/')
_PASTE_SPECIAL_TRANS = str.maketrans("", "", ''.join(c for c in map(chr, range(128)) if not c.isalpha()))

def _is_likely_paste(text: str) -> bool:
    """Detect pasted terminal output / log lines / config dumps.

    Heuristics: low alphabetic ratio, high special-character density,
    or many colons in a short line (timestamp / key-value patterns).
    """
    if not text or len(text) < 10:
        return False
    text_len = len(text)
    # Count alpha chars via translate (C-level, ~5x faster than generator)
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / text_len < 0.30:
        return True
    special_count = sum(1 for c in text if c in _PASTE_SPECIAL)
    if special_count / text_len > 0.15:
        return True
    if text.count(":") >= 3 and text_len < 200:
        return True
    return False


def _is_repetitive(text: str) -> bool:
    """Detect repetitive turns like 'help help help help'."""
    tokens = text.split()
    if len(tokens) < 4:
        return False
    most_common_count = Counter(tokens).most_common(1)[0][1]
    return (most_common_count / len(tokens)) > 0.50


_RE_PLACEHOLDERS = re.compile(r"__url__|__path__|__cmd__|__number__")

def _is_english_response(text: str) -> bool:
    """Return True if text is predominantly ASCII (English).

    Non-English IRC users typically have enough non-ASCII characters to
    trip this check. Responses with fewer than 3 alpha chars are passed
    through — other filters handle trivially short responses.
    """
    clean = _RE_PLACEHOLDERS.sub(" ", text)
    alpha_count = 0
    ascii_count = 0
    for c in clean:
        if c.isalpha():
            alpha_count += 1
            if c < '\x80':
                ascii_count += 1
    if alpha_count < 3:
        return True
    return (ascii_count / alpha_count) >= 0.80


def _is_placeholder_only(text: str) -> bool:
    """Return True if response contains only placeholder tokens."""
    return bool(_RE_PLACEHOLDER_ONLY.match(text.strip()))


_RE_ECHO_STRIP = re.compile(r"__eot__|__user__")

def _is_echo_pair(resp_text: str, ctx_text: str) -> bool:
    """Return True if the response appears verbatim inside the context."""
    if len(resp_text) < 6:
        return False
    ctx_norm = _RE_ECHO_STRIP.sub(" ", ctx_text.lower())
    ctx_norm = _RE_MULTI_SP.sub(" ", ctx_norm).strip()
    return resp_text.lower() in ctx_norm


_BOT_RESPONSE_PREFIXES = tuple(_BOT_RESPONSE_BLACKLIST)

def _is_bot_response(text: str) -> bool:
    """Return True if text matches a known bot/boilerplate string."""
    return text.startswith(_BOT_RESPONSE_PREFIXES)


# ── Date parsing ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=500_000)
def _parse_date(date_str: str) -> Optional[datetime]:
    """Try several common date formats; return UTC datetime or None.

    Cached: the corpus has many repeated date strings across turns.
    With 16.6M turns but only ~2-3M unique timestamps, the cache
    eliminates ~80% of strptime calls.
    """
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def _dialogue_first_date(dlg: Dict) -> Optional[datetime]:
    """Return the parsed date of the first turn that has a valid date."""
    for t in dlg["turns"]:
        d = _parse_date(t.get("date", ""))
        if d is not None:
            return d
    return None


# ── Same-speaker turn merging ─────────────────────────────────────────────────

def _merge_same_speaker_turns(turns: List[Dict]) -> List[Dict]:
    """Merge consecutive turns by the same speaker into one turn.

    Utterances are joined with a single space. Duplicate text within the
    same merged turn is silently deduplicated.
    """
    if not turns:
        return []
    merged = [dict(turns[0])]
    for t in turns[1:]:
        if t["from"].lower() == merged[-1]["from"].lower():
            existing = {u.strip() for u in merged[-1]["text"].split(" ")}
            candidate = t["text"].strip()
            if candidate and candidate not in existing:
                merged[-1]["text"] += " " + candidate
        else:
            merged.append(dict(t))
    return merged


# ── Stage 2 worker (top-level for multiprocessing pickling) ──────────────────

def _filter_worker(args: Tuple) -> Tuple[List[Dict], Dict]:
    """Filter and clean one chunk of dialogues.

    Top-level function required for multiprocessing pickling.
    Returns (kept_dialogues, reason_counts).
    """
    chunk, cfg = args
    kept: List[Dict] = []
    counts: Dict[str, int] = defaultdict(int)

    for dlg in chunk:
        result, reason = _filter_dialogue(dlg, cfg)
        counts[reason] += 1
        if result is not None:
            kept.append(result)

    return kept, dict(counts)


def _filter_dialogue(dlg: Dict, cfg: dict) -> Tuple[Optional[Dict], str]:
    """Apply all dialogue-level quality filters and clean turn text.

    Returns (cleaned_dialogue, reason) where reason is 'kept' on success
    or a short label describing why the dialogue was dropped.
    """
    # Sort turns by date; require at least 2 parseable dates
    valid = []
    for t in dlg["turns"]:
        dt = _parse_date(t.get("date", ""))
        if dt is not None:
            valid.append((dt, t))
    if len(valid) < 2:
        return None, "too_few_valid_dates"
    valid.sort(key=lambda x: x[0])
    sorted_turns = [t for _dt, t in valid]

    # Per-turn filtering
    cleaned_turns: List[Dict] = []
    for t in sorted_turns:
        speaker = t["from"].lower().strip()
        if speaker in _BOTS:
            continue
        raw = t["text"].strip()
        if cfg.get("filter_irc_actions", True) and _RE_ACTION.match(raw):
            continue
        if cfg.get("filter_paste", True) and _is_likely_paste(raw):
            continue
        ctext = _clean_text(raw)
        if not ctext:
            continue
        if cfg.get("filter_repetitive", True) and _is_repetitive(ctext):
            continue
        cleaned_turns.append({
            "date": t.get("date", ""),
            "from": speaker,
            "text": ctext,
        })

    if len(cleaned_turns) < cfg.get("min_turns", 2):
        return None, "too_few_turns"

    # Dyadic check
    if cfg.get("dyadic_only", True):
        speakers = {t["from"] for t in cleaned_turns}
        if len(speakers) != 2:
            return None, "not_dyadic"

    # Speaker dominance filter — drop monologue-style threads where one speaker
    # dominates >max_single_speaker_ratio of turns (e.g. help-flooding bots).
    max_sr = cfg.get("max_single_speaker_ratio", 1.0)
    if max_sr < 1.0:
        speaker_counts = Counter(t["from"] for t in cleaned_turns)
        dominant_ratio = max(speaker_counts.values()) / len(cleaned_turns)
        if dominant_ratio > max_sr:
            return None, "speaker_dominance"

    # Alternation ratio filter — drop non-alternating floods like A,A,A,B,B,B.
    # Requires at least min_alternation_ratio fraction of consecutive-turn speaker changes.
    min_alt = cfg.get("min_alternation_ratio", 0.0)
    if min_alt > 0.0 and len(cleaned_turns) >= 2:
        alt_count = sum(
            1 for i in range(1, len(cleaned_turns))
            if cleaned_turns[i]["from"] != cleaned_turns[i - 1]["from"]
        )
        alt_ratio = alt_count / (len(cleaned_turns) - 1)
        if alt_ratio < min_alt:
            return None, "low_alternation"

    # Temporal coherence: hard ceiling + large-gap ratio in a single pass
    if cfg.get("filter_temporal", True):
        hard_ceiling = cfg.get("max_turn_gap_seconds", 0)
        gap_threshold = cfg.get("large_gap_threshold", 600)
        max_gap_ratio = cfg.get("max_large_gap_ratio", 1.0)

        if hard_ceiling > 0 or max_gap_ratio < 1.0:
            times = [_parse_date(t["date"]) for t in cleaned_turns]
            times = [x for x in times if x is not None]
            if len(times) >= 2:
                n_large = 0
                for i in range(1, len(times)):
                    gap = abs((times[i] - times[i - 1]).total_seconds())
                    if hard_ceiling > 0 and gap > hard_ceiling:
                        return None, "temporal_hard_ceiling"
                    if gap > gap_threshold:
                        n_large += 1
                if max_gap_ratio < 1.0 and n_large / (len(times) - 1) > max_gap_ratio:
                    return None, "temporal_gap_ratio"

    return {"id": dlg["id"], "turns": cleaned_turns}, "kept"


# ── Stage 0: Kaggle download ──────────────────────────────────────────────────

# Each entry: (kaggle_slug, local_subdir_name)
KAGGLE_DATASETS = [
    ("rtatman/ubuntu-dialogue-corpus",                                "Ubuntu-dialogue-corpus"),
    ("saurabhshahane/wikiqa-corpus",                                  "wikiqa-corpus"),
    ("rtatman/questionanswer-dataset",                                "question-answer-dataset"),
    ("jeromeblanchet/conversational-question-answering-dataset-coqa", "coqa"),
    ("shadimsadiq/english-movie-subtitle-dataset",                    "english-movie-subtitles"),
    ("Cornell-University/movie-dialog-corpus",                        "movie-dialog-corpus"),
]


def _kaggle_download(slug: str, dest_dir: Path) -> None:
    """Download a single Kaggle dataset to dest_dir (API or CLI)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"  Downloading {slug} → {dest_dir} …")
        api.dataset_download_files(slug, path=str(dest_dir), unzip=True)
    except ImportError:
        print("  kaggle package not installed — trying kaggle CLI …")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug,
                 "-p", str(dest_dir), "--unzip"],
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Could not download corpus: install the kaggle package "
                "(pip install kaggle) or ensure the kaggle CLI is on PATH."
            )


def stage0_download_corpus(cfg: dict) -> None:
    """Download all training corpora from Kaggle if not already present.

    Downloads:
      - Ubuntu Dialogue Corpus (primary — used by Stage 1+)
      - WikiQA Corpus (Microsoft Research)
      - Question-Answer Dataset (factoid Q&A pairs)
      - CoQA (Conversational Question Answering)
      - English Movie Subtitle Dataset
      - Movie Dialog Corpus (Cornell)

    Requires either:
      - The ``kaggle`` Python package (``pip install kaggle``), or
      - The ``kaggle`` CLI on PATH.

    Authentication: place kaggle.json in ~/.kaggle/ (Linux/Mac) or
    C:\\Users\\<user>\\.kaggle\\ (Windows), or set KAGGLE_USERNAME +
    KAGGLE_KEY environment variables.
    """
    data_dir = Path(cfg["corpus_dir"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 0 — Download corpora from Kaggle")
    print("=" * 60)
    t0 = time.time()

    downloaded = 0
    for slug, subdir in KAGGLE_DATASETS:
        dest = data_dir / subdir
        if dest.exists() and any(dest.iterdir()):
            print(f"  ✓ {subdir}/ already present — skipping")
            continue

        dest.mkdir(parents=True, exist_ok=True)
        _kaggle_download(slug, dest)

        # Some datasets extract flat into dest, others create a subfolder.
        # If dest is empty but a nested folder appeared, it's fine.
        if not any(dest.iterdir()):
            print(f"  ⚠ Warning: {dest} is empty after download — check manually")
        else:
            n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            print(f"  ✓ {subdir}/ — {n_files} files ({total_size / 1e6:.0f} MB)")
            downloaded += 1

    # Verify primary corpus (Ubuntu) has CSVs — needed for Stage 1
    corpus_dir = Path(cfg["corpus_dir"])
    if not corpus_dir.exists():
        # CSVs may have landed directly in data_dir
        flat_csvs = list(data_dir.glob("dialogueText*.csv"))
        if flat_csvs:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            for f in flat_csvs:
                shutil.move(str(f), str(corpus_dir / f.name))
            print(f"  Moved {len(flat_csvs)} CSV files → {corpus_dir}")

    if not corpus_dir.exists() or not any(corpus_dir.glob("*.csv")):
        raise FileNotFoundError(
            f"Ubuntu corpus CSV files not found in {corpus_dir}. "
            f"Check the extracted contents in {data_dir}."
        )

    # Keep only dialogueText_301.csv (superset) — remove smaller variants to save disk
    kept = corpus_dir / "dialogueText_301.csv"
    if kept.exists():
        for f in list(corpus_dir.glob("dialogueText*.csv")):
            if f.name != "dialogueText_301.csv":
                f.unlink()
                print(f"  Removed {f.name} (subset of 301)")

    print(f"\n  Stage 0 done — {downloaded} new dataset(s) downloaded "
          f"({time.time() - t0:.0f}s)\n")


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1_load_corpus(cfg: dict) -> List[Dict]:
    """Load raw Ubuntu Dialogue Corpus CSV files → list of dialogue dicts.

    Uses DuckDB for fast CSV parsing, filtering, and sorting (~10-15x faster
    than csv.DictReader). Falls back to the original Python implementation
    if DuckDB is not available.

    Each dialogue is structured as:
        {"id": str, "turns": [{"date": str, "from": str, "text": str}]}
    Turns within each dialogue are sorted by date before return.
    The unique dialogue key is 'folder/dialogueID' to avoid merging
    unrelated IRC sessions from different folders.
    """
    source_dir = Path(cfg["corpus_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"corpus_dir not found: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    # Prefer the largest file (dialogueText_301.csv) — it's a superset of the others.
    largest = max(csv_files, key=lambda p: p.stat().st_size)
    print(f"  Using {largest.name} ({largest.stat().st_size / 1e9:.1f} GB)")

    try:
        return _stage1_duckdb(largest)
    except (ImportError, Exception) as e:
        if "OutOfMemory" in type(e).__name__ or "OutOfMemory" in str(e):
            print(f"  DuckDB out of memory — falling back to csv.DictReader")
        elif isinstance(e, ImportError):
            print("  DuckDB not available — falling back to csv.DictReader")
        else:
            print(f"  DuckDB failed ({e}) — falling back to csv.DictReader")
        return _stage1_csv(cfg, csv_files)


def _stage1_duckdb(csv_path: Path) -> List[Dict]:
    """Fast Stage 1 using DuckDB: CSV parse + filter + sort in C++."""
    import duckdb

    print("  Loading with DuckDB …")
    t0 = time.time()

    # DuckDB reads, filters, concatenates folder/dialogueID, and sorts in one query.
    # CAST(date AS VARCHAR) keeps the raw date string for downstream _parse_date().
    # DuckDB groups and aggregates in C++ — avoids 16.6M-row Python loop.
    grouped = duckdb.sql(f"""
        WITH raw AS (
            SELECT
                CASE WHEN CAST(folder AS VARCHAR) IS NOT NULL AND CAST(folder AS VARCHAR) != ''
                     THEN CAST(folder AS VARCHAR) || '/' || CAST(dialogueID AS VARCHAR)
                     ELSE CAST(dialogueID AS VARCHAR)
                END AS id,
                CAST(date AS VARCHAR) AS date,
                CAST("from" AS VARCHAR) AS speaker,
                CAST(text AS VARCHAR) AS text
            FROM read_csv_auto('{csv_path}', all_varchar=true, ignore_errors=true)
            WHERE "from" IS NOT NULL AND TRIM("from") != ''
              AND text IS NOT NULL AND TRIM(text) != ''
              AND dialogueID IS NOT NULL AND TRIM(dialogueID) != ''
        )
        SELECT id,
               LIST(struct_pack(date := date, speaker := speaker, text := text)
                    ORDER BY date) AS turns
        FROM raw
        GROUP BY id
    """).fetchall()

    total_turns_est = sum(len(row[1]) for row in grouped)
    print(f"  DuckDB read ~{total_turns_est:,} turns in {time.time() - t0:.1f}s")

    # Convert DuckDB structs to Python dicts
    t1 = time.time()
    result: List[Dict] = []
    for row in grouped:
        dlg_id = row[0]
        turns = []
        for t in row[1]:
            date_str = t["date"] if t["date"] and str(t["date"]) != "None" else ""
            turns.append({
                "date": date_str,
                "from": t["speaker"],
                "text": t["text"],
            })
        result.append({"id": dlg_id, "turns": turns})
    print(f"  Total turns: {total_turns_est:,}   Dialogues: {len(result):,}")
    return result


def _stage1_csv(cfg: dict, csv_files: List[Path]) -> List[Dict]:
    """Original Stage 1 fallback using csv.DictReader."""
    csv.field_size_limit(2 ** 24)
    print(f"  Found {len(csv_files)} CSV file(s)")

    dialogues: Dict[str, List[Dict]] = defaultdict(list)
    total_turns = 0

    for csv_path in csv_files:
        print(f"  Reading {csv_path.name} …")
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                did = row.get("dialogueID", "").strip()
                fld = row.get("folder", "").strip()
                if not did:
                    continue
                unique_id = f"{fld}/{did}" if fld else did
                date_val = row.get("date", "").strip()
                from_val = row.get("from", "").strip()
                text_val = row.get("text", "").strip()
                if not from_val or not text_val:
                    continue
                dialogues[unique_id].append({
                    "date": date_val,
                    "from": from_val,
                    "text": text_val,
                })
                total_turns += 1
                if total_turns % 5_000_000 == 0:
                    print(f"    … {total_turns:,} turns, {len(dialogues):,} dialogues")

    # Sort turns by date within each dialogue; skip undated turns
    result: List[Dict] = []
    for uid, turns in dialogues.items():
        dated = []
        undated = []
        for t in turns:
            dt = _parse_date(t["date"])
            if dt is not None:
                dated.append((dt, t))
            else:
                undated.append(t)
        dated.sort(key=lambda x: x[0])
        sorted_turns = [t for _dt, t in dated] + undated
        if sorted_turns:
            result.append({"id": uid, "turns": sorted_turns})

    print(f"  Total turns: {total_turns:,}   Dialogues: {len(result):,}")
    return result


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def stage2_clean_and_filter(dialogues: List[Dict], cfg: dict) -> Tuple[List[Dict], Dict]:
    """Apply all quality filters and text cleaning in parallel chunks.

    Returns (clean_dialogues, stats) where stats includes per-filter
    discard counts. Uses multiprocessing.Pool with fallback to sequential
    processing if the pool fails.
    """
    total_in = len(dialogues)
    chunk_size = cfg.get("chunk_size", 50_000)
    # OpenShift: cpu_count() returns host CPUs (254), not pod limit (4).
    num_workers = cfg.get("num_workers", 4)

    chunks = [dialogues[i: i + chunk_size] for i in range(0, total_in, chunk_size)]
    print(f"  {total_in:,} dialogues → {len(chunks)} chunks (size {chunk_size:,}), {num_workers} workers")

    kept_all: List[Dict] = []
    reason_totals: Dict[str, int] = defaultdict(int)

    worker_args = [(chunk, cfg) for chunk in chunks]

    try:
        # Use "fork" on Linux (OpenShift AI / Colab) — spawn can't pickle notebook
        # functions. Use "spawn" on Windows/macOS to avoid fork-safety issues.
        _mp_method = "fork" if sys.platform.startswith("linux") else "spawn"
        _mp_ctx = mp.get_context(_mp_method)
        with _mp_ctx.Pool(num_workers) as pool:
            for ci, (kept_chunk, reason_counts) in enumerate(
                pool.imap_unordered(_filter_worker, worker_args), 1
            ):
                kept_all.extend(kept_chunk)
                for r, n in reason_counts.items():
                    reason_totals[r] += n
                if ci % max(1, len(chunks) // 10) == 0 or ci == len(chunks):
                    print(f"    chunk {ci}/{len(chunks)}  kept={len(kept_all):,}")
    except Exception as exc:
        print(f"  Pool failed ({exc}); falling back to sequential processing …")
        kept_all = []
        reason_totals = defaultdict(int)
        for ci, (chunk, _cfg) in enumerate(worker_args, 1):
            kept_chunk, reason_counts = _filter_worker((chunk, _cfg))
            kept_all.extend(kept_chunk)
            for r, n in reason_counts.items():
                reason_totals[r] += n
            if ci % max(1, len(chunks) // 10) == 0 or ci == len(chunks):
                print(f"    chunk {ci}/{len(chunks)}  kept={len(kept_all):,}")

    total_out = len(kept_all)
    total_disc = total_in - total_out

    print(f"\n  Filter breakdown (input: {total_in:,}):")
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        tag = "kept" if reason == "kept" else "disc"
        print(f"    {tag}  {reason:<28}  {count:>8,}  ({count/total_in*100:.1f}%)")
    print(f"  TOTAL KEPT: {total_out:,} ({total_out/total_in*100:.1f}%)")

    stats = {
        "stage": 2,
        "n_input": total_in,
        "n_output": total_out,
        "n_discarded": total_disc,
        "filter_breakdown": dict(reason_totals),
    }
    return kept_all, stats


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def stage3_temporal_split(dialogues: List[Dict], cfg: dict) -> Tuple[List, List, List, Dict]:
    """Split dialogues by THREAD first-turn date → (train, val, test, stats).

    Thread boundary safety (G6): each dialogue is assigned to exactly one
    split based on its FIRST turn's date. This prevents a single IRC thread
    from appearing in both train and test (data leakage).

    Dialogues with no parseable date are assigned to train.
    """
    train_cutoff = datetime.strptime(
        cfg["train_cutoff_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)
    val_cutoff = datetime.strptime(
        cfg["val_cutoff_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []
    no_date = 0

    for dlg in dialogues:
        first_date = _dialogue_first_date(dlg)
        if first_date is None:
            no_date += 1
            train.append(dlg)
            continue
        if first_date < train_cutoff:
            train.append(dlg)
        elif first_date < val_cutoff:
            val.append(dlg)
        else:
            test.append(dlg)

    total = len(train) + len(val) + len(test)

    # Verify thread boundary integrity
    train_ids = {d["id"] for d in train}
    val_ids   = {d["id"] for d in val}
    test_ids  = {d["id"] for d in test}
    tv_overlap = len(train_ids & val_ids)
    tt_overlap = len(train_ids & test_ids)
    vt_overlap = len(val_ids & test_ids)
    assert tv_overlap == 0, f"Thread boundary violation: {tv_overlap} dialogues in train ∩ val"
    assert tt_overlap == 0, f"Thread boundary violation: {tt_overlap} dialogues in train ∩ test"
    assert vt_overlap == 0, f"Thread boundary violation: {vt_overlap} dialogues in val ∩ test"

    stats = {
        "stage": 3,
        "n_train_dialogues": len(train),
        "n_val_dialogues":   len(val),
        "n_test_dialogues":  len(test),
        "n_no_date":         no_date,
        "train_pct":         round(len(train) / total * 100, 1) if total else 0,
        "val_pct":           round(len(val) / total * 100, 1) if total else 0,
        "test_pct":          round(len(test) / total * 100, 1) if total else 0,
        "overlap_train_val":  tv_overlap,
        "overlap_train_test": tt_overlap,
        "overlap_val_test":   vt_overlap,
        "zero_overlap_confirmed": True,
    }
    print(f"  Train: {len(train):,} ({stats['train_pct']}%)  "
          f"Val: {len(val):,} ({stats['val_pct']}%)  "
          f"Test: {len(test):,} ({stats['test_pct']}%)  "
          f"No-date: {no_date:,}")
    print(f"  Overlaps — T∩V: {tv_overlap}, T∩Te: {tt_overlap}, V∩Te: {vt_overlap}  ✓ zero overlap confirmed")

    return train, val, test, stats


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def _generate_pairs_for_split(
    dialogues: List[Dict],
    cfg: dict,
    apply_diversity_filter: bool,
) -> Tuple[List[Dict], Dict]:
    """Generate (ctx, resp) string pairs from a list of dialogues.

    For each dialogue, same-speaker turns are merged, then for each
    response position i the context is the max_ctx_turns most-recent
    preceding turns joined by a space.

    The response diversity filter (apply_diversity_filter=True) limits
    how many times any identical response string can appear.

    Returns (pairs, discard_counts).
    """
    max_ctx_turns  = cfg["max_ctx_turns"]
    max_resp_tokens = cfg["max_resp_tokens"]
    min_resp_tokens = cfg["min_resp_tokens"]
    min_ctx_tokens  = cfg.get("min_ctx_tokens", 3)

    resp_counter: Counter = Counter()
    max_occ = cfg.get("max_response_occurrences", 500)
    do_diversity = apply_diversity_filter and cfg.get("filter_response_diversity", True)

    disc = defaultdict(int)
    pairs: List[Dict] = []

    for dlg in dialogues:
        merged = _merge_same_speaker_turns(dlg["turns"])
        if len(merged) < 2:
            continue

        # Precompile speaker-name patterns once per dialogue (not per pair)
        speaker_patterns = []
        for sp in {t["from"].lower() for t in merged}:
            if _RE_SPEAKER_SPECIAL.search(sp) or len(sp) > 9:
                speaker_patterns.append(re.compile(r"\b" + re.escape(sp) + r"\b"))

        for i in range(1, len(merged)):
            start = max(0, i - max_ctx_turns)
            ctx_text  = " __eot__ ".join(merged[j]["text"] for j in range(start, i))
            resp_text = merged[i]["text"]

            # Mask speaker names with precompiled patterns
            for pat in speaker_patterns:
                ctx_text  = pat.sub("__user__", ctx_text)
                resp_text = pat.sub("__user__", resp_text)

            ctx_text  = _RE_BOT_NAMES.sub("__user__", ctx_text)
            resp_text = _RE_BOT_NAMES.sub("__user__", resp_text)

            # Rough token estimate (word count) for filtering
            ctx_words  = ctx_text.split()
            resp_words = resp_text.split()
            if len(ctx_words) < min_ctx_tokens:
                disc["ctx_too_short"] += 1
                continue
            if len(resp_words) < min_resp_tokens:
                disc["resp_too_short"] += 1
                continue
            if len(resp_words) > max_resp_tokens:
                disc["resp_too_long"] += 1
                continue

            # Pair-level quality filters
            if cfg.get("filter_non_english", True) and not _is_english_response(resp_text):
                disc["non_english"] += 1
                continue
            if cfg.get("filter_bot_responses", True) and _is_bot_response(resp_text):
                disc["bot_response"] += 1
                continue
            if cfg.get("filter_echo_pairs", True) and _is_echo_pair(resp_text, ctx_text):
                disc["echo_pair"] += 1
                continue
            if cfg.get("filter_placeholder", True) and _is_placeholder_only(resp_text):
                disc["placeholder_only"] += 1
                continue

            # Turn-coherence filter: discard pairs where no recent substantive
            # ctx turn shares content words with the response — catches
            # dialogue-window misalignments.
            #
            # Blind-spot fix: the last turn is often a short ack ("yes", "ok",
            # "sure") which has < 5 content words.  We now walk BACKWARD through
            # ctx turns to find the most recent one that is substantive (≥ 5
            # content words) before running the overlap check.  This catches the
            # pattern: [long question] __eot__ [cingular] __eot__ [yes] / [XP resp]
            if cfg.get("filter_incoherent_pairs", True):
                ctx_turns = ctx_text.split(" __eot__ ")
                ctx_content: set = set()
                for turn in reversed(ctx_turns):
                    words = {
                        w for w in turn.split()
                        if w.isalpha() and len(w) >= 4 and w not in _COHERENCE_STOPWORDS
                    }
                    if len(words) >= 5:
                        ctx_content = words
                        break
                resp_content = {
                    w for w in resp_text.split()
                    if w.isalpha() and len(w) >= 4 and w not in _COHERENCE_STOPWORDS
                }
                if len(ctx_content) >= 5 and len(resp_content) >= 5:
                    if not (ctx_content & resp_content):
                        disc["incoherent_pair"] += 1
                        continue

            # Response diversity cap (train only)
            if do_diversity:
                if resp_counter[resp_text] >= max_occ:
                    disc["diversity_cap"] += 1
                    continue
                resp_counter[resp_text] += 1

            pairs.append({"ctx": ctx_text, "resp": resp_text})

    return pairs, dict(disc)


def _generate_pairs_worker(args: Tuple) -> Tuple[List[Dict], Dict]:
    """Worker for parallel pair generation (top-level for pickling)."""
    chunk, cfg, apply_diversity = args
    return _generate_pairs_for_split(chunk, cfg, apply_diversity)


def _write_stage4_samples(
    splits: Dict[str, List[Dict]],
    artifact_dir: Path,
    n: int = 200,
    seed: int = 42,
) -> None:
    """Write n randomly-sampled pairs from each split as a readable text file.

    Files are saved alongside the main stage 4 JSONs as:
        stage4_train_samples.txt
        stage4_val_samples.txt
        stage4_test_samples.txt

    Each pair is formatted as:
        ── Pair 001 ──────────────────────────────────────
        CTX : <context with __eot__ separators>
        RESP: <response>

    Re-running always overwrites so the samples stay fresh.
    """
    rng = random.Random(seed)
    for split_name, pairs in splits.items():
        sample = rng.sample(pairs, min(n, len(pairs)))
        out_path = artifact_dir / f"stage4_{split_name}_samples.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Stage 4 — {split_name.upper()} samples  "
                    f"({len(sample)} of {len(pairs):,} pairs, seed={seed})\n")
            f.write("=" * 70 + "\n\n")
            for idx, pair in enumerate(sample, 1):
                f.write(f"── Pair {idx:03d} {'─' * 50}\n")
                f.write(f"CTX : {pair['ctx']}\n")
                f.write(f"RESP: {pair['resp']}\n\n")
        print(f"  Saved {len(sample)} {split_name} samples → {out_path.name}")


def stage4_generate_pairs(
    train_dialogues: List[Dict],
    val_dialogues: List[Dict],
    test_dialogues: List[Dict],
    cfg: dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Extract (context, response) string pairs from all three splits.

    Applies the response diversity filter on train only.
    Caps train to cfg['max_train_pairs'] if > 0.
    Returns (train_pairs, val_pairs, test_pairs, stats).
    """
    print("  Generating train pairs …")
    n_workers = cfg.get("num_workers", 4)
    if n_workers > 1 and len(train_dialogues) > 10_000:
        # Parallel: split dialogues across workers, merge results
        chunk_size = (len(train_dialogues) + n_workers - 1) // n_workers
        chunks = [train_dialogues[i:i + chunk_size]
                  for i in range(0, len(train_dialogues), chunk_size)]
        # diversity_filter=False per chunk — we apply global cap after merge
        work = [(c, cfg, False) for c in chunks]
        try:
            with mp.Pool(n_workers) as pool:
                results = pool.map(_generate_pairs_worker, work)
            train_pairs = []
            train_disc: Dict[str, int] = defaultdict(int)
            for pairs_chunk, disc_chunk in results:
                train_pairs.extend(pairs_chunk)
                for k, v in disc_chunk.items():
                    train_disc[k] += v
            train_disc = dict(train_disc)
            # Apply global diversity cap after merging
            if cfg.get("filter_response_diversity", True):
                max_occ = cfg.get("max_response_occurrences", 500)
                resp_counter: Counter = Counter()
                filtered = []
                n_capped = 0
                for p in train_pairs:
                    if resp_counter[p["resp"]] >= max_occ:
                        n_capped += 1
                        continue
                    resp_counter[p["resp"]] += 1
                    filtered.append(p)
                if n_capped:
                    train_disc["diversity_cap"] = n_capped
                train_pairs = filtered
        except Exception as e:
            print(f"    Pool failed ({e}); falling back to sequential")
            train_pairs, train_disc = _generate_pairs_for_split(
                train_dialogues, cfg, apply_diversity_filter=True)
    else:
        train_pairs, train_disc = _generate_pairs_for_split(
            train_dialogues, cfg, apply_diversity_filter=True)
    print(f"    train raw: {len(train_pairs):,}  discards: {train_disc}")

    max_pairs = cfg.get("max_train_pairs", 0)
    if max_pairs > 0 and len(train_pairs) > max_pairs:
        # Random shuffle before cap to avoid temporal/alphabetical bias.
        # Without this, we'd only see dialogues from the earliest part of
        # the corpus — identical to the old phase1.py approach (ported back).
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:max_pairs]
        print(f"    train capped to {max_pairs:,} (randomly sampled)")

    if len(train_pairs) == 0:
        raise ValueError("Stage 4: 0 train pairs produced — check filters and corpus path.")

    print("  Generating val pairs …")
    val_pairs, val_disc = _generate_pairs_for_split(val_dialogues, cfg, apply_diversity_filter=False)
    print(f"    val: {len(val_pairs):,}  discards: {val_disc}")

    print("  Generating test pairs …")
    test_pairs, test_disc = _generate_pairs_for_split(test_dialogues, cfg, apply_diversity_filter=False)
    print(f"    test: {len(test_pairs):,}  discards: {test_disc}")

    stats = {
        "stage": 4,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs":   len(val_pairs),
        "n_test_pairs":  len(test_pairs),
        "train_discards": train_disc,
        "val_discards":   val_disc,
        "test_discards":  test_disc,
    }
    return train_pairs, val_pairs, test_pairs, stats


# ── Stage 4.5 — Domain-focused filtering ─────────────────────────────────────

def _is_command_related(text: str) -> bool:
    """Return True if text contains a Linux command or a __path__ token."""
    return bool(_DOMAIN_CMD_RE.search(text)) or "__path__" in text


def _last_substantive_turn(ctx: str, max_lookback: int = 3) -> str:
    """Return the last turn in ctx with ≥ 4 words (walk back up to max_lookback).

    IRC contexts frequently end with short acknowledgment turns ("ok", "right",
    "yes") that are not the actual question.  Walking back finds the most recent
    substantive turn so question patterns have meaningful text to match against.
    """
    turns = ctx.split(" __eot__ ")
    for turn in reversed(turns[-max_lookback:]):
        if len(turn.split()) >= 4:
            return turn
    return turns[-1]


def _is_question_pair(pair: dict) -> bool:
    """Return True if the last substantive context turn matches a question pattern.

    Patterns are designed for post-_clean_text text:
      • No ?-based patterns  — ? stripped by _RE_NONALPHA
      • Uses 'i cannot' not "can't" — more specific than bare 'cannot'
      • Scans last 1-3 substantive turns (max_lookback=3 catches 3 consecutive
        short acks before the actual question)
    """
    turn = _last_substantive_turn(pair["ctx"])
    return bool(_DOMAIN_Q_RE.search(turn))


def stage4_5_domain_filter(
    train_pairs: List[Dict],
    val_pairs:   List[Dict],
    test_pairs:  List[Dict],
    cfg: dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Filter (ctx, resp) pairs to retain domain-relevant content.

    Applies Strategy A (command/path signals) and/or Strategy B (question
    patterns on last substantive context turn) according to cfg['domain_filter_strategy'].

    Applied AFTER Stage 4 generates pairs and BEFORE Stage 5 trains SPM, so the
    BPE vocabulary and FastText embeddings reflect the filtered domain.

    The filter is applied to all three splits to maintain distribution consistency.
    Val and test are filtered identically to train (no diversity cap differences).

    Args:
        train_pairs / val_pairs / test_pairs: lists of {"ctx": str, "resp": str}
        cfg: pipeline config dict

    Returns:
        filtered_train, filtered_val, filtered_test, stats_dict
    """
    strategy = cfg.get("domain_filter_strategy", "union").lower()
    if strategy not in ("command", "question", "union", "intersection"):
        raise ValueError(
            f"domain_filter_strategy must be one of command/question/union/intersection, "
            f"got {strategy!r}"
        )

    def _keep(pair: dict) -> bool:
        cmd = _is_command_related(pair["ctx"]) or _is_command_related(pair["resp"])
        q   = _is_question_pair(pair)
        if strategy == "command":
            return cmd
        if strategy == "question":
            return q
        if strategy == "union":
            return cmd or q
        # intersection
        return cmd and q

    def _filter_split(pairs: List[Dict], split_name: str) -> Tuple[List[Dict], Dict]:
        kept = []
        total = len(pairs)
        n_cmd = n_q = n_both = 0
        for p in pairs:
            cmd = _is_command_related(p["ctx"]) or _is_command_related(p["resp"])
            q   = _is_question_pair(p)
            if cmd:
                n_cmd += 1
            if q:
                n_q += 1
            if cmd and q:
                n_both += 1
            # Apply strategy filter
            if strategy == "command" and cmd:
                kept.append(p)
            elif strategy == "question" and q:
                kept.append(p)
            elif strategy == "union" and (cmd or q):
                kept.append(p)
            elif strategy == "intersection" and (cmd and q):
                kept.append(p)
        pct = 100 * len(kept) / total if total else 0
        print(
            f"  [{split_name}] {total:,} → {len(kept):,} pairs kept ({pct:.1f}%)  "
            f"cmd={n_cmd:,}  q={n_q:,}  both={n_both:,}"
        )
        return kept, {
            "total":  total,
            "kept":   len(kept),
            "pct":    round(pct, 2),
            "n_cmd":  n_cmd,
            "n_question": n_q,
            "n_both": n_both,
        }

    print(f"  Strategy: {strategy!r}")
    f_train, s_train = _filter_split(train_pairs, "train")
    f_val,   s_val   = _filter_split(val_pairs,   "val")
    f_test,  s_test  = _filter_split(test_pairs,  "test")

    if len(f_train) == 0:
        raise ValueError(
            "stage4_5_domain_filter: 0 training pairs survived. "
            "Check strategy and corpus content."
        )

    stats = {
        "stage": "4.5",
        "strategy": strategy,
        "train": s_train,
        "val":   s_val,
        "test":  s_test,
    }
    return f_train, f_val, f_test, stats


# ── Stage 5 ───────────────────────────────────────────────────────────────────

def stage5_train_spm(train_pairs: List[Dict], cfg: dict) -> str:
    """Train BPE tokenizer on training context + response text.

    Prefers HuggingFace Tokenizers (Rust, ~10x faster) with SentencePiece
    fallback. Both produce the same token ID contract: pad=0, unk=1, sos=2, eos=3.

    Returns the path to the saved tokenizer file (.json or .model).
    """
    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(artifact_dir / "stage5_spm")
    target_vocab = cfg["spm_vocab_size"]

    # Build sentence iterator in-memory
    def _sentence_iter():
        for pair in train_pairs:
            ctx = pair["ctx"].strip()
            if ctx:
                yield ctx
            resp = pair["resp"].strip()
            if resp:
                yield resp

    user_symbols = [
        "__url__", "__path__", "__ip__", "__cmd__", "__number__",
        "__eot__", "__user__",
    ]

    # Try HuggingFace Tokenizers first (Rust, multithreaded, ~10x faster)
    try:
        from tokenizer_utils import train_bpe_tokenizer
        print(f"  Training BPE (HuggingFace Tokenizers)  vocab_size={target_vocab}  ({len(train_pairs):,} pairs) …")
        t_hf = time.time()
        model_path = train_bpe_tokenizer(
            sentence_iterator=_sentence_iter(),
            model_prefix=model_prefix,
            vocab_size=target_vocab,
            user_defined_symbols=user_symbols,
        )
        print(f"  ✓ Token ID contract verified: pad=0, unk=1, sos=2, eos=3")
        print(f"  HF Tokenizer saved → {model_path}  ({time.time() - t_hf:.1f}s)")
        return model_path
    except ImportError:
        print("  HuggingFace tokenizers not available — falling back to SentencePiece")

    # Fallback: SentencePiece (C++, single-threaded BPE)
    import sentencepiece as spm

    print(f"  Training SentencePiece BPE  vocab_size={target_vocab}  ({len(train_pairs):,} pairs) …")

    # SPM logs progress to stderr in C++.  We capture it in a background
    # thread and display a compact progress bar instead of thousands of lines.
    import io, os, threading

    _spm_re = re.compile(r"size=(\d+)")
    _bar_lock = threading.Lock()
    _last_size = [0]

    def _print_spm_bar(current, total):
        pct = current / total
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {current:,}/{total:,} pieces ({pct:.0%})", end="", flush=True)

    # Redirect stderr through a pipe so we can parse SPM's C++ output
    old_stderr_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def _reader():
        with os.fdopen(r_fd, "r", errors="replace") as f:
            for line in f:
                m = _spm_re.search(line)
                if m:
                    sz = int(m.group(1))
                    with _bar_lock:
                        if sz > _last_size[0]:
                            _last_size[0] = sz
                            _print_spm_bar(sz, target_vocab)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        spm.SentencePieceTrainer.train(
            sentence_iterator=_sentence_iter(),
            model_prefix=model_prefix,
            vocab_size=target_vocab,
            model_type=cfg.get("spm_model_type", "bpe"),
            character_coverage=cfg.get("spm_character_coverage", 0.9999),
            input_sentence_size=cfg.get("spm_input_sentence_size", 2_000_000),
            shuffle_input_sentence=True,
            pad_id=0,    pad_piece="<pad>",
            unk_id=1,    unk_piece="<unk>",
            bos_id=2,    bos_piece="<sos>",
            eos_id=3,    eos_piece="<eos>",
            user_defined_symbols=user_symbols,
        )
    finally:
        # Restore stderr so normal output works again
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        reader_thread.join(timeout=2)
        print()  # newline after progress bar

    model_path = model_prefix + ".model"

    # Verify token ID contract
    sp = spm.SentencePieceProcessor(model_file=model_path)
    assert sp.piece_to_id("<pad>") == 0, f"<pad> ID mismatch: got {sp.piece_to_id('<pad>')}"
    assert sp.piece_to_id("<unk>") == 1, f"<unk> ID mismatch: got {sp.piece_to_id('<unk>')}"
    assert sp.piece_to_id("<sos>") == 2, f"<sos> ID mismatch: got {sp.piece_to_id('<sos>')}"
    assert sp.piece_to_id("<eos>") == 3, f"<eos> ID mismatch: got {sp.piece_to_id('<eos>')}"
    print(f"  ✓ Token ID contract verified: pad=0, unk=1, sos=2, eos=3")

    print(f"  SPM model saved → {model_path}")
    return model_path


# ── Stage 6 ───────────────────────────────────────────────────────────────────

# Use orjson for ~5x faster JSONL serialisation if available; fallback to json.
try:
    import orjson as _json_fast  # type: ignore[import-untyped]
    def _dumps_line(obj: dict) -> str:
        return _json_fast.dumps(obj).decode("utf-8") + "\n"
except ImportError:
    _json_fast = None  # noqa: F841
    def _dumps_line(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":")) + "\n"


def _encode_split(
    pairs: List[Dict],
    sp,
    out_path: Path,
    max_ctx_tokens: int,
    max_resp_tokens: int,
    sos_id: int,
    eos_id: int,
) -> int:
    """Encode one split to JSONL and return number of lines written.

    Uses batch encoding (sp.encode on list of strings) which is ~3-4x
    faster than encoding one string at a time.
    """
    tmp = out_path.with_suffix(".tmp")
    batch_size = 50_000

    written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            ctx_texts  = [p["ctx"] for p in batch]
            resp_texts = [p["resp"] for p in batch]

            # Batch encode: returns list of list of ints
            ctx_encoded  = sp.encode(ctx_texts, out_type=int)
            resp_encoded = sp.encode(resp_texts, out_type=int)

            # Buffer lines and write in one chunk (fewer syscalls)
            lines = []
            for ctx_ids_full, resp_ids_raw in zip(ctx_encoded, resp_encoded):
                ctx_ids = ctx_ids_full[-max_ctx_tokens:]
                resp_ids = [sos_id] + resp_ids_raw[:max_resp_tokens] + [eos_id]
                lines.append(_dumps_line({"ctx": ctx_ids, "resp": resp_ids}))
            f.write("".join(lines))
            written += len(lines)
    os.replace(tmp, out_path)
    return written


def stage6_encode_pairs(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict],
    spm_model_path: str,
    cfg: dict,
) -> Tuple[str, str, str, Dict, Dict]:
    """Encode all pairs to BPE token ID sequences and write JSONL files.

    Context encoding (M3 — no <sos> on encoder):
        ctx_ids = sp.encode(ctx_text)[-max_ctx_tokens:]   # keep LAST N tokens (most recent context)

    Response encoding:
        resp_ids = [sos_id] + sp.encode(resp_text)[:max_resp_tokens] + [eos_id]

    Returns (train_path, val_path, test_path, vocab_dict, stats).
    """
    from tokenizer_utils import load_tokenizer

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sp = load_tokenizer(artifact_dir)

    sos_id = sp.piece_to_id("<sos>")   # == 2
    eos_id = sp.piece_to_id("<eos>")   # == 3
    vocab_size = sp.get_piece_size()

    max_ctx_tokens  = cfg["max_ctx_tokens"]
    max_resp_tokens = cfg["max_resp_tokens"]

    train_path = artifact_dir / "stage6_train_ids.jsonl"
    val_path   = artifact_dir / "stage6_val_ids.jsonl"
    test_path  = artifact_dir / "stage6_test_ids.jsonl"

    print(f"  Encoding train ({len(train_pairs):,} pairs) …")
    n_train = _encode_split(train_pairs, sp, train_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    print(f"  Encoding val ({len(val_pairs):,} pairs) …")
    n_val = _encode_split(val_pairs, sp, val_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    print(f"  Encoding test ({len(test_pairs):,} pairs) …")
    n_test = _encode_split(test_pairs, sp, test_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    # Build vocab dict {piece_str: id}
    vocab: Dict[str, int] = {sp.id_to_piece(i): i for i in range(vocab_size)}

    vocab_path = artifact_dir / "stage6_vocab.json"
    tmp_vocab  = vocab_path.with_suffix(".tmp")
    with open(tmp_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    os.replace(tmp_vocab, vocab_path)
    print(f"  Vocab saved → {vocab_path.name}  ({vocab_size:,} entries)")

    # Save reverse mapping (id → piece) for evaluation decoding.
    idx2word: Dict[int, str] = {i: sp.id_to_piece(i) for i in range(vocab_size)}
    idx2word_path = artifact_dir / "stage6_idx2word.json"
    tmp_i2w = idx2word_path.with_suffix(".tmp")
    with open(tmp_i2w, "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False)
    os.replace(tmp_i2w, idx2word_path)
    print(f"  idx2word saved → {idx2word_path.name}")

    stats = {
        "stage": 6,
        "vocab_size":  vocab_size,
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      n_test,
        "sos_id":      sos_id,
        "eos_id":      eos_id,
    }
    stats_path = artifact_dir / "stage6_stats.json"
    _save_json(stats, stats_path)

    return str(train_path), str(val_path), str(test_path), vocab, stats


# ── Stage 7 ───────────────────────────────────────────────────────────────────

def stage7_train_fasttext(spm_model_path: str, all_pairs: List[Dict], cfg: dict) -> str:
    """Tokenise all corpus pairs with SPM and train a Word2Vec (skip-gram) model.

    Word2Vec is used instead of FastText because the input is BPE tokens
    (already subword units), so FastText's character n-gram feature adds
    no benefit — only ~3x overhead.

    Uses all available pairs (train + val + test combined) for richer
    embedding coverage.

    Args:
        spm_model_path: Path to the trained SentencePiece .model file.
        all_pairs:      Combined list of dicts with 'ctx' and 'resp' keys.
        cfg:            Pipeline config dict.

    Returns path to the saved model file.
    """
    from tokenizer_utils import load_tokenizer
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sp = load_tokenizer(artifact_dir)

    corpus_path = artifact_dir / "stage7_bpe_corpus.tmp"
    print(f"  Tokenising {len(all_pairs):,} pairs with SPM (batch) …")
    batch_size = 50_000
    with open(corpus_path, "w", encoding="utf-8") as f:
        # Batch-encode ctx and resp separately for speed
        for field in ("ctx", "resp"):
            for start in range(0, len(all_pairs), batch_size):
                batch = [p[field].strip() for p in all_pairs[start : start + batch_size]]
                encoded = sp.encode(batch, out_type=str)
                for pieces in encoded:
                    if pieces:
                        f.write(" ".join(pieces) + "\n")

    n_workers = cfg.get("num_workers", 4)
    n_epochs = cfg.get("fasttext_epochs", 10)
    print(f"  Training Word2Vec  dim={cfg['fasttext_dim']}  epochs={n_epochs}  sg={cfg.get('fasttext_sg', 1)}  workers={n_workers} …")

    # Suppress gensim per-second progress spam; show one bar per epoch instead
    import logging as _logging
    _gensim_logger = _logging.getLogger("gensim.models.word2vec")
    _prev_level = _gensim_logger.level
    _gensim_logger.setLevel(_logging.WARNING)

    from gensim.models.callbacks import CallbackAny2Vec

    class _EpochBar(CallbackAny2Vec):
        """Callback that prints a per-epoch progress bar for Word2Vec training."""
        def __init__(self, total_epochs):
            self.total = total_epochs
            self.current = 0
            self.t0 = time.time()
            self.epoch_t0 = time.time()
        def on_epoch_begin(self, model):
            self.epoch_t0 = time.time()
        def on_epoch_end(self, model):
            self.current += 1
            epoch_time = time.time() - self.epoch_t0
            elapsed = time.time() - self.t0
            eta = elapsed / self.current * (self.total - self.current) if self.current else 0
            bar_len = 30
            filled = int(bar_len * self.current / self.total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  Epoch {self.current:2d}/{self.total} [{bar}] "
                  f"{epoch_time:.0f}s  (total {elapsed:.0f}s, ~{eta:.0f}s remaining)")

    epoch_bar = _EpochBar(n_epochs)

    sentences = LineSentence(str(corpus_path))
    model = Word2Vec(
        sentences,
        vector_size=cfg.get("fasttext_dim", 300),
        epochs=n_epochs,
        min_count=cfg.get("fasttext_min_count", 3),
        window=cfg.get("fasttext_window", 5),
        sg=cfg.get("fasttext_sg", 1),
        workers=n_workers,
        callbacks=[epoch_bar],
    )

    _gensim_logger.setLevel(_prev_level)

    model_path = str(artifact_dir / "stage7_fasttext.model")
    model.save(model_path)
    print(f"  Word2Vec model saved → {model_path}")

    corpus_path.unlink(missing_ok=True)
    return model_path


# ── Stage 8 ───────────────────────────────────────────────────────────────────

def stage8_build_embedding_matrix(
    vocab: Dict[str, int],
    fasttext_model_path: str,
    cfg: dict,
) -> Tuple[str, Dict]:
    """Build numpy embedding matrix [vocab_size × embed_dim] from Word2Vec.

    For each vocab piece (by integer ID order), the embedding is looked up
    using the EXACT piece string including the ▁ word-initial prefix (M5 fix).
    The <pad> row (index 0) is forced to all zeros.

    Returns (matrix_path, stats).
    """
    from gensim.models import Word2Vec

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading Word2Vec model from {fasttext_model_path} …")
    ft_model = Word2Vec.load(fasttext_model_path)

    vocab_size = len(vocab)
    embed_dim  = cfg.get("fasttext_dim", 300)
    print(f"  Building embedding matrix  [{vocab_size} × {embed_dim}] …")

    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    n_found = 0
    for piece_str, idx in vocab.items():
        if idx == 0:
            # <pad> row stays all zeros
            continue
        try:
            vec = ft_model.wv[piece_str]   # exact ▁-prefixed piece string
            matrix[idx] = vec
            n_found += 1
        except KeyError:
            pass   # FastText subword fallback is automatic; this shouldn't happen

    # Guarantee pad row is zero (in case FastText returned something)
    matrix[0] = 0.0

    assert matrix[0].sum() == 0.0, "<pad> row is not all zeros"
    print(f"  Vectors filled: {n_found:,}/{vocab_size:,}  (pad row forced to zeros)")

    matrix_path = artifact_dir / "stage8_embedding_matrix.npy"
    tmp_path    = artifact_dir / "stage8_embedding_matrix_tmp.npy"   # must end .npy so np.save writes here
    np.save(str(tmp_path), matrix)
    os.replace(tmp_path, matrix_path)
    print(f"  Embedding matrix saved → {matrix_path.name}  shape={matrix.shape}")

    stats = {
        "stage": 8,
        "vocab_size":  vocab_size,
        "embed_dim":   embed_dim,
        "n_filled":    n_found,
        "pad_row_sum": float(matrix[0].sum()),
        "matrix_shape": list(matrix.shape),
    }
    _save_json(stats, artifact_dir / "stage8_stats.json")
    return str(matrix_path), stats


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main(cfg: Optional[Dict] = None, script_name: str = "phase1") -> None:
    """Run all pipeline stages sequentially, skipping completed ones.

    Each stage checks whether all its STAGE_ARTIFACTS exist; if so it
    loads from disk and skips computation. Otherwise it runs, saves
    artifacts atomically, and logs timing statistics.

    Args:
        cfg: Config dict; defaults to PHASE1_CONFIG.
        script_name: Used for the run log filename (e.g. "phase1_mini").
    """
    if cfg is None:
        cfg = PHASE1_CONFIG

    from logging_utils import setup_run_logging
    setup_run_logging(script_name, log_dir=cfg.get("log_dir", "new/logs"))

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Configuration summary ─────────────────────────────────────────────────
    domain_filter    = cfg.get("domain_filter", False)
    filter_strategy  = cfg.get("domain_filter_strategy", "union")
    print("=" * 60)
    print("PHASE 1 — CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  artifact_dir       : {cfg['artifact_dir']}")
    print(f"  vocab_size         : {cfg.get('vocab_size', 16000):,}")
    print(f"  max_train_pairs    : {cfg.get('max_train_pairs', 0):,}  (0 = no cap)")
    print(f"  min_ctx_tokens     : {cfg.get('min_ctx_tokens', 3)}")
    print(f"  min_resp_tokens    : {cfg.get('min_resp_tokens', 5)}")
    print(f"  max_resp_tokens    : {cfg.get('max_resp_tokens', 40)}")
    print(f"  max_ctx_turns      : {cfg.get('max_ctx_turns', 8)}")
    print(f"  domain_filter      : {domain_filter}  (stage 4.5)")
    if domain_filter:
        print(f"  filter_strategy    : {filter_strategy}")
        print(f"  ⚡ Stage 4.5 ENABLED — ~73% of pairs retained (union A+B)")
    else:
        print(f"  filter_strategy    : disabled")
    print(f"  spm_vocab_size     : {cfg.get('spm_vocab_size', 16000):,}")
    print(f"  fasttext_dim       : {cfg.get('fasttext_dim', 300)}")
    print(f"  fasttext_epochs    : {cfg.get('fasttext_epochs', 10)}")
    print("=" * 60)
    print()

    # ── Stage 0 — Download corpus from Kaggle if needed ────────────────────
    stage0_download_corpus(cfg)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    s1_path = artifact_dir / "stage1_dialogues.pkl"
    if _stage_done(1, artifact_dir):
        print("✓ Stage 1 already complete — loading dialogues …")
        dialogues = _load_pickle(s1_path)
    else:
        print("=" * 60)
        print("STAGE 1 — Load corpus")
        print("=" * 60)
        t0 = time.time()
        dialogues = stage1_load_corpus(cfg)
        _save_pickle(dialogues, s1_path)
        print(f"Stage 1 done ({_elapsed(t0)})  {len(dialogues):,} dialogues\n")

    # Optional subsample — used by phase1_mini.py to work on 10% of dialogues.
    # Applied after the stage 1 pickle so a full run can share its stage 1
    # cache and mini only re-runs stages 2-8 on the subset.
    subsample_frac = float(cfg.get("stage1_subsample_frac", 1.0))
    if subsample_frac < 1.0:
        n_keep = max(1, int(len(dialogues) * subsample_frac))
        rng = random.Random(42)
        dialogues = rng.sample(dialogues, n_keep)
        print(f"  [mini] Subsampled to {len(dialogues):,} dialogues "
              f"({subsample_frac:.0%} of corpus, seed=42)\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    s2_path       = artifact_dir / "stage2_clean_dialogues.pkl"
    s2_stats_path = artifact_dir / "stage2_stats.json"
    if _stage_done(2, artifact_dir):
        print("✓ Stage 2 already complete — loading clean dialogues …")
        clean_dialogues = _load_pickle(s2_path)
    else:
        print("=" * 60)
        print("STAGE 2 — Clean and filter")
        print("=" * 60)
        t0 = time.time()
        clean_dialogues, s2_stats = stage2_clean_and_filter(dialogues, cfg)
        s2_stats["elapsed"] = _elapsed(t0)
        _save_pickle(clean_dialogues, s2_path)
        _save_json(s2_stats, s2_stats_path)
        del dialogues
        gc.collect()
        _cleanup_intermediate(artifact_dir, ["stage1_dialogues.pkl"])
        print(f"Stage 2 done ({_elapsed(t0)})  {len(clean_dialogues):,} dialogues kept\n")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    s3_train_path = artifact_dir / "stage3_train.pkl"
    s3_val_path   = artifact_dir / "stage3_val.pkl"
    s3_test_path  = artifact_dir / "stage3_test.pkl"
    s3_stats_path = artifact_dir / "stage3_stats.json"
    if _stage_done(3, artifact_dir):
        print("✓ Stage 3 already complete — loading splits …")
        train_dlg = _load_pickle(s3_train_path)
        val_dlg   = _load_pickle(s3_val_path)
        test_dlg  = _load_pickle(s3_test_path)
    else:
        print("=" * 60)
        print("STAGE 3 — Temporal split (by thread first-turn date)")
        print("=" * 60)
        t0 = time.time()
        train_dlg, val_dlg, test_dlg, s3_stats = stage3_temporal_split(clean_dialogues, cfg)
        s3_stats["elapsed"] = _elapsed(t0)
        _save_pickle(train_dlg, s3_train_path)
        _save_pickle(val_dlg,   s3_val_path)
        _save_pickle(test_dlg,  s3_test_path)
        _save_json(s3_stats, s3_stats_path)
        del clean_dialogues
        gc.collect()
        _cleanup_intermediate(artifact_dir, ["stage2_clean_dialogues.pkl"])
        print(f"Stage 3 done ({_elapsed(t0)})\n")

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    s4_train_pkl  = artifact_dir / "stage4_train_pairs.pkl"
    s4_val_pkl    = artifact_dir / "stage4_val_pairs.pkl"
    s4_test_pkl   = artifact_dir / "stage4_test_pairs.pkl"
    s4_stats_path = artifact_dir / "stage4_stats.json"
    if _stage_done(4, artifact_dir):
        print("✓ Stage 4 already complete — loading pairs …")
        train_pairs = _load_pickle(s4_train_pkl)
        val_pairs   = _load_pickle(s4_val_pkl)
        test_pairs  = _load_pickle(s4_test_pkl)
    else:
        print("=" * 60)
        print("STAGE 4 — Generate context-response pairs")
        print("=" * 60)
        t0 = time.time()
        train_pairs, val_pairs, test_pairs, s4_stats = stage4_generate_pairs(
            train_dlg, val_dlg, test_dlg, cfg
        )
        s4_stats["elapsed"] = _elapsed(t0)
        _save_pickle(train_pairs, s4_train_pkl)
        _save_pickle(val_pairs,   s4_val_pkl)
        _save_pickle(test_pairs,  s4_test_pkl)
        _save_json(s4_stats,    s4_stats_path)
        del train_dlg, val_dlg, test_dlg
        gc.collect()
        _cleanup_intermediate(artifact_dir, ["stage3_train.pkl", "stage3_val.pkl", "stage3_test.pkl"])
        print(f"Stage 4 done ({_elapsed(t0)})  train={len(train_pairs):,}  val={len(val_pairs):,}  test={len(test_pairs):,}\n")

    # Write 200-pair human-readable sample files for each split so you can
    # inspect data quality without opening the full multi-hundred-MB JSONs.
    # Samples are drawn randomly each run (reproducible via seed below).
    _write_stage4_samples(
        {"train": train_pairs, "val": val_pairs, "test": test_pairs},
        artifact_dir,
        n=200,
        seed=42,
    )

    # ── Stage 4.5 — Domain filter (optional) ─────────────────────────────────
    s45_train_pkl  = artifact_dir / "stage4_5_train_pairs.pkl"
    s45_val_pkl    = artifact_dir / "stage4_5_val_pairs.pkl"
    s45_test_pkl   = artifact_dir / "stage4_5_test_pairs.pkl"
    s45_stats_path = artifact_dir / "stage4_5_filter_stats.json"
    if cfg.get("domain_filter", False):
        print("=" * 60)
        print("STAGE 4.5 — Domain-focused filtering (A+B union)")
        print("=" * 60)
        t0 = time.time()
        train_pairs, val_pairs, test_pairs, s45_stats = stage4_5_domain_filter(
            train_pairs, val_pairs, test_pairs, cfg
        )
        s45_stats["elapsed"] = _elapsed(t0)
        _save_pickle(train_pairs, s45_train_pkl)
        _save_pickle(val_pairs,   s45_val_pkl)
        _save_pickle(test_pairs,  s45_test_pkl)
        _save_json(s45_stats,   s45_stats_path)
        # Overwrite sample files with filtered pairs so inspect files are accurate
        _write_stage4_samples(
            {"train": train_pairs, "val": val_pairs, "test": test_pairs},
            artifact_dir,
            n=200,
            seed=42,
        )
        # Stage 4 raw pairs superseded by filtered 4.5 pairs
        _cleanup_intermediate(artifact_dir, [
            "stage4_train_pairs.pkl", "stage4_val_pairs.pkl", "stage4_test_pairs.pkl",
            "stage4_train_pairs.json", "stage4_val_pairs.json", "stage4_test_pairs.json",
        ])
        print(
            f"Stage 4.5 done ({_elapsed(t0)})  "
            f"train={len(train_pairs):,}  val={len(val_pairs):,}  test={len(test_pairs):,}\n"
        )
    else:
        print("✓ Stage 4.5 skipped (domain_filter=False)\n")

    # ── Stage 4.6 — Load additional corpora ─────────────────────────────────
    if cfg.get("extra_corpora", True):
        from corpus_loaders import load_all_extra_corpora
        data_dir = Path(cfg["corpus_dir"]).parent
        extra_pairs = load_all_extra_corpora(data_dir, cfg)
        if extra_pairs:
            train_ratio = len(train_pairs) / max(1, len(train_pairs) + len(val_pairs) + len(test_pairs))
            val_ratio = len(val_pairs) / max(1, len(train_pairs) + len(val_pairs) + len(test_pairs))
            # Split extra pairs proportionally: ~80% train, ~10% val, ~10% test
            import random as _rnd
            _rnd.Random(42).shuffle(extra_pairs)
            n_train = int(len(extra_pairs) * train_ratio)
            n_val = int(len(extra_pairs) * val_ratio)
            extra_train = extra_pairs[:n_train]
            extra_val = extra_pairs[n_train:n_train + n_val]
            extra_test = extra_pairs[n_train + n_val:]
            train_pairs.extend(extra_train)
            val_pairs.extend(extra_val)
            test_pairs.extend(extra_test)
            print(f"  Merged: train +{len(extra_train):,}  val +{len(extra_val):,}  "
                  f"test +{len(extra_test):,}")
            print(f"  New totals: train={len(train_pairs):,}  val={len(val_pairs):,}  "
                  f"test={len(test_pairs):,}\n")

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    # Stage 5 produces either .json (HF Tokenizers) or .model (SentencePiece).
    _s5_hf_path  = artifact_dir / "stage5_spm.json"
    _s5_spm_path = artifact_dir / "stage5_spm.model"
    _s5_done = _s5_hf_path.exists() or _s5_spm_path.exists()

    # Cache-invalidation guard for domain_filter
    if cfg.get("domain_filter", False) and _s5_done:
        print("⚠️  WARNING: domain_filter=True but Stage 5 (tokenizer) is already cached.")
        print("   The cached tokenizer may have been trained on UNFILTERED pairs.")
        print("   To fix: delete artifacts/stage5_spm.* and stages 6-8, then rerun.")
        print()
    if _s5_done:
        _s5_found = str(_s5_hf_path) if _s5_hf_path.exists() else str(_s5_spm_path)
        print(f"✓ Stage 5 already complete — tokenizer exists ({Path(_s5_found).name})")
        spm_model_path = _s5_found
    else:
        print("=" * 60)
        print("STAGE 5 — Train BPE tokenizer")
        print("=" * 60)
        t0 = time.time()
        spm_model_path = stage5_train_spm(train_pairs, cfg)
        print(f"Stage 5 done ({_elapsed(t0)})\n")

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    s6_vocab_path = artifact_dir / "stage6_vocab.json"
    if _stage_done(6, artifact_dir):
        print("✓ Stage 6 already complete — loading vocab …")
        vocab = _load_json(s6_vocab_path)
    else:
        print("=" * 60)
        print("STAGE 6 — Encode pairs to BPE IDs")
        print("=" * 60)
        t0 = time.time()
        _, _, _, vocab, s6_stats = stage6_encode_pairs(
            train_pairs, val_pairs, test_pairs, spm_model_path, cfg
        )
        del train_pairs, val_pairs, test_pairs
        gc.collect()
        # Clean up JSON files from stage 4/4.5 (pair pickles kept for stage 7)
        _cleanup_intermediate(artifact_dir, [
            "stage4_5_train_pairs.json", "stage4_5_val_pairs.json", "stage4_5_test_pairs.json",
        ])
        print(f"Stage 6 done ({_elapsed(t0)})\n")

    # ── Stage 7 ──────────────────────────────────────────────────────────────
    s7_model_path = artifact_dir / "stage7_fasttext.model"
    if _stage_done(7, artifact_dir):
        print("✓ Stage 7 already complete — FastText model exists")
        ft_model_path = str(s7_model_path)
    else:
        print("=" * 60)
        print("STAGE 7 — Train FastText on BPE-tokenised corpus")
        print("=" * 60)
        t0 = time.time()
        # Load all 3 splits so FastText sees full vocabulary coverage.
        # NOTE: val/test token co-occurrence statistics are encoded in the initial
        # embedding vectors. Since freeze=False, this is progressively overwritten
        # during Seq2Seq fine-tuning, but the initialisation carries mild leakage.
        # This is a deliberate trade-off: train-only FastText risks OOV tokens in
        # val/test that would receive random (zero) initialisations instead.
        # Reload pairs from pickle (deleted after stage 6 to free memory)
        _s4_prefix = "stage4_5" if cfg.get("domain_filter", False) else "stage4"
        ft_pairs = _load_pickle(artifact_dir / f"{_s4_prefix}_train_pairs.pkl")
        ft_pairs += _load_pickle(artifact_dir / f"{_s4_prefix}_val_pairs.pkl")
        ft_pairs += _load_pickle(artifact_dir / f"{_s4_prefix}_test_pairs.pkl")
        print(f"  FastText corpus: {len(ft_pairs):,} pairs (train + val + test)")
        ft_model_path = stage7_train_fasttext(spm_model_path, ft_pairs, cfg)
        del ft_pairs
        gc.collect()
        # Pair pickles no longer needed — stage 6 JSONL is the training input
        _cleanup_intermediate(artifact_dir, [
            "stage4_train_pairs.pkl", "stage4_val_pairs.pkl", "stage4_test_pairs.pkl",
            "stage4_5_train_pairs.pkl", "stage4_5_val_pairs.pkl", "stage4_5_test_pairs.pkl",
        ])
        print(f"Stage 7 done ({_elapsed(t0)})\n")

    # ── Stage 8 ──────────────────────────────────────────────────────────────
    if _stage_done(8, artifact_dir):
        print("✓ Stage 8 already complete — embedding matrix exists")
    else:
        print("=" * 60)
        print("STAGE 8 — Build embedding matrix")
        print("=" * 60)
        t0 = time.time()
        # Ensure spm_model_path is set in cfg for stage8 lookup
        cfg_with_spm = dict(cfg)
        cfg_with_spm["spm_model_path"] = spm_model_path
        matrix_path, s8_stats = stage8_build_embedding_matrix(vocab, ft_model_path, cfg_with_spm)
        # Word2Vec model no longer needed — only the embedding matrix is used by train.py
        _cleanup_intermediate(artifact_dir, [
            "stage7_fasttext.model", "stage7_fasttext.model.*",
        ])
        print(f"Stage 8 done ({_elapsed(t0)})  matrix shape: {s8_stats['matrix_shape']}\n")

    print("=" * 60)
    print(f"✓ Phase 1 pipeline complete  ({_elapsed(t_total)})")
    print(f"  Artifacts in: {artifact_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
