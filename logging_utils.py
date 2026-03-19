"""
logging_utils.py — Timestamped run logging for all main() entry points.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Timestamped run logging with tee to file + terminal
          v3.0.0 — Initial version

Captures ALL console output (print + logging) to a timestamped log file
while keeping output visible in the terminal/notebook simultaneously.

Usage (add as first line in any main()):
    from logging_utils import setup_run_logging
    log_path = setup_run_logging("phase1", log_dir="new/logs")

Log files are named:  {log_dir}/{script_name}_{YYYY-MM-DD_HH-MM-SS}.log
Allows side-by-side comparison of multiple runs over time.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Matches common ANSI escape sequences (colours, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so log files stay readable as plain text."""
    return _ANSI_RE.sub("", text)


class _TeeWriter:
    """Writes to both the original stream AND a log file simultaneously.

    The original stream receives text as-is (preserving colours/progress bars).
    The log file receives text with ANSI codes stripped (clean plain text).
    """

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        # Always write to terminal as-is
        n = self._original.write(text)
        # Write ANSI-stripped version to log file
        clean = _strip_ansi(text)
        if clean:
            self._log_file.write(clean)
        return n or 0

    def flush(self) -> None:
        self._original.flush()
        try:
            self._log_file.flush()
        except ValueError:
            pass  # file already closed

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False

    # Forward attribute access so libraries that probe stream attributes
    # (e.g. rich, tqdm) don't crash
    def __getattr__(self, name):
        return getattr(self._original, name)


def setup_run_logging(
    script_name: str,
    log_dir: str = "new/logs",
    also_capture_stderr: bool = True,
) -> Path:
    """Set up run logging: tee stdout (and optionally stderr) to a timestamped file.

    Call this as the VERY FIRST statement inside any main() function.
    Safe to call from notebooks — each invocation creates a separate log file.

    Args:
        script_name:         Short name used in the log filename (e.g. "phase1").
        log_dir:             Directory for log files. Created if it doesn't exist.
                             Relative to the current working directory.
        also_capture_stderr: If True, stderr is also tee'd to the log file.

    Returns:
        Path to the created log file.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir_path / f"{script_name}_{timestamp}.log"

    # Line-buffered so entries appear immediately even if the process is killed
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    # ── Header ────────────────────────────────────────────────────────────────
    header_lines = [
        "=" * 60,
        f"SCRIPT  : {script_name}",
        f"STARTED : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"PID     : {os.getpid()}",
        f"CWD     : {Path.cwd()}",
        "=" * 60,
        "",
    ]
    log_file.write("\n".join(header_lines))
    log_file.flush()

    # ── Tee stdout ────────────────────────────────────────────────────────────
    # Keep a reference to the original stream so we can always fall back
    original_stdout = sys.__stdout__
    sys.stdout = _TeeWriter(sys.stdout, log_file)

    if also_capture_stderr:
        sys.stderr = _TeeWriter(sys.stderr, log_file)

    # ── Python logging module ─────────────────────────────────────────────────
    # Route logging.xxx() calls through the same tee'd stdout so they also
    # land in the log file without a separate FileHandler (avoids duplication).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,   # this is now the TeeWriter
        force=True,          # override any prior basicConfig (Python ≥3.8)
    )

    # ── Footer on exit ────────────────────────────────────────────────────────
    def _write_footer():
        try:
            footer = (
                f"\n{'=' * 60}\n"
                f"FINISHED: {script_name} — "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 60}\n"
            )
            log_file.write(footer)
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_write_footer)

    # ── Announce log file location ────────────────────────────────────────────
    print(f"📋  Run log → {log_path.resolve()}")

    return log_path
