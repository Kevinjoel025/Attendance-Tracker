"""
core/logging_setup.py
─────────────────────
Configures structured rotating file logging + console output.

SECURITY RULES (enforced by convention — not automatically filtered):
- NEVER log passwords, session tokens, cookies, or authentication headers.
- Log attendance data is acceptable (not a credential).
- Log file is gitignored and must not be committed.

Usage:
    from attendance_tracker.core.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from attendance_tracker.core.config import config

_CONFIGURED = False


def setup_logging() -> None:
    """
    Configure root logger once.
    Safe to call multiple times — only configures on first call.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Ensure log directory exists
    logs_dir = Path(config.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "attendance_tracker.log"

    # ── Formatter ────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Rotating file handler ────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(log_level)

    # ── Console handler (WARNING and above only) ─────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.WARNING)

    # ── Root logger ──────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Ensures logging is configured first.

    Example:
        logger = get_logger(__name__)
        logger.info("Snapshot saved.")
    """
    setup_logging()
    return logging.getLogger(name)
