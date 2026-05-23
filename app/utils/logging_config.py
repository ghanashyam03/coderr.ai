from __future__ import annotations

import logging
import sys
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Console formatter with ANSI color codes for different log levels."""

    GREY = "\x1b[38;5;245m"
    CYAN = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            fmt=f"{color}{self.FORMAT}{self.RESET}",
            datefmt=self.DATE_FORMAT,
        )
        return formatter.format(record)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure root logger with:
    - Console handler (colored, human-readable)
    - Optional file handler (plain text, machine-readable)
    """
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate logs on repeated calls
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ColorFormatter())
    root.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # always full detail in file
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
