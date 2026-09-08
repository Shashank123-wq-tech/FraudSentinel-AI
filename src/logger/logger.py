"""Centralized logging utilities for FraudSentinel AI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = str(level).upper().strip()
    return getattr(logging, value, logging.INFO)


def get_logger(
    name: str = "fraudsentinel",
    level: str | int = "INFO",
    file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger without duplicating handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    # Configure the stream handler once.
    has_stream = any(
        getattr(handler, "_fraudsentinel_stream", False)
        for handler in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(_resolve_level(level))
        stream_handler._fraudsentinel_stream = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    # Configure file handler once when a path is provided.
    if file:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(file_path.resolve())
        has_file = any(
            getattr(handler, "_fraudsentinel_file", None) == resolved
            for handler in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(_resolve_level(level))
            file_handler._fraudsentinel_file = resolved  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(_resolve_level(level))

    return logger
