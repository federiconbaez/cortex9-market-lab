"""Structured JSON logging and lightweight in-process metrics."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line; extra fields are merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=False)


class _CurrentStderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is *now* (survives stream swaps, e.g. in tests)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.setStream(sys.stderr)
        super().emit(record)


def reset_logging() -> None:
    """Detach and close every handler installed by :func:`configure_logging`."""
    logger = logging.getLogger("cortex9_market_lab")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(level: str = "INFO", file: str | None = None) -> None:
    """Install JSON handlers on the ``cortex9_market_lab`` logger.

    Idempotent: previous handlers are closed first, so repeated calls never leak file
    handles or keep writing to a stream that was closed in the meantime.
    """
    reset_logging()
    logger = logging.getLogger("cortex9_market_lab")
    logger.setLevel(level)
    stream = _CurrentStderrHandler()
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)
    if file:
        from pathlib import Path

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"cortex9_market_lab.{name}")


class Metrics:
    """Thread-safe counters and gauges, serializable to a dict."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {"counters": dict(self._counters), "gauges": dict(self._gauges)}

    def load(self, snap: dict[str, dict[str, float]]) -> None:
        with self._lock:
            self._counters = defaultdict(float, snap.get("counters", {}))
            self._gauges = dict(snap.get("gauges", {}))
