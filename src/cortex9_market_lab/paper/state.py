"""Atomic JSON persistence for paper-trading state (resumable runs)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

STATE_VERSION = 1


def fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of the parts of the config that make a state file reusable."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    """Write via temp file + os.replace so a crash never leaves a torn file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": STATE_VERSION, **state}
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_state(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported state version {data.get('version')}")
    return data
