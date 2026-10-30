"""Strategy interface and pluggable registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class Strategy(ABC):
    """A strategy maps a closed-bar OHLCV frame to a target direction per bar.

    Contract (enforced by tests):
      * output index == input index, values in {-1, 0, +1};
      * value at bar ``t`` depends only on bars ``<= t`` (causal, no lookahead);
      * the engine executes the signal of bar ``t`` at the open of bar ``t+1``.
    """

    name: ClassVar[str] = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars needed before the signal is meaningful (signal is 0 before)."""

    @abstractmethod
    def _signals(self, df: pd.DataFrame) -> np.ndarray:
        """Return an int array of {-1,0,1} aligned with ``df``."""

    def generate(self, df: pd.DataFrame) -> pd.Series:
        sig = np.asarray(self._signals(df), dtype=int)
        if sig.shape[0] != len(df):
            raise ValueError("signal length mismatch")
        sig[: min(self.warmup, len(sig))] = 0
        sig = np.clip(sig, -1, 1)
        return pd.Series(sig, index=df.index, name=self.name)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}


_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in _REGISTRY:
        raise ValueError(f"strategy {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def create(name: str, params: dict[str, Any] | None = None) -> Strategy:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}") from exc
    return cls(**(params or {}))


StrategyFactory = Callable[..., Strategy]
