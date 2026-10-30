"""Shared fixtures: an offline fake ccxt exchange and synthetic data."""

from __future__ import annotations

from typing import Any

import ccxt
import pandas as pd
import pytest

from cortex9_market_lab.data.synthetic import gbm
from cortex9_market_lab.observability.logging import reset_logging


@pytest.fixture(autouse=True)
def _clean_logging():
    """Each test starts and ends without handlers bound to an old (closed) stream."""
    reset_logging()
    yield
    reset_logging()


class FakeExchange:
    """Minimal stand-in for a ccxt exchange backed by in-memory frames."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        fail_load: int = 0,
        fail_fetch: int = 0,
        last_price: dict[str, float] | None = None,
        page_cap: int = 100,
    ) -> None:
        self.frames = frames
        self.fail_load = fail_load
        self.fail_fetch = fail_fetch
        self.last_price = last_price or {}
        self.page_cap = page_cap
        self.calls: list[tuple[str, Any]] = []
        # optional "now" in ms: bars that open after it are not served (like a real venue)
        self.now_ms: Any = None

    def load_markets(self) -> dict[str, Any]:
        self.calls.append(("load_markets", None))
        if self.fail_load > 0:
            self.fail_load -= 1
            raise ccxt.NetworkError("simulated network failure")
        return {s: {} for s in self.frames}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", since: int | None = None, limit: int = 100
    ) -> list[list[float]]:
        self.calls.append(("fetch_ohlcv", (symbol, since, limit)))
        if self.fail_fetch > 0:
            self.fail_fetch -= 1
            raise ccxt.RateLimitExceeded("simulated 429")
        df = self._visible(symbol)
        ts = df.index.view("int64") // 1_000_000
        limit = min(limit, self.page_cap)
        sel = df[ts >= since] if since is not None else df
        sel = sel.iloc[:limit] if since is not None else sel.iloc[-limit:]
        sel_ts = sel.index.view("int64") // 1_000_000
        return [
            [int(t), r.open, r.high, r.low, r.close, r.volume]
            for t, r in zip(sel_ts, sel.itertuples(), strict=True)
        ]

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetch_ticker", symbol))
        if symbol in self.last_price:
            return {"last": self.last_price[symbol]}
        return {"last": float(self._visible(symbol)["close"].iloc[-1])}

    def _visible(self, symbol: str) -> pd.DataFrame:
        df = self.frames[symbol]
        if self.now_ms is None:
            return df
        ts = df.index.view("int64") // 1_000_000
        return df[ts <= self.now_ms()]


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return {
        "AAA/USDT": gbm(500, seed=1, timeframe="1h", start="2024-01-01"),
        "BBB/USDT": gbm(500, seed=2, timeframe="1h", start="2024-01-01"),
    }


@pytest.fixture
def no_sleep() -> list[float]:
    return []
