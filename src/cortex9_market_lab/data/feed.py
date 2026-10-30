"""Polling live feed: emits newly closed bars and latest prices for paper trading."""

from __future__ import annotations

import pandas as pd

from cortex9_market_lab.config import timeframe_seconds
from cortex9_market_lab.data.exchange import MarketDataClient


class PollingFeed:
    """Keeps a rolling window of *closed* bars per symbol by polling REST endpoints.

    A bar is considered closed when ``bar_open + timeframe <= now``. The still-forming
    bar returned by the exchange is never exposed to strategies (no lookahead).
    """

    def __init__(
        self, client: MarketDataClient, symbols: list[str], timeframe: str, window: int
    ) -> None:
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.window = window
        self.bars: dict[str, pd.DataFrame] = {}
        self._step = pd.Timedelta(seconds=timeframe_seconds(timeframe))

    def _closed(self, df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return df[df.index + self._step <= now]

    def poll_bars(self, now: pd.Timestamp) -> dict[str, bool]:
        """Refresh bars; returns {symbol: True if a new closed bar appeared}."""
        new: dict[str, bool] = {}
        for sym in self.symbols:
            fresh = self._closed(
                self.client.fetch_recent_ohlcv(sym, self.timeframe, min(self.window, 300)), now
            )
            prev = self.bars.get(sym)
            last_prev = prev.index[-1] if prev is not None and not prev.empty else None
            merged = fresh if prev is None else pd.concat([prev, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index().iloc[-self.window :]
            self.bars[sym] = merged
            advanced = last_prev is None or merged.index[-1] > last_prev
            new[sym] = bool(not merged.empty and advanced)
        return new

    def last_prices(self) -> dict[str, float]:
        return {sym: self.client.fetch_last_price(sym) for sym in self.symbols}
