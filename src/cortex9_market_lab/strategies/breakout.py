"""Donchian-channel volatility breakout with an ATR margin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex9_market_lab.strategies.base import Strategy, register


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder-style average true range (simple rolling mean of true range)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


@register
class DonchianBreakout(Strategy):
    """Long when close > highest high of the *previous* ``entry`` bars + atr_mult*ATR.

    Exit when close < lowest low of the previous ``exit`` bars. Mirror for short.
    Channels are shifted by one bar so the current bar never sees itself.
    """

    name = "breakout"

    def __init__(
        self, entry: int = 48, exit: int = 24, atr_n: int = 24, atr_mult: float = 0.0
    ) -> None:
        if entry < 2 or exit < 1 or atr_n < 1 or atr_mult < 0:
            raise ValueError("invalid breakout params")
        super().__init__(entry=entry, exit=exit, atr_n=atr_n, atr_mult=atr_mult)
        self.entry, self.exit, self.atr_n, self.atr_mult = entry, exit, atr_n, atr_mult

    @property
    def warmup(self) -> int:
        return max(self.entry, self.exit, self.atr_n) + 1

    def _signals(self, df: pd.DataFrame) -> np.ndarray:
        hi_in = df["high"].rolling(self.entry).max().shift(1).to_numpy()
        lo_in = df["low"].rolling(self.entry).min().shift(1).to_numpy()
        hi_out = df["high"].rolling(self.exit).max().shift(1).to_numpy()
        lo_out = df["low"].rolling(self.exit).min().shift(1).to_numpy()
        a = atr(df, self.atr_n).to_numpy()
        close = df["close"].to_numpy()
        out = np.zeros(len(close), dtype=int)
        pos = 0
        for i in range(len(close)):
            if np.isnan(hi_in[i]) or np.isnan(a[i]) or np.isnan(lo_out[i]):
                out[i] = 0
                continue
            margin = self.atr_mult * a[i]
            if pos == 0:
                if close[i] > hi_in[i] + margin:
                    pos = 1
                elif close[i] < lo_in[i] - margin:
                    pos = -1
            elif (pos == 1 and close[i] < lo_out[i]) or (pos == -1 and close[i] > hi_out[i]):
                pos = 0
            out[i] = pos
        return out
