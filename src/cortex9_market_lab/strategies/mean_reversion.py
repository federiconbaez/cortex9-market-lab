"""Z-score mean reversion around a rolling mean (Bollinger-style bands)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex9_market_lab.strategies.base import Strategy, register


@register
class ZScoreMeanReversion(Strategy):
    """Long when z <= -entry_z, flat again when z >= -exit_z (mirror for short).

    z_t = (close_t - SMA_n(close)_t) / STD_n(close)_t, rolling windows end at bar t.
    """

    name = "mean_reversion"

    def __init__(self, lookback: int = 48, entry_z: float = 2.0, exit_z: float = 0.0) -> None:
        if lookback < 2 or entry_z <= 0 or exit_z >= entry_z:
            raise ValueError("invalid mean_reversion params")
        super().__init__(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        self.lookback, self.entry_z, self.exit_z = lookback, entry_z, exit_z

    @property
    def warmup(self) -> int:
        return self.lookback

    def zscore(self, close: pd.Series) -> pd.Series:
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std(ddof=0)
        return (close - mean) / std.replace(0.0, np.nan)

    def _signals(self, df: pd.DataFrame) -> np.ndarray:
        z = self.zscore(df["close"]).to_numpy()
        out = np.zeros(len(z), dtype=int)
        pos = 0
        for i, zi in enumerate(z):
            if np.isnan(zi):
                pos = 0
            elif pos == 0:
                if zi <= -self.entry_z:
                    pos = 1
                elif zi >= self.entry_z:
                    pos = -1
            elif (pos == 1 and zi >= -self.exit_z) or (pos == -1 and zi <= self.exit_z):
                pos = 0
            out[i] = pos
        return out
