"""Moving-average crossover momentum (trend following)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex9_market_lab.strategies.base import Strategy, register


@register
class MACrossMomentum(Strategy):
    """+1 when EMA(fast) > EMA(slow) by more than ``band`` (relative), -1 when below."""

    name = "momentum"

    def __init__(self, fast: int = 24, slow: int = 96, band: float = 0.0) -> None:
        if not 1 <= fast < slow or band < 0:
            raise ValueError("invalid momentum params")
        super().__init__(fast=fast, slow=slow, band=band)
        self.fast, self.slow, self.band = fast, slow, band

    @property
    def warmup(self) -> int:
        return self.slow

    def _signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        f = close.ewm(span=self.fast, adjust=False).mean()
        s = close.ewm(span=self.slow, adjust=False).mean()
        rel = (f / s - 1.0).to_numpy()
        out = np.zeros(len(rel), dtype=int)
        out[rel > self.band] = 1
        out[rel < -self.band] = -1
        return out
