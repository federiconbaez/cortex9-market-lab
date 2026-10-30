"""Deterministic synthetic OHLCV series (GBM and two-state regime switching)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex9_market_lab.config import periods_per_year, timeframe_seconds


def _to_ohlcv(
    close: np.ndarray, rng: np.random.Generator, index: pd.DatetimeIndex, intrabar_vol: np.ndarray
) -> pd.DataFrame:
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick_hi = np.abs(rng.normal(0.0, 1.0, close.size)) * intrabar_vol * close * 0.5
    wick_lo = np.abs(rng.normal(0.0, 1.0, close.size)) * intrabar_vol * close * 0.5
    high = np.maximum(open_, close) + wick_hi
    low = np.maximum(np.minimum(open_, close) - wick_lo, 1e-9)
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=close.size)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def _index(n: int, timeframe: str, start: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        periods=n,
        freq=pd.Timedelta(seconds=timeframe_seconds(timeframe)),
        name="timestamp",
    )


def gbm(
    n: int,
    *,
    seed: int,
    timeframe: str = "1h",
    s0: float = 100.0,
    mu_annual: float = 0.0,
    sigma_annual: float = 0.6,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Geometric Brownian motion with constant drift/volatility."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year(timeframe)
    shocks = rng.normal(0.0, 1.0, n)
    log_ret = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * shocks
    log_ret[0] = 0.0
    close = s0 * np.exp(np.cumsum(log_ret))
    vol = np.full(n, sigma_annual * np.sqrt(dt))
    return _to_ohlcv(close, rng, _index(n, timeframe, start), vol)


def regime_switching(
    n: int,
    *,
    seed: int,
    timeframe: str = "1h",
    s0: float = 100.0,
    regimes: tuple[tuple[float, float], ...] = ((0.8, 0.4), (-0.6, 0.9)),
    switch_prob: float = 0.01,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Markov regime-switching GBM; each regime is (mu_annual, sigma_annual)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year(timeframe)
    state = 0
    log_ret = np.zeros(n)
    vol = np.zeros(n)
    for i in range(n):
        if i > 0 and rng.random() < switch_prob:
            state = (state + 1) % len(regimes)
        mu, sigma = regimes[state]
        vol[i] = sigma * np.sqrt(dt)
        if i > 0:
            log_ret[i] = (mu - 0.5 * sigma**2) * dt + vol[i] * rng.normal()
    close = s0 * np.exp(np.cumsum(log_ret))
    return _to_ohlcv(close, rng, _index(n, timeframe, start), vol)


def synthetic_universe(
    symbols: list[str], n: int, *, seed: int, timeframe: str, model: str = "regime"
) -> dict[str, pd.DataFrame]:
    """One independent deterministic series per symbol (seed offset by position)."""
    out: dict[str, pd.DataFrame] = {}
    for k, sym in enumerate(symbols):
        if model == "gbm":
            out[sym] = gbm(n, seed=seed + k, timeframe=timeframe)
        else:
            out[sym] = regime_switching(n, seed=seed + k, timeframe=timeframe)
    return out
