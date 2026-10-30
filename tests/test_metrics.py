import math

import pandas as pd
import pytest

from cortex9_market_lab.backtest import metrics as m
from cortex9_market_lab.execution.portfolio import Trade


def _eq(values, freq="1D"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_sharpe_sortino_drawdown_hand_computed():
    eq = _eq([100.0, 110.0, 99.0, 108.9])
    r = m.returns(eq)
    assert r.tolist() == pytest.approx([0.1, -0.1, 0.1])
    mean = 0.1 / 3
    sd = math.sqrt(((0.1 - mean) ** 2 * 2 + (-0.1 - mean) ** 2) / 2)  # ddof=1
    assert m.sharpe(r, 1.0) == pytest.approx(mean / sd)
    assert m.sharpe(r, 365.0) == pytest.approx(mean / sd * math.sqrt(365.0))
    downside = math.sqrt(0.01 / 3)
    assert m.sortino(r, 1.0) == pytest.approx(mean / downside)
    assert m.max_drawdown(eq) == pytest.approx(99.0 / 110.0 - 1.0)


def test_cagr_two_years_ten_percent():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=2 * 365.25),
        ]
    )
    eq = pd.Series([100.0, 121.0], index=idx)
    assert m.cagr(eq) == pytest.approx(0.10)


def _trade(pnl):
    return Trade("X", 1, 1.0, "a", 100.0, "b", 100.0, 0.0, pnl, "signal")


def test_trade_stats():
    s = m.trade_stats([_trade(10), _trade(-5), _trade(20), _trade(-5)])
    assert s["n_trades"] == 4
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["profit_factor"] == pytest.approx(3.0)
    assert s["avg_trade_pnl"] == pytest.approx(5.0)


def test_degenerate_cases_are_nan_not_errors():
    flat = _eq([100.0, 100.0, 100.0])
    assert math.isnan(m.sharpe(m.returns(flat), 1.0))
    assert math.isnan(m.sortino(m.returns(flat), 1.0))
    s = m.trade_stats([])
    assert s["n_trades"] == 0 and math.isnan(s["win_rate"])
    assert math.isnan(m.trade_stats([_trade(5)])["profit_factor"])


def test_compute_metrics_exposure():
    eq = _eq([100.0, 101.0, 102.0, 103.0])
    in_mkt = pd.Series([False, True, True, False], index=eq.index)
    out = m.compute_metrics(eq, [], in_mkt, 365.0, fees_paid=1.5)
    assert out["exposure"] == pytest.approx(0.5)
    assert out["total_return"] == pytest.approx(0.03)
    assert out["fees_paid"] == 1.5
