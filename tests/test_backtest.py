import numpy as np
import pandas as pd
import pytest

from cortex9_market_lab import strategies
from cortex9_market_lab.backtest.engine import align, run_backtest
from cortex9_market_lab.config import CostConfig, RiskConfig
from cortex9_market_lab.data.synthetic import gbm, synthetic_universe
from cortex9_market_lab.strategies.base import Strategy

NO_COST = CostConfig(fee_bps=0, slippage_bps=0)


class OneShot(Strategy):
    """Long only on bars [start, end)."""

    name = "one_shot_test"

    def __init__(self, start: int, end: int) -> None:
        super().__init__(start=start, end=end)
        self.start, self.end = start, end

    @property
    def warmup(self) -> int:
        return 0

    def _signals(self, df):
        s = np.zeros(len(df), dtype=int)
        s[self.start : self.end] = 1
        return s


class AlwaysLong(OneShot):
    name = "always_long_test"

    def __init__(self) -> None:
        super().__init__(0, 10**9)


def _risk(**kw):
    base = dict(
        fraction=1.0,
        max_symbol_exposure=1.0,
        max_total_exposure=1.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        max_drawdown_kill=None,
    )
    base.update(kw)
    return RiskConfig(**base)


def test_signal_executes_at_next_open_with_slippage():
    df = gbm(50, seed=1)
    res = run_backtest(
        {"X": df},
        OneShot(10, 20),
        timeframe="1h",
        costs=CostConfig(fee_bps=10, slippage_bps=5),
        risk_cfg=_risk(fraction=0.5, max_symbol_exposure=0.5),
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_time == df.index[11].isoformat()
    assert t.entry_price == pytest.approx(df["open"].iloc[11] * 1.0005)
    assert t.exit_time == df.index[21].isoformat()
    assert t.exit_price == pytest.approx(df["open"].iloc[21] * 0.9995)
    assert res.metrics["fees_paid"] == pytest.approx(t.fees)
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 + t.pnl)


def test_engine_has_no_lookahead_on_future_shock():
    data = synthetic_universe(["A/U", "B/U"], 600, seed=3, timeframe="1h")
    strat = strategies.create("momentum", {"fast": 5, "slow": 30})
    base = run_backtest(data, strat, timeframe="1h", costs=CostConfig(), risk_cfg=RiskConfig())
    shocked = {k: v.copy() for k, v in data.items()}
    for v in shocked.values():
        v.iloc[400:, :4] *= 0.3
    alt = run_backtest(shocked, strat, timeframe="1h", costs=CostConfig(), risk_cfg=RiskConfig())
    # equity up to bar 399 (marked at close of 399) must be identical
    pd.testing.assert_series_equal(base.equity.iloc[:400], alt.equity.iloc[:400])


def test_buy_and_hold_without_costs_matches_price_path():
    df = gbm(300, seed=9)
    res = run_backtest({"X": df}, AlwaysLong(), timeframe="1h", costs=NO_COST, risk_cfg=_risk())
    expected = 10_000.0 * df["close"].iloc[-1] / df["open"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert res.metrics["n_trades"] == 1
    assert res.trades[0].exit_reason == "end_of_data"


def test_costs_reduce_equity_by_exactly_fees_and_slippage_on_flat_prices():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx
    )
    res = run_backtest(
        {"X": df},
        OneShot(2, 5),
        timeframe="1h",
        costs=CostConfig(fee_bps=10, slippage_bps=5),
        risk_cfg=_risk(fraction=0.5, max_symbol_exposure=0.5),
    )
    t = res.trades[0]
    qty = 5_000.0 / 100.05
    expected = -(1_000 * 0 + qty * (100.05 - 99.95)) - 5.0 - qty * 99.95 * 0.001
    assert t.pnl == pytest.approx(expected)
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 + expected)


def test_stop_loss_triggers_and_blocks_reentry():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    close = np.array([100, 100, 100, 99, 90, 90, 90, 90, 90, 90], dtype=float)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx
    )
    res = run_backtest(
        {"X": df}, AlwaysLong(), timeframe="1h", costs=NO_COST, risk_cfg=_risk(stop_loss_pct=0.05)
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(90.0)  # gapped through 95 -> filled at the open
    assert res.counters["counters"]["exit.stop_loss"] == 1


def test_kill_switch_flattens_and_halts():
    idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    close = np.array([100, 100, 95, 90, 85, 80, 75, 80, 90, 100, 110, 120], dtype=float)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx
    )
    res = run_backtest(
        {"X": df},
        AlwaysLong(),
        timeframe="1h",
        costs=NO_COST,
        risk_cfg=_risk(max_drawdown_kill=0.1),
    )
    assert res.halted_at is not None
    assert [t.exit_reason for t in res.trades] == ["kill_switch"]
    assert res.metrics["n_trades"] == 1


def test_total_exposure_cap_across_symbols():
    data = {s: gbm(50, seed=k) for k, s in enumerate(["A/U", "B/U", "C/U"])}
    res = run_backtest(
        data,
        OneShot(5, 40),
        timeframe="1h",
        costs=NO_COST,
        risk_cfg=_risk(fraction=0.4, max_symbol_exposure=0.4, max_total_exposure=0.8),
    )
    # 0.4 + 0.4 fills the 0.8 budget at bar 6; the third symbol gets nothing then.
    first = [t for t in res.trades if t.entry_time == data["A/U"].index[6].isoformat()]
    assert sorted(t.symbol for t in first) == ["A/U", "B/U"]
    # a blocked entry is retried on later bars only with whatever room price moves free up
    for t in res.trades:
        if t.symbol == "C/U":
            assert t.entry_time > data["A/U"].index[6].isoformat()
            assert t.qty * t.entry_price < 0.05 * 10_000.0


def test_deterministic_reruns():
    data = synthetic_universe(["A/U", "B/U"], 800, seed=21, timeframe="1h")
    s = strategies.create("breakout", {"entry": 20, "exit": 10})
    a = run_backtest(data, s, timeframe="1h", costs=CostConfig(), risk_cfg=RiskConfig())
    b = run_backtest(data, s, timeframe="1h", costs=CostConfig(), risk_cfg=RiskConfig())
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.metrics == b.metrics


def test_align_inner_join_and_empty():
    a = gbm(10, seed=1)
    b = gbm(10, seed=2).iloc[3:]
    out = align({"a": a, "b": b})
    assert len(out["a"]) == 7 and out["a"].index.equals(out["b"].index)
    with pytest.raises(ValueError):
        align({"a": a.iloc[:2], "b": a.iloc[5:]})


def test_result_save(tmp_path):
    df = gbm(100, seed=1)
    res = run_backtest(
        {"X": df}, OneShot(10, 20), timeframe="1h", costs=CostConfig(), risk_cfg=RiskConfig()
    )
    out = res.save(tmp_path / "r")
    assert {p.name for p in out.iterdir()} == {"equity.csv", "trades.csv", "report.json"}
