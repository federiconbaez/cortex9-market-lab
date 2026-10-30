"""Deterministic event-driven backtester over closed bars.

Timeline for bar i (all symbols share one aligned index):
  1. at the OPEN of bar i, execute the targets decided at the CLOSE of bar i-1;
  2. during bar i, check stop-loss / take-profit against bar i's high/low;
  3. at the CLOSE of bar i, mark to market, update the drawdown kill-switch and
     compute the next targets from signals that only use bars <= i.
So a signal never trades on the bar that produced it (no lookahead).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cortex9_market_lab.backtest.metrics import compute_metrics
from cortex9_market_lab.config import CostConfig, RiskConfig, periods_per_year
from cortex9_market_lab.execution.costs import CostModel
from cortex9_market_lab.execution.portfolio import Portfolio, Trade
from cortex9_market_lab.execution.router import apply_protective_exit, apply_targets, flatten_all
from cortex9_market_lab.observability.logging import Metrics, get_logger
from cortex9_market_lab.risk.manager import RiskManager
from cortex9_market_lab.strategies.base import Strategy

log = get_logger("backtest")


@dataclass
class BacktestResult:
    strategy: dict[str, Any]
    equity: pd.Series
    in_market: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]
    benchmark: dict[str, float]
    halted_at: str | None = None
    counters: dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.equity.rename("equity").to_csv(out / "equity.csv")
        pd.DataFrame([asdict(t) for t in self.trades]).to_csv(out / "trades.csv", index=False)
        report = {
            "strategy": self.strategy,
            "metrics": self.metrics,
            "benchmark_buy_and_hold": self.benchmark,
            "halted_at": self.halted_at,
            "counters": self.counters,
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        return out


def align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every symbol to the common set of timestamps (inner join)."""
    idx: pd.DatetimeIndex | None = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None or len(idx) == 0:
        raise ValueError("no overlapping bars across symbols")
    return {s: df.loc[idx] for s, df in data.items()}


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: Strategy,
    *,
    timeframe: str,
    costs: CostConfig,
    risk_cfg: RiskConfig,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    data = align(data)
    symbols = list(data)
    index = next(iter(data.values())).index
    n = len(index)
    ppy = periods_per_year(timeframe)

    signals = {s: strategy.generate(df).to_numpy() for s, df in data.items()}
    vols = {
        s: np.log(df["close"]).diff().rolling(risk_cfg.vol_lookback).std(ddof=1).to_numpy()
        for s, df in data.items()
    }
    o = {s: df["open"].to_numpy() for s, df in data.items()}
    h = {s: df["high"].to_numpy() for s, df in data.items()}
    lo = {s: df["low"].to_numpy() for s, df in data.items()}
    c = {s: df["close"].to_numpy() for s, df in data.items()}

    portfolio = Portfolio(initial_cash, CostModel(costs.fee_bps, costs.slippage_bps))
    risk = RiskManager(risk_cfg, ppy, peak_equity=initial_cash)
    counters = Metrics()
    equity = np.empty(n)
    in_mkt = np.zeros(n, dtype=bool)
    pending: dict[str, int] | None = None
    pending_vol: dict[str, float | None] = {}

    for i in range(n):
        ts = index[i].isoformat()
        if pending is not None:
            opens = {s: o[s][i] for s in symbols}
            for t in apply_targets(portfolio, risk, pending, opens, ts, pending_vol):
                counters.inc(f"exit.{t.exit_reason}")
        for s in symbols:
            t = apply_protective_exit(portfolio, risk, s, o[s][i], h[s][i], lo[s][i], ts)
            if t is not None:
                counters.inc(f"exit.{t.exit_reason}")
        closes = {s: c[s][i] for s in symbols}
        in_mkt[i] = bool(portfolio.positions)
        equity[i] = portfolio.equity(closes)
        if risk.update_equity(equity[i], ts):
            counters.inc("kill_switch")
            log.warning("kill switch tripped", extra={"ts": ts, "equity": equity[i]})
        pending = {s: risk.desired_side(s, int(signals[s][i])) for s in symbols}
        pending_vol = {s: (None if np.isnan(vols[s][i]) else float(vols[s][i])) for s in symbols}

    last_ts = index[-1].isoformat()
    closes = {s: c[s][-1] for s in symbols}
    for t in flatten_all(portfolio, closes, last_ts, "end_of_data"):
        counters.inc(f"exit.{t.exit_reason}")
    equity[-1] = portfolio.cash  # final equity after paying exit costs

    eq = pd.Series(equity, index=index, name="equity")
    metrics = compute_metrics(
        eq, portfolio.trades, pd.Series(in_mkt, index=index), ppy, portfolio.fees_paid
    )
    bench = {s: float(c[s][-1] / c[s][0] - 1.0) for s in symbols}
    bench["equal_weight"] = float(np.mean([bench[s] for s in symbols]))
    log.info(
        "backtest done",
        extra={"strategy": strategy.name, "bars": n, "trades": len(portfolio.trades)},
    )
    return BacktestResult(
        strategy.describe(),
        eq,
        pd.Series(in_mkt, index=index),
        portfolio.trades,
        metrics,
        bench,
        risk.halted_at,
        counters.snapshot(),
    )
