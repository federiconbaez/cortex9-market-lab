"""Performance metrics. All formulas are plain and documented so they can be audited.

* simple returns r_t = E_t / E_{t-1} - 1
* Sharpe  = mean(r) / std(r, ddof=1) * sqrt(P)           (risk-free rate = 0)
* Sortino = mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(P)
* CAGR    = (E_T / E_0) ** (1 / years) - 1, years = elapsed calendar time / 365.25 d
* max drawdown = min_t (E_t / max_{s<=t} E_s - 1)        (reported as a negative number)
* win rate = winning trades / closed trades; profit factor = gross wins / |gross losses|
* exposure = share of bars with at least one open position
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from cortex9_market_lab.execution.portfolio import Trade


def returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(r: pd.Series, periods_per_year: float) -> float:
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return (
        float("nan")
        if sd == 0 or math.isnan(sd)
        else float(r.mean() / sd * math.sqrt(periods_per_year))
    )


def sortino(r: pd.Series, periods_per_year: float) -> float:
    if len(r) < 2:
        return float("nan")
    dd = math.sqrt(float((np.minimum(r.to_numpy(), 0.0) ** 2).mean()))
    return float("nan") if dd == 0 else float(r.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    if years <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def trade_stats(trades: Sequence[Trade]) -> dict[str, float]:
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_loss = -sum(losses)
    return {
        "n_trades": float(len(pnls)),
        "win_rate": len(wins) / len(pnls) if pnls else float("nan"),
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else float("nan"),
        "avg_trade_pnl": float(np.mean(pnls)) if pnls else float("nan"),
    }


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    in_market: pd.Series,
    periods_per_year: float,
    fees_paid: float = 0.0,
) -> dict[str, float]:
    r = returns(equity)
    out: dict[str, float] = {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "exposure": float(in_market.mean()) if len(in_market) else 0.0,
        "fees_paid": float(fees_paid),
    }
    out.update(trade_stats(trades))
    return out
