"""Turns target sides into simulated fills. Shared by backtest and paper engines so both
use exactly the same sizing, cost and risk logic."""

from __future__ import annotations

from cortex9_market_lab.execution.portfolio import Portfolio, Trade
from cortex9_market_lab.risk.manager import RiskManager

MIN_FRACTION = 1e-6


def apply_targets(
    portfolio: Portfolio,
    risk: RiskManager,
    targets: dict[str, int],
    prices: dict[str, float],
    ts: str,
    vols: dict[str, float | None] | None = None,
) -> list[Trade]:
    """Close positions whose side differs from the target, then open new ones.

    Closes run first so that freed exposure is available to new entries.
    ``prices`` must contain every symbol that is held or targeted.
    """
    closed: list[Trade] = []
    for sym, target in targets.items():
        cur = portfolio.side(sym)
        if cur != 0 and target != cur:
            reason = "kill_switch" if risk.halted else "signal"
            closed.append(portfolio.close(sym, prices[sym], ts, reason))
    for sym, target in targets.items():
        if target == 0 or portfolio.side(sym) != 0:
            continue
        equity = portfolio.equity(prices)
        if equity <= 0:
            break
        frac = risk.target_fraction((vols or {}).get(sym))
        frac = risk.cap_for_total(frac, portfolio.gross_exposure(prices))
        if frac < MIN_FRACTION:
            continue
        est_fill = portfolio.costs.fill_price(prices[sym], buy=target > 0)
        stop, take = risk.stop_levels(target, est_fill)
        portfolio.open(sym, target, frac * equity, prices[sym], ts, stop, take)
    return closed


def apply_protective_exit(
    portfolio: Portfolio,
    risk: RiskManager,
    symbol: str,
    open_: float,
    high: float,
    low: float,
    ts: str,
) -> Trade | None:
    """Check stop-loss / take-profit for one held symbol and close it if hit."""
    pos = portfolio.positions.get(symbol)
    if pos is None:
        return None
    hit = risk.check_exit(pos.side, pos.stop_price, pos.take_price, open_, high, low)
    if hit is None:
        return None
    side = pos.side
    trade = portfolio.close(symbol, hit.price, ts, hit.reason)
    if hit.reason == "stop_loss":
        risk.mark_stopped(symbol, side)
    return trade


def flatten_all(
    portfolio: Portfolio, prices: dict[str, float], ts: str, reason: str
) -> list[Trade]:
    return [portfolio.close(s, prices[s], ts, reason) for s in list(portfolio.positions)]
