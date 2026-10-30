"""Explicit, testable risk controls: sizing, stops, exposure caps, drawdown kill-switch."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cortex9_market_lab.config import RiskConfig


@dataclass
class ExitDecision:
    reason: str  # "stop_loss" | "take_profit"
    price: float


@dataclass
class RiskManager:
    cfg: RiskConfig
    periods_per_year: float
    peak_equity: float = 0.0
    halted: bool = False
    halted_at: str | None = None
    # side that was stopped out per symbol; re-entry in that side is blocked
    # until the strategy signal changes away from it.
    stopped: dict[str, int] = field(default_factory=dict)

    # -- direction -----------------------------------------------------------------
    def desired_side(self, symbol: str, signal: int) -> int:
        """Translate a raw strategy signal into an allowed target side."""
        if self.halted:
            return 0
        side = signal if (signal >= 0 or self.cfg.allow_short) else 0
        blocked = self.stopped.get(symbol)
        if blocked is not None:
            if side == blocked:
                return 0
            del self.stopped[symbol]
        return side

    def mark_stopped(self, symbol: str, side: int) -> None:
        self.stopped[symbol] = side

    # -- sizing ----------------------------------------------------------------------
    def target_fraction(self, realized_vol_per_bar: float | None) -> float:
        """Fraction of equity to allocate to a new position before exposure caps."""
        if self.cfg.sizing == "fixed_fractional":
            frac = self.cfg.fraction
        else:
            if realized_vol_per_bar is None or not math.isfinite(realized_vol_per_bar):
                return 0.0
            ann = realized_vol_per_bar * math.sqrt(self.periods_per_year)
            frac = self.cfg.vol_target_annual / ann if ann > 0 else 0.0
        return max(0.0, min(frac, self.cfg.max_symbol_exposure))

    def cap_for_total(self, fraction: float, current_gross: float) -> float:
        """Reduce ``fraction`` so that total gross exposure stays <= max_total_exposure."""
        room = max(0.0, self.cfg.max_total_exposure - current_gross)
        return max(0.0, min(fraction, room))

    # -- protective exits ------------------------------------------------------------
    def stop_levels(self, side: int, entry: float) -> tuple[float | None, float | None]:
        sl = self.cfg.stop_loss_pct
        tp = self.cfg.take_profit_pct
        stop = None if sl is None else entry * (1 - side * sl)
        take = None if tp is None else entry * (1 + side * tp)
        return stop, take

    @staticmethod
    def check_exit(
        side: int,
        stop: float | None,
        take: float | None,
        open_: float,
        high: float,
        low: float,
    ) -> ExitDecision | None:
        """Intrabar exit check. Gaps fill at the open; if both levels are touched in the
        same bar the stop is assumed first (conservative)."""
        if side > 0:
            if stop is not None and open_ <= stop:
                return ExitDecision("stop_loss", open_)
            if take is not None and open_ >= take:
                return ExitDecision("take_profit", open_)
            if stop is not None and low <= stop:
                return ExitDecision("stop_loss", stop)
            if take is not None and high >= take:
                return ExitDecision("take_profit", take)
        else:
            if stop is not None and open_ >= stop:
                return ExitDecision("stop_loss", open_)
            if take is not None and open_ <= take:
                return ExitDecision("take_profit", open_)
            if stop is not None and high >= stop:
                return ExitDecision("stop_loss", stop)
            if take is not None and low <= take:
                return ExitDecision("take_profit", take)
        return None

    # -- kill switch -------------------------------------------------------------------
    def update_equity(self, equity: float, ts: str) -> bool:
        """Track peak equity; returns True exactly when the kill-switch trips."""
        self.peak_equity = max(self.peak_equity, equity)
        lim = self.cfg.max_drawdown_kill
        if self.halted or lim is None or self.peak_equity <= 0:
            return False
        if 1.0 - equity / self.peak_equity >= lim - 1e-12:  # float tolerance
            self.halted = True
            self.halted_at = ts
            return True
        return False

    def drawdown(self, equity: float) -> float:
        return 0.0 if self.peak_equity <= 0 else 1.0 - equity / self.peak_equity

    # -- serialization -------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "peak_equity": self.peak_equity,
            "halted": self.halted,
            "halted_at": self.halted_at,
            "stopped": dict(self.stopped),
        }

    def load_state(self, d: dict[str, Any]) -> None:
        self.peak_equity = float(d.get("peak_equity", 0.0))
        self.halted = bool(d.get("halted", False))
        self.halted_at = d.get("halted_at")
        self.stopped = {k: int(v) for k, v in d.get("stopped", {}).items()}
