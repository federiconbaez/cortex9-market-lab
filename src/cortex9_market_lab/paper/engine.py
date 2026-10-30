"""Paper trading loop on live public data. Simulation only: no order is ever sent.

Each iteration:
  1. poll closed bars and last prices (REST polling, retried with backoff);
  2. check stop-loss / take-profit against the last price;
  3. for symbols with a newly closed bar, compute the strategy signal on closed bars
     only and rebalance at the last price with the same cost model as the backtest;
  4. mark to market, update the drawdown kill-switch, persist state atomically.
The run ends after ``duration_s``, after ``max_iterations``, on SIGINT/SIGTERM, or after
too many consecutive data errors. State is saved on every exit path and is resumed on
the next start if the config fingerprint matches.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from cortex9_market_lab.config import AppConfig, periods_per_year
from cortex9_market_lab.data.exchange import MarketDataClient
from cortex9_market_lab.data.feed import PollingFeed
from cortex9_market_lab.execution.costs import CostModel
from cortex9_market_lab.execution.portfolio import Portfolio
from cortex9_market_lab.execution.router import apply_protective_exit, apply_targets
from cortex9_market_lab.observability.logging import Metrics, get_logger
from cortex9_market_lab.paper.state import fingerprint, load_state, save_state
from cortex9_market_lab.risk.manager import RiskManager
from cortex9_market_lab.strategies.base import Strategy

log = get_logger("paper")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class PaperTrader:
    def __init__(
        self,
        cfg: AppConfig,
        strategy: Strategy,
        client: MarketDataClient,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        state_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.state_path = state_path or cfg.paper.state_path
        self.symbols = cfg.data.symbols
        self.timeframe = cfg.paper.timeframe
        self.metrics = Metrics()
        self.feed = PollingFeed(
            client, self.symbols, self.timeframe, max(cfg.paper.warmup_bars, strategy.warmup + 5)
        )
        self.portfolio = Portfolio(
            cfg.paper.initial_cash, CostModel(cfg.costs.fee_bps, cfg.costs.slippage_bps)
        )
        self.risk = RiskManager(
            cfg.risk, periods_per_year(self.timeframe), peak_equity=cfg.paper.initial_cash
        )
        self.last_signal: dict[str, int] = {}
        self.equity_history: list[tuple[str, float]] = []
        self.iterations = 0
        self.started_at = _iso(self.clock())
        self.resumed = False
        self._stop = threading.Event()
        self.fp = fingerprint(
            {
                "strategy": strategy.describe(),
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "costs": cfg.costs.model_dump(),
                "risk": cfg.risk.model_dump(),
            }
        )

    # -- lifecycle -----------------------------------------------------------------
    def request_stop(self, *_: Any) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)

    def resume(self) -> bool:
        st = load_state(self.state_path)
        if st is None:
            return False
        if st.get("fingerprint") != self.fp:
            raise ValueError(
                "state file belongs to a different config; use another --state path or delete it"
            )
        self.portfolio = Portfolio.from_dict(st["portfolio"])
        self.risk.load_state(st["risk"])
        self.metrics.load(st.get("metrics", {}))
        self.last_signal = {k: int(v) for k, v in st.get("last_signal", {}).items()}
        self.equity_history = [tuple(x) for x in st.get("equity_history", [])]  # type: ignore[misc]
        self.iterations = int(st.get("iterations", 0))
        self.started_at = st.get("started_at", self.started_at)
        self.resumed = True
        log.info("resumed", extra={"iterations": self.iterations, "path": self.state_path})
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fp,
            "strategy": self.strategy.describe(),
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "exchange": self.client.active_exchange,
            "started_at": self.started_at,
            "updated_at": _iso(self.clock()),
            "iterations": self.iterations,
            "portfolio": self.portfolio.to_dict(),
            "risk": self.risk.state(),
            "metrics": self.metrics.snapshot(),
            "last_signal": self.last_signal,
            "equity_history": self.equity_history[-2000:],
        }

    def persist(self) -> None:
        save_state(self.state_path, self.snapshot())

    # -- one iteration ---------------------------------------------------------------
    def step(self) -> None:
        now = self.clock()
        ts = _iso(now)
        new_bars = self.feed.poll_bars(pd.Timestamp(now, unit="s", tz="UTC"))
        prices = self.feed.last_prices()
        self.metrics.inc("polls")

        for sym in self.symbols:
            p = prices[sym]
            t = apply_protective_exit(self.portfolio, self.risk, sym, p, p, p, ts)
            if t is not None:
                self.metrics.inc(f"exit.{t.exit_reason}")
                log.info(
                    "protective exit", extra={"symbol": sym, "reason": t.exit_reason, "pnl": t.pnl}
                )

        targets: dict[str, int] = {}
        vols: dict[str, float | None] = {}
        for sym in self.symbols:
            bars = self.feed.bars.get(sym)
            if bars is None or len(bars) <= self.strategy.warmup:
                continue
            if new_bars.get(sym) or sym not in self.last_signal:
                self.metrics.inc("bars.closed")
                sig = int(self.strategy.generate(bars).iloc[-1])
                self.last_signal[sym] = sig
            targets[sym] = self.risk.desired_side(sym, self.last_signal[sym])
            lr = np.log(bars["close"]).diff().tail(self.cfg.risk.vol_lookback)
            v = float(lr.std(ddof=1)) if lr.count() > 2 else None
            vols[sym] = v

        held_prices = {s: prices[s] for s in self.symbols}
        for t in apply_targets(self.portfolio, self.risk, targets, held_prices, ts, vols):
            self.metrics.inc(f"exit.{t.exit_reason}")
            log.info("closed", extra={"symbol": t.symbol, "reason": t.exit_reason, "pnl": t.pnl})
        for sym, pos in self.portfolio.positions.items():
            if pos.entry_time == ts:
                self.metrics.inc("entries")
                log.info(
                    "opened",
                    extra={
                        "symbol": sym,
                        "side": pos.side,
                        "qty": pos.qty,
                        "price": pos.entry_price,
                    },
                )

        eq = self.portfolio.equity(held_prices)
        if self.risk.update_equity(eq, ts):
            self.metrics.inc("kill_switch")
            log.warning("kill switch tripped", extra={"equity": eq})
        self.equity_history.append((ts, eq))
        self.metrics.set_gauge("equity", eq)
        self.metrics.set_gauge("gross_exposure", self.portfolio.gross_exposure(held_prices))
        self.metrics.set_gauge("drawdown", self.risk.drawdown(eq))
        self.iterations += 1
        log.info(
            "tick",
            extra={
                "iteration": self.iterations,
                "equity": round(eq, 4),
                "prices": prices,
                "signals": dict(self.last_signal),
                "positions": {s: p.qty for s, p in self.portfolio.positions.items()},
            },
        )

    # -- main loop -------------------------------------------------------------------
    def run(self, duration_s: float, max_iterations: int | None = None) -> dict[str, Any]:
        self._install_signal_handlers()
        self.resume()
        if self.client.active_exchange is None:
            self.client.connect()
        deadline = self.clock() + duration_s
        errors = 0
        n = 0
        stop_reason = "duration"
        log.info(
            "paper start",
            extra={
                "duration_s": duration_s,
                "strategy": self.strategy.name,
                "symbols": self.symbols,
            },
        )
        try:
            while True:
                if self._stop.is_set():
                    stop_reason = "signal"
                    break
                if self.clock() >= deadline:
                    break
                if max_iterations is not None and n >= max_iterations:
                    stop_reason = "max_iterations"
                    break
                try:
                    self.step()
                    errors = 0
                except Exception as exc:  # keep running on transient failures
                    errors += 1
                    self.metrics.inc("errors")
                    log.error("iteration failed", extra={"error": repr(exc), "consecutive": errors})
                    if errors >= self.cfg.paper.max_consecutive_errors:
                        stop_reason = "too_many_errors"
                        break
                n += 1
                self.persist()
                self._wait(min(self.cfg.paper.poll_seconds, max(0.0, deadline - self.clock())))
        finally:
            self.persist()
        summary = self.summary(stop_reason)
        log.info("paper stop", extra=summary)
        return summary

    def _wait(self, seconds: float) -> None:
        end = self.clock() + seconds
        while not self._stop.is_set() and self.clock() < end:
            self.sleep(min(0.5, max(0.0, end - self.clock())))

    def summary(self, stop_reason: str) -> dict[str, Any]:
        last_eq = self.equity_history[-1][1] if self.equity_history else self.portfolio.cash
        return {
            "stop_reason": stop_reason,
            "exchange": self.client.active_exchange,
            "resumed": self.resumed,
            "iterations": self.iterations,
            "start_equity": self.cfg.paper.initial_cash,
            "last_equity": last_eq,
            "open_positions": {s: p.qty for s, p in self.portfolio.positions.items()},
            "closed_trades": len(self.portfolio.trades),
            "fills": self.portfolio.fills,
            "fees_paid": self.portfolio.fees_paid,
            "halted": self.risk.halted,
            "last_signal": dict(self.last_signal),
            "counters": self.metrics.snapshot()["counters"],
            "state_path": self.state_path,
        }
