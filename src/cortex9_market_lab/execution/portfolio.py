"""Simulated portfolio: cash, signed positions, fills and closed trades (no real orders)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cortex9_market_lab.execution.costs import CostModel


@dataclass
class Position:
    symbol: str
    qty: float  # signed: > 0 long, < 0 short
    entry_price: float  # fill price incl. slippage
    entry_time: str
    entry_fee: float
    stop_price: float | None = None
    take_price: float | None = None

    @property
    def side(self) -> int:
        return 1 if self.qty > 0 else -1


@dataclass
class Trade:
    symbol: str
    side: int
    qty: float
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    fees: float
    pnl: float  # net of both fees
    exit_reason: str

    @property
    def return_pct(self) -> float:
        notional = abs(self.qty) * self.entry_price
        return self.pnl / notional if notional else 0.0


@dataclass
class Portfolio:
    cash: float
    costs: CostModel = field(default_factory=CostModel)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    fees_paid: float = 0.0
    fills: int = 0

    # -- valuation ---------------------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices[s] for s, p in self.positions.items())

    def gross_exposure(self, prices: dict[str, float]) -> float:
        """Sum of |position notional| divided by equity (0 if equity <= 0)."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(abs(p.qty) * prices[s] for s, p in self.positions.items()) / eq

    def side(self, symbol: str) -> int:
        p = self.positions.get(symbol)
        return 0 if p is None else p.side

    # -- execution -----------------------------------------------------------------
    def open(
        self,
        symbol: str,
        side: int,
        notional: float,
        ref_price: float,
        ts: str,
        stop_price: float | None = None,
        take_price: float | None = None,
    ) -> Position | None:
        if symbol in self.positions:
            raise ValueError(f"position already open for {symbol}")
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        if notional <= 0:
            return None
        fill = self.costs.fill_price(ref_price, buy=side > 0)
        qty = side * notional / fill
        fee = self.costs.fee(qty, fill)
        self.cash -= qty * fill + fee
        self.fees_paid += fee
        self.fills += 1
        pos = Position(symbol, qty, fill, ts, fee, stop_price, take_price)
        self.positions[symbol] = pos
        return pos

    def close(self, symbol: str, ref_price: float, ts: str, reason: str) -> Trade:
        pos = self.positions.pop(symbol)
        fill = self.costs.fill_price(ref_price, buy=pos.qty < 0)
        fee = self.costs.fee(pos.qty, fill)
        self.cash += pos.qty * fill - fee
        self.fees_paid += fee
        self.fills += 1
        pnl = pos.qty * (fill - pos.entry_price) - pos.entry_fee - fee
        trade = Trade(
            symbol,
            pos.side,
            abs(pos.qty),
            pos.entry_time,
            pos.entry_price,
            ts,
            fill,
            pos.entry_fee + fee,
            pnl,
            reason,
        )
        self.trades.append(trade)
        return trade

    # -- serialization -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "costs": asdict(self.costs),
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
            "fees_paid": self.fees_paid,
            "fills": self.fills,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Portfolio:
        return cls(
            cash=float(d["cash"]),
            costs=CostModel(**d["costs"]),
            positions={s: Position(**p) for s, p in d["positions"].items()},
            trades=[Trade(**t) for t in d["trades"]],
            fees_paid=float(d["fees_paid"]),
            fills=int(d["fills"]),
        )
