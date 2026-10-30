"""Fee and slippage model shared by the backtest and the paper engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Proportional costs in basis points.

    * slippage moves the fill against the trader: buy at ref*(1+s), sell at ref*(1-s);
    * the fee is charged on the filled notional: |qty| * fill * fee.
    """

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0

    @property
    def slip_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def fill_price(self, ref_price: float, buy: bool) -> float:
        if ref_price <= 0:
            raise ValueError("reference price must be positive")
        return ref_price * (1.0 + self.slip_rate) if buy else ref_price * (1.0 - self.slip_rate)

    def fee(self, qty: float, fill_price: float) -> float:
        return abs(qty) * fill_price * self.fee_rate
