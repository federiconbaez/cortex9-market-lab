import pytest

from cortex9_market_lab.execution.costs import CostModel
from cortex9_market_lab.execution.portfolio import Portfolio


def test_fill_price_and_fee():
    c = CostModel(fee_bps=10, slippage_bps=5)
    assert c.fill_price(100.0, buy=True) == pytest.approx(100.05)
    assert c.fill_price(100.0, buy=False) == pytest.approx(99.95)
    assert c.fee(2.0, 100.0) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        c.fill_price(0.0, buy=True)


def test_long_round_trip_accounting():
    p = Portfolio(10_000.0, CostModel(10, 5))
    pos = p.open("X", 1, 1_000.0, 100.0, "t0")
    assert pos is not None
    assert pos.entry_price == pytest.approx(100.05)
    assert pos.qty == pytest.approx(1_000.0 / 100.05)
    assert p.cash == pytest.approx(10_000.0 - 1_000.0 - 1.0)  # fee = 10 bps of 1000
    tr = p.close("X", 110.0, "t1", "signal")
    exit_fill = 110.0 * (1 - 0.0005)
    exit_fee = pos.qty * exit_fill * 0.001
    expected_pnl = pos.qty * (exit_fill - 100.05) - 1.0 - exit_fee
    assert tr.exit_price == pytest.approx(exit_fill)
    assert tr.pnl == pytest.approx(expected_pnl)
    assert tr.fees == pytest.approx(1.0 + exit_fee)
    assert p.cash == pytest.approx(10_000.0 + expected_pnl)
    assert p.fees_paid == pytest.approx(tr.fees)
    assert p.fills == 2


def test_flat_round_trip_loses_exactly_costs():
    p = Portfolio(10_000.0, CostModel(10, 5))
    p.open("X", 1, 5_000.0, 50.0, "t0")
    tr = p.close("X", 50.0, "t1", "signal")
    assert tr.pnl < 0
    assert p.cash == pytest.approx(10_000.0 + tr.pnl)


def test_short_round_trip_profits_when_price_falls():
    p = Portfolio(10_000.0, CostModel(0, 0))
    p.open("X", -1, 1_000.0, 100.0, "t0")
    assert p.side("X") == -1
    assert p.equity({"X": 100.0}) == pytest.approx(10_000.0)
    assert p.equity({"X": 90.0}) == pytest.approx(10_100.0)
    tr = p.close("X", 90.0, "t1", "signal")
    assert tr.pnl == pytest.approx(100.0)


def test_gross_exposure_and_duplicate_open():
    p = Portfolio(10_000.0, CostModel(0, 0))
    p.open("X", 1, 3_000.0, 100.0, "t0")
    p.open("Y", -1, 2_000.0, 10.0, "t0")
    assert p.gross_exposure({"X": 100.0, "Y": 10.0}) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        p.open("X", 1, 1.0, 100.0, "t1")
    assert p.open("Z", 1, 0.0, 1.0, "t1") is None


def test_portfolio_serialization_round_trip():
    p = Portfolio(10_000.0, CostModel(7, 3))
    p.open("X", 1, 1_000.0, 100.0, "t0", stop_price=95.0, take_price=110.0)
    p.open("Y", 1, 500.0, 10.0, "t0")
    p.close("Y", 11.0, "t1", "signal")
    q = Portfolio.from_dict(p.to_dict())
    assert q.to_dict() == p.to_dict()
    assert q.positions["X"].stop_price == 95.0
