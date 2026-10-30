import math

import pytest

from cortex9_market_lab.config import RiskConfig
from cortex9_market_lab.risk.manager import RiskManager


def rm(**kw):
    return RiskManager(RiskConfig(**kw), periods_per_year=8760.0)


def test_fixed_fractional_capped_by_symbol_limit():
    assert rm(fraction=0.3, max_symbol_exposure=0.5).target_fraction(None) == pytest.approx(0.3)
    assert rm(fraction=0.9, max_symbol_exposure=0.5).target_fraction(None) == pytest.approx(0.5)


def test_volatility_target_sizing():
    r = rm(sizing="volatility_target", vol_target_annual=0.2, max_symbol_exposure=1.0)
    ann = 0.01 * math.sqrt(8760.0)
    assert r.target_fraction(0.01) == pytest.approx(0.2 / ann)
    # very low vol -> capped by max_symbol_exposure
    assert r.target_fraction(1e-6) == pytest.approx(1.0)
    assert r.target_fraction(None) == 0.0
    assert r.target_fraction(float("nan")) == 0.0


def test_total_exposure_cap():
    r = rm(max_symbol_exposure=0.5, max_total_exposure=0.8)
    assert r.cap_for_total(0.5, 0.0) == pytest.approx(0.5)
    assert r.cap_for_total(0.5, 0.5) == pytest.approx(0.3)
    assert r.cap_for_total(0.5, 0.9) == 0.0


def test_stop_levels_long_and_short():
    r = rm(stop_loss_pct=0.05, take_profit_pct=0.10)
    assert r.stop_levels(1, 100.0) == pytest.approx((95.0, 110.0))
    assert r.stop_levels(-1, 100.0) == pytest.approx((105.0, 90.0))
    assert rm(stop_loss_pct=None).stop_levels(1, 100.0)[0] is None


@pytest.mark.parametrize(
    ("side", "o", "h", "lo", "expected"),
    [
        (1, 100, 101, 99, None),
        (1, 100, 101, 94, ("stop_loss", 95.0)),
        (1, 93, 94, 90, ("stop_loss", 93.0)),  # gap below stop fills at open
        (1, 100, 111, 99, ("take_profit", 110.0)),
        (1, 100, 111, 94, ("stop_loss", 95.0)),  # both touched -> stop first
        (-1, 100, 106, 99, ("stop_loss", 105.0)),
        (-1, 100, 101, 89, ("take_profit", 90.0)),
        (-1, 107, 108, 106, ("stop_loss", 107.0)),
    ],
)
def test_check_exit(side, o, h, lo, expected):
    r = rm(stop_loss_pct=0.05, take_profit_pct=0.10)
    stop, take = r.stop_levels(side, 100.0)
    hit = r.check_exit(side, stop, take, o, h, lo)
    if expected is None:
        assert hit is None
    else:
        assert (hit.reason, hit.price) == (expected[0], pytest.approx(expected[1]))


def test_kill_switch_trips_once_and_blocks_entries():
    r = rm(max_drawdown_kill=0.2)
    assert not r.update_equity(100.0, "t0")
    assert not r.update_equity(120.0, "t1")
    assert not r.update_equity(97.0, "t2")  # dd = 19.2%
    assert r.update_equity(96.0, "t3")  # dd = 20%
    assert r.halted and r.halted_at == "t3"
    assert not r.update_equity(50.0, "t4")  # already halted
    assert r.desired_side("X", 1) == 0


def test_desired_side_short_and_stop_memory():
    r = rm(allow_short=False)
    assert r.desired_side("X", -1) == 0
    assert rm(allow_short=True).desired_side("X", -1) == -1
    r.mark_stopped("X", 1)
    assert r.desired_side("X", 1) == 0  # blocked until the signal changes
    assert r.desired_side("X", 0) == 0  # signal changed -> memory cleared
    assert r.desired_side("X", 1) == 1


def test_risk_state_round_trip():
    r = rm()
    r.update_equity(100.0, "t0")
    r.mark_stopped("X", 1)
    q = rm()
    q.load_state(r.state())
    assert q.state() == r.state()
