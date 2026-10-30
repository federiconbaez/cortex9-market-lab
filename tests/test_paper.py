import json

import pandas as pd
import pytest

from conftest import FakeExchange
from cortex9_market_lab import strategies
from cortex9_market_lab.config import AppConfig
from cortex9_market_lab.data.exchange import MarketDataClient
from cortex9_market_lab.data.retry import BackoffPolicy
from cortex9_market_lab.data.synthetic import gbm
from cortex9_market_lab.paper.engine import PaperTrader
from cortex9_market_lab.paper.state import fingerprint, load_state, save_state


class FakeClock:
    def __init__(self, t0: float) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _cfg(tmp_path, **paper):
    p = {
        "timeframe": "1m",
        "poll_seconds": 60,
        "warmup_bars": 60,
        "state_path": str(tmp_path / "state.json"),
    }
    p.update(paper)
    return AppConfig.model_validate(
        {
            "data": {"source": "synthetic", "symbols": ["AAA/USDT", "BBB/USDT"], "timeframe": "1m"},
            "strategies": [{"name": "momentum", "params": {"fast": 3, "slow": 10}}],
            "risk": {"stop_loss_pct": 0.02, "max_drawdown_kill": 0.5},
            "paper": p,
        }
    )


def _setup(tmp_path, **paper):
    frames = {
        s: gbm(400, seed=k, timeframe="1m", sigma_annual=1.5, start="2024-01-01")
        for k, s in enumerate(["AAA/USDT", "BBB/USDT"])
    }
    fx = FakeExchange(frames)
    client = MarketDataClient(
        ["okx"],
        policy=BackoffPolicy(2, 0.01, 0.01, 0.0),
        factory=lambda _: fx,
        sleep=lambda s: None,
    )
    cfg = _cfg(tmp_path, **paper)
    strat = strategies.create("momentum", {"fast": 3, "slow": 10})
    # the fake venue serves bars up to the fake clock, including the still-forming one,
    # so the feed's closed-bar filter is what keeps the run causal.
    clock = FakeClock(frames["AAA/USDT"].index[300].timestamp() + 1)
    fx.now_ms = lambda: clock() * 1000
    return cfg, strat, client, clock, fx


def test_paper_run_persists_and_resumes(tmp_path):
    cfg, strat, client, clock, _ = _setup(tmp_path)
    t = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    s1 = t.run(duration_s=10_000, max_iterations=5)
    assert s1["stop_reason"] == "max_iterations" and s1["iterations"] == 5
    st = load_state(cfg.paper.state_path)
    assert st["iterations"] == 5 and st["fingerprint"] == t.fp
    assert len(st["equity_history"]) == 5
    # every bar seen by the strategy was closed relative to the fake clock
    assert t.feed.bars["AAA/USDT"].index[-1] + pd.Timedelta(minutes=1) <= pd.Timestamp(
        clock(), unit="s", tz="UTC"
    )

    t2 = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    s2 = t2.run(duration_s=10_000, max_iterations=2)
    assert s2["resumed"] is True and s2["iterations"] == 7
    assert t2.portfolio.fills >= t.portfolio.fills


def test_paper_respects_duration(tmp_path):
    cfg, strat, client, clock, _ = _setup(tmp_path, poll_seconds=30)
    t = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    s = t.run(duration_s=95)
    assert s["stop_reason"] == "duration"
    assert s["iterations"] == 4  # t=0, 30, 60, 90


def test_paper_stops_after_consecutive_errors(tmp_path):
    cfg, strat, client, clock, fx = _setup(tmp_path, max_consecutive_errors=2)
    fx.fail_fetch = 10**6
    t = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    s = t.run(duration_s=10_000)
    assert s["stop_reason"] == "too_many_errors"
    assert s["counters"]["errors"] == 2
    assert load_state(cfg.paper.state_path) is not None  # state saved on failure path


def test_paper_graceful_stop_request(tmp_path):
    cfg, strat, client, clock, _ = _setup(tmp_path)
    t = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    t.request_stop()
    s = t.run(duration_s=10_000)
    assert s["stop_reason"] == "signal" and s["iterations"] == 0


def test_state_fingerprint_mismatch_rejected(tmp_path):
    cfg, strat, client, clock, _ = _setup(tmp_path)
    PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep).run(10_000, max_iterations=1)
    other = strategies.create("momentum", {"fast": 4, "slow": 10})
    with pytest.raises(ValueError):
        PaperTrader(cfg, other, client, clock=clock, sleep=clock.sleep).resume()


def test_save_state_is_atomic_and_versioned(tmp_path):
    p = tmp_path / "s.json"
    save_state(p, {"a": 1})
    assert json.loads(p.read_text())["version"] == 1
    assert load_state(p)["a"] == 1
    assert list(tmp_path.iterdir()) == [p]  # no temp files left behind
    p.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError):
        load_state(p)
    assert load_state(tmp_path / "missing.json") is None
    assert fingerprint({"x": 1, "y": 2}) == fingerprint({"y": 2, "x": 1})


def test_paper_actually_trades_with_shared_cost_model(tmp_path):
    cfg, strat, client, clock, _ = _setup(tmp_path)
    t = PaperTrader(cfg, strat, client, clock=clock, sleep=clock.sleep)
    s = t.run(duration_s=10**6, max_iterations=40)
    assert s["fills"] > 0 and s["closed_trades"] > 0
    for tr in t.portfolio.trades:
        assert tr.fees > 0  # 10 bps fee charged on both legs
    assert s["fees_paid"] == pytest.approx(
        sum(tr.fees for tr in t.portfolio.trades)
        + sum(p.entry_fee for p in t.portfolio.positions.values())
    )
