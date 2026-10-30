import numpy as np
import pandas as pd
import pytest

from cortex9_market_lab import strategies
from cortex9_market_lab.data.synthetic import gbm, regime_switching
from cortex9_market_lab.strategies.base import Strategy, register

PARAMS = {
    "mean_reversion": {"lookback": 20, "entry_z": 1.5, "exit_z": 0.0},
    "momentum": {"fast": 5, "slow": 30},
    "breakout": {"entry": 20, "exit": 10, "atr_n": 14, "atr_mult": 0.5},
}


def _frame(close):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )


def test_registry_lists_builtins_and_rejects_unknown():
    assert {"mean_reversion", "momentum", "breakout"} <= set(strategies.available())
    with pytest.raises(KeyError):
        strategies.create("does_not_exist")
    with pytest.raises(ValueError):

        @register
        class Dup(Strategy):
            name = "momentum"

            @property
            def warmup(self):
                return 0

            def _signals(self, df):
                return np.zeros(len(df))


@pytest.mark.parametrize("name", list(PARAMS))
def test_signal_contract(name):
    df = regime_switching(800, seed=3)
    s = strategies.create(name, PARAMS[name])
    sig = s.generate(df)
    assert sig.index.equals(df.index)
    assert set(np.unique(sig)) <= {-1, 0, 1}
    assert (sig.iloc[: s.warmup] == 0).all()
    assert (sig != 0).any(), "strategy never trades on 800 bars of regime data"


@pytest.mark.parametrize("name", list(PARAMS))
def test_no_lookahead_truncation_invariance(name):
    """Signal at bar t computed on the full series equals the one computed on bars <= t."""
    df = regime_switching(600, seed=11)
    s = strategies.create(name, PARAMS[name])
    full = s.generate(df)
    for t in range(s.warmup, len(df), 37):
        assert s.generate(df.iloc[: t + 1]).iloc[-1] == full.iloc[t], f"lookahead at bar {t}"


@pytest.mark.parametrize("name", list(PARAMS))
def test_future_perturbation_does_not_change_past(name):
    df = gbm(400, seed=5)
    s = strategies.create(name, PARAMS[name])
    base = s.generate(df)
    shocked = df.copy()
    shocked.iloc[300:, :4] *= 3.0
    assert s.generate(shocked).iloc[:300].equals(base.iloc[:300])


def test_momentum_follows_trend():
    up = _frame(np.linspace(100, 200, 200))
    down = _frame(np.linspace(200, 100, 200))
    s = strategies.create("momentum", {"fast": 5, "slow": 30})
    assert s.generate(up).iloc[-1] == 1
    assert s.generate(down).iloc[-1] == -1


def test_mean_reversion_buys_the_dip_and_exits_on_recovery():
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 0.1, 100)
    close = np.concatenate([close, [95.0], np.full(30, 100.0)])
    s = strategies.create("mean_reversion", {"lookback": 20, "entry_z": 2.0, "exit_z": 0.0})
    sig = s.generate(_frame(close))
    assert sig.iloc[100] == 1
    assert sig.iloc[-1] == 0


def test_breakout_enters_on_new_high_and_exits_on_channel_low():
    close = np.concatenate([np.full(60, 100.0), [105.0, 106, 107], np.full(5, 107.0), [90.0]])
    s = strategies.create("breakout", {"entry": 20, "exit": 5, "atr_n": 5, "atr_mult": 0.0})
    sig = s.generate(_frame(close))
    assert sig.iloc[59] == 0
    assert sig.iloc[60] == 1
    assert sig.iloc[-1] == 0


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("mean_reversion", {"lookback": 1}),
        ("momentum", {"fast": 30, "slow": 10}),
        ("breakout", {"entry": 1}),
    ],
)
def test_invalid_params_rejected(name, bad):
    with pytest.raises(ValueError):
        strategies.create(name, bad)
