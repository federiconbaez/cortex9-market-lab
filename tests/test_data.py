from datetime import datetime, timezone

import ccxt
import pandas as pd
import pytest

from conftest import FakeExchange
from cortex9_market_lab.config import DataConfig
from cortex9_market_lab.data.cache import OhlcvCache
from cortex9_market_lab.data.exchange import MarketDataClient, ohlcv_to_frame
from cortex9_market_lab.data.feed import PollingFeed
from cortex9_market_lab.data.loader import load_history
from cortex9_market_lab.data.retry import BackoffPolicy, RetryExhaustedError, call_with_retry
from cortex9_market_lab.data.synthetic import gbm, regime_switching, synthetic_universe


def test_backoff_delays_exponential_and_capped():
    p = BackoffPolicy(max_attempts=6, base_delay_s=1.0, max_delay_s=5.0, jitter=0.0)
    assert [p.delay(a) for a in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]
    j = BackoffPolicy(base_delay_s=1.0, jitter=0.25, seed=1)
    assert all(0.75 <= j.delay(1) <= 1.25 for _ in range(50))


def test_retry_succeeds_after_transient_errors():
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ccxt.RequestTimeout("t")
        return "ok"

    out = call_with_retry(
        flaky,
        policy=BackoffPolicy(5, 1.0, 30.0, 0.0),
        retry_on=(ccxt.NetworkError,),
        sleep=sleeps.append,
    )
    assert out == "ok" and calls["n"] == 3 and sleeps == [1.0, 2.0]


def test_retry_exhaustion_and_non_transient_passthrough():
    sleeps: list[float] = []

    def always():
        raise ccxt.ExchangeNotAvailable("down")

    with pytest.raises(RetryExhaustedError) as ei:
        call_with_retry(
            always,
            policy=BackoffPolicy(3, 1.0, 30.0, 0.0),
            retry_on=(ccxt.NetworkError,),
            sleep=sleeps.append,
        )
    assert isinstance(ei.value.__cause__, ccxt.ExchangeNotAvailable)
    assert sleeps == [1.0, 2.0]

    def bad_symbol():
        raise ccxt.BadSymbol("nope")

    sleeps.clear()
    with pytest.raises(ccxt.BadSymbol):
        call_with_retry(
            bad_symbol, policy=BackoffPolicy(3), retry_on=(ccxt.NetworkError,), sleep=sleeps.append
        )
    assert sleeps == []


def test_client_falls_back_when_primary_unreachable(frames):
    primary = FakeExchange(frames, fail_load=99)
    backup = FakeExchange(frames)
    made = {"okx": primary, "binance": backup}
    client = MarketDataClient(
        ["okx", "binance"],
        policy=BackoffPolicy(2, 0.01, 0.01, 0.0),
        factory=lambda ex: made[ex],
        sleep=lambda s: None,
    )
    assert client.connect() == "binance"
    assert client.metrics.counter("exchange.connect_failures") == 1


def test_client_reconnects_and_paginates(frames):
    created: list[FakeExchange] = []

    def factory(_ex):
        fx = FakeExchange(frames, fail_fetch=1 if not created else 0, page_cap=50)
        created.append(fx)
        return fx

    client = MarketDataClient(
        ["okx"], policy=BackoffPolicy(3, 0.01, 0.01, 0.0), factory=factory, sleep=lambda s: None
    )
    client.connect()
    df = frames["AAA/USDT"]
    start = int(df.index[10].timestamp() * 1000)
    end = int(df.index[260].timestamp() * 1000)
    out = client.fetch_ohlcv_range("AAA/USDT", "1h", start, end, page_limit=100)
    assert len(out) == 250
    assert out.index[0] == df.index[10] and out.index[-1] == df.index[259]
    assert client.metrics.counter("exchange.reconnects") == 1
    assert len(created) == 2  # original + rebuilt after the 429
    pd.testing.assert_series_equal(out["close"], df["close"].iloc[10:260], check_freq=False)


def test_ohlcv_to_frame_cleans_rows():
    rows = [
        [2000, 1, 2, 0.5, 1.5, 10],
        [1000, 1, 2, 0.5, 1.5, 10],
        [1000, 1, 2, 0.5, 1.5, 10],
        [3000, 1, 0.1, 0.5, 1.5, 10],
    ]  # high < low -> dropped
    df = ohlcv_to_frame(rows)
    assert list(df.index.view("int64") // 1_000_000) == [1000, 2000]
    assert ohlcv_to_frame([]).empty


def test_cache_round_trip_and_bounds(tmp_path, frames):
    cache = OhlcvCache(tmp_path / "c.sqlite")
    df = frames["AAA/USDT"].iloc[:100]
    assert cache.upsert("okx", "AAA/USDT", "1h", df) == 100
    assert cache.upsert("okx", "AAA/USDT", "1h", df.iloc[:10]) == 10  # idempotent upsert
    b = cache.bounds("okx", "AAA/USDT", "1h")
    assert b == (int(df.index[0].timestamp() * 1000), int(df.index[-1].timestamp() * 1000))
    back = cache.load("okx", "AAA/USDT", "1h", b[0], b[1] + 1)
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    assert cache.bounds("okx", "ZZZ/USDT", "1h") is None
    cache.close()


def test_load_history_uses_cache_on_second_call(tmp_path, frames):
    fx = FakeExchange(frames)
    client = MarketDataClient(["okx"], factory=lambda _: fx, sleep=lambda s: None)
    idx = frames["AAA/USDT"].index
    cfg = DataConfig(
        source="exchange",
        symbols=["AAA/USDT", "BBB/USDT"],
        timeframe="1h",
        start=idx[0].to_pydatetime(),
        end=idx[200].to_pydatetime(),
        cache_path=str(tmp_path / "c.sqlite"),
    )
    data, src = load_history(cfg, client=client)
    assert src == "okx:public-ohlcv" and len(data["AAA/USDT"]) == 200
    n_fetch = sum(1 for c in fx.calls if c[0] == "fetch_ohlcv")
    data2, _ = load_history(cfg, client=client)
    assert sum(1 for c in fx.calls if c[0] == "fetch_ohlcv") == n_fetch
    pd.testing.assert_frame_equal(data["BBB/USDT"], data2["BBB/USDT"])


def test_synthetic_is_deterministic_and_valid():
    a = regime_switching(300, seed=4)
    pd.testing.assert_frame_equal(a, regime_switching(300, seed=4))
    assert not a.equals(regime_switching(300, seed=5))
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert (a["low"] > 0).all()
    u = synthetic_universe(["A/U", "B/U"], 50, seed=1, timeframe="1h", model="gbm")
    pd.testing.assert_frame_equal(u["A/U"], gbm(50, seed=1))


def test_feed_never_exposes_forming_bar(frames):
    fx = FakeExchange(frames)
    client = MarketDataClient(["okx"], factory=lambda _: fx, sleep=lambda s: None)
    client.connect()
    feed = PollingFeed(client, ["AAA/USDT"], "1h", window=50)
    idx = frames["AAA/USDT"].index
    # "now" is 30 minutes into the last bar -> last bar is still forming
    now = idx[-1] + pd.Timedelta(minutes=30)
    new = feed.poll_bars(now)
    assert new == {"AAA/USDT": True}
    assert feed.bars["AAA/USDT"].index[-1] == idx[-2]
    assert feed.poll_bars(now) == {"AAA/USDT": False}
    assert feed.poll_bars(idx[-1] + pd.Timedelta(hours=1)) == {"AAA/USDT": True}
    assert feed.last_prices()["AAA/USDT"] == pytest.approx(frames["AAA/USDT"]["close"].iloc[-1])


def test_load_history_synthetic_label():
    cfg = DataConfig(source="synthetic", symbols=["A/U"], synthetic_bars=60)
    data, src = load_history(cfg, seed=3)
    assert src == "synthetic:regime:seed=3" and len(data["A/U"]) == 60
    assert data["A/U"].index[0] == pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
