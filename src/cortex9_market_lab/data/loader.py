"""High-level loading: cache-aware historical fetch for a DataConfig."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cortex9_market_lab.config import DataConfig, timeframe_seconds
from cortex9_market_lab.data.cache import OhlcvCache
from cortex9_market_lab.data.exchange import MarketDataClient
from cortex9_market_lab.data.retry import BackoffPolicy
from cortex9_market_lab.data.synthetic import synthetic_universe
from cortex9_market_lab.observability.logging import get_logger

log = get_logger("data.loader")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def make_client(cfg: DataConfig, **kwargs: object) -> MarketDataClient:
    r = cfg.retry
    policy = BackoffPolicy(r.max_attempts, r.base_delay_s, r.max_delay_s, r.jitter)
    return MarketDataClient([cfg.exchange, *cfg.fallback_exchanges], policy=policy, **kwargs)  # type: ignore[arg-type]


def load_history(
    cfg: DataConfig,
    *,
    client: MarketDataClient | None = None,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Return ({symbol: ohlcv}, source_label). Uses the SQLite cache when it covers the range."""
    if cfg.source == "synthetic":
        data = synthetic_universe(
            cfg.symbols,
            cfg.synthetic_bars,
            seed=seed,
            timeframe=cfg.timeframe,
            model=cfg.synthetic_model,
        )
        return data, f"synthetic:{cfg.synthetic_model}:seed={seed}"

    assert cfg.start is not None
    end = cfg.end or datetime.now(tz=timezone.utc)
    start_ms, end_ms = _ms(cfg.start), _ms(end)
    step = timeframe_seconds(cfg.timeframe) * 1000
    # only closed bars: drop anything whose close time is in the future
    end_ms = min(end_ms, (_ms(datetime.now(tz=timezone.utc)) // step) * step)

    client = client or make_client(cfg)
    exchange_id = client.active_exchange or client.connect()
    cache = OhlcvCache(cfg.cache_path)
    out: dict[str, pd.DataFrame] = {}
    try:
        for sym in cfg.symbols:
            b = cache.bounds(exchange_id, sym, cfg.timeframe)
            fetch_from = start_ms
            if b is not None and b[0] <= start_ms:
                fetch_from = max(start_ms, b[1] + step)
            if fetch_from < end_ms:
                fresh = client.fetch_ohlcv_range(
                    sym, cfg.timeframe, fetch_from, end_ms, page_limit=cfg.page_limit
                )
                n = cache.upsert(exchange_id, sym, cfg.timeframe, fresh)
                log.info("fetched", extra={"symbol": sym, "bars": n, "exchange": exchange_id})
            df = cache.load(exchange_id, sym, cfg.timeframe, start_ms, end_ms)
            if df.empty:
                raise RuntimeError(f"no data for {sym} {cfg.timeframe} on {exchange_id}")
            out[sym] = df
    finally:
        cache.close()
    return out, f"{exchange_id}:public-ohlcv"
