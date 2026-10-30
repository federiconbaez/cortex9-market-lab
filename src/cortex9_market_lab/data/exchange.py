"""Public-market-data client over ccxt with retries, reconnection and exchange fallback.

Read-only by construction:
  * the ccxt instance is built without credentials (no API key, secret or password is
    read from config, environment or disk);
  * every exchange call goes through ``_call``, which only accepts the method names in
    ``PUBLIC_METHODS``. There is no code path that places, edits or cancels an order or
    reads balances; tests assert this.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import ccxt
import pandas as pd

from cortex9_market_lab.config import timeframe_seconds
from cortex9_market_lab.data.retry import BackoffPolicy, RetryExhaustedError, call_with_retry
from cortex9_market_lab.observability.logging import Metrics, get_logger

log = get_logger("data.exchange")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Transient errors worth retrying. ccxt.NetworkError covers RequestTimeout,
# ExchangeNotAvailable, DDoSProtection and RateLimitExceeded.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (ccxt.NetworkError,)

ExchangeFactory = Callable[[str], Any]

# The only exchange methods this project is allowed to call (all public market data).
PUBLIC_METHODS: frozenset[str] = frozenset({"load_markets", "fetch_ohlcv", "fetch_ticker"})


class ForbiddenExchangeCallError(RuntimeError):
    """Raised if code tries to call an exchange method outside ``PUBLIC_METHODS``."""


def default_factory(exchange_id: str) -> Any:
    """Build a public (unauthenticated) ccxt exchange with built-in rate limiting."""
    klass = getattr(ccxt, exchange_id)
    client = klass({"enableRateLimit": True, "timeout": 20_000})
    # make sure no credential sneaks in through ccxt defaults
    for attr in ("apiKey", "secret", "password", "uid", "privateKey", "walletAddress"):
        if getattr(client, attr, None):
            raise ForbiddenExchangeCallError(f"unexpected credential attribute {attr!r} set")
    return client


class PublicOnlyExchange:
    """Wrapper around a ccxt instance that exposes nothing but ``PUBLIC_METHODS``.

    Every attribute lookup goes through ``__getattribute__``: whitelisted names are
    forwarded to the wrapped instance, anything else (``create_order``, ``createOrder``,
    ``fetch_balance``, ``apiKey``, the wrapped instance itself, ...) raises
    ``ForbiddenExchangeCallError``.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattribute__(self, name: str) -> Any:
        if name not in PUBLIC_METHODS:
            raise ForbiddenExchangeCallError(f"exchange attribute {name!r} is not allowed")
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise ForbiddenExchangeCallError("the exchange wrapper is read-only")


class MarketDataClient:
    """Fetch OHLCV/tickers from the first reachable exchange in a preference list.

    * every call is retried with exponential backoff on transient errors;
    * after a retry the underlying ccxt instance is rebuilt ("reconnect");
    * if the primary exchange cannot be reached at all, the next fallback is used.
    """

    def __init__(
        self,
        exchanges: list[str],
        *,
        policy: BackoffPolicy | None = None,
        factory: ExchangeFactory = default_factory,
        sleep: Callable[[float], None] = time.sleep,
        metrics: Metrics | None = None,
    ) -> None:
        if not exchanges:
            raise ValueError("at least one exchange id is required")
        self.exchanges = list(exchanges)
        self.policy = policy or BackoffPolicy()
        self._factory = factory
        self._sleep = sleep
        self.metrics = metrics or Metrics()
        self.__exchange: PublicOnlyExchange | None = None
        self.active_exchange: str | None = None

    # -- connection management -------------------------------------------------
    def _connect(self, exchange_id: str) -> PublicOnlyExchange:
        client = PublicOnlyExchange(self._factory(exchange_id))
        call_with_retry(
            lambda: _public(client, "load_markets")(),
            policy=self.policy,
            retry_on=TRANSIENT_ERRORS,
            sleep=self._sleep,
            op=f"{exchange_id}.load_markets",
        )
        return client

    def connect(self) -> str:
        """Connect to the first reachable exchange; returns its id."""
        errors: dict[str, str] = {}
        for ex_id in self.exchanges:
            try:
                self.__exchange = self._connect(ex_id)
                self.active_exchange = ex_id
                self.metrics.inc("exchange.connects")
                log.info("connected", extra={"exchange": ex_id})
                return ex_id
            except (RetryExhaustedError, ccxt.BaseError, AttributeError) as exc:
                errors[ex_id] = repr(exc)
                self.metrics.inc("exchange.connect_failures")
                log.warning("exchange unreachable", extra={"exchange": ex_id, "error": repr(exc)})
        raise ConnectionError(f"no exchange reachable: {errors}")

    def _exchange(self) -> PublicOnlyExchange:
        """The connected exchange wrapper (private: there is no public handle to it)."""
        if self.__exchange is None:
            self.connect()
        assert self.__exchange is not None
        return self.__exchange

    def _reconnect(self, _attempt: int, _exc: BaseException) -> None:
        if self.active_exchange is None:
            return
        self.metrics.inc("exchange.reconnects")
        try:
            self.__exchange = PublicOnlyExchange(self._factory(self.active_exchange))
        except Exception as exc:  # pragma: no cover - defensive
            log.error("reconnect failed", extra={"error": repr(exc)})

    def _call(self, op: str, method: str, *args: Any, **kwargs: Any) -> Any:
        self.metrics.inc("exchange.requests")
        return call_with_retry(
            lambda: _public(self._exchange(), method)(*args, **kwargs),
            policy=self.policy,
            retry_on=TRANSIENT_ERRORS,
            sleep=self._sleep,
            on_retry=self._reconnect,
            op=op,
        )

    # -- public data -----------------------------------------------------------
    def fetch_ohlcv_page(
        self, symbol: str, timeframe: str, since_ms: int | None, limit: int
    ) -> list[list[float]]:
        return self._call(
            f"fetch_ohlcv {symbol} {timeframe}",
            "fetch_ohlcv",
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit,
        )

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        *,
        page_limit: int = 100,
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        """Page forward from ``start_ms`` until ``end_ms`` (exclusive). Returns a clean frame."""
        step = timeframe_seconds(timeframe) * 1000
        since = start_ms
        rows: list[list[float]] = []
        for _ in range(max_pages):
            if since >= end_ms:
                break
            page = self.fetch_ohlcv_page(symbol, timeframe, since, page_limit)
            if not page:
                break
            rows.extend(r for r in page if start_ms <= r[0] < end_ms)
            last_ts = int(page[-1][0])
            if last_ts + step <= since:  # exchange did not advance; avoid infinite loop
                break
            since = last_ts + step
        return ohlcv_to_frame(rows)

    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        page = self.fetch_ohlcv_page(symbol, timeframe, None, limit)
        return ohlcv_to_frame(page)

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self._call(f"fetch_ticker {symbol}", "fetch_ticker", symbol)
        price = ticker.get("last") or ticker.get("close")
        if price is None:
            raise ValueError(f"ticker for {symbol} has no last price")
        return float(price)


def _public(client: Any, method: str) -> Callable[..., Any]:
    """Return ``client.<method>`` only if it is a whitelisted public market-data method.

    This and ``PublicOnlyExchange.__getattribute__`` are the only places where an exchange
    attribute is looked up by name; a test enforces it.
    """
    if method not in PUBLIC_METHODS:
        raise ForbiddenExchangeCallError(f"exchange method {method!r} is not allowed")
    return getattr(client, method)


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert ccxt OHLCV rows into a UTC-indexed, de-duplicated, sorted DataFrame."""
    if not rows:
        empty = pd.DataFrame(columns=OHLCV_COLUMNS[1:], dtype=float)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return empty
    df = pd.DataFrame([r[:6] for r in rows], columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = df.astype(float)
    return df[(df["high"] >= df["low"]) & (df["close"] > 0)]
