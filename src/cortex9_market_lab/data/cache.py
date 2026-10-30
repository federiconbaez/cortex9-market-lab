"""SQLite cache for OHLCV bars keyed by (exchange, symbol, timeframe, timestamp)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from cortex9_market_lab.data.exchange import ohlcv_to_frame

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL, volume REAL NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, ts)
);
"""


class OhlcvCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        ts_ms = (df.index.view("int64") // 1_000_000).tolist()
        rows = [
            (exchange, symbol, timeframe, int(t), float(o), float(h), float(lo), float(c), float(v))
            for t, o, h, lo, c, v in zip(
                ts_ms, df["open"], df["high"], df["low"], df["close"], df["volume"], strict=True
            )
        ]
        self._conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self._conn.commit()
        return len(rows)

    def load(
        self, exchange: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        cur = self._conn.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv "
            "WHERE exchange=? AND symbol=? AND timeframe=? AND ts>=? AND ts<? ORDER BY ts",
            (exchange, symbol, timeframe, start_ms, end_ms),
        )
        return ohlcv_to_frame([list(r) for r in cur.fetchall()])

    def bounds(self, exchange: str, symbol: str, timeframe: str) -> tuple[int, int] | None:
        row = self._conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe=?",
            (exchange, symbol, timeframe),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0]), int(row[1])
