"""Configuration models loaded from external YAML files and validated with pydantic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
}


def timeframe_seconds(timeframe: str) -> int:
    """Return the length of a timeframe in seconds (raises ValueError if unknown)."""
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe {timeframe!r}") from exc


def periods_per_year(timeframe: str) -> float:
    """Number of bars per year for a 24/7 market (crypto)."""
    return 365.0 * 86400.0 / timeframe_seconds(timeframe)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetryConfig(_Strict):
    max_attempts: int = Field(5, ge=1, le=20)
    base_delay_s: float = Field(1.0, gt=0)
    max_delay_s: float = Field(30.0, gt=0)
    jitter: float = Field(0.25, ge=0, le=1)


class DataConfig(_Strict):
    source: Literal["exchange", "synthetic"] = "exchange"
    exchange: str = "okx"
    fallback_exchanges: list[str] = Field(default_factory=lambda: ["binance", "kraken"])
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    start: datetime | None = None
    end: datetime | None = None
    cache_path: str = "data/ohlcv.sqlite"
    page_limit: int = Field(100, ge=10, le=1000)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # synthetic-only options
    synthetic_bars: int = Field(2000, ge=50)
    synthetic_model: Literal["gbm", "regime"] = "regime"

    @field_validator("symbols")
    @classmethod
    def _symbols_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one symbol is required")
        for s in v:
            if "/" not in s:
                raise ValueError(f"symbol {s!r} must look like BASE/QUOTE")
        if len(set(v)) != len(v):
            raise ValueError("duplicate symbols")
        return v

    @field_validator("timeframe")
    @classmethod
    def _tf_known(cls, v: str) -> str:
        timeframe_seconds(v)
        return v

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def _range(self) -> DataConfig:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("data.start must be before data.end")
        if self.source == "exchange" and self.start is None:
            raise ValueError("data.start is required when source is 'exchange'")
        return self


class StrategyConfig(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class CostConfig(_Strict):
    fee_bps: float = Field(10.0, ge=0, le=500)
    slippage_bps: float = Field(5.0, ge=0, le=500)


class RiskConfig(_Strict):
    sizing: Literal["fixed_fractional", "volatility_target"] = "fixed_fractional"
    fraction: float = Field(0.25, gt=0, le=1)
    vol_target_annual: float = Field(0.20, gt=0, le=5)
    vol_lookback: int = Field(48, ge=5)
    stop_loss_pct: float | None = Field(0.05, gt=0, lt=1)
    take_profit_pct: float | None = Field(None, gt=0)
    max_symbol_exposure: float = Field(0.5, gt=0, le=1)
    max_total_exposure: float = Field(1.0, gt=0, le=1)
    max_drawdown_kill: float | None = Field(0.25, gt=0, lt=1)
    allow_short: bool = False

    @model_validator(mode="after")
    def _limits(self) -> RiskConfig:
        if self.max_symbol_exposure > self.max_total_exposure:
            raise ValueError("max_symbol_exposure cannot exceed max_total_exposure")
        return self


class BacktestConfig(_Strict):
    initial_cash: float = Field(10_000.0, gt=0)
    seed: int = 42
    output_dir: str = "runs/backtest"


class PaperConfig(_Strict):
    timeframe: str = "1m"
    poll_seconds: float = Field(10.0, ge=1)
    warmup_bars: int = Field(200, ge=10, le=1000)
    initial_cash: float = Field(10_000.0, gt=0)
    state_path: str = "state/paper_state.json"
    max_consecutive_errors: int = Field(10, ge=1)

    @field_validator("timeframe")
    @classmethod
    def _tf_known(cls, v: str) -> str:
        timeframe_seconds(v)
        return v


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = None


class AppConfig(_Strict):
    data: DataConfig
    strategies: list[StrategyConfig]
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("strategies")
    @classmethod
    def _strategies_nonempty(cls, v: list[StrategyConfig]) -> list[StrategyConfig]:
        if not v:
            raise ValueError("at least one strategy is required")
        return v


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AppConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return AppConfig.model_validate(raw)
