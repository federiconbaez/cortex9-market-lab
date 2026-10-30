"""Command line interface: fetch, backtest, paper, strategies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from cortex9_market_lab import strategies
from cortex9_market_lab.backtest.engine import run_backtest
from cortex9_market_lab.config import AppConfig, load_config
from cortex9_market_lab.data.loader import load_history, make_client
from cortex9_market_lab.observability.logging import configure_logging
from cortex9_market_lab.paper.engine import PaperTrader

METRIC_ROWS = [
    ("total_return", "pct"),
    ("cagr", "pct"),
    ("sharpe", "num"),
    ("sortino", "num"),
    ("max_drawdown", "pct"),
    ("win_rate", "pct"),
    ("profit_factor", "num"),
    ("exposure", "pct"),
    ("n_trades", "int"),
    ("fees_paid", "num"),
    ("end_equity", "num"),
]


def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.3f}"


def _table(results: dict[str, dict[str, float]]) -> str:
    names = list(results)
    head = "| metric | " + " | ".join(names) + " |"
    sep = "|---|" + "---:|" * len(names)
    lines = [head, sep]
    for key, kind in METRIC_ROWS:
        lines.append(f"| {key} | " + " | ".join(_fmt(results[n][key], kind) for n in names) + " |")
    return "\n".join(lines)


def _load(path: str) -> AppConfig:
    cfg = load_config(path)
    configure_logging(cfg.logging.level, cfg.logging.file)
    return cfg


def cmd_strategies(_: argparse.Namespace) -> int:
    for name in strategies.available():
        s = strategies.create(name)
        print(f"{name}: {type(s).__name__} default params={s.params}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    data, source = load_history(cfg.data, seed=cfg.backtest.seed)
    for sym, df in data.items():
        print(
            f"{source} {sym} {cfg.data.timeframe}: {len(df)} bars "
            f"{df.index[0].isoformat()} -> {df.index[-1].isoformat()}"
        )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if args.fee_bps is not None:
        cfg.costs.fee_bps = args.fee_bps
    if args.slippage_bps is not None:
        cfg.costs.slippage_bps = args.slippage_bps
    data, source = load_history(cfg.data, seed=cfg.backtest.seed)
    selected = [s for s in cfg.strategies if not args.strategy or s.name in args.strategy]
    if not selected:
        print(f"no strategy matches {args.strategy}", file=sys.stderr)
        return 2
    out_root = Path(args.out or cfg.backtest.output_dir)
    any_df = next(iter(data.values()))
    header = {
        "data_source": source,
        "symbols": cfg.data.symbols,
        "timeframe": cfg.data.timeframe,
        "first_bar": any_df.index[0].isoformat(),
        "last_bar": any_df.index[-1].isoformat(),
        "bars_per_symbol": {s: len(df) for s, df in data.items()},
        "costs": cfg.costs.model_dump(),
        "risk": cfg.risk.model_dump(),
        "initial_cash": cfg.backtest.initial_cash,
    }
    results: dict[str, dict[str, float]] = {}
    bench: dict[str, float] = {}
    for sc in selected:
        strat = strategies.create(sc.name, sc.params)
        res = run_backtest(
            data,
            strat,
            timeframe=cfg.data.timeframe,
            costs=cfg.costs,
            risk_cfg=cfg.risk,
            initial_cash=cfg.backtest.initial_cash,
        )
        res.save(out_root / sc.name)
        results[sc.name] = res.metrics
        bench = res.benchmark
    summary: dict[str, Any] = {**header, "results": results, "benchmark_buy_and_hold": bench}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(header, indent=2, default=str))
    print()
    print(_table(results))
    print()
    print(
        "buy & hold (no costs, reference only): "
        + ", ".join(f"{k}={v * 100:.2f}%" for k, v in bench.items())
    )
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    name = args.strategy or cfg.strategies[0].name
    sc = next((s for s in cfg.strategies if s.name == name), None)
    strat = strategies.create(name, sc.params if sc else None)
    client = make_client(cfg.data)
    trader = PaperTrader(cfg, strat, client, state_path=args.state)
    summary = trader.run(args.duration, args.max_iterations)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["stop_reason"] != "too_many_errors" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="market-lab",
        description="Crypto research, backtest and paper trading (simulation only, no orders).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("strategies", help="list registered strategies")
    s.set_defaults(func=cmd_strategies)

    f = sub.add_parser("fetch", help="download historical OHLCV into the local cache")
    f.add_argument("--config", required=True)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="run backtests for the configured strategies")
    b.add_argument("--config", required=True)
    b.add_argument("--strategy", action="append", help="restrict to this strategy (repeatable)")
    b.add_argument("--out", help="output directory (default: backtest.output_dir)")
    b.add_argument("--fee-bps", type=float, help="override costs.fee_bps (sensitivity runs)")
    b.add_argument("--slippage-bps", type=float, help="override costs.slippage_bps")
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="paper trade on live public data (no real orders)")
    pp.add_argument("--config", required=True)
    pp.add_argument("--strategy", help="strategy name (default: first in config)")
    pp.add_argument("--duration", type=float, default=120.0, help="run length in seconds")
    pp.add_argument("--max-iterations", type=int, default=None)
    pp.add_argument("--state", help="state file path (default: paper.state_path)")
    pp.set_defaults(func=cmd_paper)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
