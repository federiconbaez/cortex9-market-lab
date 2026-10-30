from pathlib import Path

import pytest
from pydantic import ValidationError

from cortex9_market_lab.config import AppConfig, load_config, periods_per_year, timeframe_seconds

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")))
def test_shipped_configs_validate(path):
    cfg = load_config(path)
    assert cfg.strategies


def _base():
    return {
        "data": {"source": "synthetic", "symbols": ["A/B"]},
        "strategies": [{"name": "momentum"}],
    }


def test_minimal_config_defaults():
    cfg = AppConfig.model_validate(_base())
    assert cfg.costs.fee_bps == 10 and cfg.risk.sizing == "fixed_fractional"


@pytest.mark.parametrize(
    "patch",
    [
        {"data": {"source": "synthetic", "symbols": ["A/B"], "timeframe": "7m"}},
        {"data": {"source": "synthetic", "symbols": ["AB"]}},
        {"data": {"source": "synthetic", "symbols": []}},
        {"data": {"source": "synthetic", "symbols": ["A/B", "A/B"]}},
        {"data": {"source": "exchange", "symbols": ["A/B"]}},  # missing start
        {
            "data": {
                "source": "exchange",
                "symbols": ["A/B"],
                "start": "2024-02-01",
                "end": "2024-01-01",
            }
        },
        {"strategies": []},
        {"risk": {"max_symbol_exposure": 0.9, "max_total_exposure": 0.5}},
        {"risk": {"fraction": 1.5}},
        {"costs": {"fee_bps": -1}},
        {"unknown_section": {}},
    ],
)
def test_invalid_configs_rejected(patch):
    raw = _base()
    raw.update(patch)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_naive_dates_become_utc():
    raw = _base()
    raw["data"] = {"source": "exchange", "symbols": ["A/B"], "start": "2024-01-01T00:00:00"}
    cfg = AppConfig.model_validate(raw)
    assert cfg.data.start.utcoffset().total_seconds() == 0


def test_timeframe_helpers():
    assert timeframe_seconds("1h") == 3600
    assert periods_per_year("1h") == pytest.approx(8760.0)
    with pytest.raises(ValueError):
        timeframe_seconds("2w")


def test_load_config_rejects_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_cli_cost_overrides_and_synthetic_backtest(tmp_path, capsys):
    from cortex9_market_lab.cli import main

    cfg = ROOT / "configs" / "synthetic.yaml"
    rc = main(
        [
            "backtest",
            "--config",
            str(cfg),
            "--strategy",
            "momentum",
            "--fee-bps",
            "35",
            "--slippage-bps",
            "0",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["costs"] == {"fee_bps": 35.0, "slippage_bps": 0.0}
    assert list(summary["results"]) == ["momentum"]


def test_configure_logging_is_idempotent_and_follows_stderr(tmp_path, capsys):
    import logging

    from cortex9_market_lab.observability.logging import configure_logging, get_logger

    log_file = tmp_path / "x.jsonl"
    configure_logging("INFO", str(log_file))
    configure_logging("INFO", str(log_file))
    root = logging.getLogger("cortex9_market_lab")
    assert len(root.handlers) == 2  # one stream + one file, not duplicated
    get_logger("t").info("hello", extra={"k": 1})
    err = capsys.readouterr().err
    assert '"event": "hello"' in err and '"k": 1' in err
    assert log_file.read_text().count('"hello"') == 1
