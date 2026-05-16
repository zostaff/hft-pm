"""Unit tests for hft_pm.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hft_pm.config import MarketConfig, load_config
from hft_pm.fees.polymarket import FeeCategory

_MIN_YAML = {
    "market": {"token_id": "1234", "tick": 0.01},
    "strategy": {"kind": "constant_spread", "params": {"half_spread": 0.02}},
}


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_load_minimal_config(tmp_path: Path) -> None:
    path = _write(tmp_path, _MIN_YAML)
    cfg = load_config(path)
    assert cfg.market.token_id == "1234"
    assert cfg.market.tick == 0.01
    # Default fee category is OTHER (zero fees).
    assert cfg.market.fee_category == FeeCategory.OTHER
    assert cfg.strategy.kind == "constant_spread"
    # Risk section is optional; defaults apply.
    assert cfg.risk.max_drawdown_pct == 0.20


def test_fee_category_string_resolves_to_enum(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["market"] = {**_MIN_YAML["market"], "fee_category": "sports"}
    cfg = load_config(_write(tmp_path, body))
    assert cfg.market.fee_category == FeeCategory.SPORTS


def test_unknown_fee_category_raises(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["market"] = {**_MIN_YAML["market"], "fee_category": "MYSTERY"}
    with pytest.raises(ValueError, match="unknown fee_category"):
        load_config(_write(tmp_path, body))


def test_unknown_strategy_kind_rejected(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["strategy"] = {"kind": "made_up_strategy", "params": {}}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, body))


def test_extra_top_level_keys_rejected(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["bogus"] = {"x": 1}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, body))


def test_negative_tick_rejected(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["market"] = {"token_id": "1234", "tick": -0.01}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, body))


def test_risk_section_to_limits_round_trip(tmp_path: Path) -> None:
    body = dict(_MIN_YAML)
    body["risk"] = {
        "max_drawdown_pct": 0.15,
        "max_inventory": 50.0,
        "heartbeat_timeout_s": 20.0,
        "daily_loss_limit": 30.0,
        "baseline_capital": 200.0,
    }
    cfg = load_config(_write(tmp_path, body))
    limits = cfg.risk.to_limits()
    assert limits.max_drawdown_pct == 0.15
    assert limits.max_inventory == 50.0
    assert limits.heartbeat_timeout_s == 20.0
    assert limits.daily_loss_limit == 30.0
    assert limits.baseline_capital == 200.0


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_market_config_direct_instantiation() -> None:
    """Sanity: MarketConfig can be built from kwargs (not only YAML)."""
    mc = MarketConfig(token_id="x", tick=0.01, fee_category=FeeCategory.FINANCE)
    assert mc.fee_category.cat_name == "finance"


def test_app_config_repr_contains_market_and_strategy(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _MIN_YAML))
    s = repr(cfg)
    assert "MarketConfig" in s
    assert "StrategyConfig" in s
