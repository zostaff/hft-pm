"""YAML-driven configuration for backtest / paper / live runners.

A single config bundles three concerns: which market to trade, what
strategy and parameters to use, and what risk limits to enforce. Each
section is a Pydantic model so YAML errors fail fast at load time
rather than silently mis-typing values.

Example::

    market:
      token_id: "13915..."
      condition_id: "0xa0f4..."
      fee_category: SPORTS
      tick: 0.01
    strategy:
      kind: avellaneda_stoikov_with_signals
      params:
        gamma: 1.0
        sigma: 0.005
        kappa: 60.0
        horizon_ms: 600000
        size: 1
        use_microprice: true
        alpha_beta: 5.0e-7
        vpin_max: 3.0
    risk:
      max_drawdown_pct: 0.20
      max_inventory: 50
      heartbeat_timeout_s: 30
      daily_loss_limit: 25
      baseline_capital: 100
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .fees.polymarket import FeeCategory
from .risk.limits import RiskLimits

StrategyKind = Literal[
    "constant_spread",
    "avellaneda_stoikov",
    "avellaneda_stoikov_with_signals",
    "glt",
]


class MarketConfig(BaseModel):
    """Which market to subscribe to and trade."""

    model_config = ConfigDict(extra="forbid")

    token_id: str
    condition_id: str | None = None
    fee_category: FeeCategory = FeeCategory.OTHER
    tick: float = Field(gt=0)

    @field_validator("fee_category", mode="before")
    @classmethod
    def _coerce_fee_category(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return FeeCategory[v.upper()]
            except KeyError as e:
                valid = [c.name for c in FeeCategory]
                raise ValueError(f"unknown fee_category {v!r}; valid: {valid}") from e
        return v


class StrategyConfig(BaseModel):
    """Strategy kind + raw parameter dict (validated by the strategy itself)."""

    model_config = ConfigDict(extra="forbid")

    kind: StrategyKind
    params: dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Mirror of :class:`RiskLimits` for YAML loading."""

    model_config = ConfigDict(extra="forbid")

    max_drawdown_pct: float = 0.20
    max_inventory: float = 100.0
    heartbeat_timeout_s: float = 30.0
    daily_loss_limit: float | None = None
    baseline_capital: float = 100.0

    def to_limits(self) -> RiskLimits:
        return RiskLimits(
            max_drawdown_pct=self.max_drawdown_pct,
            max_inventory=self.max_inventory,
            heartbeat_timeout_s=self.heartbeat_timeout_s,
            daily_loss_limit=self.daily_loss_limit,
            baseline_capital=self.baseline_capital,
        )


class AppConfig(BaseModel):
    """Top-level config combining the three sections."""

    model_config = ConfigDict(extra="forbid")

    market: MarketConfig
    strategy: StrategyConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config. Raises on schema / type errors."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config {p} must contain a YAML mapping at the top level")
    return AppConfig.model_validate(raw)


__all__ = [
    "AppConfig",
    "MarketConfig",
    "RiskConfig",
    "StrategyConfig",
    "StrategyKind",
    "load_config",
]
