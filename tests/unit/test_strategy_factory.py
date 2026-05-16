"""Unit tests for hft_pm.strategies.factory."""

from __future__ import annotations

import pytest

from hft_pm.strategies.avellaneda_stoikov import (
    AvellanedaStoikov,
    AvellanedaStoikovWithSignals,
)
from hft_pm.strategies.constant_spread import ConstantSpread
from hft_pm.strategies.factory import build_strategy
from hft_pm.strategies.glt import GLT


def test_builds_constant_spread() -> None:
    s = build_strategy("constant_spread", {"half_spread": 0.02, "size": 1})
    assert isinstance(s, ConstantSpread)
    assert s.half_spread == 0.02


def test_builds_avellaneda_stoikov() -> None:
    s = build_strategy(
        "avellaneda_stoikov",
        {"gamma": 1.0, "sigma": 0.01, "kappa": 50, "horizon_ms": 10_000, "size": 1},
    )
    assert isinstance(s, AvellanedaStoikov)


def test_builds_as_with_signals_with_ofi_and_vpin() -> None:
    s = build_strategy(
        "avellaneda_stoikov_with_signals",
        {
            "gamma": 1.0,
            "sigma": 0.01,
            "kappa": 50,
            "horizon_ms": 10_000,
            "size": 1,
            "use_microprice": True,
            "ofi_window_s": 2.0,
            "alpha_beta": 5e-7,
            "vpin_bucket_volume": 100.0,
            "vpin_n_buckets": 20,
            "vpin_max": 3.0,
        },
    )
    assert isinstance(s, AvellanedaStoikovWithSignals)
    assert s.ofi_calc is not None
    assert s.vpin_calc is not None


def test_builds_as_with_signals_without_optional_components() -> None:
    s = build_strategy(
        "avellaneda_stoikov_with_signals",
        {"gamma": 1.0, "sigma": 0.01, "kappa": 50, "horizon_ms": 10_000, "size": 1},
    )
    assert s.ofi_calc is None
    assert s.vpin_calc is None


def test_builds_glt() -> None:
    s = build_strategy("glt", {"gamma": 1.0, "sigma": 0.01, "kappa": 50, "A": 2.0, "size": 1})
    assert isinstance(s, GLT)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy kind"):
        build_strategy("mystery_strategy", {})


def test_factory_passes_through_typeerror_on_bad_params() -> None:
    """Misspelled param surfaces as TypeError from the strategy constructor."""
    with pytest.raises(TypeError):
        build_strategy("constant_spread", {"bad_arg": 0.02})
