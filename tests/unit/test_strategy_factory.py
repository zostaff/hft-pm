"""Unit tests for hft_pm.strategies.factory."""

from __future__ import annotations

import logging

import pytest

from hft_pm.strategies.avellaneda_stoikov import (
    AvellanedaStoikov,
    AvellanedaStoikovWithSignals,
)
from hft_pm.strategies.constant_spread import ConstantSpread
from hft_pm.strategies.factory import build_strategy, merge_calibrated_params
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


# ---------------------------------------------------------------------------
# merge_calibrated_params — calibration → strategy params with safety guards
# ---------------------------------------------------------------------------


_DEFAULT_AS_PARAMS = {"gamma": 1.0, "sigma": 0.005, "kappa": 60.0, "horizon_ms": 600_000, "size": 1}


def test_merge_applies_calibrated_values_when_valid() -> None:
    cal = {"sigma_per_sqrts": 0.01, "kappa": 100.0, "alpha_beta": 5e-7}
    out = merge_calibrated_params("avellaneda_stoikov", _DEFAULT_AS_PARAMS, cal)
    assert out["sigma"] == 0.01
    assert out["kappa"] == 100.0
    # alpha_beta is allowed only for the signals variant; AS plain drops it.
    assert "alpha_beta" not in out


def test_merge_filters_by_strategy_kind() -> None:
    cal = {"sigma_per_sqrts": 0.01, "kappa": 100.0, "A_per_side": 0.5, "alpha_beta": 5e-7}
    out_as = merge_calibrated_params("avellaneda_stoikov", _DEFAULT_AS_PARAMS, cal)
    out_glt = merge_calibrated_params(
        "glt", {"gamma": 1.0, "sigma": 0.005, "kappa": 60.0, "A": 1.0, "size": 1}, cal
    )
    # AS doesn't take A; GLT does.
    assert "A" not in out_as
    assert out_glt["A"] == 0.5


def test_merge_rejects_non_positive_sigma_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: 2026-05-17 Thunder calibration produced sigma_per_sqrts=0
    (the mid never moved during the capture). The pre-fix merge passed it
    through unchanged and AvellanedaStoikov raised ValueError("sigma must
    be positive") inside the backtest. The fix keeps the YAML default and
    logs a warning."""
    cal = {"sigma_per_sqrts": 0.0, "kappa": 200.0}
    caplog.set_level(logging.WARNING, logger="hft_pm.strategies.factory")
    out = merge_calibrated_params("avellaneda_stoikov", _DEFAULT_AS_PARAMS, cal)
    # YAML default preserved, not the bad calibrated value.
    assert out["sigma"] == 0.005
    # Good values still apply.
    assert out["kappa"] == 200.0
    assert any("sigma_per_sqrts" in r.message for r in caplog.records), (
        f"expected sigma warning, got records: {[r.message for r in caplog.records]}"
    )


def test_merge_rejects_nan_and_inf_for_positive_params() -> None:
    cal = {"sigma_per_sqrts": float("nan"), "kappa": float("inf")}
    out = merge_calibrated_params("avellaneda_stoikov", _DEFAULT_AS_PARAMS, cal)
    assert out["sigma"] == 0.005  # default kept
    assert out["kappa"] == 60.0  # default kept


def test_merge_accepts_zero_and_negative_alpha_beta() -> None:
    """alpha_beta=0 means "OFI signal is noise"; negative means "OFI predicts
    the opposite direction". Both are semantically valid — the merge must
    NOT filter them out the way it does for sigma/kappa."""
    sig_params = dict(_DEFAULT_AS_PARAMS, alpha_beta=5e-7, ofi_window_s=2.0)
    cal_zero = {"alpha_beta": 0.0}
    out_zero = merge_calibrated_params("avellaneda_stoikov_with_signals", sig_params, cal_zero)
    assert out_zero["alpha_beta"] == 0.0

    cal_neg = {"alpha_beta": -3e-8}
    out_neg = merge_calibrated_params("avellaneda_stoikov_with_signals", sig_params, cal_neg)
    assert out_neg["alpha_beta"] == -3e-8


def test_merge_does_not_mutate_inputs() -> None:
    cal = {"sigma_per_sqrts": 0.01}
    orig_params = dict(_DEFAULT_AS_PARAMS)
    out = merge_calibrated_params("avellaneda_stoikov", orig_params, cal)
    assert orig_params == _DEFAULT_AS_PARAMS, "input params dict was mutated"
    assert out is not orig_params
