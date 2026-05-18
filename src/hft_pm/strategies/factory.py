"""Build a :class:`Strategy` from a :class:`StrategyConfig`.

Centralised so that CLI tools (backtest, paper trade, live) construct
strategies the same way and validate parameter names early.
"""

from __future__ import annotations

import logging
from typing import Any

from ..signals.ofi import OFICalculator
from ..signals.vpin import VPINCalculator
from .avellaneda_stoikov import AvellanedaStoikov, AvellanedaStoikovWithSignals
from .base import Strategy
from .constant_spread import ConstantSpread
from .glt import GLT

logger = logging.getLogger(__name__)

# Which strategy kinds accept which parameter names. Used by
# :func:`merge_calibrated_params` to drop calibrated values that wouldn't
# be consumed by the target strategy.
_PARAMS_BY_STRATEGY: dict[str, set[str]] = {
    "constant_spread": {"half_spread", "size", "tick"},
    "avellaneda_stoikov": {"gamma", "sigma", "kappa", "horizon_ms", "size"},
    "avellaneda_stoikov_with_signals": {
        "gamma",
        "sigma",
        "kappa",
        "horizon_ms",
        "size",
        "use_microprice",
        "alpha_beta",
        "ofi_window_s",
        "vpin_bucket_volume",
        "vpin_n_buckets",
        "vpin_max",
        "jump_schedule_ms",
        "pre_jump_withdraw_ms",
        "post_jump_resume_ms",
    },
    "glt": {"gamma", "sigma", "kappa", "A", "size"},
}

# Calibration JSON key -> target strategy parameter name.
_CALIBRATED_KEY_MAPPING: dict[str, str] = {
    "sigma_per_sqrts": "sigma",
    "kappa": "kappa",
    "A_per_side": "A",
    "alpha_beta": "alpha_beta",
}

# Parameters that strategies validate as strictly positive (sigma > 0,
# kappa > 0 etc.). A calibrated value of 0 or NaN passed through unchecked
# crashes the strategy constructor; instead we keep the YAML default and
# log a warning. ``alpha_beta`` is signed (negative means "OFI predicts
# the opposite direction") and ``0`` means "OFI is uninformative" — both
# are valid, so it is not in this set.
_STRICTLY_POSITIVE_PARAMS: frozenset[str] = frozenset({"sigma", "kappa", "A"})


def merge_calibrated_params(
    strategy_kind: str,
    strategy_params: dict[str, Any],
    calibrated: dict[str, Any],
) -> dict[str, Any]:
    """Merge calibrated values into the strategy params, with safety guards.

    Steps:

    1. For each ``(calibration_key, strategy_param)`` mapping, only apply
       the calibrated value if the strategy actually accepts that param
       (different strategy kinds use different subsets).
    2. For strictly-positive params (``sigma``, ``kappa``, ``A``), drop
       calibrated values that are ``<= 0`` or non-finite. This protects
       against thin-data captures where the estimator collapsed (e.g.
       a stuck mid yields ``sigma_per_sqrts = 0`` and would crash
       :class:`AvellanedaStoikov`). The YAML default is kept and a
       warning is logged so the user knows the calibration was rejected.
    3. ``alpha_beta`` passes through untouched: 0 and negative values are
       both semantically valid (no signal / opposite-sign predictor).
    """
    import math

    allowed = _PARAMS_BY_STRATEGY.get(strategy_kind, set())
    out = dict(strategy_params)
    for src, dst in _CALIBRATED_KEY_MAPPING.items():
        if src not in calibrated or dst not in allowed:
            continue
        value = calibrated[src]
        if dst in _STRICTLY_POSITIVE_PARAMS:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "merge_calibrated_params: %s=%r is non-numeric; keeping default",
                    src,
                    value,
                )
                continue
            if not math.isfinite(fv) or fv <= 0:
                logger.warning(
                    "merge_calibrated_params: %s=%s is non-positive / non-finite — "
                    "calibration likely ran on too-thin a capture; keeping default %s=%s",
                    src,
                    fv,
                    dst,
                    out.get(dst),
                )
                continue
        out[dst] = value
    return out


def build_strategy(kind: str, params: dict[str, Any]) -> Strategy:
    """Instantiate a strategy by kind name from a parameter dict.

    Recognised kinds:
      * ``constant_spread`` — params: ``half_spread``, ``size``
      * ``avellaneda_stoikov`` — params: ``gamma``, ``sigma``, ``kappa``, ``horizon_ms``, ``size``
      * ``avellaneda_stoikov_with_signals`` — params above plus optional
        ``use_microprice`` (bool), ``ofi_window_s`` (float, enables OFI),
        ``alpha_beta`` (float), ``vpin_bucket_volume`` (float, enables VPIN),
        ``vpin_n_buckets`` (int), ``vpin_max`` (float),
        ``jump_schedule_ms`` (list[int]), ``pre_jump_withdraw_ms`` (int),
        ``post_jump_resume_ms`` (int)
      * ``glt`` — params: ``gamma``, ``sigma``, ``kappa``, ``A``, ``size``
    """
    p = dict(params)
    if kind == "constant_spread":
        return ConstantSpread(**p)
    if kind == "avellaneda_stoikov":
        return AvellanedaStoikov(**p)
    if kind == "avellaneda_stoikov_with_signals":
        ofi_window_s = p.pop("ofi_window_s", None)
        ofi = OFICalculator(window_seconds=ofi_window_s) if ofi_window_s else None
        vpin_bucket = p.pop("vpin_bucket_volume", None)
        vpin_n = p.pop("vpin_n_buckets", 50)
        vpin = VPINCalculator(bucket_volume=vpin_bucket, n_buckets=vpin_n) if vpin_bucket else None
        return AvellanedaStoikovWithSignals(ofi=ofi, vpin=vpin, **p)
    if kind == "glt":
        return GLT(**p)
    raise ValueError(f"unknown strategy kind: {kind!r}")


__all__ = ["build_strategy", "merge_calibrated_params"]
