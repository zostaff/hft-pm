"""Build a :class:`Strategy` from a :class:`StrategyConfig`.

Centralised so that CLI tools (backtest, paper trade, live) construct
strategies the same way and validate parameter names early.
"""

from __future__ import annotations

from typing import Any

from ..signals.ofi import OFICalculator
from ..signals.vpin import VPINCalculator
from .avellaneda_stoikov import AvellanedaStoikov, AvellanedaStoikovWithSignals
from .base import Strategy
from .constant_spread import ConstantSpread
from .glt import GLT


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


__all__ = ["build_strategy"]
