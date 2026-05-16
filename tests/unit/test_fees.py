"""Unit tests for hft_pm.fees.polymarket."""

from __future__ import annotations

import math

from hft_pm.fees.polymarket import (
    FeeCategory,
    effective_half_spread_with_rebate,
    maker_rebate,
    taker_fee,
)


def test_taker_fee_peaks_at_midpoint() -> None:
    # At p=0.5, fee == peak_rate x size for every category.
    for cat in FeeCategory:
        if cat.peak_rate == 0:
            continue
        size = 100.0
        fee = taker_fee(0.5, size, cat)
        expected = cat.peak_rate * size
        assert math.isclose(fee, expected, rel_tol=1e-12), f"{cat.name}: {fee} != {expected}"


def test_taker_fee_zero_at_boundaries() -> None:
    for cat in FeeCategory:
        assert taker_fee(0.0, 100, cat) == 0.0
        assert taker_fee(1.0, 100, cat) == 0.0


def test_taker_fee_symmetry_around_half() -> None:
    # fee(p) == fee(1-p) for any p; comes from p(1-p) symmetry.
    for cat in FeeCategory:
        if cat.peak_rate == 0:
            continue
        for p in (0.1, 0.25, 0.4, 0.49):
            assert math.isclose(taker_fee(p, 100, cat), taker_fee(1 - p, 100, cat))


def test_maker_rebate_is_share_of_taker_fee() -> None:
    for cat in FeeCategory:
        rebate = maker_rebate(0.5, 100, cat)
        fee = taker_fee(0.5, 100, cat)
        assert math.isclose(rebate, fee * cat.rebate_share)


def test_geopolitics_is_fee_free() -> None:
    assert taker_fee(0.5, 100, FeeCategory.GEOPOLITICS) == 0.0
    assert maker_rebate(0.5, 100, FeeCategory.GEOPOLITICS) == 0.0


def test_finance_has_50pct_rebate_share() -> None:
    # Spec call-out: Finance's 50% rebate is the deliberate Polymarket
    # liquidity subsidy. Guard the value so a refactor cannot drop it.
    assert FeeCategory.FINANCE.rebate_share == 0.50


def test_taker_fee_clamps_invalid_price() -> None:
    # Negative or >1 price should not produce a negative fee.
    assert taker_fee(-0.1, 100, FeeCategory.FINANCE) == 0.0
    assert taker_fee(1.1, 100, FeeCategory.FINANCE) == 0.0


def test_effective_half_spread_includes_rebate() -> None:
    # Spec example: at p=0.5 in FINANCE, rebate per $1 notional == 0.005.
    eff = effective_half_spread_with_rebate(0.005, 0.5, FeeCategory.FINANCE)
    assert math.isclose(eff, 0.005 + 0.005)


def test_unknown_category_default_is_zero() -> None:
    assert taker_fee(0.5, 100, FeeCategory.OTHER) == 0.0
    assert maker_rebate(0.5, 100, FeeCategory.OTHER) == 0.0
