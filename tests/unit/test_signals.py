"""Unit tests for hft_pm.signals.{ofi,microprice,vpin}."""

from __future__ import annotations

import pytest

from hft_pm.signals.microprice import imbalance, microprice
from hft_pm.signals.ofi import OFICalculator
from hft_pm.signals.vpin import VPINCalculator

# ----------------------------------------------------------------------
# OFICalculator
# ----------------------------------------------------------------------


def test_ofi_first_event_returns_zero() -> None:
    o = OFICalculator(window_seconds=1.0)
    assert o.update(0.0, 0.50, 100, 0.51, 100) == 0.0


def test_ofi_bid_price_up_adds_new_bid_size() -> None:
    """Bid moved up → buying pressure → +bid_sz."""
    o = OFICalculator(window_seconds=10.0)
    o.update(0.0, 0.50, 100, 0.51, 100)  # baseline
    val = o.update(0.5, 0.51, 200, 0.52, 100)
    # bid_px up: e_bid = +200; ask_px up: e_ask = +prev_ask_sz = +100. e_n = 300.
    assert val == pytest.approx(300.0)


def test_ofi_bid_price_down_subtracts_old_bid() -> None:
    o = OFICalculator(window_seconds=10.0)
    o.update(0.0, 0.50, 100, 0.51, 100)
    val = o.update(0.5, 0.49, 50, 0.51, 100)
    # bid_px down: e_bid = -100; ask unchanged: e_ask = -(100-100) = 0. e_n = -100.
    assert val == pytest.approx(-100.0)


def test_ofi_window_evicts_old_events() -> None:
    o = OFICalculator(window_seconds=1.0)
    o.update(0.0, 0.50, 100, 0.51, 100)
    o.update(0.1, 0.51, 200, 0.52, 100)  # +300
    val_before = o.value()
    # advance 2s — should evict the +300 event from window
    o.update(2.5, 0.51, 200, 0.52, 100)
    assert val_before == pytest.approx(300.0)
    # After eviction, only the most recent zero-flow event remains.
    assert o.value() == pytest.approx(0.0)


def test_ofi_rejects_zero_or_negative_window() -> None:
    with pytest.raises(ValueError):
        OFICalculator(window_seconds=0)


# ----------------------------------------------------------------------
# microprice
# ----------------------------------------------------------------------


def test_microprice_equals_mid_for_balanced_book() -> None:
    assert microprice(0.49, 100, 0.51, 100) == pytest.approx(0.50)


def test_microprice_pulls_toward_ask_when_bid_heavy() -> None:
    """Big bid → microprice closer to ask (next print likely takes the ask)."""
    mp = microprice(0.49, 1000, 0.51, 100)
    mid = 0.50
    assert mp > mid


def test_microprice_pulls_toward_bid_when_ask_heavy() -> None:
    mp = microprice(0.49, 100, 0.51, 1000)
    mid = 0.50
    assert mp < mid


def test_microprice_falls_back_to_mid_when_zero_size() -> None:
    assert microprice(0.49, 0, 0.51, 0) == pytest.approx(0.50)


def test_imbalance_sign_and_range() -> None:
    assert imbalance(100, 0) == pytest.approx(1.0)
    assert imbalance(0, 100) == pytest.approx(-1.0)
    assert imbalance(50, 50) == 0.0
    assert imbalance(0, 0) == 0.0


# ----------------------------------------------------------------------
# VPINCalculator
# ----------------------------------------------------------------------


def test_vpin_returns_zero_with_no_buckets() -> None:
    v = VPINCalculator(bucket_volume=100)
    v.add_trade(50, is_buy=True, price=0.5)
    # Bucket not yet closed.
    assert v.n_closed_buckets == 0
    assert v.value() == 0.0


def test_vpin_balanced_buckets_give_low_value() -> None:
    v = VPINCalculator(bucket_volume=100, n_buckets=10)
    for _ in range(20):
        v.add_trade(50, is_buy=True, price=0.5)
        v.add_trade(50, is_buy=False, price=0.5)
    # Balanced → |buy − sell| ≈ 0 in each bucket → VPIN ≈ 0
    assert v.value() < 0.05


def test_vpin_one_sided_buckets_give_high_value() -> None:
    v = VPINCalculator(bucket_volume=100, n_buckets=10)
    for _ in range(20):
        v.add_trade(100, is_buy=True, price=0.5)
    # All buy: |buy − sell| / (sqrt(0.25)·total) = total / (0.5·total) = 2
    val = v.value()
    assert val > 1.5  # well above 0


def test_vpin_pm_normalisation_shrinks_near_boundary() -> None:
    """At p near 0 or 1, sqrt(p(1−p)) shrinks the denom — wait it shrinks
    the denom, INCREASING the VPIN value. The PM normalisation amplifies
    boundary toxicity rather than damping it. Test reflects that direction.
    """
    v_mid = VPINCalculator(bucket_volume=100, n_buckets=10)
    v_edge = VPINCalculator(bucket_volume=100, n_buckets=10)
    for _ in range(20):
        v_mid.add_trade(100, is_buy=True, price=0.5)
        v_edge.add_trade(100, is_buy=True, price=0.05)
    assert v_edge.value() > v_mid.value()


def test_vpin_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        VPINCalculator(bucket_volume=0)
    with pytest.raises(ValueError):
        VPINCalculator(bucket_volume=100, n_buckets=0)


def test_vpin_handles_zero_volume_trades() -> None:
    v = VPINCalculator(bucket_volume=100)
    v.add_trade(0, is_buy=True, price=0.5)  # no-op
    assert v.value() == 0.0
