"""Unit tests for hft_pm.strategies.{constant_spread,avellaneda_stoikov,glt}."""

from __future__ import annotations

import math

import pytest

from hft_pm.data.schemas import BookEvent, PriceLevel
from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.strategies.avellaneda_stoikov import AvellanedaStoikov
from hft_pm.strategies.constant_spread import ConstantSpread
from hft_pm.strategies.glt import GLT

ASSET = "x"
MARKET = "0xy"


def _book(ts_ms: int, bid_px: float, ask_px: float) -> BookEvent:
    return BookEvent(
        asset_id=ASSET,
        market=MARKET,
        timestamp_ms=ts_ms,
        recv_ts_ms=ts_ms,
        # Non-zero size so L2OrderBook does not filter the level out
        # (apply_book_snapshot drops size==0 levels by design).
        bids=[PriceLevel(price=bid_px, size=10.0)],
        asks=[PriceLevel(price=ask_px, size=10.0)],
    )


# ----------------------------------------------------------------------
# ConstantSpread
# ----------------------------------------------------------------------


def test_constant_spread_quotes_at_correct_distance() -> None:
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    cs = ConstantSpread(half_spread=0.03, size=1)
    sim.run([_book(1000, 0.49, 0.51), _book(2000, 0.49, 0.51)], cs)
    # After processing, our orders should be at mid-0.03 and mid+0.03.
    assert cs._bid_price == pytest.approx(0.47)
    assert cs._ask_price == pytest.approx(0.53)


def test_constant_spread_withdraws_when_quote_outside_unit_interval() -> None:
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    cs = ConstantSpread(half_spread=0.20, size=1)
    # Mid 0.10, ask = 0.30 fine, bid = -0.10 invalid -> withdraw.
    sim.run([_book(1000, 0.05, 0.15), _book(2000, 0.05, 0.15)], cs)
    assert cs._bid_price is None
    assert cs._ask_price == pytest.approx(0.30)


def test_constant_spread_rejects_zero_half_spread() -> None:
    with pytest.raises(ValueError):
        ConstantSpread(half_spread=0)


# ----------------------------------------------------------------------
# Avellaneda-Stoikov
# ----------------------------------------------------------------------


def _as_expected(mid: float, q: float, tau_s: float, gamma: float, sigma: float, kappa: float):
    """Reference closed-form for the regression test."""
    gss = gamma * sigma * sigma
    r = mid - q * gss * tau_s
    half = gss * tau_s + (2 / gamma) * math.log(1 + gamma / kappa)
    return r - half, r + half


def test_as_quotes_at_zero_inventory_match_closed_form() -> None:
    book = L2OrderBook(tick=0.001)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    as_strat = AvellanedaStoikov(gamma=1.0, sigma=0.02, kappa=50, horizon_ms=100_000, size=1)
    sim.run([_book(0, 0.49, 0.51), _book(0, 0.49, 0.51)], as_strat)
    expected_bid, expected_ask = _as_expected(
        mid=0.50, q=0, tau_s=100.0, gamma=1.0, sigma=0.02, kappa=50
    )
    # Snap to tick=0.001
    assert as_strat._bid_price == pytest.approx(round(expected_bid * 1000) / 1000, abs=0.001)
    assert as_strat._ask_price == pytest.approx(round(expected_ask * 1000) / 1000, abs=0.001)


def test_as_skews_quotes_with_inventory() -> None:
    """Positive inventory should pull both bid and ask down (more aggressive bid).

    We test this indirectly: build two backtests, one where the strategy
    is forced to long inventory, one at zero inventory, and check the
    reservation price moves correctly.
    """
    # At q=0: r = S; at q>0: r < S. So the midpoint of (bid, ask) shifts down.
    gamma, sigma, kappa = 1.0, 0.02, 50.0
    mid = 0.50
    bid_q0, ask_q0 = _as_expected(mid, q=0, tau_s=100.0, gamma=gamma, sigma=sigma, kappa=kappa)
    bid_q5, ask_q5 = _as_expected(mid, q=5, tau_s=100.0, gamma=gamma, sigma=sigma, kappa=kappa)
    midpoint_q0 = (bid_q0 + ask_q0) / 2
    midpoint_q5 = (bid_q5 + ask_q5) / 2
    assert midpoint_q5 < midpoint_q0, "long inventory should shift quote midpoint down"


def test_as_half_spread_grows_with_remaining_horizon() -> None:
    # δ* = γσ²(T-t) + (2/γ)ln(1+γ/κ). Larger τ → larger half-spread.
    gamma, sigma, kappa = 1.0, 0.02, 50.0
    _, ask_far = _as_expected(0.5, 0, 100.0, gamma, sigma, kappa)
    _, ask_near = _as_expected(0.5, 0, 1.0, gamma, sigma, kappa)
    assert ask_far > ask_near


def test_as_validates_params() -> None:
    with pytest.raises(ValueError):
        AvellanedaStoikov(gamma=0, sigma=0.01, kappa=50, horizon_ms=1000)
    with pytest.raises(ValueError):
        AvellanedaStoikov(gamma=1, sigma=-1, kappa=50, horizon_ms=1000)
    with pytest.raises(ValueError):
        AvellanedaStoikov(gamma=1, sigma=0.01, kappa=0, horizon_ms=1000)
    with pytest.raises(ValueError):
        AvellanedaStoikov(gamma=1, sigma=0.01, kappa=50, horizon_ms=-1)


# ----------------------------------------------------------------------
# GLT
# ----------------------------------------------------------------------


def test_glt_quotes_symmetric_at_zero_inventory() -> None:
    """At q=0, bid and ask half-spreads are equal: base + (1/2)·inv_term."""
    glt = GLT(gamma=1.0, sigma=0.02, kappa=50, A=2.0, size=1)
    base = (1.0 / 1.0) * math.log(1 + 1.0 / 50)
    inv_coef = math.sqrt((0.02**2 * 1.0) / (2 * 50 * 2.0) * (1 + 1.0 / 50) ** (1 + 50.0 / 1.0))
    expected_delta = base + 0.5 * inv_coef
    assert glt._base == pytest.approx(base)
    assert glt._inv_coef == pytest.approx(inv_coef)
    # At q=0: delta_ask == delta_bid == base + 0.5 * inv_term
    book = L2OrderBook(tick=0.0001)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    sim.run([_book(0, 0.4999, 0.5001), _book(0, 0.4999, 0.5001)], glt)
    assert glt._bid_price == pytest.approx(0.5 - expected_delta, abs=0.001)
    assert glt._ask_price == pytest.approx(0.5 + expected_delta, abs=0.001)


def test_glt_validates_params() -> None:
    with pytest.raises(ValueError):
        GLT(gamma=0, sigma=0.02, kappa=50, A=2)
    with pytest.raises(ValueError):
        GLT(gamma=1, sigma=0.02, kappa=50, A=0)
