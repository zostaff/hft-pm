"""Unit tests for hft_pm.simulator.synthetic."""

from __future__ import annotations

import pytest

from hft_pm.data.schemas import BookEvent, LastTradePriceEvent
from hft_pm.simulator.synthetic import generate_as_world


def test_generator_yields_timestamp_ordered_events() -> None:
    events = list(generate_as_world(duration_s=10, sigma=0.01, kappa=50, A=2.0, seed=0))
    timestamps = [e.timestamp_ms for e in events]
    assert timestamps == sorted(timestamps)
    assert len(events) > 0


def test_generator_includes_both_event_types() -> None:
    events = list(generate_as_world(duration_s=20, sigma=0.01, kappa=50, A=5.0, seed=0))
    n_book = sum(1 for e in events if isinstance(e, BookEvent))
    n_trade = sum(1 for e in events if isinstance(e, LastTradePriceEvent))
    assert n_book > 0
    assert n_trade > 0


def test_generator_seed_is_reproducible() -> None:
    a = list(generate_as_world(duration_s=5, sigma=0.01, kappa=30, A=2.0, seed=123))
    b = list(generate_as_world(duration_s=5, sigma=0.01, kappa=30, A=2.0, seed=123))
    assert len(a) == len(b)
    for ea, eb in zip(a, b, strict=True):
        assert ea.timestamp_ms == eb.timestamp_ms
        assert type(ea) is type(eb)


def test_book_snapshot_count_is_deterministic() -> None:
    """Snapshot interval is fixed; total snapshots = duration / interval + 1."""
    duration_s = 60
    snapshot_interval_ms = 100
    events = list(
        generate_as_world(
            duration_s=duration_s,
            sigma=0.01,
            kappa=50,
            A=2.0,
            snapshot_interval_ms=snapshot_interval_ms,
            seed=0,
        )
    )
    n_book = sum(1 for e in events if isinstance(e, BookEvent))
    expected = duration_s * 1000 // snapshot_interval_ms + 1
    assert n_book == expected


def test_total_trade_events_increase_with_arrival_rate() -> None:
    """Doubling A should roughly double total trade events emitted."""
    base_events = list(generate_as_world(duration_s=120, sigma=0.005, kappa=50, A=1.0, seed=0))
    high_events = list(generate_as_world(duration_s=120, sigma=0.005, kappa=50, A=4.0, seed=0))
    base_trades = sum(1 for e in base_events if isinstance(e, LastTradePriceEvent))
    high_trades = sum(1 for e in high_events if isinstance(e, LastTradePriceEvent))
    # 4× the rate should give roughly 4× the trades — allow 30 % slack
    # for Poisson variance and walk-length variance.
    assert 2.5 * base_trades < high_trades < 6.0 * base_trades


def test_walk_emits_multiple_levels_per_arrival() -> None:
    """Each Poisson arrival produces ≥ 1 trade event; deeper budgets → more events."""
    events = list(
        generate_as_world(
            duration_s=60,
            sigma=0.001,
            kappa=20,  # mean depth 0.05 → ~50 ticks at tick=0.001
            A=2.0,
            tick=0.001,
            seed=11,
        )
    )
    trades = [e for e in events if isinstance(e, LastTradePriceEvent)]
    # At κ=20, mean walk length is ≈ 50 ticks, so the trade count
    # should massively exceed the arrival count (~120 arrivals).
    assert len(trades) > 200


def test_generator_validates_inputs() -> None:
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=-1, sigma=0.01, kappa=50, A=1))
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=10, sigma=-1, kappa=50, A=1))
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=10, sigma=0.01, kappa=0, A=1))
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=10, sigma=0.01, kappa=50, A=-1))
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=10, sigma=0.01, kappa=50, A=1, mid_start=0))
    with pytest.raises(ValueError):
        list(generate_as_world(duration_s=10, sigma=0.01, kappa=50, A=1, mid_start=1.5))
