"""Unit tests for hft_pm.simulator.latency."""

from __future__ import annotations

import pytest

from hft_pm.simulator.latency import ConstantLatency, GaussianLatency


def test_constant_latency_adds_fixed_offset() -> None:
    lat = ConstantLatency(50)
    assert lat.sample(1000) == 1050
    assert lat.sample(0) == 50


def test_constant_latency_rejects_negative() -> None:
    with pytest.raises(ValueError):
        ConstantLatency(-1)


def test_gaussian_latency_is_seeded_reproducible() -> None:
    a = GaussianLatency(50, 10, seed=42)
    b = GaussianLatency(50, 10, seed=42)
    samples_a = [a.sample(1000) for _ in range(10)]
    samples_b = [b.sample(1000) for _ in range(10)]
    assert samples_a == samples_b


def test_gaussian_latency_clamps_at_zero() -> None:
    # Mean 0, large stddev — many draws should be negative pre-clamp.
    # Post-clamp: never below decision time.
    lat = GaussianLatency(0, 100, seed=0)
    for _ in range(200):
        assert lat.sample(1000) >= 1000


def test_gaussian_with_zero_stddev_is_deterministic() -> None:
    lat = GaussianLatency(75, 0, seed=123)
    assert all(lat.sample(0) == 75 for _ in range(10))


def test_gaussian_rejects_negative_params() -> None:
    with pytest.raises(ValueError):
        GaussianLatency(-1, 10)
    with pytest.raises(ValueError):
        GaussianLatency(50, -1)
