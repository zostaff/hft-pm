"""Unit tests for hft_pm.signals.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from hft_pm.signals.calibration import (
    estimate_arrival_rate,
    estimate_kappa,
    estimate_sigma,
)


def test_estimate_sigma_recovers_ground_truth() -> None:
    """Generate a Brownian path with known σ, recover it within tolerance."""
    rng = np.random.default_rng(0)
    true_sigma = 0.02  # per √s
    n = 5000
    dt_s = 0.1
    increments = rng.normal(0.0, true_sigma * np.sqrt(dt_s), size=n)
    mids = 0.5 + np.cumsum(increments)
    timestamps_ms = (np.arange(n + 1) * int(dt_s * 1000)).tolist()
    mids_full = [0.5, *mids.tolist()]

    estimated = estimate_sigma(timestamps_ms, mids_full)
    # 5000 samples should give standard error ~true_sigma / sqrt(2*5000) ≈ 0.0002.
    assert abs(estimated - true_sigma) < 5e-3


def test_estimate_sigma_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError):
        estimate_sigma([1000, 500, 1500], [0.5, 0.51, 0.52])


def test_estimate_sigma_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError):
        estimate_sigma([1000], [0.5])


def test_estimate_arrival_rate_poisson() -> None:
    """N events uniform over T s should give λ̂ ≈ N/T."""
    rng = np.random.default_rng(0)
    true_lambda = 5.0  # events/sec
    duration_s = 1000
    n = int(true_lambda * duration_s)
    arrivals = np.sort(rng.uniform(0, duration_s, size=n))
    timestamps_ms = (arrivals * 1000).astype(np.int64).tolist()
    estimated = estimate_arrival_rate(timestamps_ms, observation_window_ms=duration_s * 1000)
    assert abs(estimated - true_lambda) < 0.5


def test_estimate_arrival_rate_uses_span_when_window_omitted() -> None:
    timestamps_ms = [0, 1000, 2000, 3000]  # span 3000 ms = 3s, n=4
    rate = estimate_arrival_rate(timestamps_ms)
    assert rate == pytest.approx(4.0 * 1000 / 3000)


def test_estimate_kappa_recovers_exponential_rate() -> None:
    rng = np.random.default_rng(0)
    true_kappa = 50.0  # 1/price-units
    depths = rng.exponential(1.0 / true_kappa, size=10000)
    estimated = estimate_kappa(depths)
    # MLE 1/mean has standard error ~true_kappa/sqrt(N) ≈ 0.5.
    assert abs(estimated - true_kappa) < 2.0


def test_estimate_kappa_rejects_negative_depth() -> None:
    with pytest.raises(ValueError):
        estimate_kappa([0.01, -0.005, 0.02])


def test_estimate_kappa_rejects_empty() -> None:
    with pytest.raises(ValueError):
        estimate_kappa([])
