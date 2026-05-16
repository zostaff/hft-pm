"""Unit tests for hft_pm.hawkes.{intensity,mle}."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hft_pm.hawkes.intensity import HawkesIntensity
from hft_pm.hawkes.mle import fit_hawkes, hawkes_log_likelihood

# ----------------------------------------------------------------------
# HawkesIntensity
# ----------------------------------------------------------------------


def test_intensity_equals_mu_with_no_events() -> None:
    h = HawkesIntensity(mu=0.5, alpha=1.0, beta=2.0)
    assert h.update(0.0, event=False) == pytest.approx(0.5)
    assert h.update(10.0, event=False) == pytest.approx(0.5)


def test_intensity_jumps_on_event() -> None:
    h = HawkesIntensity(mu=0.5, alpha=1.0, beta=2.0)
    h.update(0.0, event=False)
    val = h.update(1.0, event=True)
    # state was 0; decayed = 0; then += alpha = 1; λ = 0.5 + 1 = 1.5
    assert val == pytest.approx(1.5)


def test_intensity_decays_between_events() -> None:
    h = HawkesIntensity(mu=0.0, alpha=1.0, beta=2.0)
    h.update(0.0, event=True)
    val = h.update(1.0, event=False)
    # state was 1; decayed by exp(-2 * 1) = 0.1353; λ = 0.1353
    assert val == pytest.approx(math.exp(-2.0))


def test_intensity_rejects_backward_time() -> None:
    h = HawkesIntensity(mu=0.5, alpha=1.0, beta=2.0)
    h.update(5.0, event=False)
    with pytest.raises(ValueError):
        h.update(4.0, event=False)


def test_intensity_validates_params() -> None:
    with pytest.raises(ValueError):
        HawkesIntensity(mu=-1, alpha=1, beta=1)
    with pytest.raises(ValueError):
        HawkesIntensity(mu=1, alpha=-1, beta=1)
    with pytest.raises(ValueError):
        HawkesIntensity(mu=1, alpha=1, beta=0)


def test_branching_ratio() -> None:
    h = HawkesIntensity(mu=1.0, alpha=0.5, beta=2.0)
    assert h.branching_ratio() == pytest.approx(0.25)


# ----------------------------------------------------------------------
# MLE
# ----------------------------------------------------------------------


def _simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed: int) -> np.ndarray:
    """Generate one Hawkes path via Ogata thinning."""
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    lam_bar = mu  # upper bound on intensity, updated when needed
    while t < T:
        # Sample exponential with current upper bound λ̄.
        u = rng.exponential(1.0 / lam_bar)
        t += u
        if t >= T:
            break
        # Compute actual intensity at t.
        intensity = mu + sum(alpha * math.exp(-beta * (t - s)) for s in events)
        if rng.random() < intensity / lam_bar:
            events.append(t)
            lam_bar = intensity + alpha  # new upper bound after the jump
        else:
            # No event; lower λ̄ down toward intensity for next iter.
            lam_bar = max(intensity, mu)
    return np.asarray(events, dtype=np.float64)


def test_mle_log_likelihood_returns_finite_for_simulated_path() -> None:
    mu, alpha, beta = 0.5, 0.3, 1.0
    T = 100.0
    ts = _simulate_hawkes(mu, alpha, beta, T, seed=0)
    val = hawkes_log_likelihood((mu, alpha, beta), ts, T)
    assert math.isfinite(val)


def test_mle_log_likelihood_inf_for_invalid_params() -> None:
    ts = np.array([0.1, 0.5, 0.9])
    assert hawkes_log_likelihood((-1, 0.3, 1.0), ts, 1.0) == float("inf")
    assert hawkes_log_likelihood((0.5, -0.3, 1.0), ts, 1.0) == float("inf")
    assert hawkes_log_likelihood((0.5, 0.3, 0.0), ts, 1.0) == float("inf")


def test_mle_recovers_params_on_long_simulated_path() -> None:
    """Fit on 2000s of simulated Hawkes; recover (μ, α, β) within tolerance."""
    mu_true, alpha_true, beta_true = 0.8, 0.4, 1.5
    T = 2000.0
    ts = _simulate_hawkes(mu_true, alpha_true, beta_true, T, seed=42)
    assert len(ts) > 200
    result = fit_hawkes(ts, T)
    # Tight on μ; α and β are coupled (only their ratio is well-identified
    # on short paths), so allow generous tolerance.
    assert abs(result["mu"] - mu_true) / mu_true < 0.30
    assert result["branching_ratio"] == pytest.approx(alpha_true / beta_true, abs=0.10)
    assert result["stationary"] is True
    assert result["success"] is True


def test_mle_warns_on_explosive_fit() -> None:
    """If the fit lands at α/β ≥ 1, emit RuntimeWarning."""
    # Force explosion by giving an initial guess in the unstable region.
    # We feed a short, irregular sequence that the optimiser may fit
    # explosively; the warning code path is what we test.
    ts = np.array([0.1, 0.15, 0.2, 0.21, 0.22])
    T = 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fit_hawkes(ts, T, initial_guess=(0.5, 5.0, 1.0))
    if not res["stationary"]:
        assert any(issubclass(rec.category, RuntimeWarning) for rec in w)


def test_mle_rejects_too_few_events() -> None:
    with pytest.raises(ValueError):
        fit_hawkes([0.1, 0.5], T=1.0)


def test_mle_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError):
        fit_hawkes([0.5, 0.1, 0.9], T=1.0)


def test_mle_rejects_window_smaller_than_last_event() -> None:
    with pytest.raises(ValueError):
        fit_hawkes([0.1, 0.5, 0.9], T=0.5)
