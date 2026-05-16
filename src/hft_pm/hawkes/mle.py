"""Exponential-kernel Hawkes MLE calibration (docs §6.3).

Closed-form negative log-likelihood for the univariate exponential-kernel
Hawkes process with parameters ``(μ, α, β)``::

    log L = Σ_i log(μ + α R_i) - μ T - (α/β) Σ_i (1 - exp(-β (T - t_i)))

where ``R_i = Σ_{j<i} exp(-β (t_i - t_j))`` is computed via the
recurrence ``R_i = exp(-β (t_i - t_{i-1})) (R_{i-1} + 1)`` with
``R_1 = 0``. Implementation is the one inlined in docs §6.3, factored
out into a fittable form.

The :func:`fit_hawkes` wrapper enforces the stationarity constraint
``α / β < 1`` by penalising the objective heavily — explosive fits are
flagged with a warning rather than silently returned.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize


def hawkes_log_likelihood(
    params: np.ndarray | Sequence[float],
    event_times: np.ndarray,
    T: float,
) -> float:
    """Negative log-likelihood for use with ``scipy.optimize.minimize``.

    Parameters
    ----------
    params:
        ``[μ, α, β]``. All must be positive in the optimiser bounds.
    event_times:
        Sorted ascending event timestamps in the same units as ``T``.
    T:
        Observation window length (so the absent-events term integrates
        over ``[0, T]``).

    Returns
    -------
    Negative log-likelihood (smaller = better fit).
    """
    mu, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    if mu <= 0 or alpha < 0 or beta <= 0:
        return float("inf")

    ts = np.asarray(event_times, dtype=np.float64)
    n = len(ts)
    if n == 0:
        return mu * T

    # Compute R_i via the exponential-decay recurrence.
    R = np.empty(n, dtype=np.float64)
    R[0] = 0.0
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        R[i] = math.exp(-beta * dt) * (R[i - 1] + 1.0)

    intensities = mu + alpha * R
    if np.any(intensities <= 0):
        return float("inf")
    sum_log = float(np.sum(np.log(intensities)))

    # Compensator term — closed form.
    integral_kernel = float(np.sum(1.0 - np.exp(-beta * (T - ts))))
    compensator = mu * T + (alpha / beta) * integral_kernel

    return -(sum_log - compensator)


def fit_hawkes(
    event_times: Sequence[float] | np.ndarray,
    T: float,
    *,
    initial_guess: tuple[float, float, float] | None = None,
    enforce_stationarity: bool = True,
) -> dict[str, float | bool]:
    """Fit ``(μ, α, β)`` from observed event times.

    Returns a dict with ``mu``, ``alpha``, ``beta``, ``branching_ratio``,
    ``stationary`` (whether ``α/β < 1``), ``n_events``, ``log_likelihood``,
    and ``success``.

    If ``enforce_stationarity`` and the fit is explosive (``α/β ≥ 1``),
    emits a ``RuntimeWarning``. The caller should not use explosive fits
    in production — see docs §6.3.
    """
    ts = np.asarray(event_times, dtype=np.float64)
    if len(ts) < 3:
        raise ValueError("need at least 3 events to fit Hawkes")
    if np.any(np.diff(ts) < 0):
        raise ValueError("event_times must be sorted ascending")
    if ts[-1] >= T:
        raise ValueError("observation window T must exceed the last event time")

    if initial_guess is None:
        # Heuristic: μ ≈ N/(2T) (half the rate is background), α small,
        # β moderate. The optimiser walks from here.
        mu0 = max(len(ts) / (2.0 * T), 1e-6)
        initial_guess = (mu0, mu0 * 0.5, mu0 * 2.0)

    result = minimize(
        hawkes_log_likelihood,
        x0=np.asarray(initial_guess, dtype=np.float64),
        args=(ts, T),
        method="L-BFGS-B",
        bounds=[(1e-9, None), (1e-9, None), (1e-9, None)],
    )

    mu, alpha, beta = float(result.x[0]), float(result.x[1]), float(result.x[2])
    branching = alpha / beta
    stationary = branching < 1.0
    if enforce_stationarity and not stationary:
        warnings.warn(
            f"Hawkes fit is explosive: alpha/beta = {branching:.3f} >= 1. "
            "Do not use in production — see docs §6.3.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": branching,
        "stationary": stationary,
        "n_events": len(ts),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
    }


__all__ = ["fit_hawkes", "hawkes_log_likelihood"]
