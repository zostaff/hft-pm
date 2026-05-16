"""Univariate exponential-kernel Hawkes intensity (docs §6).

The conditional intensity given history :math:`\\mathcal{H}_t` is

.. math::

    \\lambda_t = \\mu + \\sum_{t_i < t} \\alpha \\, e^{-\\beta (t - t_i)}.

State is kept compact: a single scalar ``state`` holds the
exponentially-decayed sum of past events' impulses. Between calls, the
state decays as ``state · exp(−β · Δt)``; when an event arrives, the
state gains ``α``. The current intensity is then ``μ + state``.

Reference: docs §6 (online tracker) and §6.3 (calibration).
"""

from __future__ import annotations

import math


class HawkesIntensity:
    """Online intensity for a univariate exponential-kernel Hawkes process.

    Parameters
    ----------
    mu:
        Background (baseline) intensity, must be ≥ 0.
    alpha:
        Jump in intensity when an event arrives, must be ≥ 0.
    beta:
        Exponential decay rate of past-event impact, must be > 0.

    The stability (branching-ratio) condition ``alpha / beta < 1`` is
    not enforced at construction — it is a *fit* property and the
    intensity tracker is well-defined even with explosive parameters
    over a finite horizon. :func:`hft_pm.hawkes.mle.fit_hawkes` does
    enforce it.
    """

    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        if mu < 0:
            raise ValueError("mu must be non-negative")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.state: float = 0.0
        self.last_t: float | None = None

    def update(self, t: float, event: bool) -> float:
        """Decay state to time ``t``, optionally add an event impulse, return λ(t).

        Parameters
        ----------
        t:
            Current time in seconds. Must be ≥ the previous ``t``.
        event:
            Whether an event just occurred at ``t``. If ``True``, the
            state gains ``alpha`` *after* decay (so the returned λ
            includes the new event's contribution).
        """
        if self.last_t is not None:
            if t < self.last_t:
                raise ValueError("time must be non-decreasing")
            dt = t - self.last_t
            self.state *= math.exp(-self.beta * dt)
        if event:
            self.state += self.alpha
        self.last_t = t
        return self.mu + self.state

    def value(self) -> float:
        """Return λ at the last update time without advancing."""
        return self.mu + self.state

    def branching_ratio(self) -> float:
        """Return ``alpha / beta`` — must be < 1 for a stationary process."""
        return self.alpha / self.beta


__all__ = ["HawkesIntensity"]
