"""Latency models for the simulator.

The simulator never lets a strategy's order arrive at the book at the
decision time. Instead, the engine asks the configured
:class:`LatencyModel` what arrival time to schedule, and pushes a
``place_arrival`` event on the heap at that timestamp. Same for
cancellations.

This is the seam Phase 6 will exercise with delay-injection (+100ms /
+500ms / +2s) and shuffle tests. Keeping latency as an injectable object
means those tests never need to monkey-patch.

Models
------
* :class:`ConstantLatency` — deterministic shift. Use in unit tests and
  as a baseline.
* :class:`GaussianLatency` — Normal(mean, std) clamped to ``≥ 0``. Seedable
  for reproducibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LatencyModel(ABC):
    """Decide the arrival timestamp for a strategy action."""

    @abstractmethod
    def sample(self, t_decision_ms: int) -> int:
        """Return the arrival timestamp (epoch ms) given the decision time.

        Implementations must ensure the returned value is at least
        ``t_decision_ms`` (negative latency would let the strategy
        act in the past — a look-ahead leak).
        """


class ConstantLatency(LatencyModel):
    """Always add a fixed number of milliseconds."""

    def __init__(self, latency_ms: int) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        self.latency_ms = int(latency_ms)

    def sample(self, t_decision_ms: int) -> int:
        return t_decision_ms + self.latency_ms


class GaussianLatency(LatencyModel):
    """Draw latency from a Normal distribution, clamped to ``≥ 0``.

    Parameters
    ----------
    mean_ms, stddev_ms:
        Normal parameters in milliseconds.
    seed:
        Optional seed for the internal :class:`numpy.random.Generator`.
        Pass an explicit seed in tests to get reproducible draws.
    """

    def __init__(self, mean_ms: float, stddev_ms: float, *, seed: int | None = None) -> None:
        if mean_ms < 0:
            raise ValueError("mean_ms must be non-negative")
        if stddev_ms < 0:
            raise ValueError("stddev_ms must be non-negative")
        self.mean_ms = float(mean_ms)
        self.stddev_ms = float(stddev_ms)
        self._rng = np.random.default_rng(seed)

    def sample(self, t_decision_ms: int) -> int:
        if self.stddev_ms == 0:
            draw = self.mean_ms
        else:
            draw = float(self._rng.normal(self.mean_ms, self.stddev_ms))
        # Clamp at zero — a negative draw would let the strategy act in
        # the past, which is a look-ahead leak.
        return t_decision_ms + max(0, round(draw))


__all__ = ["ConstantLatency", "GaussianLatency", "LatencyModel"]
