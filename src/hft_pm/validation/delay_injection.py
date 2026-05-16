"""Delay-injection robustness test (CLAUDE.md Phase 6 acceptance).

Wraps a backtest's :class:`LatencyModel` to add a constant extra delay.
A genuine alpha degrades smoothly as latency grows: +100 ms shaves
some edge, +500 ms shaves more, +2 s typically reduces Sharpe to a
fraction of the no-delay baseline. **A strategy whose Sharpe collapses
at +100 ms is almost certainly leaking look-ahead information** — the
delay reveals that it was reacting to information faster than physically
possible.

The wrapper composes with any concrete latency model (``ConstantLatency``,
``GaussianLatency``, etc.) without modifying it.
"""

from __future__ import annotations

from ..simulator.latency import LatencyModel


class DelayInjector(LatencyModel):
    """Add ``extra_ms`` on top of the inner model's sample.

    Parameters
    ----------
    inner:
        Any concrete :class:`LatencyModel` (constant, Gaussian, etc.).
    extra_ms:
        Constant extra latency in milliseconds. Must be non-negative.
    """

    def __init__(self, inner: LatencyModel, extra_ms: int) -> None:
        if extra_ms < 0:
            raise ValueError("extra_ms must be non-negative")
        self.inner = inner
        self.extra_ms = int(extra_ms)

    def sample(self, t_decision_ms: int) -> int:
        return self.inner.sample(t_decision_ms) + self.extra_ms


__all__ = ["DelayInjector"]
