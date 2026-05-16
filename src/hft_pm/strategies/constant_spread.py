"""Constant-spread baseline strategy.

Quotes ``bid = mid − δ`` and ``ask = mid + δ`` for a fixed half-spread δ,
ignoring inventory entirely. Used as the Phase 3 baseline against which
AS / GLT must demonstrate a Sharpe improvement.
"""

from __future__ import annotations

import math

from .base import SimulatorAPI
from .quoting import QuotingStrategy


class ConstantSpread(QuotingStrategy):
    """Symmetric, inventory-blind quoter at fixed half-spread.

    Parameters
    ----------
    half_spread:
        Distance from the mid at which to quote each side, in price units.
    size:
        Order size on each side.
    tick:
        Optional explicit tick size used to round quotes. Defaults to
        the book's own tick at quoting time.
    """

    def __init__(self, *, half_spread: float, size: float = 1.0, tick: float | None = None) -> None:
        super().__init__(size=size)
        if half_spread <= 0:
            raise ValueError("half_spread must be positive")
        self.half_spread = float(half_spread)
        self._tick = tick

    def desired_quotes(self, sim: SimulatorAPI) -> tuple[float | None, float | None]:
        mid = sim.book.mid()
        if mid is None:
            return None, None
        tick = self._tick or sim.book.tick
        bid_raw = _snap_down(mid - self.half_spread, tick)
        ask_raw = _snap_up(mid + self.half_spread, tick)
        # Withdraw each side individually if it would land outside the
        # unit interval; the other side may still be valid.
        bid: float | None = bid_raw if bid_raw > 0 else None
        ask: float | None = ask_raw if ask_raw < 1 else None
        if bid is not None and ask is not None and bid >= ask:
            return None, None
        return bid, ask


def _snap_down(price: float, tick: float) -> float:
    """Snap to the nearest tick at or below ``price`` (used for bid quotes).

    Adds half a tick of slack to absorb float arithmetic noise so that
    inputs like 0.470000000001 still snap to 0.47 rather than dropping
    to 0.46.
    """
    return math.floor(price / tick + 1e-9) * tick


def _snap_up(price: float, tick: float) -> float:
    """Snap to the nearest tick at or above ``price`` (used for ask quotes)."""
    return math.ceil(price / tick - 1e-9) * tick


__all__ = ["ConstantSpread"]
