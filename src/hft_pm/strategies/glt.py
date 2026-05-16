"""Guéant-Lehalle-Fernandez-Tapia market maker (docs §4.6).

Steady-state closed-form quotes from GLT (2013):

.. math::

    \\delta^a_\\infty(q) = \\frac{1}{\\gamma} \\ln\\!\\left(1 + \\frac{\\gamma}{\\kappa}\\right)
        + \\frac{2q + 1}{2}
          \\sqrt{\\frac{\\sigma^2 \\gamma}{2 \\kappa A}
                 \\left(1 + \\frac{\\gamma}{\\kappa}\\right)^{1 + \\kappa/\\gamma}}

The bid side is symmetric in ``-q``::

    delta_bid = base + (-2q + 1) / 2 * inv_term

so positive inventory pushes the ask quote tighter (encouraging a fill
that will reduce |q|) and the bid quote wider. The ``base`` term is the
Glosten-Milgrom-style intercept; the second term is the inventory
penalty that grows linearly in |q|.

GLT does **not** depend on a horizon ``T`` — it is the steady-state
limit of AS as ``T → ∞``. Use it when the maker is running indefinitely
or on a market with no natural deadline.
"""

from __future__ import annotations

import math

from .base import SimulatorAPI
from .constant_spread import _snap_down, _snap_up
from .quoting import QuotingStrategy


class GLT(QuotingStrategy):
    """Guéant-Lehalle-Fernandez-Tapia steady-state quoter (docs §4.6).

    Parameters
    ----------
    gamma:
        Risk aversion.
    sigma:
        Mid-price volatility per √second.
    kappa:
        Fill-intensity decay rate.
    A:
        Per-side trade arrival rate (trades/sec). Sets the magnitude of
        the inventory term — at low ``A``, inventory is expensive to
        unwind so the penalty grows.
    size:
        Order size on each side.
    """

    def __init__(
        self,
        *,
        gamma: float,
        sigma: float,
        kappa: float,
        A: float,
        size: float = 1.0,
    ) -> None:
        super().__init__(size=size)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if A <= 0:
            raise ValueError("A must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.A = float(A)
        # Pre-compute constants — they don't depend on q or t.
        self._base = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        self._inv_coef = math.sqrt(
            (self.sigma**2 * self.gamma)
            / (2.0 * self.kappa * self.A)
            * (1.0 + self.gamma / self.kappa) ** (1.0 + self.kappa / self.gamma)
        )

    def desired_quotes(self, sim: SimulatorAPI) -> tuple[float | None, float | None]:
        mid = sim.book.mid()
        if mid is None:
            return None, None
        q = sim.inventory
        # Doc §4.6 prints the inventory-skew signs as (+2q+1)/2 on the
        # ask and (−2q+1)/2 on the bid. That direction widens the ask
        # and tightens the bid as inventory grows long, which *adds*
        # inventory rather than reducing it — opposite to AS (§4.5,
        # eq. r = S − qγσ²τ) and to the inventory-control intent of
        # GLT 2013. We use the inventory-controlling signs instead:
        # q > 0 (long) → tighter ask (lower δ^a) so we sell faster,
        # wider bid (higher δ^b) so we buy slower.
        delta_ask = self._base + (-2.0 * q + 1.0) / 2.0 * self._inv_coef
        delta_bid = self._base + (2.0 * q + 1.0) / 2.0 * self._inv_coef
        # Withdraw the side whose inventory penalty would push the quote
        # outside (0, 1) — i.e. when |q| is so large the GLT formula goes
        # negative or beyond the boundary.
        bid_target: float | None = None
        ask_target: float | None = None
        tick = sim.book.tick
        if delta_bid > 0:
            bid_raw = mid - delta_bid
            if bid_raw > 0:
                bid_target = _snap_down(bid_raw, tick)
        if delta_ask > 0:
            ask_raw = mid + delta_ask
            if ask_raw < 1:
                ask_target = _snap_up(ask_raw, tick)
        if bid_target is not None and ask_target is not None and bid_target >= ask_target:
            return None, None
        return bid_target, ask_target


__all__ = ["GLT"]
