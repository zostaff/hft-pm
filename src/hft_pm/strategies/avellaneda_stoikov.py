"""Avellaneda-Stoikov market maker (docs §4.5 closed form).

Reservation price (linear-in-q approximation):

.. math::

    r(t, q) = S_t - q \\gamma \\sigma^2 (T - t)

Half-spread (asymptotic):

.. math::

    \\delta^* = \\gamma \\sigma^2 (T - t)
        + \\frac{2}{\\gamma} \\ln\\!\\left(1 + \\frac{\\gamma}{\\kappa}\\right)

Quotes:

.. math::

    p^b = r - \\delta^*, \\quad p^a = r + \\delta^*

The inventory effect is automatic: positive q skews the reservation
price below the mid, making the maker more aggressive on the bid (to
get back to flat). The half-spread widens as ``T - t`` grows because
the maker faces more residual mid-price risk over the remaining
horizon.

This implementation uses raw (unbounded) prices — fine for the Phase 3
acceptance test on synthetic AS-conforming data. Docs §4.7 lists the
regimes where the closed form breaks (large |q|, long horizon × high
sigma, p near 0 or 1); the logit-space variant in §5 lives in
:mod:`hft_pm.strategies.logit_market_maker` (Phase 5).
"""

from __future__ import annotations

import math

from ..orderbook.events import SimEvent
from ..signals.microprice import microprice as _microprice
from ..signals.ofi import OFICalculator
from ..signals.vpin import VPINCalculator
from .base import SimulatorAPI
from .constant_spread import _snap_down, _snap_up
from .quoting import QuotingStrategy


class AvellanedaStoikov(QuotingStrategy):
    """AS quoter (docs §4.5).

    Parameters
    ----------
    gamma:
        Risk aversion. Larger γ → tighter inventory control, wider spread.
    sigma:
        Mid-price volatility per √second.
    kappa:
        Fill-intensity decay rate.
    horizon_ms:
        Total backtest horizon in ms; used to compute remaining time
        ``T − t`` from ``sim.now_ms``. The strategy assumes ``t = 0`` is
        the timestamp of the first event it sees (set lazily on the
        first callback).
    size:
        Order size on each side.
    """

    def __init__(
        self,
        *,
        gamma: float,
        sigma: float,
        kappa: float,
        horizon_ms: int,
        size: float = 1.0,
    ) -> None:
        super().__init__(size=size)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.horizon_ms = int(horizon_ms)
        self._t0_ms: int | None = None

    def desired_quotes(self, sim: SimulatorAPI) -> tuple[float | None, float | None]:
        mid = sim.book.mid()
        if mid is None:
            return None, None
        if self._t0_ms is None:
            self._t0_ms = sim.now_ms

        elapsed_ms = max(0, sim.now_ms - self._t0_ms)
        tau_s = max(0.0, (self.horizon_ms - elapsed_ms) / 1000.0)
        q = sim.inventory

        gss = self.gamma * self.sigma * self.sigma  # γσ²
        reservation = mid - q * gss * tau_s
        half_spread = gss * tau_s + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)

        tick = sim.book.tick
        bid = _snap_down(reservation - half_spread, tick)
        ask = _snap_up(reservation + half_spread, tick)
        if bid <= 0 or ask >= 1 or bid >= ask:
            return None, None
        return bid, ask


class AvellanedaStoikovWithSignals(QuotingStrategy):
    """AS quoter that consumes microprice / OFI / VPIN signals (docs §6, §8).

    Three opt-in modifications layered on top of vanilla AS:

    * **Microprice as fair value.** Replaces ``mid`` with Stoikov's
      imbalance-weighted microprice. When the book is imbalanced, the
      reservation price tracks the side that's about to print.
    * **OFI alpha skew.** Adds ``alpha_beta · OFI / (γσ²) · (T−t)`` to
      the reservation price (docs §6 alpha-skewed AS, eq. for ``p^a``
      with predictable drift).
    * **VPIN gate.** When VPIN exceeds ``vpin_max``, withdraws both
      sides — adverse-selection protection (§8.3, §4.8).

    Toggle each via constructor flags; tests can A/B them in isolation.
    """

    def __init__(
        self,
        *,
        gamma: float,
        sigma: float,
        kappa: float,
        horizon_ms: int,
        size: float = 1.0,
        use_microprice: bool = False,
        ofi: OFICalculator | None = None,
        alpha_beta: float = 0.0,
        vpin: VPINCalculator | None = None,
        vpin_max: float = 1.0,
        jump_schedule_ms: list[int] | None = None,
        pre_jump_withdraw_ms: int = 0,
        post_jump_resume_ms: int = 0,
    ) -> None:
        super().__init__(size=size)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.horizon_ms = int(horizon_ms)
        self.use_microprice = use_microprice
        self.ofi_calc = ofi
        self.alpha_beta = float(alpha_beta)
        self.vpin_calc = vpin
        self.vpin_max = float(vpin_max)
        self.jump_schedule_ms = sorted(jump_schedule_ms) if jump_schedule_ms else []
        self.pre_jump_withdraw_ms = int(pre_jump_withdraw_ms)
        self.post_jump_resume_ms = int(post_jump_resume_ms)
        self._t0_ms: int | None = None

    def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
        # Side-effect: update signals on every relevant event before the
        # base class decides on quotes. Order matters — signals must
        # reflect *current* event before the requote check reads them.
        self._update_signals(sim, event)
        super().on_event(sim, event)

    def _update_signals(self, sim: SimulatorAPI, event: SimEvent) -> None:
        if self.ofi_calc is not None and event.kind in ("book", "price_change"):
            bb = sim.book.best_bid()
            ba = sim.book.best_ask()
            if bb is not None and ba is not None:
                self.ofi_calc.update(sim.now_ms / 1000.0, bb[0], bb[1], ba[0], ba[1])
        if self.vpin_calc is not None and event.kind == "trade":
            payload = event.payload
            # event.payload is a LastTradePriceEvent
            self.vpin_calc.add_trade(
                volume=payload.size,
                is_buy=payload.side == "BUY",
                price=payload.price,
            )

    def desired_quotes(self, sim: SimulatorAPI) -> tuple[float | None, float | None]:
        bb = sim.book.best_bid()
        ba = sim.book.best_ask()
        if bb is None or ba is None:
            return None, None

        # VPIN gate: withdraw both sides when toxicity is high.
        if self.vpin_calc is not None and self.vpin_calc.value() > self.vpin_max:
            return None, None

        # Scheduled-jump gate: withdraw both sides in a window around
        # each scheduled jump. ``pre_jump_withdraw_ms`` before the
        # jump and ``post_jump_resume_ms`` after, the maker is dark.
        if self.jump_schedule_ms and self._in_jump_window(sim.now_ms):
            return None, None

        if self.use_microprice:
            fair = _microprice(bb[0], bb[1], ba[0], ba[1])
        else:
            fair = (bb[0] + ba[0]) / 2.0

        if self._t0_ms is None:
            self._t0_ms = sim.now_ms
        elapsed_ms = max(0, sim.now_ms - self._t0_ms)
        tau_s = max(0.0, (self.horizon_ms - elapsed_ms) / 1000.0)
        q = sim.inventory

        gss = self.gamma * self.sigma * self.sigma
        reservation = fair - q * gss * tau_s

        # OFI alpha skew: add a direct prediction of the next ΔS from
        # the current OFI value. ``alpha_beta`` is the regression slope
        # of (forward Δmid) on (OFI), in price-units per OFI-unit — i.e.
        # the output of :func:`calibrate_ofi_alpha`. Docs §6 also offers
        # a τ-scaled form, but for the short-horizon quoting use case
        # the single-step linear prediction is what makes empirical
        # sense and is what the calibrator returns.
        if self.ofi_calc is not None and self.alpha_beta != 0.0:
            ofi_now = self.ofi_calc.value()
            reservation += self.alpha_beta * ofi_now

        half_spread = gss * tau_s + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)

        tick = sim.book.tick
        bid = _snap_down(reservation - half_spread, tick)
        ask = _snap_up(reservation + half_spread, tick)
        bid_t: float | None = bid if bid > 0 else None
        ask_t: float | None = ask if ask < 1 else None
        if bid_t is not None and ask_t is not None and bid_t >= ask_t:
            return None, None
        return bid_t, ask_t

    def _in_jump_window(self, now_ms: int) -> bool:
        """True iff ``now_ms`` lies in any scheduled-jump withdraw window."""
        # Binary-search-friendly linear scan — schedules are typically short.
        for jt in self.jump_schedule_ms:
            if jt - self.pre_jump_withdraw_ms <= now_ms <= jt + self.post_jump_resume_ms:
                return True
            if jt - self.pre_jump_withdraw_ms > now_ms:
                # Schedule is sorted; nothing further can match either.
                break
        return False


__all__ = ["AvellanedaStoikov", "AvellanedaStoikovWithSignals"]
