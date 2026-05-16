"""Shared quoting-strategy machinery.

Most market-making strategies follow the same loop:

1. On each book / trade event, decide a target ``(bid_price, ask_price)``.
2. Reconcile with currently-resting orders: cancel any quote that no
   longer matches the target; place a new quote where one is missing.

:class:`QuotingStrategy` factors that loop out so concrete strategies
(constant-spread, AS, GLT) only need to implement the pricing function
:meth:`desired_quotes`. Returning ``(None, None)`` for either side
withdraws that side; the reconciler will cancel any outstanding order.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from ..orderbook.events import SimEvent
from .base import SimulatorAPI, Strategy

Side = Literal["bid", "ask"]


class QuotingStrategy(Strategy):
    """Base class for two-sided makers with cancel-and-replace reconcile.

    Parameters
    ----------
    size:
        Order size (in contracts) submitted on each side.
    requote_kinds:
        Event kinds that trigger a re-quote check. Defaults to
        book / price_change / trade — the events that change the book
        state on which our quotes depend. Strategies that need to
        re-quote on every tick can override.
    """

    def __init__(
        self,
        *,
        size: float = 1.0,
        requote_kinds: tuple[str, ...] = ("book", "price_change", "trade"),
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.requote_kinds = requote_kinds
        self._bid_oid: int | None = None
        self._bid_price: float | None = None
        self._ask_oid: int | None = None
        self._ask_price: float | None = None

    @abstractmethod
    def desired_quotes(self, sim: SimulatorAPI) -> tuple[float | None, float | None]:
        """Return ``(bid_price, ask_price)``.

        Either may be ``None`` to withdraw that side.
        """

    def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
        if event.kind not in self.requote_kinds:
            return
        bid_target, ask_target = self.desired_quotes(sim)
        self._reconcile_side(sim, "bid", bid_target)
        self._reconcile_side(sim, "ask", ask_target)

    def _reconcile_side(self, sim: SimulatorAPI, side: Side, target: float | None) -> None:
        if side == "bid":
            cur_oid, cur_price = self._bid_oid, self._bid_price
        else:
            cur_oid, cur_price = self._ask_oid, self._ask_price

        tick = sim.book.tick
        same_price = (
            cur_price is not None and target is not None and abs(cur_price - target) < tick / 2
        )

        if cur_oid is not None and not same_price:
            sim.cancel(cur_oid)
            cur_oid, cur_price = None, None

        if target is not None and cur_oid is None:
            cur_oid = sim.place_limit(side, target, self.size)
            cur_price = target

        if side == "bid":
            self._bid_oid, self._bid_price = cur_oid, cur_price
        else:
            self._ask_oid, self._ask_price = cur_oid, cur_price


__all__ = ["QuotingStrategy"]
