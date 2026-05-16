"""Strategy ABC and the narrow API the simulator exposes to strategies.

CLAUDE.md rule #1 — "every feature uses only data with timestamp < decision
timestamp" — is enforced structurally here: strategies receive each event
*after* the engine has applied it to the book, and they cannot reach into
the engine's heap or future state. The :class:`SimulatorAPI` Protocol
declares exactly what they may touch.

Place/cancel return order ids (ints). Strategies hold those ids and pass
them back to ``cancel``. There is no live ``Order`` object to mutate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol

from ..orderbook.events import SimEvent
from ..orderbook.l2_book import L2OrderBook

Side = Literal["bid", "ask"]


class SimulatorAPI(Protocol):
    """The minimal slice of the engine that strategies are allowed to use."""

    @property
    def now_ms(self) -> int: ...
    @property
    def inventory(self) -> float: ...
    @property
    def book(self) -> L2OrderBook: ...

    def place_limit(
        self,
        side: Side,
        price: float,
        size: float,
        *,
        post_only: bool = True,
    ) -> int:
        """Submit a limit order; return the assigned order id."""

    def cancel(self, order_id: int) -> None:
        """Submit a cancellation. Silently ignored if the id is unknown."""


class Strategy(ABC):
    """Base class for backtest strategies.

    Subclasses override :meth:`on_event`. The engine calls it once per
    event in heap order, *after* applying the event to the book.
    Strategies typically inspect ``sim.book``, possibly call
    ``sim.place_limit`` or ``sim.cancel``, and return.
    """

    @abstractmethod
    def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None: ...


class DoNothing(Strategy):
    """Reference strategy. Never places, cancels, or holds inventory.

    Used by the Phase 2 acceptance test ``do_nothing_returns_zero_pnl``.
    """

    def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
        return None


__all__ = ["DoNothing", "SimulatorAPI", "Strategy"]
