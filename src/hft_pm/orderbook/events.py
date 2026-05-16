"""Internal event types for the event-driven simulator.

The :class:`MarketEvent` family in :mod:`hft_pm.data.schemas` is what the
``Replay`` yields. Inside the simulator we also need to schedule internal
actions (order arrival, cancellation arrival) on the same heap as the
market events. :class:`SimEvent` is that union: it carries a timestamp,
a stable tiebreaker sequence, a discriminator, and a typed payload.

Fill records are reported back to the strategy via the result object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SimEventKind = Literal[
    # External (replayed from the captured market channel)
    "book",
    "price_change",
    "trade",
    "tick_size_change",
    # Internal (scheduled by the engine when the strategy calls place/cancel)
    "place_arrival",
    "cancel_arrival",
]


@dataclass(order=True, frozen=True)
class SimEvent:
    """Heap-sortable simulator event.

    ``timestamp_ms`` is the primary sort key; ``seq`` is a stable tiebreaker
    assigned by the engine in the order events were pushed. ``kind`` and
    ``payload`` are not part of the comparison (see ``field(compare=False)``).
    """

    timestamp_ms: int
    seq: int
    kind: SimEventKind = field(compare=False)
    payload: Any = field(compare=False, default=None)


@dataclass(frozen=True)
class FillRecord:
    """One realised fill, returned in :class:`SimulationResult.fills`."""

    timestamp_ms: int
    order_id: int
    side: Literal["bid", "ask"]
    price: float
    size: float
    is_maker: bool
    fee_paid: float
    rebate_received: float
    queue_ahead_at_fill: float
    time_in_book_ms: int


__all__ = ["FillRecord", "SimEvent", "SimEventKind"]
