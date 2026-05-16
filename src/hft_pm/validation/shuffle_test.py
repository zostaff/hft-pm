"""Timestamp-shuffle test (CLAUDE.md Phase 6 acceptance).

Permutes the timestamps of an event stream while leaving the payload
identifiers and event types intact. Under the null hypothesis that the
strategy is exploiting time-correlated microstructure, the shuffle
destroys the signal — Sharpe should collapse toward zero. If a
shuffled-Sharpe stays comparable to the in-order Sharpe, the strategy
is **not** exploiting microstructure; it's exploiting some calendar
artifact or feature-leakage, which is a bug.

Implementation: we collect every event into a list, generate a random
permutation of the timestamps, then re-attach them in order. The events
are then re-sorted by the new timestamps so the engine still receives
them monotonically. Payload contents (prices, sizes, sides) are not
changed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..data.schemas import (
    BookEvent,
    LastTradePriceEvent,
    MarketEvent,
    PriceChangeEvent,
    TickSizeChangeEvent,
)


def _with_timestamps(ev: MarketEvent, ts_ms: int) -> MarketEvent:
    """Rebuild ``ev`` with new ``timestamp_ms`` and ``recv_ts_ms``.

    The Pydantic models are frozen, so we use ``model_copy`` rather
    than mutating in place.
    """
    return ev.model_copy(update={"timestamp_ms": ts_ms, "recv_ts_ms": ts_ms})


def shuffle_event_timestamps(
    events: Iterable[MarketEvent],
    *,
    seed: int | None = None,
) -> list[MarketEvent]:
    """Return a list of the same events with shuffled timestamps.

    Reuses the original set of timestamps so the time density (overall
    event rate) is preserved; only the *association* between content
    and time is randomised.
    """
    evs = list(events)
    n = len(evs)
    if n == 0:
        return []
    timestamps = np.array([e.timestamp_ms for e in evs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(timestamps)
    new_events = [_with_timestamps(e, int(t)) for e, t in zip(evs, permuted, strict=True)]
    # Re-sort by new timestamps; tie-break by id() of original event for stability.
    new_events.sort(key=lambda e: e.timestamp_ms)
    return new_events


def _is_supported(ev: MarketEvent) -> bool:
    return isinstance(ev, BookEvent | PriceChangeEvent | LastTradePriceEvent | TickSizeChangeEvent)


__all__ = ["shuffle_event_timestamps"]
