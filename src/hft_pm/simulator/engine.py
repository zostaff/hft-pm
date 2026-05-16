"""Event-driven backtester (docs §10) with L2 queue tracking and V2 fees.

Pipeline per event:

1. Engine pops the next ``SimEvent`` off the min-heap (key:
   ``(timestamp_ms, seq)``).
2. The event is applied to the L2 book (or, for ``place_arrival`` /
   ``cancel_arrival``, to the our-orders state).
3. For ``trade`` events, the book's ``process_trade`` reports any of
   our orders that got hit; the engine builds :class:`FillRecord`\\ s,
   updates cash, inventory, and the PnL series.
4. The strategy callback runs *after* the state update.

Latency injection: when the strategy calls ``place_limit`` /
``cancel``, the engine asks the configured :class:`LatencyModel` for
an arrival timestamp and schedules a ``place_arrival`` /
``cancel_arrival`` event there. The order only becomes live on the
book at that moment.

post-only handling: by default ``place_limit(post_only=True)`` and the
book accepts the order even if it would cross — there is no observable
``post_only`` rejection in the simulator yet (that requires modelling
the matching engine's behaviour). If a strategy explicitly passes
``post_only=False`` and the price crosses the spread *at arrival*, the
engine fills it as a taker (immediate fill against the opposite top of
book, paying taker fee, no rebate).
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from ..data.schemas import (
    BookEvent,
    LastTradePriceEvent,
    MarketEvent,
    PriceChangeEvent,
    TickSizeChangeEvent,
)
from ..fees.polymarket import FeeCategory, maker_rebate, taker_fee
from ..orderbook.events import FillRecord, SimEvent
from ..orderbook.l2_book import L2OrderBook
from ..strategies.base import SimulatorAPI, Strategy
from .latency import LatencyModel
from .metrics import PnLSnapshot, mark_to_mid

Side = Literal["bid", "ask"]


@dataclass
class _PlacePayload:
    order_id: int
    side: Side
    price: float
    size: float
    post_only: bool


@dataclass
class _CancelPayload:
    order_id: int


@dataclass
class SimulationResult:
    """Output of :meth:`Backtester.run`."""

    pnl: float
    n_fills: int
    n_maker_fills: int
    n_taker_fills: int
    final_inventory: float
    cash: float
    fees_paid: float
    rebates_received: float
    fills: list[FillRecord] = field(default_factory=list)
    pnl_series: list[PnLSnapshot] = field(default_factory=list)


class Backtester(SimulatorAPI):
    """Event-driven simulator (docs §10).

    Parameters
    ----------
    book:
        :class:`L2OrderBook` instance. The simulator does not own it
        outright — passing the book in lets tests pre-seed levels.
    latency:
        Decides when strategy actions arrive at the book.
    fee_category:
        Polymarket fee tier; defaults to ``GEOPOLITICS`` (zero fees,
        zero rebates) so smoke tests of pure mechanics remain pure.
    record_pnl_series:
        If ``True``, push a :class:`PnLSnapshot` after every event.
        Off by default for performance on large replays.
    """

    def __init__(
        self,
        *,
        book: L2OrderBook,
        latency: LatencyModel,
        fee_category: FeeCategory = FeeCategory.GEOPOLITICS,
        record_pnl_series: bool = False,
    ) -> None:
        self._book = book
        self._latency = latency
        self._fee_category = fee_category
        self._record_pnl_series = record_pnl_series

        self._heap: list[SimEvent] = []
        self._seq_counter = itertools.count()
        self._now_ms: int = 0

        self._cash: float = 0.0
        self._inventory: float = 0.0
        self._fees_paid: float = 0.0
        self._rebates_received: float = 0.0
        self._fills: list[FillRecord] = []
        self._pnl_series: list[PnLSnapshot] = []
        self._last_mid: float | None = None

        self._next_order_id = itertools.count(start=1)
        # order_id -> (side, price, size, post_only, pending_decision_ts).
        # Lives between place_arrival and cancel/fill so cancels of a
        # not-yet-arrived order can be honoured.
        self._pending_places: dict[int, _PlacePayload] = {}

    # ------------------------------------------------------------------
    # SimulatorAPI surface
    # ------------------------------------------------------------------

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def inventory(self) -> float:
        return self._inventory

    @property
    def book(self) -> L2OrderBook:
        return self._book

    def place_limit(
        self,
        side: Side,
        price: float,
        size: float,
        *,
        post_only: bool = True,
    ) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        if side not in ("bid", "ask"):
            raise ValueError(f"invalid side: {side!r}")
        order_id = next(self._next_order_id)
        arrival_ms = self._latency.sample(self._now_ms)
        payload = _PlacePayload(order_id, side, price, size, post_only)
        self._pending_places[order_id] = payload
        self._push_internal(arrival_ms, "place_arrival", payload)
        return order_id

    def cancel(self, order_id: int) -> None:
        arrival_ms = self._latency.sample(self._now_ms)
        self._push_internal(arrival_ms, "cancel_arrival", _CancelPayload(order_id))

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------

    def _push_internal(self, timestamp_ms: int, kind: str, payload: object) -> None:
        heapq.heappush(
            self._heap,
            SimEvent(timestamp_ms, next(self._seq_counter), kind, payload),  # type: ignore[arg-type]
        )

    def _push_market(self, ev: MarketEvent) -> None:
        kind = _market_event_to_kind(ev)
        heapq.heappush(
            self._heap,
            SimEvent(ev.timestamp_ms, next(self._seq_counter), kind, ev),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, events: Iterable[MarketEvent], strategy: Strategy) -> SimulationResult:
        """Drive the strategy against ``events`` and return the result.

        ``events`` is consumed lazily — the caller may pass a
        :class:`hft_pm.data.replay.Replay` directly, which yields events
        in timestamp order. The heap merges them with internal arrivals.
        """
        events_iter: Iterator[MarketEvent] = iter(events)
        # Prime the heap with the first event so we have a notion of "now".
        first = next(events_iter, None)
        if first is None:
            return self._finalise()
        self._push_market(first)

        while self._heap:
            sim_ev = heapq.heappop(self._heap)
            self._now_ms = sim_ev.timestamp_ms

            self._apply(sim_ev)

            # Refill the heap with the next market event if available so
            # the strategy sees a consistent ordering.
            with contextlib.suppress(StopIteration):
                self._push_market(next(events_iter))

            strategy.on_event(self, sim_ev)

            if self._record_pnl_series:
                self._pnl_series.append(self._snapshot())

        return self._finalise()

    # ------------------------------------------------------------------
    # Event application
    # ------------------------------------------------------------------

    def _apply(self, sim_ev: SimEvent) -> None:
        kind = sim_ev.kind
        payload = sim_ev.payload

        if kind == "book":
            assert isinstance(payload, BookEvent)
            bids = [(lvl.price, lvl.size) for lvl in payload.bids]
            asks = [(lvl.price, lvl.size) for lvl in payload.asks]
            self._book.apply_book_snapshot(bids, asks)
            self._refresh_last_mid()

        elif kind == "price_change":
            assert isinstance(payload, PriceChangeEvent)
            for change in payload.changes:
                self._book.apply_price_change(change.side, change.price, change.size)
            self._refresh_last_mid()

        elif kind == "trade":
            assert isinstance(payload, LastTradePriceEvent)
            fills = self._book.process_trade(
                payload.price, payload.size, payload.side, payload.timestamp_ms
            )
            for oid, side, fill_size, qa_at_fill, time_in_book in fills:
                self._record_fill(
                    oid=oid,
                    side=side,
                    price=payload.price,
                    size=fill_size,
                    is_maker=True,
                    queue_ahead_at_fill=qa_at_fill,
                    time_in_book_ms=time_in_book,
                    ts_ms=payload.timestamp_ms,
                )
            self._refresh_last_mid()

        elif kind == "tick_size_change":
            assert isinstance(payload, TickSizeChangeEvent)
            self._book.tick = float(payload.new_tick_size)

        elif kind == "place_arrival":
            assert isinstance(payload, _PlacePayload)
            self._handle_place_arrival(payload)

        elif kind == "cancel_arrival":
            assert isinstance(payload, _CancelPayload)
            self._book.remove_our_order(payload.order_id)
            self._pending_places.pop(payload.order_id, None)

    def _handle_place_arrival(self, p: _PlacePayload) -> None:
        # The order may have been cancelled before arrival.
        if p.order_id not in self._pending_places:
            return
        del self._pending_places[p.order_id]

        if not p.post_only and _crosses_spread(self._book, p.side, p.price):
            # Immediate taker fill against the opposite top of book.
            opp = self._book.best_ask() if p.side == "bid" else self._book.best_bid()
            assert opp is not None  # crosses_spread guaranteed presence
            fill_price = opp[0]
            available = opp[1]
            fill_size = min(p.size, available)
            self._record_fill(
                oid=p.order_id,
                side=p.side,
                price=fill_price,
                size=fill_size,
                is_maker=False,
                queue_ahead_at_fill=0.0,
                time_in_book_ms=0,
                ts_ms=self._now_ms,
            )
            # Any remainder rests at the limit price as maker liquidity.
            remainder = p.size - fill_size
            if remainder > 0:
                self._book.add_our_order(p.order_id, p.side, p.price, remainder, self._now_ms)
            return

        self._book.add_our_order(p.order_id, p.side, p.price, p.size, self._now_ms)

    # ------------------------------------------------------------------
    # PnL accounting
    # ------------------------------------------------------------------

    def _record_fill(
        self,
        *,
        oid: int,
        side: Side,
        price: float,
        size: float,
        is_maker: bool,
        queue_ahead_at_fill: float,
        time_in_book_ms: int,
        ts_ms: int,
    ) -> None:
        if is_maker:
            fee = 0.0
            rebate = maker_rebate(price, size, self._fee_category)
        else:
            fee = taker_fee(price, size, self._fee_category)
            rebate = 0.0

        sign = 1 if side == "bid" else -1
        notional = price * size
        self._cash += -sign * notional + rebate - fee
        # Inventory is summed as float — Polymarket book sizes are
        # often fractional (e.g. 115.89, 30468.75). Rounding per-fill
        # would silently drift the sum under repeated half-contract
        # fills. We round once, at reporting, in :meth:`_finalise`.
        self._inventory += sign * size
        self._fees_paid += fee
        self._rebates_received += rebate

        self._fills.append(
            FillRecord(
                timestamp_ms=ts_ms,
                order_id=oid,
                side=side,
                price=price,
                size=size,
                is_maker=is_maker,
                fee_paid=fee,
                rebate_received=rebate,
                queue_ahead_at_fill=queue_ahead_at_fill,
                time_in_book_ms=time_in_book_ms,
            )
        )

    def _refresh_last_mid(self) -> None:
        mid = self._book.mid()
        if mid is not None:
            self._last_mid = mid

    def _snapshot(self) -> PnLSnapshot:
        return PnLSnapshot(
            timestamp_ms=self._now_ms,
            cash=self._cash,
            inventory=self._inventory,
            mark_price=self._last_mid,
            pnl=mark_to_mid(self._cash, self._inventory, self._last_mid),
        )

    def _finalise(self) -> SimulationResult:
        n_maker = sum(1 for f in self._fills if f.is_maker)
        return SimulationResult(
            pnl=mark_to_mid(self._cash, self._inventory, self._last_mid),
            n_fills=len(self._fills),
            n_maker_fills=n_maker,
            n_taker_fills=len(self._fills) - n_maker,
            final_inventory=self._inventory,
            cash=self._cash,
            fees_paid=self._fees_paid,
            rebates_received=self._rebates_received,
            fills=list(self._fills),
            pnl_series=list(self._pnl_series),
        )


def _market_event_to_kind(ev: MarketEvent) -> str:
    if isinstance(ev, BookEvent):
        return "book"
    if isinstance(ev, PriceChangeEvent):
        return "price_change"
    if isinstance(ev, LastTradePriceEvent):
        return "trade"
    if isinstance(ev, TickSizeChangeEvent):
        return "tick_size_change"
    raise TypeError(f"unsupported market event type: {type(ev).__name__}")


def _crosses_spread(book: L2OrderBook, side: Side, price: float) -> bool:
    if side == "bid":
        best_ask = book.best_ask()
        return best_ask is not None and price >= best_ask[0]
    best_bid = book.best_bid()
    return best_bid is not None and price <= best_bid[0]


__all__ = ["Backtester", "SimulationResult"]
