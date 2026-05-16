"""Unit tests for hft_pm.simulator.engine.

Phase 2 acceptance:
* ``test_do_nothing_strategy_returns_zero_pnl``
* ``test_latency_injection_blocks_premature_fill``
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from hft_pm.data.schemas import (
    BookEvent,
    LastTradePriceEvent,
    PriceChangeEvent,
    PriceLevel,
    PriceLevelChange,
)
from hft_pm.fees.polymarket import FeeCategory
from hft_pm.orderbook.events import SimEvent
from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.strategies.base import DoNothing, SimulatorAPI, Strategy

ASSET = "12345"
MARKET = "0xdeadbeef"


def _book(
    ts_ms: int,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> BookEvent:
    return BookEvent(
        asset_id=ASSET,
        market=MARKET,
        timestamp_ms=ts_ms,
        recv_ts_ms=ts_ms,
        bids=[PriceLevel(price=p, size=s) for p, s in bids],
        asks=[PriceLevel(price=p, size=s) for p, s in asks],
    )


def _trade(ts_ms: int, price: float, size: float, side: str) -> LastTradePriceEvent:
    return LastTradePriceEvent(
        asset_id=ASSET,
        market=MARKET,
        timestamp_ms=ts_ms,
        recv_ts_ms=ts_ms,
        price=price,
        size=size,
        side=side,  # type: ignore[arg-type]
    )


def _price_change(ts_ms: int, changes: list[tuple[float, str, float]]) -> PriceChangeEvent:
    return PriceChangeEvent(
        asset_id=ASSET,
        market=MARKET,
        timestamp_ms=ts_ms,
        recv_ts_ms=ts_ms,
        changes=[
            PriceLevelChange(price=p, side=s, size=sz)  # type: ignore[arg-type]
            for p, s, sz in changes
        ],
    )


# ----------------------------------------------------------------------
# Acceptance: do-nothing strategy
# ----------------------------------------------------------------------


def test_do_nothing_strategy_returns_zero_pnl() -> None:
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(50))
    events = [
        _book(1000, bids=[(0.49, 100)], asks=[(0.51, 100)]),
        _trade(1100, 0.51, 50, "BUY"),
        _trade(1200, 0.49, 30, "SELL"),
        _price_change(1300, [(0.49, "BUY", 70), (0.51, "SELL", 50)]),
        _trade(1400, 0.49, 70, "SELL"),
    ]
    result = sim.run(events, DoNothing())
    assert result.pnl == 0.0
    assert result.n_fills == 0
    assert result.final_inventory == 0
    assert result.fees_paid == 0.0
    assert result.rebates_received == 0.0


# ----------------------------------------------------------------------
# Acceptance: latency injection
# ----------------------------------------------------------------------


class _PlaceOnceStrategy(Strategy):
    """Place a single bid the first time it sees an event, then nothing."""

    def __init__(self, side: str, price: float, size: float) -> None:
        self.side = side
        self.price = price
        self.size = size
        self.placed = False
        self.order_id: int | None = None

    def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
        if not self.placed:
            self.order_id = sim.place_limit(self.side, self.price, self.size)  # type: ignore[arg-type]
            self.placed = True


def test_latency_injection_blocks_premature_fill() -> None:
    """Strategy places at t=1000 with 100ms latency → arrival at t=1100.
    A trade at t=1050 must NOT fill us; a later trade at t=1200 must."""
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(100))
    events = [
        _book(1000, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        # Trade at our intended price, BEFORE our order arrives.
        _trade(1050, 0.50, 50, "SELL"),
        # After arrival (1100), another trade fills us.
        _trade(1200, 0.50, 50, "SELL"),
    ]
    strat = _PlaceOnceStrategy("bid", 0.50, 20)
    result = sim.run(events, strat)
    assert result.n_fills == 1, "exactly one fill expected (the late trade)"
    fill = result.fills[0]
    assert fill.timestamp_ms == 1200
    assert fill.size == 20
    assert fill.is_maker is True


def test_latency_injection_with_zero_latency_fills_immediately() -> None:
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    events = [
        _book(1000, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(1050, 0.50, 50, "SELL"),
    ]
    strat = _PlaceOnceStrategy("bid", 0.50, 20)
    result = sim.run(events, strat)
    assert result.n_fills == 1
    assert result.fills[0].size == 20


# ----------------------------------------------------------------------
# Sanity / mechanics
# ----------------------------------------------------------------------


def test_strategy_sees_book_after_event_applied() -> None:
    """Strategy callback must run AFTER the event is applied to the book."""
    seen: list[tuple[int, float | None]] = []

    class Spy(Strategy):
        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            mid = sim.book.mid()
            seen.append((event.timestamp_ms, mid))

    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    events = [
        _book(1000, bids=[(0.49, 100)], asks=[(0.51, 100)]),
        _book(1100, bids=[(0.50, 100)], asks=[(0.52, 100)]),
    ]
    sim.run(events, Spy())
    # The first call sees mid=0.50; the second sees mid=0.51 (post-update).
    assert seen == [(1000, 0.50), (1100, 0.51)]


def test_pnl_accounting_is_consistent_for_round_trip() -> None:
    """A maker fill on bid then maker fill on ask at higher price → positive PnL.

    Setup: empty bid level (no queue). Strategy places bid 0.50 size 10.
    A SELL aggressor at 0.50 size 10 fills us → inventory +10.
    Then strategy places ask 0.52 size 10. BUY aggressor at 0.52 → fills, inventory 0.
    Cash change: -0.50*10 + 0.52*10 = +0.20. PnL == 0.20.
    """
    sequence: Iterator[int] = iter(range(100))

    class TwoSided(Strategy):
        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            n = next(sequence)
            if n == 0:
                sim.place_limit("bid", 0.50, 10)
            elif sim.inventory == 10 and event.kind == "trade":
                sim.place_limit("ask", 0.52, 10)

    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    events = [
        _book(1000, bids=[(0.50, 0)], asks=[(0.52, 0)]),
        _trade(1100, 0.50, 10, "SELL"),  # fills our bid → inventory +10
        _trade(1300, 0.52, 10, "BUY"),  # fills our ask (placed in 1100 callback)
    ]
    result = sim.run(events, TwoSided())
    assert result.n_fills == 2
    assert result.final_inventory == 0
    # Cash: -0.50*10 + 0.52*10 = 0.20; rebates+fees zero in GEOPOLITICS.
    assert pytest.approx(result.cash) == 0.20
    assert pytest.approx(result.pnl) == 0.20


def test_taker_strategy_pays_fee_in_finance_category() -> None:
    """Strategy crosses spread (post_only=False) → taker fee, no rebate."""
    book = L2OrderBook(tick=0.01)
    sim = Backtester(
        book=book,
        latency=ConstantLatency(0),
        fee_category=FeeCategory.FINANCE,
    )

    class Taker(Strategy):
        placed = False

        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            if not self.placed and sim.book.best_ask() is not None:
                sim.place_limit("bid", 0.51, 10, post_only=False)
                self.placed = True

    events = [
        _book(1000, bids=[(0.49, 100)], asks=[(0.51, 100)]),
        _book(1100, bids=[(0.49, 100)], asks=[(0.51, 100)]),  # second event needed
    ]
    result = sim.run(events, Taker())
    assert result.n_taker_fills == 1
    assert result.fees_paid > 0
    assert result.rebates_received == 0


def test_cancel_before_arrival_prevents_order() -> None:
    """Strategy places then cancels before arrival; no order should rest."""

    class PlaceAndCancel(Strategy):
        done = False
        oid: int | None = None

        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            if not self.done and event.kind == "book":
                self.oid = sim.place_limit("bid", 0.50, 10)
                sim.cancel(self.oid)
                self.done = True

    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(50))
    events = [
        _book(1000, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(2000, 0.50, 50, "SELL"),  # would have filled us if order rested
    ]
    result = sim.run(events, PlaceAndCancel())
    assert result.n_fills == 0
    assert len(book.our_orders) == 0


def test_empty_event_stream_returns_zero_pnl() -> None:
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    result = sim.run([], DoNothing())
    assert result.pnl == 0.0
    assert result.n_fills == 0


def test_strategy_does_not_see_future_book_state() -> None:
    """No-look-ahead invariant.

    At each callback, ``sim.book`` reflects events with timestamp ≤ now_ms,
    never anything from the next event in the stream. The reviewer was
    concerned that pre-loading the heap with the next event leaks state;
    this test pins the invariant.
    """
    observed: list[tuple[int, float | None, float | None]] = []

    class Spy(Strategy):
        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            bb = sim.book.best_bid()
            ba = sim.book.best_ask()
            observed.append((event.timestamp_ms, bb[0] if bb else None, ba[0] if ba else None))

    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))
    events = [
        _book(1000, bids=[(0.40, 100)], asks=[(0.60, 100)]),
        _book(2000, bids=[(0.45, 100)], asks=[(0.55, 100)]),
        _book(3000, bids=[(0.50, 100)], asks=[(0.51, 100)]),
    ]
    sim.run(events, Spy())
    # At t=1000 the strategy must see (0.40, 0.60) — NOT the t=2000 quotes
    # even though the engine has already pushed the t=2000 event onto the heap.
    assert observed[0] == (1000, 0.40, 0.60)
    assert observed[1] == (2000, 0.45, 0.55)
    assert observed[2] == (3000, 0.50, 0.51)


def test_fractional_fill_size_does_not_drift_inventory() -> None:
    """Polymarket allows fractional book sizes (e.g. 115.89). Sequential
    fills of 0.5 contracts must accumulate exactly, not drift to zero
    via per-fill ``round()``.
    """
    book = L2OrderBook(tick=0.01)
    sim = Backtester(book=book, latency=ConstantLatency(0))

    class HalfShareScalper(Strategy):
        n_placed = 0

        def on_event(self, sim: SimulatorAPI, event: SimEvent) -> None:
            if self.n_placed < 4 and event.kind == "book":
                sim.place_limit("bid", 0.50, 0.5)
                self.n_placed += 1

    events = [
        _book(1000, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(1100, 0.50, 0.5, "SELL"),
        _book(1200, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(1300, 0.50, 0.5, "SELL"),
        _book(1400, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(1500, 0.50, 0.5, "SELL"),
        _book(1600, bids=[(0.50, 0)], asks=[(0.51, 100)]),
        _trade(1700, 0.50, 0.5, "SELL"),
    ]
    result = sim.run(events, HalfShareScalper())
    assert result.n_fills == 4
    # Four 0.5-fills must sum to exactly 2.0 (not 0 from banker's rounding).
    assert result.final_inventory == pytest.approx(2.0)
