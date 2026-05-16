"""Unit tests for hft_pm.orderbook.l2_book."""

from __future__ import annotations

import pytest

from hft_pm.orderbook.l2_book import L2OrderBook


def test_apply_book_snapshot_populates_best_bid_ask() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(
        bids=[(0.49, 100), (0.48, 200), (0.50, 50)],
        asks=[(0.51, 80), (0.52, 150)],
    )
    assert book.best_bid() == (0.50, 50)
    assert book.best_ask() == (0.51, 80)
    assert book.mid() == 0.505


def test_zero_size_levels_dropped() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(
        bids=[(0.49, 100), (0.48, 0)],
        asks=[(0.51, 80), (0.52, 0)],
    )
    assert 0.48 not in book.bids
    assert 0.52 not in book.asks


def test_apply_price_change_buy_updates_bid_side() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.apply_price_change("BUY", 0.50, 60)
    assert book.bids[0.50] == 60
    book.apply_price_change("BUY", 0.50, 0)
    assert 0.50 not in book.bids


def test_apply_price_change_sell_updates_ask_side() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.apply_price_change("SELL", 0.51, 40)
    assert book.asks[0.51] == 40


def test_mid_none_when_one_side_empty() -> None:
    book = L2OrderBook()
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[])
    assert book.mid() is None


def test_add_our_order_records_queue_ahead() -> None:
    """Our order joins behind the existing public size at the level."""
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 200)], asks=[(0.51, 100)])
    book.add_our_order(order_id=1, side="bid", price=0.50, size=50, placed_at_ms=0)
    assert book.queue_position(1) == 200
    # Public size on the level was not mutated.
    assert book.bids[0.50] == 200


def test_add_our_order_duplicate_oid_raises() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.50, 10, 0)
    with pytest.raises(ValueError):
        book.add_our_order(1, "bid", 0.50, 5, 0)


def test_add_our_order_negative_size_raises() -> None:
    book = L2OrderBook(tick=0.01)
    with pytest.raises(ValueError):
        book.add_our_order(1, "bid", 0.50, 0, 0)


def test_remove_our_order_idempotent() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.50, 50, 0)
    book.remove_our_order(1)
    book.remove_our_order(1)  # no raise on second call
    assert book.queue_position(1) is None


def test_process_trade_no_fill_when_queue_ahead_absorbs_all() -> None:
    """Our order at 0.50, queue ahead 200. A 150-size sell aggressor at
    0.50 only consumes part of the queue in front of us — no fill."""
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 200)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.50, 50, placed_at_ms=0)
    fills = book.process_trade(0.50, 150, "SELL", trade_ts_ms=10)
    assert fills == []
    # Our queue position advances by 150.
    qp = book.queue_position(1)
    assert qp is not None and qp == 50


def test_process_trade_fills_after_queue_consumed() -> None:
    """Same setup; the next 100-size trade should fill 50 (queue 50 then us 50)."""
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 200)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.50, 50, placed_at_ms=0)
    # First trade: 150 — all queue ahead.
    book.process_trade(0.50, 150, "SELL", trade_ts_ms=10)
    # Simulate the public book shrinking to reflect the 150 that traded.
    book.apply_price_change("BUY", 0.50, 50)
    # Second trade: 100. 50 in front, 50 fills us.
    fills = book.process_trade(0.50, 100, "SELL", trade_ts_ms=20)
    assert len(fills) == 1
    oid, side, fill_size, _qa_at_fill, time_in_book = fills[0]
    assert oid == 1
    assert side == "bid"
    assert fill_size == 50
    assert time_in_book == 20  # 20 - 0


def test_process_trade_partial_fill() -> None:
    """Trade size larger than (queue + our size) only fills our size."""
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 0)], asks=[(0.51, 100)])
    # Queue ahead of us is 0; our order is at the front.
    book.bids[0.50] = 0  # explicit
    book.add_our_order(1, "bid", 0.50, 20, placed_at_ms=0)
    fills = book.process_trade(0.50, 100, "SELL", trade_ts_ms=5)
    assert len(fills) == 1
    _, _, fill_size, qa, _ = fills[0]
    assert fill_size == 20
    assert qa == 0
    # Order should be removed entirely.
    assert book.queue_position(1) is None


def test_process_trade_skips_different_level() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.49, 100), (0.50, 100)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.49, 10, 0)
    # A trade at 0.50 should not touch our 0.49 order.
    fills = book.process_trade(0.50, 200, "SELL", trade_ts_ms=5)
    assert fills == []


def test_process_trade_skips_wrong_side() -> None:
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.add_our_order(1, "ask", 0.51, 10, 0)
    # A sell aggressor hits the bid side, not our ask.
    fills = book.process_trade(0.50, 50, "SELL", trade_ts_ms=5)
    assert fills == []


def test_constructor_rejects_zero_or_negative_tick() -> None:
    with pytest.raises(ValueError):
        L2OrderBook(tick=0)
    with pytest.raises(ValueError):
        L2OrderBook(tick=-0.01)


def test_process_trade_two_orders_at_same_level_share_queue_consumption() -> None:
    """Regression for the queue-accounting bug surfaced by code review.

    Setup: two of our orders A (size 5) and B (size 3) sit at level 0.50,
    behind 100 of public liquidity (placed in that order).
    A trade of 200 hits the level. It should consume 100 (queue), then
    fill A entirely (5), then fill B entirely (3), leaving 92 of trade.

    Before the fix, B's queue_remaining was computed against the original
    public size (100) rather than the post-fill state, so B saw queue=100
    and was mistakenly skipped.
    """
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.50, 100)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.50, 5, placed_at_ms=0)
    book.add_our_order(2, "bid", 0.50, 3, placed_at_ms=1)
    fills = book.process_trade(0.50, 200, "SELL", trade_ts_ms=10)
    fills_by_oid = {oid: (size, qa) for oid, _, size, qa, _ in fills}
    assert 1 in fills_by_oid, "order A must be filled"
    assert 2 in fills_by_oid, "order B must be filled (was skipped pre-fix)"
    assert fills_by_oid[1][0] == 5
    assert fills_by_oid[2][0] == 3
    # A had 100 queue ahead at fill; B had effectively 0 (after A).
    assert fills_by_oid[1][1] == 100
    assert fills_by_oid[2][1] == 0


def test_process_trade_uses_tick_index_not_float_distance() -> None:
    """Regression for the tick-comparison bug surfaced by code review.

    With tick=0.01 and our order at 0.49, a trade reported at 0.495
    must NOT match — they're different tick indices (49 vs 50). The
    old ``abs(price - trade_price) > tick / 2`` would incorrectly admit
    0.495 as "same level" because the float distance is exactly tick/2.
    """
    book = L2OrderBook(tick=0.01)
    book.apply_book_snapshot(bids=[(0.49, 0)], asks=[(0.51, 100)])
    book.add_our_order(1, "bid", 0.49, 10, placed_at_ms=0)
    fills = book.process_trade(0.495, 50, "SELL", trade_ts_ms=10)
    assert fills == [], "0.495 and 0.49 must not be treated as same level"
