"""Full L2 order book with our resting-order queue tracking (docs §8.6).

Two state spaces live side by side here:

* **The public book.** ``self.bids`` and ``self.asks`` mirror exactly what
  the Polymarket WebSocket feed publishes — total resting size per level,
  *excluding* our own orders. We never mutate these from our own
  placements; only the public-feed appliers
  (:meth:`apply_book_snapshot`, :meth:`apply_price_change`) touch them.
* **Our orders.** ``self.our_orders`` indexes our resting orders by id and
  remembers, at the moment we joined the queue, how much public size was
  ahead of us. ``self.level_size_at_placement`` snapshots the public size
  at that moment so :meth:`process_trade` can estimate how much of the
  queue in front of us has been consumed since.

The queue-front consumption estimator follows §8.6::

    consumed_from_front = level_size_at_placement - current_public_level_size
    queue_remaining     = max(0, queue_ahead_at_placement - consumed_from_front)

When a trade fills our order, we update our internal queue/level snapshots
so subsequent trades — which may arrive before the next ``price_change``
on the same level — see consistent state.
"""

from __future__ import annotations

from typing import Literal

from sortedcontainers import SortedDict

from .events import FillRecord

Side = Literal["bid", "ask"]
AggressorSide = Literal["BUY", "SELL"]


class L2OrderBook:
    """Two-sided L2 book + our-order queue tracker (docs §8.6).

    Parameters
    ----------
    tick:
        Tick size. Used only for "same price level" comparisons in
        :meth:`process_trade`. Defaults to 0.01 (Polymarket default).
    """

    def __init__(self, tick: float = 0.01) -> None:
        if tick <= 0:
            raise ValueError("tick must be positive")
        self.tick = tick
        # Per-side sorted books: price -> public (theirs) size at the level.
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()
        # Our resting orders, indexed by order id.
        # order_id -> (side, price, size, queue_ahead_at_placement)
        self.our_orders: dict[int, tuple[Side, float, float, float]] = {}
        # Public level size at the moment we placed, per order id.
        self.level_size_at_placement: dict[int, float] = {}
        # Placement timestamps so the engine can compute time-in-book.
        self.placed_at_ms: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public-feed appliers
    # ------------------------------------------------------------------

    def apply_book_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Replace the public book from a WebSocket ``book`` snapshot.

        Zero-size levels are filtered out. Our resting-order state is
        preserved — queue estimates rebuild against the new public sizes
        on subsequent trades.
        """
        self.bids = SortedDict({px: sz for px, sz in bids if sz > 0})
        self.asks = SortedDict({px: sz for px, sz in asks if sz > 0})

    def apply_price_change(self, side: AggressorSide, price: float, new_size: float) -> None:
        """Apply one entry from a ``price_change`` event.

        ``side`` follows Polymarket's wire convention: ``"BUY"`` updates
        the bid book, ``"SELL"`` updates the ask book.
        """
        book = self.bids if side == "BUY" else self.asks
        if new_size <= 0:
            book.pop(price, None)
        else:
            book[price] = new_size

    # ------------------------------------------------------------------
    # Top of book
    # ------------------------------------------------------------------

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        px = self.bids.keys()[-1]
        return px, self.bids[px]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        px = self.asks.keys()[0]
        return px, self.asks[px]

    def mid(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2.0

    # ------------------------------------------------------------------
    # Our orders
    # ------------------------------------------------------------------

    def add_our_order(
        self,
        order_id: int,
        side: Side,
        price: float,
        size: float,
        placed_at_ms: int,
    ) -> None:
        """Register that *we* just placed an order at this level.

        Does **not** mutate the public book. The public size at the level
        becomes our ``queue_ahead_at_placement``.
        """
        if size <= 0:
            raise ValueError("order size must be positive")
        if order_id in self.our_orders:
            raise ValueError(f"duplicate order_id: {order_id}")
        book = self.bids if side == "bid" else self.asks
        queue_ahead = book.get(price, 0.0)
        self.our_orders[order_id] = (side, price, size, queue_ahead)
        self.level_size_at_placement[order_id] = queue_ahead
        self.placed_at_ms[order_id] = placed_at_ms

    def remove_our_order(self, order_id: int) -> None:
        """Cancel-side removal. Does not mutate the public book."""
        self.our_orders.pop(order_id, None)
        self.level_size_at_placement.pop(order_id, None)
        self.placed_at_ms.pop(order_id, None)

    # ------------------------------------------------------------------
    # Trades against the book
    # ------------------------------------------------------------------

    def process_trade(
        self,
        trade_price: float,
        trade_size: float,
        aggressor: AggressorSide,
        trade_ts_ms: int,
    ) -> list[tuple[int, Side, float, float, int]]:
        """Walk our resting orders on the side hit by ``aggressor``.

        Returns ``(order_id, side, fill_size, queue_ahead_at_fill,
        time_in_book_ms)`` tuples for each of our orders that got a
        fill. The engine wraps these in :class:`FillRecord`\\ s with
        fee/rebate attached.

        Queue accounting: ``trade_size`` is consumed first from
        ``queue_remaining`` (volume still in front of us); whatever is
        left fills our order, up to its remaining size. When multiple
        of our orders sit at the same level, the volume eaten by
        earlier orders within this single ``process_trade`` call must
        also count as "consumed in front" for later orders — tracked
        via ``consumed_at_price``.
        """
        our_side: Side = "ask" if aggressor == "BUY" else "bid"
        fills: list[tuple[int, Side, float, float, int]] = []
        remaining_trade = trade_size
        # Maps price level -> volume consumed from this trade so far at
        # that level. Lets multiple orders at the same level account
        # for each other's queue consumption within one trade.
        consumed_at_price: dict[float, float] = {}
        for oid, (side, price, size, qa_initial) in list(self.our_orders.items()):
            if side != our_side:
                continue
            if side == "bid" and price < trade_price:
                continue
            if side == "ask" and price > trade_price:
                continue
            if not _same_tick_level(price, trade_price, self.tick):
                continue

            book = self.bids if side == "bid" else self.asks
            current_public = book.get(price, 0.0)
            initial_public = self.level_size_at_placement[oid]
            real_consumed = max(0.0, initial_public - current_public)
            in_loop_consumed = consumed_at_price.get(price, 0.0)
            consumed_from_front = real_consumed + in_loop_consumed
            queue_remaining = max(0.0, qa_initial - consumed_from_front)

            if remaining_trade <= queue_remaining:
                new_qa = max(0.0, qa_initial - consumed_from_front - remaining_trade)
                self.our_orders[oid] = (side, price, size, new_qa)
                self.level_size_at_placement[oid] = current_public
                consumed_at_price[price] = in_loop_consumed + remaining_trade
                remaining_trade = 0.0
                break

            remaining_trade -= queue_remaining
            fill_size = min(remaining_trade, size)
            remaining_trade -= fill_size
            new_size = size - fill_size
            time_in_book_ms = trade_ts_ms - self.placed_at_ms.get(oid, trade_ts_ms)
            fills.append((oid, side, fill_size, queue_remaining, time_in_book_ms))
            consumed_at_price[price] = in_loop_consumed + queue_remaining + fill_size
            if new_size <= 0:
                self.our_orders.pop(oid, None)
                self.level_size_at_placement.pop(oid, None)
                self.placed_at_ms.pop(oid, None)
            else:
                self.our_orders[oid] = (side, price, new_size, 0.0)
                self.level_size_at_placement[oid] = current_public
            if remaining_trade <= 0:
                break
        return fills

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def queue_position(self, order_id: int) -> float | None:
        """Estimate volume currently ahead of our order at its level.

        Returns ``None`` if the order is not tracked.
        """
        info = self.our_orders.get(order_id)
        if info is None:
            return None
        side, price, _size, qa_initial = info
        book = self.bids if side == "bid" else self.asks
        current_public = book.get(price, 0.0)
        initial_public = self.level_size_at_placement[order_id]
        consumed = max(0.0, initial_public - current_public)
        return max(0.0, qa_initial - consumed)


def _same_tick_level(a: float, b: float, tick: float) -> bool:
    """Return True iff ``a`` and ``b`` snap to the same tick index.

    Compares by ``round(price / tick)`` so floats with sub-tick noise
    (e.g. 0.504 and 0.505 on a 0.01-tick book) cannot be confused.
    """
    return round(a / tick) == round(b / tick)


__all__ = ["FillRecord", "L2OrderBook"]
