"""Live and paper-trading runners.

* :class:`PaperTrader` — Tier 2 runner. Subscribes to the live Polymarket
  WS feed, drives a :class:`~hft_pm.strategies.base.Strategy` against a
  local :class:`~hft_pm.orderbook.l2_book.L2OrderBook`, and simulates
  fills locally. No orders are sent to Polymarket.

The Polymarket V2 SDK wrapper (``client_v2``) for Tier 3 live trading
arrives separately and is not imported here.
"""

from .paper_trade import PaperTrader

__all__ = ["PaperTrader"]
