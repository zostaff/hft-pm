"""PnL and PnL-series helpers for the simulator.

Phase 2 acceptance only needs total PnL and a per-event PnL series.
Sharpe, drawdown, and validation-grade metrics arrive in Phase 6.

The mark-to-mid convention: open inventory is valued at the current
mid price. If the book has no mid (one side empty), inventory is held
at the last observed mid; if no mid has ever been observed, inventory
PnL is reported as zero (acceptable for Phase 2 smoke tests).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..orderbook.events import FillRecord


@dataclass(frozen=True)
class PnLSnapshot:
    """One point on the PnL time series."""

    timestamp_ms: int
    cash: float
    inventory: float
    mark_price: float | None
    pnl: float


def realised_cash(fills: Iterable[FillRecord]) -> float:
    """Return cash change from a sequence of fills (signed).

    Maker rebates increase cash; taker fees decrease cash. Bid fills
    spend cash; ask fills receive cash.
    """
    cash = 0.0
    for f in fills:
        sign = -1 if f.side == "bid" else 1
        cash += sign * f.price * f.size
        cash += f.rebate_received
        cash -= f.fee_paid
    return cash


def inventory_from_fills(fills: Iterable[FillRecord]) -> float:
    """Net inventory in contracts. Bid fills increase, ask fills decrease."""
    net = 0.0
    for f in fills:
        sign = 1 if f.side == "bid" else -1
        net += sign * f.size
    return net


def mark_to_mid(cash: float, inventory: float, mark_price: float | None) -> float:
    """PnL = cash + inventory * mid. Falls back to ``cash`` if mid is unknown."""
    if mark_price is None:
        return cash
    return cash + inventory * mark_price


__all__ = ["PnLSnapshot", "inventory_from_fills", "mark_to_mid", "realised_cash"]
