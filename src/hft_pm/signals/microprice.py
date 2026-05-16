"""Stoikov (2018) microprice — imbalance-weighted fair value (docs §8.2).

The microprice weights each side of the BBO by the **opposite** side's
size, so a bid-heavy book pulls the fair-value estimate toward the ask::

    microprice = (V_ask · p_bid + V_bid · p_ask) / (V_bid + V_ask)

Intuition: a deep bid means buyers are queueing up — the next print is
more likely to take the ask. The microprice is a martingale-by-construction
estimate of the next trade price.

When both sides are zero-size (degenerate book), we fall back to the
arithmetic midpoint so the function never returns NaN.
"""

from __future__ import annotations


def microprice(
    bid_px: float,
    bid_sz: float,
    ask_px: float,
    ask_sz: float,
) -> float:
    """Stoikov's first-iteration imbalance-weighted mid.

    Parameters
    ----------
    bid_px, bid_sz, ask_px, ask_sz:
        Top of book, both sides.

    Returns
    -------
    Microprice in the same units as the input prices. Falls back to
    arithmetic mid if total size is zero.
    """
    total = bid_sz + ask_sz
    if total <= 0:
        return (bid_px + ask_px) / 2.0
    return (ask_sz * bid_px + bid_sz * ask_px) / total


def imbalance(bid_sz: float, ask_sz: float) -> float:
    """Signed top-of-book imbalance ``(V_b − V_a) / (V_b + V_a)``.

    Range [−1, 1]. Positive → bid-heavy → microprice pulls toward ask.
    Returns 0 when both sides are zero (degenerate book).
    """
    total = bid_sz + ask_sz
    if total <= 0:
        return 0.0
    return (bid_sz - ask_sz) / total


__all__ = ["imbalance", "microprice"]
