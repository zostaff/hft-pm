"""Polymarket V2 fee + rebate calculations (docs §7.6, post-March 2026).

Taker fees are category-dependent and probability-dependent:

.. math::

    \\text{fee}(p) = F_\\text{cat} \\cdot \\text{contracts} \\cdot 4 \\cdot p (1-p)

The ``x 4`` normalises so that the peak fee at :math:`p = 0.5` equals
``F_cat x size``. The fee vanishes at :math:`p = 0` and :math:`p = 1`.

Maker rebates are a fixed share of the same per-fill taker fee — the
counterparty's fee is the source of liquidity providers' rebate.

The ``GEOPOLITICS`` category is fee-free (and therefore rebate-free)
because Polymarket reserved that category for world-events markets.
``OTHER`` is the safe default for unknown categories: zero fees, zero
rebates.
"""

from __future__ import annotations

from enum import Enum


class FeeCategory(Enum):
    """Polymarket fee categories (peak rate F, rebate share R).

    Values are ``(category_name, peak_taker_rate, maker_rebate_share)``.
    """

    CRYPTO = ("crypto", 0.0180, 0.20)
    ECONOMICS = ("economics", 0.0150, 0.25)
    MENTIONS = ("mentions", 0.0156, 0.25)
    CULTURE = ("culture", 0.0125, 0.25)
    WEATHER = ("weather", 0.0125, 0.25)
    FINANCE = ("finance", 0.0100, 0.50)
    POLITICS = ("politics", 0.0100, 0.25)
    TECH = ("tech", 0.0100, 0.25)
    SPORTS = ("sports", 0.0075, 0.25)
    GEOPOLITICS = ("geopolitics", 0.0, 0.0)
    OTHER = ("other", 0.0, 0.0)

    def __init__(self, cat_name: str, peak_rate: float, rebate_share: float) -> None:
        self.cat_name = cat_name
        self.peak_rate = peak_rate
        self.rebate_share = rebate_share


def taker_fee(price: float, size: float, category: FeeCategory) -> float:
    """Dollar taker fee for a fill at ``price`` x ``size`` in ``category``.

    Symmetric around :math:`p = 0.5`, peaks at the midpoint, zero at both
    boundaries.
    """
    if category.peak_rate == 0:
        return 0.0
    # Bound to [0, 1] to guard against malformed inputs; ``p(1-p) * 4``
    # peaks at 1 at p=0.5 and is zero at p=0 and p=1.
    p = max(0.0, min(1.0, price))
    return category.peak_rate * size * p * (1.0 - p) * 4.0


def maker_rebate(price: float, size: float, category: FeeCategory) -> float:
    """Dollar maker rebate per fill — a fixed share of the taker fee."""
    return taker_fee(price, size, category) * category.rebate_share


def effective_half_spread_with_rebate(
    half_spread: float,
    price: float,
    category: FeeCategory,
) -> float:
    """Half-spread plus per-unit maker rebate (docs §7.6, §4.8).

    Both terms are expressed in dollars per unit of size, so this is the
    effective edge the maker captures per share at ``price``.
    """
    return half_spread + maker_rebate(price, 1.0, category)


__all__ = [
    "FeeCategory",
    "effective_half_spread_with_rebate",
    "maker_rebate",
    "taker_fee",
]
