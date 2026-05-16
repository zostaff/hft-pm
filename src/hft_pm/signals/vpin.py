"""VPIN — volume-bucketed adverse-selection toxicity (docs §8.3).

Standard VPIN (Easley, López de Prado, O'Hara 2012) summed in equal-volume
buckets; normalised for prediction markets to account for the binary
payoff geometry::

    VPIN_PM = (1/N) Σ |V_buy − V_sell| / (√(p(1−p)) · (V_buy + V_sell))

The denominator's ``√(p(1−p))`` term shrinks the contribution of buckets
near the {0, 1} boundary, where the absolute imbalance is small in
*payoff* terms even when the count imbalance is large.

When VPIN is high, the maker is being adversely selected (one side keeps
hitting). The strategy should widen or withdraw — gated typically at
VPIN > 0.4 in our acceptance test, calibrated from real markets.
"""

from __future__ import annotations

import math
from collections import deque


class VPINCalculator:
    """Volume-bucketed VPIN with PM normalisation.

    Parameters
    ----------
    bucket_volume:
        Total (buy + sell) volume that closes one bucket. Tune to your
        market: e.g. 1000 contracts on a busy Polymarket leg gives ~5–20
        buckets per hour.
    n_buckets:
        Number of trailing buckets kept; older buckets are evicted.
    """

    def __init__(self, bucket_volume: float, n_buckets: int = 50) -> None:
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        self.bucket_volume = float(bucket_volume)
        self.n_buckets = int(n_buckets)
        self._buckets: deque[tuple[float, float, float]] = deque()  # (buy, sell, p_mean)
        self._cur_buy: float = 0.0
        self._cur_sell: float = 0.0
        self._cur_p_sum: float = 0.0
        self._cur_n: int = 0

    def add_trade(self, volume: float, is_buy: bool, price: float) -> None:
        """Accumulate one trade into the current bucket."""
        if volume <= 0:
            return
        if is_buy:
            self._cur_buy += volume
        else:
            self._cur_sell += volume
        self._cur_p_sum += price
        self._cur_n += 1

        while self._cur_buy + self._cur_sell >= self.bucket_volume:
            p_mean = self._cur_p_sum / max(self._cur_n, 1)
            self._buckets.append((self._cur_buy, self._cur_sell, p_mean))
            if len(self._buckets) > self.n_buckets:
                self._buckets.popleft()
            # Carry residual proportionally so a single huge trade does
            # not all land in one bucket.
            excess = self._cur_buy + self._cur_sell - self.bucket_volume
            total_now = self._cur_buy + self._cur_sell
            sell_frac = self._cur_sell / total_now if total_now > 0 else 0.5
            self._cur_sell = excess * sell_frac
            self._cur_buy = excess * (1.0 - sell_frac)
            if excess > 0:
                # Residual volume continues into the next bucket. Seed
                # the running price tracker with the just-closed bucket's
                # mean so the next bucket's p_mean is well-defined.
                self._cur_p_sum = p_mean
                self._cur_n = 1
            else:
                # Bucket drained exactly; start fresh.
                self._cur_p_sum = 0.0
                self._cur_n = 0

    def value(self) -> float:
        """Current PM-normalised VPIN in roughly [0, 1].

        Returns 0 when fewer than one bucket has closed.
        """
        if not self._buckets:
            return 0.0
        total = 0.0
        for buy, sell, p in self._buckets:
            p_clamped = max(1e-6, min(1.0 - 1e-6, p))
            denom = math.sqrt(p_clamped * (1.0 - p_clamped)) * (buy + sell)
            if denom > 0:
                total += abs(buy - sell) / denom
        return total / len(self._buckets)

    @property
    def n_closed_buckets(self) -> int:
        return len(self._buckets)


__all__ = ["VPINCalculator"]
