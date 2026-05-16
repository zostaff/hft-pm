"""Order Flow Imbalance (Cont-Kukanov-Stoikov 2014, docs §8.1).

For each book event, a signed scalar ``e_n`` measures one-sided pressure::

    e_bid =   bid_size           if bid_price went up
            − prev_bid_size      if bid_price went down
            (bid_size − prev)    if bid_price unchanged

    e_ask = − ask_size           if ask_price went down
            + prev_ask_size      if ask_price went up
            − (ask_size − prev)  if ask_price unchanged

    e_n = e_bid + e_ask

Positive ``e_n`` → buying pressure (bid built or ask shrank). The
running sum over a rolling time window is OFI; empirically OFI predicts
the next short-horizon Δ-logit-mid with a small but persistent slope
(see ``calibrate_ofi_alpha`` in :mod:`hft_pm.signals.calibration`).
"""

from __future__ import annotations

from collections import deque


class OFICalculator:
    """Rolling-window OFI accumulator.

    Parameters
    ----------
    window_seconds:
        Length of the trailing time window in seconds. Older events are
        evicted on each ``update`` call. Typical value: 1–5 seconds.
    """

    def __init__(self, window_seconds: float = 1.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window = float(window_seconds)
        self._events: deque[tuple[float, float]] = deque()  # (ts_s, e_n)
        self._running_sum: float = 0.0
        self._prev_bid_px: float | None = None
        self._prev_bid_sz: float | None = None
        self._prev_ask_px: float | None = None
        self._prev_ask_sz: float | None = None

    def update(
        self,
        ts_seconds: float,
        bid_px: float,
        bid_sz: float,
        ask_px: float,
        ask_sz: float,
    ) -> float:
        """Apply one BBO observation and return the current rolling OFI."""
        if self._prev_bid_px is None:
            self._prev_bid_px, self._prev_bid_sz = bid_px, bid_sz
            self._prev_ask_px, self._prev_ask_sz = ask_px, ask_sz
            return 0.0

        if bid_px > self._prev_bid_px:
            e_bid = bid_sz
        elif bid_px < self._prev_bid_px:
            e_bid = -self._prev_bid_sz
        else:
            e_bid = bid_sz - self._prev_bid_sz

        if ask_px < self._prev_ask_px:
            e_ask = -ask_sz
        elif ask_px > self._prev_ask_px:
            e_ask = self._prev_ask_sz
        else:
            e_ask = -(ask_sz - self._prev_ask_sz)

        e_n = e_bid + e_ask
        self._events.append((ts_seconds, e_n))
        self._running_sum += e_n
        self._evict(ts_seconds)

        self._prev_bid_px, self._prev_bid_sz = bid_px, bid_sz
        self._prev_ask_px, self._prev_ask_sz = ask_px, ask_sz
        return self._running_sum

    def value(self) -> float:
        """Current rolling OFI without applying a new event."""
        return self._running_sum

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._events and self._events[0][0] < cutoff:
            _, e = self._events.popleft()
            self._running_sum -= e


__all__ = ["OFICalculator"]
