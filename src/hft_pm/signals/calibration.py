"""Parameter calibration for the AS / GLT family (docs §4, §6.3, §8.5).

Three estimators ship in Phase 3:

* :func:`estimate_sigma` — annualised mid-price volatility from a series
  of (timestamp_ms, mid) samples. Standard Brownian-motion estimator
  σ̂ = std(Δlog S) / √Δt. Returned in *price units per √second* so
  downstream AS / GLT can consume it directly.

* :func:`estimate_arrival_rate` — Poisson rate λ̂ = N / T from event
  timestamps. Used for the AS ``A`` parameter (per-side trade arrival
  intensity at best price).

* :func:`estimate_kappa` — exponential decay of fill intensity in
  depth from mid (Cont-Kukanov-Stoikov 2014, §8.5). Given a sample of
  fill depths δ_i, the MLE for λ(δ) = A e^{−κδ} reduces to
  κ̂ = 1 / mean(δ_i). Returns κ in 1/price-units.

These are deliberately simple. Phase 6 will add Hawkes (§6.3) and
proper L2 calibration; here we want defensible estimates for the
Phase 3 acceptance test, not state-of-the-art.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def estimate_sigma(
    timestamps_ms: Sequence[int],
    mids: Sequence[float],
    *,
    use_log_returns: bool = False,
) -> float:
    """Estimate volatility from a mid-price time series.

    Parameters
    ----------
    timestamps_ms:
        Sample times in epoch ms. Must be strictly increasing.
    mids:
        Mid-prices in the same order.
    use_log_returns:
        If True, use log-returns Δlog S; otherwise use raw differences ΔS.
        Prediction-market mids live in (0, 1); raw differences are the
        right unit because PnL is dollar-linear in price.

    Returns
    -------
    σ in *price units per √second*. For prediction-market mids, a typical
    value is 0.001–0.05.
    """
    ts = np.asarray(timestamps_ms, dtype=np.int64)
    s = np.asarray(mids, dtype=np.float64)
    if len(ts) != len(s):
        raise ValueError("timestamps_ms and mids must have equal length")
    if len(ts) < 2:
        raise ValueError("need at least 2 samples to estimate sigma")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps_ms must be strictly increasing")

    dt_s = np.diff(ts).astype(np.float64) / 1000.0
    if use_log_returns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.diff(np.log(s))
    else:
        ret = np.diff(s)

    # Scale each return by sqrt(dt) to normalise to per-√second units,
    # then take the sample standard deviation.
    scaled = ret / np.sqrt(dt_s)
    return float(np.std(scaled, ddof=1)) if len(scaled) > 1 else float(abs(scaled[0]))


def estimate_arrival_rate(
    event_timestamps_ms: Sequence[int],
    *,
    observation_window_ms: int | None = None,
) -> float:
    """Estimate a Poisson arrival rate λ̂ = N / T (events per second).

    If ``observation_window_ms`` is omitted, T defaults to the span
    (max − min) of the timestamps. Pass it explicitly if your sample
    is censored (capture started before first event or ended after
    last event).
    """
    ts = np.asarray(event_timestamps_ms, dtype=np.int64)
    n = len(ts)
    if n == 0:
        return 0.0
    if observation_window_ms is None:
        if n < 2:
            raise ValueError("need >=2 events when observation_window_ms not given")
        window_ms = int(ts.max() - ts.min())
    else:
        window_ms = int(observation_window_ms)
    if window_ms <= 0:
        raise ValueError("observation_window_ms must be positive")
    return float(n) * 1000.0 / float(window_ms)


def estimate_kappa(fill_depths: Sequence[float]) -> float:
    """Estimate κ for λ(δ) = A e^{−κδ} from observed fill depths.

    The MLE for an exponential distribution with rate κ is
    κ̂ = 1 / sample_mean. ``fill_depths`` is the distance from mid at
    each observed fill (positive scalar in price units).

    Returns
    -------
    κ in *1 / price units*. Typical Polymarket value is in the range
    20–200 (i.e. e-folding depth ≈ 0.005–0.05 of price).
    """
    arr = np.asarray(fill_depths, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("need >=1 fill depth")
    if np.any(arr < 0):
        raise ValueError("fill depths must be non-negative")
    mean = float(arr.mean())
    if mean == 0.0:
        raise ValueError("mean fill depth is zero — degenerate input")
    return 1.0 / mean


__all__ = ["estimate_arrival_rate", "estimate_kappa", "estimate_sigma"]
