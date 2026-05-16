"""Synthetic event generator for AS-conforming markets (docs §4 Setup).

Produces a stream of :class:`MarketEvent` events that obey the
Avellaneda-Stoikov assumptions:

* Mid-price is a Brownian motion with volatility ``sigma`` (per √second).
* Trade-aggressor arrivals on each side are Poisson with rate ``A``.
* Fill depth from mid is exponential with rate ``kappa``: a SELL
  aggressor at depth δ is priced at ``mid − δ``, so a maker bid at
  depth δ_maker is filled iff δ ≥ δ_maker, with probability
  P(fill) = e^{−κ δ_maker} per arrival.

The book is intentionally one-tick-wide and empty of public liquidity
on each side: this makes the maker the only liquidity provider, so
the strategy's quotes alone determine fill rate. Phase 4+ will swap
in a richer book model.

Output schema matches what :class:`hft_pm.simulator.engine.Backtester`
expects: a sequence of ``BookEvent`` snapshots interleaved with
``LastTradePriceEvent`` events in timestamp order.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from ..data.schemas import BookEvent, LastTradePriceEvent, PriceLevel


def generate_as_world(
    *,
    duration_s: float,
    sigma: float,
    kappa: float,
    A: float,
    mid_start: float = 0.5,
    tick: float = 0.01,
    snapshot_interval_ms: int = 100,
    seed: int | None = None,
    asset_id: str = "synthetic",
    market: str = "synthetic",
) -> Iterator:
    """Generate AS-conforming synthetic market events.

    Parameters
    ----------
    duration_s:
        Total simulated time in seconds.
    sigma:
        Mid-price volatility per √second (e.g. 0.01 means a 1¢ standard
        deviation per second).
    kappa:
        Exponential decay of fill probability with depth from mid.
        Larger κ = fills concentrated near the mid.
    A:
        Per-side trade arrival rate, in trades per second.
    mid_start:
        Starting mid-price.
    tick:
        Tick size used to round trade prices to a level.
    snapshot_interval_ms:
        How often to emit a fresh ``BookEvent`` so the maker sees the
        updated mid. Trades are interleaved at their own Poisson times.
    seed:
        Optional seed for reproducibility.
    asset_id, market:
        Identifiers stamped on every emitted event.

    Yields
    ------
    BookEvent | LastTradePriceEvent
        In strict ascending timestamp order.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if A < 0:
        raise ValueError("A must be non-negative")
    if not 0 < mid_start < 1:
        raise ValueError("mid_start must lie in (0, 1)")
    if tick <= 0:
        raise ValueError("tick must be positive")
    if snapshot_interval_ms <= 0:
        raise ValueError("snapshot_interval_ms must be positive")

    rng = np.random.default_rng(seed)

    duration_ms = int(duration_s * 1000)
    snapshot_times_ms = list(range(0, duration_ms + 1, snapshot_interval_ms))

    n_snapshots = len(snapshot_times_ms)
    # Brownian path: dS = σ dW, dt in seconds.
    dt_s = snapshot_interval_ms / 1000.0
    increments = rng.normal(0.0, sigma * np.sqrt(dt_s), size=n_snapshots - 1)
    mids = np.empty(n_snapshots, dtype=np.float64)
    mids[0] = mid_start
    np.cumsum(increments, out=mids[1:])
    mids[1:] += mid_start
    # Reflect at the boundaries to keep mid in (tick, 1-tick) without
    # distorting the AS-world's local dynamics (an absorbing boundary
    # would freeze the simulation).
    lo, hi = 2 * tick, 1 - 2 * tick
    mids = np.clip(mids, lo, hi)

    # Trade arrivals (combined both sides) — one homogeneous Poisson process.
    n_trades_expected = int(2 * A * duration_s) + 1
    # Generate slightly more inter-arrival times than expected to avoid
    # truncation; oversample factor 4 covers the upper-tail safely.
    inter_arrivals = rng.exponential(1.0 / max(2 * A, 1e-12), size=n_trades_expected * 4)
    trade_times_s = np.cumsum(inter_arrivals)
    trade_times_s = trade_times_s[trade_times_s < duration_s]
    trade_times_ms = (trade_times_s * 1000.0).astype(np.int64)

    # Per-trade: side (BUY/SELL with prob 0.5 each) and depth ~ Exp(κ).
    n_trades = len(trade_times_ms)
    sides = rng.choice([b"BUY", b"SELL"], size=n_trades).astype(str)
    depths = rng.exponential(1.0 / kappa, size=n_trades)

    # Convert to a sortable list of (timestamp_ms, kind, payload) tuples
    # then yield in timestamp order via two-pointer merge.
    snap_ix = 0
    trade_ix = 0
    base_ts = 0  # epoch-ms anchor; tests just want monotonic ordering

    # Public-liquidity buffer at the BBO. Used only so that the L2
    # book has a defined mid; the maker quotes deeper, where there is
    # no public liquidity, and trades at deep prices land directly on
    # the maker's price level with zero queue ahead.
    bbo_size = 1000.0

    def _book_event(ts_offset_ms: int, mid: float) -> BookEvent:
        bid_px = round((mid - tick) / tick) * tick
        ask_px = round((mid + tick) / tick) * tick
        return BookEvent(
            asset_id=asset_id,
            market=market,
            timestamp_ms=base_ts + ts_offset_ms,
            recv_ts_ms=base_ts + ts_offset_ms,
            bids=[PriceLevel(price=max(tick, bid_px), size=bbo_size)],
            asks=[PriceLevel(price=min(1 - tick, ask_px), size=bbo_size)],
        )

    def _walk_trade(ts_ms: int, mid: float, side: str, depth_budget: float):
        """Yield one ``LastTradePriceEvent`` per tick the aggressor walks.

        A market order with depth budget ``d`` consumes liquidity from
        the BBO outward to ``mid ± d`` in tick increments. Each per-tick
        trade lands at that exact level, so a maker resting on any of
        those ticks is filled. This matches Polymarket's behaviour —
        the matching engine emits one trade event per traded price level.
        """
        if side == "BUY":
            start = max(tick, min(1 - tick, round((mid + tick) / tick) * tick))
            end_raw = mid + depth_budget
            end = max(tick, min(1 - tick, round(end_raw / tick) * tick))
            n_levels = round((end - start) / tick) + 1
            prices = [round((start + i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        else:
            start = max(tick, min(1 - tick, round((mid - tick) / tick) * tick))
            end_raw = mid - depth_budget
            end = max(tick, min(1 - tick, round(end_raw / tick) * tick))
            n_levels = round((start - end) / tick) + 1
            prices = [round((start - i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        for i, p in enumerate(prices):
            yield LastTradePriceEvent(
                asset_id=asset_id,
                market=market,
                # Stagger by 1 ms per level so heap ordering is stable
                # and price_change events at the same logical time are
                # not interleaved between per-tick trades.
                timestamp_ms=base_ts + ts_ms + i,
                recv_ts_ms=base_ts + ts_ms + i,
                price=p,
                size=1.0,
                side=side,  # type: ignore[arg-type]
            )

    # Two-pointer merge of snapshot and trade streams in timestamp order.
    while snap_ix < n_snapshots or trade_ix < n_trades:
        if trade_ix >= n_trades or (
            snap_ix < n_snapshots and snapshot_times_ms[snap_ix] <= int(trade_times_ms[trade_ix])
        ):
            yield _book_event(snapshot_times_ms[snap_ix], float(mids[snap_ix]))
            snap_ix += 1
        else:
            ts = int(trade_times_ms[trade_ix])
            ref_ix = max(0, snap_ix - 1)
            yield from _walk_trade(
                ts, float(mids[ref_ix]), sides[trade_ix], float(depths[trade_ix])
            )
            trade_ix += 1


def generate_drifted_world(
    *,
    duration_s: float,
    sigma: float,
    kappa: float,
    A: float,
    drift_strength: float,
    flow_decay_s: float = 5.0,
    mid_start: float = 0.5,
    tick: float = 0.001,
    snapshot_interval_ms: int = 100,
    seed: int | None = None,
    asset_id: str = "synthetic",
    market: str = "synthetic",
) -> Iterator:
    """Drifted variant of :func:`generate_as_world` with predictable order flow.

    Adds a hidden mean-reverting "flow" state ``φ_t`` (Ornstein-Uhlenbeck
    around 0 with decay ``flow_decay_s``). It controls two things:

    1. **Trade aggressor side** is biased: P(BUY) = sigmoid(φ_t) so that
       positive φ → more BUYs → trade arrivals on the ask side.
    2. **BBO asymmetry**: bid size grows when φ > 0 (buyers queueing),
       ask size shrinks. This means the imbalance is observable in the
       book — OFI / microprice can detect it and predict the next mid
       move.
    3. **Mid drift**: ``dmid = drift_strength · φ · dt``.

    A strategy that consumes OFI / microprice can tilt its reservation
    price in the direction of the upcoming drift; vanilla AS ignores it
    and quotes at the lagging mid. That's the Phase 4 acceptance setup.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if A < 0:
        raise ValueError("A must be non-negative")
    if drift_strength < 0:
        raise ValueError("drift_strength must be non-negative")
    if flow_decay_s <= 0:
        raise ValueError("flow_decay_s must be positive")
    if not 0 < mid_start < 1:
        raise ValueError("mid_start must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    duration_ms = int(duration_s * 1000)
    dt_s = snapshot_interval_ms / 1000.0
    snapshot_times_ms = list(range(0, duration_ms + 1, snapshot_interval_ms))
    n_snapshots = len(snapshot_times_ms)

    # Simulate the OU flow process and the drifted Brownian mid jointly.
    flow = np.empty(n_snapshots, dtype=np.float64)
    mids = np.empty(n_snapshots, dtype=np.float64)
    flow[0] = 0.0
    mids[0] = mid_start
    flow_sigma = 1.0  # OU innovation scale; arbitrary, only ratio matters
    flow_decay = math.exp(-dt_s / flow_decay_s)
    flow_innov = rng.normal(0.0, flow_sigma * math.sqrt(dt_s), size=n_snapshots - 1)
    mid_innov = rng.normal(0.0, sigma * math.sqrt(dt_s), size=n_snapshots - 1)
    for i in range(1, n_snapshots):
        flow[i] = flow[i - 1] * flow_decay + flow_innov[i - 1]
        mids[i] = mids[i - 1] + drift_strength * flow[i - 1] * dt_s + mid_innov[i - 1]
    lo, hi = 2 * tick, 1 - 2 * tick
    mids = np.clip(mids, lo, hi)

    # Trade arrival times
    n_expected = int(2 * A * duration_s) + 1
    inter = rng.exponential(1.0 / max(2 * A, 1e-12), size=n_expected * 4)
    trade_times_s = np.cumsum(inter)
    trade_times_s = trade_times_s[trade_times_s < duration_s]
    trade_times_ms = (trade_times_s * 1000.0).astype(np.int64)
    n_trades = len(trade_times_ms)

    depths = rng.exponential(1.0 / kappa, size=n_trades)

    base_size = 500.0  # baseline BBO size per side
    flow_amp = 500.0  # how much flow tilts the BBO sizes
    base_ts = 0

    def _book_event_with_flow(ts_offset_ms: int, mid: float, phi: float) -> BookEvent:
        bid_px = max(tick, min(1 - tick, round((mid - tick) / tick) * tick))
        ask_px = max(tick, min(1 - tick, round((mid + tick) / tick) * tick))
        # phi > 0 → bid stacks, ask thins.
        bid_sz = max(1.0, base_size + flow_amp * phi)
        ask_sz = max(1.0, base_size - flow_amp * phi)
        return BookEvent(
            asset_id=asset_id,
            market=market,
            timestamp_ms=base_ts + ts_offset_ms,
            recv_ts_ms=base_ts + ts_offset_ms,
            bids=[PriceLevel(price=bid_px, size=bid_sz)],
            asks=[PriceLevel(price=ask_px, size=ask_sz)],
        )

    def _walk_trade(ts_ms: int, mid: float, side: str, depth_budget: float):
        if side == "BUY":
            start = max(tick, min(1 - tick, round((mid + tick) / tick) * tick))
            end = max(tick, min(1 - tick, round((mid + depth_budget) / tick) * tick))
            n_levels = round((end - start) / tick) + 1
            prices = [round((start + i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        else:
            start = max(tick, min(1 - tick, round((mid - tick) / tick) * tick))
            end = max(tick, min(1 - tick, round((mid - depth_budget) / tick) * tick))
            n_levels = round((start - end) / tick) + 1
            prices = [round((start - i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        for i, p in enumerate(prices):
            yield LastTradePriceEvent(
                asset_id=asset_id,
                market=market,
                timestamp_ms=base_ts + ts_ms + i,
                recv_ts_ms=base_ts + ts_ms + i,
                price=p,
                size=1.0,
                side=side,  # type: ignore[arg-type]
            )

    snap_ix = 0
    trade_ix = 0
    while snap_ix < n_snapshots or trade_ix < n_trades:
        if trade_ix >= n_trades or (
            snap_ix < n_snapshots and snapshot_times_ms[snap_ix] <= int(trade_times_ms[trade_ix])
        ):
            yield _book_event_with_flow(
                snapshot_times_ms[snap_ix], float(mids[snap_ix]), float(flow[snap_ix])
            )
            snap_ix += 1
        else:
            ts = int(trade_times_ms[trade_ix])
            ref_ix = max(0, snap_ix - 1)
            # Bias side: P(BUY) = sigmoid(2 * phi).
            p_buy = 1.0 / (1.0 + math.exp(-2.0 * float(flow[ref_ix])))
            side = "BUY" if rng.random() < p_buy else "SELL"
            yield from _walk_trade(ts, float(mids[ref_ix]), side, float(depths[trade_ix]))
            trade_ix += 1


def generate_jumpy_world(
    *,
    duration_s: float,
    sigma: float,
    kappa: float,
    A: float,
    jump_times_s: list[float],
    jump_magnitudes: list[float],
    mid_start: float = 0.5,
    tick: float = 0.001,
    snapshot_interval_ms: int = 100,
    seed: int | None = None,
    asset_id: str = "synthetic",
    market: str = "synthetic",
) -> Iterator:
    """AS-conforming synthetic with scheduled mid-price jumps (docs §6).

    Between jumps the world is the same as :func:`generate_as_world`:
    Brownian mid + Poisson trade walks. At each scheduled time
    ``jump_times_s[i]``, the underlying mid steps by ``jump_magnitudes[i]``
    (signed) — modelling a news release or scheduled announcement.

    A maker that withdraws shortly before the scheduled time avoids
    being run over by aggressive aggressors hitting stale quotes; a
    maker that ignores the schedule eats the full adverse move.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if A < 0:
        raise ValueError("A must be non-negative")
    if len(jump_times_s) != len(jump_magnitudes):
        raise ValueError("jump_times_s and jump_magnitudes must align")
    if not 0 < mid_start < 1:
        raise ValueError("mid_start must lie in (0, 1)")
    sorted_jumps = sorted(zip(jump_times_s, jump_magnitudes, strict=True))
    if any(t <= 0 or t >= duration_s for t, _ in sorted_jumps):
        raise ValueError("each jump time must lie strictly inside (0, duration_s)")

    rng = np.random.default_rng(seed)
    duration_ms = int(duration_s * 1000)
    dt_s = snapshot_interval_ms / 1000.0
    snapshot_times_ms = list(range(0, duration_ms + 1, snapshot_interval_ms))
    n_snapshots = len(snapshot_times_ms)

    # Brownian path with discrete jumps at the scheduled times.
    increments = rng.normal(0.0, sigma * np.sqrt(dt_s), size=n_snapshots - 1)
    mids = np.empty(n_snapshots, dtype=np.float64)
    mids[0] = mid_start
    jump_idx = 0
    for i in range(1, n_snapshots):
        ts_s = snapshot_times_ms[i] / 1000.0
        # Apply any pending jump whose time has just passed.
        jump_total = 0.0
        while jump_idx < len(sorted_jumps) and sorted_jumps[jump_idx][0] <= ts_s:
            jump_total += sorted_jumps[jump_idx][1]
            jump_idx += 1
        mids[i] = mids[i - 1] + increments[i - 1] + jump_total

    lo, hi = 2 * tick, 1 - 2 * tick
    mids = np.clip(mids, lo, hi)

    # Trade arrivals — homogeneous Poisson; aggressor side biased toward
    # the direction of the just-applied jump for a brief shock window.
    n_expected = int(2 * A * duration_s) + 1
    inter = rng.exponential(1.0 / max(2 * A, 1e-12), size=n_expected * 4)
    trade_times_s = np.cumsum(inter)
    trade_times_s = trade_times_s[trade_times_s < duration_s]
    trade_times_ms = (trade_times_s * 1000.0).astype(np.int64)
    n_trades = len(trade_times_ms)
    depths = rng.exponential(1.0 / kappa, size=n_trades)

    bbo_size = 1000.0
    base_ts = 0
    shock_window_s = 2.0  # how long after a jump the side bias persists

    def _book(ts_offset_ms: int, mid: float) -> BookEvent:
        bid_px = round((mid - tick) / tick) * tick
        ask_px = round((mid + tick) / tick) * tick
        return BookEvent(
            asset_id=asset_id,
            market=market,
            timestamp_ms=base_ts + ts_offset_ms,
            recv_ts_ms=base_ts + ts_offset_ms,
            bids=[PriceLevel(price=max(tick, bid_px), size=bbo_size)],
            asks=[PriceLevel(price=min(1 - tick, ask_px), size=bbo_size)],
        )

    def _walk(ts_ms: int, mid: float, side: str, depth_budget: float):
        if side == "BUY":
            start = max(tick, min(1 - tick, round((mid + tick) / tick) * tick))
            end = max(tick, min(1 - tick, round((mid + depth_budget) / tick) * tick))
            n_levels = round((end - start) / tick) + 1
            prices = [round((start + i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        else:
            start = max(tick, min(1 - tick, round((mid - tick) / tick) * tick))
            end = max(tick, min(1 - tick, round((mid - depth_budget) / tick) * tick))
            n_levels = round((start - end) / tick) + 1
            prices = [round((start - i * tick) * 1e9) / 1e9 for i in range(max(1, n_levels))]
        for i, p in enumerate(prices):
            yield LastTradePriceEvent(
                asset_id=asset_id,
                market=market,
                timestamp_ms=base_ts + ts_ms + i,
                recv_ts_ms=base_ts + ts_ms + i,
                price=p,
                size=1.0,
                side=side,  # type: ignore[arg-type]
            )

    def _aggressor_side(trade_ts_s: float) -> str:
        # Within shock_window_s after a positive jump, BUYs dominate; after
        # a negative jump, SELLs dominate. Otherwise 50/50.
        for jt, jm in sorted_jumps:
            if 0 <= trade_ts_s - jt <= shock_window_s:
                p_buy = 0.80 if jm > 0 else 0.20
                return "BUY" if rng.random() < p_buy else "SELL"
        return "BUY" if rng.random() < 0.5 else "SELL"

    snap_ix = 0
    trade_ix = 0
    while snap_ix < n_snapshots or trade_ix < n_trades:
        if trade_ix >= n_trades or (
            snap_ix < n_snapshots and snapshot_times_ms[snap_ix] <= int(trade_times_ms[trade_ix])
        ):
            yield _book(snapshot_times_ms[snap_ix], float(mids[snap_ix]))
            snap_ix += 1
        else:
            ts_ms_val = int(trade_times_ms[trade_ix])
            ref_ix = max(0, snap_ix - 1)
            side = _aggressor_side(ts_ms_val / 1000.0)
            yield from _walk(ts_ms_val, float(mids[ref_ix]), side, float(depths[trade_ix]))
            trade_ix += 1


__all__ = ["generate_as_world", "generate_drifted_world", "generate_jumpy_world"]
