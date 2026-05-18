"""Calibrate strategy parameters (σ, κ, A, alpha_beta) from a captured replay.

Usage::

    python scripts/calibrate_strategy.py \\
        --data data/raw/ \\
        --asset 13915... \\
        --date 2026-05-16 \\
        --out my_params.json

Reads the JSONL files written by :mod:`hft_pm.data.writer`, computes:

* **σ** — from successive mid-price differences (Brownian fit)
* **A** — total trade arrival rate per side
* **κ** — exponential MLE on observed trade depths from mid
* **alpha_beta** — linear regression of forward Δmid on rolling OFI

and writes the result as JSON so a backtest / paper-trade config can
load it. None of these are full-precision research-grade fits; they
are a sane starting point for the Phase 6 validation suite to refine.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# Allow `python scripts/...` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hft_pm.data.replay import Replay  # noqa: E402
from hft_pm.data.schemas import (  # noqa: E402
    BookEvent,
    LastTradePriceEvent,
)
from hft_pm.signals.calibration import (  # noqa: E402
    estimate_arrival_rate,
    estimate_kappa,
    estimate_sigma,
)
from hft_pm.signals.ofi import OFICalculator  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate_strategy.py",
        description="Calibrate sigma, kappa, A, alpha_beta from a captured replay.",
    )
    parser.add_argument("--data", required=True, help="Root directory of captured JSONL.")
    parser.add_argument("--asset", required=True, help="Token id (asset_id) to calibrate against.")
    parser.add_argument(
        "--date",
        required=True,
        help="UTC date of the capture to load (YYYY-MM-DD). Single day for now.",
    )
    parser.add_argument(
        "--ofi-window-s",
        type=float,
        default=2.0,
        help="Rolling window for OFI accumulation (seconds).",
    )
    parser.add_argument(
        "--ofi-horizon-s",
        type=float,
        default=1.0,
        help="Forward horizon (seconds) for the OFI→Δmid regression.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser.parse_args()


def _extract_streams(
    events: list, asset_id: str
) -> tuple[list[tuple[int, float, float, float, float]], list[LastTradePriceEvent]]:
    """Return (book_obs, trades) for one asset.

    ``book_obs`` is a list of (timestamp_ms, bid_px, bid_sz, ask_px, ask_sz).
    """
    book_obs = []
    trades = []
    last_ts = -1
    for ev in events:
        if ev.asset_id != asset_id:
            continue
        if isinstance(ev, BookEvent):
            if not ev.bids or not ev.asks:
                continue
            bb_px = max(b.price for b in ev.bids)
            bb_sz = next(b.size for b in ev.bids if b.price == bb_px)
            ba_px = min(a.price for a in ev.asks)
            ba_sz = next(a.size for a in ev.asks if a.price == ba_px)
            ts = ev.timestamp_ms
            if ts <= last_ts:
                # Polymarket occasionally republishes a snapshot with the
                # same logical timestamp; nudge forward by 1 ms so the
                # estimators (which require strictly increasing ts) accept it.
                ts = last_ts + 1
            last_ts = ts
            book_obs.append((ts, bb_px, bb_sz, ba_px, ba_sz))
        elif isinstance(ev, LastTradePriceEvent):
            trades.append(ev)
    return book_obs, trades


def _calibrate_ofi_alpha(
    book_obs: list[tuple[int, float, float, float, float]],
    *,
    window_s: float,
    horizon_s: float,
) -> dict[str, float | int]:
    """Fit ``Δmid_{t+h} = β · OFI_t``, return slope + R²."""
    if len(book_obs) < 50:
        return {"alpha_beta": 0.0, "r2": 0.0, "n_samples": len(book_obs)}
    ofi_calc = OFICalculator(window_seconds=window_s)
    ofi_series = []
    mid_series = []
    ts_series = []
    for ts_ms, bb_px, bb_sz, ba_px, ba_sz in book_obs:
        ofi = ofi_calc.update(ts_ms / 1000.0, bb_px, bb_sz, ba_px, ba_sz)
        ofi_series.append(ofi)
        mid_series.append((bb_px + ba_px) / 2.0)
        ts_series.append(ts_ms)
    ts_arr = np.array(ts_series, dtype=np.int64)
    mids = np.array(mid_series, dtype=np.float64)
    ofis = np.array(ofi_series, dtype=np.float64)
    horizon_ms = int(horizon_s * 1000)
    # For each sample i, find the smallest j such that ts[j] >= ts[i] + horizon.
    target_ts = ts_arr + horizon_ms
    target_idx = np.searchsorted(ts_arr, target_ts, side="left")
    valid = target_idx < len(ts_arr)
    if valid.sum() < 30:
        return {"alpha_beta": 0.0, "r2": 0.0, "n_samples": int(valid.sum())}
    x = ofis[valid]
    y = mids[target_idx[valid]] - mids[valid]
    if np.std(x, ddof=1) == 0:
        return {"alpha_beta": 0.0, "r2": 0.0, "n_samples": len(x)}
    # OLS through origin (intercept absorbed by the mid drift).
    slope = float(np.sum(x * y) / np.sum(x * x))
    pred = slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"alpha_beta": slope, "r2": r2, "n_samples": len(x)}


def main() -> None:
    args = _parse_args()
    day = date.fromisoformat(args.date)
    replay = Replay(root=Path(args.data), assets=[args.asset], date_range=(day, day))
    events = list(replay)
    if not events:
        raise SystemExit(f"no events found in {args.data} for asset={args.asset} on {day}")

    book_obs, trades = _extract_streams(events, args.asset)
    if len(book_obs) < 5:
        raise SystemExit(f"too few book events for asset {args.asset}: {len(book_obs)}")

    # σ from successive mid changes
    ts = [b[0] for b in book_obs]
    mids = [(b[1] + b[3]) / 2.0 for b in book_obs]
    sigma = estimate_sigma(ts, mids)

    # A from trade arrival rate
    if len(trades) >= 2:
        first_ts = min(t.timestamp_ms for t in trades)
        last_ts = max(t.timestamp_ms for t in trades)
        window_ms = max(1, last_ts - first_ts)
        per_side_rate = (
            estimate_arrival_rate([t.timestamp_ms for t in trades], observation_window_ms=window_ms)
            / 2.0
        )
    else:
        per_side_rate = 0.0

    # κ from observed trade depth (|trade_price - last mid|)
    depths = []
    if trades:
        sorted_book_ts = np.array(ts, dtype=np.int64)
        sorted_mids = np.array(mids, dtype=np.float64)
        for trade in trades:
            idx = int(np.searchsorted(sorted_book_ts, trade.timestamp_ms, side="right") - 1)
            if idx < 0:
                continue
            mid = float(sorted_mids[idx])
            d = abs(trade.price - mid)
            if d > 0:
                depths.append(d)
    kappa = estimate_kappa(depths) if depths else 0.0

    ofi_fit = _calibrate_ofi_alpha(
        book_obs, window_s=args.ofi_window_s, horizon_s=args.ofi_horizon_s
    )

    warnings: list[str] = []
    # Thresholds are pragmatic: hft-pm's 2026-05-17 Thunder NBA capture had
    # n_book=193, n_trade=91 over 11 h and produced sigma=0 (mid stuck),
    # which crashed AvellanedaStoikov on the next backtest. Captures around
    # 500 book + 200 trade events give barely-usable params (R^2 ≈ 0);
    # below ~200/100 the estimator collapses regularly. The numbers are
    # warnings, not errors — calibration still writes the JSON so a
    # downstream consumer (merge_calibrated_params) can apply its own
    # safeguards (non-positive sigma/kappa rejected).
    if len(book_obs) < 500:
        warnings.append(
            f"n_book_events={len(book_obs)} < 500: sigma and kappa estimates may be "
            "unstable; consider a longer capture window"
        )
    if len(trades) < 200:
        warnings.append(
            f"n_trade_events={len(trades)} < 200: kappa and A estimates may be unstable"
        )
    if sigma <= 0:
        warnings.append(
            f"sigma_per_sqrts={sigma} ≤ 0: the mid likely never moved during the "
            "capture; downstream backtests using this value will reject it and "
            "fall back to the YAML default"
        )
    if kappa <= 0:
        warnings.append(f"kappa={kappa} ≤ 0: no trades observed at non-zero depth from mid")
    if abs(ofi_fit["r2"]) < 0.005:
        warnings.append(
            f"alpha_beta R²={ofi_fit['r2']:.4f}: OFI has effectively zero predictive "
            "power on this capture; alpha_beta is noise"
        )

    result: dict[str, Any] = {
        "asset_id": args.asset,
        "date": str(day),
        "n_book_events": len(book_obs),
        "n_trade_events": len(trades),
        "sigma_per_sqrts": sigma,
        "A_per_side": per_side_rate,
        "kappa": kappa,
        "alpha_beta": ofi_fit["alpha_beta"],
        "alpha_beta_r2": ofi_fit["r2"],
        "alpha_beta_n_samples": ofi_fit["n_samples"],
        "alpha_beta_window_s": args.ofi_window_s,
        "alpha_beta_horizon_s": args.ofi_horizon_s,
        "calibration_warnings": warnings,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
