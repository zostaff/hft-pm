"""End-to-end backtest CLI: config + replay → SimulationResult + summary.

Usage::

    python scripts/run_backtest.py \\
        --config configs/example.yaml \\
        --data data/raw/ \\
        --date 2026-05-16 \\
        [--params my_params.json] \\
        [--latency-ms 50] \\
        [--out report.json]

Reads the YAML config, optionally merges calibrated params from JSON
(``--params``), replays the captured events for the configured market
on the requested date(s), runs the strategy through the
:class:`Backtester`, and prints / writes a structured report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hft_pm.config import load_config  # noqa: E402
from hft_pm.data.replay import Replay  # noqa: E402
from hft_pm.orderbook.l2_book import L2OrderBook  # noqa: E402
from hft_pm.risk.limits import KillSwitch  # noqa: E402
from hft_pm.simulator.engine import Backtester  # noqa: E402
from hft_pm.simulator.latency import ConstantLatency  # noqa: E402
from hft_pm.strategies.factory import build_strategy, merge_calibrated_params  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_backtest.py",
        description="Run a backtest from a YAML config against captured data.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data", required=True, help="Root of captured JSONL.")
    parser.add_argument(
        "--date",
        required=True,
        help="UTC date (YYYY-MM-DD) to replay. Single day for now.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end UTC date (YYYY-MM-DD) for multi-day replay.",
    )
    parser.add_argument(
        "--params",
        help="Optional calibrated-params JSON from scripts/calibrate_strategy.py "
        "— merged into strategy.params, overriding the config values.",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=0,
        help="Constant order-arrival latency in milliseconds.",
    )
    parser.add_argument("--out", help="Optional path to write a JSON report.")
    return parser.parse_args()


# merge_calibrated_params lives in hft_pm.strategies.factory so the paper-trade
# runner and any future live runner all apply the same calibration→strategy
# mapping plus the same sanity guards (non-positive sigma/kappa rejected).


def _summarise(result, kill_switch: KillSwitch) -> dict[str, Any]:
    snaps = result.pnl_series
    if len(snaps) >= 2:
        pnl_arr = np.array([s.pnl for s in snaps], dtype=np.float64)
        diffs = np.diff(pnl_arr)
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        sharpe = float(np.mean(diffs) / sd) if sd > 0 else 0.0
        running_peak = np.maximum.accumulate(pnl_arr)
        max_dd = float((running_peak - pnl_arr).max())
    else:
        sharpe = 0.0
        max_dd = 0.0

    return {
        "pnl": float(result.pnl),
        "cash": float(result.cash),
        "n_fills": result.n_fills,
        "n_maker_fills": result.n_maker_fills,
        "n_taker_fills": result.n_taker_fills,
        "final_inventory": float(result.final_inventory),
        "fees_paid": float(result.fees_paid),
        "rebates_received": float(result.rebates_received),
        "sharpe_per_event": sharpe,
        "max_drawdown": max_dd,
        "kill_switch_halted": kill_switch.halted,
        "kill_switch_reason": kill_switch.halt_reason.value,
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    strat_params = cfg.strategy.params
    if args.params:
        calibrated = json.loads(Path(args.params).read_text())
        strat_params = merge_calibrated_params(cfg.strategy.kind, strat_params, calibrated)

    strategy = build_strategy(cfg.strategy.kind, strat_params)

    start = date.fromisoformat(args.date)
    end = date.fromisoformat(args.end_date) if args.end_date else start
    replay = Replay(
        root=Path(args.data),
        assets=[cfg.market.token_id],
        date_range=(start, end),
    )

    book = L2OrderBook(tick=cfg.market.tick)
    backtester = Backtester(
        book=book,
        latency=ConstantLatency(args.latency_ms),
        fee_category=cfg.market.fee_category,
        record_pnl_series=True,
    )

    # Kill switch shadows the backtest — it doesn't stop fills, but its
    # final state is reported. A halted run during backtest is a flag
    # that the strategy would have been pulled in production.
    kill_switch = KillSwitch(cfg.risk.to_limits())

    # Run the strategy. After completion, replay the PnL series through
    # the kill switch to detect would-have-been halts.
    result = backtester.run(replay, strategy)
    for snap in result.pnl_series:
        kill_switch.tick(
            now_s=snap.timestamp_ms / 1000.0,
            current_pnl=snap.pnl,
            inventory=snap.inventory,
        )
        if kill_switch.halted:
            break

    summary = _summarise(result, kill_switch)
    summary["config_path"] = str(args.config)
    summary["data_root"] = str(args.data)
    summary["date_range"] = [str(start), str(end)]
    summary["latency_ms"] = args.latency_ms

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
