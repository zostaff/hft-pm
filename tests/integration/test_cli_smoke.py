"""End-to-end smoke test for the calibrate + backtest CLIs.

Generates a tiny synthetic capture on disk (via the JsonlWriter and
generate_as_world), then shells out to both CLIs and checks the JSON
report is well-formed and the kill switch did not falsely trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest

from hft_pm.data.schemas import BookEvent, LastTradePriceEvent
from hft_pm.data.writer import JsonlWriter
from hft_pm.simulator.synthetic import generate_as_world

pytestmark = pytest.mark.integration


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _seed_capture(tmp_path: Path, asset_id: str, day: date) -> Path:
    """Write a small AS-conforming synthetic capture under tmp_path/data/raw."""
    data_root = tmp_path / "data" / "raw"
    writer = JsonlWriter(data_root)
    base_ms = int((day - date(1970, 1, 1)).days * 86_400 * 1000 + 10_000)  # ~10s into the UTC day
    events = list(
        generate_as_world(
            duration_s=20,
            sigma=0.005,
            kappa=60,
            A=2.0,
            tick=0.01,
            asset_id=asset_id,
            market="0xdeadbeef",
            seed=0,
        )
    )
    for ev in events:
        raw: dict = {
            "asset_id": ev.asset_id,
            "market": ev.market,
            "timestamp": str(base_ms + ev.timestamp_ms),
            "event_type": ev.event_type,
        }
        if isinstance(ev, BookEvent):
            raw["bids"] = [{"price": str(b.price), "size": str(b.size)} for b in ev.bids]
            raw["asks"] = [{"price": str(a.price), "size": str(a.size)} for a in ev.asks]
        elif isinstance(ev, LastTradePriceEvent):
            raw["price"] = str(ev.price)
            raw["size"] = str(ev.size)
            raw["side"] = ev.side
        writer.write(raw, base_ms + ev.timestamp_ms)
    writer.close()
    return data_root


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_PROJECT_ROOT / "src"), "PATH": ""},
        check=False,
    )


def test_calibrate_then_backtest_end_to_end(tmp_path: Path) -> None:
    asset_id = "11111"
    day = date(2026, 5, 16)
    data_root = _seed_capture(tmp_path, asset_id, day)

    # 1. Calibrate
    params_path = tmp_path / "params.json"
    res = _run_cli(
        [
            "scripts/calibrate_strategy.py",
            "--data",
            str(data_root),
            "--asset",
            asset_id,
            "--date",
            day.isoformat(),
            "--out",
            str(params_path),
        ]
    )
    assert res.returncode == 0, res.stderr
    params = json.loads(params_path.read_text())
    assert params["n_book_events"] > 0
    assert params["sigma_per_sqrts"] >= 0
    assert params["kappa"] >= 0

    # 2. Write a minimal config that targets this asset.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(f"""\
        market:
          token_id: "{asset_id}"
          fee_category: OTHER
          tick: 0.01
        strategy:
          kind: avellaneda_stoikov_with_signals
          params:
            gamma: 1.0
            sigma: 0.005
            kappa: 60.0
            horizon_ms: 20000
            size: 1
            use_microprice: true
            ofi_window_s: 1.0
            alpha_beta: 0.0
        risk:
          max_drawdown_pct: 0.5
          max_inventory: 100
          heartbeat_timeout_s: 60
          baseline_capital: 100
    """)
    )

    # 3. Backtest with calibrated params merged in.
    report_path = tmp_path / "report.json"
    res = _run_cli(
        [
            "scripts/run_backtest.py",
            "--config",
            str(cfg_path),
            "--data",
            str(data_root),
            "--date",
            day.isoformat(),
            "--params",
            str(params_path),
            "--out",
            str(report_path),
        ]
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(report_path.read_text())
    # The strategy may or may not have fills against the synthetic capture,
    # but the report must be well-formed and the kill switch must not trip
    # on a clean run.
    assert "pnl" in report
    assert "sharpe_per_event" in report
    assert "max_drawdown" in report
    assert report["kill_switch_halted"] is False
    assert report["kill_switch_reason"] == "none"
