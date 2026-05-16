"""Paper-trade against the live Polymarket feed (Tier 2 entrypoint).

Usage::

    python scripts/paper_trade.py \\
        --config configs/example.yaml \\
        --log-root data/paper/ \\
        --latency-ms 50 \\
        [--params my_params.json]

Strategy, market, fee tier, and risk limits are all read from the YAML
config (same schema as scripts/run_backtest.py). Optional calibrated
parameters from scripts/calibrate_strategy.py override the in-config
values.

Output: one JSONL file at ``{log-root}/{YYYY-MM-DD UTC}/{token_id}.jsonl``
with one record per event / order / fill / halt. Stop with Ctrl-C or
let the kill switch halt the run on drawdown / heartbeat / daily-loss
breach.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hft_pm.live.paper_trade import main  # noqa: E402

if __name__ == "__main__":
    main()
