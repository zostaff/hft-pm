"""Minimal Phase-1 capture entrypoint.

Thin wrapper around ``python -m hft_pm.data.polymarket_ws`` so the runbook
in CLAUDE.md works as printed:

    python scripts/capture_data.py --assets <id1>,<id2> --out data/raw/
"""

from __future__ import annotations

import sys


def main() -> None:
    from hft_pm.data import polymarket_ws

    sys.argv[0] = "scripts/capture_data.py"
    polymarket_ws.main()


if __name__ == "__main__":
    main()
