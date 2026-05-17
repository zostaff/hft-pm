# Background processes — quick reference

Started 2026-05-17 UTC. Both processes write JSONL under `data/`, partitioned by UTC date.

## What's running

| process | pid file | command | log |
|---|---|---|---|
| WS capture (8 tokens, 4 markets) | `/tmp/capture.pid` | `python -m hft_pm.data.polymarket_ws` | `logs/capture.log` |
| Paper trade — Thunder NBA YES | `/tmp/paper.pid` | `python scripts/paper_trade.py` | `logs/paper_thunder.log` |

## Markets being captured

| market | yes price | tick | vol24 | category |
|---|---|---|---|---|
| France WC 2026 | 0.176 | 0.001 | $300k | sports / long-dated |
| Spurs NBA Finals 2026 | 0.230 | 0.001 | $288k | sports / medium-term |
| Thunder NBA Finals 2026 *(paper-trade target)* | 0.585 | 0.01 | $57k | sports / mid-range price |
| Newsom 2028 Dem nominee | 0.244 | 0.001 | $213k | politics / very-long |

YES + NO token per market → 8 tokens total on one WS subscription.

## Inspect

```bash
cd /Users/admin/Desktop/hft-bot/hft-pm

# Are both alive?
ps -p $(cat /tmp/capture.pid) $(cat /tmp/paper.pid) -o pid=,stat=,etime= 2>/dev/null

# Capture progress (events per token)
find data/raw/$(date -u +%Y-%m-%d)/ -name '*.jsonl' -exec wc -l {} \;

# Paper-trade log tail (last 20 records, one per event)
tail -20 data/paper/$(date -u +%Y-%m-%d)/49500299856831034491021962156746701298730459370557900271970866855042624695770.jsonl | python3 -m json.tool --json-lines

# Paper-trade running PnL
PYTHONPATH=src python3 -c "
import json, sys
from pathlib import Path
log = list(Path('data/paper').rglob('*.jsonl'))[-1]
last_pnl = None
fills = 0
halt = None
for line in log.read_text().splitlines():
    r = json.loads(line)
    if r['type'] == 'pnl': last_pnl = r
    elif r['type'] == 'fill': fills += 1
    elif r['type'] == 'halt': halt = r
print(f'fills={fills}  last_pnl={last_pnl}  halt={halt}')
"
```

## Stop

```bash
kill $(cat /tmp/capture.pid) $(cat /tmp/paper.pid)
```

## Next steps after a few hours of data

1. `python scripts/calibrate_strategy.py --data data/raw --asset <token> --date 2026-05-17 --out <name>_params.json`
2. `python scripts/run_backtest.py --config configs/<market>_paper.yaml --data data/raw --date 2026-05-17 --params <name>_params.json --latency-ms 50`
3. Compare PnL / Sharpe across markets, look for one where AS actually beats CS (would be evidence that calibrated AS works on a non-Pistons real market)
