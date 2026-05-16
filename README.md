# hft-pm

Framework / scaffold for an HFT market-making bot on prediction markets
(Polymarket V2 primary). Phases 1-6 of the roadmap in
[`CLAUDE.md`](CLAUDE.md) are implemented: WebSocket data capture,
event-driven simulator, Avellaneda-Stoikov + GLT quoting, OFI /
microprice / VPIN signals, Hawkes-based event-driven extensions, and
the full purged-CV / DSR / PBO / delay / shuffle validation suite.

The theoretical contract is in
[`docs/hft_prediction_markets_EN.md`](docs/hft_prediction_markets_EN.md);
read it before reading code.

## What's included

- **Data layer**: production WebSocket client with reconnect + heartbeat
  watchdog + sequence tracking; JSONL writer with UTC-date partitioning;
  deterministic replay with gap detection.
- **Simulator**: event-driven backtester with proper L2 queue tracking
  (docs §8.6), Polymarket V2 fee accounting (docs §7.6), injectable
  latency model, fractional-inventory accounting.
- **Strategies**: ConstantSpread baseline, Avellaneda-Stoikov (docs §4.5),
  Guéant-Lehalle-Fernandez-Tapia (docs §4.6, with the inventory-skew
  signs corrected to match AS's direction — see inline comments),
  AS-with-signals variant consuming microprice / OFI / VPIN / scheduled
  jumps.
- **Signals**: rolling-window OFI, Stoikov microprice, PM-normalised VPIN,
  Hawkes intensity tracker + MLE.
- **Validation suite**: purged combinatorial CV, Deflated Sharpe (Bailey
  & López de Prado 2014), Probability of Backtest Overfit, Diebold-Mariano
  with Newey-West HAC, delay-injection wrapper, timestamp-shuffle test.
- **Risk**: `KillSwitch` with max-drawdown, heartbeat-timeout,
  daily-loss-limit, and per-side inventory cap (CLAUDE.md rule #9).
- **CLIs**: end-to-end calibrate + backtest scripts driven by a single
  YAML config.

## What's **NOT** included

- **No live trading.** `py-clob-client-v2` is in the dependency list as
  a placeholder; the live wrapper (`live/client_v2.py`) is not built.
- **No real-data validation.** All acceptance tests run on synthetic
  data we generate ourselves. The validation suite is ready; real data
  has not been fed through it.
- **No paper-trading runner.** The pieces are there (WS client + L2 book
  + strategy + risk limits) but the glue (`paper_trade.py`) is not built.
- **No logit-space market maker** (docs §5) — the AS variant for prices
  near `{0, 1}`. Use AS only on mid-range markets until this is built.

**Use this as a framework / scaffold to build your own bot, not as a
production trading system.** See `DISCLAIMER.md` before doing anything
with real money.

## Setup

```bash
# Preferred: uv
uv sync --extra dev

# Fallback if uv is not installed:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## End-to-end workflow

### 1. Capture WebSocket data on a market

```bash
python -m hft_pm.data.polymarket_ws \
    --assets <token_id_yes>,<token_id_no> \
    --out data/raw/
```

Captures land in `data/raw/{YYYY-MM-DD}/{asset_id}.jsonl`. Run for at
least a few hours (longer for less-active markets) before calibration.

### 2. Calibrate σ, κ, A, alpha_beta from your capture

```bash
python scripts/calibrate_strategy.py \
    --data data/raw/ \
    --asset <token_id> \
    --date 2026-05-16 \
    --out my_params.json
```

Output is a JSON file ready to be merged into a backtest config via
`--params`.

### 3. Backtest your strategy against the captured replay

```bash
python scripts/run_backtest.py \
    --config configs/example.yaml \
    --data data/raw/ \
    --date 2026-05-16 \
    --params my_params.json \
    --latency-ms 50 \
    --out report.json
```

Prints a JSON summary: PnL, Sharpe per event, max drawdown, kill-switch
status, fees/rebates, fill counts. Edit `configs/example.yaml` to point
at your token id, choose a strategy kind, and set risk limits.

### 4. Run the validation suite

```bash
pytest -m integration tests/integration/test_validation_suite.py
```

The suite asserts PBO < 0.30, DSR > 0.95, no Sharpe collapse under
+100/+500/+2000ms delay injection, and a Sharpe collapse under
timestamp shuffle (which is the *expected* failure mode). It currently
runs on synthetic data; point it at your real captures for the
final go-live decision.

## Running tests

```bash
pytest -m "not integration"   # unit tests, ~3 s
pytest -m integration         # integration tests, ~2 min
ruff check src tests scripts
ruff format --check src tests scripts
```

## Phase status

- [x] Phase 1 — Data: WebSocket capture + deterministic replay
- [x] Phase 2 — Simulator: event-driven engine with L2 queue tracking
- [x] Phase 3 — Naive MM: constant-spread → AS → GLT
- [x] Phase 4 — Signals: OFI + microprice + VPIN
- [x] Phase 5 — Event-driven: Hawkes + scheduled-jump withdraw
- [x] Phase 6 — Validation: CPCV + DSR + PBO + delay/shuffle
- [ ] Phase 7 — Paper trading (user responsibility)
- [ ] Phase 8 — Tiny live (user responsibility)

## Numbers (current state)

- 184 tests passing (175 unit + 5 phase-3-5 integration + 4 phase-6
  integration); unit-test suite runs in ~3 s.
- `ruff check`: All checks passed.
- Verified WebSocket client on a 40-min live capture of the Pistons vs.
  Cavaliers market (vol24h $4.8M): median latency 52 ms, matches spec.
- Phase 6 acceptance passes on synthetic data: PBO ≈ 0, DSR > 0.99,
  Sharpe robust to +100/+500/+2000ms delays, Sharpe collapses under
  timestamp shuffle.
