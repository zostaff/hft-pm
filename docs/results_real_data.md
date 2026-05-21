# Real-data results: 2026-05-16 → 2026-05-22

Six days of live Polymarket WebSocket captures across eight tokens
(four markets × YES/NO), then calibrated and backtested. This document
records what the validation suite — which had until now only been run
on synthetic data — said when pointed at real captures.

This is **not** an endorsement of any of the strategies for live
trading. It is the honest result of running the framework end-to-end
on real data.

---

## Setup

* **Tokens captured (8):** Spurs / Cavaliers / Thunder / Pistons (NBA
  Finals 2026), plus France WC 2026 YES + Newsom 2028 Dem nominee.
* **Capture duration:** ~6 days continuous via
  `python -m hft_pm.data.polymarket_ws --assets <ids> --out data/raw/`,
  one process holding a single WebSocket subscription to all 8 tokens.
* **Activity by day:** the NBA Finals game on **2026-05-21** produced the
  only dataset with enough trade flow for stable calibration on a
  single market — the Spurs YES token recorded **3 251 trade events**
  that day. The other days and tokens had between 50 and 1 300 trades.
* **Calibrated parameters on Spurs 2026-05-21:**
  | param | value | note |
  |---|---|---|
  | `sigma_per_sqrts` | 1.56 · 10⁻⁴ | ~30× smaller than the synthetic 0.005 default |
  | `kappa` | 1 463 | ~24× larger than the synthetic 60 default |
  | `A_per_side` | 0.020 trades / ms | ≈20 trades / s per side |
  | `alpha_beta` | **−8.95 · 10⁻⁷** | **negative** — OFI predicts mean-reversion |
  | OFI regression R² | 0.096 | first real non-zero signal across the captures |

---

## Headline result: AS-family backtest matrix on Spurs YES 2026-05-21

All four strategies driven through the event-driven backtester with
`ConstantLatency(50ms)`, Polymarket V2 SPORTS fee category, calibrated
σ / κ / α from the JSON above. Strategy-specific params from
`configs/example.yaml`, `gamma=1.0`, `horizon_ms=600 000` (10 min).

| strategy   | PnL ($) | fills | maker | final inv | max DD ($) | Sharpe / event | rebates ($) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ConstantSpread** | **+0.308** | 19 | 19 | −7 | 0.236 | +0.0097 | 0.031 |
| **GLT**            | **+0.305** | 22 | 22 | −5 | 0.236 | +0.0095 | 0.034 |
| Avellaneda-Stoikov | +0.172 | 13 | 13 | −1 | 0.118 | +0.0108 | 0.021 |
| AS + signals       | +0.004 |  5 |  5 | −5 | 0.038 | +0.0020 | 0.008 |

(Bankroll = $100, `max_drawdown_pct = 0.20`, `daily_loss_limit = 25`.
None of the strategies tripped the kill switch — peak $ drawdowns
remain below 1 % of bankroll because the absolute PnLs are small.)

### What the table says

1. **Symmetric quoters (CS, GLT) win on absolute PnL.** They both
   collect ~$0.30 by quoting at mid ± 1 tick and re-pegging when the
   public book moves. GLT's small inventory-skew term does not change
   the outcome materially on this data.
2. **Avellaneda-Stoikov plain underperforms by ~44 %.** It has the
   best Sharpe-per-event of the four (+0.0108 vs +0.0097 for CS) and
   half the dollar drawdown, but at this bankroll scale the absolute
   dollars matter more than the risk-adjusted ratio.
3. **AS with signals is the worst.** Adding the calibrated OFI
   skew + microprice cuts PnL by 98 %. The signal is real
   (R² = 0.096) but the AS-with-signals framework cannot extract
   value from it on this market.

---

## Why γ does not help AS on PM data

Sweeping `gamma` from 0.01 to 5.0 on the same backtest:

```
γ=0.01  pnl=+0.1779  fills=14  inv=-2.0
γ=0.10  pnl=+0.1779  fills=14  inv=-2.0
γ=0.50  pnl=+0.1779  fills=14  inv=-2.0
γ=1.00  pnl=+0.1779  fills=14  inv=-2.0
γ=2.00  pnl=+0.1779  fills=14  inv=-2.0
γ=5.00  pnl=+0.1779  fills=14  inv=-2.0
```

Bit-for-bit identical. With the calibrated σ ≈ 1.56 · 10⁻⁴ and
T = 600 s:

* AS half-spread term `½(γσ²T) + (1/γ)·ln(1 + γ/κ)`
  ≈ 7.3·10⁻⁶ · γ + 0.00069   → ≈ 0.00069 for every γ in the sweep
* AS inventory drift `q · γ · σ²T`
  ≈ q · γ · 1.46·10⁻⁵         → ≤ 1.5·10⁻⁴ for any realistic q · γ

Both quantities are **below the 0.001 tick size**, so both quotes snap
to mid ± 1 tick regardless of γ. AS degenerates into "ConstantSpread
with 1-tick spread minus 5 fills that get refused for inventory-cap
reasons elsewhere in the strategy".

This is not a bug in our implementation — it is the AS formula
applied to a market where σ is so small that the price-volatility
component of the spread is smaller than the minimum tick. AS is
designed for fast-moving markets where σ·√T comfortably exceeds the
tick.

---

## Negative α: OFI mean-reverts on PM markets

The synthetic Phase-4 acceptance test calibrated `alpha_beta ≈ 5·10⁻⁷`,
positive: aggressive buying (OFI > 0) predicted price up. On
Spurs 2026-05-21 the same regression yielded **α = −8.95·10⁻⁷** with
R² = 0.096.

Sweep over `alpha_beta` ∈ {−2·10⁻⁶, −8.95·10⁻⁷, −5·10⁻⁷, 0, +5·10⁻⁷}
with everything else fixed:

```
α=    +5e-7    pnl=+0.0028  fills=7  inv=-5.0
α=        0    pnl=-0.0004  fills=6  inv=-4.0
α=    -5e-7    pnl=-0.0035  fills=5  inv=-5.0
α=-8.946e-7    pnl=-0.0035  fills=5  inv=-5.0
α=    -2e-6    pnl=+0.0003  fills=3  inv=-3.0
```

The signal is real but the AS-with-signals architecture (treat α as a
microprice-style reservation shift) cannot translate it into PnL on
this data. The shift moves both quotes by `α · OFI`; the resulting
asymmetric quoting locks us out of fills without compensating
profitability. A different architecture — for instance, **widening the
spread when |OFI| is extreme rather than skewing the mid** — would be
the next thing to try if continuing on PM-data strategy work.

---

## Calibration quality across other days / markets

| token / day | n_book | n_trade | σ | κ | R² | warnings |
|---|---:|---:|---:|---:|---:|---|
| Pistons YES 2026-05-16 | 1 147 | 1 117 | 6.4·10⁻³ | 342 | 0.015 | low R² |
| France WC YES 2026-05-17 | 616 | 296 | 3.4·10⁻⁵ | 1 966 | 0.001 | low R² |
| Thunder NBA YES 2026-05-17 | 193 | 91 | 0.0 | 200 | 0.000 | low n_book, low n_trade, σ=0, low R² |
| Spurs NBA YES 2026-05-17 | 181 | 119 | 4.7·10⁻⁴ | 1 073 | −0.004 | low n_book, low n_trade, low R² |
| **Spurs NBA YES 2026-05-21** | **3 356** | **3 251** | **1.6·10⁻⁴** | **1 463** | **+0.096** | **clean — only useful sample** |

Five of six market × day combinations produce calibration that
`scripts/calibrate_strategy.py` flags as unstable. The single clean
calibration is the NBA Finals game day on the Spurs YES token. This
strongly argues against trying to fit AS-family parameters on anything
shorter than a high-activity event window.

---

## Economics

CS / GLT delivered ~$0.30 PnL over the eleven hours of capture on
2026-05-21 at a $100 nominal bankroll. Scaled linearly:

| bankroll | PnL / day | annualised | notes |
|---|---:|---:|---|
| $100 | $0.30 | ~$110 | proof-of-concept |
| $1 000 | $3 | ~$1 100 | small hobby |
| $10 000 | $30 | ~$11 000 | starts being meaningful |
| $100 000 | $300 | ~$110 000 | size-limited by market depth |

Real returns will be lower because (a) only a fraction of days produce
the kind of flow Spurs had on 2026-05-21 and (b) inventory drift on
trending markets eats into the spread the maker can capture. The
~$0.30/day figure is best-case, not expected-case.

For comparison: the Sniper strategy on a separate codebase (Kalshi /
older Polymarket bot at `polymarket-trading-bot/`) showed 5.1 % per-snipe
win rate and +165 % ROI over 39 weeks of backtest, which is in a
materially different regime even granting backtest optimism.

---

## What this means for the project

1. **Phase 6 acceptance ("PBO < 0.3, DSR > 0.95")** was passed on
   synthetic data. The numbers above are NOT the validation suite — they
   are single-day backtests. Pointing the actual CPCV + DSR + PBO
   pipeline at real captures across multiple market×day folds is the
   next legitimate validation step before any live trading decision.
2. **The AS family is mismatched to PM-market microstructure** when the
   tick is large relative to σ·√T. CS-style symmetric quoting plus a
   weak signal overlay is a more promising architecture than chasing
   tighter AS calibration.
3. **The negative-α finding is the most useful piece of signal here**.
   A future strategy could treat large |OFI| not as a price-shift hint
   but as a microstructure-noise indicator — widen quotes when the
   market is being moved, stay tight when flow is two-sided.
4. **The framework works.** Capture, replay, simulator, kill switch,
   paper trader, calibration, and backtest all run end-to-end on real
   data and produce numbers. That was the original goal.

If the project continues, the next moves in increasing order of effort
are:

* Run `tests/integration/test_validation_suite.py` over real-data
  folds rather than synthetic.
* Build the proposed "widen-spread-on-extreme-OFI" overlay on top of
  ConstantSpread.
* Build `src/hft_pm/live/client_v2.py` to actually trade.

If the project stops here, the artifacts to point to are this document
plus `docs/hft_prediction_markets_EN.md` (theory) plus the test suite
(infrastructure correctness).
