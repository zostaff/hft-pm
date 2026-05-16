# Disclaimer

This project is a **research and educational framework**, not a
production trading system. Read this in full before using any part of
this code with real money.

## No financial advice

Nothing in this repository constitutes financial, investment, or
trading advice. The strategies implemented here (Avellaneda-Stoikov,
GLT, OFI-skewed AS, jump-aware AS) are textbook market-making
algorithms; their suitability for any particular market, account size,
or risk tolerance is your responsibility to assess.

## What this code has and has not been validated against

- The **validation suite** (CPCV / DSR / PBO / delay / shuffle) has
  been run only on synthetic data we generate ourselves. The Phase 6
  acceptance thresholds passing on synthetic data **does not imply**
  that the strategies are profitable, safe, or low-risk on real
  Polymarket markets.
- The **WebSocket client** has been smoke-tested on live Polymarket
  feeds for under one hour.
- **No real order has ever been placed** through this code. The
  Polymarket V2 client wrapper is not implemented.

## Known limitations relevant to losing money

1. The Avellaneda-Stoikov closed form breaks down at prices near 0 or 1
   (docs §4.7). The logit-space variant for boundary-priced markets is
   not built — using the standard AS quotes there is unsafe.
2. The OFI alpha-skew coefficient (`alpha_beta`) shipped in the example
   config was calibrated on synthetic data. On real markets the slope
   may be different, near zero, or of the wrong sign.
3. The kill switches (`max_drawdown_pct`, `max_inventory`,
   `heartbeat_timeout_s`, `daily_loss_limit`) only trip when fed live
   state. They are not wired into any live runner because that runner
   does not exist yet.
4. The Polymarket V2 `/balance-allowance/update` call, mandatory after
   every funding event (docs §7.5), is not implemented anywhere in
   this codebase.

## 87 % of Polymarket wallets lose money

Per the source document this project is built from. The math
implemented here, applied carefully, is what puts a maker in the other
13 %. Applied carelessly, or applied without the missing pieces above,
it puts you back in the 87 %.

## Use of this code is at your own risk

By using this code you agree that the authors are not liable for any
losses, direct or indirect, that arise from its use.
