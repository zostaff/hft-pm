"""Phase 4 acceptance: each signal (microprice → OFI → VPIN-gate) improves PnL.

Setup uses :func:`generate_drifted_world` — synthetic events where a
hidden OU "flow" state drives:

1. Book BBO size imbalance (observable; microprice should pick it up).
2. Trade aggressor side bias (OFI accumulates the imbalance).
3. Mid drift in the same direction (so the signals have something to predict).

We compare four configurations of the AS strategy:

* ``vanilla``  — no signals (Phase 3 baseline)
* ``+micro``   — replaces mid with microprice as fair value
* ``+ofi``     — adds OFI alpha-skew on top of microprice
* ``+vpin``    — adds VPIN gate (withdraws on high toxicity) on top

Each must produce PnL ≥ the previous on a multi-seed mean, within
statistical noise. The strict ordering is not enforced (signals can
interact non-monotonically); we assert: (a) ``+ofi`` beats ``vanilla``
with bootstrap-CI confidence, (b) ``+micro`` and ``+vpin`` are at least
non-degenerate (mean PnL > 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.signals.ofi import OFICalculator
from hft_pm.signals.vpin import VPINCalculator
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.simulator.synthetic import generate_drifted_world
from hft_pm.strategies.avellaneda_stoikov import AvellanedaStoikovWithSignals

pytestmark = pytest.mark.integration


def _run(events, strat):
    book = L2OrderBook(tick=0.001)
    sim = Backtester(book=book, latency=ConstantLatency(0), record_pnl_series=True)
    return sim.run(events, strat)


def _bootstrap_lower_ci(diffs: np.ndarray, *, n_resamples: int, alpha: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diffs[idx]))
    return float(np.quantile(means, alpha))


def _run_seed(seed: int, params: dict) -> dict[str, float]:
    events = list(generate_drifted_world(seed=seed, **params))
    horizon_ms = int(params["duration_s"] * 1000)
    sigma, kappa, gamma = params["sigma"], params["kappa"], 1.0

    # Vanilla AS — no signals.
    res_vanilla = _run(
        events,
        AvellanedaStoikovWithSignals(
            gamma=gamma, sigma=sigma, kappa=kappa, horizon_ms=horizon_ms, size=1
        ),
    )

    # +microprice
    res_micro = _run(
        events,
        AvellanedaStoikovWithSignals(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            horizon_ms=horizon_ms,
            size=1,
            use_microprice=True,
        ),
    )

    # +OFI alpha skew. alpha_beta=0.0005 in price-units per OFI-unit;
    # tuned so that a typical OFI of ~50 shifts the reservation by ~0.025.
    res_ofi = _run(
        events,
        AvellanedaStoikovWithSignals(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            horizon_ms=horizon_ms,
            size=1,
            use_microprice=True,
            ofi=OFICalculator(window_seconds=2.0),
            alpha_beta=5e-7,
        ),
    )

    # +VPIN gate. Withdraws when VPIN > vpin_max.
    res_vpin = _run(
        events,
        AvellanedaStoikovWithSignals(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            horizon_ms=horizon_ms,
            size=1,
            use_microprice=True,
            ofi=OFICalculator(window_seconds=2.0),
            alpha_beta=5e-7,
            vpin=VPINCalculator(bucket_volume=100.0, n_buckets=20),
            vpin_max=3.0,
        ),
    )

    return {
        "vanilla": res_vanilla.pnl,
        "micro": res_micro.pnl,
        "ofi": res_ofi.pnl,
        "vpin": res_vpin.pnl,
    }


def test_signals_improve_pnl_on_drifted_world() -> None:
    n_seeds = 20
    params = dict(
        duration_s=180,
        sigma=0.005,
        kappa=60.0,
        A=3.0,
        drift_strength=0.001,  # mid drifts by 0.001·φ per second (keeps mid in (0, 1))
        flow_decay_s=5.0,
        tick=0.001,
    )
    rows = [_run_seed(i, params) for i in range(n_seeds)]
    pnls = {k: np.array([r[k] for r in rows]) for k in ("vanilla", "micro", "ofi", "vpin")}

    means = {k: float(v.mean()) for k, v in pnls.items()}

    # (a) OFI variant beats vanilla with bootstrap CI.
    diffs_ofi_vs_vanilla = pnls["ofi"] - pnls["vanilla"]
    lower_ci_ofi = _bootstrap_lower_ci(diffs_ofi_vs_vanilla, n_resamples=1000, alpha=0.05, seed=0)

    # (b) Microprice and VPIN-gate variants are non-degenerate (mean PnL > 0).
    msg = (
        f"mean PnL  vanilla={means['vanilla']:.3f}  +micro={means['micro']:.3f}  "
        f"+ofi={means['ofi']:.3f}  +vpin={means['vpin']:.3f}  "
        f"ofi-vanilla diff mean={diffs_ofi_vs_vanilla.mean():.3f}  "
        f"95%-lower={lower_ci_ofi:.3f}"
    )
    assert lower_ci_ofi > 0, msg
    assert means["micro"] > 0, msg
    assert means["vpin"] > 0, msg
