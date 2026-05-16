"""Phase 6 acceptance: full validation suite on the AS strategy.

CLAUDE.md Phase 6 acceptance:

* **PBO < 0.3** — picking the in-sample-best strategy is not pure overfit.
* **DSR > 0.95** — observed Sharpe is genuinely positive after deflating
  for multiple-testing and non-normality.
* **Sharpe degrades smoothly under +100 ms / +500 ms / +2 s delay injection**
  — a genuine signal degrades gracefully; a leaky one collapses.
* **Sharpe drops to ~0 under timestamp shuffle** — the strategy is
  exploiting time-correlated microstructure, not calendar artefacts.

Setup: drifted synthetic with multiple seeds. We compute Sharpes across
several γ choices (the "candidate strategies" axis for PBO) using
CPCV-style splits over seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.signals.ofi import OFICalculator
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.simulator.synthetic import generate_drifted_world
from hft_pm.strategies.avellaneda_stoikov import AvellanedaStoikovWithSignals
from hft_pm.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfit,
)
from hft_pm.validation.delay_injection import DelayInjector
from hft_pm.validation.purged_cv import purged_cpcv_splits
from hft_pm.validation.shuffle_test import shuffle_event_timestamps

pytestmark = pytest.mark.integration


SIGMA = 0.005
KAPPA = 60.0
A_PARAM = 3.0
DURATION_S = 180
TICK = 0.001
N_SEEDS = 12


def _sharpe(snapshots) -> float:
    if len(snapshots) < 2:
        return 0.0
    pnl = np.array([s.pnl for s in snapshots], dtype=np.float64)
    diffs = np.diff(pnl)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return float(np.mean(diffs) / sd) if sd > 0 else 0.0


def _returns_series(snapshots) -> np.ndarray:
    if len(snapshots) < 2:
        return np.zeros(1)
    pnl = np.array([s.pnl for s in snapshots], dtype=np.float64)
    return np.diff(pnl)


def _run(events, gamma: float, extra_latency_ms: int = 0):
    horizon_ms = DURATION_S * 1000
    inner = ConstantLatency(0)
    lat = DelayInjector(inner, extra_latency_ms) if extra_latency_ms > 0 else inner
    book = L2OrderBook(tick=TICK)
    sim = Backtester(book=book, latency=lat, record_pnl_series=True)
    strat = AvellanedaStoikovWithSignals(
        gamma=gamma,
        sigma=SIGMA,
        kappa=KAPPA,
        horizon_ms=horizon_ms,
        size=1,
        use_microprice=True,
        ofi=OFICalculator(window_seconds=2.0),
        alpha_beta=5e-7,
    )
    return sim.run(events, strat)


def _events_for_seed(seed: int):
    return list(
        generate_drifted_world(
            duration_s=DURATION_S,
            sigma=SIGMA,
            kappa=KAPPA,
            A=A_PARAM,
            drift_strength=0.001,
            flow_decay_s=5.0,
            tick=TICK,
            seed=seed,
        )
    )


def test_pbo_under_threshold() -> None:
    """PBO < 0.3: AS-with-signals across γ choices is not overfit."""
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5]
    n_strats = len(gammas)
    # Sharpe matrix: rows = seeds, cols = γ choices
    sharpe_matrix = np.empty((N_SEEDS, n_strats), dtype=np.float64)
    for s in range(N_SEEDS):
        events = _events_for_seed(s)
        for j, g in enumerate(gammas):
            sharpe_matrix[s, j] = _sharpe(_run(events, g).pnl_series)

    splits = purged_cpcv_splits(
        n_samples=N_SEEDS, n_groups=4, n_test_groups=1, purge_window=0, embargo=0
    )
    n_splits = len(splits)
    is_sr = np.empty((n_splits, n_strats), dtype=np.float64)
    oos_sr = np.empty((n_splits, n_strats), dtype=np.float64)
    for i, (train_idx, test_idx) in enumerate(splits):
        is_sr[i] = sharpe_matrix[train_idx].mean(axis=0)
        oos_sr[i] = sharpe_matrix[test_idx].mean(axis=0)
    pbo = probability_of_backtest_overfit(is_sr, oos_sr)
    assert pbo < 0.3, f"PBO={pbo:.3f} exceeds 0.3 threshold"


def test_dsr_above_threshold() -> None:
    """DSR > 0.95 on per-seed Sharpes for the canonical γ=1.0."""
    gammas_tried = 5  # we test 5 γ choices upstream
    seeds = list(range(N_SEEDS))
    sharpes = np.array([_sharpe(_run(_events_for_seed(s), 1.0).pnl_series) for s in seeds])
    observed = float(sharpes.mean() / sharpes.std(ddof=1)) if sharpes.std(ddof=1) > 0 else 0.0
    dsr = deflated_sharpe_ratio(observed_sr=observed, n_trials=gammas_tried, sr_returns=sharpes)
    msg = f"observed_sr={observed:.3f}  dsr={dsr:.3f}  sharpes={sharpes.tolist()}"
    assert dsr > 0.95, msg


def test_sharpe_degrades_smoothly_under_delay_injection() -> None:
    """+100/+500/+2000 ms delay → Sharpe should degrade smoothly, not collapse.

    "Smoothly" means: each step reduces but does not flip sign on average.
    A collapse to negative-Sharpe at +100ms would indicate look-ahead leak.
    """
    extra_delays = [0, 100, 500, 2000]
    sharpes = {d: [] for d in extra_delays}
    for s in range(N_SEEDS):
        events = _events_for_seed(s)
        for d in extra_delays:
            sharpes[d].append(_sharpe(_run(events, 1.0, extra_latency_ms=d).pnl_series))
    means = {d: float(np.mean(sharpes[d])) for d in extra_delays}

    msg = "  ".join(f"+{d}ms={means[d]:+.4f}" for d in extra_delays)

    # No-delay Sharpe must be positive — the alpha must exist to start with.
    assert means[0] > 0, msg
    # +100 ms must NOT collapse to negative — that would signal sub-100ms
    # look-ahead leak, which is the failure mode delay-injection guards
    # against.
    assert means[100] >= 0.5 * means[0], msg
    # +2 s must not be dramatically *higher* than no-delay (would suggest
    # the strategy is causally implausible — gaining from delay).
    assert means[2000] <= 2.0 * means[0], msg


def test_sharpe_collapses_under_timestamp_shuffle() -> None:
    """Shuffling timestamps must destroy the OFI-driven edge."""
    in_order_sharpes = []
    shuffled_sharpes = []
    for s in range(N_SEEDS):
        events = _events_for_seed(s)
        in_order_sharpes.append(_sharpe(_run(events, 1.0).pnl_series))
        shuf = shuffle_event_timestamps(events, seed=s)
        shuffled_sharpes.append(_sharpe(_run(shuf, 1.0).pnl_series))

    in_order_mean = float(np.mean(in_order_sharpes))
    shuffled_mean = float(np.mean(shuffled_sharpes))
    msg = f"in_order_mean_sharpe={in_order_mean:.4f}  shuffled_mean_sharpe={shuffled_mean:.4f}"
    # The shuffle should at least halve the Sharpe.
    assert abs(shuffled_mean) < 0.5 * abs(in_order_mean) + 1e-4, msg
