"""Phase 3 acceptance: AS beats Constant-Spread on Sharpe.

Setup:

* Generate AS-conforming synthetic events at known (σ, κ, A).
* Constant-spread quoter at half-spread δ_CS = AS's q=0 half-spread —
  i.e. when AS has zero inventory and zero remaining horizon term, both
  strategies quote at the same width. The acceptance question is then
  whether AS's inventory skew gives it a Sharpe advantage.
* Run both on the same event stream (same RNG seed) for many seeds.
* Bootstrap the distribution of (Sharpe_AS − Sharpe_CS).
* Assert the lower 95 % CI bound is strictly positive.

We tune γ small enough that AS's reservation skew does not blow past
the unit interval at typical inventory levels, but large enough that
the inventory penalty actually changes quotes. The acceptance is
robust across reasonable γ in [0.5, 2].
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.simulator.synthetic import generate_as_world
from hft_pm.strategies.avellaneda_stoikov import AvellanedaStoikov
from hft_pm.strategies.constant_spread import ConstantSpread

pytestmark = pytest.mark.integration


def _sharpe_from_pnl_series(snapshots) -> float:
    """Annualised-style Sharpe from per-event PnL points.

    Sharpe = mean(ΔPnL) / std(ΔPnL). Returns 0 if std is 0 or fewer
    than 2 points.
    """
    if len(snapshots) < 2:
        return 0.0
    pnl = np.array([s.pnl for s in snapshots], dtype=np.float64)
    diffs = np.diff(pnl)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs) / sd)


def _bootstrap_lower_ci(diffs: np.ndarray, *, n_resamples: int, alpha: float, seed: int) -> float:
    """Lower bound of a (1−2α)-CI for the mean via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diffs[idx]))
    return float(np.quantile(means, alpha))


def _run_one_seed(
    *, seed: int, sigma: float, kappa: float, A: float, gamma: float, duration_s: int
):
    horizon_ms = duration_s * 1000
    half_cs = (2.0 / gamma) * math.log(1.0 + gamma / kappa)

    events = list(
        generate_as_world(
            duration_s=duration_s,
            sigma=sigma,
            kappa=kappa,
            A=A,
            seed=seed,
            tick=0.001,
        )
    )

    sim_as = Backtester(
        book=L2OrderBook(tick=0.001),
        latency=ConstantLatency(0),
        record_pnl_series=True,
    )
    as_strat = AvellanedaStoikov(
        gamma=gamma, sigma=sigma, kappa=kappa, horizon_ms=horizon_ms, size=1
    )
    res_as = sim_as.run(events, as_strat)

    sim_cs = Backtester(
        book=L2OrderBook(tick=0.001),
        latency=ConstantLatency(0),
        record_pnl_series=True,
    )
    cs_strat = ConstantSpread(half_spread=half_cs, size=1)
    res_cs = sim_cs.run(events, cs_strat)

    return _sharpe_from_pnl_series(res_as.pnl_series), _sharpe_from_pnl_series(res_cs.pnl_series)


def test_avellaneda_stoikov_beats_constant_spread_on_sharpe() -> None:
    """ACCEPTANCE: AS Sharpe > CS Sharpe with statistical confidence."""
    n_seeds = 30
    sigma = 0.005
    kappa = 60.0
    A = 3.0
    gamma = 1.0
    duration_s = 180

    sharpes_as = np.empty(n_seeds)
    sharpes_cs = np.empty(n_seeds)
    for i in range(n_seeds):
        s_as, s_cs = _run_one_seed(
            seed=i, sigma=sigma, kappa=kappa, A=A, gamma=gamma, duration_s=duration_s
        )
        sharpes_as[i] = s_as
        sharpes_cs[i] = s_cs

    diffs = sharpes_as - sharpes_cs
    lower_ci = _bootstrap_lower_ci(diffs, n_resamples=1000, alpha=0.05, seed=0)

    msg = (
        f"AS mean Sharpe = {sharpes_as.mean():.4f}; "
        f"CS mean Sharpe = {sharpes_cs.mean():.4f}; "
        f"diff mean = {diffs.mean():.4f}; "
        f"95% lower CI on diff = {lower_ci:.4f}"
    )
    assert lower_ci > 0, msg
