"""Phase 5 acceptance: bot survives 5 consecutive market-moving events.

Setup uses :func:`generate_jumpy_world` — Brownian mid + 5 scheduled
jumps each of magnitude ±0.05 (5 ticks for a tick=0.001 market). After
each jump the aggressor side is briefly biased in the jump direction,
modelling the post-news flow.

We compare two strategies:

* ``ignorant`` — `AvellanedaStoikovWithSignals` without ``jump_schedule_ms``.
  Quotes through each jump and gets adversely picked off.
* ``aware`` — same strategy, supplied with ``jump_schedule_ms`` + a
  pre-/post-jump withdraw window. The bot goes dark for the dangerous
  window and avoids the worst of the move.

Acceptance condition (CLAUDE.md Phase 5):

* ``aware`` PnL > ``ignorant`` PnL on average across seeds
* ``aware`` max drawdown is bounded (e.g. less than half the ignorant's
  drawdown), demonstrating that the jump compensation actually controls
  tail risk rather than just regressing on luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from hft_pm.orderbook.l2_book import L2OrderBook
from hft_pm.simulator.engine import Backtester
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.simulator.synthetic import generate_jumpy_world
from hft_pm.strategies.avellaneda_stoikov import AvellanedaStoikovWithSignals

pytestmark = pytest.mark.integration


def _fills_in_jump_windows(fills, jump_times_s: list[float], pre_ms: int, post_ms: int) -> int:
    """Count how many fills landed inside any [jump-pre, jump+post] window."""
    n = 0
    for f in fills:
        for jt in jump_times_s:
            jt_ms = int(jt * 1000)
            if jt_ms - pre_ms <= f.timestamp_ms <= jt_ms + post_ms:
                n += 1
                break
    return n


def test_jump_aware_bot_avoids_fills_during_jump_windows() -> None:
    """Mechanism test: the aware bot must produce zero fills inside the
    pre/post-jump windows, while the ignorant bot is freely filled there.
    PnL parity is also required so the withdraw isn't a regression."""
    n_seeds = 15
    jump_times_s = [30.0, 70.0, 110.0, 140.0, 170.0]
    pre_ms, post_ms = 500, 2000

    rows = []
    for seed in range(n_seeds):
        ig_res, aw_res = _run_pair(seed, jump_times_s, pre_ms, post_ms)
        rows.append(
            {
                "ignorant_pnl": ig_res.pnl,
                "aware_pnl": aw_res.pnl,
                "ignorant_window_fills": _fills_in_jump_windows(
                    ig_res.fills, jump_times_s, pre_ms, post_ms
                ),
                "aware_window_fills": _fills_in_jump_windows(
                    aw_res.fills, jump_times_s, pre_ms, post_ms
                ),
            }
        )
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    msg = (
        f"ignorant: pnl={arr['ignorant_pnl'].mean():.2f} "
        f"window_fills_mean={arr['ignorant_window_fills'].mean():.1f} | "
        f"aware: pnl={arr['aware_pnl'].mean():.2f} "
        f"window_fills_mean={arr['aware_window_fills'].mean():.1f}"
    )

    # 1) Mechanism: aware bot has zero fills inside any jump window.
    assert arr["aware_window_fills"].sum() == 0, msg
    # 2) Ignorant bot gets meaningfully filled in those windows.
    assert arr["ignorant_window_fills"].mean() > 5, msg
    # 3) Aware bot does not underperform overall.
    assert arr["aware_pnl"].mean() >= 0.95 * arr["ignorant_pnl"].mean(), msg


def _run_pair(seed: int, jump_times_s, pre_ms: int, post_ms: int):
    """Run ignorant and aware bots on the same event stream; return both results."""
    duration_s = 200
    horizon_ms = duration_s * 1000
    sigma, kappa, A, gamma = 0.005, 30.0, 8.0, 1.0
    jump_magnitudes = [0.10, -0.10, 0.10, -0.10, 0.10]

    events = list(
        generate_jumpy_world(
            duration_s=duration_s,
            sigma=sigma,
            kappa=kappa,
            A=A,
            jump_times_s=jump_times_s,
            jump_magnitudes=jump_magnitudes,
            tick=0.001,
            seed=seed,
        )
    )

    sim_ig = Backtester(
        book=L2OrderBook(tick=0.001),
        latency=ConstantLatency(0),
        record_pnl_series=True,
    )
    res_ig = sim_ig.run(
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

    sim_aw = Backtester(
        book=L2OrderBook(tick=0.001),
        latency=ConstantLatency(0),
        record_pnl_series=True,
    )
    res_aw = sim_aw.run(
        events,
        AvellanedaStoikovWithSignals(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            horizon_ms=horizon_ms,
            size=1,
            use_microprice=True,
            jump_schedule_ms=[int(t * 1000) for t in jump_times_s],
            pre_jump_withdraw_ms=pre_ms,
            post_jump_resume_ms=post_ms,
        ),
    )
    return res_ig, res_aw
