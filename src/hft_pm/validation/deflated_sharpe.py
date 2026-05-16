"""Deflated Sharpe + Probability of Backtest Overfit + Diebold-Mariano.

All three are from López de Prado's *Advances in Financial Machine
Learning* (2018) and Bailey & López de Prado (2014).

Reading the metrics (docs §11):

* **DSR > 0.95** — strong evidence the observed Sharpe is genuinely
  positive after deflating for multiple-testing and non-normality.
* **PBO < 0.3** — robust strategy selection. PBO > 0.5 is pure overfit.
* **DM p-value < 0.05** vs a baseline — rejects equal predictive accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    sr_returns: np.ndarray,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns the probability that the *true* Sharpe is positive given:

    * ``observed_sr`` — empirical Sharpe of the candidate strategy
    * ``n_trials`` — total number of strategies / parameter sets tested
      (Sidak-type multiple-testing correction)
    * ``sr_returns`` — the per-period return series the Sharpe was
      computed on; its skew and excess-kurtosis enter the variance
      correction.
    """
    sr_returns = np.asarray(sr_returns, dtype=np.float64)
    T = len(sr_returns)
    if T < 4 or n_trials < 1:
        return 0.0

    gamma3 = float(stats.skew(sr_returns))
    gamma4 = float(stats.kurtosis(sr_returns, fisher=True))

    euler = 0.5772156649
    # Expected maximum of n_trials draws from N(0, 1) (Sidak approximation).
    if n_trials == 1:
        e_max_sr = 0.0
    else:
        e_max_sr = (1 - euler) * stats.norm.ppf(1 - 1 / n_trials) + euler * stats.norm.ppf(
            1 - 1 / (n_trials * np.e)
        )

    sr_var = (1 - gamma3 * observed_sr + (gamma4 / 4) * observed_sr**2) / (T - 1)
    if sr_var <= 0:
        return 0.0

    z = (observed_sr - e_max_sr) / np.sqrt(sr_var)
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfit(
    insample_sr: np.ndarray,
    oos_sr: np.ndarray,
) -> float:
    """López de Prado's PBO metric (docs §11).

    For each combinatorial CV split you produced a vector of per-strategy
    in-sample Sharpes and out-of-sample Sharpes. PBO is the fraction of
    splits where the in-sample winner ranked below the OOS median —
    i.e. the empirical probability that the chosen "best" strategy is
    actually overfit.

    Parameters
    ----------
    insample_sr, oos_sr:
        Arrays of shape ``(n_splits, n_strategies)``. Each row is one
        CV split; each column is one candidate strategy.
    """
    insample_sr = np.asarray(insample_sr, dtype=np.float64)
    oos_sr = np.asarray(oos_sr, dtype=np.float64)
    if insample_sr.shape != oos_sr.shape:
        raise ValueError("insample_sr and oos_sr must have matching shapes")
    if insample_sr.ndim != 2:
        raise ValueError("inputs must be 2-D arrays (n_splits, n_strategies)")
    n_splits, n_strategies = insample_sr.shape
    if n_strategies < 2:
        raise ValueError("need at least 2 candidate strategies for PBO")

    is_best_idx = insample_sr.argmax(axis=1)
    # Convert OOS Sharpes to ranks (low = bad, high = good).
    oos_ranks = oos_sr.argsort(axis=1).argsort(axis=1)
    oos_rank_of_is_best = oos_ranks[np.arange(n_splits), is_best_idx]
    median_rank = (n_strategies - 1) / 2.0
    return float((oos_rank_of_is_best < median_rank).mean())


def diebold_mariano(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
) -> dict[str, float | bool]:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests :math:`H_0: \\mathbb{E}[\\ell_a - \\ell_b] = 0` against a
    two-sided alternative. Negative statistic + small p-value means
    strategy A has lower loss (better).

    Parameters
    ----------
    losses_a, losses_b:
        Per-period loss series of equal length. For Sharpe-style
        comparisons, use ``losses = -returns`` or ``losses = returns**2``
        as appropriate.
    h:
        Forecast horizon. ``h=1`` uses the plain sample variance;
        ``h>1`` uses a Newey-West HAC correction.
    """
    losses_a = np.asarray(losses_a, dtype=np.float64)
    losses_b = np.asarray(losses_b, dtype=np.float64)
    if losses_a.shape != losses_b.shape:
        raise ValueError("loss series must have equal length")
    n = len(losses_a)
    if n < 10:
        return {"statistic": 0.0, "p_value": 1.0, "a_better": False}
    if h < 1:
        raise ValueError("h must be >= 1")

    d = losses_a - losses_b
    d_mean = float(d.mean())
    if h == 1:
        d_var = float(d.var(ddof=1) / n)
    else:
        gamma_0 = float(d.var(ddof=1))
        gamma_sum = 0.0
        for k in range(1, h):
            if k >= n:
                break
            cov_k = float(np.cov(d[:-k], d[k:], ddof=1)[0, 1])
            gamma_sum += (1 - k / h) * cov_k
        d_var = (gamma_0 + 2 * gamma_sum) / n

    if d_var <= 0:
        return {"statistic": 0.0, "p_value": 1.0, "a_better": False}

    dm_stat = d_mean / float(np.sqrt(d_var))
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return {
        "statistic": float(dm_stat),
        "p_value": p_value,
        "a_better": bool(dm_stat < 0),
    }


__all__ = [
    "deflated_sharpe_ratio",
    "diebold_mariano",
    "probability_of_backtest_overfit",
]
