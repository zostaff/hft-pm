"""Unit tests for hft_pm.validation.*"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hft_pm.data.schemas import BookEvent, PriceLevel
from hft_pm.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    diebold_mariano,
    probability_of_backtest_overfit,
)
from hft_pm.validation.delay_injection import DelayInjector
from hft_pm.validation.purged_cv import purged_cpcv_splits
from hft_pm.validation.shuffle_test import shuffle_event_timestamps

# ----------------------------------------------------------------------
# purged_cpcv_splits
# ----------------------------------------------------------------------


def test_cpcv_split_count_matches_combinatorial_choose() -> None:
    splits = purged_cpcv_splits(n_samples=1000, n_groups=6, n_test_groups=2)
    expected = math.comb(6, 2)
    assert len(splits) == expected


def test_cpcv_train_and_test_disjoint() -> None:
    splits = purged_cpcv_splits(n_samples=1000, n_groups=5, n_test_groups=1)
    for train_idx, test_idx in splits:
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0


def test_cpcv_purge_excludes_adjacent_samples() -> None:
    splits = purged_cpcv_splits(
        n_samples=1000, n_groups=5, n_test_groups=1, purge_window=20, embargo=10
    )
    for train_idx, test_idx in splits:
        test_lo, test_hi = test_idx.min(), test_idx.max()
        # No training sample in [test_lo - purge_window, test_lo).
        assert not any(test_lo - 20 <= i < test_lo for i in train_idx)
        # No training sample in (test_hi, test_hi + embargo].
        assert not any(test_hi < i <= test_hi + 10 for i in train_idx)


def test_cpcv_validates_inputs() -> None:
    with pytest.raises(ValueError):
        purged_cpcv_splits(n_samples=0)
    with pytest.raises(ValueError):
        purged_cpcv_splits(n_samples=100, n_groups=1)
    with pytest.raises(ValueError):
        purged_cpcv_splits(n_samples=100, n_groups=5, n_test_groups=5)
    with pytest.raises(ValueError):
        purged_cpcv_splits(n_samples=100, n_groups=5, purge_window=-1)
    with pytest.raises(ValueError):
        purged_cpcv_splits(n_samples=3, n_groups=10)


# ----------------------------------------------------------------------
# DSR
# ----------------------------------------------------------------------


def test_dsr_high_for_strong_signal_few_trials() -> None:
    # SR must clearly exceed the Sidak expected-max-of-n_trials,
    # which is sqrt(2 ln n_trials) ≈ 1.79 for n_trials=5. Pick a
    # mean/std ratio of 3 so the observed SR sits well above that.
    rng = np.random.default_rng(0)
    returns = rng.normal(3.0, 1.0, size=500)
    observed_sr = returns.mean() / returns.std(ddof=1)
    dsr = deflated_sharpe_ratio(observed_sr, n_trials=5, sr_returns=returns)
    assert dsr > 0.95


def test_dsr_low_for_moderate_signal_under_many_trials() -> None:
    """The same SR is "weaker" evidence when many trials were tried."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.5, 1.0, size=500)
    observed_sr = returns.mean() / returns.std(ddof=1)
    dsr_few = deflated_sharpe_ratio(observed_sr, n_trials=2, sr_returns=returns)
    dsr_many = deflated_sharpe_ratio(observed_sr, n_trials=10000, sr_returns=returns)
    assert dsr_few > dsr_many


def test_dsr_low_for_weak_signal_many_trials() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 1.0, size=500)  # zero true mean
    observed_sr = returns.mean() / returns.std(ddof=1)
    dsr = deflated_sharpe_ratio(observed_sr, n_trials=1000, sr_returns=returns)
    # Zero mean + many trials → no evidence of positive Sharpe.
    assert dsr < 0.5


def test_dsr_handles_short_series() -> None:
    assert deflated_sharpe_ratio(2.0, n_trials=10, sr_returns=np.array([0.1, 0.2])) == 0.0


# ----------------------------------------------------------------------
# PBO
# ----------------------------------------------------------------------


def test_pbo_zero_when_in_sample_ranks_match_oos() -> None:
    # Perfect correspondence: IS-best is also OOS-best every split.
    n_splits, n_strategies = 10, 5
    rng = np.random.default_rng(0)
    is_sr = rng.normal(0, 1, size=(n_splits, n_strategies))
    oos_sr = is_sr.copy()  # identical rankings
    assert probability_of_backtest_overfit(is_sr, oos_sr) == 0.0


def test_pbo_high_when_rankings_anti_correlated() -> None:
    n_splits, n_strategies = 20, 6
    rng = np.random.default_rng(0)
    is_sr = rng.normal(0, 1, size=(n_splits, n_strategies))
    oos_sr = -is_sr  # invert: IS-best becomes OOS-worst
    assert probability_of_backtest_overfit(is_sr, oos_sr) > 0.7


def test_pbo_validates_shapes() -> None:
    with pytest.raises(ValueError):
        probability_of_backtest_overfit(np.zeros((5, 3)), np.zeros((5, 4)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfit(np.zeros((5,)), np.zeros((5,)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfit(np.zeros((5, 1)), np.zeros((5, 1)))


# ----------------------------------------------------------------------
# Diebold-Mariano
# ----------------------------------------------------------------------


def test_dm_detects_a_better() -> None:
    rng = np.random.default_rng(0)
    losses_a = rng.normal(1.0, 0.5, size=200)
    losses_b = rng.normal(2.0, 0.5, size=200)  # B strictly worse
    res = diebold_mariano(losses_a, losses_b)
    assert res["a_better"] is True
    assert res["p_value"] < 0.05


def test_dm_no_difference_when_identical() -> None:
    rng = np.random.default_rng(0)
    losses = rng.normal(1.0, 0.5, size=200)
    res = diebold_mariano(losses, losses.copy())
    # Identical series → statistic 0 → p-value 1.
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0


def test_dm_short_series_returns_nondecisive() -> None:
    res = diebold_mariano(np.zeros(5), np.ones(5))
    assert res["p_value"] == 1.0


def test_dm_validates_inputs() -> None:
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(20), np.zeros(20), h=0)


# ----------------------------------------------------------------------
# DelayInjector
# ----------------------------------------------------------------------


class _FakeLatency:
    def sample(self, t_decision_ms: int) -> int:
        return t_decision_ms + 50


def test_delay_injector_adds_constant() -> None:
    di = DelayInjector(inner=_FakeLatency(), extra_ms=200)
    assert di.sample(1000) == 1000 + 50 + 200


def test_delay_injector_rejects_negative() -> None:
    with pytest.raises(ValueError):
        DelayInjector(inner=_FakeLatency(), extra_ms=-1)


# ----------------------------------------------------------------------
# shuffle_event_timestamps
# ----------------------------------------------------------------------


def _book(ts_ms: int) -> BookEvent:
    return BookEvent(
        asset_id="x",
        market="0xy",
        timestamp_ms=ts_ms,
        recv_ts_ms=ts_ms,
        bids=[PriceLevel(price=0.49, size=10.0)],
        asks=[PriceLevel(price=0.51, size=10.0)],
    )


def test_shuffle_preserves_event_count_and_timestamp_set() -> None:
    events = [_book(i * 100) for i in range(10)]
    shuffled = shuffle_event_timestamps(events, seed=0)
    assert len(shuffled) == len(events)
    assert sorted(e.timestamp_ms for e in shuffled) == sorted(e.timestamp_ms for e in events)


def test_shuffle_changes_event_to_timestamp_pairing() -> None:
    # Build events with distinct bid prices so we can detect re-pairing.
    events = [
        BookEvent(
            asset_id="x",
            market="0xy",
            timestamp_ms=i * 100,
            recv_ts_ms=i * 100,
            bids=[PriceLevel(price=0.40 + i * 0.01, size=10.0)],
            asks=[PriceLevel(price=0.60, size=10.0)],
        )
        for i in range(20)
    ]
    shuffled = shuffle_event_timestamps(events, seed=42)
    pairing_before = [(e.timestamp_ms, e.bids[0].price) for e in events]
    pairing_after = [(e.timestamp_ms, e.bids[0].price) for e in shuffled]
    # At least some pairs must have changed.
    assert pairing_before != pairing_after


def test_shuffle_is_monotonic_after_resort() -> None:
    events = [_book(i * 100) for i in range(50)]
    shuffled = shuffle_event_timestamps(events, seed=1)
    ts = [e.timestamp_ms for e in shuffled]
    assert ts == sorted(ts)


def test_shuffle_seed_reproducible() -> None:
    events = [_book(i * 100) for i in range(20)]
    a = shuffle_event_timestamps(events, seed=7)
    b = shuffle_event_timestamps(events, seed=7)
    assert [e.timestamp_ms for e in a] == [e.timestamp_ms for e in b]


def test_shuffle_handles_empty_stream() -> None:
    assert shuffle_event_timestamps([]) == []
