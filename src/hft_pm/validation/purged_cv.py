"""Purged combinatorial cross-validation (docs §11, López de Prado).

Standard k-fold CV leaks future information into training when samples
are sequentially correlated (which order-book / fill data always is).
Walk-forward CV avoids the leak but tests only one path, so it has
high variance and is gameable. Purged combinatorial CV (CPCV) sits in
between: it generates many (train, test) splits, each with explicit
*purging* (samples adjacent to test sets are dropped from training)
and *embargo* (samples immediately after a test set are also dropped).

Direct transcription of docs §11 with type hints and edge-case guards.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def purged_cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_window: int = 100,
    embargo: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate ``(train_idx, test_idx)`` splits with purging and embargo.

    Each sample is assigned to one of ``n_groups`` sequential groups.
    Each split picks ``n_test_groups`` of those groups as the test set
    (``C(n_groups, n_test_groups)`` splits total). The training set is
    every other index *except* a window of ``purge_window`` samples
    immediately before each test group and ``embargo`` samples
    immediately after.

    Parameters
    ----------
    n_samples:
        Total number of sequential samples.
    n_groups:
        Partition the samples into this many groups.
    n_test_groups:
        Test group count per split. Must be ``< n_groups``.
    purge_window, embargo:
        Number of adjacent samples to drop from training on each side
        of every test group.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_groups <= 1:
        raise ValueError("n_groups must be > 1")
    if n_test_groups <= 0 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in 1..n_groups-1")
    if purge_window < 0 or embargo < 0:
        raise ValueError("purge_window and embargo must be non-negative")
    if n_samples < n_groups:
        raise ValueError("n_samples must be >= n_groups")

    group_size = n_samples // n_groups
    group_ranges = [
        (i * group_size, (i + 1) * group_size if i < n_groups - 1 else n_samples)
        for i in range(n_groups)
    ]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate(
            [np.arange(group_ranges[g][0], group_ranges[g][1]) for g in test_combo]
        )
        forbidden = set(test_idx.tolist())
        for g in test_combo:
            lo, hi = group_ranges[g]
            forbidden.update(range(max(0, lo - purge_window), lo))
            forbidden.update(range(hi, min(n_samples, hi + embargo)))
        train_idx = np.array([i for i in range(n_samples) if i not in forbidden], dtype=np.int64)
        splits.append((train_idx, test_idx))
    return splits


__all__ = ["purged_cpcv_splits"]
