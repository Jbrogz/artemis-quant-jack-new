"""Probability of Backtest Overfitting via CSCV (Lopez de Prado, 2015).

Ported verbatim from the author's earlier ``cmom/overfitting/pbo.py``.

Combinatorially-symmetric cross-validation splits the backtest into S equal
blocks, forms every way of choosing S/2 blocks as in-sample (the rest
out-of-sample), and checks whether the in-sample-best configuration stays
above-median out-of-sample. PBO is the fraction of splits where it does not;
PBO > 0.5 indicates the selection process is overfitting.
"""
from __future__ import annotations

import itertools

import numpy as np


def _sharpe_per_column(block: np.ndarray) -> np.ndarray:
    """Sharpe ratio of each column; zero-variance columns become NaN."""
    mean = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    sd = np.where(sd == 0.0, np.nan, sd)
    return mean / sd


def probability_of_backtest_overfitting(
    pnl, n_splits: int = 16
) -> float:
    """Return PBO for a (observations x trials) PnL matrix.

    Each column of `pnl` is one strategy configuration's per-period return.
    Raises ValueError if `pnl` holds non-finite values, or if any in-sample
    split is so degenerate that no trial is rankable (all constant).
    """
    matrix = np.asarray(pnl, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("pnl must be a 2-D (observations x trials) array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pnl contains non-finite values (NaN/inf)")
    n_obs, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError("need at least 2 trials")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even number >= 2")
    if n_obs < n_splits:
        raise ValueError("need at least n_splits observations")

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits = []
    for is_blocks in itertools.combinations(range(n_splits), half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[i] for i in is_blocks])
        oos_rows = np.concatenate(
            [blocks[i] for i in range(n_splits) if i not in is_set]
        )
        is_perf = _sharpe_per_column(matrix[is_rows])
        if np.all(np.isnan(is_perf)):
            raise ValueError(
                "an in-sample split has no rankable trials "
                "(every configuration is constant over those observations)"
            )
        oos_perf = _sharpe_per_column(matrix[oos_rows])
        best = int(np.nanargmax(is_perf))
        # If the in-sample-best configuration is constant out-of-sample its
        # OOS Sharpe is NaN; `nan <= nan` is False, so the rank would
        # silently collapse to 0 and the split be misclassified as
        # overfitting. Such a split is genuinely unrankable -- skip it.
        if np.isnan(oos_perf[best]):
            continue
        # out-of-sample relative rank of the in-sample-best configuration
        oos_rank = int(np.sum(oos_perf <= oos_perf[best]))   # 1 .. n_trials
        omega = oos_rank / (n_trials + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(np.log(omega / (1.0 - omega)))
    if not logits:
        raise ValueError(
            "no rankable splits: the in-sample-best configuration is "
            "constant out-of-sample in every combination"
        )
    logits = np.asarray(logits)
    return float(np.mean(logits <= 0.0))
