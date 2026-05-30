"""Stage-2.7 stationary block bootstrap — the bootstrap of record (spec §2.7).

``arch.bootstrap.StationaryBootstrap`` (Politis & Romano, 1994) is the **sole
bootstrap of record** for the empirical mean p-value; the ported ``cmom``
bootstrap (``amom.stats`` does not re-export it) may only cross-check, never be
the reported result.

The p-value tests ``H0: mean = 0`` against ``H1: mean > 0`` one-sided. The
series is resampled in geometric-length blocks (preserving the serial
correlation that 30-day overlapping holds induce); to impose the null we
resample the **recentered** series ``r - mean(r)`` so the bootstrap world has a
true mean of zero, then read off how often a null resample mean is at least as
positive as the observed mean. A strongly-positive series yields a small p; a
series with a zero sample mean yields p ≈ 0.5.

On a Newey-West / bootstrap **disagreement** — the two p-values falling on
opposite sides of the (Bonferroni-adjusted) threshold — the bootstrap is the
reported verdict (spec §2.7); ``disagrees`` only flags the condition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap


def _mean(sample: np.ndarray) -> float:
    return float(sample.mean())


def stationary_bootstrap_pvalue(
    returns: pd.Series,
    *,
    reps: int,
    block_size: int,
    seed: int,
) -> float:
    """One-sided stationary-bootstrap p-value for ``H1: mean > 0`` (spec §2.7).

    Resamples the recentered series (so the bootstrap world satisfies
    ``H0: mean = 0``) with ``arch.bootstrap.StationaryBootstrap`` and returns the
    fraction of null resample means at least as positive as the observed mean.
    The ``(count + 1) / (reps + 1)`` form keeps the p-value strictly inside
    ``(0, 1]`` and never reports an impossible exact zero.

    Args:
        returns: per-rebalance factor returns (NaNs are dropped).
        reps: number of bootstrap resamples.
        block_size: expected (geometric) block length; covers serial overlap.
        seed: RNG seed — fixing it makes the p-value deterministic.

    Returns:
        The empirical one-sided p-value in ``(0, 1]``; ``NaN`` if fewer than two
        finite observations remain (no resampling possible).
    """
    r = np.asarray(returns.dropna().values, dtype=float)
    if r.size < 2:
        return float("nan")

    observed_mean = r.mean()
    centered = r - observed_mean  # impose H0: bootstrap-world mean == 0

    bs = StationaryBootstrap(block_size, centered, seed=seed)
    null_means = np.asarray(bs.apply(_mean, reps), dtype=float).ravel()

    n_at_least = int(np.sum(null_means >= observed_mean))
    return (n_at_least + 1) / (reps + 1)


def disagrees(hac_p: float, boot_p: float, threshold: float) -> bool:
    """Do the HAC and bootstrap verdicts fall on opposite sides of ``threshold``?

    Disagreement (spec §2.7) is defined as exactly one of the two p-values being
    ``<= threshold`` (the survive/fail boundary), matching
    ``bonferroni_correction``'s ``p <= threshold`` survival rule. On a True
    return the caller takes the **bootstrap** result as the reported verdict.

    A non-finite p-value is an insufficient-data verdict, not a disagreement, so
    it returns ``False``.
    """
    if not (np.isfinite(hac_p) and np.isfinite(boot_p)):
        return False
    hac_significant = hac_p <= threshold
    boot_significant = boot_p <= threshold
    return hac_significant != boot_significant
