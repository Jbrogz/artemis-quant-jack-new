"""Tests for the Stage-2.7 stationary block bootstrap (arch — bootstrap of record).

Task T3 (plan §T3, spec §2.7). Two units under test:

* ``stationary_bootstrap_pvalue(returns, *, reps, block_size, seed)`` — a
  one-sided empirical p-value for ``H1: mean > 0`` built on
  ``arch.bootstrap.StationaryBootstrap`` (the **sole bootstrap of record**; the
  ported ``cmom`` bootstrap may only cross-check). The test resamples the
  *recentered* series so the bootstrap world satisfies ``H0: mean = 0`` and
  counts how often a null resample mean is at least as positive as the observed
  mean. A strongly-positive series gives a small p; a series whose sample mean
  is zero gives p ≈ 0.5; the result is deterministic under a fixed seed.

* ``disagrees(hac_p, boot_p, threshold)`` — the Newey-West and bootstrap
  verdicts fall on **opposite sides** of the (Bonferroni-adjusted) threshold,
  i.e. exactly one of the two p-values is ``<= threshold``. On disagreement the
  bootstrap is the reported verdict (spec §2.7); ``disagrees`` only flags it.
"""

import numpy as np
import pandas as pd

from amom.stats.bootstrap import disagrees, stationary_bootstrap_pvalue


# ---------------------------------------------------------------------------
# stationary_bootstrap_pvalue: one-sided empirical p for mean > 0
# ---------------------------------------------------------------------------

def test_positive_series_gives_small_pvalue():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.04, 0.02, 300))  # strong positive drift
    p = stationary_bootstrap_pvalue(r, reps=2000, block_size=4, seed=7)
    assert 0.0 <= p <= 1.0
    assert p < 0.05  # clear evidence of a positive mean


def test_zero_mean_series_gives_pvalue_near_half():
    # A series whose *sample* mean is exactly zero carries no evidence for
    # mean > 0, so the one-sided p sits around 0.5 (symmetric null).
    base = pd.Series(np.tile([0.03, -0.03], 200))
    assert np.isclose(base.mean(), 0.0)
    p = stationary_bootstrap_pvalue(base, reps=3000, block_size=4, seed=7)
    assert abs(p - 0.5) < 0.1


def test_negative_series_gives_large_pvalue():
    rng = np.random.default_rng(43)
    r = pd.Series(rng.normal(-0.04, 0.02, 300))
    p = stationary_bootstrap_pvalue(r, reps=2000, block_size=4, seed=7)
    assert p > 0.5  # no evidence for a positive mean


def test_deterministic_under_fixed_seed():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.03, 250))
    p1 = stationary_bootstrap_pvalue(r, reps=1500, block_size=5, seed=99)
    p2 = stationary_bootstrap_pvalue(r, reps=1500, block_size=5, seed=99)
    assert p1 == p2  # exact equality, not approximate


def test_different_seeds_can_differ():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0, 0.03, 250))
    p_a = stationary_bootstrap_pvalue(r, reps=1500, block_size=5, seed=1)
    p_b = stationary_bootstrap_pvalue(r, reps=1500, block_size=5, seed=2)
    # not asserting a specific value, only that the seed actually drives the RNG
    assert p_a != p_b


def test_pvalue_is_a_probability_and_drops_nans():
    rng = np.random.default_rng(3)
    vals = rng.normal(0.02, 0.02, 200)
    r = pd.Series(np.concatenate([vals, [np.nan, np.nan]]))
    p = stationary_bootstrap_pvalue(r, reps=1000, block_size=4, seed=7)
    assert 0.0 <= p <= 1.0


def test_degenerate_series_returns_nan():
    # Fewer than two observations -> no resampling possible.
    assert np.isnan(stationary_bootstrap_pvalue(pd.Series([0.01]), reps=1000, block_size=4, seed=7))
    assert np.isnan(stationary_bootstrap_pvalue(pd.Series([], dtype=float), reps=1000, block_size=4, seed=7))


# ---------------------------------------------------------------------------
# disagrees: opposite sides of the Bonferroni-adjusted threshold (spec §2.7)
# ---------------------------------------------------------------------------

def test_agree_when_both_below_threshold():
    # Both significant -> agreement.
    assert disagrees(0.001, 0.002, threshold=0.05 / 7) is False


def test_agree_when_both_above_threshold():
    # Both not significant -> agreement.
    assert disagrees(0.20, 0.30, threshold=0.05 / 7) is False


def test_disagree_when_hac_significant_but_bootstrap_not():
    # NW says survive, bootstrap says no -> disagreement (bootstrap overrides).
    assert disagrees(0.001, 0.20, threshold=0.05 / 7) is True


def test_disagree_when_bootstrap_significant_but_hac_not():
    assert disagrees(0.20, 0.001, threshold=0.05 / 7) is True


def test_threshold_boundary_is_inclusive_below():
    # p == threshold counts as "below" (survives), matching bonferroni_correction's
    # p <= threshold rule, so equal-to-threshold on both sides is agreement.
    thr = 0.05 / 7
    assert disagrees(thr, thr, threshold=thr) is False
    # one exactly on the threshold (survives) vs one above (fails) -> disagreement
    assert disagrees(thr, thr + 0.01, threshold=thr) is True


def test_disagree_returns_false_when_a_pvalue_is_nan():
    # A NaN p-value is an insufficient-data verdict, not a disagreement.
    assert disagrees(float("nan"), 0.001, threshold=0.05 / 7) is False
    assert disagrees(0.001, float("nan"), threshold=0.05 / 7) is False
