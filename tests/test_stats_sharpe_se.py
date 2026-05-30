"""Tests for the Lo (2002) Sharpe SE + HAC bandwidth rule + power labelling.

Task T1 (plan §T1, spec §2.0 / §2.3). Three units under test:

* ``maxlags_for(n_obs, holding_obs)`` = ``max(holding_obs - 1, ceil(n_obs**0.25))``
  -- the HAC bandwidth must cover the holding-period overlap (spec §2.0); the
  rule is kept general even though the non-overlapping 30-day series has
  ``holding_obs == 1`` and reduces to ``ceil(T**0.25)``.
* ``lo_sharpe_se(returns, periods_per_year)`` -> ``(sharpe, se, used_autocorr_correction)``.
  The iid SE is the closed form ``sqrt((1 + 0.5*SR**2)/T)`` (annualized via the
  naive ``sqrt(periods_per_year)`` scaling). When Ljung-Box flags autocorrelation
  the annualization is replaced by Lo's serial-correlation-corrected factor and
  the flag is True (spec §2.3 / §2.0 -- the trigger is automatic, not optional).
* ``effective_n_and_power(returns, holding_obs)`` -> effective non-overlapping n
  and an approximate power; a variant with effective ``n < MIN_EFFECTIVE_N`` is
  labelled ``"inconclusive (underpowered)"`` (distinct from "insignificant").
"""

import math

import numpy as np
import pandas as pd

from amom.config import MIN_EFFECTIVE_N
from amom.stats.sharpe_se import (
    effective_n_and_power,
    lo_sharpe_se,
    maxlags_for,
)


# ---------------------------------------------------------------------------
# maxlags_for: HAC bandwidth covers the holding-period overlap (spec §2.0)
# ---------------------------------------------------------------------------

def test_maxlags_reduces_to_t_quarter_when_non_overlapping():
    # holding_obs == 1 -> holding_obs - 1 == 0, so the rule is ceil(T**0.25).
    assert maxlags_for(99, holding_obs=1) == math.ceil(99 ** 0.25)  # ceil(3.155) == 4
    assert maxlags_for(16, holding_obs=1) == 2  # 16**0.25 == 2.0 exactly


def test_maxlags_covers_holding_overlap_when_it_dominates():
    # An overlapping hold of 30 obs forces maxlags >= 29 regardless of T**0.25.
    assert maxlags_for(99, holding_obs=30) == 29
    # When T is large and the hold is short the rate term ceil(T**0.25) wins;
    # the rule takes the max of the two terms.
    assert maxlags_for(100_000, holding_obs=2) == math.ceil(100_000 ** 0.25)  # 18 > 1


def test_maxlags_is_at_least_holding_obs_minus_one():
    # spec §7 compliance-matrix assertion: maxlags >= holding_period_obs - 1.
    for n in (50, 99, 250):
        for hold in (1, 5, 30):
            assert maxlags_for(n, holding_obs=hold) >= hold - 1


# ---------------------------------------------------------------------------
# lo_sharpe_se: iid closed form, and the autocorr trigger
# ---------------------------------------------------------------------------

def _iid_se_closed_form(returns: pd.Series, ppy: float) -> float:
    """Oracle iid SE of the *annualized* Sharpe: sqrt(ppy) * sqrt((1+0.5 SR_p^2)/T)."""
    r = returns.dropna().values
    t = len(r)
    sr_period = r.mean() / r.std(ddof=1)
    se_period = math.sqrt((1.0 + 0.5 * sr_period**2) / t)
    return math.sqrt(ppy) * se_period


def test_lo_se_matches_iid_closed_form_when_no_autocorr():
    rng = np.random.default_rng(2024)
    # iid normal -> Ljung-Box should NOT flag autocorrelation.
    r = pd.Series(rng.normal(0.01, 0.05, 400))
    sharpe, se, used = lo_sharpe_se(r, periods_per_year=12)
    assert used is False
    assert np.isclose(se, _iid_se_closed_form(r, 12), rtol=0.0, atol=1e-12)
    # sharpe returned is the annualized Sharpe
    expected_sr = (r.mean() * 12) / (r.std(ddof=1) * math.sqrt(12))
    assert np.isclose(sharpe, expected_sr, rtol=1e-9, atol=0.0)


def test_lo_se_triggers_autocorr_correction_on_ar1_series():
    # Strong AR(1) -> Ljung-Box flags it -> corrected SE, flag True, and the
    # corrected SE differs from the naive iid SE (positive autocorr inflates it).
    rng = np.random.default_rng(11)
    n = 500
    eps = rng.normal(scale=0.04, size=n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = 0.5 * r[t - 1] + eps[t]
    r = pd.Series(r + 0.01)  # nonzero mean
    sharpe, se, used = lo_sharpe_se(r, periods_per_year=12)
    assert used is True
    iid = _iid_se_closed_form(r, 12)
    assert not np.isclose(se, iid, rtol=1e-6, atol=0.0)
    assert se > iid  # positive serial correlation widens the SE


def test_lo_se_nan_for_degenerate_series():
    sharpe, se, used = lo_sharpe_se(pd.Series([0.01]), periods_per_year=12)
    assert np.isnan(sharpe) and np.isnan(se)
    assert used is False
    # zero variance -> NaN, never a divide-by-zero
    s2, se2, used2 = lo_sharpe_se(pd.Series([0.02, 0.02, 0.02, 0.02]), periods_per_year=12)
    assert np.isnan(s2) and np.isnan(se2)


# ---------------------------------------------------------------------------
# effective_n_and_power + underpowered labelling (spec §2.0)
# ---------------------------------------------------------------------------

def test_effective_n_non_overlapping_equals_obs():
    r = pd.Series(np.zeros(99))
    out = effective_n_and_power(r, holding_obs=1)
    assert out["effective_n"] == 99  # non-overlapping -> n unchanged


def test_effective_n_discounts_overlap():
    r = pd.Series(np.zeros(99))
    out = effective_n_and_power(r, holding_obs=30)
    # overlapping obs do not each count as one independent draw
    assert out["effective_n"] < 99
    assert out["effective_n"] >= 1


def test_underpowered_label_when_below_min_effective_n():
    n = MIN_EFFECTIVE_N - 1
    r = pd.Series(np.zeros(n))
    out = effective_n_and_power(r, holding_obs=1)
    assert out["underpowered"] is True
    assert out["label"] == "inconclusive (underpowered)"


def test_not_underpowered_at_or_above_min_effective_n():
    r = pd.Series(np.zeros(MIN_EFFECTIVE_N))
    out = effective_n_and_power(r, holding_obs=1)
    assert out["underpowered"] is False
    assert out["label"] != "inconclusive (underpowered)"


def test_power_is_a_probability():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.01, 0.05, 99))
    out = effective_n_and_power(r, holding_obs=1)
    assert 0.0 <= out["power"] <= 1.0
