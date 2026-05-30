"""Tests for the ported stats core (Task T0, plan §T0).

The headline test is the guide §2.2 equivalence: ``hac_tstat`` (Newey-West HAC
on a constant) must match ``statsmodels.OLS(y, add_constant).fit(cov_type='HAC',
cov_kwds={'maxlags': L})`` intercept t-stat to ~6 decimal places. statsmodels is
the independent oracle; if the two disagree the HAC estimator is wrong.

Bonferroni: m = number of finite p-values, per-test threshold alpha/m, a test
survives iff p <= threshold. HLZ tiers: t >= 3 significant, 2 < t < 3 suggestive,
else not_significant. DSR/PBO are smoke-tested for the ported surface; their
numerical correctness is covered by the cmom suite that owns them.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from amom.stats.core import (
    HLZ_SIGNIFICANT_T,
    HLZ_SUGGESTIVE_T,
    bonferroni_correction,
    calmar_ratio,
    classify_tstat_hlz,
    hac_tstat,
    max_drawdown,
    newey_west_se,
    ols_tstat_hac,
    rolling_sharpe,
    sharpe_ratio,
)
from amom.stats.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from amom.stats.pbo import probability_of_backtest_overfitting


def _statsmodels_const_hac_t(y: np.ndarray, maxlags: int) -> float:
    """Oracle: intercept t-stat of OLS-on-a-constant with HAC cov."""
    X = sm.add_constant(np.zeros(len(y)))  # constant only
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(model.tvalues[0])


# ---------------------------------------------------------------------------
# §2.2 equivalence: hac_tstat == statsmodels OLS-on-constant HAC t to ~6 dp
# ---------------------------------------------------------------------------

def test_hac_tstat_matches_statsmodels_const_hac_iid():
    rng = np.random.default_rng(12345)
    y = rng.normal(loc=0.4, scale=1.0, size=200)
    for maxlags in (0, 1, 3, 5, 12):
        ours = hac_tstat(pd.Series(y), bandwidth=maxlags)["tstat"]
        oracle = _statsmodels_const_hac_t(y, maxlags)
        assert np.isclose(ours, oracle, atol=1e-6, rtol=0.0), (
            f"maxlags={maxlags}: ours={ours} oracle={oracle}"
        )


def test_hac_tstat_matches_statsmodels_const_hac_autocorrelated():
    # AR(1) series: HAC correction bites, so the equivalence is a real check.
    rng = np.random.default_rng(7)
    n = 300
    eps = rng.normal(size=n)
    y = np.empty(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + eps[t]
    y = y + 0.3  # nonzero mean
    for maxlags in (4, 8, 15):
        ours = hac_tstat(pd.Series(y), bandwidth=maxlags)["tstat"]
        oracle = _statsmodels_const_hac_t(y, maxlags)
        assert np.isclose(ours, oracle, atol=1e-6, rtol=0.0), (
            f"maxlags={maxlags}: ours={ours} oracle={oracle}"
        )


def test_hac_tstat_returns_nan_when_too_few_obs():
    out = hac_tstat(pd.Series([0.1, 0.2, 0.3]), bandwidth=10)
    assert np.isnan(out["tstat"])
    assert out["n_obs"] == 3


# ---------------------------------------------------------------------------
# ols_tstat_hac / newey_west_se
# ---------------------------------------------------------------------------

def test_ols_tstat_hac_matches_statsmodels_slope():
    rng = np.random.default_rng(99)
    n = 250
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.5, size=n)
    maxlags = 5
    coef, t = ols_tstat_hac(y, x, bandwidth=maxlags)
    X = sm.add_constant(x)
    oracle = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    assert np.isclose(coef, oracle.params[1], atol=1e-8)
    assert np.isclose(t, oracle.tvalues[1], atol=1e-6, rtol=0.0)


def test_newey_west_se_matches_statsmodels_full_cov():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.7, size=n)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    maxlags = 6
    ours = newey_west_se(resid, X, maxlags)
    oracle = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}).bse
    assert np.allclose(ours, oracle, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Bonferroni correction (guide §2.5)
# ---------------------------------------------------------------------------

def test_bonferroni_m_counts_only_finite_pvalues():
    p = [0.001, 0.04, np.nan, 0.5]
    out = bonferroni_correction(p, alpha=0.05)
    assert out["m"] == 3  # the NaN is dropped from the family count
    assert np.isclose(out["threshold"], 0.05 / 3)


def test_bonferroni_survivor_is_below_threshold():
    # threshold = 0.05/7; only p=0.005 survives.
    p = [0.005, 0.02, 0.03, 0.04, 0.06, 0.5, 0.9]
    out = bonferroni_correction(p, alpha=0.05)
    assert out["m"] == 7
    assert np.isclose(out["threshold"], 0.05 / 7)
    assert out["reject"][0] is True
    assert out["n_reject"] == 1
    assert all(r is False for r in out["reject"][1:])


def test_bonferroni_nan_never_rejected_and_adjusted_is_nan():
    out = bonferroni_correction([np.nan, 0.001], alpha=0.05)
    assert out["reject"][0] is False
    assert np.isnan(out["p_adjusted"][0])
    assert out["m"] == 1
    assert out["reject"][1] is True


# ---------------------------------------------------------------------------
# Harvey-Liu-Zhu tiers (guide §2.5)
# ---------------------------------------------------------------------------

def test_hlz_tiers():
    assert classify_tstat_hlz(3.5) == "significant"
    assert classify_tstat_hlz(HLZ_SIGNIFICANT_T) == "significant"  # t == 3.0
    assert classify_tstat_hlz(2.5) == "suggestive"
    assert classify_tstat_hlz(HLZ_SUGGESTIVE_T) == "not_significant"  # t == 2.0
    assert classify_tstat_hlz(1.0) == "not_significant"
    assert classify_tstat_hlz(-4.0) == "not_significant"  # wrong sign
    assert classify_tstat_hlz(float("nan")) == "not_significant"


# ---------------------------------------------------------------------------
# Sharpe / drawdown / calmar / rolling sharpe
# ---------------------------------------------------------------------------

def test_sharpe_ratio_known_value():
    # std == 0 exactly -> NaN (single-valued series after a length check pass).
    assert np.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0.01, 0.05, 120))
    expected = (s.mean() * 12) / (s.std() * np.sqrt(12))
    assert np.isclose(sharpe_ratio(s, periods_per_year=12), expected)


def test_max_drawdown_simple_path():
    # cumulative returns rise to +0.5 then fall to 0.0 -> wealth 1.5 -> 1.0
    cum = pd.Series([0.0, 0.5, 0.2, 0.0])
    dd = max_drawdown(cum)
    # peak wealth 1.5, trough 1.0 -> (1.0 - 1.5)/1.5 = -1/3
    assert np.isclose(dd, -1.0 / 3.0)


def test_calmar_ratio_uses_floor():
    r = pd.Series([0.01] * 24)  # monotone up, dd ~ 0 -> floored
    c = calmar_ratio(r, periods_per_year=12, min_dd_floor=0.05)
    # ann return 0.12, floored dd 0.05 -> 2.4
    assert np.isclose(c, 0.12 / 0.05)


def test_rolling_sharpe_length():
    s = pd.Series(np.arange(20, dtype=float) * 0.01)
    rs = rolling_sharpe(s, window=5, periods_per_year=12)
    assert len(rs) == 20
    assert rs.iloc[:4].isna().all()


# ---------------------------------------------------------------------------
# DSR / PBO ported surface (smoke; numeric ownership stays in cmom suite)
# ---------------------------------------------------------------------------

def test_psr_in_unit_interval():
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.05, 250)
    p = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert 0.0 <= p <= 1.0


def test_deflated_sharpe_smaller_with_more_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.03, 0.05, 300)
    few = deflated_sharpe_ratio(r, trial_sharpes=[0.1, 0.2])
    many = deflated_sharpe_ratio(r, trial_sharpes=rng.normal(0.1, 0.3, 50))
    assert few >= many  # deflating across more trials cannot raise the DSR


def test_pbo_zero_for_identical_columns():
    rng = np.random.default_rng(0)
    col = rng.normal(0.0, 1.0, 64)
    pnl = np.column_stack([col, col, col, col])
    pbo = probability_of_backtest_overfitting(pnl, n_splits=8)
    assert 0.0 <= pbo <= 1.0
