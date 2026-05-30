"""Tests for the Stage-2.4 spanning regression (Task T2, spec §2.4 / §3.2).

The spanning test asks whether the momentum factor's mean return survives once
its exposure to a small set of benchmark regressors is partialled out: regress
the factor return on {equal-weighted market return, small-minus-big size control}
and report the **intercept (alpha)** with a Newey-West HAC t-stat (guide §2.4).
The size control is a TEST-ONLY regressor (a small-minus-big long/short on Artemis
``MC`` quintiles); it is never formed as a deployed portfolio (spec §2, §3.2).

The discriminating tests pin the three behaviours the spec names:
  - a factor that is *exactly* ``2×market`` (plus a constant) has alpha ≈ the
    constant and the market beta ≈ 2 — the market fully spans the slope part;
  - a factor *orthogonal* to the regressors keeps its full mean as alpha (the
    regressors explain none of it);
  - the HAC alpha t-stat matches ``statsmodels`` OLS-with-HAC to ~6 dp (the same
    independent-oracle check the §2.2 core uses).

The market/size builders are checked point-in-time over the ``(r, next_r]``
holding window and on the in-sample eligibility mask, mirroring the factor
formation's window convention. Every fixture is synthetic and offline.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from amom.stats.spanning import (
    build_market_return,
    build_size_control,
    spanning_alpha,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _holding_returns(values: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    """Long ``[date, symbol, holding_return]`` from a ``{symbol: [series]}`` dict."""
    n = len(next(iter(values.values())))
    idx = pd.date_range(start, periods=n, freq="D")
    rows = []
    for sym, series in values.items():
        for d, v in zip(idx, series):
            rows.append({"date": d, "symbol": sym, "holding_return": float(v)})
    return pd.DataFrame(rows)


def _universe_all_eligible(symbols: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long universe panel marking every symbol eligible on every date."""
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"date": d, "symbol": s, "eligible": True})
    return pd.DataFrame(rows)


def _mc_panel(values: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    """Long ``[date, symbol, mc]`` market-cap panel."""
    n = len(next(iter(values.values())))
    idx = pd.date_range(start, periods=n, freq="D")
    rows = []
    for sym, series in values.items():
        for d, v in zip(idx, series):
            rows.append({"date": d, "symbol": sym, "mc": float(v)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# spanning_alpha: 2×market => alpha ≈ const, market beta ≈ 2
# ---------------------------------------------------------------------------

def test_spanning_alpha_two_times_market_has_near_zero_alpha():
    rng = np.random.default_rng(11)
    n = 200
    market = rng.normal(0.01, 0.05, n)
    size = rng.normal(0.0, 0.04, n)
    const = 0.002
    # factor is exactly 2*market (+ a small constant) — market fully spans it.
    factor = const + 2.0 * market
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    y = pd.Series(factor, index=idx)
    X = pd.DataFrame({"market_return": market, "size_control": size}, index=idx)

    out = spanning_alpha(y, X, bandwidth=5)
    assert np.isclose(out["alpha"], const, atol=1e-9)
    assert np.isclose(out["betas"]["market_return"], 2.0, atol=1e-9)
    assert np.isclose(out["betas"]["size_control"], 0.0, atol=1e-6)
    assert out["n"] == n
    assert out["r2"] > 0.99  # market explains essentially all variation


# ---------------------------------------------------------------------------
# spanning_alpha: factor orthogonal to regressors keeps its mean as alpha
# ---------------------------------------------------------------------------

def test_spanning_alpha_orthogonal_factor_keeps_mean_as_alpha():
    rng = np.random.default_rng(22)
    n = 250
    market = rng.normal(0.0, 0.05, n)
    size = rng.normal(0.0, 0.05, n)
    # factor is pure noise around a positive mean, independent of the regressors.
    factor = rng.normal(0.03, 0.02, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    y = pd.Series(factor, index=idx)
    X = pd.DataFrame({"market_return": market, "size_control": size}, index=idx)

    out = spanning_alpha(y, X, bandwidth=4)
    # alpha recovers the factor mean to within sampling noise; betas ~ 0.
    assert np.isclose(out["alpha"], factor.mean(), atol=0.01)
    assert abs(out["betas"]["market_return"]) < 0.2
    assert abs(out["betas"]["size_control"]) < 0.2


# ---------------------------------------------------------------------------
# spanning_alpha: HAC alpha t-stat matches statsmodels OLS-with-HAC to ~6 dp
# ---------------------------------------------------------------------------

def test_spanning_alpha_tstat_matches_statsmodels_hac():
    rng = np.random.default_rng(33)
    n = 220
    market = rng.normal(0.0, 0.05, n)
    size = rng.normal(0.0, 0.05, n)
    factor = 0.004 + 0.7 * market - 0.3 * size + rng.normal(0.0, 0.02, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    y = pd.Series(factor, index=idx)
    X = pd.DataFrame({"market_return": market, "size_control": size}, index=idx)

    bandwidth = 6
    out = spanning_alpha(y, X, bandwidth=bandwidth)

    Xmat = sm.add_constant(X.to_numpy())
    oracle = sm.OLS(y.to_numpy(), Xmat).fit(
        cov_type="HAC", cov_kwds={"maxlags": bandwidth}
    )
    # params order: [const(alpha), market, size]
    assert np.isclose(out["alpha"], oracle.params[0], atol=1e-9)
    assert np.isclose(out["alpha_tstat"], oracle.tvalues[0], atol=1e-6, rtol=0.0)
    assert np.isclose(out["betas"]["market_return"], oracle.params[1], atol=1e-9)
    assert np.isclose(out["betas"]["size_control"], oracle.params[2], atol=1e-9)
    assert np.isclose(out["r2"], oracle.rsquared, atol=1e-9)


def test_spanning_alpha_drops_nonfinite_rows_and_reports_n():
    rng = np.random.default_rng(44)
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    y = pd.Series(rng.normal(0.01, 0.02, 20), index=idx)
    y.iloc[3] = np.nan  # one factor obs missing
    X = pd.DataFrame(
        {
            "market_return": rng.normal(0.0, 0.05, 20),
            "size_control": rng.normal(0.0, 0.05, 20),
        },
        index=idx,
    )
    X.iloc[5, 0] = np.nan  # one regressor obs missing
    out = spanning_alpha(y, X, bandwidth=1)
    assert out["n"] == 18  # 20 - 2 dropped rows
    assert np.isfinite(out["alpha"])
    assert np.isfinite(out["alpha_tstat"])


# ---------------------------------------------------------------------------
# build_market_return: equal-weighted eligible-universe return per window
# ---------------------------------------------------------------------------

def test_market_return_is_equal_weighted_eligible_universe_over_window():
    # 3 names, all eligible; window (r, next_r] = (idx[2], idx[5]].
    rets = _holding_returns(
        {
            "a": [0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0],
            "b": [0.0, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0, 0.0],
        }
    )
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    uni = _universe_all_eligible(["a", "b", "c"], idx)
    r, next_r = idx[2], idx[5]

    mr = build_market_return(rets, uni, pd.DatetimeIndex([r, next_r]))
    # Each coin's compounded window return: a=0.10, b=0.20, c=0.30; EW mean = 0.20.
    assert r in mr.index
    assert np.isclose(mr.loc[r, "market_return"], 0.20)


def test_market_return_excludes_ineligible_names_pointintime():
    rets = _holding_returns(
        {
            "a": [0.0, 0.0, 0.10, 0.0],
            "b": [0.0, 0.0, 0.50, 0.0],  # huge, but b is ineligible at r
        }
    )
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    uni = _universe_all_eligible(["a", "b"], idx)
    # b ineligible as-of r=idx[1] => excluded from the market basket.
    uni.loc[(uni["date"] == idx[1]) & (uni["symbol"] == "b"), "eligible"] = False
    r, next_r = idx[1], idx[3]

    mr = build_market_return(rets, uni, pd.DatetimeIndex([r, next_r]))
    # Only 'a' eligible => market return == a's window return (0.10), not blended.
    assert np.isclose(mr.loc[r, "market_return"], 0.10)


# ---------------------------------------------------------------------------
# build_size_control: small-minus-big over MC quintiles (test-only)
# ---------------------------------------------------------------------------

def test_size_control_is_small_minus_big_over_mc_quintiles():
    # 10 names; MC strictly increasing s0..s9 as-of r. With QUANTILE=0.20 the
    # small leg = bottom-2 {s0,s1}, big leg = top-2 {s8,s9}.
    syms = [f"s{i}" for i in range(10)]
    n_days = 6
    # Returns: small names rally +0.10 over the window, big names fall -0.10.
    series = {}
    for i, s in enumerate(syms):
        col = [0.0] * n_days
        col[3] = 0.10 if i < 5 else -0.10  # one in-window return on idx[3]
        series[s] = col
    rets = _holding_returns(series)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    uni = _universe_all_eligible(syms, idx)
    mc = _mc_panel({s: [float(i + 1) * 1e6] * n_days for i, s in enumerate(syms)})
    r, next_r = idx[2], idx[5]

    sc = build_size_control(rets, uni, mc, pd.DatetimeIndex([r, next_r]))
    # small leg {s0,s1} = +0.10, big leg {s8,s9} = -0.10 => SMB = 0.10 - (-0.10).
    assert r in sc.index
    assert np.isclose(sc.loc[r, "size_control"], 0.20)


def test_size_control_ranks_on_pointintime_mc_asof_r():
    # MC ordering FLIPS after r; the size legs must use MC as-of r only.
    syms = [f"s{i}" for i in range(10)]
    n_days = 6
    series = {s: [0.0] * n_days for s in syms}
    # small (low-MC as-of r) names s0,s1 rally; big s8,s9 fall.
    for i, s in enumerate(syms):
        series[s][3] = 0.10 if i < 5 else -0.10
    rets = _holding_returns(series)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    uni = _universe_all_eligible(syms, idx)
    r, next_r = idx[2], idx[5]

    mc_vals = {}
    for i, s in enumerate(syms):
        col = [float(i + 1) * 1e6] * n_days       # ascending as-of r
        # Flip the cross-section on every date strictly after r — must be ignored.
        for j in range(3, n_days):
            col[j] = float(10 - i) * 1e6
        mc_vals[s] = col
    mc = _mc_panel(mc_vals)

    sc = build_size_control(rets, uni, mc, pd.DatetimeIndex([r, next_r]))
    # If MC as-of r (ascending) is used: small={s0,s1}(+0.10), big={s8,s9}(-0.10).
    assert np.isclose(sc.loc[r, "size_control"], 0.20)
