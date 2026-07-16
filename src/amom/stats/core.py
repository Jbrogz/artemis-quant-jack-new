"""Pure statistical functions for factor evaluation.

Ported verbatim from the author's earlier ``factor_eval/stats.py`` (guide §2.1-§2.5).
Only the module docstring and provenance note differ; the numerics are
unchanged so the §2.2 statsmodels equivalence carries over identically.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 12) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Annualization factor (e.g. 12 for monthly, 26 for biweekly).
    """
    if len(returns) < 2:
        return np.nan
    mean_ret = returns.mean() * periods_per_year
    std_ret = returns.std() * np.sqrt(periods_per_year)
    return mean_ret / std_ret if std_ret > 0 else np.nan


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown from a cumulative return series.

    Returns a negative number (e.g. -0.25 means 25% drawdown).
    """
    wealth = 1 + cumulative_returns
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return dd.min()


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: float = 12,
    min_dd_floor: float = 0.05,
) -> float:
    """Calmar ratio: annualized return / max drawdown.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Annualization factor.
        min_dd_floor: Floor for max drawdown to prevent extreme ratios.
    """
    if len(returns) < 2:
        return np.nan
    cum = (1 + returns).cumprod() - 1
    ann_return = returns.mean() * periods_per_year
    dd = max_drawdown(cum)
    dd_floored = min(dd, -min_dd_floor)
    if dd_floored == 0:
        return np.nan
    return ann_return / abs(dd_floored)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 10,
    periods_per_year: float = 12,
) -> pd.Series:
    """Rolling annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns.
        window: Rolling window size (in periods).
        periods_per_year: Annualization factor.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(periods_per_year)


def hac_tstat(returns: pd.Series, bandwidth: int = 21) -> dict:
    """Newey-West HAC t-statistic for mean return being nonzero.

    Args:
        returns: Series of returns to test.
        bandwidth: Number of lags for Newey-West kernel.

    Returns:
        Dict with keys: mean, tstat, se, n_obs.
    """
    y = returns.dropna().values
    n = len(y)
    if n < bandwidth + 2:
        return {"mean": np.nan, "tstat": np.nan, "se": np.nan, "n_obs": n}

    mean = y.mean()
    resid = y - mean

    # Newey-West HAC variance estimator
    gamma_0 = np.dot(resid, resid) / n
    nw_var = gamma_0
    for lag in range(1, bandwidth + 1):
        weight = 1 - lag / (bandwidth + 1)  # Bartlett kernel
        gamma_j = np.dot(resid[lag:], resid[:-lag]) / n
        nw_var += 2 * weight * gamma_j

    se = np.sqrt(nw_var / n)
    tstat = mean / se if se > 0 else np.nan

    return {"mean": mean, "tstat": tstat, "se": se, "n_obs": n}


def newey_west_se(residuals: np.ndarray, X: np.ndarray, bandwidth: int) -> np.ndarray:
    """Newey-West HAC standard errors for OLS regression coefficients.

    Args:
        residuals: OLS residuals (n,).
        X: Design matrix (n, k).
        bandwidth: Number of lags for Bartlett kernel.

    Returns:
        Array of standard errors for each coefficient.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    # Gamma_0
    S = np.zeros((k, k))
    for t in range(n):
        xt = X[t : t + 1].T  # (k, 1)
        S += (residuals[t] ** 2) * (xt @ xt.T)

    # Gamma_j with Bartlett weights
    for lag in range(1, bandwidth + 1):
        weight = 1 - lag / (bandwidth + 1)
        G_j = np.zeros((k, k))
        for t in range(lag, n):
            xt = X[t : t + 1].T
            xt_lag = X[t - lag : t - lag + 1].T
            G_j += residuals[t] * residuals[t - lag] * (xt @ xt_lag.T)
        S += weight * (G_j + G_j.T)

    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


def ols_tstat_hac(
    y: np.ndarray,
    x: np.ndarray,
    bandwidth: int = 21,
) -> tuple[float, float]:
    """Univariate OLS with HAC t-stat.

    Args:
        y: Dependent variable.
        x: Independent variable.
        bandwidth: Newey-West bandwidth.

    Returns:
        (coefficient, t-statistic)
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < bandwidth + 5:
        return np.nan, np.nan

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = newey_west_se(resid, X, bandwidth)
    tstat = beta[1] / se[1] if se[1] > 0 else np.nan
    return beta[1], tstat


# --- Multiple-testing correction (guide section 2.5) ------------------------

# Harvey-Liu-Zhu (2016) t-stat tiers. With hundreds of factors mined in the
# literature, the conventional |t| > 2 bar over-rejects; a newly proposed factor
# needs roughly t > 3 to be credible. Guide section 2.5 instructs: treat
# 2 < t < 3 as suggestive, not conclusive.
HLZ_SUGGESTIVE_T = 2.0
HLZ_SIGNIFICANT_T = 3.0


def bonferroni_correction(pvalues, alpha: float = 0.05) -> dict:
    """Bonferroni family-wise error correction (guide section 2.5).

    Given m raw p-values from a family of m hypothesis tests, the Bonferroni
    per-test threshold is alpha / m; a test is rejected (the factor "survives")
    only if its p-value <= alpha / m. This controls the family-wise error rate
    (probability of one or more false rejections) at alpha.

    NaN p-values are treated as insufficient-data tests: dropped before counting
    m, never rejected, and reported with a NaN adjusted p-value.

    Args:
        pvalues: Sequence of raw p-values, one per test in the family.
        alpha: Family-wise error rate. Spec-fixed default 0.05.

    Returns:
        Dict with:
          m: number of valid (finite) tests -- the recorded "total number of
             tests run" (guide section 2.5).
          alpha: the family-wise level used.
          threshold: alpha / m (per-test rejection threshold; NaN if m == 0).
          reject: list[bool] aligned to the INPUT order; True where the test
             survives correction (p <= threshold). NaN inputs map to False.
          n_reject: number of survivors.
          p_adjusted: list[float] aligned to input; min(p * m, 1.0); NaN->NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return {
            "m": 0,
            "alpha": alpha,
            "threshold": float("nan"),
            "reject": [False] * len(p),
            "n_reject": 0,
            "p_adjusted": [float("nan")] * len(p),
        }
    threshold = alpha / m
    reject = [bool(finite[i] and p[i] <= threshold) for i in range(len(p))]
    p_adjusted = [
        float(min(p[i] * m, 1.0)) if finite[i] else float("nan")
        for i in range(len(p))
    ]
    return {
        "m": m,
        "alpha": alpha,
        "threshold": threshold,
        "reject": reject,
        "n_reject": int(sum(reject)),
        "p_adjusted": p_adjusted,
    }


def classify_tstat_hlz(tstat: float) -> str:
    """Harvey-Liu-Zhu (2016) significance tier for an upper-tail t-stat.

    The factor framework tests E[return] > 0 one-sided, so the tier is keyed on
    the (signed) upper-tail t-stat. Guide section 2.5: treat 2 < t < 3 as
    suggestive, not conclusive.

        t <= 2.0        -> "not_significant"  (includes wrong-sign / negative t)
        2.0 < t < 3.0   -> "suggestive"
        t >= 3.0        -> "significant"

    A non-finite t-stat returns "not_significant".
    """
    if not np.isfinite(tstat) or tstat <= HLZ_SUGGESTIVE_T:
        return "not_significant"
    if tstat < HLZ_SIGNIFICANT_T:
        return "suggestive"
    return "significant"
