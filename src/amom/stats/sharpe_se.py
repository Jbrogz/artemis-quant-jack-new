"""Lo (2002) Sharpe-ratio standard error, HAC bandwidth rule, power labelling.

Stage-2 conventions (spec §2.0 / §2.3). These are the autocorrelation-aware
additions on top of the ported ``stats.core`` HAC machinery; they are NEW (no
counterpart in the earlier ``factor_eval.stats``).

* ``maxlags_for`` -- the HAC bandwidth must cover holding-period overlap, not
  just satisfy the asymptotic ``T**0.25`` rate, or the Newey-West SE
  under-corrects (the exact downward-SE bias the guide forbids, §2.1).
* ``lo_sharpe_se`` -- Sharpe ratios are never reported without a standard error.
  The iid SE is Lo's closed form ``sqrt((1 + 0.5*SR**2)/T)``; when the
  return series is autocorrelated (Ljung-Box flags it automatically) the SE is
  inflated by the autocovariance structure of the mean estimator, so positive
  serial correlation widens the SE rather than spuriously shrinking it.
* ``effective_n_and_power`` -- few non-overlapping draws => low power. A variant
  with effective ``n < MIN_EFFECTIVE_N`` is "inconclusive (underpowered)", a
  distinct verdict from "insignificant" (honesty over false nulls, spec §2.0).

All functions are pure and perform no I/O.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats as _scs
from statsmodels.stats.diagnostic import acorr_ljungbox

from amom.config import MIN_EFFECTIVE_N

# Significance level for the automatic Ljung-Box autocorrelation trigger and for
# the one-sided power approximation. Pinned by convention (the guide's 5% bar),
# not tuned to a result.
_AUTOCORR_ALPHA = 0.05
_POWER_ALPHA = 0.05


def maxlags_for(n_obs: int, holding_obs: int) -> int:
    """HAC Newey-West bandwidth that covers the holding-period overlap (spec §2.0).

    ``maxlags = max(holding_obs - 1, ceil(n_obs ** 0.25))``.

    The first term covers the serial dependence induced by overlapping holding
    windows (``holding_obs`` observations per hold => up to ``holding_obs - 1``
    lags of mechanical overlap); the second is the standard ``T**(1/4)`` growth
    rate. For the non-overlapping 30-day series ``holding_obs == 1`` and the rule
    reduces to ``ceil(n_obs ** 0.25)`` -- but the rule is kept general so an
    overlapping configuration is bandwidth-covered, not silently under-corrected.

    Args:
        n_obs: number of return observations (T).
        holding_obs: holding-window length in observation units (1 if the series
            is already non-overlapping).

    Returns:
        Integer bandwidth >= ``holding_obs - 1``.
    """
    rate_term = math.ceil(n_obs ** 0.25)
    overlap_term = holding_obs - 1
    return int(max(overlap_term, rate_term))


def _ljung_box_flags_autocorr(r: np.ndarray, max_lag: int) -> bool:
    """True if Ljung-Box rejects no-autocorrelation at any lag up to ``max_lag``."""
    if max_lag < 1 or len(r) < max_lag + 2:
        return False
    lb = acorr_ljungbox(r, lags=max_lag, return_df=True)
    return bool((lb["lb_pvalue"] < _AUTOCORR_ALPHA).any())


def _autocorr_vif(r: np.ndarray, mean: float, max_lag: int) -> float:
    """Bartlett-weighted variance-inflation factor for the sample-mean variance.

    ``VIF = 1 + 2 * sum_{k=1}^{max_lag} (1 - k/(max_lag+1)) * rho_k`` where
    ``rho_k`` is the lag-k autocorrelation of the centred returns. This is the
    serial-correlation term in Lo's (2002) GMM Sharpe SE: it reduces to 1 when
    the autocorrelations vanish (recovering the iid form) and inflates the SE
    when positive serial correlation is present. Clamped at a small positive
    floor so a (finite-sample) negative estimate can never yield an imaginary SE.
    """
    n = len(r)
    resid = r - mean
    gamma_0 = float(np.dot(resid, resid) / n)
    if gamma_0 <= 0.0:
        return 1.0
    vif = 1.0
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1)
        rho_k = float(np.dot(resid[k:], resid[:-k]) / n) / gamma_0
        vif += 2.0 * weight * rho_k
    return max(vif, 1e-8)


def lo_sharpe_se(
    returns: pd.Series,
    periods_per_year: float = 12,
) -> tuple[float, float, bool]:
    """Annualized Sharpe ratio and its Lo (2002) standard error.

    The iid standard error of the per-period Sharpe is the closed form
    ``sqrt((1 + 0.5 * SR_period**2) / T)`` (Lo 2002); annualized by the naive
    ``sqrt(periods_per_year)`` scaling. If the return series is autocorrelated
    -- detected automatically by Ljung-Box at the ``maxlags_for`` bandwidth
    (spec §2.0, the trigger is not optional) -- the per-period sampling variance
    is inflated by the autocovariance structure of the mean estimator, so the SE
    grows under positive serial correlation instead of under-stating it.

    Args:
        returns: periodic factor returns.
        periods_per_year: annualization factor.

    Returns:
        ``(sharpe, se, used_autocorr_correction)`` -- the *annualized* Sharpe,
        the SE of that annualized Sharpe, and whether the autocorrelation
        correction was applied. Degenerate input (<2 obs or zero variance)
        returns ``(nan, nan, False)``.
    """
    r = returns.dropna().to_numpy(dtype=float)
    t = len(r)
    if t < 2:
        return float("nan"), float("nan"), False
    mean = float(r.mean())
    # ddof=1 (sample std) — consistent with the naive t-stat and with Lo (2002)
    # which derives the iid SE under the sample-mean / sample-std estimator.
    std = float(r.std(ddof=1))
    if not std > 0.0:
        return float("nan"), float("nan"), False

    sr_period = mean / std
    sharpe = sr_period * math.sqrt(periods_per_year)
    iid_var_period = (1.0 + 0.5 * sr_period**2) / t

    max_lag = maxlags_for(t, holding_obs=1)
    used = _ljung_box_flags_autocorr(r, max_lag)
    if used:
        vif = _autocorr_vif(r, mean, max_lag)
        var_period = iid_var_period * vif
    else:
        var_period = iid_var_period

    se = math.sqrt(periods_per_year) * math.sqrt(var_period)
    return sharpe, se, used


def effective_n_and_power(returns: pd.Series, holding_obs: int) -> dict:
    """Effective (non-overlapping) sample size and approximate power.

    Overlapping holding windows share information, so ``T`` raw observations of a
    ``holding_obs``-period hold are not ``T`` independent draws; the effective
    count is ``floor(T / holding_obs)`` (== T when ``holding_obs == 1``). A
    variant whose effective ``n`` falls below ``config.MIN_EFFECTIVE_N`` is
    labelled ``"inconclusive (underpowered)"`` -- a distinct verdict from
    "insignificant" (spec §2.0): too few independent draws to separate a true
    zero mean from a small positive one.

    Power is a one-sided t-test approximation at the observed per-period Sharpe:
    non-centrality ``SR_period * sqrt(effective_n)``, ``power = P(Z > z_alpha - ncp)``.

    Args:
        returns: periodic factor returns.
        holding_obs: holding-window length in observation units (1 => non-overlapping).

    Returns:
        Dict: ``effective_n`` (int), ``n_obs`` (int), ``power`` (float in [0,1]),
        ``underpowered`` (bool), ``label`` (str).
    """
    r = returns.dropna().to_numpy(dtype=float)
    n_obs = len(r)
    hold = max(int(holding_obs), 1)
    effective_n = n_obs // hold

    underpowered = effective_n < MIN_EFFECTIVE_N
    label = "inconclusive (underpowered)" if underpowered else "powered"

    if effective_n >= 1 and n_obs >= 2 and r.std(ddof=1) > 0.0:
        sr_period = float(r.mean()) / float(r.std(ddof=1))
        ncp = sr_period * math.sqrt(effective_n)
        z_alpha = _scs.norm.ppf(1.0 - _POWER_ALPHA)
        power = float(_scs.norm.sf(z_alpha - ncp))
    else:
        power = float("nan")

    return {
        "effective_n": int(effective_n),
        "n_obs": int(n_obs),
        "power": power,
        "underpowered": bool(underpowered),
        "label": label,
    }
