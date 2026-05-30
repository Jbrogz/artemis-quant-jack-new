"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Ported verbatim from Project 1 ``cmom/overfitting/dsr.py``.

The DSR corrects an observed Sharpe ratio for (a) the number of strategy
configurations tried during research and (b) non-normal returns. It returns
the probability that the true Sharpe exceeds a selection-adjusted benchmark.
A DSR above ~0.95 means the result survives multiple-testing deflation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe estimator
_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns) -> float:
    """Non-annualised Sharpe ratio of a return series."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd)


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark), adjusting for skew, kurtosis and T.

    PSR = Z[ (SR - SR*) * sqrt(T - 1)
             / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) ]
    where g3 is skewness, g4 is (non-excess) kurtosis, T the sample length,
    and Z the standard-normal CDF.
    """
    r = np.asarray(returns, dtype=float)
    t = r.size
    if t < 2:
        raise ValueError("need at least 2 return observations")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain non-finite values (NaN/inf)")
    if r.std(ddof=1) == 0.0:
        raise ValueError("PSR undefined for constant (zero-variance) returns")
    sr = sharpe_ratio(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))   # non-excess kurtosis
    denominator = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("degenerate PSR denominator")
    z = (sr - sr_benchmark) * np.sqrt(t - 1) / denominator
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe across n_trials independent strategies (H0).

    E[max SR] ~ sqrt(V) * [ (1 - g) * Z^-1(1 - 1/N)
                            + g * Z^-1(1 - 1/(N*e)) ]
    with g the Euler-Mascheroni constant. Zero for a single trial.

    Assumes the N trial Sharpe ratios are mutually independent (Bailey &
    Lopez de Prado 2014, Prop. 2). Strategies sharing one backtest window
    are positively correlated, so E[max SR] is overestimated -- which makes
    the resulting DSR conservative (a lower bound on the true DSR).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(returns, trial_sharpes) -> float:
    """Deflated Sharpe Ratio.

    `trial_sharpes` is the array of Sharpe ratios of every configuration
    tested. The benchmark is the expected maximum Sharpe across that many
    trials; DSR is the PSR evaluated at that benchmark.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n = trials.size
    if n < 1:
        raise ValueError("need at least one trial Sharpe")
    sr_variance = float(trials.var(ddof=1)) if n > 1 else 0.0
    sr_benchmark = expected_max_sharpe(sr_variance, n)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_benchmark)
