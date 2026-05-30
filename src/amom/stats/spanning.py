"""Stage-2.4 spanning regression vs {market, size control} (spec §2.4 / §3.2).

The spanning test partials the momentum factor's exposure to a small benchmark
set out of its mean: regress the per-rebalance ``factor_return`` on
{equal-weighted market return, small-minus-big size control} and report the
**intercept (alpha)** with a Newey-West HAC t-stat (guide §2.4). A variant is
kept only if alpha is significant; this is a reduced (not full) LTW redundancy
set and is disclosed as such.

The **size control** is a TEST-ONLY regressor: a small-minus-big long/short
return formed on Artemis ``MC`` quintiles. It is a significance test instrument
only — never formed as a deployed portfolio, never reported as a strategy (spec
§2). Including it makes the redundancy test non-vacuous without breaching
momentum-only scope.

Both benchmark series are built **point-in-time** over each factor holding window
``(r, next_r]`` — the same window convention the factor formation uses — on the
in-sample eligibility mask:

  * ``build_market_return``  — equal-weighted compounded return of the names
    eligible as-of ``r`` over ``(r, next_r]``.
  * ``build_size_control``   — rank the eligible-as-of-``r`` names by ``MC``
    as-of ``r`` into quintiles; ``size_control = small_leg − big_leg`` where each
    leg is the equal-weighted compounded window return (small = bottom MC
    quintile, big = top). MC is read as-of ``r`` only (no look-ahead).

``spanning_alpha`` is a pure HAC-OLS that reuses ``stats.core.newey_west_se`` so
the alpha t-stat is identical to ``statsmodels`` OLS-with-HAC. The builders are
pure and perform no I/O; the in-sample slicing and the MC/return/universe panels
are supplied by the Stage-2 runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import QUANTILE
from amom.stats.core import newey_west_se

_MARKET_COLUMNS = ["rebalance_date", "market_return"]
_SIZE_COLUMNS = ["rebalance_date", "size_control"]


def spanning_alpha(
    factor_returns: pd.Series | pd.DataFrame,
    regressors_df: pd.DataFrame,
    bandwidth: int,
) -> dict:
    """OLS spanning regression with a Newey-West HAC alpha t-stat (guide §2.4).

    Regresses the factor return on a constant plus the columns of
    ``regressors_df`` and reports the intercept (alpha) and its HAC t-stat. The
    HAC standard errors reuse ``stats.core.newey_west_se``, so the alpha t-stat
    matches ``statsmodels`` OLS-with-HAC to numerical precision.

    Args:
        factor_returns: per-rebalance factor returns, indexed by rebalance date
            (a 1-column DataFrame is accepted and squeezed to a Series).
        regressors_df: aligned regressors (e.g. ``market_return``,
            ``size_control``), indexed by the same rebalance dates.
        bandwidth: Newey-West HAC bandwidth (use ``maxlags_for``).

    Returns:
        Dict with:
          ``alpha``        intercept (the spanning alpha),
          ``alpha_tstat``  HAC t-stat of the intercept,
          ``betas``        {regressor_name: coefficient},
          ``r2``           OLS R-squared,
          ``n``            number of finite, aligned observations used.
        Degenerate input (too few aligned obs for the bandwidth) returns NaNs.
    """
    if isinstance(factor_returns, pd.DataFrame):
        if factor_returns.shape[1] != 1:
            raise ValueError("factor_returns DataFrame must have exactly one column")
        factor_returns = factor_returns.iloc[:, 0]

    reg_names = list(regressors_df.columns)
    aligned = pd.concat(
        [factor_returns.rename("__y__"), regressors_df], axis=1, join="inner"
    ).dropna()
    n = len(aligned)
    k = len(reg_names) + 1  # +1 for the intercept
    nan_betas = {name: float("nan") for name in reg_names}
    if n < bandwidth + k + 1:
        return {
            "alpha": float("nan"),
            "alpha_tstat": float("nan"),
            "betas": nan_betas,
            "r2": float("nan"),
            "n": int(n),
        }

    y = aligned["__y__"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), aligned[reg_names].to_numpy(dtype=float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    try:
        se = newey_west_se(resid, X, bandwidth)
    except np.linalg.LinAlgError:
        # A rank-deficient design (e.g. a constant regressor) has no HAC SE;
        # report NaNs rather than raising so the runner records the variant.
        return {
            "alpha": float(beta[0]),
            "alpha_tstat": float("nan"),
            "betas": {name: float(beta[i + 1]) for i, name in enumerate(reg_names)},
            "r2": float("nan"),
            "n": int(n),
        }

    alpha = float(beta[0])
    alpha_se = float(se[0])
    alpha_tstat = alpha / alpha_se if alpha_se > 0 else float("nan")

    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    betas = {name: float(beta[i + 1]) for i, name in enumerate(reg_names)}
    return {
        "alpha": alpha,
        "alpha_tstat": float(alpha_tstat),
        "betas": betas,
        "r2": float(r2),
        "n": int(n),
    }


def _eligible_symbols_asof(
    universe_panel: pd.DataFrame, rebalance_date: pd.Timestamp
) -> list[str]:
    """Symbols eligible at ``rebalance_date`` (or the latest dated ``<=`` it).

    Point-in-time: eligibility is read as-of the rebalance/entry date ``r``, the
    same convention the factor formation uses. Returns the symbols whose
    ``eligible`` flag is True on that as-of row.
    """
    dates = universe_panel["date"]
    asof_dates = dates[dates <= rebalance_date]
    if asof_dates.empty:
        return []
    asof = asof_dates.max()
    row = universe_panel[(universe_panel["date"] == asof) & universe_panel["eligible"]]
    return row["symbol"].tolist()


def _window_return(
    returns_wide: pd.DataFrame,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Compounded simple return for ``symbol`` over ``(start_date, end_date]``.

    Mirrors the factor formation's window convention (strictly after the entry
    date ``start_date``, through ``end_date``); NaN cells inside the window
    compound as 0% (held, no price observed). All-NaN window -> NaN.
    """
    if symbol not in returns_wide.columns:
        return float("nan")
    mask = (returns_wide.index > start_date) & (returns_wide.index <= end_date)
    window = returns_wide.loc[mask, symbol]
    if window.dropna().empty:
        return float("nan")
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def _equal_weighted_leg(
    returns_wide: pd.DataFrame,
    symbols: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Equal-weighted compounded window return over ``symbols`` (NaN legs dropped)."""
    rets = [_window_return(returns_wide, s, start_date, end_date) for s in symbols]
    rets = [x for x in rets if not np.isnan(x)]
    if not rets:
        return float("nan")
    return float(np.mean(rets))


def _returns_wide(holding_returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long holding-return panel to wide (dates x symbols)."""
    return holding_returns.pivot_table(
        index="date", columns="symbol", values="holding_return"
    ).sort_index()


def build_market_return(
    holding_returns: pd.DataFrame,
    universe_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Equal-weighted eligible-universe return per rebalance window (spec §2.4).

    For each rebalance date ``r`` (except the last), the market return is the
    equal-weighted compounded return over ``(r, next_r]`` of the names eligible
    as-of ``r`` (point-in-time). This is the benchmark the momentum factor is
    regressed on — the "is it just the market?" leg of the spanning test.

    Args:
        holding_returns: long ``[date, symbol, holding_return]`` panel.
        universe_panel: long ``[date, symbol, eligible, ...]`` panel.
        rebalance_dates: ordered rebalance dates (windows are consecutive pairs).

    Returns:
        DataFrame indexed by ``rebalance_date`` with one ``market_return`` column.
    """
    returns_wide = _returns_wide(holding_returns)
    dates = list(rebalance_dates)
    rows = []
    for i, r in enumerate(dates[:-1]):
        next_r = dates[i + 1]
        eligible = _eligible_symbols_asof(universe_panel, r)
        mr = _equal_weighted_leg(returns_wide, eligible, r, next_r)
        if np.isnan(mr):
            continue
        rows.append({"rebalance_date": r, "market_return": mr})
    if not rows:
        return pd.DataFrame(columns=_MARKET_COLUMNS[1:]).rename_axis("rebalance_date")
    return pd.DataFrame(rows).set_index("rebalance_date")


def _mc_asof(mc_panel: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.Series:
    """Market cap per symbol as-of ``rebalance_date`` (latest row dated ``<=`` it).

    Point-in-time: the size sort uses MC as-of the rebalance date only; MC values
    dated after ``r`` are never consulted (no look-ahead).
    """
    sub = mc_panel[mc_panel["date"] <= rebalance_date]
    if sub.empty:
        return pd.Series(dtype=float)
    latest = sub.sort_values("date").groupby("symbol")["mc"].last()
    return latest.dropna()


def build_size_control(
    holding_returns: pd.DataFrame,
    universe_panel: pd.DataFrame,
    mc_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    quantile: float = QUANTILE,
) -> pd.DataFrame:
    """Small-minus-big size control per rebalance window (TEST-ONLY, spec §2.4).

    For each rebalance date ``r``, rank the names eligible as-of ``r`` by ``MC``
    as-of ``r`` into quintiles; the *small* leg is the bottom ``quantile`` of MC
    and the *big* leg is the top ``quantile``. Each leg's equal-weighted
    compounded return over ``(r, next_r]`` is computed, and
    ``size_control = small_leg − big_leg``.

    This is a significance-test regressor only: it is never formed as a deployed
    portfolio and never reported as a strategy (spec §2, §3.2). A rebalance with
    too few MC-ranked eligible names to field both legs is skipped.

    Args:
        holding_returns: long ``[date, symbol, holding_return]`` panel.
        universe_panel: long ``[date, symbol, eligible, ...]`` panel.
        mc_panel: long ``[date, symbol, mc]`` market-cap panel.
        rebalance_dates: ordered rebalance dates (consecutive-pair windows).
        quantile: top/bottom MC fraction per leg (default ``QUANTILE`` = quintile).

    Returns:
        DataFrame indexed by ``rebalance_date`` with one ``size_control`` column.
    """
    returns_wide = _returns_wide(holding_returns)
    dates = list(rebalance_dates)
    rows = []
    for i, r in enumerate(dates[:-1]):
        next_r = dates[i + 1]
        eligible = set(_eligible_symbols_asof(universe_panel, r))
        if not eligible:
            continue
        mc = _mc_asof(mc_panel, r)
        mc = mc[mc.index.isin(eligible)]
        if len(mc) < 2:
            continue

        big_threshold = np.nanquantile(mc.to_numpy(), 1.0 - quantile)
        small_threshold = np.nanquantile(mc.to_numpy(), quantile)
        small_syms = sorted(mc[mc <= small_threshold].index.tolist())
        big_syms = sorted(mc[mc >= big_threshold].index.tolist())
        if not small_syms or not big_syms:
            continue

        small_leg = _equal_weighted_leg(returns_wide, small_syms, r, next_r)
        big_leg = _equal_weighted_leg(returns_wide, big_syms, r, next_r)
        if np.isnan(small_leg) or np.isnan(big_leg):
            continue
        rows.append({"rebalance_date": r, "size_control": small_leg - big_leg})

    if not rows:
        return pd.DataFrame(columns=_SIZE_COLUMNS[1:]).rename_axis("rebalance_date")
    return pd.DataFrame(rows).set_index("rebalance_date")
