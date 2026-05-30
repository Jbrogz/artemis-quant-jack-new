"""Dollar-neutral momentum long/short formation (Task S2, spec §1.4).

Ported from Project 1 ``factors/momentum_portfolio.py`` with **funding stripped**:
the factor P&L is driven by the spot ``holding_return`` panel (Task S0, spec §3.1),
never a funding-adjusted return. Artemis serves no funding and this is a spot book.

Construction (spec §1.4 / guide §1.4):
  - sort the eligible cross-section by signal into quintiles (``QUANTILE = 0.20``):
    long the top 20%, short the bottom 20%;
  - **equal-weight** within each leg; **dollar-neutral** — the long leg's dollar
    weight (Σ = +1) exactly offsets the short leg's (Σ = −1), so Σ weights = 0;
  - ``factor_return = long_leg_return − short_leg_return`` per rebalance;
  - eligibility-masked per the universe panel; min-bucket gate (``MIN_BUCKET_SIZE``)
    skips a rebalance whose long or short bucket is too thin.

No look-ahead (cardinal rule, spec §7), in the rebalance-date framing:
  - the rebalance/entry date ``r`` is the **close t+1** execution date;
  - the signal it uses is the most recent one dated ``<= r − LAG_DAYS`` (= close
    ``t``) — the formation never reads the signal dated ``r`` (= ``t+1``) itself;
  - the holding window is the strictly-later ``(r, r + HOLDING_DAYS]`` — the
    position is opened at close ``r`` and earns returns dated after ``r``.

Cross-sectional ranking uses the signal value directly; log/simple ranking is
monotone so the sort is identical (spec §1.4 — no convention-mix). The functions
are pure and perform no I/O; panels are passed in (loaders live in the script).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import HOLDING_DAYS, LAG_DAYS, MIN_BUCKET_SIZE, QUANTILE

_FACTOR_RETURN_COLUMNS = [
    "rebalance_date",
    "factor_return",
    "long_return",
    "short_return",
    "n_long",
    "n_short",
]


def select_buckets(
    signal_panel: pd.DataFrame,
    eligibility: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    *,
    quantile: float = QUANTILE,
    lag_days: int = LAG_DAYS,
    min_bucket_size: int = MIN_BUCKET_SIZE,
) -> pd.DataFrame | None:
    """Pick the long/short buckets for one rebalance date and assign weights.

    Args:
        signal_panel: wide (dates x symbols) momentum signal (Task S1 output).
        eligibility: wide (dates x symbols) bool eligibility mask (from the
            universe panel's ``eligible`` column).
        rebalance_date: the entry/execution date ``r`` (= close ``t+1``).
        quantile: top/bottom fraction (default ``QUANTILE`` = quintile).
        lag_days: execution lag; the signal used is dated ``<= r − lag_days``.
        min_bucket_size: a leg with fewer names than this gates the rebalance.

    Returns:
        Long DataFrame ``[rebalance_date, symbol, position, weight, signal_value]``
        with equal weights (+1/n_long on longs, −1/n_short on shorts, dollar-
        neutral), or ``None`` when either bucket is below ``min_bucket_size`` (the
        rebalance is skipped). Sorted by ``(position, symbol)``.

    No look-ahead: the signal is looked up at the most recent date ``<= r −
    lag_days``; signal values dated after that are never consulted. Eligibility is
    evaluated AT the rebalance date ``r`` (the trade happens at ``r``), so a name
    whose eligibility lapsed between the signal date and ``r`` is excluded.
    """
    # --- Signal lookup at the lagged date (no look-ahead: <= r − lag_days). ---
    signal_lookup_date = rebalance_date - pd.Timedelta(days=lag_days)
    valid_signal_dates = signal_panel.index[signal_panel.index <= signal_lookup_date]
    if len(valid_signal_dates) == 0:
        return None
    sig_date = valid_signal_dates[-1]
    sig_row = signal_panel.loc[sig_date]

    # --- Eligibility AT the rebalance date r (the trade happens at r). ---
    elig_row = _eligibility_row_asof(eligibility, rebalance_date)
    if elig_row is None:
        return None

    common = sig_row.index.intersection(elig_row.index)
    sig_row = sig_row[common]
    elig_row = elig_row[common]

    eligible_sig = sig_row[elig_row.astype(bool) & sig_row.notna()]
    # Need at least min_bucket_size names per leg to field both buckets.
    if len(eligible_sig) < 2 * min_bucket_size:
        return None

    long_threshold = np.nanquantile(eligible_sig.to_numpy(), 1.0 - quantile)
    short_threshold = np.nanquantile(eligible_sig.to_numpy(), quantile)

    long_syms = sorted(eligible_sig[eligible_sig >= long_threshold].index.tolist())
    short_syms = sorted(eligible_sig[eligible_sig <= short_threshold].index.tolist())

    if len(long_syms) < min_bucket_size or len(short_syms) < min_bucket_size:
        return None

    long_w = 1.0 / len(long_syms)
    short_w = -1.0 / len(short_syms)
    rows = []
    for s in long_syms:
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "symbol": s,
                "position": "long",
                "weight": long_w,
                "signal_value": float(eligible_sig[s]),
            }
        )
    for s in short_syms:
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "symbol": s,
                "position": "short",
                "weight": short_w,
                "signal_value": float(eligible_sig[s]),
            }
        )
    return pd.DataFrame(rows).sort_values(["position", "symbol"]).reset_index(drop=True)


def _eligibility_row_asof(
    eligibility: pd.DataFrame, rebalance_date: pd.Timestamp
) -> pd.Series | None:
    """The eligibility row at ``rebalance_date`` (or the latest dated ``<=`` it)."""
    if rebalance_date in eligibility.index:
        return eligibility.loc[rebalance_date]
    valid = eligibility.index[eligibility.index <= rebalance_date]
    if len(valid) == 0:
        return None
    return eligibility.loc[valid[-1]]


def _holding_period_return(
    returns_wide: pd.DataFrame,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Compounded simple return for ``symbol`` over ``(start_date, end_date]``.

    The window is strictly after ``start_date`` (the entry/close-t+1 date) and
    through ``end_date`` — the entry-date return is excluded because the position
    is opened at that close. NaN cells inside the window compound as 0% (position
    held, no price observed), which implicitly closes a coin that delists mid-hold
    at its last observed price (spec §1.2). All-NaN window -> NaN (no return).
    """
    if symbol not in returns_wide.columns:
        return float("nan")
    mask = (returns_wide.index > start_date) & (returns_wide.index <= end_date)
    window = returns_wide.loc[mask, symbol]
    if window.dropna().empty:
        return float("nan")
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def compute_factor_returns(
    holdings: pd.DataFrame,
    returns_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    min_bucket_size: int = MIN_BUCKET_SIZE,
    holding_days: int = HOLDING_DAYS,
) -> pd.DataFrame:
    """Per-rebalance factor returns from a holdings table and the return panel.

    For each rebalance date ``r`` (except the last, which has no next rebalance),
    compute the equal-weighted long- and short-leg returns over ``(r, next_r]``
    and ``factor_return = long_leg − short_leg``. ``holding_days`` is accepted for
    signature symmetry with ``build_factor_returns`` (the actual window end is the
    next scheduled rebalance, which the caller spaces ``holding_days`` apart).

    Returns:
        DataFrame indexed by ``rebalance_date`` with columns ``factor_return,
        long_return, short_return, n_long, n_short``. A rebalance whose long or
        short leg has fewer than ``min_bucket_size`` names with a realized return
        over the window is skipped (no gappy / single-name leg).
    """
    rows = []
    sorted_dates = list(rebalance_dates)
    for i, r in enumerate(sorted_dates[:-1]):
        next_r = sorted_dates[i + 1]
        rebal = holdings[holdings["rebalance_date"] == r]
        if rebal.empty:
            continue
        long_syms = rebal.loc[rebal["position"] == "long", "symbol"].tolist()
        short_syms = rebal.loc[rebal["position"] == "short", "symbol"].tolist()

        long_rets = [_holding_period_return(returns_wide, s, r, next_r) for s in long_syms]
        short_rets = [_holding_period_return(returns_wide, s, r, next_r) for s in short_syms]
        long_rets = [x for x in long_rets if not np.isnan(x)]
        short_rets = [x for x in short_rets if not np.isnan(x)]

        if len(long_rets) < min_bucket_size or len(short_rets) < min_bucket_size:
            continue

        long_leg = float(np.mean(long_rets))
        short_leg = float(np.mean(short_rets))
        rows.append(
            {
                "rebalance_date": r,
                "factor_return": long_leg - short_leg,
                "long_return": long_leg,
                "short_return": short_leg,
                "n_long": len(long_rets),
                "n_short": len(short_rets),
            }
        )

    if not rows:
        return pd.DataFrame(columns=_FACTOR_RETURN_COLUMNS[1:]).rename_axis("rebalance_date")
    return pd.DataFrame(rows).set_index("rebalance_date")


def build_rebalance_dates(
    signal_index: pd.DatetimeIndex, *, holding_days: int = HOLDING_DAYS
) -> pd.DatetimeIndex:
    """Rebalance dates every ``holding_days``-th date in the signal panel index.

    Anchoring on actual grid dates (not calendar months) keeps the holding window
    ``(r, next_r]`` aligned to the daily panel. Warm-up dates where the signal is
    NaN are skipped implicitly by ``select_buckets`` (insufficient eligible names).
    """
    return signal_index[::holding_days]


def build_factor_returns(
    signal_panel: pd.DataFrame,
    eligibility: pd.DataFrame,
    returns_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    quantile: float = QUANTILE,
    lag_days: int = LAG_DAYS,
    min_bucket_size: int = MIN_BUCKET_SIZE,
    holding_days: int = HOLDING_DAYS,
) -> pd.DataFrame:
    """Build the per-rebalance dollar-neutral factor-return series (pure, no I/O).

    For each rebalance date, ``select_buckets`` chooses the dollar-neutral
    long/short book from the lagged signal; ``compute_factor_returns`` then prices
    each book over its ``(r, next_r]`` holding window from the spot return panel.

    Returns:
        Long DataFrame ``[rebalance_date, factor_return, long_return,
        short_return, n_long, n_short]`` (one row per non-gated rebalance with a
        realized window), sorted by ``rebalance_date``.
    """
    frames = []
    for r in rebalance_dates:
        frame = select_buckets(
            signal_panel,
            eligibility,
            r,
            quantile=quantile,
            lag_days=lag_days,
            min_bucket_size=min_bucket_size,
        )
        if frame is not None:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_FACTOR_RETURN_COLUMNS)

    holdings = pd.concat(frames, ignore_index=True)
    fr = compute_factor_returns(
        holdings,
        returns_wide,
        pd.DatetimeIndex(rebalance_dates),
        min_bucket_size=min_bucket_size,
        holding_days=holding_days,
    )
    return fr.reset_index()[_FACTOR_RETURN_COLUMNS] if not fr.empty else pd.DataFrame(
        columns=_FACTOR_RETURN_COLUMNS
    )
