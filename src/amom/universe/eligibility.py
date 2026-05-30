"""Point-in-time eligibility — the anti-bias core (guide §1.1, spec §4 Stage 1.1).

A coin is eligible *as of* a date when, using only information available at or
before that date, it satisfies every filter:

  - history:     ``(as_of - first_date).days >= MIN_HISTORY_DAYS`` (90 days),
  - obs-density: ``n_obs_90d / MIN_HISTORY_DAYS >= MIN_OBS_DENSITY`` (calendar
                 age alone is not enough; the window must be densely observed),
  - liquidity:   ``mc >= MIN_MC_USD`` AND a trailing-window 24H-volume gate that
                 is robust to broken prints yet requires sustained volume —
                 ``median(positive prints) >= MIN_MEDIAN_VOL_USD`` AND
                 ``sum(positive prints) / window_days >= MIN_MEDIAN_VOL_USD``
                 (spec §3.5). The full window is the ADV denominator, so a
                 2-print $5M window FAILS (the rev-2 mean-of-present-rows bug),
                 while a few broken/sub-dollar prints do not flip a liquid coin
                 (the median is robust),
  - tradeability: ``(as_of - last_price_date).days <= LISTING_STALENESS_DAYS``
                 (a stopped/dead coin exits eligibility within the grace window),
  - exclusion:   ``symbol not in EXCLUDED`` (stablecoins ∪ wrapped tokens).

No look-ahead by construction: ``is_eligible`` consumes only as-of scalars and a
pre-sliced trailing volume window (which carries no dates) — there is no
parameter through which a future-dated series could enter the decision.
``eligible_mask`` slices each symbol's volume to the trailing window dated at or
before ``as_of`` and never any later row. Both functions are pure and perform
no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import (
    LIQUIDITY_VOL_WINDOW_DAYS,
    LISTING_STALENESS_DAYS,
    MIN_HISTORY_DAYS,
    MIN_MC_USD,
    MIN_MEDIAN_VOL_USD,
    MIN_OBS_DENSITY,
)


def window_liquidity(
    vol_window, *, window_days: int = LIQUIDITY_VOL_WINDOW_DAYS
) -> tuple[float, float]:
    """Robust trailing-window volume stats, winsorizing non-positive prints.

    Args:
        vol_window: array-like of the trailing-window 24H volume prints (already
            point-in-time: only observations dated ``<= as_of`` within the
            window). May contain NaN / zero / negative garbage prints.
        window_days: the window length in days; this is the ADV denominator so
            a sparse window of a few large prints yields a small ADV.

    Returns:
        ``(median_vol, adv)`` where ``median_vol`` is the median over the
        positive prints (NaN if there are none) and ``adv`` is the sum of
        positive prints divided by the FULL ``window_days`` (0.0 if none). The
        full-window denominator is what removes the rev-2 mean-of-present-rows
        bug: two $5M prints give ADV = 2·5M / 30 ≪ the floor.
    """
    arr = np.asarray(vol_window, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size == 0:
        return float("nan"), 0.0
    median_vol = float(np.median(positive))
    adv = float(positive.sum()) / float(window_days)
    return median_vol, adv


def is_eligible(
    symbol: str,
    as_of: pd.Timestamp,
    *,
    first_date: pd.Timestamp,
    last_price_date: pd.Timestamp,
    n_obs_90d: int,
    mc: float,
    vol_window,
    excluded,
) -> bool:
    """Return whether ``symbol`` is eligible as of ``as_of``.

    Args:
        symbol: the coin symbol (case-sensitive; compared against ``excluded``).
        as_of: the evaluation date; only this date's information is consulted.
        first_date: the coin's first observed price date (listing proxy).
        last_price_date: the coin's latest observed price date dated ``<= as_of``
            (the tradeability/staleness anchor). NaT fails tradeability.
        n_obs_90d: count of priced days in the trailing ``MIN_HISTORY_DAYS``
            window as of ``as_of`` (the obs-density numerator).
        mc: market cap (USD) as of ``as_of``. NaN fails the MC floor.
        vol_window: the trailing-``LIQUIDITY_VOL_WINDOW_DAYS`` 24H-volume prints
            dated ``<= as_of`` (array-like; broken prints tolerated).
        excluded: a set/collection of excluded symbols (stablecoins ∪ wrapped).

    Returns:
        True iff history, obs-density, liquidity, tradeability, and exclusion all
        pass.
    """
    if symbol in excluded:
        return False

    # History: calendar age as-of.
    if pd.isna(first_date) or (as_of - first_date).days < MIN_HISTORY_DAYS:
        return False

    # Observation density: the trailing-90d window must be densely observed.
    if n_obs_90d / MIN_HISTORY_DAYS < MIN_OBS_DENSITY:
        return False

    # Tradeability: the latest price must be within the grace window.
    if pd.isna(last_price_date) or (as_of - last_price_date).days > LISTING_STALENESS_DAYS:
        return False

    # Liquidity: MC floor AND robust-but-sustained trailing volume (§3.5).
    if pd.isna(mc) or mc < MIN_MC_USD:
        return False
    median_vol, adv = window_liquidity(vol_window)
    if pd.isna(median_vol) or median_vol < MIN_MEDIAN_VOL_USD:
        return False
    if adv < MIN_MEDIAN_VOL_USD:
        return False

    return True


def eligible_mask(
    as_of: pd.Timestamp,
    coverage_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    excluded,
) -> set:
    """Return the set of symbols eligible as of ``as_of``.

    Args:
        as_of: the evaluation date.
        coverage_df: DataFrame with columns
            ``[symbol, price_first_date, price_last_date, n_obs]`` recomputed
            as-of (the listing-date / density reconstruction).
        volume_df: long DataFrame ``[date, symbol, volume]``. For each symbol the
            trailing-``LIQUIDITY_VOL_WINDOW_DAYS`` window of observations dated
            ``<= as_of`` is used; rows dated after ``as_of`` are ignored entirely
            (no look-ahead). A symbol with no row in the window fails liquidity.
        mc_df: DataFrame with columns ``[symbol, mc]`` of as-of market cap. A
            symbol absent here maps to NaN (fails the MC floor).
        excluded: a set/collection of excluded symbols.

    Returns:
        The set of eligible symbols.
    """
    window_lo = as_of - pd.Timedelta(days=LIQUIDITY_VOL_WINDOW_DAYS)
    in_window = (volume_df["date"] > window_lo) & (volume_df["date"] <= as_of)
    vol_window = volume_df[in_window]
    vol_by_symbol = {
        sym: g["volume"].to_numpy(dtype=float)
        for sym, g in vol_window.groupby("symbol", sort=False)
    }
    mc_as_of = mc_df.set_index("symbol")["mc"].to_dict()

    eligible = set()
    for row in coverage_df.itertuples(index=False):
        if is_eligible(
            row.symbol,
            as_of,
            first_date=row.price_first_date,
            last_price_date=row.price_last_date,
            n_obs_90d=int(row.n_obs),
            mc=mc_as_of.get(row.symbol, float("nan")),
            vol_window=vol_by_symbol.get(row.symbol, np.array([])),
            excluded=excluded,
        ):
            eligible.add(row.symbol)

    return eligible
