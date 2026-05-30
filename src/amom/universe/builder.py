"""Universe panel builder — survivorship + point-in-time + min-universe gate.

This is the gated deliverable (guide §1.1, spec §4 Stage 1.1). It assembles a
long ``[date, symbol, eligible, adv_30d, price_last_date, delisted_asof, gated]``
panel that:

  - covers **every symbol ever seen** across ``dates`` — including coins that
    later collapsed or delisted. They are kept as rows; they simply become
    ineligible once they no longer pass the as-of filters. Dropping them is the
    survivorship bias the guide forbids.
  - rebuilds point-in-time eligibility per (symbol, date) using the rev-3
    liquidity gate (MC floor + robust-but-sustained trailing-window 24H volume),
    plus obs-density and tradeability/staleness checks (spec §3.5, §4 Stage 1.1),
  - reports ``adv_30d`` = the trailing-30d ADV (sum of positive prints divided
    by the full window) used by the liquidity gate, point-in-time,
  - carries the **death signal** (spec §4 Stage 1.1): ``price_last_date`` is the
    latest priced date dated ``<= as_of`` (NaT before the coin lists), and
    ``delisted_asof`` is ``(as_of - price_last_date) > LISTING_STALENESS_DAYS``
    — a point-in-time flag that fires once a collapsed/stopped coin's reporting
    lapses past the tradeability grace, so the returns layer (§Stage 1.2) knows
    where to book the terminal crash return. Both use only data ``<= as_of``.
  - applies a **minimum-universe gate**: a date with fewer than
    ``MIN_ELIGIBLE_NAMES`` eligible coins is marked ``gated=True`` (the
    rebalance is skipped downstream), using as-of information only. The floor is
    **derived from** ``MIN_BUCKET_SIZE`` (5 quintile buckets × ``MIN_BUCKET_SIZE``
    names) so a non-gated date can field a full quintile sort, not a
    coincidental constant.

The builder is pure and performs no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import (
    LIQUIDITY_VOL_WINDOW_DAYS,
    LISTING_STALENESS_DAYS,
    MIN_ELIGIBLE_NAMES,
    MIN_HISTORY_DAYS,
)
from amom.universe.coverage import first_seen_dates
from amom.universe.eligibility import is_eligible, window_liquidity

# MIN_ELIGIBLE_NAMES is derived from MIN_BUCKET_SIZE in config (N_QUINTILES *
# MIN_BUCKET_SIZE): a non-gated rebalance date must be able to field a full
# quintile sort with at least MIN_BUCKET_SIZE names per bucket (spec §4 Stage 1.1).

_OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "eligible",
    "adv_30d",
    "price_last_date",
    "delisted_asof",
    "left_censored",
    "gated",
]


def _trailing_windows(
    volume_panel: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[tuple[pd.Timestamp, str], np.ndarray]:
    """Trailing-window volume prints per (date, symbol), point-in-time.

    For each grid date ``d`` and symbol, the value is the array of that symbol's
    volume observations with ``d - LIQUIDITY_VOL_WINDOW_DAYS < vol_date <= d``.
    Volume dated after ``d`` is never consulted. A symbol with no volume in the
    window maps to an empty array (fails liquidity).
    """
    windows: dict[tuple[pd.Timestamp, str], np.ndarray] = {}
    window = pd.Timedelta(days=LIQUIDITY_VOL_WINDOW_DAYS)
    empty = np.array([])

    by_symbol = {sym: g for sym, g in volume_panel.groupby("symbol", sort=False)}
    for sym in symbols:
        g = by_symbol.get(sym)
        if g is None:
            for d in dates:
                windows[(d, sym)] = empty
            continue
        vol_dates = g["date"].to_numpy()
        vol_vals = g["volume"].to_numpy(dtype=float)
        for d in dates:
            lo = d - window
            in_window = (vol_dates > np.datetime64(lo)) & (vol_dates <= np.datetime64(d))
            windows[(d, sym)] = vol_vals[in_window]
    return windows


def _trailing_n_obs(
    price_panel: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[tuple[pd.Timestamp, str], int]:
    """Count of priced days in the trailing-``MIN_HISTORY_DAYS`` window per
    (date, symbol), point-in-time (the obs-density numerator)."""
    counts: dict[tuple[pd.Timestamp, str], int] = {}
    window = pd.Timedelta(days=MIN_HISTORY_DAYS)
    priced = price_panel.dropna(subset=["price"])
    by_symbol = {sym: g for sym, g in priced.groupby("symbol", sort=False)}
    for sym in symbols:
        g = by_symbol.get(sym)
        if g is None:
            for d in dates:
                counts[(d, sym)] = 0
            continue
        pdates = g["date"].to_numpy()
        for d in dates:
            lo = d - window
            in_window = (pdates > np.datetime64(lo)) & (pdates <= np.datetime64(d))
            counts[(d, sym)] = int(in_window.sum())
    return counts


def _last_price_dates(
    price_panel: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[tuple[pd.Timestamp, str], pd.Timestamp]:
    """Latest priced date dated ``<= d`` per (date, symbol), point-in-time (the
    tradeability/staleness anchor). NaT when no price exists at or before ``d``."""
    last: dict[tuple[pd.Timestamp, str], pd.Timestamp] = {}
    priced = price_panel.dropna(subset=["price"])
    by_symbol = {sym: g for sym, g in priced.groupby("symbol", sort=False)}
    for sym in symbols:
        g = by_symbol.get(sym)
        if g is None:
            for d in dates:
                last[(d, sym)] = pd.NaT
            continue
        pdates = np.sort(g["date"].to_numpy())
        for d in dates:
            past = pdates[pdates <= np.datetime64(d)]
            last[(d, sym)] = pd.Timestamp(past[-1]) if past.size else pd.NaT
    return last


def _mc_as_of(
    mc_panel: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[tuple[pd.Timestamp, str], float]:
    """Latest market cap dated ``<= d`` per (date, symbol), point-in-time. NaN
    when no MC observation exists at or before ``d``."""
    mc: dict[tuple[pd.Timestamp, str], float] = {}
    by_symbol = {sym: g for sym, g in mc_panel.groupby("symbol", sort=False)}
    for sym in symbols:
        g = by_symbol.get(sym)
        if g is None:
            for d in dates:
                mc[(d, sym)] = float("nan")
            continue
        g = g.sort_values("date")
        mc_dates = g["date"].to_numpy()
        mc_vals = g["mc"].to_numpy(dtype=float)
        for d in dates:
            past = mc_vals[mc_dates <= np.datetime64(d)]
            mc[(d, sym)] = float(past[-1]) if past.size else float("nan")
    return mc


def build_universe_history(
    price_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    mc_panel: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    excluded,
) -> pd.DataFrame:
    """Build the point-in-time universe eligibility panel.

    Args:
        price_panel: long DataFrame ``[date, symbol, price]`` covering every
            symbol ever seen (including collapsed/delisted coins).
        volume_panel: long DataFrame ``[date, symbol, volume]`` of daily USD
            24H volume. Symbols/days may be missing (illiquid -> ineligible).
        mc_panel: long DataFrame ``[date, symbol, mc]`` of market cap. A symbol
            with no MC at or before a date maps to NaN (fails the MC floor).
        dates: the rebalance/grid dates to evaluate eligibility on.
        excluded: collection of excluded symbols (stablecoins ∪ wrapped).

    Returns:
        Long DataFrame ``[date, symbol, eligible, adv_30d, price_last_date,
        delisted_asof, left_censored, gated]`` with one row per (date, symbol)
        for every symbol ever seen, sorted by (date, symbol). ``eligible``,
        ``delisted_asof``, ``left_censored``, and ``gated`` are bool;
        ``adv_30d`` is float; ``price_last_date`` is datetime (NaT when no
        price yet observed).
    """
    dates = pd.DatetimeIndex(dates).normalize()

    # Normalize ingest boundaries: strip intraday timestamps so half-open
    # (<= date) comparisons are robust to any intraday API delivery.
    price_panel = price_panel.copy()
    price_panel["date"] = price_panel["date"].dt.normalize()
    volume_panel = volume_panel.copy()
    volume_panel["date"] = volume_panel["date"].dt.normalize()
    mc_panel = mc_panel.copy()
    mc_panel["date"] = mc_panel["date"].dt.normalize()

    coverage = first_seen_dates(price_panel)
    first_by_symbol = dict(
        zip(coverage["symbol"], coverage["price_first_date"], strict=True)
    )

    # Left-censoring: the pull-start is the earliest date in the price panel.
    # An asset whose first observed price equals the pull-start may have listed
    # before the pull window began — its true listing date is unknown.
    if not price_panel.empty:
        pull_start = price_panel["date"].min()
    else:
        pull_start = pd.NaT
    left_censored_symbols: set[str] = set()
    if pd.notna(pull_start):
        for sym, first_date in first_by_symbol.items():
            if pd.notna(first_date) and first_date == pull_start:
                left_censored_symbols.add(sym)

    symbols = sorted(
        set(price_panel["symbol"]).union(volume_panel["symbol"])
    )

    vol_windows = _trailing_windows(volume_panel, dates, symbols)
    n_obs = _trailing_n_obs(price_panel, dates, symbols)
    last_price = _last_price_dates(price_panel, dates, symbols)
    mc_lookup = _mc_as_of(mc_panel, dates, symbols)

    staleness = pd.Timedelta(days=LISTING_STALENESS_DAYS)

    rows = []
    for d in dates:
        for sym in symbols:
            window = vol_windows[(d, sym)]
            _, adv = window_liquidity(window)
            last_date = last_price[(d, sym)]
            eligible = is_eligible(
                sym,
                d,
                first_date=first_by_symbol.get(sym, pd.NaT),
                last_price_date=last_date,
                n_obs_90d=n_obs[(d, sym)],
                mc=mc_lookup[(d, sym)],
                vol_window=window,
                excluded=excluded,
            )
            # Point-in-time death signal: a coin whose latest price <= d is
            # older than the staleness grace has stopped reporting (delisted /
            # collapsed). A coin with no price at or before d (NaT) has simply
            # not listed yet, so it is not delisted. Uses only data <= d.
            delisted = pd.notna(last_date) and (d - last_date) > staleness
            # Left-censoring: true listing date unknown if first price == pull-start.
            censored = sym in left_censored_symbols
            rows.append((d, sym, eligible, adv, last_date, delisted, censored))

    panel = pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "eligible",
            "adv_30d",
            "price_last_date",
            "delisted_asof",
            "left_censored",
        ],
    )
    panel["eligible"] = panel["eligible"].astype(bool)
    panel["adv_30d"] = panel["adv_30d"].astype(float)
    panel["price_last_date"] = pd.to_datetime(panel["price_last_date"])
    panel["delisted_asof"] = panel["delisted_asof"].astype(bool)
    panel["left_censored"] = panel["left_censored"].astype(bool)

    # Point-in-time min-universe gate: per date, gated iff too few eligible.
    eligible_counts = panel.groupby("date")["eligible"].transform("sum")
    panel["gated"] = (eligible_counts < MIN_ELIGIBLE_NAMES).astype(bool)

    panel = (
        panel.sort_values(["date", "symbol"])
        .reset_index(drop=True)[_OUTPUT_COLUMNS]
    )
    return panel
