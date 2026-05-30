"""Universe panel builder — survivorship + point-in-time + min-universe gate.

This is the gated deliverable (guide §1.1, spec §4 Stage 1.1). It assembles a
long ``[date, symbol, eligible, adv_30d, gated]`` panel that:

  - covers **every symbol ever seen** across ``dates`` — including coins that
    later collapsed or delisted. They are kept as rows; they simply become
    ineligible once they no longer pass the as-of filters. Dropping them is the
    survivorship bias the guide forbids.
  - rebuilds Task-4 eligibility per (symbol, date), strictly point-in-time,
  - computes trailing-30d average daily USD volume (``adv_30d``) from volume
    rows dated ``<= date`` only (no look-ahead),
  - applies a **minimum-universe gate**: a date with fewer than
    ``MIN_ELIGIBLE_NAMES`` eligible coins is marked ``gated=True`` (the
    rebalance is skipped downstream), using as-of information only.

The builder is pure and performs no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import MIN_ELIGIBLE_NAMES
from amom.universe.coverage import first_seen_dates
from amom.universe.eligibility import is_eligible

# Trailing window for the average-daily-volume liquidity measure.
_ADV_WINDOW_DAYS = 30

_OUTPUT_COLUMNS = ["date", "symbol", "eligible", "adv_30d", "gated"]


def _trailing_adv(
    volume_panel: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[tuple[pd.Timestamp, str], float]:
    """Trailing-30d mean daily volume per (date, symbol), point-in-time.

    For each grid date ``d`` and symbol, the value is the mean of that symbol's
    volume observations with ``d - 30d < vol_date <= d``. Volume dated after
    ``d`` is never consulted. A symbol with no volume in the window maps to NaN.
    """
    adv: dict[tuple[pd.Timestamp, str], float] = {}
    window = pd.Timedelta(days=_ADV_WINDOW_DAYS)

    by_symbol = {sym: g for sym, g in volume_panel.groupby("symbol", sort=False)}
    for sym in symbols:
        g = by_symbol.get(sym)
        if g is None:
            for d in dates:
                adv[(d, sym)] = float("nan")
            continue
        vol_dates = g["date"].to_numpy()
        vol_vals = g["volume"].to_numpy(dtype=float)
        for d in dates:
            lo = d - window
            in_window = (vol_dates > np.datetime64(lo)) & (vol_dates <= np.datetime64(d))
            vals = vol_vals[in_window]
            adv[(d, sym)] = float(np.nanmean(vals)) if vals.size else float("nan")
    return adv


def build_universe_history(
    price_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    excluded,
) -> pd.DataFrame:
    """Build the point-in-time universe eligibility panel.

    Args:
        price_panel: long DataFrame ``[date, symbol, price]`` covering every
            symbol ever seen (including collapsed/delisted coins).
        volume_panel: long DataFrame ``[date, symbol, volume]`` of daily USD
            volume. Symbols may be missing (their ADV is NaN -> illiquid).
        dates: the rebalance/grid dates to evaluate eligibility on.
        excluded: collection of excluded symbols (stablecoins ∪ wrapped).

    Returns:
        Long DataFrame ``[date, symbol, eligible, adv_30d, gated]`` with one row
        per (date, symbol) for every symbol ever seen, sorted by (date, symbol).
        ``eligible`` and ``gated`` are bool; ``adv_30d`` is float.
    """
    dates = pd.DatetimeIndex(dates)

    coverage = first_seen_dates(price_panel)
    first_by_symbol = dict(
        zip(coverage["symbol"], coverage["price_first_date"], strict=True)
    )

    # Every symbol ever seen in EITHER panel (price drives first-seen; volume
    # may carry symbols with no usable price, kept for completeness).
    symbols = sorted(
        set(price_panel["symbol"]).union(volume_panel["symbol"])
    )

    adv = _trailing_adv(volume_panel, dates, symbols)

    rows = []
    for d in dates:
        for sym in symbols:
            adv_30d = adv[(d, sym)]
            first_date = first_by_symbol.get(sym, pd.NaT)
            eligible = is_eligible(
                sym, d, first_date=first_date, adv_30d=adv_30d, excluded=excluded
            )
            rows.append((d, sym, eligible, adv_30d))

    panel = pd.DataFrame(rows, columns=["date", "symbol", "eligible", "adv_30d"])
    panel["eligible"] = panel["eligible"].astype(bool)
    panel["adv_30d"] = panel["adv_30d"].astype(float)

    # Point-in-time min-universe gate: per date, gated iff too few eligible.
    eligible_counts = panel.groupby("date")["eligible"].transform("sum")
    panel["gated"] = (eligible_counts < MIN_ELIGIBLE_NAMES).astype(bool)

    panel = (
        panel.sort_values(["date", "symbol"])
        .reset_index(drop=True)[_OUTPUT_COLUMNS]
    )
    return panel
