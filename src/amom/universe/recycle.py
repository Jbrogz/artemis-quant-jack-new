"""Recycled-ticker splitter (Task R1, spec §3.6).

A single Artemis ticker is sometimes reused by a *new* project after the
original one collapses (the canonical case being ``ust``). Treating the whole
series as one asset splices a healthy post-revival run onto the dead project's
crash — fabricating a recovery and erasing the realized loss. That is the
survivorship trap §1.1 forbids.

``split_recycled`` detects, per symbol, a **terminal drawdown to near-zero**
(price falls by ``>= drawdown_thresh`` from its running peak *and* reaches a
near-zero fraction of that peak) immediately followed by a **gap of at least
``gap_days``** before the series resumes. Each such crash+gap boundary opens a
new synthetic segment; the post-revival rows are relabeled ``{symbol}__seg{k}``
and the dead segment keeps its crash as its final value.

Properties:
  - **Pure / point-in-time-safe.** The split decision uses only the realized
    price series (a crash + gap that already occurred). Appending later rows to
    a revived segment never moves where an earlier (dead) segment ends.
  - **Survivorship-preserving.** A crash with *no* later resumption is a genuine
    death, left as a single segment ending at the crash — never dropped.
  - **Conservative.** A shallow drawdown, or a gap shorter than ``gap_days``, is
    treated as continuation and the symbol is left untouched (no suffix).
"""
from __future__ import annotations

import pandas as pd

# Fraction of the running peak at/below which a price counts as "near-zero".
# A terminal drawdown of >= drawdown_thresh already implies price <=
# (1 - drawdown_thresh) * peak; this constant makes the near-zero requirement
# explicit and independent so a 90% drop to a still-sizable level (relative to a
# later much-higher peak) is not mistaken for a death.
_NEAR_ZERO_FRACTION = 0.1


def _segment_boundaries(
    dates: pd.Series, prices: pd.Series, *, drawdown_thresh: float, gap_days: int
) -> list[int]:
    """Return the positional indices (into the sorted series) at which a new
    segment begins, i.e. the first observation *after* a death+gap boundary.

    A boundary exists at position ``i`` (i >= 1) iff:
      - the series up to ``i-1`` reached a terminal drawdown to near-zero
        (running-peak drawdown >= ``drawdown_thresh`` and the last price before
        the gap is <= ``_NEAR_ZERO_FRACTION`` * that peak), AND
      - the calendar gap from ``i-1`` to ``i`` is > ``gap_days``.
    """
    starts: list[int] = []
    peak = float(prices.iloc[0])
    crashed = False  # has the current (open) segment hit a terminal near-zero?

    for i in range(1, len(prices)):
        prev_price = float(prices.iloc[i - 1])
        cur_price = float(prices.iloc[i])

        # Track the running peak and whether we've terminally crashed to ~0.
        if prev_price > peak:
            peak = prev_price
        if peak > 0:
            drawdown = 1.0 - (prev_price / peak)
            if drawdown >= drawdown_thresh and prev_price <= _NEAR_ZERO_FRACTION * peak:
                crashed = True

        gap = (dates.iloc[i] - dates.iloc[i - 1]).days
        if crashed and gap > gap_days:
            # Recycled-ticker boundary: open a fresh segment at position i.
            starts.append(i)
            # Reset the per-segment crash state and peak for the new project.
            peak = cur_price
            crashed = False
        else:
            # Continuation within the same segment.
            if cur_price > peak:
                peak = cur_price

    return starts


def split_recycled(
    price_panel: pd.DataFrame,
    *,
    drawdown_thresh: float = 0.9,
    gap_days: int = 45,
) -> pd.DataFrame:
    """Split recycled tickers into distinct synthetic assets.

    Args:
        price_panel: long DataFrame ``[date, symbol, price]``.
        drawdown_thresh: minimum running-peak drawdown (e.g. 0.9 = 90%) that,
            together with a near-zero terminal price, marks a death.
        gap_days: minimum calendar gap (days) after a death before a resumption
            counts as a new project rather than a data outage.

    Returns:
        A new long DataFrame with the same columns and row count. Symbols whose
        realized series contains one or more death+gap boundaries are relabeled
        ``{symbol}__seg0``, ``{symbol}__seg1``, … in chronological order; all
        other symbols (and their rows) are returned unchanged.
    """
    if price_panel.empty:
        return price_panel.copy()

    out_frames: list[pd.DataFrame] = []

    for symbol, group in price_panel.groupby("symbol", sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        starts = _segment_boundaries(
            g["date"], g["price"], drawdown_thresh=drawdown_thresh, gap_days=gap_days
        )

        if not starts:
            # Healthy / single-death series: leave the symbol untouched.
            out_frames.append(g)
            continue

        # Assign a segment id by counting how many boundaries precede each row.
        bounds = [0, *starts, len(g)]
        relabeled = g.copy()
        seg_symbols = relabeled["symbol"].to_numpy(dtype=object)
        for seg_idx in range(len(bounds) - 1):
            lo, hi = bounds[seg_idx], bounds[seg_idx + 1]
            seg_symbols[lo:hi] = f"{symbol}__seg{seg_idx}"
        relabeled["symbol"] = seg_symbols
        out_frames.append(relabeled)

    result = pd.concat(out_frames, ignore_index=True)
    return result[list(price_panel.columns)]
