"""Momentum signal grid (Task S1, spec §1.3).

The momentum signal is a **trailing return over a fixed ``lookback_days`` window,
skipping the most recent ``skip_days``** — the academic convention that skips the
recent days to avoid contaminating the trend with short-term reversal. It is
built from daily price returns as a **log-sum** (log returns are the convention
for compounding a series through time; spec §1.2/§1.3). Ported from the author's earlier
``factors/momentum.py`` with the funding term irrelevant by construction: a
momentum signal is a price-trajectory predictor, never a holding cost.

Definition, per ``(lookback, skip)`` and per symbol ``s``:

    daily_log[t, s] = ln(price[t, s] / price[t-1, s])
    signal[t, s]    = Σ daily_log[k, s] for k in (t - skip - lookback, t - skip]

i.e. the sum of ``lookback`` consecutive daily log returns ending ``skip`` days
before ``t``. The most recent ``skip`` daily returns are excluded.

**No look-ahead (cardinal rule, spec §7):** ``signal[t]`` is a function of prices
dated ``<= t`` only. The rolling sum looks strictly backward and the ``skip``
shift only moves the window further into the past, so mutating any price dated
``> t`` cannot change ``signal[t]``. This is unit-tested per-date.

NaN where the ``(lookback + skip)`` window has not fully filled (warm-up rows).
This function is pure and performs no I/O; eligibility masking and t+1 execution
are applied downstream in portfolio formation (Task S2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pivot_price(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Long ``[date, symbol, price]`` -> wide (dates x symbols) of prices, sorted."""
    priced = price_panel.dropna(subset=["price"]).copy()
    priced["date"] = pd.to_datetime(priced["date"])
    wide = priced.pivot(index="date", columns="symbol", values="price")
    return wide.sort_index()


def build_momentum_signal(
    price_panel: pd.DataFrame, lookback_days: int, skip_days: int
) -> pd.DataFrame:
    """Build the wide momentum signal panel for one ``(lookback, skip)`` variant.

    Args:
        price_panel: long DataFrame ``[date, symbol, price]``. Each symbol's
            realized price series is used as-is; daily returns come from its own
            consecutive prices.
        lookback_days: window length in days (number of daily returns summed).
            Must be a positive integer.
        skip_days: number of most-recent daily returns to exclude (>= 0). The
            primary convention is ``PRIMARY_SKIP_DAYS = 1``.

    Returns:
        Wide DataFrame (dates x symbols) where each cell is the log-sum of
        ``lookback_days`` consecutive daily log returns ending ``skip_days``
        before that date. Cells where that window is not fully filled are NaN.
        Index sorted ascending; columns are the symbols present in the panel.

    Raises:
        ValueError: if ``lookback_days < 1`` or ``skip_days < 0``.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    if skip_days < 0:
        raise ValueError(f"skip_days must be >= 0, got {skip_days}")

    prices = _pivot_price(price_panel)

    # Daily log returns from each symbol's own consecutive prices (strictly
    # backward-looking: row t uses prices t and t-1 only).
    daily_log = np.log(prices / prices.shift(1))

    # Trailing log-sum over the lookback window; require a full window (no
    # partial warm-up sums), then shift by skip to drop the most recent days.
    trailing = daily_log.rolling(window=lookback_days, min_periods=lookback_days).sum()
    signal = trailing.shift(skip_days)

    return signal
