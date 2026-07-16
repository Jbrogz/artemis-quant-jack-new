"""Tests for the momentum signal grid (Task S1, spec §1.3).

The signal is a trailing return over a fixed ``lookback_days`` window, **skipping
the most recent ``skip_days``** — the academic momentum convention (skip the
recent days to avoid short-term reversal). It is built from daily price returns
as a **log-sum** (log returns compound a series through time; spec §1.2 / §1.3),
ported from the author's earlier ``factors/momentum.py`` (funding never enters the signal —
a momentum signal is a price-trajectory predictor).

The cardinal rule asserted here is **no look-ahead**: the signal at date ``t``
uses only data dated ``<= t``. The discriminating test mutates every price
strictly after ``t`` to garbage and asserts ``signal[t]`` is byte-for-byte
unchanged — per-date, not merely on the last date.

Every fixture is synthetic and offline; no API calls.
"""

import numpy as np
import pandas as pd

from amom.config import LOOKBACKS_DAYS, PRIMARY_SKIP_DAYS, ROBUSTNESS_SKIPS
from amom.factor.momentum import build_momentum_signal


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _price_panel(symbol: str, prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="D")
    return pd.DataFrame(
        {"date": dates, "symbol": symbol, "price": [float(p) for p in prices]},
        columns=["date", "symbol", "price"],
    )


# ---------------------------------------------------------------------------
# Frozen grid lives in config (spec §1.3: fixed in advance, not selected)
# ---------------------------------------------------------------------------

def test_frozen_grid_constants():
    assert LOOKBACKS_DAYS == (1, 3, 5, 7, 14, 28, 56)
    assert PRIMARY_SKIP_DAYS == 1
    assert ROBUSTNESS_SKIPS == (2, 3)


# ---------------------------------------------------------------------------
# CARDINAL RULE: no look-ahead — signal[t] uses only data <= t
# ---------------------------------------------------------------------------

def test_no_look_ahead_mutating_future_price_leaves_signal_unchanged():
    """Mutating every price strictly after t must not change signal at any date <= t.

    Discriminating per-date check (not just the last row): garbage future prices
    are injected after a cut date; the entire signal slice up to the cut must be
    identical to the un-mutated baseline.
    """
    # Strictly increasing distinct prices so every daily return is non-trivial.
    prices = [100.0 + i for i in range(40)]
    base_panel = _price_panel("a", prices)

    lookback, skip = 5, PRIMARY_SKIP_DAYS
    base = build_momentum_signal(base_panel, lookback, skip)

    cut = ts("2024-01-20")
    mut_panel = base_panel.copy()
    mut_panel.loc[mut_panel["date"] > cut, "price"] = 1e9  # garbage future
    mut = build_momentum_signal(mut_panel, lookback, skip)

    on_or_before = base.index <= cut
    pd.testing.assert_frame_equal(
        base.loc[on_or_before], mut.loc[mut.index <= cut], check_names=False
    )


# ---------------------------------------------------------------------------
# Hand-computed value: log-sum of daily returns over lookback, shifted by skip
# ---------------------------------------------------------------------------

def test_hand_computed_log_sum_value():
    """A small known price path matches the log-sum-over-lookback-with-skip value.

    Prices: 100,110,121,...  (each day +10%). Daily log return = ln(1.1) each day.
    With lookback=3, skip=1: signal[t] = sum of the 3 daily log returns ending at
    t-skip = t-1, i.e. 3 * ln(1.1).
    """
    # 8 days at a constant +10% step.
    prices = [100.0 * (1.10 ** i) for i in range(8)]
    panel = _price_panel("a", prices)

    sig = build_momentum_signal(panel, lookback_days=3, skip_days=1)
    expected = 3.0 * np.log(1.10)

    # signal at t = day index 5 (2024-01-06): window covers the 3 daily log
    # returns ending at day 4 (skip=1 shifts off day 5's own return).
    s = sig["a"]
    assert np.isclose(s.loc[ts("2024-01-06")], expected)
    # Every fully-warmed date carries the same constant 3*ln(1.1).
    assert np.allclose(s.dropna().to_numpy(), expected)


def test_skip_zero_includes_the_most_recent_return():
    """With skip=0 the window ends at t and includes t's own daily return."""
    prices = [100.0 * (1.10 ** i) for i in range(8)]
    panel = _price_panel("a", prices)

    sig0 = build_momentum_signal(panel, lookback_days=3, skip_days=0)
    # day index 3 (2024-01-04): sum of daily log returns over days 1,2,3.
    assert np.isclose(sig0["a"].loc[ts("2024-01-04")], 3.0 * np.log(1.10))


# ---------------------------------------------------------------------------
# Skip excludes the most recent skip_days
# ---------------------------------------------------------------------------

def test_skip_excludes_the_most_recent_days():
    """The skip drops the most recent daily return(s) from the window.

    Path is flat except a single +50% jump on the LAST day. With skip>=1 the
    jump is excluded from the trailing window, so the signal does not see it;
    with skip=0 it does.
    """
    prices = [100.0] * 9 + [150.0]  # flat, then a +50% jump on the final day
    panel = _price_panel("a", prices)
    last = ts("2024-01-10")

    sig_skip1 = build_momentum_signal(panel, lookback_days=3, skip_days=1)
    sig_skip0 = build_momentum_signal(panel, lookback_days=3, skip_days=0)

    # skip=1 excludes the last day's jump -> trailing window is flat -> 0.
    assert np.isclose(sig_skip1["a"].loc[last], 0.0)
    # skip=0 includes the jump -> signal = ln(1.5) over a window of 2 flat + jump.
    assert np.isclose(sig_skip0["a"].loc[last], np.log(1.5))


def test_skip_two_excludes_two_recent_days():
    """skip=2 (a robustness skip) drops the two most recent daily returns."""
    # +50% jumps on the last two days; skip=2 should exclude both.
    prices = [100.0] * 8 + [150.0, 225.0]
    panel = _price_panel("a", prices)
    last = ts("2024-01-10")

    sig = build_momentum_signal(panel, lookback_days=3, skip_days=2)
    assert np.isclose(sig["a"].loc[last], 0.0)


# ---------------------------------------------------------------------------
# NaN where insufficient history
# ---------------------------------------------------------------------------

def test_nan_on_insufficient_history():
    """Warm-up rows where the (lookback + skip) window is not full are NaN.

    With lookback=5, skip=1 the first non-NaN signal needs 5 daily returns
    ending skip=1 days before t, i.e. 6 prior daily returns -> 7 prices.
    """
    prices = [100.0 + i for i in range(10)]
    panel = _price_panel("a", prices)

    sig = build_momentum_signal(panel, lookback_days=5, skip_days=1)
    s = sig["a"]

    # 10 prices -> 9 daily returns. Need lookback(5)+skip(1)=6 returns to fill;
    # the first fully-warmed signal is on the price-index-6 date (2024-01-07).
    warmup_dates = pd.date_range("2024-01-01", periods=6, freq="D")
    assert s.loc[warmup_dates].isna().all()
    # The first fully-warmed signal lands on 2024-01-07, non-NaN, and every
    # date from there onward is non-NaN (no holes after warm-up).
    assert not np.isnan(s.loc[ts("2024-01-07")])
    assert s.loc[ts("2024-01-07"):].notna().all()


def test_multi_symbol_wide_panel_shape_and_independence():
    """Output is a wide (dates x symbols) panel; symbols computed independently."""
    a = _price_panel("a", [100.0 * (1.10 ** i) for i in range(8)])
    b = _price_panel("b", [100.0 * (0.95 ** i) for i in range(8)])
    panel = pd.concat([a, b], ignore_index=True)

    sig = build_momentum_signal(panel, lookback_days=3, skip_days=1)
    assert set(sig.columns) == {"a", "b"}
    assert sig.index.is_monotonic_increasing
    # a rises (positive signal), b falls (negative signal).
    assert sig["a"].dropna().iloc[-1] > 0
    assert sig["b"].dropna().iloc[-1] < 0
