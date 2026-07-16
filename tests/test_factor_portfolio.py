"""Tests for dollar-neutral momentum long/short formation (Task S2, spec §1.4).

Formation ports the author's earlier ``momentum_portfolio.py`` with **funding stripped**:
the factor P&L is driven by the spot ``holding_return`` panel (Task S0), never a
funding-adjusted return (Artemis is spot-only; spec §3.1).

Construction (spec §1.4 / guide §1.4):
  - quintile sort (``QUANTILE = 0.20``): long the top 20%, short the bottom 20%;
  - **equal-weight** within each leg; **dollar-neutral** (Σ weights = 0, the long
    leg's dollar weight equals the short leg's);
  - ``factor_return = long_leg_return − short_leg_return``;
  - eligibility-masked per the universe panel; min-bucket gate
    (``MIN_BUCKET_SIZE``): a rebalance whose long or short bucket is too thin is
    skipped.

**Cardinal rule — no look-ahead (spec §7):** the signal computed through close
``t`` drives the trade entered at close ``t+1``. In the rebalance-date framing
the rebalance/entry date is ``r``; the signal it uses is the most recent one
dated ``<= r − LAG_DAYS`` (i.e. ``t``); the holding window is the strictly-later
``(r, r + HOLDING_DAYS]``. The discriminating test mutates *post-``t``* signal
values to garbage and asserts the bucket chosen for the ``t → t+1`` trade is
unchanged — the formation never reads the signal at ``r`` (= ``t+1``) itself.

Every fixture is synthetic and offline; no API calls.
"""

import numpy as np
import pandas as pd

from amom.config import HOLDING_DAYS, LAG_DAYS, MIN_BUCKET_SIZE, QUANTILE
from amom.factor.portfolio import (
    build_factor_returns,
    compute_factor_returns,
    select_buckets,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _wide(values: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    """Build a wide (dates x symbols) panel from a {symbol: [series]} dict."""
    n = len(next(iter(values.values())))
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(values, index=idx)


def _elig_all_true(like: pd.DataFrame) -> pd.DataFrame:
    """A wide eligibility mask that is True everywhere ``like`` is shaped."""
    return pd.DataFrame(True, index=like.index, columns=like.columns)


def _grid_signal(
    n_names: int = 15, start: str = "2024-01-01", periods: int = 40
) -> pd.DataFrame:
    """A signal panel for ``n_names`` symbols (s0..) with a clean monotone XS.

    On every date symbol ``si`` carries signal value ``i`` (highest = last index,
    lowest = s0). With ``n_names = 15`` and ``QUANTILE = 0.20`` the quintile legs
    are exactly 3 names each (top-3 / bottom-3), meeting ``MIN_BUCKET_SIZE = 3``.
    """
    idx = pd.date_range(start, periods=periods, freq="D")
    cols = [f"s{i:02d}" for i in range(n_names)]
    data = {c: [float(i)] * periods for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Config grid is frozen (spec §1.4)
# ---------------------------------------------------------------------------

def test_formation_constants_frozen():
    assert QUANTILE == 0.20
    assert LAG_DAYS == 1
    assert HOLDING_DAYS == 30
    assert MIN_BUCKET_SIZE == 3


# ---------------------------------------------------------------------------
# (a) dollar-neutrality: equal leg dollar weight, Σ weights ≈ 0
# ---------------------------------------------------------------------------

def test_dollar_neutral_weights_sum_to_zero():
    sig = _grid_signal()
    elig = _elig_all_true(sig)
    r = ts("2024-01-20")

    holdings = select_buckets(sig, elig, r)
    assert holdings is not None

    total = float(holdings["weight"].sum())
    assert np.isclose(total, 0.0, atol=1e-12)

    long_dollars = float(holdings.loc[holdings["position"] == "long", "weight"].sum())
    short_dollars = float(holdings.loc[holdings["position"] == "short", "weight"].sum())
    # Equal dollar on each leg: long sums to +1, short to -1 (gross 2, net 0).
    assert np.isclose(long_dollars, 1.0)
    assert np.isclose(short_dollars, -1.0)
    assert np.isclose(long_dollars + short_dollars, 0.0)


# ---------------------------------------------------------------------------
# (c) equal-weight within each leg
# ---------------------------------------------------------------------------

def test_equal_weight_within_each_leg():
    sig = _grid_signal()
    elig = _elig_all_true(sig)
    r = ts("2024-01-20")

    holdings = select_buckets(sig, elig, r)
    assert holdings is not None

    longs = holdings[holdings["position"] == "long"]
    shorts = holdings[holdings["position"] == "short"]

    # Every long weight identical; every short weight identical.
    assert longs["weight"].nunique() == 1
    assert shorts["weight"].nunique() == 1
    assert np.isclose(longs["weight"].iloc[0], 1.0 / len(longs))
    assert np.isclose(shorts["weight"].iloc[0], -1.0 / len(shorts))

    # The quintile picks for N=15, q=0.2: top-3 = {s12,s13,s14}, bottom-3 = {s00,s01,s02}.
    assert set(longs["symbol"]) == {"s12", "s13", "s14"}
    assert set(shorts["symbol"]) == {"s00", "s01", "s02"}


# ---------------------------------------------------------------------------
# (b) NO LOOK-AHEAD — signal at t (=r-LAG), not t+1 (=r), picks buckets; and
#     mutating post-t signal data does not change the t -> t+1 trade.
# ---------------------------------------------------------------------------

def test_buckets_use_signal_at_r_minus_lag_not_at_r():
    """The signal driving the rebalance at r is the one dated r − LAG_DAYS.

    Set a *clean* cross-section on the lagged date (r-1) and a *flipped* one on r
    itself. The buckets must reflect the r-1 ordering; if the formation peeked at
    r (= t+1, the entry date) the long/short sets would flip.
    """
    sig = _grid_signal()
    n = sig.shape[1]
    r = ts("2024-01-20")
    lag_date = r - pd.Timedelta(days=LAG_DAYS)

    # Flip the ordering on the entry date r itself (s00 highest, s14 lowest).
    flipped = sig.copy()
    flipped.loc[r] = [float(n - 1 - i) for i in range(n)]
    elig = _elig_all_true(flipped)

    holdings = select_buckets(flipped, elig, r)
    assert holdings is not None
    longs = set(holdings.loc[holdings["position"] == "long", "symbol"])
    shorts = set(holdings.loc[holdings["position"] == "short", "symbol"])

    # Driven by r-1 (clean ordering): top = {s12,s13,s14}, bottom = {s00,s01,s02}.
    assert longs == {"s12", "s13", "s14"}, f"buckets must use signal at {lag_date}, not {r}"
    assert shorts == {"s00", "s01", "s02"}


def test_no_look_ahead_mutating_post_t_signal_does_not_change_trade():
    """Mutating every signal value strictly after t (= r − LAG_DAYS) to garbage
    must not change the bucket chosen for the t → t+1 trade at r."""
    sig = _grid_signal()
    elig = _elig_all_true(sig)
    r = ts("2024-01-20")
    t = r - pd.Timedelta(days=LAG_DAYS)

    base = select_buckets(sig, elig, r)
    assert base is not None

    mutated = sig.copy()
    mutated.loc[mutated.index > t] = 1e9  # garbage on every date strictly after t
    after = select_buckets(mutated, _elig_all_true(mutated), r)
    assert after is not None

    pd.testing.assert_frame_equal(
        base.sort_values("symbol").reset_index(drop=True),
        after.sort_values("symbol").reset_index(drop=True),
    )


def test_holding_window_is_strictly_after_entry_date():
    """factor_return for rebalance r compounds returns over (r, r+HOLDING_DAYS] —
    strictly after the entry date r (entry at close r = close t+1). The entry-date
    return itself is NOT earned (the position is opened at that close)."""
    # 3 long, 3 short, all eligible; signal is a clean monotone cross-section.
    n = 12
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(6)]
    sig = pd.DataFrame({c: [float(i)] * n for i, c in enumerate(cols)}, index=idx)
    elig = _elig_all_true(sig)

    # Returns: put a HUGE return ON the entry date r for a long name; if the
    # window wrongly included r it would dominate. It must be excluded.
    r = idx[5]
    next_r = idx[8]
    rets = pd.DataFrame(0.0, index=idx, columns=cols)
    rets.loc[r, "s5"] = 10.0          # entry-date return -> must be EXCLUDED
    rets.loc[idx[6], "s5"] = 0.10     # day after entry -> included
    rets.loc[idx[7], "s5"] = 0.10     # included

    holdings = pd.concat(
        [select_buckets(sig, elig, r, min_bucket_size=2)], ignore_index=True
    )
    fr = compute_factor_returns(
        holdings, rets, pd.DatetimeIndex([r, next_r]), min_bucket_size=2
    )
    # s5's contribution over (r, next_r] = (1.10)(1.10) - 1 = 0.21, NOT ~10.
    assert r in fr.index
    long_ret = fr.loc[r, "long_return"]
    assert long_ret < 1.0, "entry-date return leaked into the holding window"
    assert long_ret > 0.0


# ---------------------------------------------------------------------------
# (d) survivorship: a collapsed coin in the short leg flows its crash into P&L
# ---------------------------------------------------------------------------

def test_collapsed_short_leg_coin_crash_flows_into_pnl():
    """A coin in the SHORT leg that crashes ~−95% over the hold makes the short
    leg very negative; since factor = long − short, the crash is a positive
    payoff to the dollar-neutral book (the survivorship payoff flows through)."""
    n = 12
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(6)]
    # s0,s1,s2 are losers (short); s3,s4,s5 winners (long).
    sig = pd.DataFrame({c: [float(i)] * n for i, c in enumerate(cols)}, index=idx)
    elig = _elig_all_true(sig)

    r = idx[5]
    next_r = idx[8]
    rets = pd.DataFrame(0.0, index=idx, columns=cols)
    # s0 (a short name) collapses: -95% realized on the first held day, then NaN
    # (it stops reporting — closed at the last observed price).
    rets.loc[idx[6], "s0"] = -0.95
    rets.loc[idx[7], "s0"] = np.nan

    holdings = select_buckets(sig, elig, r, min_bucket_size=2)
    assert holdings is not None
    assert "s0" in set(holdings.loc[holdings["position"] == "short", "symbol"])

    fr = compute_factor_returns(
        holdings, rets, pd.DatetimeIndex([r, next_r]), min_bucket_size=2
    )
    # short leg average return is dragged sharply negative by s0's -95%.
    assert fr.loc[r, "short_return"] < -0.25
    # factor = long − short: the short crash is a POSITIVE contribution.
    assert fr.loc[r, "factor_return"] > 0.25


# ---------------------------------------------------------------------------
# (e) min-bucket gate: too-thin a bucket skips the rebalance
# ---------------------------------------------------------------------------

def test_min_bucket_gate_skips_thin_rebalance():
    """When the eligible cross-section can't field MIN_BUCKET_SIZE names per leg,
    the rebalance is skipped (select_buckets returns None)."""
    # Only 4 eligible names -> quintile legs would be 1 each < MIN_BUCKET_SIZE(3).
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(4)]
    sig = pd.DataFrame({c: [float(i)] * n for i, c in enumerate(cols)}, index=idx)
    elig = _elig_all_true(sig)
    r = ts("2024-01-06")

    # default MIN_BUCKET_SIZE = 3; 4 names can't field two buckets of >=3.
    assert select_buckets(sig, elig, r) is None


def test_eligibility_mask_excludes_ineligible_names():
    """Ineligible names on the rebalance date are dropped from the cross-section
    before bucketing, even if their (lagged) signal is finite."""
    sig = _grid_signal()
    elig = _elig_all_true(sig)
    r = ts("2024-01-20")

    # Make the two would-be top names (s13, s14) ineligible on the rebalance date.
    elig = elig.copy()
    elig.loc[r, "s14"] = False
    elig.loc[r, "s13"] = False

    holdings = select_buckets(sig, elig, r)
    assert holdings is not None
    longs = set(holdings.loc[holdings["position"] == "long", "symbol"])
    # With s13,s14 removed, the top-3 of the remaining 13 (s00..s12) are {s10,s11,s12}.
    assert "s14" not in longs and "s13" not in longs
    assert longs == {"s10", "s11", "s12"}


# ---------------------------------------------------------------------------
# End-to-end pure orchestrator: build_factor_returns over a rebalance schedule
# ---------------------------------------------------------------------------

def test_build_factor_returns_end_to_end_no_lookahead_and_neutral():
    """The pure orchestrator builds a gap-free per-rebalance factor-return series;
    mutating signal data after each rebalance's lagged signal date leaves the
    series unchanged (no look-ahead end-to-end)."""
    periods = 80
    sig = _grid_signal(periods=periods)
    n = sig.shape[1]
    elig = _elig_all_true(sig)
    # Winners (high signal) keep rising; losers (low signal) keep falling, so the
    # long-minus-short factor return is reliably positive.
    rets = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    mid = (n - 1) / 2.0
    for i, c in enumerate(sig.columns):
        rets[c] = 0.01 * (i - mid)  # s14 -> +0.07/day, s00 -> -0.07/day

    rebal = sig.index[::HOLDING_DAYS]
    base = build_factor_returns(sig, elig, rets, rebal)

    assert not base.empty
    assert list(base.columns) == [
        "rebalance_date",
        "factor_return",
        "long_return",
        "short_return",
        "n_long",
        "n_short",
    ]
    # Dollar-neutral + winners>losers -> every factor return positive.
    assert (base["factor_return"] > 0).all()

    # No look-ahead end-to-end: garble all signal data after the LAST rebalance's
    # lagged signal date; the produced series must be byte-for-byte identical
    # (the last rebalance has no next rebalance, so it produces no return row).
    last_lag = rebal[-1] - pd.Timedelta(days=LAG_DAYS)
    mut = sig.copy()
    mut.loc[mut.index > last_lag] = -1e9
    after = build_factor_returns(mut, elig, rets, rebal)
    pd.testing.assert_frame_equal(base, after)
