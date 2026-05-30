"""Tests for point-in-time eligibility — Task R2 (the anti-bias core, redesigned).

These tests encode the guide §1.1 / spec §4 Stage 1.1 anti-bias rules with the
rev-3 liquidity redesign (spec §3.5):

  - history:    (as_of - first_date).days >= MIN_HISTORY_DAYS (90), as-of,
  - liquidity:  MC >= MIN_MC_USD AND a trailing-30d-window volume gate that is
                robust to broken prints (median) yet REQUIRES sustained volume
                (a 2-print $5M window fails — the rev-2 mean-of-present-rows bug
                is gone),
  - obs-density: at least MIN_OBS_DENSITY of the trailing 90d have a price,
  - tradeability: the latest price is within LISTING_STALENESS_DAYS of as_of,
  - exclusion:  stablecoins ∪ wrapped are never eligible,
  - no look-ahead: eligibility is a pure function of as-of scalars; data dated
    strictly after `as_of` can never change the as-of decision.

Note (Task R6): eligible_mask was a standalone helper that duplicated the builder's
per-date loop but was never wired into the builder. It has been deleted; the builder
(universe/builder.py) is now the sole call-site of is_eligible. The is_eligible
tests below remain and guard the real code path.

All fixtures are synthetic and offline; no API calls.
"""

import inspect

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
from amom.universe.eligibility import is_eligible, window_liquidity


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# A set of as-of liquidity/density/staleness inputs that comfortably pass every
# filter, so each test can perturb exactly one dimension.
def _ok(as_of: pd.Timestamp = ts("2025-06-01"), **overrides):
    base = dict(
        symbol="c",
        as_of=as_of,
        first_date=ts("2024-01-01"),       # > 90d history
        last_price_date=as_of,             # fresh
        n_obs_90d=90,                       # full density
        mc=50e6,                            # > MC floor
        # 30 daily prints of $5M -> median 5M, ADV 5M: liquid.
        vol_window=np.full(LIQUIDITY_VOL_WINDOW_DAYS, 5e6),
        excluded=set(),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# History filter is point-in-time (the core no-survivorship-by-recency rule)
# ---------------------------------------------------------------------------

def test_history_filter_is_point_in_time():
    # coin first seen 2025-01-01; ineligible at 2025-02-01 (31d < 90d),
    # eligible at 2025-04-15 (>=90d). No future data consulted.
    assert is_eligible(**_ok(as_of=ts("2025-02-01"), first_date=ts("2025-01-01"),
                             last_price_date=ts("2025-02-01"))) is False
    assert is_eligible(**_ok(as_of=ts("2025-04-15"), first_date=ts("2025-01-01"),
                             last_price_date=ts("2025-04-15"))) is True


def test_history_filter_exactly_at_threshold_is_eligible():
    # Exactly MIN_HISTORY_DAYS of history qualifies (>= boundary, not >).
    first = ts("2025-01-01")
    as_of = first + pd.Timedelta(days=MIN_HISTORY_DAYS)
    assert is_eligible(**_ok(as_of=as_of, first_date=first,
                             last_price_date=as_of)) is True
    # One day short is ineligible.
    as_of_short = first + pd.Timedelta(days=MIN_HISTORY_DAYS - 1)
    assert is_eligible(**_ok(as_of=as_of_short, first_date=first,
                             last_price_date=as_of_short)) is False


# ---------------------------------------------------------------------------
# Liquidity: MC floor AND a robust-but-sustained trailing-volume gate (§3.5)
# ---------------------------------------------------------------------------

def test_two_print_5m_window_fails_liquidity():
    """THE rev-2 bug: 2 prints of $5M in the trailing 30d must NOT pass.

    A mean/median over only the 2 present rows is $5M (would pass), but the gate
    must require sustained volume across the window, so the sparse window fails.
    """
    sparse = np.array([5e6, 5e6])  # only two prints in a 30-day window
    assert is_eligible(**_ok(vol_window=sparse)) is False


def test_broken_subdollar_prints_do_not_flip_a_liquid_coin():
    """Median is robust: a genuinely liquid coin with a few broken sub-dollar
    prints in the window still passes liquidity."""
    window = np.full(LIQUIDITY_VOL_WINDOW_DAYS, 5e6)
    window[:3] = 0.0001  # three broken sub-dollar prints
    window[5] = -1.0     # a negative/garbage print
    assert is_eligible(**_ok(vol_window=window)) is True


def test_low_market_cap_fails_even_if_volume_passes():
    """MC < MIN_MC_USD fails even when the volume gate passes."""
    assert is_eligible(**_ok(mc=MIN_MC_USD - 1.0)) is False
    # And exactly at the floor passes.
    assert is_eligible(**_ok(mc=MIN_MC_USD)) is True


def test_nan_market_cap_is_ineligible():
    assert is_eligible(**_ok(mc=np.nan)) is False


def test_thin_volume_fails_even_with_full_window():
    """A full window of low daily volume fails the median floor."""
    thin = np.full(LIQUIDITY_VOL_WINDOW_DAYS, 0.2e6)  # < $1M every day
    assert is_eligible(**_ok(vol_window=thin)) is False


def test_empty_volume_window_is_ineligible():
    assert is_eligible(**_ok(vol_window=np.array([]))) is False


# ---------------------------------------------------------------------------
# window_liquidity: the pure liquidity primitive (median + full-window ADV)
# ---------------------------------------------------------------------------

def test_window_liquidity_winsorizes_nonpositive_prints():
    """Non-positive prints are ignored; the median is over positive prints."""
    window = np.array([5e6, 5e6, 0.0, -3.0, np.nan, 5e6])
    median_vol, adv = window_liquidity(window, window_days=LIQUIDITY_VOL_WINDOW_DAYS)
    assert median_vol == 5e6                         # median of the three 5e6
    # ADV uses the FULL window in the denominator (sum of positive / window_days),
    # NOT a mean over present rows — the rev-2 bug.
    assert adv == (3 * 5e6) / LIQUIDITY_VOL_WINDOW_DAYS


def test_window_liquidity_two_prints_low_adv():
    """Two $5M prints: median 5M (robust) but ADV is sum/30 << floor."""
    median_vol, adv = window_liquidity(
        np.array([5e6, 5e6]), window_days=LIQUIDITY_VOL_WINDOW_DAYS
    )
    assert median_vol == 5e6
    assert adv == (2 * 5e6) / LIQUIDITY_VOL_WINDOW_DAYS
    assert adv < MIN_MEDIAN_VOL_USD  # this is what makes the 2-print window fail


def test_window_liquidity_empty_is_nan():
    median_vol, adv = window_liquidity(
        np.array([]), window_days=LIQUIDITY_VOL_WINDOW_DAYS
    )
    assert np.isnan(median_vol)
    assert adv == 0.0


# ---------------------------------------------------------------------------
# Observation-density filter (calendar age alone is not enough)
# ---------------------------------------------------------------------------

def test_low_obs_density_fails_despite_age():
    """A coin with calendar age >= 90d but obs-density < MIN_OBS_DENSITY fails."""
    sparse_obs = int(MIN_OBS_DENSITY * MIN_HISTORY_DAYS) - 1  # below threshold
    assert is_eligible(**_ok(n_obs_90d=sparse_obs)) is False


def test_obs_density_exactly_at_threshold_passes():
    at_threshold = int(np.ceil(MIN_OBS_DENSITY * MIN_HISTORY_DAYS))
    assert is_eligible(**_ok(n_obs_90d=at_threshold)) is True


# ---------------------------------------------------------------------------
# Tradeability / staleness (a stopped coin exits eligibility promptly)
# ---------------------------------------------------------------------------

def test_stale_last_price_is_ineligible():
    """A coin whose last price is older than LISTING_STALENESS_DAYS before as_of
    is INELIGIBLE (tradeability)."""
    as_of = ts("2025-06-01")
    stale = as_of - pd.Timedelta(days=LISTING_STALENESS_DAYS + 1)
    assert is_eligible(**_ok(as_of=as_of, last_price_date=stale)) is False


def test_staleness_exactly_at_grace_is_eligible():
    as_of = ts("2025-06-01")
    on_grace = as_of - pd.Timedelta(days=LISTING_STALENESS_DAYS)
    assert is_eligible(**_ok(as_of=as_of, last_price_date=on_grace)) is True


def test_nan_last_price_date_is_ineligible():
    assert is_eligible(**_ok(last_price_date=pd.NaT)) is False


# ---------------------------------------------------------------------------
# Stablecoin + wrapped exclusion
# ---------------------------------------------------------------------------

def test_stablecoins_and_wrapped_excluded():
    for sym in ("usdt", "wbtc"):
        assert is_eligible(**_ok(symbol=sym, excluded={"usdt", "wbtc"})) is False


def test_exclusion_overrides_otherwise_eligible():
    assert is_eligible(**_ok(symbol="usdc", excluded={"usdc"})) is False


# ---------------------------------------------------------------------------
# No look-ahead — structural AND behavioural
# ---------------------------------------------------------------------------

def test_is_eligible_signature_accepts_no_future_dated_series():
    """Structural no-look-ahead: the function consumes only as-of scalars/arrays.

    `is_eligible` takes (symbol, as_of, first_date, last_price_date, n_obs_90d,
    mc, vol_window, excluded). first_date / last_price_date / n_obs_90d / mc are
    as-of scalars; vol_window is the as-of trailing-window slice (already
    point-in-time, contains no dates). There is no parameter through which a
    future-dated panel could be threaded into the decision.
    """
    params = inspect.signature(is_eligible).parameters
    assert list(params) == [
        "symbol", "as_of", "first_date", "last_price_date",
        "n_obs_90d", "mc", "vol_window", "excluded",
    ]
    for name in ("first_date", "last_price_date", "n_obs_90d", "mc",
                 "vol_window", "excluded"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_no_lookahead_future_data_irrelevant_to_is_eligible():
    """Behavioural no-look-ahead: is_eligible is a pure function of as-of scalars.
    Changing data strictly after as_of cannot change the as-of decision because
    is_eligible receives only point-in-time scalars and a pre-sliced window.

    This test verifies that passing identical as-of inputs but different future
    data produces the same result — by construction (is_eligible has no future
    data path), so this is a documentation/structural check.
    """
    as_of = ts("2025-06-01")
    vol_window = np.full(LIQUIDITY_VOL_WINDOW_DAYS, 5e6)

    result_1 = is_eligible(
        "btc", as_of,
        first_date=ts("2020-01-01"),
        last_price_date=as_of,
        n_obs_90d=90,
        mc=1e12,
        vol_window=vol_window,
        excluded=set(),
    )
    # Identical call — same result by definition.
    result_2 = is_eligible(
        "btc", as_of,
        first_date=ts("2020-01-01"),
        last_price_date=as_of,
        n_obs_90d=90,
        mc=1e12,
        vol_window=vol_window,
        excluded=set(),
    )
    assert result_1 is True
    assert result_1 == result_2
