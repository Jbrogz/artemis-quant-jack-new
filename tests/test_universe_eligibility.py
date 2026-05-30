"""Tests for point-in-time eligibility — Task 4 (the anti-bias core).

These tests encode the guide §1.1 / spec §4 Stage 1.1 anti-bias rules:
  - history: (as_of - first_date).days >= MIN_HISTORY_DAYS (90), evaluated as-of,
  - liquidity: trailing-30d ADV >= MIN_ADV_USD ($1M),
  - exclusion: stablecoins ∪ wrapped are never eligible,
  - no look-ahead: eligibility is a pure function of as-of inputs; data dated
    strictly after `as_of` can never change the as-of decision.

All fixtures are synthetic and offline; no API calls.
"""

import inspect

import numpy as np
import pandas as pd

from amom.config import MIN_ADV_USD, MIN_HISTORY_DAYS
from amom.universe.eligibility import eligible_mask, is_eligible


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# History filter is point-in-time (the core no-survivorship-by-recency rule)
# ---------------------------------------------------------------------------

def test_history_filter_is_point_in_time():
    # coin first seen 2025-01-01; ineligible at 2025-02-01 (31d < 90d),
    # eligible at 2025-04-15 (>=90d). No future data consulted.
    assert is_eligible("c", as_of=ts("2025-02-01"), first_date=ts("2025-01-01"),
                       adv_30d=5e6, excluded=set()) is False
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2025-01-01"),
                       adv_30d=5e6, excluded=set()) is True


def test_history_filter_exactly_at_threshold_is_eligible():
    # Exactly MIN_HISTORY_DAYS of history qualifies (>= boundary, not >).
    first = ts("2025-01-01")
    as_of = first + pd.Timedelta(days=MIN_HISTORY_DAYS)
    assert is_eligible("c", as_of=as_of, first_date=first,
                       adv_30d=5e6, excluded=set()) is True
    # One day short is ineligible.
    as_of_short = first + pd.Timedelta(days=MIN_HISTORY_DAYS - 1)
    assert is_eligible("c", as_of=as_of_short, first_date=first,
                       adv_30d=5e6, excluded=set()) is False


# ---------------------------------------------------------------------------
# Liquidity floor
# ---------------------------------------------------------------------------

def test_liquidity_filter():
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2024-01-01"),
                       adv_30d=0.5e6, excluded=set()) is False  # below $1M


def test_liquidity_exactly_at_floor_is_eligible():
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2024-01-01"),
                       adv_30d=MIN_ADV_USD, excluded=set()) is True


def test_nan_adv_is_ineligible():
    # Missing/NaN ADV as-of must fail liquidity, never pass silently.
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2024-01-01"),
                       adv_30d=np.nan, excluded=set()) is False


# ---------------------------------------------------------------------------
# Stablecoin + wrapped exclusion
# ---------------------------------------------------------------------------

def test_stablecoins_and_wrapped_excluded():
    for sym in ("usdt", "wbtc"):
        assert is_eligible(sym, as_of=ts("2025-04-15"), first_date=ts("2020-01-01"),
                           adv_30d=1e9, excluded={"usdt", "wbtc"}) is False


def test_exclusion_overrides_otherwise_eligible():
    # A coin that passes history + liquidity is still rejected if excluded.
    assert is_eligible("usdc", as_of=ts("2025-04-15"), first_date=ts("2018-01-01"),
                       adv_30d=1e12, excluded={"usdc"}) is False


# ---------------------------------------------------------------------------
# No look-ahead — structural AND behavioural
# ---------------------------------------------------------------------------

def test_is_eligible_signature_accepts_no_future_dated_series():
    """Structural no-look-ahead: the function consumes only as-of scalars.

    `is_eligible` must take exactly (symbol, as_of, first_date, adv_30d,
    excluded). first_date and adv_30d are as-of scalars, not time-indexed
    series; there is no parameter through which a future-dated panel could be
    threaded into the decision. This is the contract that makes look-ahead
    impossible by construction.
    """
    params = inspect.signature(is_eligible).parameters
    assert list(params) == ["symbol", "as_of", "first_date", "adv_30d", "excluded"]
    # first_date / adv_30d / excluded are keyword-only (the as-of contract).
    for name in ("first_date", "adv_30d", "excluded"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_no_lookahead_future_data_irrelevant():
    """Behavioural no-look-ahead: mutating data strictly AFTER as_of cannot
    change the eligible set computed at as_of.

    We build a coverage + ADV panel, compute the mask at as_of, then mutate
    every observation dated after as_of (prices vanish, ADV explodes/collapses,
    a coin "delists") and recompute. The two masks must be identical because
    eligible_mask only reads as-of information.
    """
    as_of = ts("2025-06-01")

    coverage = pd.DataFrame({
        "symbol": ["btc", "eth", "newcoin"],
        # btc + eth have long history; newcoin crosses 90d well before as_of.
        "price_first_date": [ts("2020-01-01"), ts("2021-01-01"), ts("2025-02-01")],
    })

    # ADV panel: long [date, symbol, adv_30d] with rows on both sides of as_of.
    dates = pd.date_range("2025-04-01", "2025-08-01", freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for sym in ("btc", "eth", "newcoin"):
            rows.append({"date": d, "symbol": sym, "adv_30d": 5e6 + rng.uniform(0, 1e6)})
    adv = pd.DataFrame(rows)

    excluded = set()
    mask_before = eligible_mask(as_of, coverage, adv, excluded)

    # Mutate the future: rows strictly after as_of get garbage values, and we
    # even drop newcoin's future rows entirely (simulating a later delisting).
    future = adv["date"] > as_of
    adv_mut = adv.copy()
    adv_mut.loc[future, "adv_30d"] = 0.0  # would fail liquidity if consulted
    adv_mut = adv_mut[~(future & (adv_mut["symbol"] == "newcoin"))]

    mask_after = eligible_mask(as_of, coverage, adv_mut, excluded)

    assert mask_before == mask_after
    # And the decision is the real one: all three pass history + liquidity here.
    assert mask_before == {"btc", "eth", "newcoin"}


def test_eligible_mask_uses_latest_adv_at_or_before_as_of():
    """eligible_mask reads the most recent ADV observation <= as_of, never later."""
    as_of = ts("2025-06-01")
    coverage = pd.DataFrame({
        "symbol": ["a", "b"],
        "price_first_date": [ts("2020-01-01"), ts("2020-01-01")],
    })
    adv = pd.DataFrame([
        # 'a': as-of ADV is below floor; only a FUTURE row is above floor.
        {"date": ts("2025-05-31"), "symbol": "a", "adv_30d": 0.4e6},
        {"date": ts("2025-06-30"), "symbol": "a", "adv_30d": 9e9},
        # 'b': as-of ADV is above floor; a future row collapses it.
        {"date": ts("2025-05-31"), "symbol": "b", "adv_30d": 9e9},
        {"date": ts("2025-06-30"), "symbol": "b", "adv_30d": 0.0},
    ])
    mask = eligible_mask(as_of, coverage, adv, set())
    assert mask == {"b"}  # 'a' fails on its as-of ADV; future high ADV ignored.


# ---------------------------------------------------------------------------
# Vectorized eligible_mask over a small coverage + ADV frame
# ---------------------------------------------------------------------------

def test_eligible_mask_applies_all_three_filters():
    as_of = ts("2025-06-01")
    coverage = pd.DataFrame({
        "symbol": ["old_liquid", "young", "illiquid", "usdt", "wbtc"],
        "price_first_date": [
            ts("2020-01-01"),   # old + liquid -> eligible
            ts("2025-05-15"),   # < 90d history -> out
            ts("2020-01-01"),   # old but illiquid -> out
            ts("2018-01-01"),   # stablecoin -> out
            ts("2018-01-01"),   # wrapped -> out
        ],
    })
    adv = pd.DataFrame([
        {"date": as_of, "symbol": "old_liquid", "adv_30d": 5e6},
        {"date": as_of, "symbol": "young", "adv_30d": 5e6},
        {"date": as_of, "symbol": "illiquid", "adv_30d": 0.2e6},
        {"date": as_of, "symbol": "usdt", "adv_30d": 1e10},
        {"date": as_of, "symbol": "wbtc", "adv_30d": 1e10},
    ])
    mask = eligible_mask(as_of, coverage, adv, excluded={"usdt", "wbtc"})
    assert mask == {"old_liquid"}


def test_eligible_mask_missing_adv_symbol_is_ineligible():
    """A symbol in coverage with no ADV row <= as_of fails liquidity."""
    as_of = ts("2025-06-01")
    coverage = pd.DataFrame({
        "symbol": ["has_adv", "no_adv"],
        "price_first_date": [ts("2020-01-01"), ts("2020-01-01")],
    })
    adv = pd.DataFrame([
        {"date": as_of, "symbol": "has_adv", "adv_30d": 5e6},
    ])
    mask = eligible_mask(as_of, coverage, adv, set())
    assert mask == {"has_adv"}


def test_eligible_mask_returns_a_set():
    as_of = ts("2025-06-01")
    coverage = pd.DataFrame({"symbol": ["a"], "price_first_date": [ts("2020-01-01")]})
    adv = pd.DataFrame([{"date": as_of, "symbol": "a", "adv_30d": 5e6}])
    mask = eligible_mask(as_of, coverage, adv, set())
    assert isinstance(mask, set)
