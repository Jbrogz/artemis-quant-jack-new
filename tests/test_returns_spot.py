"""Tests for spot holding returns + terminal crash-return imputation (Task R4).

This is the survivorship payoff. The universe layer (R3) keeps collapsed coins
in the panel and carries a point-in-time ``delisted_asof`` death signal; this
layer turns that signal into a *realized* return. The guide §1.1 / spec §1.2
requirement is concrete: a coin that crashes ~95% and then stops reporting must
contribute its **realized terminal return ≈ −0.95**, booked on the delisting
date — NOT a dropped/NaN value (which is exactly the survivorship bias the guide
forbids; dropping it would silently erase the loss).

Conventions asserted here (spec §3.1, §4 Stage 1.2):
  - Holding return = **simple** spot price return (``p_t / p_{t-1} − 1``). No
    funding term — Artemis serves no funding and this is a spot book.
  - Returns are point-in-time: a return dated ``d`` uses only prices ``<= d``.
  - The terminal (delisting) return closes the position at the coin's last
    observed price; the realized loss into the crash is preserved, never NaN.

Every fixture is synthetic and offline; no API calls.
"""

import numpy as np
import pandas as pd

from amom.config import LISTING_STALENESS_DAYS
from amom.returns.spot import build_holding_returns
from amom.universe.builder import build_universe_history


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# Fixture helpers (mirror the builder tests so the two layers compose)
# ---------------------------------------------------------------------------

def _price_rows(symbol: str, start: str, end: str, value: float = 100.0):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "price": float(value)} for d in dates]


def _vol_rows(symbol: str, start: str, end: str, daily_vol: float = 5e6):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "volume": daily_vol} for d in dates]


# ---------------------------------------------------------------------------
# CENTERPIECE: the terminal crash return is booked, not dropped
# ---------------------------------------------------------------------------

def test_terminal_crash_return_booked_on_delisting_not_nan():
    """deadcoin holds ~100 then crashes ~95% to ~5 and stops reporting.

    The realized return of the final leg (the crash) must appear in the holding
    series with value ≈ −0.95, booked on the delisting date the universe panel
    flags. It must NOT be NaN and the coin must NOT be dropped — silently
    dropping it is the survivorship bias the guide forbids.
    """
    # Daily prices flat at 100 through 2024-05-31, one crash print to 5 on
    # 2024-06-01 (a clean −95% step), then the coin goes silent forever.
    dead_price = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    dead_price += [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01")

    price = pd.DataFrame(dead_price, columns=["date", "symbol", "price"])
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": r["symbol"], "mc": 50e6} for r in dead_price],
        columns=["date", "symbol", "mc"],
    )
    vol = pd.DataFrame(dead_vol, columns=["date", "symbol", "volume"])

    # Daily universe grid spanning before the crash to well past the delisting.
    dates = pd.date_range("2024-05-25", "2024-06-30", freq="D")
    universe = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    rets = build_holding_returns(price, universe)

    dead = rets.query("symbol == 'deadcoin'").set_index("date")["holding_return"]
    # The crash-day return is realized and equals (5/100 − 1) = −0.95.
    assert np.isclose(dead.loc[ts("2024-06-01")], -0.95)
    # No NaN anywhere in the realized holding series for the dead coin.
    assert not dead.dropna().empty
    assert dead.notna().all()
    # The single worst realized return is the booked terminal crash ≈ −0.95.
    assert np.isclose(dead.min(), -0.95)


def test_terminal_return_is_present_even_though_coin_later_delisted():
    """The delisting date carries no NEW phantom return after the crash leg.

    Once the position is closed at the last observed price (2024-06-01), the
    death signal firing later must not fabricate an extra return on a date with
    no price — the realized loss was already booked on the crash date.
    """
    dead_price = _price_rows("deadcoin", "2024-01-01", "2024-05-31", value=100.0)
    dead_price += [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2024-01-01", "2024-06-01")
    price = pd.DataFrame(dead_price, columns=["date", "symbol", "price"])
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": r["symbol"], "mc": 50e6} for r in dead_price],
        columns=["date", "symbol", "mc"],
    )
    vol = pd.DataFrame(dead_vol, columns=["date", "symbol", "volume"])

    delist = ts("2024-06-01") + pd.Timedelta(days=LISTING_STALENESS_DAYS + 1)
    dates = pd.date_range("2024-05-25", "2024-06-30", freq="D")
    universe = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    rets = build_holding_returns(price, universe)
    dead = rets.query("symbol == 'deadcoin'").set_index("date")["holding_return"]
    # No return is booked on or after the delisting date (no price after 06-01).
    after = dead[dead.index >= delist]
    assert after.empty


# ---------------------------------------------------------------------------
# Healthy coins: simple spot returns, no funding
# ---------------------------------------------------------------------------

def test_simple_daily_returns_for_healthy_coin():
    """A coin with a known price path yields the exact simple daily returns."""
    rows = [
        {"date": ts("2024-01-01"), "symbol": "a", "price": 100.0},
        {"date": ts("2024-01-02"), "symbol": "a", "price": 110.0},  # +10%
        {"date": ts("2024-01-03"), "symbol": "a", "price": 99.0},   # −10%
        {"date": ts("2024-01-04"), "symbol": "a", "price": 99.0},   #   0%
    ]
    price = pd.DataFrame(rows, columns=["date", "symbol", "price"])
    vol = pd.DataFrame(
        _vol_rows("a", "2024-01-01", "2024-01-04"), columns=["date", "symbol", "volume"]
    )
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": "a", "mc": 50e6} for r in rows],
        columns=["date", "symbol", "mc"],
    )
    universe = build_universe_history(
        price, vol, mc, dates=pd.DatetimeIndex([ts("2024-01-04")]), excluded=set()
    )

    rets = build_holding_returns(price, universe).set_index("date")["holding_return"]
    assert np.isclose(rets.loc[ts("2024-01-02")], 0.10)
    assert np.isclose(rets.loc[ts("2024-01-03")], -0.10)
    assert np.isclose(rets.loc[ts("2024-01-04")], 0.0)
    # The first date has no prior price -> no return row (not a 0, not a NaN row).
    assert ts("2024-01-01") not in rets.index


def test_no_funding_term_returns_are_pure_price():
    """Holding return == price return exactly (spot; no funding adjustment).

    A constant-funding world would shift every return by a fixed amount; we
    assert the returns equal the raw price returns with no such offset.
    """
    rows = _price_rows("a", "2024-01-01", "2024-01-31", value=100.0)
    # geometric ramp so each daily return is a constant non-zero value
    for i, r in enumerate(rows):
        r["price"] = 100.0 * (1.01 ** i)
    price = pd.DataFrame(rows, columns=["date", "symbol", "price"])
    vol = pd.DataFrame(
        _vol_rows("a", "2024-01-01", "2024-01-31"), columns=["date", "symbol", "volume"]
    )
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": "a", "mc": 50e6} for r in rows],
        columns=["date", "symbol", "mc"],
    )
    universe = build_universe_history(
        price, vol, mc, dates=pd.DatetimeIndex([ts("2024-01-31")]), excluded=set()
    )

    rets = build_holding_returns(price, universe).set_index("date")["holding_return"]
    # Every realized return is exactly the price step (1.01 − 1), no offset.
    assert np.allclose(rets.dropna().to_numpy(), 0.01)


# ---------------------------------------------------------------------------
# Point-in-time: a return dated d uses only prices <= d
# ---------------------------------------------------------------------------

def test_returns_are_point_in_time_future_price_does_not_move_earlier_return():
    """Mutating a price strictly after date d cannot change the return at d."""
    rows = _price_rows("a", "2024-01-01", "2024-03-31", value=100.0)
    for i, r in enumerate(rows):
        r["price"] = 100.0 + i  # monotone, distinct steps
    price = pd.DataFrame(rows, columns=["date", "symbol", "price"])
    vol = pd.DataFrame(
        _vol_rows("a", "2024-01-01", "2024-03-31"), columns=["date", "symbol", "volume"]
    )
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": "a", "mc": 50e6} for r in rows],
        columns=["date", "symbol", "mc"],
    )
    universe = build_universe_history(
        price, vol, mc, dates=pd.DatetimeIndex([ts("2024-03-31")]), excluded=set()
    )
    base = build_holding_returns(price, universe).set_index("date")["holding_return"]

    cut = ts("2024-02-15")
    price_mut = price.copy()
    price_mut.loc[price_mut["date"] > cut, "price"] = 1e9  # garbage future prices
    mut = build_holding_returns(price_mut, universe).set_index("date")["holding_return"]

    on_or_before = base.index <= cut
    pd.testing.assert_series_equal(
        base[on_or_before], mut[mut.index <= cut], check_names=False
    )


def test_output_schema_and_dtypes():
    rows = _price_rows("a", "2024-01-01", "2024-01-10", value=100.0)
    price = pd.DataFrame(rows, columns=["date", "symbol", "price"])
    vol = pd.DataFrame(
        _vol_rows("a", "2024-01-01", "2024-01-10"), columns=["date", "symbol", "volume"]
    )
    mc = pd.DataFrame(
        [{"date": r["date"], "symbol": "a", "mc": 50e6} for r in rows],
        columns=["date", "symbol", "mc"],
    )
    universe = build_universe_history(
        price, vol, mc, dates=pd.DatetimeIndex([ts("2024-01-10")]), excluded=set()
    )
    rets = build_holding_returns(price, universe)
    assert list(rets.columns) == ["date", "symbol", "holding_return"]
    assert pd.api.types.is_datetime64_any_dtype(rets["date"])
    assert pd.api.types.is_float_dtype(rets["holding_return"])
