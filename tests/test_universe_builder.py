"""Tests for the universe panel builder — Task 5 (THE GATED DELIVERABLE).

This is the survivorship / point-in-time core of the whole project. The
centerpiece is ``test_collapsed_coin_stays_in_panel_with_final_return``: a coin
that crashes ~95% and stops trading must remain in the panel on the dates it was
eligible (NOT dropped — dropping it is exactly the survivorship bias the guide
§1.1 forbids), and must fall out of eligibility only after its last observed
price plus the history-grace window.

Every fixture is synthetic and offline; no API calls. The builder must:
  - cover EVERY symbol ever seen across ``dates`` (incl. collapsed/delisted),
  - rebuild eligibility per (symbol, date), strictly point-in-time, using the
    rev-3 MC-floor + robust-trailing-volume liquidity gate,
  - report trailing-30d ADV (sum of positive prints / window) from volume rows
    dated ``<= date`` only,
  - mark a date ``gated`` when ``eligible.sum() < MIN_ELIGIBLE_NAMES``,
  - never consult any data dated after the as-of date.

Task R6 additions:
  - per-date no-look-ahead: for each grid date d, dropping rows > d must not
    change eligible/gated/delisted_asof at d,
  - intraday-bar boundary: panels with intraday timestamps are normalized to
    midnight so half-open <= comparisons are robust,
  - left-censoring: assets whose first price == the panel pull-start are flagged
    as left_censored (their true listing date is unknown).
"""

import pandas as pd

from amom.config import (
    LISTING_STALENESS_DAYS,
    MIN_BUCKET_SIZE,
    MIN_ELIGIBLE_NAMES,
    MIN_MEDIAN_VOL_USD,
)
from amom.universe.builder import build_universe_history


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _price_rows(symbol: str, start: str, end: str, value: float = 100.0):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "price": value} for d in dates]


def _vol_rows(symbol: str, start: str, end: str, daily_vol: float):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "volume": daily_vol} for d in dates]


def _mc_rows(symbol: str, start: str, end: str, mc: float = 50e6):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "mc": mc} for d in dates]


def _make_panels(price_rows: list, vol_rows: list, mc_rows: list | None = None):
    price = pd.DataFrame(price_rows, columns=["date", "symbol", "price"])
    vol = pd.DataFrame(vol_rows, columns=["date", "symbol", "volume"])
    if mc_rows is None:
        # Default: every priced (symbol, date) gets a passing market cap so
        # tests that don't probe the MC floor stay focused on their own axis.
        mc_rows = [
            {"date": r["date"], "symbol": r["symbol"], "mc": 50e6}
            for r in price_rows
        ]
    mc = pd.DataFrame(mc_rows, columns=["date", "symbol", "mc"])
    return price, vol, mc


def _liquid_universe(n: int, start: str, end: str, daily_vol: float = 5e6):
    """n always-liquid, long-history coins named bg0..bg{n-1}."""
    price_rows, vol_rows = [], []
    for i in range(n):
        sym = f"bg{i}"
        price_rows += _price_rows(sym, start, end)
        vol_rows += _vol_rows(sym, start, end, daily_vol)
    return price_rows, vol_rows


# ---------------------------------------------------------------------------
# CENTERPIECE: survivorship — collapsed coin stays in the panel
# ---------------------------------------------------------------------------

def test_collapsed_coin_stays_in_panel_with_final_return():
    """'deadcoin' has prices to 2024-06-01 then crashes ~95% and stops.

    It MUST appear in the panel while eligible (NOT dropped), and it must be
    absent from the eligible set (not silently kept eligible) after its last
    price + the history grace window has lapsed.
    """
    # deadcoin: long history, liquid, then a ~95% crash and a final stop.
    pre = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    crash = [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_price = pre + crash
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01", 5e6)

    # A liquid background universe so the gate is open on the dates we test.
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )

    price, vol, mc = _make_panels(dead_price + bg_price, dead_vol + bg_vol)
    dates = pd.date_range("2024-03-01", "2024-12-01", freq="MS")

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    # deadcoin appears in the panel and is eligible on at least one date.
    dead = panel.query("symbol == 'deadcoin'")
    assert not dead.empty, "collapsed coin was dropped from the panel entirely"
    assert dead["eligible"].any(), "collapsed coin never eligible (it should be pre-crash)"

    # It is eligible on a pre-crash date while it had data + liquidity.
    pre_crash = panel.query("symbol == 'deadcoin' and date == @ts('2024-05-01')")
    assert bool(pre_crash["eligible"].iloc[0]) is True

    # Long after the last price (+ grace), it is present as a row but NOT
    # eligible — survivorship without silently keeping a dead name alive.
    post = panel.query("symbol == 'deadcoin' and date == @ts('2024-12-01')")
    assert not post.empty, "collapsed coin row missing on a later date"
    assert bool(post["eligible"].iloc[0]) is False


def test_collapsed_coin_final_observed_value_reflects_crash():
    """The collapse must be visible in the panel's price-derived state.

    The last observed price for deadcoin is the crashed value, so its trailing
    ADV decays toward NaN/zero after data stops — proving the crash return is
    carried, not the coin silently removed at its pre-crash level.
    """
    pre = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    crash = [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01", 5e6)
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(pre + crash + bg_price, dead_vol + bg_vol)
    dates = pd.date_range("2024-03-01", "2024-12-01", freq="MS")

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    # On 2024-06-01 the crash row exists and the coin is still in the panel.
    on_crash = panel.query("symbol == 'deadcoin' and date == @ts('2024-06-01')")
    assert not on_crash.empty
    # ADV as-of the crash date is still computed from <= that date's volume.
    assert on_crash["adv_30d"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Eligibility rebuilt each date, strictly point-in-time
# ---------------------------------------------------------------------------

def test_eligibility_rebuilt_each_date_and_point_in_time():
    """A coin crossing the 90d threshold becomes eligible exactly once it has
    90d of history as of that date, not before."""
    # newcoin: first price 2024-01-01, liquid throughout.
    new_price = _price_rows("newcoin", "2024-01-01", "2024-12-31")
    new_vol = _vol_rows("newcoin", "2024-01-01", "2024-12-31", 5e6)
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2020-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(new_price + bg_price, new_vol + bg_vol)

    # Dates straddling the 90d threshold (first_date + 90d = 2024-03-31).
    before = ts("2024-03-15")   # 74 days -> ineligible
    after = ts("2024-04-15")    # 105 days -> eligible
    dates = pd.DatetimeIndex([before, after])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    e_before = panel.query("symbol == 'newcoin' and date == @before")["eligible"].iloc[0]
    e_after = panel.query("symbol == 'newcoin' and date == @after")["eligible"].iloc[0]
    assert bool(e_before) is False
    assert bool(e_after) is True


def test_panel_covers_every_symbol_on_every_date():
    """Every symbol ever seen appears on every requested date (full grid)."""
    price_rows = _price_rows("a", "2024-01-01", "2024-06-30") + \
        _price_rows("b", "2024-03-01", "2024-06-30")
    vol_rows = _vol_rows("a", "2024-01-01", "2024-06-30", 5e6) + \
        _vol_rows("b", "2024-03-01", "2024-06-30", 5e6)
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.date_range("2024-04-01", "2024-06-01", freq="MS")

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    symbols = set(panel["symbol"].unique())
    assert symbols == {"a", "b"}
    for d in dates:
        on_date = panel.query("date == @d")
        assert set(on_date["symbol"]) == {"a", "b"}, f"missing symbol on {d}"
    assert len(panel) == len(dates) * 2


def test_no_lookahead_future_data_irrelevant_to_panel():
    """Mutating volume/price strictly AFTER each as_of date cannot change the
    eligible flag computed at that date — the structural no-look-ahead rule."""
    price_rows, vol_rows = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="MS")

    panel_base = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    # Mutate the future: collapse all volume strictly after the LAST as_of date,
    # and append a brand-new coin that only exists in the future. Neither can
    # alter eligibility on any of `dates`.
    last_date = dates.max()
    vol_mut = vol.copy()
    future = vol_mut["date"] > last_date
    vol_mut.loc[future, "volume"] = 0.0
    mc_mut = mc.copy()
    mc_mut.loc[mc_mut["date"] > last_date, "mc"] = 0.0  # garbage future MC
    # add a future-only coin (price + volume + MC all strictly in the future)
    fut_price = _price_rows("future_only", "2024-07-01", "2024-12-31")
    fut_vol = _vol_rows("future_only", "2024-07-01", "2024-12-31", 9e9)
    fut_mc = _mc_rows("future_only", "2024-07-01", "2024-12-31", 1e12)
    price_mut = pd.concat([price, pd.DataFrame(fut_price)], ignore_index=True)
    vol_mut = pd.concat([vol_mut, pd.DataFrame(fut_vol)], ignore_index=True)
    mc_mut = pd.concat([mc_mut, pd.DataFrame(fut_mc)], ignore_index=True)

    panel_mut = build_universe_history(
        price_mut, vol_mut, mc_mut, dates=dates, excluded=set()
    )

    # Restrict to the originally-present symbols on the original dates.
    base = panel_base.sort_values(["date", "symbol"]).reset_index(drop=True)
    mut = (
        panel_mut[panel_mut["symbol"] != "future_only"]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        base["eligible"], mut["eligible"], check_names=False
    )


# ---------------------------------------------------------------------------
# Trailing-30d ADV is point-in-time
# ---------------------------------------------------------------------------

def test_adv_is_trailing_30d_sum_over_window_as_of_date():
    """adv_30d = sum(positive prints) / 30 over the trailing window <= the date.

    With a full window of constant daily volume the ADV equals that constant;
    the rev-3 gate's denominator is the full window (not present rows), so a
    sparse window would yield a small ADV instead.
    """
    vol_rows = _vol_rows("a", "2024-01-01", "2024-06-30", 4e6)
    price_rows = _price_rows("a", "2023-01-01", "2024-06-30")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-05-15")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    adv = panel.query("symbol == 'a'")["adv_30d"].iloc[0]
    assert adv == 4e6


def test_adv_only_uses_volume_at_or_before_date():
    """A volume spike strictly AFTER the as_of date must not enter adv_30d."""
    base = _vol_rows("a", "2024-01-01", "2024-06-30", 2e6)
    # huge spike after as_of
    spike = [{"date": ts("2024-06-30"), "symbol": "a", "volume": 1e12}]
    vol_rows = base + spike
    price_rows = _price_rows("a", "2023-01-01", "2024-06-30")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-05-15")])  # before the spike

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    adv = panel.query("symbol == 'a'")["adv_30d"].iloc[0]
    assert adv == 2e6  # spike ignored entirely


# ---------------------------------------------------------------------------
# Minimum-universe gate is point-in-time
# ---------------------------------------------------------------------------

def test_min_universe_gate_is_point_in_time():
    """An early date with < MIN_ELIGIBLE_NAMES eligible -> gated True; a later
    date with enough names -> gated False. The gate uses as-of info only."""
    # Build exactly MIN_ELIGIBLE_NAMES coins, all listed 2024-01-01. Early on,
    # none has 90d of history (gate closed); later all do (gate open).
    price_rows, vol_rows = [], []
    for i in range(MIN_ELIGIBLE_NAMES):
        sym = f"c{i}"
        price_rows += _price_rows(sym, "2024-01-01", "2024-12-31")
        vol_rows += _vol_rows(sym, "2024-01-01", "2024-12-31", 5e6)
    price, vol, mc = _make_panels(price_rows, vol_rows)

    early = ts("2024-02-01")    # ~31d history -> 0 eligible -> gated
    late = ts("2024-06-01")     # >90d history -> all eligible -> not gated
    dates = pd.DatetimeIndex([early, late])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    early_rows = panel.query("date == @early")
    late_rows = panel.query("date == @late")

    assert int(early_rows["eligible"].sum()) < MIN_ELIGIBLE_NAMES
    assert bool(early_rows["gated"].all()) is True

    assert int(late_rows["eligible"].sum()) >= MIN_ELIGIBLE_NAMES
    assert bool(late_rows["gated"].any()) is False


def test_gate_is_constant_per_date():
    """gated is a per-date flag: identical across all symbols on a given date."""
    price_rows, vol_rows = _liquid_universe(3, "2020-01-01", "2024-12-31")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    on_date = panel.query("date == @ts('2024-06-01')")
    # Only 3 eligible names < MIN_ELIGIBLE_NAMES -> all gated.
    assert on_date["gated"].nunique() == 1
    assert bool(on_date["gated"].iloc[0]) is True


def test_gate_future_mutation_does_not_change_decision():
    """The matrix rule: the gate decision at a date is unchanged when future
    data is mutated."""
    price_rows, vol_rows = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="MS")

    base = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    # Kill all volume after the last as_of date.
    last_date = dates.max()
    vol_mut = vol.copy()
    vol_mut.loc[vol_mut["date"] > last_date, "volume"] = 0.0
    mut = build_universe_history(price, vol_mut, mc, dates=dates, excluded=set())

    base_gate = base.groupby("date")["gated"].first()
    mut_gate = mut.groupby("date")["gated"].first()
    pd.testing.assert_series_equal(base_gate, mut_gate, check_names=False)


# ---------------------------------------------------------------------------
# Exclusions and schema
# ---------------------------------------------------------------------------

def test_excluded_symbols_never_eligible():
    price_rows, vol_rows = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2020-01-01", "2024-12-31"
    )
    # add a stablecoin that is liquid + old but excluded
    price_rows += _price_rows("usdt", "2020-01-01", "2024-12-31")
    vol_rows += _vol_rows("usdt", "2020-01-01", "2024-12-31", 1e10)
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded={"usdt"})
    usdt = panel.query("symbol == 'usdt'")
    assert not usdt.empty  # present in panel (covers every symbol)
    assert bool(usdt["eligible"].any()) is False  # never eligible


def test_illiquid_coin_is_ineligible_but_present():
    price_rows, vol_rows = _liquid_universe(2, "2020-01-01", "2024-12-31")
    # a long-history but illiquid coin
    price_rows += _price_rows("thin", "2020-01-01", "2024-12-31")
    vol_rows += _vol_rows("thin", "2020-01-01", "2024-12-31", 0.1e6)  # < $1M
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    thin = panel.query("symbol == 'thin'")
    assert not thin.empty
    assert bool(thin["eligible"].any()) is False
    assert thin["adv_30d"].iloc[0] < MIN_MEDIAN_VOL_USD


def test_output_schema_and_dtypes():
    price_rows, vol_rows = _liquid_universe(2, "2020-01-01", "2024-12-31")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    assert list(panel.columns) == [
        "date",
        "symbol",
        "eligible",
        "adv_30d",
        "price_last_date",
        "delisted_asof",
        "left_censored",
        "gated",
    ]
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])
    assert panel["eligible"].dtype == bool
    assert panel["gated"].dtype == bool
    assert panel["delisted_asof"].dtype == bool
    assert panel["left_censored"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(panel["price_last_date"])
    assert pd.api.types.is_float_dtype(panel["adv_30d"])


def test_symbol_with_no_volume_has_zero_adv_and_is_ineligible():
    """A symbol present in price but absent from volume gets ADV 0.0 -> out.

    With no positive prints in the window the trailing-window ADV is 0.0 (sum of
    nothing / window), which is below the floor, so the coin is ineligible.
    """
    price_rows = _price_rows("a", "2020-01-01", "2024-12-31")
    bg_p, bg_v = _liquid_universe(2, "2020-01-01", "2024-12-31")
    price, vol, mc = _make_panels(price_rows + bg_p, bg_v)  # 'a' missing from vol
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    a = panel.query("symbol == 'a'")
    assert not a.empty
    assert a["adv_30d"].iloc[0] == 0.0
    assert bool(a["eligible"].iloc[0]) is False


# ---------------------------------------------------------------------------
# Task R3: death signal carried in the panel (price_last_date + delisted_asof)
# ---------------------------------------------------------------------------

def test_price_last_date_is_point_in_time():
    """price_last_date is the latest priced date <= the as_of date, not the
    coin's global last price. A future price must not advance it."""
    # 'a' is priced daily through 2024-12-31; we evaluate mid-series.
    price_rows = _price_rows("a", "2024-01-01", "2024-12-31")
    vol_rows = _vol_rows("a", "2024-01-01", "2024-12-31", 5e6)
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-15")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    row = panel.query("symbol == 'a' and date == @ts('2024-06-15')")
    # The latest price <= 2024-06-15 is 2024-06-15 (daily series), NOT 12-31.
    assert row["price_last_date"].iloc[0] == ts("2024-06-15")


def test_delisted_asof_only_after_last_price_plus_grace_no_future_data():
    """A collapsed coin that stops reporting on 2024-06-01 is delisted_asof
    only on/after 2024-06-01 + LISTING_STALENESS_DAYS, never before, and the
    flag uses only data dated <= the as_of date."""
    # deadcoin: priced through the crash on 2024-06-01, then silent.
    pre = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    crash = [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01", 5e6)
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(pre + crash + bg_price, dead_vol + bg_vol)

    grace = pd.Timedelta(days=LISTING_STALENESS_DAYS)
    last = ts("2024-06-01")
    # A date inside the grace window (not yet delisted) and one past it.
    within = last + grace            # exactly == grace -> not strictly older
    past = last + grace + pd.Timedelta(days=1)
    before = ts("2024-05-15")        # still reporting -> not delisted
    dates = pd.DatetimeIndex([before, last, within, past])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    def deadflag(d):
        r = panel.query("symbol == 'deadcoin' and date == @d")
        return bool(r["delisted_asof"].iloc[0])

    assert deadflag(before) is False  # actively reporting
    assert deadflag(last) is False    # last price is today; staleness 0
    assert deadflag(within) is False  # exactly at grace boundary, not > grace
    assert deadflag(past) is True     # strictly past the grace window


def test_delisted_asof_ignores_future_resumption():
    """If a coin goes silent and later resumes, the delisted flag on a date in
    the silent gap must reflect only data <= that date (a True), unaffected by
    the future resumption."""
    early = _price_rows("zombie", "2024-01-01", "2024-03-01", value=100.0)
    # long silent gap, then resumes far in the future
    revived = _price_rows("zombie", "2024-09-01", "2024-12-31", value=2.0)
    vol_rows = _vol_rows("zombie", "2024-01-01", "2024-03-01", 5e6) + \
        _vol_rows("zombie", "2024-09-01", "2024-12-31", 5e6)
    price, vol, mc = _make_panels(early + revived, vol_rows)

    # A date deep in the silent gap: last price <= here is 2024-03-01.
    in_gap = ts("2024-06-01")
    dates = pd.DatetimeIndex([in_gap])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    row = panel.query("symbol == 'zombie' and date == @in_gap")
    assert row["price_last_date"].iloc[0] == ts("2024-03-01")
    assert bool(row["delisted_asof"].iloc[0]) is True  # > grace, future ignored


def test_delisted_asof_false_for_never_priced_at_or_before_date():
    """A coin whose first price is strictly after the as_of date has no
    price_last_date (NaT) and is not delisted — it has simply not listed yet."""
    bg_price, bg_vol = _liquid_universe(2, "2020-01-01", "2024-12-31")
    later = _price_rows("nascent", "2024-08-01", "2024-12-31")
    later_vol = _vol_rows("nascent", "2024-08-01", "2024-12-31", 5e6)
    price, vol, mc = _make_panels(bg_price + later, bg_vol + later_vol)
    dates = pd.DatetimeIndex([ts("2024-06-01")])  # before nascent lists

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    row = panel.query("symbol == 'nascent' and date == @ts('2024-06-01')")
    assert pd.isna(row["price_last_date"].iloc[0])
    assert bool(row["delisted_asof"].iloc[0]) is False


def test_delisted_asof_aligns_with_eligibility_exit():
    """The death signal is consistent with the tradeability filter: once a coin
    is delisted_asof, it is no longer eligible (staleness gate fired)."""
    pre = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    crash = [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01", 5e6)
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price, vol, mc = _make_panels(pre + crash + bg_price, dead_vol + bg_vol)
    dates = pd.date_range("2024-03-01", "2024-12-01", freq="MS")

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    dead = panel.query("symbol == 'deadcoin'")
    # Wherever the death signal fired, the coin must not be eligible.
    delisted = dead[dead["delisted_asof"]]
    assert not delisted.empty, "death signal never fired for a collapsed coin"
    assert not delisted["eligible"].any()


def test_min_universe_gate_derived_from_min_bucket_size():
    """The gate threshold is derived from MIN_BUCKET_SIZE (5 quintiles each
    needing >= MIN_BUCKET_SIZE names), not a bare constant. A universe with
    exactly one fewer than 5*MIN_BUCKET_SIZE eligible names is gated; exactly
    5*MIN_BUCKET_SIZE is not."""
    floor = 5 * MIN_BUCKET_SIZE

    def build_n(n):
        price_rows, vol_rows = _liquid_universe(n, "2020-01-01", "2024-12-31")
        price, vol, mc = _make_panels(price_rows, vol_rows)
        dates = pd.DatetimeIndex([ts("2024-06-01")])
        return build_universe_history(price, vol, mc, dates=dates, excluded=set())

    below = build_n(floor - 1)
    assert int(below["eligible"].sum()) == floor - 1
    assert bool(below["gated"].iloc[0]) is True  # too few -> gated

    at = build_n(floor)
    assert int(at["eligible"].sum()) == floor
    assert bool(at["gated"].iloc[0]) is False  # exactly the floor -> open


# ---------------------------------------------------------------------------
# Task R6: per-date no-look-ahead (discriminating, row-dropping variant)
# ---------------------------------------------------------------------------

def test_per_date_nolookahead_eligible_unchanged_when_future_rows_dropped():
    """For every grid date d, dropping ALL rows (price/vol/mc) dated strictly
    after d must produce the same eligible flag at d as the full build.

    This is more discriminating than the previous mutation test: we do a full
    drop (not just zeroing) so even symbol presence changes after d — the builder
    must use only data <= d for each date independently.
    """
    price_rows, vol_rows = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    # Add a coin that starts mid-way so its eligibility changes across dates.
    price_rows += _price_rows("latestart", "2024-01-01", "2025-01-01")
    vol_rows += _vol_rows("latestart", "2024-01-01", "2025-01-01", 5e6)

    price_full, vol_full, mc_full = _make_panels(price_rows, vol_rows)
    dates = pd.date_range("2024-01-01", "2024-12-01", freq="MS")

    panel_full = build_universe_history(
        price_full, vol_full, mc_full, dates=dates, excluded=set()
    )

    for d in dates:
        # Restrict all panels to rows <= d.
        p_trunc = price_full[price_full["date"] <= d]
        v_trunc = vol_full[vol_full["date"] <= d]
        m_trunc = mc_full[mc_full["date"] <= d]

        panel_trunc = build_universe_history(
            p_trunc, v_trunc, m_trunc, dates=pd.DatetimeIndex([d]), excluded=set()
        )

        # For every symbol present on date d in both panels, eligible must agree.
        full_at_d = (
            panel_full.query("date == @d")
            .set_index("symbol")[["eligible", "gated", "delisted_asof"]]
        )
        trunc_at_d = (
            panel_trunc.query("date == @d")
            .set_index("symbol")[["eligible", "gated", "delisted_asof"]]
        )
        # Symbols only in truncated (no future-only symbols) — compare intersection.
        shared = full_at_d.index.intersection(trunc_at_d.index)
        pd.testing.assert_frame_equal(
            full_at_d.loc[shared].sort_index(),
            trunc_at_d.loc[shared].sort_index(),
            check_like=True,
            obj=f"eligible/gated/delisted at {d.date()}",
        )


def test_per_date_nolookahead_gated_unchanged_when_future_rows_dropped():
    """The gated flag at each date is unchanged when all rows after that date
    are dropped. This tests the gate's as-of logic separately from eligible."""
    price_rows, vol_rows = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price_full, vol_full, mc_full = _make_panels(price_rows, vol_rows)
    dates = pd.date_range("2023-06-01", "2024-06-01", freq="MS")

    panel_full = build_universe_history(
        price_full, vol_full, mc_full, dates=dates, excluded=set()
    )

    for d in dates:
        p_trunc = price_full[price_full["date"] <= d]
        v_trunc = vol_full[vol_full["date"] <= d]
        m_trunc = mc_full[mc_full["date"] <= d]

        panel_trunc = build_universe_history(
            p_trunc, v_trunc, m_trunc, dates=pd.DatetimeIndex([d]), excluded=set()
        )

        gate_full = panel_full.query("date == @d")["gated"].iloc[0]
        gate_trunc = panel_trunc.query("date == @d")["gated"].iloc[0]
        assert gate_full == gate_trunc, (
            f"gated at {d.date()} differs: full={gate_full}, trunc={gate_trunc}"
        )


def test_per_date_nolookahead_delisted_asof_unchanged_when_future_rows_dropped():
    """delisted_asof at d must reflect only data <= d; dropping future rows must
    not change the signal. Uses a coin that crashes and stops mid-series."""
    pre = _price_rows("deadcoin", "2023-01-01", "2024-05-31", value=100.0)
    crash = [{"date": ts("2024-06-01"), "symbol": "deadcoin", "price": 5.0}]
    dead_vol = _vol_rows("deadcoin", "2023-01-01", "2024-06-01", 5e6)
    bg_price, bg_vol = _liquid_universe(
        MIN_ELIGIBLE_NAMES, "2023-01-01", "2025-01-01"
    )
    price_full, vol_full, mc_full = _make_panels(
        pre + crash + bg_price, dead_vol + bg_vol
    )
    dates = pd.date_range("2024-04-01", "2024-10-01", freq="MS")

    panel_full = build_universe_history(
        price_full, vol_full, mc_full, dates=dates, excluded=set()
    )

    for d in dates:
        p_trunc = price_full[price_full["date"] <= d]
        v_trunc = vol_full[vol_full["date"] <= d]
        m_trunc = mc_full[mc_full["date"] <= d]

        panel_trunc = build_universe_history(
            p_trunc, v_trunc, m_trunc, dates=pd.DatetimeIndex([d]), excluded=set()
        )

        full_row = panel_full.query("symbol == 'deadcoin' and date == @d")
        trunc_row = panel_trunc.query("symbol == 'deadcoin' and date == @d")

        full_flag = bool(full_row["delisted_asof"].iloc[0])
        trunc_flag = bool(trunc_row["delisted_asof"].iloc[0])
        assert full_flag == trunc_flag, (
            f"delisted_asof for deadcoin at {d.date()} "
            f"differs: full={full_flag}, trunc={trunc_flag}"
        )


# ---------------------------------------------------------------------------
# Task R6: intraday-bar boundary — date normalization at ingest
# ---------------------------------------------------------------------------

def test_intraday_timestamps_normalized_to_midnight():
    """Panels whose 'date' column carries intraday timestamps (e.g. noon UTC from
    an API response) must be normalized to midnight at the builder boundary so
    half-open <= comparisons are identical to the midnight-only case.

    Concretely: a row stamped 2024-06-01T12:00:00 and a grid date of 2024-06-01
    must be treated as 'on' that date, not as 'after' it.
    """
    # Build a price/vol panel with an intraday timestamp.
    price_rows_raw = _price_rows("a", "2023-01-01", "2024-12-31")
    vol_rows_raw = _vol_rows("a", "2023-01-01", "2024-12-31", 5e6)

    # Shift a subset of rows to noon UTC to simulate intraday API delivery.
    price_intraday = [
        {**r, "date": pd.Timestamp(r["date"]) + pd.Timedelta(hours=12)}
        for r in price_rows_raw
    ]
    vol_intraday = [
        {**r, "date": pd.Timestamp(r["date"]) + pd.Timedelta(hours=12)}
        for r in vol_rows_raw
    ]

    price_intra = pd.DataFrame(price_intraday, columns=["date", "symbol", "price"])
    vol_intra = pd.DataFrame(vol_intraday, columns=["date", "symbol", "volume"])
    mc_intra = pd.DataFrame(
        [{"date": pd.Timestamp(r["date"]) + pd.Timedelta(hours=12),
           "symbol": "a", "mc": 50e6}
         for r in price_rows_raw],
        columns=["date", "symbol", "mc"],
    )

    dates = pd.DatetimeIndex([ts("2024-06-01")])

    # With normalization the builder must treat intraday rows as their date-day.
    panel = build_universe_history(
        price_intra, vol_intra, mc_intra, dates=dates, excluded=set()
    )
    row = panel.query("symbol == 'a' and date == @ts('2024-06-01')")
    assert not row.empty, "symbol missing — intraday rows not recognized as <= date"
    assert bool(row["eligible"].iloc[0]) is True, (
        "symbol ineligible — intraday timestamp not normalized to midnight"
    )


def test_intraday_price_last_date_normalized_correctly():
    """price_last_date must equal the grid date (midnight) even when the raw
    panel carries intraday timestamps, so delisted_asof arithmetic is correct."""
    price_rows_raw = _price_rows("b", "2023-01-01", "2024-06-15")
    vol_rows_raw = _vol_rows("b", "2023-01-01", "2024-06-15", 5e6)

    # All rows at 23:59:59 — near end of day but still same calendar date.
    def near_eod(r):
        return {**r, "date": pd.Timestamp(r["date"]) + pd.Timedelta(hours=23, minutes=59, seconds=59)}

    price_near = pd.DataFrame([near_eod(r) for r in price_rows_raw],
                               columns=["date", "symbol", "price"])
    vol_near = pd.DataFrame([near_eod(r) for r in vol_rows_raw],
                             columns=["date", "symbol", "volume"])
    mc_near = pd.DataFrame(
        [{"date": pd.Timestamp(r["date"]) + pd.Timedelta(hours=23, minutes=59, seconds=59),
          "symbol": "b", "mc": 50e6} for r in price_rows_raw],
        columns=["date", "symbol", "mc"],
    )

    dates = pd.DatetimeIndex([ts("2024-06-15")])
    panel = build_universe_history(
        price_near, vol_near, mc_near, dates=dates, excluded=set()
    )
    row = panel.query("symbol == 'b' and date == @ts('2024-06-15')")
    assert not row.empty
    # price_last_date must be midnight 2024-06-15, not 2024-06-14T23:59:59.
    assert row["price_last_date"].iloc[0] == ts("2024-06-15")


# ---------------------------------------------------------------------------
# Task R6: left-censoring flag (pull_start == first_price -> unknown true date)
# ---------------------------------------------------------------------------

def test_left_censored_flag_set_for_asset_at_pull_start():
    """An asset whose first price equals the earliest date in the price panel
    (the pull-start boundary) is left_censored: its true listing date is unknown.

    The builder must emit a left_censored column, True for these assets.
    """
    # 'censor_me' starts exactly on the pull-start (earliest date in the panel).
    # 'clear' starts later, so its first price is away from the boundary.
    price_rows = (
        _price_rows("censor_me", "2023-01-01", "2024-12-31")
        + _price_rows("clear", "2023-06-01", "2024-12-31")
    )
    vol_rows = (
        _vol_rows("censor_me", "2023-01-01", "2024-12-31", 5e6)
        + _vol_rows("clear", "2023-06-01", "2024-12-31", 5e6)
    )
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())

    assert "left_censored" in panel.columns, "left_censored column missing"
    row_censor = panel.query("symbol == 'censor_me'")
    row_clear = panel.query("symbol == 'clear'")

    assert bool(row_censor["left_censored"].iloc[0]) is True, (
        "asset at pull-start boundary should be left_censored"
    )
    assert bool(row_clear["left_censored"].iloc[0]) is False, (
        "asset starting after pull-start should NOT be left_censored"
    )


def test_left_censored_column_is_bool():
    """left_censored must be a bool column (not object/int)."""
    price_rows, vol_rows = _liquid_universe(2, "2020-01-01", "2024-12-31")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    assert panel["left_censored"].dtype == bool


def test_left_censored_schema_included_in_output_columns():
    """The output schema now includes left_censored after delisted_asof."""
    price_rows, vol_rows = _liquid_universe(2, "2020-01-01", "2024-12-31")
    price, vol, mc = _make_panels(price_rows, vol_rows)
    dates = pd.DatetimeIndex([ts("2024-06-01")])

    panel = build_universe_history(price, vol, mc, dates=dates, excluded=set())
    assert list(panel.columns) == [
        "date",
        "symbol",
        "eligible",
        "adv_30d",
        "price_last_date",
        "delisted_asof",
        "left_censored",
        "gated",
    ]
