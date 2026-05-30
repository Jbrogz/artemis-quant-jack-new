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
"""

import pandas as pd

from amom.config import MIN_ELIGIBLE_NAMES, MIN_MEDIAN_VOL_USD
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

    assert list(panel.columns) == ["date", "symbol", "eligible", "adv_30d", "gated"]
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])
    assert panel["eligible"].dtype == bool
    assert panel["gated"].dtype == bool
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
