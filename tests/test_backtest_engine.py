"""Tests for the cost-aware backtest engine (Task B2, spec §4.1/§4.3/§4.4/§3.5).

``run_backtest(weights_by_rebal, holding_returns, universe_panel, *, aum,
cost_model, vol_target)`` walks the per-rebalance dollar-neutral coin weights
through time: it caps each coin at ``PER_COIN_CAP`` and the book gross at
``GROSS_LEVERAGE_CAP = 2.0``, **walk-forward vol-targets** the book to
``vol_target`` using only trailing (<= t) realized vol, executes the rebalance at
the **t+1 close** charging ``cost_model`` on the traded notional (target -
current), and carries the book forward with realized price P&L. It returns an
equity curve, a position history, and a trade log.

Discriminating behaviours (plan §B2):
  * dollar-neutrality is preserved each rebalance (Σ signed weight ~= 0);
  * **gross <= 2x** asserted EVERY period (the cap binds, never exceeded);
  * the vol scalar uses ONLY trailing (<= t) data — mutating *future* returns
    does not change today's scalar (no look-ahead, the cardinal rule);
  * costs reduce the NET equity curve below the gross (no-cost) one;
  * a collapsed short-leg coin's crash flows into the book P&L (a short of a
    coin that craters earns a positive return; survivorship still books).

All fixtures are synthetic and offline; no API calls, no disk reads.
"""

import math

import pandas as pd

from amom.backtest.costs import trade_cost
from amom.backtest.engine import run_backtest
from amom.config import ANNUAL_VOL_TARGET, GROSS_LEVERAGE_CAP, PER_COIN_CAP


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# A frictionless cost model (always zero) isolates the P&L / vol-target / cap
# behaviours from the cost behaviour; the real ``trade_cost`` is used where the
# net-vs-gross test needs friction.
def _zero_cost(*args, **kwargs) -> float:
    return 0.0


def _long_weights(rows: list[dict]) -> pd.DataFrame:
    """A long-form weights-by-rebalance frame [rebalance_date, symbol, weight]."""
    return pd.DataFrame(rows, columns=["rebalance_date", "symbol", "weight"])


def _holding_returns(rows: list[dict]) -> pd.DataFrame:
    """Long-form holding-return panel [date, symbol, holding_return]."""
    return pd.DataFrame(rows, columns=["date", "symbol", "holding_return"])


def _universe(dates, symbols, *, adv: float = 1e9) -> pd.DataFrame:
    """A long-form universe panel with a dense ADV for every (date, symbol)."""
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"date": d, "symbol": s, "adv_30d": adv, "eligible": True})
    return pd.DataFrame(rows)


# A two-coin, dollar-neutral, two-rebalance synthetic book. Long A, short B at
# each rebalance; each holding window is 2 calendar days of returns.
_REB = [ts("2024-02-01"), ts("2024-03-01"), ts("2024-04-01")]
_SYMS = ["A", "B"]


def _simple_weights() -> pd.DataFrame:
    rows = []
    for r in _REB[:-1]:  # last rebalance has no forward window to price
        rows += [
            {"rebalance_date": r, "symbol": "A", "weight": 0.5},
            {"rebalance_date": r, "symbol": "B", "weight": -0.5},
        ]
    return _long_weights(rows)


def _simple_returns(b_returns: dict[str, list[float]] | None = None) -> pd.DataFrame:
    """Holding returns on a daily grid covering the two holding windows.

    A drifts up modestly; B's path is overridable so a test can crater the short.
    """
    rows = []
    # Window 1: (2024-02-01, 2024-03-01]; Window 2: (2024-03-01, 2024-04-01].
    a_path = {
        ts("2024-02-15"): 0.10,
        ts("2024-03-01"): 0.05,
        ts("2024-03-15"): 0.02,
        ts("2024-04-01"): 0.03,
    }
    b_path = {
        ts("2024-02-15"): 0.04,
        ts("2024-03-01"): 0.06,
        ts("2024-03-15"): 0.01,
        ts("2024-04-01"): 0.02,
    }
    if b_returns is not None:
        b_path = {ts(k): v for k, v in b_returns.items()}
    for d, v in a_path.items():
        rows.append({"date": d, "symbol": "A", "holding_return": v})
    for d, v in b_path.items():
        rows.append({"date": d, "symbol": "B", "holding_return": v})
    return _holding_returns(rows)


# A longer, multi-rebalance book (8 rebalances -> 7 priced windows) on a wide
# 10-name cross-section (5 longs, 5 shorts at +/-0.1 each: raw gross = 1.0,
# per-coin |weight| = 0.1 < PER_COIN_CAP). The walk-forward vol scalar actually
# activates (by the 3rd rebalance there are >= 2 trailing book returns) and on a
# *small but non-zero* realized vol it drives the scalar > 1, levering gross
# toward the cap — so GROSS_LEVERAGE_CAP becomes the binding constraint rather
# than a trivial scalar of 1.0.
_REB_LONG = list(pd.date_range("2024-01-01", periods=8, freq="MS"))
_LONGS = [f"L{i}" for i in range(5)]
_SHORTS = [f"S{i}" for i in range(5)]
_SYMS_LONG = _LONGS + _SHORTS


def _long_book_weights() -> pd.DataFrame:
    rows = []
    for r in _REB_LONG[:-1]:
        for s in _LONGS:
            rows.append({"rebalance_date": r, "symbol": s, "weight": 0.1})
        for s in _SHORTS:
            rows.append({"rebalance_date": r, "symbol": s, "weight": -0.1})
    return _long_weights(rows)


def _long_book_returns() -> pd.DataFrame:
    """Small, *varying* per-window returns so realized book vol is positive but
    tiny -> the vol scalar wants to lever the book hard (testing the caps)."""
    rows = []
    # A small alternating wiggle on the long leg gives a non-constant (positive
    # but small) book-return series, so std(ddof=1) > 0 and the scalar > 1.
    wiggle = [0.004, 0.006, 0.005, 0.007, 0.004, 0.006, 0.005]
    for i in range(len(_REB_LONG) - 1):
        end = _REB_LONG[i + 1]
        for s in _LONGS:
            rows.append({"date": end, "symbol": s, "holding_return": wiggle[i]})
        for s in _SHORTS:
            rows.append({"date": end, "symbol": s, "holding_return": 0.001})
    return _holding_returns(rows)


def _run(weights, returns, universe, *, cost_model=_zero_cost, vol_target=None):
    return run_backtest(
        weights,
        returns,
        universe,
        aum=1_000_000.0,
        cost_model=cost_model,
        vol_target=ANNUAL_VOL_TARGET if vol_target is None else vol_target,
    )


# ---------------------------------------------------------------------------
# the engine returns the three artifacts (equity curve, positions, trades)
# ---------------------------------------------------------------------------

def test_run_backtest_returns_three_artifacts():
    res = _run(_simple_weights(), _simple_returns(), _universe(_REB, _SYMS))
    assert {"equity", "positions", "trades"}.issubset(set(res.keys()))
    assert isinstance(res["equity"], pd.DataFrame)
    assert isinstance(res["positions"], pd.DataFrame)
    assert isinstance(res["trades"], pd.DataFrame)
    assert len(res["equity"]) >= 1
    assert "missing_return_count" in res["equity"].columns


# ---------------------------------------------------------------------------
# dollar-neutrality each rebalance: Σ signed weight ~= 0
# ---------------------------------------------------------------------------

def test_positions_are_dollar_neutral_each_rebalance():
    res = _run(_simple_weights(), _simple_returns(), _universe(_REB, _SYMS))
    pos = res["positions"]
    for _, grp in pos.groupby("rebalance_date"):
        assert abs(grp["weight"].sum()) < 1e-9


# ---------------------------------------------------------------------------
# gross <= 2x asserted EVERY period
# ---------------------------------------------------------------------------

def test_gross_never_exceeds_cap_every_period():
    # An absurd vol target on a low-vol book makes the walk-forward scalar want to
    # lever far past the cap once trailing history exists; GROSS_LEVERAGE_CAP must
    # still bind EVERY period (early un-targeted periods included).
    res = _run(
        _long_book_weights(),
        _long_book_returns(),
        _universe(_REB_LONG, _SYMS_LONG),
        vol_target=100.0,  # absurd target -> scalar wants huge leverage
    )
    pos = res["positions"]
    assert len(pos) > 0
    for _, grp in pos.groupby("rebalance_date"):
        gross = grp["weight"].abs().sum()
        assert gross <= GROSS_LEVERAGE_CAP + 1e-9
    # And the cap genuinely BINDS on at least one (late) period — otherwise the
    # assertion above is vacuous (a scalar stuck at 1.0 never tests the cap).
    max_gross = pos.groupby("rebalance_date")["weight"].apply(
        lambda g: g.abs().sum()
    ).max()
    assert max_gross > GROSS_LEVERAGE_CAP - 1e-9


def test_per_coin_cap_binds():
    # On the levered low-vol book each coin would carry |weight| = GROSS/2 = 1.0
    # absent the per-coin cap; PER_COIN_CAP must clip every coin every period.
    res = _run(
        _long_book_weights(),
        _long_book_returns(),
        _universe(_REB_LONG, _SYMS_LONG),
        vol_target=100.0,
    )
    pos = res["positions"]
    assert (pos["weight"].abs() <= PER_COIN_CAP + 1e-9).all()
    # The cap binds: at least one coin sits exactly at PER_COIN_CAP late in the run.
    assert (pos["weight"].abs() >= PER_COIN_CAP - 1e-9).any()


# ---------------------------------------------------------------------------
# no look-ahead: the vol scalar uses ONLY trailing (<= t) data
# ---------------------------------------------------------------------------

def _mutate_long_book_returns(window_overrides: dict[int, float]) -> pd.DataFrame:
    """Long-book returns with the LONG leg of chosen window indices overridden.

    ``window_overrides`` maps a priced-window index ``i`` (the window
    ``(_REB_LONG[i], _REB_LONG[i+1]]``, closing at ``_REB_LONG[i+1]``) to a new
    long-leg return; every other window keeps the baseline wiggle. Shorts stay
    flat. This lets a test perturb exactly one window — past or future — relative
    to a target rebalance and observe whether that rebalance's vol scalar moved.
    """
    base_wiggle = [0.004, 0.006, 0.005, 0.007, 0.004, 0.006, 0.005]
    rows = []
    for i in range(len(_REB_LONG) - 1):
        end = _REB_LONG[i + 1]
        long_ret = window_overrides.get(i, base_wiggle[i])
        for s in _LONGS:
            rows.append({"date": end, "symbol": s, "holding_return": long_ret})
        for s in _SHORTS:
            rows.append({"date": end, "symbol": s, "holding_return": 0.001})
    return _holding_returns(rows)


def test_vol_scalar_uses_only_trailing_data():
    # The 3rd rebalance (index 2, at _REB_LONG[2]) is the FIRST rebalance whose
    # walk-forward vol scalar is genuinely ACTIVE: it consumes the two closed
    # windows before it (book_returns[0] from window 0 closing at _REB_LONG[1],
    # book_returns[1] from window 1 closing at _REB_LONG[2]), so the scalar there
    # is != 1.0 and levers the book toward the cap. Asserting at this rebalance —
    # not the hardcoded-1.0 first one — makes the no-look-ahead claim non-vacuous.
    weights = _long_book_weights()
    universe = _universe(_REB_LONG, _SYMS_LONG)
    target_r = _REB_LONG[2]

    base = _run(weights, _long_book_returns(), universe, vol_target=0.20)

    # The scalar at the target rebalance must genuinely differ from 1.0 (gross
    # != raw), else any equality assertion below is trivially satisfied.
    base_pos = base["positions"][base["positions"]["rebalance_date"] == target_r]
    base_gross = float(base_pos["weight"].abs().sum())
    assert abs(base_gross - 1.0) > 1e-6, "vol scalar is inactive — test is vacuous"

    def weights_at(res):
        sel = res["positions"][res["positions"]["rebalance_date"] == target_r]
        return sel.set_index("symbol")["weight"].sort_index()

    # --- No-look-ahead: mutating a FUTURE window (index 3, closing AFTER the
    #     target rebalance at _REB_LONG[3]) must NOT touch the target's weights. -
    future = _run(
        weights, _mutate_long_book_returns({3: -0.95}), universe, vol_target=0.20
    )
    pd.testing.assert_series_equal(weights_at(base), weights_at(future))

    # --- Positive control: mutating a PAST window (index 0, closing at
    #     _REB_LONG[1] <= the target rebalance) feeds book_returns[0], so it MUST
    #     change the target rebalance's scalar and hence its weights. A test that
    #     fails to react to in-window data is not discriminating. ---------------
    past = _run(
        weights, _mutate_long_book_returns({0: 0.25}), universe, vol_target=0.20
    )
    base_w = weights_at(base)
    past_w = weights_at(past)
    assert not base_w.equals(past_w), "past-window mutation did not move the scalar"


# ---------------------------------------------------------------------------
# costs reduce net vs gross
# ---------------------------------------------------------------------------

def test_costs_reduce_net_below_gross():
    weights = _simple_weights()
    returns = _simple_returns()
    universe = _universe(_REB, _SYMS)
    gross = _run(weights, returns, universe, cost_model=_zero_cost)
    net = _run(weights, returns, universe, cost_model=trade_cost)
    gross_final = gross["equity"]["equity"].iloc[-1]
    net_final = net["equity"]["equity"].iloc[-1]
    # Friction can only subtract — net terminal equity is strictly below gross
    # whenever any notional was traded (it always is here).
    assert net_final < gross_final
    assert (net["trades"]["cost"] >= 0).all()
    assert net["trades"]["cost"].sum() > 0


def test_nan_adv_does_not_poison_costs_or_equity():
    weights = _simple_weights()
    returns = _simple_returns()
    universe = _universe(_REB, _SYMS)
    universe.loc[universe["symbol"] == "B", "adv_30d"] = math.nan

    res = _run(weights, returns, universe, cost_model=trade_cost)

    assert res["trades"]["cost"].notna().all()
    assert res["trades"]["cost"].map(math.isfinite).all()
    assert res["equity"]["equity"].notna().all()
    assert res["equity"]["equity"].map(math.isfinite).all()


# ---------------------------------------------------------------------------
# a collapsed short-leg coin's crash flows into the book P&L
# ---------------------------------------------------------------------------

def test_collapsed_short_leg_crash_flows_to_pnl():
    weights = _simple_weights()
    universe = _universe(_REB, _SYMS)

    # Baseline: B drifts mildly in window 1. Crashed: B craters -90% in window 1.
    # Shorting B, the crash is a large POSITIVE contribution to the book, so the
    # equity after the first window must be higher when B craters.
    baseline = _run(weights, _simple_returns(), universe)
    crashed = _run(
        weights,
        _simple_returns(
            b_returns={
                "2024-02-15": -0.90,  # window-1 crash on the short leg
                "2024-03-01": 0.0,
                "2024-03-15": 0.01,
                "2024-04-01": 0.02,
            }
        ),
        universe,
    )
    # Equity at the close of the first holding window (the 2nd equity point).
    base_eq = baseline["equity"]["equity"].iloc[1]
    crash_eq = crashed["equity"]["equity"].iloc[1]
    assert crash_eq > base_eq


def test_all_missing_return_window_is_counted_in_equity_artifact():
    weights = _simple_weights()
    returns = _holding_returns([
        {"date": ts("2024-02-15"), "symbol": "A", "holding_return": 0.01},
        {"date": ts("2024-03-15"), "symbol": "A", "holding_return": 0.01},
        # B is held but has no observed return in either priced window.
    ])
    res = _run(weights, returns, _universe(_REB, _SYMS))
    priced = res["equity"].iloc[1:]
    assert (priced["missing_return_count"] >= 1).all()


# ---------------------------------------------------------------------------
# trade log: first rebalance trades the full target from a flat (zero) book
# ---------------------------------------------------------------------------

def test_first_rebalance_trades_full_target_from_flat_book():
    res = _run(_simple_weights(), _simple_returns(), _universe(_REB, _SYMS))
    trades = res["trades"]
    first = trades[trades["rebalance_date"] == _REB[0]]
    pos = res["positions"]
    first_pos = pos[pos["rebalance_date"] == _REB[0]].set_index("symbol")["weight"]
    # From a flat book the traded weight equals the target weight for each coin.
    for s, w in first_pos.items():
        traded = first[first["symbol"] == s]["traded_weight"].iloc[0]
        assert abs(traded - w) < 1e-12
