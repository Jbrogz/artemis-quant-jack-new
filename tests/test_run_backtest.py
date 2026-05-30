"""Tests for the Stage-4 cost-aware backtest runner (Task B4, spec §4.6).

``scripts/run_backtest.py`` characterizes the primary candidate
``momentum_L5d_S1d`` (and the comparator ``momentum_L28d_S1d``) net of spot
costs, then **spends the sealed OOS window exactly once** and reports the
in-sample-vs-OOS Sharpe gap plus robustness reruns (2x costs, +/-50% lookback,
regime breakdown).

Discriminating behaviours (plan §B4):
  * the OOS slice (rows >= ``OOS_START``) is read in **exactly one** code path —
    a single-use guard whose ``open()`` raises on a second call (a counter);
  * **gross >= net** for the candidate (friction can only subtract);
  * the robustness reruns **reuse the chosen spec** (the same primary variant /
    skip / quantile) rather than re-selecting from the grid.

The heavy numeric paths reuse synthetic panels; no live API, no disk writes in
the unit tests (the script's ``main`` is exercised separately, live, by the
committer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_backtest as rb  # noqa: E402

from amom.config import (  # noqa: E402
    OOS_START,
    PRIMARY_SKIP_DAYS,
    QUANTILE,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# the chosen spec is the pre-registered primary candidate (not a new selection)
# ---------------------------------------------------------------------------

def test_chosen_spec_is_the_preregistered_primary():
    spec = rb.PRIMARY_SPEC
    assert spec["variant"] == "momentum_L5d_S1d"
    assert spec["lookback"] == 5
    assert spec["skip"] == PRIMARY_SKIP_DAYS
    assert spec["quantile"] == QUANTILE
    # The comparator is the 28d academic 4-week canonical.
    assert rb.COMPARATOR_SPEC["variant"] == "momentum_L28d_S1d"
    assert rb.COMPARATOR_SPEC["lookback"] == 28


# ---------------------------------------------------------------------------
# OOS single-use guard: the OOS window is opened in exactly one code path
# ---------------------------------------------------------------------------

def test_oos_guard_opens_exactly_once():
    guard = rb.OneShotOOS()
    assert guard.opened is False
    assert guard.open_count == 0
    first = guard.open()
    assert first is True
    assert guard.opened is True
    assert guard.open_count == 1
    # A SECOND open is a programming error — the OOS window is spent once only.
    with pytest.raises(RuntimeError):
        guard.open()
    assert guard.open_count == 1


def test_oos_slice_only_returned_after_guard_opened():
    # A frame spanning the OOS boundary.
    df = pd.DataFrame(
        {
            "rebalance_date": [ts("2023-11-02"), ts("2023-12-02"), ts("2024-01-01")],
            "factor_return": [0.1, 0.2, 0.3],
        }
    )
    guard = rb.OneShotOOS()
    oos = rb.read_oos_once(df, guard)
    # Only rows >= OOS_START come back, and the guard is now spent.
    assert (oos["rebalance_date"] >= OOS_START).all()
    assert len(oos) == 2
    assert guard.opened is True
    # A second read through the (already-spent) guard raises — single use.
    with pytest.raises(RuntimeError):
        rb.read_oos_once(df, guard)


def test_combined_multi_spec_oos_read_is_single_path():
    # The runner concatenates EVERY spec's book and reads the OOS slice in ONE
    # guarded call — both specs' OOS rows come back together, and a second read
    # raises. This is the single code path that touches rows >= OOS_START.
    primary = pd.DataFrame(
        {"rebalance_date": [ts("2023-11-02"), ts("2023-12-02")], "_spec": ["primary"] * 2}
    )
    comparator = pd.DataFrame(
        {"rebalance_date": [ts("2023-11-02"), ts("2024-01-01")], "_spec": ["comparator"] * 2}
    )
    combined = pd.concat([primary, comparator], ignore_index=True)
    guard = rb.OneShotOOS()
    oos = rb.read_oos_once(combined, guard)
    # One OOS row per spec; both specs represented from the single read.
    assert set(oos["_spec"].unique()) == {"primary", "comparator"}
    assert (oos["rebalance_date"] >= OOS_START).all()
    assert guard.open_count == 1
    with pytest.raises(RuntimeError):
        rb.read_oos_once(combined, guard)


def test_oos_panels_read_in_one_guarded_call_covers_returns_and_universe():
    # The OOS engine runs must NOT receive full panels and read OOS-dated daily
    # returns / universe rows "by convention". A single guarded read returns the
    # OOS slices of ALL THREE inputs (book by rebalance_date, returns + universe
    # by date), so OOS-once is ENFORCED across every input, not just the book.
    book = pd.DataFrame(
        {
            "rebalance_date": [ts("2023-11-02"), ts("2023-12-02"), ts("2024-01-01")],
            "symbol": ["A", "A", "A"],
            "weight": [0.1, 0.2, 0.3],
            "_spec": ["primary"] * 3,
        }
    )
    returns = pd.DataFrame(
        {
            "date": [ts("2023-11-15"), ts("2023-12-15"), ts("2024-01-15")],
            "symbol": ["A", "A", "A"],
            "holding_return": [0.01, 0.02, 0.03],
        }
    )
    universe = pd.DataFrame(
        {
            "date": [ts("2023-11-02"), ts("2023-12-02"), ts("2024-01-01")],
            "symbol": ["A", "A", "A"],
            "adv_30d": [1e9, 1e9, 1e9],
        }
    )
    guard = rb.OneShotOOS()
    oos_book, oos_returns, oos_universe = rb.read_oos_panels_once(
        book, returns, universe, guard
    )
    # The guard is spent EXACTLY ONCE for all three reads.
    assert guard.open_count == 1
    # Every returned slice is OOS-only — no row dated < OOS_START leaks in, and
    # (critically) no OOS-dated returns/universe row is read outside this path.
    assert (oos_book["rebalance_date"] >= OOS_START).all()
    assert (oos_returns["date"] >= OOS_START).all()
    assert (oos_universe["date"] >= OOS_START).all()
    assert len(oos_book) == 2
    assert len(oos_returns) == 2
    assert len(oos_universe) == 2
    # A second guarded read raises — the window is provably spent once.
    with pytest.raises(RuntimeError):
        rb.read_oos_panels_once(book, returns, universe, guard)


def test_in_sample_slice_never_touches_oos_rows():
    df = pd.DataFrame(
        {
            "rebalance_date": [ts("2023-11-02"), ts("2023-12-02"), ts("2024-01-01")],
            "factor_return": [0.1, 0.2, 0.3],
        }
    )
    is_ = rb.in_sample_slice(df)
    assert (is_["rebalance_date"] < OOS_START).all()
    assert len(is_) == 1
    # in_sample_slice is a pure read — it does NOT consume the OOS guard.


# ---------------------------------------------------------------------------
# the weight-book extractor reproduces the dollar-neutral candidate book
# ---------------------------------------------------------------------------

def _toy_panels():
    """A deterministic price/eligibility panel with a clean momentum effect.

    Twenty coins over a long daily grid (so a ``QUANTILE`` = 20% quintile yields
    4 names per leg, clearing ``MIN_BUCKET_SIZE`` = 3); the strongest-trending
    coins are the longs, the weakest the shorts — a clean dollar-neutral book.
    """
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    syms = [f"C{i}" for i in range(20)]
    # Monotone-by-coin drift so the cross-sectional momentum sort is deterministic.
    rows = []
    elig_rows = []
    for i, s in enumerate(syms):
        drift = 0.0005 * (i - 9.5)  # C19 strongest up, C0 strongest down
        price = 100.0
        for d in dates:
            price *= (1.0 + drift)
            rows.append({"date": d, "symbol": s, "price": price})
            elig_rows.append({"date": d, "symbol": s, "eligible": True})
    price_panel = pd.DataFrame(rows)
    elig = pd.DataFrame(elig_rows)
    return price_panel, elig


def test_weight_book_is_dollar_neutral_for_the_chosen_spec():
    price_panel, elig = _toy_panels()
    book = rb.build_weight_book(price_panel, elig, rb.PRIMARY_SPEC)
    assert not book.empty
    assert set(["rebalance_date", "symbol", "weight"]).issubset(book.columns)
    # Each rebalance is dollar-neutral (Σ signed weight ~= 0).
    for _, grp in book.groupby("rebalance_date"):
        assert abs(grp["weight"].sum()) < 1e-9
        # The long leg is positive, the short leg negative — a real long/short book.
        assert (grp["weight"] > 0).any()
        assert (grp["weight"] < 0).any()


def test_lookback_robustness_reuses_chosen_spec_only_varying_lookback():
    # +/-50% lookback variants of L5d -> L2d / L7d (rounded), at the SAME skip /
    # quantile — a robustness rerun, not a re-selection across the 7-lookback grid.
    variants = rb.lookback_robustness_specs(rb.PRIMARY_SPEC)
    assert len(variants) == 2
    for spec in variants:
        assert spec["skip"] == rb.PRIMARY_SPEC["skip"]
        assert spec["quantile"] == rb.PRIMARY_SPEC["quantile"]
    looks = sorted(s["lookback"] for s in variants)
    # round(5*0.5)=2 (down) and round(5*1.5)=8 (up) — both within the frozen grid
    # only by coincidence; the point is they are derived from the chosen lookback,
    # not freshly selected.
    assert looks == [2, 8] or looks == [3, 8] or looks == [2, 7] or looks == [3, 7]
    # And the derived lookbacks bracket the chosen one.
    assert looks[0] < rb.PRIMARY_SPEC["lookback"] < looks[1]


# ---------------------------------------------------------------------------
# gross >= net for a real backtest run on the toy panels
# ---------------------------------------------------------------------------

def test_gross_ge_net_on_a_real_run():
    price_panel, elig = _toy_panels()
    book = rb.build_weight_book(price_panel, elig, rb.PRIMARY_SPEC)
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    gross_eq, net_eq = rb.run_gross_and_net(book, holding, universe, aum=1_000_000.0)
    # Friction can only subtract: terminal net equity <= gross terminal equity.
    assert net_eq["equity"].iloc[-1] <= gross_eq["equity"].iloc[-1] + 1e-6


# ---------------------------------------------------------------------------
# capacity candidate uses the per-rebalance TRADED delta, not mean |held weight|
# ---------------------------------------------------------------------------

def test_capacity_candidate_uses_traded_delta_not_held_weight():
    # Two coins, three rebalances. The book is held roughly flat each rebalance
    # (mean |held weight| ~= 0.5 per coin -> summed = 1.0), but the TRADED delta
    # per rebalance is far smaller because the standing book is re-established,
    # not re-traded in full. The candidate must reflect the order size that is
    # actually executed (Σ |Δw| / n_rebalances per coin), NOT the held weight.
    reb = [ts("2024-01-01"), ts("2024-02-01"), ts("2024-03-01")]
    book = pd.DataFrame(
        [
            {"rebalance_date": reb[0], "symbol": "A", "weight": 0.5},
            {"rebalance_date": reb[0], "symbol": "B", "weight": -0.5},
            {"rebalance_date": reb[1], "symbol": "A", "weight": 0.5},
            {"rebalance_date": reb[1], "symbol": "B", "weight": -0.5},
        ]
    )
    # A trade log where the FIRST rebalance trades the full target from flat, but
    # the SECOND barely trades (the book is re-established): per-coin traded
    # fractions are (0.5 + 0.02) and (0.5 + 0.02) over 2 priced rebalances ->
    # mean per-rebalance |Δw| = 0.26 each; summed one-way turnover = 0.52.
    trades = pd.DataFrame(
        [
            {"rebalance_date": reb[0], "symbol": "A", "traded_weight": 0.5,
             "traded_notional": 0.0, "cost": 0.0},
            {"rebalance_date": reb[0], "symbol": "B", "traded_weight": -0.5,
             "traded_notional": 0.0, "cost": 0.0},
            {"rebalance_date": reb[1], "symbol": "A", "traded_weight": 0.02,
             "traded_notional": 0.0, "cost": 0.0},
            {"rebalance_date": reb[1], "symbol": "B", "traded_weight": -0.02,
             "traded_notional": 0.0, "cost": 0.0},
        ]
    )
    universe = pd.DataFrame(
        [
            {"date": reb[0], "symbol": "A", "adv_30d": 1e8},
            {"date": reb[0], "symbol": "B", "adv_30d": 1e8},
        ]
    )
    cand = rb._candidate_for_capacity(universe, trades, gross_expected_return=0.01)
    summed_traded = sum(frac for frac, _adv, _rank in cand["coins"])
    # Per coin: (0.5 + 0.02) / 2 priced rebalances = 0.26 each -> summed = 0.52.
    assert abs(summed_traded - 0.52) < 1e-9
    # And it is strictly BELOW the (wrong) mean |held weight| sum of 1.0 — the
    # traded order is smaller than the standing book, which is the whole point.
    mean_held = float(book.groupby("symbol")["weight"].apply(lambda w: w.abs().mean()).sum())
    assert summed_traded < mean_held
    assert cand["gross_expected_return"] == 0.01


# ---------------------------------------------------------------------------
# regime labelling: bull/bear/chop partition by trailing market sign x vol tercile
# ---------------------------------------------------------------------------

def test_regime_labels_partition_every_period():
    # A market-return series with clear up / down / quiet stretches.
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    mkt = pd.Series(
        [0.10, 0.08, -0.09, -0.07, 0.001, -0.002, 0.05, -0.06, 0.002, 0.09, -0.08, 0.001],
        index=idx,
    )
    labels = rb.regime_labels(mkt)
    # Every period is labelled exactly one of the three regimes.
    assert set(labels.unique()).issubset({"bull", "bear", "chop"})
    assert labels.notna().all()
    assert len(labels) == len(mkt)
