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


# ===========================================================================
# Task V1 — POST-HOC widened skip>=2 candidates (in-sample cost-aware path)
# ===========================================================================
# The skip>=2 candidates are convention-fixed specs (NOT re-selected by Sharpe);
# their in-sample characterization reuses the SAME engine/cost/metric helpers as
# the pre-registered primary. These tests exercise the IS path on TOY panels only
# (no real data, no OOS, no main()).


def test_widened_candidates_are_convention_fixed_skip_ge_2_specs():
    # Exactly the five candidates the plan names, with the fixed (lookback, skip)
    # — these are pre-registered post-hoc, NOT re-selected from the grid.
    cands = {c["variant"]: c for c in rb.WIDENED_CANDIDATES}
    assert set(cands) == {
        "momentum_L3d_S3d",
        "momentum_L14d_S3d",
        "momentum_L1d_S3d",
        "momentum_L5d_S3d",
        "momentum_L5d_S2d",
    }
    expected = {
        "momentum_L3d_S3d": (3, 3),
        "momentum_L14d_S3d": (14, 3),
        "momentum_L1d_S3d": (1, 3),
        "momentum_L5d_S3d": (5, 3),
        "momentum_L5d_S2d": (5, 2),
    }
    for variant, (lb, sk) in expected.items():
        assert cands[variant]["lookback"] == lb
        assert cands[variant]["skip"] == sk
        # Every candidate has skip >= 2 (the whole point of the widened family).
        assert cands[variant]["skip"] >= 2
        # Quantile is the convention-fixed quintile, not tuned.
        assert cands[variant]["quantile"] == QUANTILE
    # The primary candidate of the widened family is L3d/S3d (the highest-t lead).
    assert rb.WIDENED_PRIMARY["variant"] == "momentum_L3d_S3d"


def _toy_widened_panels():
    """Toy panels with enough history for a skip=3, lookback up-to-14 signal.

    400 daily bars over 20 coins with a monotone cross-sectional drift so the
    momentum sort is deterministic and a clean dollar-neutral book forms for
    every skip>=2 candidate (the longest needs ~lookback+skip = 17 days of lead).
    """
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    syms = [f"C{i}" for i in range(20)]
    rows = []
    elig_rows = []
    for i, s in enumerate(syms):
        drift = 0.0006 * (i - 9.5)  # C19 strongest up, C0 strongest down
        price = 100.0
        for d in dates:
            price *= (1.0 + drift)
            rows.append({"date": d, "symbol": s, "price": price})
            elig_rows.append({"date": d, "symbol": s, "eligible": True})
    return pd.DataFrame(rows), pd.DataFrame(elig_rows)


def test_widened_candidate_net_sharpe_below_gross_on_toy_panel():
    # Costs can only subtract: for EVERY skip>=2 candidate the in-sample net
    # Sharpe must be strictly below the gross Sharpe on the toy panel (a
    # discriminating check — a zero-cost bug would make them equal).
    price_panel, elig = _toy_widened_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    for spec in rb.WIDENED_CANDIDATES:
        rec = rb.characterize_candidate_in_sample(
            price_panel, elig, holding, universe, spec
        )
        gross_sh = rec["is_gross_sharpe"]
        net_sh = rec["is_net"]["sharpe"]
        # Friction strictly reduces the Sharpe (the toy book trades every
        # rebalance, so a non-trivial cost is charged -> net < gross).
        assert net_sh < gross_sh, f"{spec['variant']}: net {net_sh} !< gross {gross_sh}"


def test_widened_candidate_turnover_is_path_independent_across_cost_multipliers():
    # Annualized turnover is a function of the TRADED WEIGHT fractions, not the
    # cost-eroded equity path, so a 1x-cost and a 2x-cost run of the SAME
    # candidate must report the SAME annualized turnover (the discriminating
    # check: a turnover divisor tied to the equity path would drift between runs).
    price_panel, elig = _toy_widened_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    book = rb.build_weight_book(price_panel, elig, rb.WIDENED_PRIMARY)
    book_is = rb.in_sample_slice(book)

    net_1x = rb.run_full(book_is, holding, universe, cost_multiplier=1.0)
    net_2x = rb.run_full(book_is, holding, universe, cost_multiplier=2.0)
    turn_1x = rb.metrics_for(net_1x["equity"], net_1x["trades"])["annual_turnover"]
    turn_2x = rb.metrics_for(net_2x["equity"], net_2x["trades"])["annual_turnover"]

    # Turnover is positive (the book actually trades) and identical across cost
    # multipliers — path-independent.
    assert turn_1x > 0.0
    assert abs(turn_1x - turn_2x) < 1e-9
    # The 2x run's equity path is genuinely different (more cost drag), so the
    # equality above is not trivially true because the runs are identical.
    assert net_2x["equity"]["equity"].iloc[-1] < net_1x["equity"]["equity"].iloc[-1]
    # The characterization record exposes the turnover so it is reportable.
    rec = rb.characterize_candidate_in_sample(
        price_panel, elig, holding, universe, rb.WIDENED_PRIMARY
    )
    assert abs(rec["is_net"]["annual_turnover"] - turn_1x) < 1e-9


def test_widened_candidate_capacity_uses_traded_order_via_metrics_capacity():
    # Capacity is computed by metrics.capacity on the per-rebalance TRADED order
    # (built by _candidate_for_capacity from the trade log), NOT the held book.
    # The record must carry a finite, positive capacity_aum and the gross edge
    # that feeds it.
    price_panel, elig = _toy_widened_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    rec = rb.characterize_candidate_in_sample(
        price_panel, elig, holding, universe, rb.WIDENED_PRIMARY
    )
    # The trending toy book has a positive gross edge -> a finite crossing AUM.
    assert rec["gross_edge"] > 0.0
    assert rec["capacity_aum"] > 0.0
    # Cross-check: the capacity is exactly metrics.capacity on the candidate that
    # _candidate_for_capacity builds from the SAME net run's trade log (proving
    # the traded-order path is used, not the held weight).
    cand = rb._candidate_for_capacity(
        universe, rec["_net_run"]["trades"], rec["gross_edge"]
    )
    from amom.backtest.costs import trade_cost
    from amom.backtest.metrics import capacity

    expected = capacity(cand, trade_cost)["capacity_aum"]
    assert rec["capacity_aum"] == expected
    # The traded order (Σ|Δw|/n_rebalances) is strictly below the summed held
    # weight — the standing book is re-established, not re-traded each rebalance.
    summed_traded = sum(frac for frac, _adv, _rank in cand["coins"])
    book_is = rb.in_sample_slice(rb.build_weight_book(price_panel, elig, rb.WIDENED_PRIMARY))
    summed_held = float(
        book_is.groupby("symbol")["weight"].apply(lambda w: w.abs().mean()).sum()
    )
    assert summed_traded < summed_held


def test_widened_section_markdown_is_post_hoc_and_below_preregistered():
    # write_markdown emits a clearly-labelled POST-HOC widened section that names
    # every candidate and the turnover/net-vs-gross cost risk; the pre-registered
    # L5d/S1d + L28d/S1d content stays INTACT and ABOVE it.
    md = rb.widened_candidates_section(_toy_widened_records())
    text = "\n".join(md)
    assert "post-hoc" in text.lower() or "widened" in text.lower()
    # Every candidate is named in the widened section.
    for variant in (
        "momentum_L3d_S3d",
        "momentum_L14d_S3d",
        "momentum_L1d_S3d",
        "momentum_L5d_S3d",
        "momentum_L5d_S2d",
    ):
        assert variant in text
    # The short-lookback turnover / net-of-cost risk is surfaced, not hidden.
    assert "turnover" in text.lower()
    # A candidate killed net of costs is flagged.
    assert "net" in text.lower()


def test_widened_section_marks_missing_candidate_as_pending_v2():
    # A candidate not yet characterized (no record) renders an explicit n/a row
    # with the OOS column pointing to V2 — it is never silently dropped.
    records = _toy_widened_records()
    del records["momentum_L5d_S2d"]
    text = "\n".join(rb.widened_candidates_section(records))
    # The missing candidate still appears (named) with a V2 placeholder.
    assert "momentum_L5d_S2d" in text
    assert "_V2_" in text


def _toy_widened_records() -> dict:
    """Minimal in-sample records (one per candidate) for the markdown builder."""
    out = {}
    for spec in rb.WIDENED_CANDIDATES:
        out[spec["variant"]] = {
            "spec": spec,
            "is_gross_sharpe": 1.0,
            "is_net": {
                "sharpe": 0.5,
                "annual_return": 0.10,
                "annual_vol": 0.20,
                "annual_turnover": 30.0,
                "max_drawdown": -0.1,
            },
            "is_net_2x": {"sharpe": 0.3, "annual_turnover": 30.0, "annual_return": 0.05},
            "capacity_aum": 5_000_000.0,
            "gross_edge": 0.01,
        }
    return out
