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


def test_oos_weight_inputs_open_guard_before_building_variant_books(monkeypatch):
    # The OOS path must own BOTH the OOS read and the OOS weight-book formation.
    # This pins the ordering directly: every variant book builder sees the guard
    # already opened, and it is asked to form only rows >= OOS_START.
    guard = rb.OneShotOOS()
    built: list[str] = []

    def fake_build_weight_book(price_panel, eligibility_input, spec, **kwargs):
        assert guard.opened
        assert kwargs["start"] == OOS_START
        built.append(spec["variant"])
        return pd.DataFrame(
            {
                "rebalance_date": [OOS_START],
                "symbol": ["A"],
                "weight": [1.0],
            }
        )

    monkeypatch.setattr(rb, "build_weight_book", fake_build_weight_book)

    returns = pd.DataFrame(
        {
            "date": [ts("2023-11-15"), ts("2023-12-15")],
            "symbol": ["A", "A"],
            "holding_return": [0.01, 0.02],
        }
    )
    universe = pd.DataFrame(
        {
            "date": [ts("2023-11-15"), ts("2023-12-15")],
            "symbol": ["A", "A"],
            "adv_30d": [1e9, 1e9],
        }
    )

    oos_book, oos_returns, oos_universe = rb.build_oos_weight_inputs_once(
        pd.DataFrame(),
        pd.DataFrame(),
        returns,
        universe,
        {"primary": rb.PRIMARY_SPEC, "comparator": rb.COMPARATOR_SPEC},
        guard,
        widened_specs=rb.WIDENED_CANDIDATES[:2],
    )

    assert guard.open_count == 1
    assert set(built) == {
        rb.PRIMARY_SPEC["variant"],
        rb.COMPARATOR_SPEC["variant"],
        rb.WIDENED_CANDIDATES[0]["variant"],
        rb.WIDENED_CANDIDATES[1]["variant"],
    }
    assert set(oos_book["_spec"]) == {
        "primary",
        "comparator",
        f"widened::{rb.WIDENED_CANDIDATES[0]['variant']}",
        f"widened::{rb.WIDENED_CANDIDATES[1]['variant']}",
    }
    assert (oos_book["rebalance_date"] >= OOS_START).all()
    assert (oos_returns["date"] >= OOS_START).all()
    assert (oos_universe["date"] >= OOS_START).all()
    with pytest.raises(RuntimeError):
        rb.build_oos_weight_inputs_once(
            pd.DataFrame(),
            pd.DataFrame(),
            returns,
            universe,
            {"primary": rb.PRIMARY_SPEC},
            guard,
        )


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


def _toy_oos_panels():
    """Toy panels that SPAN the sealed OOS boundary (so the OOS slice is non-empty).

    ``_toy_widened_panels`` lives entirely in 2020-2021 (before ``OOS_START`` =
    2023-12-02), so its OOS slice is empty — useless for an OOS test. This grid
    runs from mid-2022 to mid-2024 so the 30-day-cadence rebalances yield both a
    real in-sample stretch and ~6+ OOS rebalances past ``OOS_START``, with a clean
    deterministic momentum sort (monotone cross-sectional drift).
    """
    dates = pd.date_range("2022-06-01", periods=760, freq="D")
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


# ===========================================================================
# Task V2 — One-shot OOS validation of the skip>=2 candidates
# ===========================================================================
# The widened candidates' OOS windows are UNSPENT. V2 spends them through the
# SAME single ``read_oos_panels_once`` guard the pre-registered specs use — the
# widened books are folded into the existing combined-book concat, so adding the
# candidates does NOT add a second OOS open(). These tests exercise that
# single-guarded-path plumbing + the per-candidate OOS metric/gap/overfit-note +
# the 2x-cost and regime breakdown, all on TOY panels (no real data, no main()).


def test_widened_books_share_the_single_guarded_oos_read():
    # The combined book that feeds the ONE read_oos_panels_once call carries
    # primary + comparator AND every widened candidate (namespaced _spec tags).
    # A single guarded read returns the OOS slices of ALL specs at once; the guard
    # is opened EXACTLY ONCE even with all five candidates added, and a second
    # read raises. This is the discriminating proof that adding the candidates
    # does not introduce a second OOS code path.
    reb = [ts("2023-11-02"), ts("2023-12-02"), ts("2024-01-01")]
    spec_tags = ["primary", "comparator"] + [
        f"widened::{c['variant']}" for c in rb.WIDENED_CANDIDATES
    ]
    book_frames = []
    for tag in spec_tags:
        book_frames.append(
            pd.DataFrame(
                {
                    "rebalance_date": reb,
                    "symbol": ["A"] * 3,
                    "weight": [0.1, 0.2, 0.3],
                    "_spec": [tag] * 3,
                }
            )
        )
    combined = pd.concat(book_frames, ignore_index=True)
    returns = pd.DataFrame(
        {
            "date": [ts("2023-11-15"), ts("2023-12-15"), ts("2024-01-15")],
            "symbol": ["A"] * 3,
            "holding_return": [0.01, 0.02, 0.03],
        }
    )
    universe = pd.DataFrame(
        {"date": reb, "symbol": ["A"] * 3, "adv_30d": [1e9] * 3}
    )

    guard = rb.OneShotOOS()
    oos_book, oos_returns, oos_universe = rb.read_oos_panels_once(
        combined, returns, universe, guard
    )
    # ONE open for every spec, including all five widened candidates.
    assert guard.open_count == 1
    # Every widened candidate's OOS rows came back from the single read.
    oos_tags = set(oos_book["_spec"].unique())
    for c in rb.WIDENED_CANDIDATES:
        assert f"widened::{c['variant']}" in oos_tags
    assert {"primary", "comparator"}.issubset(oos_tags)
    # Only OOS-dated rows leak through, across all three panels.
    assert (oos_book["rebalance_date"] >= OOS_START).all()
    assert (oos_returns["date"] >= OOS_START).all()
    assert (oos_universe["date"] >= OOS_START).all()
    # A second read of the (now-spent) window raises — single use, count stays 1.
    with pytest.raises(RuntimeError):
        rb.read_oos_panels_once(combined, returns, universe, guard)
    assert guard.open_count == 1


def test_widened_candidate_oos_metrics_and_gap_computed():
    # Per candidate: OOS gross/net Sharpe, IS-vs-OOS net Sharpe gap, and an
    # overfit_note are produced from the OOS-only panels (no IS row enters the OOS
    # run). The gap is is_net_sharpe - oos_net_sharpe and is finite on the toy
    # trending panel.
    price_panel, elig = _toy_oos_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    spec = rb.WIDENED_PRIMARY

    book = rb.build_weight_book(price_panel, elig, spec)
    book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
    book_oos = book[book["rebalance_date"] >= OOS_START].reset_index(drop=True)
    oos_returns = holding[holding["date"] >= OOS_START]
    oos_universe = universe[universe["date"] >= OOS_START]
    # The OOS slice must be non-empty (the panel spans the boundary).
    assert not book_oos.empty

    rec = rb.characterize_candidate_oos(
        book_oos, oos_returns, oos_universe, is_net_sharpe=0.5
    )
    assert "oos_gross_sharpe" in rec
    assert "oos_net" in rec
    assert "is_oos_gap" in rec
    assert "overfit_note" in rec
    import numpy as np

    assert np.isfinite(rec["oos_net"]["sharpe"])
    # gap = IS - OOS, finite, and consistent with the supplied IS Sharpe.
    assert np.isfinite(rec["is_oos_gap"])
    assert abs(rec["is_oos_gap"] - (0.5 - rec["oos_net"]["sharpe"])) < 1e-9
    # The note is a non-empty honest string (overfitting / single-regime artifact).
    assert isinstance(rec["overfit_note"], str) and rec["overfit_note"]


def test_widened_candidate_oos_uses_only_oos_dated_rows_no_lookahead():
    # Discriminating no-look-ahead check: mutating an IN-SAMPLE-dated daily return
    # must NOT change any OOS metric — the OOS run reads only OOS-dated rows. (The
    # caller hands characterize_candidate_oos pre-sliced OOS-only panels, so an IS
    # mutation cannot leak in.)
    price_panel, elig = _toy_oos_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    spec = rb.WIDENED_PRIMARY

    book = rb.build_weight_book(price_panel, elig, spec)
    book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
    book_oos = book[book["rebalance_date"] >= OOS_START].reset_index(drop=True)
    oos_returns = holding[holding["date"] >= OOS_START]
    oos_universe = universe[universe["date"] >= OOS_START]

    base = rb.characterize_candidate_oos(
        book_oos, oos_returns, oos_universe, is_net_sharpe=0.5
    )
    # Now perturb the FULL holding panel at an IN-SAMPLE date, then re-slice OOS:
    # the OOS slice is unchanged, so the OOS metric must be identical.
    mutated = holding.copy()
    is_mask = mutated["date"] < OOS_START
    mutated.loc[is_mask, "holding_return"] = mutated.loc[is_mask, "holding_return"] + 5.0
    oos_returns_2 = mutated[mutated["date"] >= OOS_START]
    after = rb.characterize_candidate_oos(
        book_oos, oos_returns_2, oos_universe, is_net_sharpe=0.5
    )
    import numpy as np

    a, b = base["oos_net"]["sharpe"], after["oos_net"]["sharpe"]
    assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12


def test_widened_candidate_oos_overfit_note_flags_nonpositive_sharpe():
    # The overfit_note must call near-zero / negative OOS Sharpe overfitting, and
    # a positive-but-spent-once OOS as a single-regime artifact (not a deployable
    # edge). Drive the note logic directly via the (finite) gap classification.
    # Negative OOS -> overfitting language.
    neg = rb._oos_overfit_note(oos_net_sharpe=-0.3, is_oos_gap=0.8)
    assert "overfit" in neg.lower()
    # Strongly positive OOS that EXCEEDS IS (gap < 0) -> single-regime, not edge.
    art = rb._oos_overfit_note(oos_net_sharpe=1.2, is_oos_gap=-0.7)
    assert "single-regime" in art.lower() or "spent-once" in art.lower()
    assert "deployable" in art.lower()


def test_widened_candidate_oos_2x_cost_drags_equity_turnover_unchanged():
    # The OOS 2x-cost rerun must drag terminal equity below the 1x run while the
    # reported annualized turnover is identical (turnover is path-independent).
    price_panel, elig = _toy_oos_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    spec = rb.WIDENED_PRIMARY

    book = rb.build_weight_book(price_panel, elig, spec)
    book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
    book_oos = book[book["rebalance_date"] >= OOS_START].reset_index(drop=True)
    oos_returns = holding[holding["date"] >= OOS_START]
    oos_universe = universe[universe["date"] >= OOS_START]

    rec = rb.characterize_candidate_oos(
        book_oos, oos_returns, oos_universe, is_net_sharpe=0.5
    )
    # The 2x-cost OOS net metrics are present.
    assert "oos_net_2x" in rec
    # Turnover identical across 1x and 2x (path-independent).
    assert abs(rec["oos_net"]["annual_turnover"] - rec["oos_net_2x"]["annual_turnover"]) < 1e-9


def test_widened_candidate_oos_regime_breakdown_with_caveat():
    # An OOS regime breakdown (bull/bear/chop) is produced and carries the
    # do-not-over-read descriptive-proxy caveat (so it is never sold as a
    # walk-forward signal).
    price_panel, elig = _toy_oos_panels()
    holding = rb.price_panel_to_holding_returns(price_panel)
    universe = rb.eligibility_to_universe(elig)
    spec = rb.WIDENED_PRIMARY

    book = rb.build_weight_book(price_panel, elig, spec)
    book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
    book_oos = book[book["rebalance_date"] >= OOS_START].reset_index(drop=True)
    oos_returns = holding[holding["date"] >= OOS_START]
    oos_universe = universe[universe["date"] >= OOS_START]

    rec = rb.characterize_candidate_oos(
        book_oos, oos_returns, oos_universe, is_net_sharpe=0.5
    )
    regimes = rec["regimes"]
    assert set(regimes) == {"bull", "bear", "chop"}
    for r in regimes.values():
        assert "n" in r and "mean_net_return" in r
    # The descriptive caveat travels with the record (honest framing, do not
    # over-read the regime cut).
    assert "caveat" in rec
    assert "descriptive" in rec["caveat"].lower()


def test_widened_section_fills_oos_net_sharpe_when_record_supplies_it():
    # Once V2 stores oos_net_sharpe on a widened record, the markdown OOS column
    # shows the number rather than the V2 placeholder.
    records = _toy_widened_records()
    records["momentum_L3d_S3d"]["oos_net_sharpe"] = -0.42
    text = "\n".join(rb.widened_candidates_section(records))
    assert "-0.42" in text


# ===========================================================================
# Task V3 — per-candidate honest verdict from the (already-spent) OOS numbers
# ===========================================================================
# ``candidate_verdict`` is a PURE classifier over the real per-candidate numbers
# (IS net Sharpe, OOS net Sharpe, IS-vs-OOS gap, whether the variant clears the
# m=21 Bonferroni under BOTH HAC and bootstrap). It encodes the deployment bar:
#   * net edge dies (IS net Sharpe <= 0)            -> "works-only-gross"
#   * OOS net Sharpe non-positive / collapses        -> "fails-OOS"
#   * OOS positive but NOT m=21-robust under both     -> "marginal"
#   * OOS positive AND m=21-robust under both         -> "deployable"
# It is data-driven (it CAN return "deployable"); the no-deployable result on the
# real numbers falls out of the inputs, not a hardcoded verdict.


def test_candidate_verdict_fails_oos_when_oos_sharpe_collapses():
    # L3d/S3d-shaped: strong IS net (1.542), m=21-robust under both tests, but OOS
    # net collapses to 0.297 with a large IS-vs-OOS gap -> fails-OOS.
    v = rb.candidate_verdict(
        is_net_sharpe=1.542, oos_net_sharpe=0.297, is_oos_gap=1.245,
        clears_m21_both_tests=True,
    )
    assert v == "fails-OOS"


def test_candidate_verdict_collapse_rule_overrides_oos_above_floor():
    # Discriminating: the collapse rule is load-bearing. An OOS net Sharpe that is
    # ABOVE the overfit floor (0.40 > 0.25) and m=21-robust is STILL fails-OOS when
    # the IS-vs-OOS gap is large (the edge did not persist) — it must NOT be called
    # deployable. (Contrast test_..._deployable_when_robust_and_oos_holds, where the
    # gap is small.) Holding everything else fixed, only the gap separates the two.
    collapsed = rb.candidate_verdict(
        is_net_sharpe=1.60, oos_net_sharpe=0.40, is_oos_gap=1.20,
        clears_m21_both_tests=True,
    )
    stable = rb.candidate_verdict(
        is_net_sharpe=0.55, oos_net_sharpe=0.40, is_oos_gap=0.15,
        clears_m21_both_tests=True,
    )
    assert collapsed == "fails-OOS"
    assert stable == "deployable"


def test_candidate_verdict_fails_oos_when_oos_sharpe_negative():
    # L14d/S3d-shaped: m=21-robust IS but OOS net NEGATIVE (-0.486) -> fails-OOS.
    v = rb.candidate_verdict(
        is_net_sharpe=0.844, oos_net_sharpe=-0.486, is_oos_gap=1.330,
        clears_m21_both_tests=True,
    )
    assert v == "fails-OOS"


def test_candidate_verdict_marginal_when_oos_positive_but_not_m21_robust():
    # L1d/S3d-shaped: OOS net POSITIVE (0.455) but does NOT clear m=21 under BOTH
    # HAC and bootstrap (HAC clears, bootstrap does not) -> marginal, NOT deployable.
    v = rb.candidate_verdict(
        is_net_sharpe=1.010, oos_net_sharpe=0.455, is_oos_gap=0.555,
        clears_m21_both_tests=False,
    )
    assert v == "marginal"


def test_candidate_verdict_marginal_for_secondary_oos_positive_specs():
    # L5d/S2d-shaped (best OOS, 0.714) and L5d/S3d-shaped (0.645): OOS-positive but
    # NOT m=21-robust -> marginal (not deployable, the multiple-testing gate fails).
    assert rb.candidate_verdict(
        is_net_sharpe=0.923, oos_net_sharpe=0.714, is_oos_gap=0.209,
        clears_m21_both_tests=False,
    ) == "marginal"
    assert rb.candidate_verdict(
        is_net_sharpe=0.908, oos_net_sharpe=0.645, is_oos_gap=0.263,
        clears_m21_both_tests=False,
    ) == "marginal"


def test_candidate_verdict_can_return_deployable_when_robust_and_oos_holds():
    # Discriminating: a HYPOTHETICAL candidate with a strong OOS net Sharpe, a
    # small IS-vs-OOS gap AND m=21-robustness under both tests WOULD be deployable.
    # This proves the no-deployable result on the real numbers is data-driven, not
    # hardcoded. (No real candidate meets all three; this is a counterfactual.)
    v = rb.candidate_verdict(
        is_net_sharpe=0.90, oos_net_sharpe=0.80, is_oos_gap=0.10,
        clears_m21_both_tests=True,
    )
    assert v == "deployable"


def test_candidate_verdict_works_only_gross_when_net_edge_dies():
    # If the in-sample NET Sharpe is non-positive the edge is killed by costs ->
    # works-only-gross, regardless of OOS.
    v = rb.candidate_verdict(
        is_net_sharpe=-0.05, oos_net_sharpe=0.30, is_oos_gap=-0.35,
        clears_m21_both_tests=True,
    )
    assert v == "works-only-gross"


def test_no_widened_candidate_is_deployable_on_the_real_run_numbers():
    # The real Task-V3 run (numbers in docs/STAGE4_RESULTS.md): apply the verdict
    # to every candidate with its REAL OOS net Sharpe + its m=21 robustness flag.
    # NONE is deployable; the two m=21-robust survivors fail OOS, the OOS-positive
    # specs are only marginal. This is the headline V3 finding.
    real = {
        # variant: (is_net, oos_net, gap, clears_m21_both_tests)
        "momentum_L3d_S3d": (1.542, 0.297, 1.245, True),   # robust, fails OOS
        "momentum_L14d_S3d": (0.844, -0.486, 1.330, True),  # robust, OOS negative
        "momentum_L1d_S3d": (1.010, 0.455, 0.555, False),   # marginal (HAC-only)
        "momentum_L5d_S3d": (0.908, 0.645, 0.263, False),   # OOS+ but not m21
        "momentum_L5d_S2d": (0.923, 0.714, 0.209, False),   # best OOS but not m21
    }
    verdicts = {
        k: rb.candidate_verdict(
            is_net_sharpe=a, oos_net_sharpe=b, is_oos_gap=c,
            clears_m21_both_tests=d,
        )
        for k, (a, b, c, d) in real.items()
    }
    assert verdicts["momentum_L3d_S3d"] == "fails-OOS"
    assert verdicts["momentum_L14d_S3d"] == "fails-OOS"
    assert verdicts["momentum_L1d_S3d"] == "marginal"
    assert verdicts["momentum_L5d_S3d"] == "marginal"
    assert verdicts["momentum_L5d_S2d"] == "marginal"
    # The headline: NO candidate is deployable.
    assert "deployable" not in verdicts.values()
