"""Tests for the Stage-2 significance-battery runner (Task T4, plan §T4).

The runner computes, on the IN-SAMPLE slice only, one result row per variant with
the full battery (naive t, HAC t, mean/ann return, Lo Sharpe+SE+autocorr flag,
spanning alpha+HAC t, bootstrap p+disagreement, subsample signs+holds_sign, HLZ
tier, effective-n/power label), applies Bonferroni to the pre-registered
selection family (7 lookbacks at skip=1, m=7), records the TOTAL test count, and
runs DSR + PBO across the grid.

These tests pin the pure, offline-testable cores:
  * ``selection_family`` extracts exactly the 7 skip=1 lookback variants (m=7);
  * ``compute_variant_row`` returns one fully-populated row per variant;
  * ``apply_bonferroni`` corrects the selection family at m == 7;
  * ``build_significance_table`` yields one row per input variant (incl. failures).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_stage2  # noqa: E402
from amom.config import LOOKBACKS_DAYS, OOS_START, PRIMARY_SKIP_DAYS  # noqa: E402


def _variant(lookback: int, skip: int) -> str:
    return f"momentum_L{lookback}d_S{skip}d"


def _synthetic_factor_returns(seed: int = 0) -> pd.DataFrame:
    """A 21-variant in-sample-sized factor-return frame (skip 1/2/3 x 7 lookbacks)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-04-02", periods=69, freq="30D")
    rows = []
    for skip in (1, 2, 3):
        for lb in LOOKBACKS_DAYS:
            r = rng.normal(0.01, 0.05, len(dates))
            for d, v in zip(dates, r):
                rows.append(
                    {
                        "variant": _variant(lb, skip),
                        "rebalance_date": d,
                        "factor_return": float(v),
                        "long_return": float(v) / 2,
                        "short_return": -float(v) / 2,
                        "n_long": 5,
                        "n_short": 5,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# selection_family: the pre-registered 7 lookbacks at skip=1 (m_select = 7)
# ---------------------------------------------------------------------------

def test_selection_family_is_seven_skip1_lookbacks():
    fr = _synthetic_factor_returns()
    fam = run_stage2.selection_family(sorted(fr["variant"].unique()))
    assert len(fam) == 7  # m_select == 7
    assert all(v.endswith(f"_S{PRIMARY_SKIP_DAYS}d") for v in fam)
    # exactly the 7 frozen lookbacks at skip 1
    assert set(fam) == {_variant(lb, PRIMARY_SKIP_DAYS) for lb in LOOKBACKS_DAYS}


# ---------------------------------------------------------------------------
# apply_bonferroni: Bonferroni on the selection family uses m == 7
# ---------------------------------------------------------------------------

def test_apply_bonferroni_family_m_is_seven():
    # 7 p-values, one clearly significant at the 0.05/7 threshold.
    family = [_variant(lb, 1) for lb in LOOKBACKS_DAYS]
    pvals = {v: 0.5 for v in family}
    pvals[family[0]] = 0.001  # survives 0.05/7 ≈ 0.00714
    out = run_stage2.apply_bonferroni(family, pvals)
    assert out["m"] == 7
    assert np.isclose(out["threshold"], 0.05 / 7)
    assert out["survivors"] == [family[0]]


def test_apply_bonferroni_no_survivors_when_all_weak():
    family = [_variant(lb, 1) for lb in LOOKBACKS_DAYS]
    pvals = {v: 0.20 for v in family}
    out = run_stage2.apply_bonferroni(family, pvals)
    assert out["m"] == 7
    assert out["survivors"] == []


# ---------------------------------------------------------------------------
# compute_variant_row: one fully-populated row with the full battery
# ---------------------------------------------------------------------------

def test_compute_variant_row_has_all_battery_columns():
    rng = np.random.default_rng(5)
    n = 69
    idx = pd.date_range("2018-04-02", periods=n, freq="30D")
    fr = pd.Series(rng.normal(0.02, 0.04, n), index=idx)
    regressors = pd.DataFrame(
        {
            "market_return": rng.normal(0.0, 0.05, n),
            "size_control": rng.normal(0.0, 0.04, n),
        },
        index=idx,
    )
    row = run_stage2.compute_variant_row("momentum_L14d_S1d", fr, regressors)

    required = {
        "variant", "n_obs", "naive_t", "hac_t", "hac_maxlags",
        "mean_return", "ann_return", "sharpe", "sharpe_se", "autocorr_flag",
        "spanning_alpha", "spanning_alpha_t", "bootstrap_p", "hac_p",
        "disagreement", "reported_p", "full_sign", "holds_sign",
        "hlz_tier", "effective_n", "power", "power_label",
    }
    assert required.issubset(set(row.keys()))
    assert row["variant"] == "momentum_L14d_S1d"
    assert row["n_obs"] == n
    # naive and HAC t are distinct statistics (HAC corrects the SE)
    assert np.isfinite(row["naive_t"])
    assert np.isfinite(row["hac_t"])
    # HLZ tier is one of the three faithful labels
    assert row["hlz_tier"] in {"significant", "suggestive", "not_significant"}


def test_compute_variant_row_reported_p_follows_bootstrap_on_disagreement():
    # If HAC and bootstrap disagree across the threshold, reported_p is the
    # bootstrap p (spec §2.7 — bootstrap overrides NW on disagreement).
    rng = np.random.default_rng(6)
    n = 69
    idx = pd.date_range("2018-04-02", periods=n, freq="30D")
    fr = pd.Series(rng.normal(0.0, 0.04, n), index=idx)
    regressors = pd.DataFrame(
        {"market_return": rng.normal(0.0, 0.05, n), "size_control": rng.normal(0.0, 0.04, n)},
        index=idx,
    )
    row = run_stage2.compute_variant_row("momentum_L1d_S1d", fr, regressors)
    if row["disagreement"]:
        assert row["reported_p"] == row["bootstrap_p"]
    else:
        assert row["reported_p"] == row["hac_p"]


# ---------------------------------------------------------------------------
# build_significance_table: one row per variant (failures included)
# ---------------------------------------------------------------------------

def test_build_significance_table_one_row_per_variant():
    fr = _synthetic_factor_returns()
    # Minimal regressors aligned to the dates (one set, shared across variants).
    dates = pd.DatetimeIndex(sorted(fr["rebalance_date"].unique()))
    rng = np.random.default_rng(9)
    regressors = pd.DataFrame(
        {
            "market_return": rng.normal(0.0, 0.05, len(dates)),
            "size_control": rng.normal(0.0, 0.04, len(dates)),
        },
        index=dates,
    )
    table = run_stage2.build_significance_table(fr, regressors)
    assert len(table) == fr["variant"].nunique() == 21
    assert set(table["variant"]) == set(fr["variant"].unique())
    # every variant got a row even if its battery is weak (failures included)
    assert table["variant"].is_unique


def test_build_significance_table_marks_selection_membership():
    fr = _synthetic_factor_returns()
    dates = pd.DatetimeIndex(sorted(fr["rebalance_date"].unique()))
    rng = np.random.default_rng(10)
    regressors = pd.DataFrame(
        {
            "market_return": rng.normal(0.0, 0.05, len(dates)),
            "size_control": rng.normal(0.0, 0.04, len(dates)),
        },
        index=dates,
    )
    table = run_stage2.build_significance_table(fr, regressors)
    # exactly 7 rows flagged as the pre-registered selection family
    assert int(table["in_selection_family"].sum()) == 7
    sel = set(table.loc[table["in_selection_family"], "variant"])
    assert sel == {_variant(lb, PRIMARY_SKIP_DAYS) for lb in LOOKBACKS_DAYS}


def test_build_significance_table_seals_oos_rows_before_computing(monkeypatch):
    # The public table builder must enforce the Stage-2 OOS seal itself. Mutating
    # rows dated >= OOS_START cannot change the per-variant series handed to the
    # statistical battery, even if a caller accidentally passes the full panel.
    variants = [_variant(lb, PRIMARY_SKIP_DAYS) for lb in LOOKBACKS_DAYS]
    is_date = OOS_START - pd.Timedelta(days=30)
    rows = []
    for variant in variants:
        rows.append(
            {"variant": variant, "rebalance_date": is_date, "factor_return": 0.01}
        )
        rows.append(
            {"variant": variant, "rebalance_date": OOS_START, "factor_return": 99.0}
        )
    factor_returns = pd.DataFrame(rows)
    regressors = pd.DataFrame(index=pd.DatetimeIndex([is_date]))

    captured_runs: list[dict[str, tuple[pd.Timestamp, ...]]] = []

    def fake_compute_variant_row(variant, returns, regressors):
        captured_runs[-1][variant] = tuple(returns.index)
        return {"variant": variant, "reported_p": 0.5}

    monkeypatch.setattr(run_stage2, "compute_variant_row", fake_compute_variant_row)

    for oos_value in (99.0, -999.0):
        mutated = factor_returns.copy()
        mutated.loc[mutated["rebalance_date"] >= OOS_START, "factor_return"] = oos_value
        captured_runs.append({})
        table = run_stage2.build_significance_table(mutated, regressors)
        assert len(table) == len(variants)

    assert captured_runs[0] == captured_runs[1]
    for captured in captured_runs:
        assert set(captured) == set(variants)
        for indexes in captured.values():
            assert indexes == (is_date,)


# ---------------------------------------------------------------------------
# apply_widened_bonferroni: POST-HOC widened family (skip-as-axis, m == 21)
# ---------------------------------------------------------------------------

def _verified_widened_inputs() -> dict:
    """The five candidate stats verified from data/stats/significance.parquet.

    Only the rows that matter for the widened Bonferroni verdict are pinned with
    their real (reported_p, hac_p, bootstrap_p); every other variant is given a
    clearly non-clearing p so the m == 21 family is complete but the survivor set
    is driven by the verified candidates.
    """
    reported_p: dict[str, float] = {}
    hac_p: dict[str, float] = {}
    boot_p: dict[str, float] = {}
    for skip in (1, 2, 3):
        for lb in LOOKBACKS_DAYS:
            v = _variant(lb, skip)
            reported_p[v] = 0.20
            hac_p[v] = 0.20
            boot_p[v] = 0.20
    # Verified candidate stats (TASK V0 data facts).
    reported_p[_variant(3, 3)], hac_p[_variant(3, 3)], boot_p[_variant(3, 3)] = 2.66e-7, 2.66e-7, 0.0002
    reported_p[_variant(14, 3)], hac_p[_variant(14, 3)], boot_p[_variant(14, 3)] = 3.93e-5, 3.93e-5, 0.0006
    reported_p[_variant(1, 3)], hac_p[_variant(1, 3)], boot_p[_variant(1, 3)] = 0.00218, 0.00218, 0.0054
    reported_p[_variant(5, 3)], hac_p[_variant(5, 3)], boot_p[_variant(5, 3)] = 0.00391, 0.00391, 0.0030
    reported_p[_variant(5, 2)], hac_p[_variant(5, 2)], boot_p[_variant(5, 2)] = 0.0108, 0.00483, 0.0108
    return {"reported_p": reported_p, "hac_p": hac_p, "boot_p": boot_p}


def test_widened_bonferroni_m_is_21_and_threshold():
    inp = _verified_widened_inputs()
    out = run_stage2.apply_widened_bonferroni(
        inp["reported_p"], hac_p=inp["hac_p"], boot_p=inp["boot_p"]
    )
    assert out["m"] == 21  # 7 lookbacks x 3 skips, skip promoted to a selection axis
    assert np.isclose(out["threshold"], 0.05 / 21, atol=1e-9)
    assert np.isclose(out["threshold"], 0.0023810, atol=1e-6)


def test_widened_bonferroni_survivors_on_reported_p():
    inp = _verified_widened_inputs()
    out = run_stage2.apply_widened_bonferroni(
        inp["reported_p"], hac_p=inp["hac_p"], boot_p=inp["boot_p"]
    )
    # Clearers on reported_p at threshold 0.05/21 = 0.0023810.
    assert set(out["survivors"]) == {
        _variant(3, 3),
        _variant(14, 3),
        _variant(1, 3),
    }
    # The two next-best candidates do NOT clear (0.00391 and 0.0108 > 0.00238).
    assert _variant(5, 3) not in out["survivors"]
    assert _variant(5, 2) not in out["survivors"]


def test_widened_bonferroni_l1d_s3d_is_marginal_under_consistent_override():
    """HONESTY NUANCE: L1d/S3d clears on HAC p only.

    Its HAC p (0.00218) clears the widened 0.00238 threshold but its bootstrap p
    (0.0054) does NOT. They STRADDLE the threshold -> a consistently-applied
    bootstrap-override-on-disagreement would reject the bootstrap verdict, so
    L1d/S3d does NOT clear under both tests. It must be labelled MARGINAL, while
    L3d/S3d and L14d/S3d (clear under BOTH HAC and bootstrap) are robust.
    """
    inp = _verified_widened_inputs()
    out = run_stage2.apply_widened_bonferroni(
        inp["reported_p"], hac_p=inp["hac_p"], boot_p=inp["boot_p"]
    )
    robustness = out["robustness"]

    # L1d/S3d: HAC clears, bootstrap does not -> straddles -> MARGINAL.
    assert robustness[_variant(1, 3)] == "marginal"
    # L3d/S3d and L14d/S3d: clear under BOTH HAC and bootstrap -> robust.
    assert robustness[_variant(3, 3)] == "robust"
    assert robustness[_variant(14, 3)] == "robust"

    # The robust survivors clear under both tests; the marginal one does not.
    assert set(out["robust_survivors"]) == {_variant(3, 3), _variant(14, 3)}
    assert out["marginal_survivors"] == [_variant(1, 3)]

    # Explicit straddle determination for L1d/S3d at the widened threshold.
    thr = out["threshold"]
    assert inp["hac_p"][_variant(1, 3)] <= thr  # HAC clears
    assert inp["boot_p"][_variant(1, 3)] > thr  # bootstrap does NOT clear
