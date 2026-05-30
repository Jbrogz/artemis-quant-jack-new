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
from amom.config import LOOKBACKS_DAYS, PRIMARY_SKIP_DAYS  # noqa: E402


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
