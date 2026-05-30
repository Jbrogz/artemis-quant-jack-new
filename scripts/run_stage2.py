"""Stage-2 statistical-significance battery runner (Task T4, plan §T4, spec §2).

Answers the project's core question on the **in-sample slice only** (the OOS
window dated ``>= config.OOS_START`` is sealed for Stage 4): is the Artemis
momentum factor a true positive expected return, or a sample artifact?

For each of the 21 variants (7 lookbacks x 3 skips) the runner computes, on the
in-sample factor-return series:

  * the **naive** t-stat (reported but flagged biased — never the headline);
  * the **Newey-West HAC** t-stat at the ``maxlags_for`` bandwidth (the reported
    mean-return test, spec §2.2);
  * mean and annualized return;
  * the **Lo (2002) Sharpe** with its (autocorrelation-aware) SE + the flag;
  * the **spanning alpha** vs {equal-weighted market, small-minus-big size
    control (test-only)} with a HAC alpha t-stat (spec §2.4);
  * the **stationary-block-bootstrap** one-sided p (the bootstrap of record),
    the HAC p, the NW/bootstrap **disagreement** flag, and the reported p (the
    bootstrap on disagreement — spec §2.7);
  * **subsample** half/third signs + ``holds_sign`` deployment gate (spec §2.6);
  * the **Harvey-Liu-Zhu** tier and the **effective-n / power** label.

It then applies **Bonferroni to the pre-registered selection family** (7
lookbacks at ``PRIMARY_SKIP_DAYS=1``, ``m_select=7``), records the **total**
number of tests run (selection + diagnostics), and runs **DSR** and **PBO/CSCV**
across the variant grid. Outputs: ``data/stats/significance.parquet`` and the
human-readable ``docs/STAGE2_RESULTS.md`` (variants incl. failures, # tests,
survivors, the honest verdict).

Fully offline: the factor-return / holding-return / universe panels are read from
disk, and the size-control market-cap panel is reconstructed from the on-disk
Artemis cache — no API key is opened.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as _scs  # noqa: E402

from amom.config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    HOLDING_DAYS,
    LOOKBACKS_DAYS,
    OOS_START,
    PRIMARY_SKIP_DAYS,
    in_sample,
)
from amom.stats.bootstrap import disagrees, stationary_bootstrap_pvalue  # noqa: E402
from amom.stats.core import (  # noqa: E402
    bonferroni_correction,
    classify_tstat_hlz,
    hac_tstat,
)
from amom.stats.dsr import deflated_sharpe_ratio, sharpe_ratio  # noqa: E402
from amom.stats.pbo import probability_of_backtest_overfitting  # noqa: E402
from amom.stats.sharpe_se import (  # noqa: E402
    effective_n_and_power,
    lo_sharpe_se,
    maxlags_for,
)
from amom.stats.spanning import (  # noqa: E402
    build_market_return,
    build_size_control,
    spanning_alpha,
)
from amom.stats.subsample import sign_stability  # noqa: E402

# --- Annualization: each obs spans HOLDING_DAYS calendar days (spec §1.4). ---
DAYS_PER_YEAR = 365.0
PERIODS_PER_YEAR = DAYS_PER_YEAR / HOLDING_DAYS  # ≈ 12.17 for 30-day holds

# --- Stationary-bootstrap conventions (pinned, not tuned). ---
# block_size covers the holding-period serial overlap; for the non-overlapping
# 30-day series a small expected block length is conservative. seed makes the
# p-value deterministic (spec §2.7); reps large enough for a stable 0.05/7 p.
BOOT_REPS = 5000
BOOT_BLOCK_SIZE = 3
BOOT_SEED = 20260530

# --- Bonferroni family-wise level (spec §2.5). ---
ALPHA = 0.05

# --- PBO/CSCV split count (even, >= 2; Lopez de Prado 2015). ---
PBO_N_SPLITS = 8

FACTOR_PATH = DATA_DIR / "factor" / "factor_returns.parquet"
RETURNS_PATH = DATA_DIR / "returns" / "holding_returns.parquet"
UNIVERSE_PATH = DATA_DIR / "universe" / "universe_history.parquet"
OUTPUT_PARQUET = DATA_DIR / "stats" / "significance.parquet"
OUTPUT_MD = Path(__file__).resolve().parents[1] / "docs" / "STAGE2_RESULTS.md"


# ---------------------------------------------------------------------------
# Pre-registered selection family + Bonferroni (spec §2.5)
# ---------------------------------------------------------------------------

def selection_family(variants: list[str]) -> list[str]:
    """The pre-registered selection family: 7 lookbacks at ``PRIMARY_SKIP_DAYS``.

    Bonferroni operates on exactly this family (``m_select = 7``); skip {2,3} and
    any holding/breadth diagnostics are counted in the TOTAL test count but never
    relax the selection threshold (spec §2.5, Appendix A H1).
    """
    family = {f"momentum_L{lb}d_S{PRIMARY_SKIP_DAYS}d" for lb in LOOKBACKS_DAYS}
    return [v for v in variants if v in family]


def apply_bonferroni(
    family: list[str], pvalues_by_variant: dict[str, float], alpha: float = ALPHA
) -> dict:
    """Bonferroni-correct the pre-registered selection family (``m = 7``).

    Args:
        family: the selection-family variant names (length 7).
        pvalues_by_variant: reported p-value per family variant.
        alpha: family-wise level (default 0.05).

    Returns:
        Dict: ``m`` (== len(family) finite p-values), ``threshold`` (alpha/m),
        ``survivors`` (variants with p <= threshold), ``reject`` (aligned bools).
    """
    pvals = [pvalues_by_variant[v] for v in family]
    corr = bonferroni_correction(pvals, alpha=alpha)
    survivors = [v for v, keep in zip(family, corr["reject"]) if keep]
    return {
        "m": corr["m"],
        "threshold": corr["threshold"],
        "survivors": survivors,
        "reject": corr["reject"],
    }


# ---------------------------------------------------------------------------
# Per-variant battery (one row)
# ---------------------------------------------------------------------------

def _naive_tstat(returns: pd.Series) -> float:
    """One-sample naive (iid) t-stat for mean != 0 — flagged biased (spec §2.1)."""
    r = returns.dropna().to_numpy(dtype=float)
    n = r.size
    if n < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not sd > 0.0:
        return float("nan")
    return float(r.mean() / (sd / np.sqrt(n)))


def _hac_pvalue(tstat: float) -> float:
    """One-sided (mean > 0) normal-approx p-value for a HAC t-stat."""
    if not np.isfinite(tstat):
        return float("nan")
    return float(_scs.norm.sf(tstat))


def compute_variant_row(
    variant: str, returns: pd.Series, regressors: pd.DataFrame
) -> dict:
    """The full Stage-2 battery for one variant's in-sample factor-return series.

    Args:
        variant: variant name (e.g. ``momentum_L14d_S1d``).
        returns: in-sample per-rebalance factor returns, indexed by rebalance date.
        regressors: aligned spanning regressors (``market_return``,
            ``size_control``), indexed by the same dates.

    Returns:
        A flat dict of all battery columns for this variant (one result row).
        ``reported_p`` follows the **bootstrap** when HAC and bootstrap disagree
        across the Bonferroni-adjusted threshold (spec §2.7), else the HAC p.
    """
    r = returns.dropna()
    n_obs = int(r.size)

    bandwidth = maxlags_for(n_obs, holding_obs=1)
    hac = hac_tstat(r, bandwidth=bandwidth)
    naive_t = _naive_tstat(r)

    mean_return = float(r.mean()) if n_obs else float("nan")
    ann_return = mean_return * PERIODS_PER_YEAR

    sharpe, sharpe_se, autocorr_flag = lo_sharpe_se(r, periods_per_year=PERIODS_PER_YEAR)

    span = spanning_alpha(r, regressors, bandwidth=bandwidth)

    boot_p = stationary_bootstrap_pvalue(
        r, reps=BOOT_REPS, block_size=BOOT_BLOCK_SIZE, seed=BOOT_SEED
    )
    hac_p = _hac_pvalue(hac["tstat"])

    # Disagreement is judged at the Bonferroni-adjusted selection threshold; on
    # disagreement the bootstrap is the reported verdict (spec §2.7).
    adj_threshold = ALPHA / len(LOOKBACKS_DAYS)  # m_select == 7
    disagreement = disagrees(hac_p, boot_p, threshold=adj_threshold)
    reported_p = boot_p if disagreement else hac_p

    signs = sign_stability(r)
    power = effective_n_and_power(r, holding_obs=1)
    hlz_tier = classify_tstat_hlz(hac["tstat"])

    return {
        "variant": variant,
        "n_obs": n_obs,
        "naive_t": float(naive_t),
        "hac_t": float(hac["tstat"]),
        "hac_se": float(hac["se"]),
        "hac_maxlags": int(bandwidth),
        "mean_return": mean_return,
        "ann_return": ann_return,
        "sharpe": float(sharpe),
        "sharpe_se": float(sharpe_se),
        "autocorr_flag": bool(autocorr_flag),
        "spanning_alpha": float(span["alpha"]),
        "spanning_alpha_t": float(span["alpha_tstat"]),
        "spanning_beta_market": float(span["betas"].get("market_return", float("nan"))),
        "spanning_beta_size": float(span["betas"].get("size_control", float("nan"))),
        "spanning_n": int(span["n"]),
        "hac_p": float(hac_p),
        "bootstrap_p": float(boot_p),
        "disagreement": bool(disagreement),
        "reported_p": float(reported_p),
        "full_sign": int(signs["full_sign"]),
        "half_signs": str(signs["half_signs"]),
        "third_signs": str(signs["third_signs"]),
        "holds_sign": bool(signs["holds_sign"]),
        "hlz_tier": hlz_tier,
        "effective_n": int(power["effective_n"]),
        "power": float(power["power"]),
        "power_label": power["label"],
    }


def build_significance_table(
    factor_returns: pd.DataFrame, regressors: pd.DataFrame
) -> pd.DataFrame:
    """One battery row per variant (failures included), with selection flags.

    Args:
        factor_returns: long ``[variant, rebalance_date, factor_return, ...]``
            frame (already sliced to the in-sample window by the caller).
        regressors: spanning regressors indexed by rebalance date.

    Returns:
        DataFrame with one row per variant, an ``in_selection_family`` flag, and
        the Bonferroni-corrected ``survives_bonferroni`` flag on the selection
        family (always False for diagnostics).
    """
    variants = sorted(factor_returns["variant"].unique())
    family = set(selection_family(variants))

    rows = []
    for v in variants:
        sub = factor_returns[factor_returns["variant"] == v]
        series = sub.set_index("rebalance_date")["factor_return"].sort_index()
        row = compute_variant_row(v, series, regressors)
        row["in_selection_family"] = v in family
        rows.append(row)
    table = pd.DataFrame(rows)

    # Bonferroni on the pre-registered selection family (m == 7).
    fam_list = selection_family(variants)
    reported_p = dict(zip(table["variant"], table["reported_p"]))
    bonf = apply_bonferroni(fam_list, {v: reported_p[v] for v in fam_list})
    survivors = set(bonf["survivors"])
    table["survives_bonferroni"] = table["variant"].isin(survivors)
    return table


# ---------------------------------------------------------------------------
# Offline market-cap panel (from the on-disk Artemis cache; no API key)
# ---------------------------------------------------------------------------

def build_mc_panel(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Reconstruct the long ``[date, symbol, mc]`` market-cap panel from cache.

    The Stage-1 universe build wrote the full Artemis market frames to the parquet
    cache; the MC metric is recovered here so the size-control regressor (spec
    §2.4) is built fully offline. Dates are normalized to midnight to match the
    factor/holding-return panels' ``<= / >`` comparison convention.
    """
    frames = []
    for f in sorted(glob.glob(str(cache_dir / "*.parquet"))):
        df = pd.read_parquet(f)
        if "metric" not in df.columns:
            continue
        mc = df[(df["metric"] == "MC") & df["value"].notna()]
        if len(mc):
            frames.append(mc[["date", "symbol", "value"]])
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "mc"])
    panel = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "symbol"])
        .rename(columns={"value": "mc"})
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Overfitting controls across the grid (DSR, PBO/CSCV)
# ---------------------------------------------------------------------------

def _wide_factor_matrix(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Inner-joined (dates x variants) factor-return matrix (common dates only)."""
    wide = factor_returns.pivot_table(
        index="rebalance_date", columns="variant", values="factor_return"
    ).sort_index()
    return wide.dropna(axis=0, how="any")


def run_overfitting_controls(factor_returns: pd.DataFrame) -> dict:
    """DSR (per variant) + a single grid PBO/CSCV (spec §2 overfitting controls).

    DSR deflates each variant's Sharpe by the expected maximum across the full
    grid of trial Sharpes; PBO/CSCV reports the probability that the in-sample
    best variant fails to stay above-median out-of-sample within the in-sample
    window. Both quantify selection-induced overfit.
    """
    wide = _wide_factor_matrix(factor_returns)
    variants = list(wide.columns)
    trial_sharpes = np.array([sharpe_ratio(wide[v].to_numpy()) for v in variants])

    dsr_by_variant: dict[str, float] = {}
    for v in variants:
        series = wide[v].to_numpy(dtype=float)
        if series.size < 2 or series.std(ddof=1) == 0.0:
            dsr_by_variant[v] = float("nan")
            continue
        try:
            dsr_by_variant[v] = float(deflated_sharpe_ratio(series, trial_sharpes))
        except ValueError:
            dsr_by_variant[v] = float("nan")

    try:
        pbo = float(
            probability_of_backtest_overfitting(wide.to_numpy(), n_splits=PBO_N_SPLITS)
        )
    except ValueError as exc:
        pbo = float("nan")
        print(f"  [PBO] skipped: {exc}")

    return {"dsr_by_variant": dsr_by_variant, "pbo": pbo, "n_common_dates": len(wide)}


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------

def overall_verdict(table: pd.DataFrame) -> str:
    """The faithful one-line verdict over the selection family (spec §2 honesty).

    Reports whatever the battery shows, never massaged toward significance:
      * ``significant``  -- a Bonferroni survivor that holds its sign and is powered;
      * ``suggestive``   -- a selection variant is HLZ-suggestive (2<t<3) but no survivor;
      * ``inconclusive`` -- the selection family is underpowered (effective n too low);
      * ``null``         -- powered, but no survivor and nothing even suggestive.
    """
    fam = table[table["in_selection_family"]]
    if fam.empty:
        return "inconclusive"
    survivors = fam[fam["survives_bonferroni"] & fam["holds_sign"] & (fam["power_label"] == "powered")]
    if len(survivors) > 0:
        return "significant"
    if (fam["power_label"] != "powered").all():
        return "inconclusive (underpowered)"
    if (fam["hlz_tier"] == "suggestive").any():
        return "suggestive"
    return "null"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_markdown(
    table: pd.DataFrame,
    bonf: dict,
    total_tests: int,
    overfit: dict,
    verdict: str,
    path: Path = OUTPUT_MD,
) -> None:
    """Write the human-readable Stage-2 results table (failures included)."""
    lines: list[str] = []
    lines.append("# Stage 2 — Statistical Significance Battery Results")
    lines.append("")
    lines.append(
        f"_In-sample slice only (rebalance_date < OOS_START = "
        f"{OOS_START.date()}; the OOS window is sealed for Stage 4)._"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Honest verdict (selection family): `{verdict}`.**")
    lines.append(
        f"- Pre-registered selection family: **{bonf['m']} tests** "
        f"(7 lookbacks at skip={PRIMARY_SKIP_DAYS}); Bonferroni threshold "
        f"= 0.05 / {bonf['m']} = {_fmt(bonf['threshold'], 5)}."
    )
    lines.append(f"- **Total tests run (selection + diagnostics): {total_tests}.**")
    survivors = bonf["survivors"]
    lines.append(
        "- Bonferroni survivors: "
        + (", ".join(f"`{s}`" for s in survivors) if survivors else "**none**")
        + "."
    )
    lines.append(
        f"- Grid PBO/CSCV (probability of backtest overfitting): "
        f"{_fmt(overfit['pbo'], 3)} "
        f"(over {overfit['n_common_dates']} common in-sample dates)."
    )
    lines.append("")
    lines.append(
        "The **HAC** t-stat (Newey-West, autocorrelation-robust) is the reported "
        "mean-return test; the **naive** t-stat is shown but is biased (overstates "
        "significance) and is never the headline. Underpowered variants are labelled "
        "**inconclusive (underpowered)**, distinct from insignificant. On a "
        "Newey-West / bootstrap disagreement the **bootstrap** is the reported verdict."
    )
    lines.append("")

    lines.append("## Selection family (skip = 1) — the deployment candidates")
    lines.append("")
    _emit_table(lines, table[table["in_selection_family"]], overfit)
    lines.append("")
    lines.append("## Diagnostics (skip {2,3}) — reported, not selected")
    lines.append("")
    _emit_table(lines, table[~table["in_selection_family"]], overfit)
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `holds_sign` is the Stage-2.6 deployment gate: a sign-flip across "
        "halves/thirds disqualifies a variant from deployment regardless of t-stat."
    )
    lines.append(
        "- The spanning alpha is the factor mean after partialling out "
        "{equal-weighted market return, small-minus-big size control}; the size "
        "control is a TEST-ONLY regressor (never deployed)."
    )
    lines.append(
        "- DSR (deflated Sharpe) deflates each variant's Sharpe by the expected "
        "maximum across the full trial grid; PBO > 0.5 indicates selection overfitting."
    )
    lines.append(
        "- Survivorship: the dead/collapsed coins remain in the underlying series; "
        "this battery does not re-filter the universe."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_table(lines: list[str], sub: pd.DataFrame, overfit: dict) -> None:
    header = (
        "| variant | n | naive t | HAC t | HLZ | ann ret | Sharpe (SE) | "
        "autocorr | span α | span α t | HAC p | boot p | disagree | "
        "holds sign | power | DSR | survives |"
    )
    sep = "|" + "|".join(["---"] * 17) + "|"
    lines.append(header)
    lines.append(sep)
    dsr = overfit["dsr_by_variant"]
    for _, r in sub.sort_values("variant").iterrows():
        lines.append(
            "| `{v}` | {n} | {nt} | {ht} | {hlz} | {ann} | {sh} ({se}) | {ac} | "
            "{sa} | {sat} | {hp} | {bp} | {dis} | {hs} | {pl} | {d} | {surv} |".format(
                v=r["variant"],
                n=int(r["n_obs"]),
                nt=_fmt(r["naive_t"], 3),
                ht=_fmt(r["hac_t"], 3),
                hlz=r["hlz_tier"],
                ann=_fmt(r["ann_return"], 4),
                sh=_fmt(r["sharpe"], 3),
                se=_fmt(r["sharpe_se"], 3),
                ac="yes" if r["autocorr_flag"] else "no",
                sa=_fmt(r["spanning_alpha"], 5),
                sat=_fmt(r["spanning_alpha_t"], 3),
                hp=_fmt(r["hac_p"], 4),
                bp=_fmt(r["bootstrap_p"], 4),
                dis="yes" if r["disagreement"] else "no",
                hs="yes" if r["holds_sign"] else "**no**",
                pl=r["power_label"],
                d=_fmt(dsr.get(r["variant"], float("nan")), 3),
                surv="**yes**" if r["survives_bonferroni"] else "no",
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if not FACTOR_PATH.exists():
        print(f"ERROR: {FACTOR_PATH} missing. Run scripts/build_factor_returns.py first.")
        return 1

    factor_all = pd.read_parquet(FACTOR_PATH)
    factor_all["rebalance_date"] = pd.to_datetime(factor_all["rebalance_date"]).dt.normalize()

    # OOS DISCIPLINE: slice to the in-sample window BEFORE any statistic. No row
    # dated >= OOS_START is consumed anywhere below (spec §2.8).
    factor_is = in_sample(factor_all)
    assert (factor_is["rebalance_date"] < OOS_START).all(), "OOS leak: a row >= OOS_START survived in_sample()"
    n_variants = factor_is["variant"].nunique()
    total_tests = n_variants  # one mean-return test per variant (selection + diagnostics)

    print("=" * 78)
    print("  STAGE 2 — SIGNIFICANCE BATTERY (in-sample only; OOS sealed)")
    print("=" * 78)
    print(f"  OOS_START (sealed)          : {OOS_START.date()}")
    print(f"  variants                    : {n_variants}")
    print(f"  in-sample obs / variant     : {len(factor_is) // max(n_variants, 1)}")
    print(f"  OOS obs / variant (sealed)  : {len(factor_all[factor_all['rebalance_date'] >= OOS_START]) // max(n_variants, 1)}")
    print(f"  total tests (recorded)      : {total_tests}")
    print()

    # --- Build the spanning regressors (in-sample only). ---
    holding = pd.read_parquet(RETURNS_PATH)
    holding["date"] = pd.to_datetime(holding["date"]).dt.normalize()
    # OOS BOUNDARY: slice the holding panel to date < OOS_START before passing it
    # to build_market_return / build_size_control. The last in-sample rebalance
    # window uses the first OOS rebalance date as its *closing* boundary date
    # (next_r), so without this slice _window_return would consume daily holding-
    # return rows dated in [last_IS_rebalance, first_OOS_rebalance) — i.e. OOS-
    # dated daily rows. Slicing here closes that spanning boundary nuance without
    # changing any in-sample factor return or regressor value.
    holding = holding[holding["date"] < OOS_START]
    universe = pd.read_parquet(UNIVERSE_PATH)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    mc_panel = build_mc_panel()
    print(f"  MC panel (offline cache)    : {mc_panel.shape[0]:,} rows, {mc_panel['symbol'].nunique()} symbols")

    # Rebalance dates for the in-sample window plus the first OOS rebalance date
    # as the closing boundary for the final IS window (next_r in the (r, next_r]
    # convention). No OOS factor return is read; daily holding rows are pre-sliced.
    is_dates = sorted(factor_is["rebalance_date"].unique())
    all_dates = sorted(factor_all["rebalance_date"].unique())
    closing = [d for d in all_dates if d >= OOS_START]
    window_dates = pd.DatetimeIndex(is_dates + (closing[:1] if closing else []))

    market = build_market_return(holding, universe, window_dates)
    size = build_size_control(holding, universe, mc_panel, window_dates)
    regressors = pd.concat([market, size], axis=1, join="inner")
    # Restrict regressors to in-sample dates only (defensive; the closing boundary
    # date is never an index row because build_* skip the final window).
    regressors = regressors.loc[regressors.index < OOS_START]
    print(f"  spanning regressors         : {regressors.shape[0]} dates x {regressors.shape[1]} cols")
    print()

    table = build_significance_table(factor_is, regressors)

    fam_list = selection_family(sorted(factor_is["variant"].unique()))
    reported_p = dict(zip(table["variant"], table["reported_p"]))
    bonf = apply_bonferroni(fam_list, {v: reported_p[v] for v in fam_list})

    overfit = run_overfitting_controls(factor_is)
    table["dsr"] = table["variant"].map(overfit["dsr_by_variant"])

    verdict = overall_verdict(table)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_PARQUET, index=False)
    write_markdown(table, bonf, total_tests, overfit, verdict)

    # --- Console summary (selection family). ---
    fam = table[table["in_selection_family"]].sort_values("variant")
    print(f"  {'variant':<20} {'HAC t':>7} {'HLZ':>15} {'boot p':>8} {'holds':>6} {'power':>26} {'DSR':>6}")
    print("  " + "-" * 96)
    for _, r in fam.iterrows():
        print(
            f"  {r['variant']:<20} {r['hac_t']:>7.3f} {r['hlz_tier']:>15} "
            f"{r['bootstrap_p']:>8.4f} {'yes' if r['holds_sign'] else 'NO':>6} "
            f"{r['power_label']:>26} {_fmt(r['dsr'], 3):>6}"
        )
    print()
    print(f"  Bonferroni m                : {bonf['m']}  (threshold {bonf['threshold']:.5f})")
    print(f"  survivors                   : {bonf['survivors'] or 'none'}")
    print(f"  grid PBO/CSCV               : {_fmt(overfit['pbo'], 3)}")
    print(f"  HONEST VERDICT              : {verdict}")
    print()
    print(f"  wrote {OUTPUT_PARQUET}")
    print(f"  wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
