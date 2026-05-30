"""
build_report.py
---------------
Deterministically assembles the Stage-5 research-report PDF for the Artemis
momentum study:

    docs/report/Artemis_Momentum_Report.pdf

Pure-Python via reportlab (no system-lib dependencies). It embeds the RPT1
figures (with their takeaway captions), the full Stage-2 variant table
(INCLUDING failures, all 21 variants), the Stage-4 gross-vs-net and
IS-vs-OOS tables plus robustness, and the required-disclosures appendix.

ALL numbers are pulled VERBATIM from the committed source-of-truth artifacts:
  docs/STAGE2_RESULTS.md, docs/STAGE4_RESULTS.md, docs/AUDIT.md
  data/stats/significance.parquet
Nothing is re-derived or invented.

Run:  uv run python scripts/build_report.py
"""

from __future__ import annotations

import pathlib

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "report" / "figures"
DATA_STATS = ROOT / "data" / "stats"
OUT_PDF = ROOT / "docs" / "report" / "Artemis_Momentum_Report.pdf"

# Figure takeaway captions — copied VERBATIM from scripts/build_report_figures.py.
FIG_CAPTIONS = {
    "fig1_cumulative_pnl.png": (
        "Fig 1 — Cumulative P&L (L5d gross & net, L28d net). Takeaway: L5d net grows "
        "steadily in-sample but is a regime-sensitive path; L28d net decays to near-zero "
        "— the canonical 4-week lookback does not survive costs."
    ),
    "fig2_drawdown.png": (
        "Fig 2 — Drawdown (L5d net, primary). Takeaway: In-sample max drawdown of -48.5% "
        "dwarfs the OOS -25.2%, consistent with the sign-instability finding; drawdown is "
        "material relative to returns at both horizons."
    ),
    "fig3_rolling_sharpe.png": (
        "Fig 3 — Rolling 6-month Sharpe (L5d net). Takeaway: Rolling Sharpe oscillates "
        "widely and dips sharply negative in multiple sub-windows, confirming that any "
        "positive in-sample/OOS Sharpe is driven by episodic regimes rather than a stable factor."
    ),
    "fig4_variant_sharpe_bar.png": (
        "Fig 4 — Gross Sharpe by variant (selection family vs diagnostics). Takeaway: No "
        "selection-family variant clears the Bonferroni threshold on Newey-West; the "
        "eye-catching high Sharpes (L3d/S3, L14d/S3) are in the diagnostic zone and were "
        "never eligible for deployment."
    ),
    "fig5_variant_correlation.png": (
        "Fig 5 — Factor-return correlation (selection family). Takeaway: Short lookbacks "
        "(L1d-L5d) cluster together with high mutual correlation, while longer lookbacks "
        "(L14d-L56d) form a separate cluster; the family is far from orthogonal, reducing "
        "the independent-test count below 7."
    ),
}

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_ss = getSampleStyleSheet()

H_TITLE = ParagraphStyle(
    "HTitle", parent=_ss["Title"], fontSize=22, leading=26, spaceAfter=6,
    textColor=colors.HexColor("#1a1a1a"),
)
H_SUB = ParagraphStyle(
    "HSub", parent=_ss["Normal"], fontSize=11, leading=15, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceAfter=14,
)
H1 = ParagraphStyle(
    "H1", parent=_ss["Heading1"], fontSize=15, leading=19, spaceBefore=14,
    spaceAfter=6, textColor=colors.HexColor("#0b3d91"),
)
H2 = ParagraphStyle(
    "H2", parent=_ss["Heading2"], fontSize=12, leading=16, spaceBefore=8,
    spaceAfter=4, textColor=colors.HexColor("#1a1a1a"),
)
BODY = ParagraphStyle(
    "Body", parent=_ss["Normal"], fontSize=9.5, leading=13.5, alignment=TA_LEFT,
    spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=4,
)
CAPTION = ParagraphStyle(
    "Caption", parent=_ss["Normal"], fontSize=8.5, leading=11.5,
    textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=12,
    alignment=TA_LEFT, fontName="Helvetica-Oblique",
)
VERDICT = ParagraphStyle(
    "Verdict", parent=_ss["Normal"], fontSize=13, leading=17, alignment=TA_CENTER,
    textColor=colors.white, fontName="Helvetica-Bold",
)
TBL_CAP = ParagraphStyle(
    "TblCap", parent=_ss["Normal"], fontSize=8, leading=11,
    textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=10,
    fontName="Helvetica-Oblique",
)
CELL = ParagraphStyle("Cell", parent=_ss["Normal"], fontSize=6.6, leading=8)
CELL_H = ParagraphStyle(
    "CellH", parent=_ss["Normal"], fontSize=6.6, leading=8,
    textColor=colors.white, fontName="Helvetica-Bold",
)

GRID = colors.HexColor("#cccccc")
HEAD_BG = colors.HexColor("#0b3d91")
ZEBRA = colors.HexColor("#f2f5fb")
FLAG_BG = colors.HexColor("#fdecec")


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", BULLET)


# ---------------------------------------------------------------------------
# Verdict banner
# ---------------------------------------------------------------------------
def verdict_banner() -> Table:
    t = Table(
        [[Paragraph("VERDICT: NO-DEPLOY", VERDICT)]],
        colWidths=[6.7 * inch],
    )
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#b3261e")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    return t


# ---------------------------------------------------------------------------
# Generic styled table builder
# ---------------------------------------------------------------------------
def styled_table(header, rows, col_widths, flag_rows=None):
    """flag_rows: set of data-row indices (0-based, excl. header) to highlight."""
    flag_rows = flag_rows or set()
    data = [[Paragraph(str(h), CELL_H) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        ridx = i - 1
        if ridx in flag_rows:
            style.append(("BACKGROUND", (0, i), (-1, i), FLAG_BG))
        elif i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Stage-2 full variant table (all 21, from the parquet — verbatim)
# ---------------------------------------------------------------------------
def stage2_table():
    sig = pd.read_parquet(DATA_STATS / "significance.parquet").copy()

    def short(v):
        return v.replace("momentum_", "").replace("_S", "/S")

    # Order: selection family (skip=1) first, then diagnostics; within each, by variant.
    sig["fam"] = sig["in_selection_family"]
    sig = sig.sort_values(["fam", "variant"], ascending=[False, True]).reset_index(drop=True)

    header = [
        "variant", "fam", "n", "naive t", "HAC t", "HAC p", "boot p", "HLZ",
        "ann ret", "Sharpe (SE)", "span α", "span α t",
        "holds sign", "DSR", "survives",
    ]
    rows = []
    flag = set()
    for i, r in enumerate(sig.itertuples()):
        rows.append([
            short(r.variant),
            "sel" if r.in_selection_family else "diag",
            int(r.n_obs),
            f"{r.naive_t:.3f}",
            f"{r.hac_t:.3f}",
            f"{r.hac_p:.4f}",
            f"{r.bootstrap_p:.4f}",
            r.hlz_tier.replace("not_significant", "n.s.").replace("significant", "sig"),
            f"{r.ann_return:.4f}",
            f"{r.sharpe:.3f} ({r.sharpe_se:.3f})",
            f"{r.spanning_alpha:.5f}",
            f"{r.spanning_alpha_t:.3f}",
            "yes" if r.holds_sign else "NO",
            f"{r.dsr:.3f}",
            "YES" if r.survives_bonferroni else "no",
        ])
        if r.survives_bonferroni:
            flag.add(i)

    widths = [
        0.62, 0.30, 0.20, 0.42, 0.42, 0.42, 0.42, 0.34, 0.46,
        0.74, 0.50, 0.44, 0.42, 0.36, 0.44,
    ]
    widths = [w * inch for w in widths]
    return styled_table(header, rows, widths, flag_rows=flag)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
def build():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Artemis Momentum Factor — Research Report",
        author="Artemis Quant",
    )
    fig_w = 6.9 * inch
    story = []

    # ---------------- Cover / Section 1: Conclusion first ----------------
    story += [
        P("Artemis Momentum Factor", H_TITLE),
        P("Research Report &mdash; Stage 5 Deliverable", H_SUB),
        P("Date: 2026-05-30 &nbsp;|&nbsp; Spot long/short, Artemis-sourced universe "
          "&nbsp;|&nbsp; Survivorship-corrected, cost-aware, sealed out-of-sample", H_SUB),
        verdict_banner(),
        Spacer(1, 12),
        P("1. Conclusion first", H1),
        P("We tested cross-sectional momentum on an Artemis-sourced crypto spot universe and "
          "find <b>no deployable factor</b>. The recommendation is: <b>do not deploy</b>.", BODY),
        bullet("<b>What we tested.</b> Twenty-one momentum variants &mdash; a pre-registered "
               "selection family of <b>7 lookbacks</b> (1, 3, 5, 7, 14, 28, 56 days) at the "
               "convention skip of 1 day, plus 14 skip-{2,3} <b>diagnostics</b> reported for "
               "transparency but never eligible for deployment. The full Stage-2 battery "
               "(Newey-West HAC, Lo Sharpe SE, spanning vs market + size control, stationary "
               "block bootstrap, Bonferroni / HLZ, subsample sign-stability, DSR and PBO) ran on "
               "the in-sample slice; a sealed OOS window was opened exactly once."),
        bullet("<b>What held up: nothing, at the family level.</b> No selection-family variant is "
               "significant on Newey-West against the Bonferroni threshold (0.05/7 = 0.00714). The "
               "strongest, <font face='Courier'>momentum_L5d_S1d</font>, is HLZ-&ldquo;suggestive&rdquo; "
               "with HAC t = 2.40 and HAC p = 0.0081 &mdash; <i>above</i> the threshold. It clears "
               "Bonferroni <b>only</b> via the bootstrap-override rule (bootstrap p = 0.0058), and "
               "is then <b>disqualified</b> by the Stage-2.6 deployment gate for flipping sign "
               "across subsamples. DSR = 0.46; grid PBO = 0.114."),
        bullet("<b>Deployed-candidate net-of-cost OOS Sharpe.</b> Characterized for completeness, "
               "<font face='Courier'>momentum_L5d_S1d</font> posts OOS <b>net Sharpe 0.756</b> "
               "(gross 0.897) and IS net Sharpe 0.664 (gross 0.789). The OOS Sharpe did <b>not</b> "
               "collapse &mdash; but this is a <b>single-regime artifact</b> on ~30 overlapping "
               "observations spent exactly once (the 2024 crypto bull), not a stable edge. The "
               "academic 4-week comparator <font face='Courier'>momentum_L28d_S1d</font> "
               "<b>works only gross of costs</b>: gross Sharpe 0.073 turns to <b>net -0.031</b> "
               "in-sample and <b>net -0.270</b> out-of-sample."),
        bullet("<b>Recommendation.</b> Do not deploy momentum on the Artemis spot universe. The "
               "suggestive ~5-day signal is a <i>research lead</i>, not a strategy. Capacity "
               "(~$36M) is comfortable and is <b>not</b> the binding constraint; the "
               "disqualification rests on multiple-testing failure, subsample sign-instability, "
               "regime dependence, lookback fragility, and disclosed data limits."),
        PageBreak(),
    ]

    # ---------------- Section 2: Methodology ----------------
    story += [
        P("2. Methodology (brief)", H1),
        P("<b>Data &mdash; Artemis only, survivorship-corrected.</b> The universe is enumerated "
          "programmatically from the Artemis <font face='Courier'>/asset</font> catalog (1,013 "
          "entries; <b>846 with any price data</b>), on a daily point-in-time grid spanning "
          "<b>2018-01-01 to 2026-05-30</b> (2,598,912 panel rows). Eligibility filters "
          "(&ge;90 days history, &ge;50% observation density, $10M market-cap floor, trailing-30d "
          "median 24H volume above threshold, staleness check, stablecoin and wrapped-token "
          "exclusion) are rebuilt as-of each date. Dead and collapsed coins are <b>kept</b>: "
          "<b>238 of 846 (28%)</b> exhibit a terminal &gt;90% drawdown (15 delisted + 223 "
          "still-printing zombies), and every crash return is carried into the P&amp;L (the "
          "LUNA/<font face='Courier'>lunc</font> -99.99% collapse included). Recycled tickers are "
          "split into distinct synthetic assets. Residual survivorship &mdash; coins purged from "
          "Artemis before the query &mdash; is <b>unestimable and disclosed</b>; reported factor "
          "returns are <b>upper bounds</b>.", BODY),
        P("<b>Returns.</b> Holding return is the simple spot price return &mdash; Artemis exposes "
          "no perpetual funding, so <b>no funding term</b> is modeled (disclosed). Simple returns "
          "aggregate across coins; log returns compound through time; the two are never mixed.", BODY),
        P("<b>Momentum construction.</b> Signal = trailing log-sum return over a fixed lookback, "
          "shifted by the skip. The <b>7-lookback grid is fixed in advance</b> (academic "
          "14/28/56-day plus crypto-short 1/3/5/7-day); skip is fixed by convention at <b>1 day</b>. "
          "The eligible universe is sorted into <b>quintiles</b> (long top 20% / short bottom 20%, "
          "equal-weight, dollar-neutral). Signal uses data through close of t; position is entered "
          "at the <b>close of t+1</b> &mdash; a mandatory, unit-tested one-period lag (no "
          "look-ahead).", BODY),
        P("<b>Stage-2 significance battery.</b> Per variant: naive t-stat (reported but flagged "
          "biased, never the headline); <b>Newey-West HAC t-stat</b> (the reported mean-return "
          "test, bandwidth covering holding-period overlap); Sharpe with the <b>Lo (2002) SE</b> "
          "(autocorrelation-corrected when Ljung-Box fires); a <b>spanning regression</b> on "
          "{equal-weight market, small-minus-big size control} reporting the HAC-t alpha (size "
          "control is TEST-ONLY, never deployed); <b>Bonferroni</b> on the family of 7 and HLZ "
          "tiers; a <b>stationary block bootstrap</b> (arch) as the bootstrap of record, which "
          "overrides Newey-West on a cross-threshold disagreement; <b>subsample sign-stability</b> "
          "(a sign-flip disqualifies regardless of t-stat); and <b>DSR</b> plus <b>PBO/CSCV</b> for "
          "selection overfit.", BODY),
        P("<b>Sealed OOS and cost-aware backtest.</b> The most recent ~30% of rebalance dates "
          "(<font face='Courier'>OOS_START = 2023-12-02</font>) were reserved before any statistic "
          "was computed and opened <b>exactly once</b> under a single-use guard. The backtest "
          "applies a 10 bps spot taker fee per side plus size-scaled tiered slippage, executes at "
          "the t+1 close, and reports the full net metric set plus a capacity estimate.", BODY),
        PageBreak(),
    ]

    # ---------------- Section 3: Factor-by-factor + Fig 4 & 5 ----------------
    story += [
        P("3. Factor-by-factor (variant) results", H1),
        P("None of the selection family (skip = 1) is significant on the Newey-West HAC test at "
          "the Bonferroni threshold (0.05/7 = 0.00714). The diagnostics (skip = 2,3) are reported "
          "separately and were <b>never eligible for deployment</b>; the striking diagnostic "
          "Sharpes (L3d/S3d HAC t &asymp; 5.0, L14d/S3d HAC t &asymp; 3.9) are <b>not</b> in the "
          "selection family. The full numeric table for all 21 variants is in the Appendix (7.3).", BODY),
        bullet("<font face='Courier'>L5d/S1d</font> (strongest): HAC t = 2.403, HAC p = 0.0081 "
               "(&gt; 0.00714), bootstrap p = 0.0058, suggestive, span &alpha; t = 2.206, "
               "DSR = 0.460. <b>Survives Bonferroni only via the bootstrap override, and holds_sign "
               "= NO &mdash; DISQUALIFIED.</b>"),
        bullet("<font face='Courier'>L7d/S1d</font>: HAC t = 1.887, p = 0.0296 &mdash; not "
               "significant. &nbsp; <font face='Courier'>L14d/S1d</font>: HAC t = 1.945, p = 0.0259 "
               "&mdash; not significant. &nbsp; <font face='Courier'>L3d/S1d</font>: HAC t = 1.533, "
               "sign-unstable."),
        bullet("<font face='Courier'>L28d/S1d</font> (academic 4-week): HAC t = 0.955, p = 0.1698 "
               "&mdash; not significant. &nbsp; <font face='Courier'>L56d/S1d</font>: HAC t = 0.756 "
               "&mdash; not significant. &nbsp; <font face='Courier'>L1d/S1d</font>: HAC t = -1.300 "
               "(negative; reversal), sign-unstable."),
        bullet("Grid overfit controls: <b>PBO/CSCV = 0.114</b> (below the 0.5 threshold) but on a "
               "correlated family (Fig 5), so the effective independent-test count is below 7."),
        Spacer(1, 6),
    ]
    story += _figure_block("fig4_variant_sharpe_bar.png", fig_w)
    story += _figure_block("fig5_variant_correlation.png", 5.0 * inch)
    story += [PageBreak()]

    # ---------------- Section 4: Deployed-candidate characterization + Fig 1 ----------------
    story += [
        P("4. Deployed-candidate characterization (Stage-4 backtest)", H1),
        P("For completeness we characterized the strongest candidate, "
          "<font face='Courier'>momentum_L5d_S1d</font>, net of realistic spot costs. <b>This "
          "characterizes the candidate; it does not rescue it</b> &mdash; the Stage-2 "
          "disqualification stands.", BODY),
        P("Gross vs net, in-sample vs out-of-sample (primary L5d/S1d):", H2),
    ]
    seg_header = ["segment", "gross Sharpe", "net Sharpe", "net ann ret", "net ann vol",
                  "Sortino", "max DD", "Calmar", "hit rate", "ann turnover"]
    primary_rows = [
        ["in-sample", "0.789", "0.664", "0.2490", "0.3752", "0.756", "-0.4851", "0.513", "0.574", "24.31"],
        ["out-of-sample", "0.897", "0.756", "0.2469", "0.3266", "1.031", "-0.2516", "0.981", "0.567", "28.22"],
    ]
    seg_w = [0.95, 0.66, 0.62, 0.66, 0.66, 0.55, 0.58, 0.52, 0.56, 0.72]
    seg_w = [w * inch for w in seg_w]
    story += [
        styled_table(seg_header, primary_rows, seg_w),
        P("Gross and net are shown side by side; net is never flattered above gross. IS and OOS "
          "are shown side by side; the OOS window was opened exactly once.", TBL_CAP),
        bullet("<b>IS &minus; OOS net Sharpe gap = -0.092.</b> The OOS figure exceeds in-sample, but "
               "this is a spent-once, ~30-observation, single-regime (2024-bull) artifact &mdash; "
               "not evidence of a stable edge."),
        bullet("<b>Capacity &asymp; $36,044,481</b> &mdash; the AUM at which size-scaled slippage on "
               "the actual per-rebalance traded order (~2.0&times; summed one-way turnover) erases "
               "the gross edge (per-rebalance gross edge 0.02436). Comfortably above a $1M book, so "
               "<b>capacity is not the binding constraint</b>."),
        bullet("<b>Turnover</b> ~24 (IS) / ~28 (OOS) annualized &mdash; high, consistent with a "
               "5-day signal."),
        Spacer(1, 6),
    ]
    story += _figure_block("fig1_cumulative_pnl.png", fig_w)
    story += [PageBreak()]

    # ---------------- Section 5: Robustness + Fig 2 & 3 ----------------
    story += [
        P("5. Robustness", H1),
        P("<b>2&times; costs (in-sample, primary):</b> net Sharpe degrades from <b>0.664 &rarr; "
          "0.552</b> (gross unchanged at 0.789; max DD widens to -0.5219). The edge thins "
          "materially under a doubled cost assumption.", BODY),
        P("&plusmn;50% lookback (in-sample, net) &mdash; the construction is <b>fragile</b> to its "
          "one free parameter:", H2),
    ]
    look_header = ["variant", "net Sharpe", "net ann ret", "note"]
    look_rows = [
        ["momentum_L5d_S1d (chosen)", "0.664", "0.2490", "chosen lookback = 5d"],
        ["momentum_L2d_S1d (-50%)", "-0.328", "-0.1155", "lookback halved — net-negative"],
        ["momentum_L8d_S1d (+50%)", "0.621", "0.2645", "lookback up 50%"],
    ]
    look_w = [w * inch for w in [2.3, 1.0, 1.0, 2.4]]
    story += [
        styled_table(look_header, look_rows, look_w, flag_rows={1}),
        P("A -50% lookback flips the candidate net-negative &mdash; not robust.", TBL_CAP),
        P("Regime breakdown (in-sample, net mean return per regime):", H2),
    ]
    reg_header = ["regime", "n", "mean net return"]
    reg_rows = [["bull", "17", "-0.00624"], ["bear", "28", "0.02195"], ["chop", "22", "0.03453"]]
    reg_w = [w * inch for w in [1.4, 0.8, 1.6]]
    story += [
        styled_table(reg_header, reg_rows, reg_w),
        P("<b>Caveat (do not over-read):</b> this is a full-sample, <b>descriptive</b> partition, "
          "<b>not</b> a walk-forward signal &mdash; it could not have been traded ex-ante. The "
          "<font face='Courier'>chop</font> bucket is the top-|market-move| tercile, which on this "
          "sample skews toward large up moves, so the apparent &lsquo;negative in bull / positive "
          "in bear&rsquo; contrast is <b>overstated</b>. The disqualifying evidence is the Stage-2 "
          "&sect;2.6 sign-instability itself, not this descriptive split.", BODY),
        P("<b>One-shot OOS, single-regime character.</b> The OOS window is 30 overlapping-regime "
          "observations spent exactly once; a single favorable stretch (the 2024 crypto bull) "
          "carries the positive OOS Sharpe. A near-zero or negative OOS Sharpe would have been "
          "reported as overfitting; the positive figure is reported as-is and interpreted as regime "
          "exposure, consistent with the sign-flip disqualification.", BODY),
        Spacer(1, 6),
    ]
    story += _figure_block("fig2_drawdown.png", fig_w)
    story += _figure_block("fig3_rolling_sharpe.png", fig_w)
    story += [PageBreak()]

    # ---------------- Section 6: Recommendation ----------------
    story += [
        P("6. Recommendation", H1),
        P("<b>Do not deploy.</b> Target allocation to a momentum sleeve on the Artemis spot "
          "universe: <b>zero</b>.", BODY),
        bullet("<b>Expected Sharpe (range):</b> for the strongest candidate, net Sharpe sits in "
               "roughly <b>0.66&ndash;0.76</b> in the realized sample, but with <b>no statistically "
               "reliable edge</b> and a true expected Sharpe indistinguishable from zero once "
               "multiple testing, sign-instability, and the single-regime OOS are accounted for. We "
               "do not represent this as a deployable expectation."),
        bullet("<b>The suggestive ~5-day signal is a research lead, not a strategy.</b> It would "
               "warrant follow-up only with (a) more independent observations, (b) a cleaner, "
               "lower-turnover construction, and (c) a true point-in-time universe addressing "
               "residual survivorship."),
        bullet("<b>Key risks if (against this recommendation) deployed:</b> regime dependence "
               "(return is regime exposure, not a stable factor); selection / multiple-testing "
               "fragility (no family Bonferroni survivor on its own terms); subsample "
               "<b>sign-instability</b> (the hard disqualifier); parameter fragility (&plusmn;50% "
               "lookback breaks it); and data limitations (no funding, daily t+1-close execution, "
               "volume unreliability, residual survivorship)."),
        bullet("<b>Next steps:</b> treat momentum as closed for deployment on this universe; if "
               "revisited, pursue the three conditions above, and consider it only as one input to "
               "a broader, properly multiple-testing-controlled research program &mdash; never as a "
               "standalone sleeve."),
        PageBreak(),
    ]

    # ---------------- Section 7: Appendix ----------------
    story += [
        P("7. Appendix &mdash; full tables and required disclosures", H1),
        P("7.1 Required disclosures (spec &sect;5.4)", H2),
        bullet("<b>Gross vs net side by side</b> (Section 4): spot costs reduce every reported "
               "Sharpe; net is never flattered above gross."),
        bullet("<b>In-sample vs out-of-sample side by side</b> (Section 4): the OOS window was "
               "opened exactly once (single-use guard); a near-zero / negative OOS Sharpe would be "
               "reported as overfitting, not hidden."),
        bullet("<b>&lsquo;Works only gross&rsquo; &mdash; momentum_L28d_S1d:</b> the academic 4-week "
               "canonical lookback is positive gross (IS gross Sharpe 0.073) but <b>net-negative</b> "
               "both in-sample (<b>net -0.031</b>) and out-of-sample (<b>net -0.270</b>, gross "
               "-0.158). The canonical horizon does not work net of costs."),
        bullet("<b>No funding / spot:</b> Artemis exposes no perpetual funding; returns and costs "
               "carry no funding term (the guide's third cost component is N/A and stated)."),
        bullet("<b>Daily, t+1-close execution:</b> Artemis serves daily granularity only; fills are "
               "at the t+1 close with slippage applied &mdash; a one-period lag, not a costless "
               "intraday fill."),
        bullet("<b>Wrapped-token exclusion:</b> wrapped tokens (WBTC, WETH, stETH, &hellip;) are "
               "excluded as redundant price exposures &mdash; a deliberate, disclosed deviation."),
        bullet("<b>Residual (as-of-today catalog) survivorship:</b> the Artemis catalog is an "
               "as-of-today snapshot with no listing/delisting dates; coins fully purged before the "
               "query are unrecoverable &mdash; an <b>unestimable lower bound</b> on survivorship "
               "bias. Reported factor returns are <b>upper bounds</b>."),
        bullet("<b>Volume reliability:</b> only 24H_VOLUME is historical (30D_VOLUME is real-time "
               "only) and is unreliable cross-sectionally; liquidity is gated on a market-cap floor "
               "+ trailing-30d <b>median</b> 24H volume &mdash; a disclosed deviation from the "
               "mean-ADV rule."),
        bullet("<b>Survivorship flows into P&amp;L:</b> a collapsed short-leg coin's crash books as "
               "a positive contribution; dead coins are <b>not</b> dropped (238/846, 28% terminal "
               "collapses carried)."),
        Spacer(1, 4),
        P("7.2 Universe / survivorship figures (docs/AUDIT.md)", H2),
    ]
    audit_header = ["Metric", "Value"]
    audit_rows = [
        ["Artemis /asset catalog entries (as-of-today)", "1,013"],
        ["Assets with any price data (panel)", "846"],
        ["Date range", "2018-01-01 → 2026-05-30"],
        ["Panel rows (date × symbol)", "2,598,912"],
        ["Total terminal collapses (>90% drawdown, carried)", "238 (28%)"],
        ["  — delisted cohort", "15"],
        ["  — zombie cohort (still printing near-zero)", "223"],
        ["Left-censored assets (first price = 2018-01-01)", "272"],
        ["Assets ever eligible", "303"],
        ["Assets eligible on latest date", "226"],
    ]
    audit_w = [w * inch for w in [4.2, 2.0]]
    story += [
        styled_table(audit_header, audit_rows, audit_w),
        Spacer(1, 4),
        P("7.3 Full Stage-2 significance table &mdash; all 21 variants, INCLUDING failures", H2),
        P("Rows are the pre-registered selection family (fam = sel, skip = 1) first, then the "
          "skip-{2,3} diagnostics (fam = diag). The HAC (Newey-West) t-stat is the reported "
          "mean-return test; the naive t is shown but biased. holds_sign = NO is a hard deployment "
          "disqualifier. The highlighted row is the sole bootstrap-Bonferroni survivor "
          "(L5d/S1d), itself disqualified by sign-instability. Numbers are verbatim from "
          "data/stats/significance.parquet.", TBL_CAP),
        stage2_table(),
        Spacer(1, 6),
        P("7.4 Full Stage-4 tables", H2),
        P("Comparator L28d/S1d (academic 4-week) &mdash; gross vs net, IS vs OOS:", BODY),
    ]
    comp_rows = [
        ["in-sample", "0.073", "-0.031", "-0.0134", "0.4314", "-0.030", "-0.6809", "-0.020", "0.559", "23.90"],
        ["out-of-sample", "-0.158", "-0.270", "-0.0818", "0.3031", "-0.283", "-0.5168", "-0.158", "0.433", "26.57"],
    ]
    story += [
        styled_table(seg_header, comp_rows, seg_w, flag_rows={0, 1}),
        P("L28d works only gross of costs: net Sharpe is negative in both segments.", TBL_CAP),
        P("Additional net metrics (total return, avg win / loss):", BODY),
    ]
    add_header = ["spec", "segment", "total return", "avg win", "avg loss"]
    add_rows = [
        ["Primary L5d_S1d", "in-sample", "1.7225", "0.08584", "-0.06745"],
        ["Primary L5d_S1d", "out-of-sample", "0.6266", "0.07509", "-0.05136"],
        ["Comparator L28d_S1d", "in-sample", "-0.4518", "0.06534", "-0.08526"],
        ["Comparator L28d_S1d", "out-of-sample", "-0.2687", "0.06283", "-0.05991"],
    ]
    add_w = [w * inch for w in [1.7, 1.3, 1.2, 1.0, 1.0]]
    story += [
        styled_table(add_header, add_rows, add_w),
        P("2&times;-costs sensitivity (in-sample, primary L5d/S1d):", BODY),
    ]
    cost_rows = [
        ["1x costs", "0.789", "0.664", "0.2490", "0.3752", "0.756", "-0.4851", "0.513", "0.574", "24.31"],
        ["2x costs", "0.789", "0.552", "0.2072", "0.3753", "0.610", "-0.5219", "0.397", "0.574", "24.32"],
    ]
    story += [
        styled_table(seg_header, cost_rows, seg_w),
        P("Doubling fees + slippage cuts net Sharpe 0.664 → 0.552 (same spec; not a "
          "re-selection).", TBL_CAP),
    ]

    doc.build(story)

    # Pin document dates so the build is byte-reproducible (reportlab otherwise
    # stamps wall-clock CreationDate/ModDate, dirtying the tree on every rerun).
    _stamp_fixed_dates(OUT_PDF, "D:20260530000000+00'00'")
    return OUT_PDF


def _stamp_fixed_dates(pdf_path: pathlib.Path, fixed_date: str):
    """Rewrite CreationDate/ModDate to a fixed value for byte-reproducibility."""
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        return  # pypdf is a dev dep; skip if absent (content already deterministic)
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({
        "/Title": "Artemis Momentum Factor — Research Report",
        "/Author": "Artemis Quant",
        "/CreationDate": fixed_date,
        "/ModDate": fixed_date,
        "/Producer": "amom/build_report",
    })
    with open(pdf_path, "wb") as fh:
        writer.write(fh)


def _figure_block(fname: str, width):
    path = FIG / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing figure: {path} (run scripts/build_report_figures.py)")
    img = Image(str(path))
    # Preserve aspect ratio.
    iw, ih = img.imageWidth, img.imageHeight
    img.drawWidth = width
    img.drawHeight = width * ih / iw
    return [img, P(FIG_CAPTIONS[fname], CAPTION)]


def _page_count(pdf_path: pathlib.Path) -> int | None:
    """Page count via pypdf if available, else pdfinfo, else a raw /Type /Page count."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
        ).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        raw = pdf_path.read_bytes()
        return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    except Exception:
        return None


if __name__ == "__main__":
    out = build()
    n = _page_count(out)
    print(f"Wrote: {out}")
    if n is not None:
        print(f"Pages: {n}")
        if not (8 <= n <= 12):
            raise SystemExit(f"ERROR: page count {n} outside required 8-12 range")
    print("Done.")
