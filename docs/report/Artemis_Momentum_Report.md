# Artemis Momentum Factor — Research Report

_Date: 2026-05-30. Status: final deliverable (Stage 5). Verdict: **NO-DEPLOY.**_

_All numbers in this report are quoted verbatim from the committed source-of-truth artifacts
(`docs/STAGE2_RESULTS.md`, `docs/STAGE4_RESULTS.md`, `docs/AUDIT.md`) and the parquet outputs
(`data/stats/significance.parquet`, `data/backtest/{equity,positions,trades}.parquet`). The
PDF is regenerated deterministically by `uv run python scripts/build_report.py`._

---

## 1. Conclusion first

**We tested cross-sectional momentum on an Artemis-sourced crypto spot universe and find no
deployable factor. The recommendation is: do not deploy.**

- **What we tested.** Twenty-one momentum variants total — a **pre-registered selection family
  of 7 lookbacks** (1, 3, 5, 7, 14, 28, 56 days) at the convention skip of 1 day, plus 14
  skip-{2,3} **diagnostics** that are reported for transparency but were never eligible for
  deployment. The full Stage-2 significance battery (Newey-West HAC, Lo Sharpe SE, spanning
  vs market + size control, stationary block bootstrap, Bonferroni / Harvey-Liu-Zhu,
  subsample sign-stability, deflated Sharpe and PBO) was run on the in-sample slice; a sealed
  out-of-sample window was opened exactly once in the cost-aware backtest.

- **What held up: nothing, at the family level.** **No selection-family variant is significant
  on the Newey-West test against the Bonferroni threshold** (0.05 / 7 = 0.00714). The strongest,
  `momentum_L5d_S1d`, is HLZ-"suggestive" with HAC t = 2.40 and HAC p = 0.0081 — which is
  *above* the Bonferroni threshold. It clears Bonferroni **only** through the
  bootstrap-override rule (bootstrap p = 0.0058), and is then **disqualified outright** by the
  Stage-2.6 deployment gate because it **flips sign across subsamples** (`holds_sign = no`).
  Its deflated Sharpe is 0.46 and the grid PBO is 0.114.

- **Deployed-candidate net-of-cost OOS Sharpe.** Characterized for completeness, the primary
  candidate `momentum_L5d_S1d` posts an out-of-sample **net Sharpe of 0.756** (gross 0.897) and
  an in-sample net Sharpe of 0.664 (gross 0.789). The OOS Sharpe did **not** collapse — but this
  is a **single-regime artifact** on a window of ~30 overlapping observations spent exactly
  once (the 2024 crypto bull), not a stable edge. The academic 4-week comparator
  `momentum_L28d_S1d` **works only gross of costs**: gross Sharpe 0.073 turns to **net −0.031**
  in-sample and **net −0.270** out-of-sample.

- **Recommendation.** **Do not deploy momentum on the Artemis spot universe.** The suggestive
  ~5-day signal is a *research lead*, not a strategy. Capacity (~$36M) is comfortable and is
  **not** the binding constraint; the disqualification rests on multiple-testing failure,
  subsample sign-instability, regime dependence, fragility to the lookback parameter, and the
  disclosed data limitations.

---

## 2. Methodology (brief)

**Data — Artemis only, survivorship-corrected.** The universe is enumerated programmatically
from the Artemis `/asset` catalog (1,013 entries; 846 with any price data), on a **daily
point-in-time grid spanning 2018-01-01 to 2026-05-30** (2,598,912 panel rows). Eligibility
filters (≥90 days history, ≥50% observation density, $10M market-cap floor, trailing-30d median
24H volume above threshold, staleness check, stablecoin and wrapped-token exclusion) are
rebuilt **as-of each date**. Dead and collapsed coins are **kept**: 238 of 846 assets (28%)
exhibit a terminal >90% drawdown (15 formally delisted + 223 still-printing "zombies"), and
every one of their crash returns is carried into the P&L (the LUNA/`lunc` −99.99% collapse
included). Recycled tickers are split into distinct synthetic assets so a revived ticker is
never spliced onto a dead one. Residual survivorship — coins fully purged from Artemis before
the query — is **unestimable and disclosed** as a lower bound; reported factor returns are
therefore **upper bounds** relative to a hypothetically complete universe.

**Returns.** Holding return is the **simple spot price return** — Artemis exposes no perpetual
funding, so the guide's funding-adjusted return is not computable and **no funding term** is
modeled (disclosed). Simple returns aggregate across coins within a period; log returns compound
through time; the two conventions are never mixed.

**Momentum construction.** Signal = trailing log-sum return over a fixed lookback, shifted by a
skip. The **7-lookback grid is fixed in advance** (academic 14/28/56-day plus crypto-short
1/3/5/7-day); skip is a nuisance parameter fixed by convention at **1 day**. The eligible
universe is sorted into **quintiles**, long the top 20% / short the bottom 20%, **equal-weight**
within each leg, **dollar-neutral**. The signal uses data through the close of t; the position
is entered at the **close of t+1** (a mandatory, unit-tested one-period lag — no look-ahead).

**Stage-2 significance battery.** Per variant: naive t-stat (reported but flagged biased, never
the headline); **Newey-West HAC t-stat** (the reported mean-return test, with bandwidth covering
the holding-period overlap); Sharpe with the **Lo (2002) SE** (autocorrelation-corrected when
Ljung-Box fires); a **spanning regression** of the factor on {equal-weight market return,
small-minus-big size control} reporting the HAC-t alpha (the size control is a TEST-ONLY
regressor, never deployed); **Bonferroni** on the pre-registered family of 7 and HLZ tiers; a
**stationary block bootstrap** (arch) as the bootstrap of record, where a Newey-West/bootstrap
disagreement across the adjusted threshold defers to the bootstrap; **subsample sign-stability**
(halves and thirds) as a hard deployment gate — a sign-flip disqualifies regardless of t-stat;
and **deflated Sharpe (DSR)** plus **PBO/CSCV** to quantify selection overfit.

**Sealed OOS and cost-aware backtest.** The most recent ~30% of rebalance dates
(`OOS_START = 2023-12-02`) were reserved **before any statistic was computed** and opened
**exactly once**, in Stage 4, under a single-use guard. The backtest applies a 10 bps spot taker
fee per side plus size-scaled tiered slippage, executes at the t+1 close, and reports the full
net metric set (Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover) plus a capacity
estimate.

---

## 3. Factor-by-factor (variant) results

**Selection family (skip = 1) — the deployment candidates.** None is significant on the
Newey-West HAC test at the Bonferroni threshold (0.05 / 7 = 0.00714). The diagnostics are listed
separately and were **never eligible for deployment**; the visually striking diagnostic Sharpes
(e.g. L3d/S3d HAC t ≈ 5.0, L14d/S3d HAC t ≈ 3.9) are **not in the selection family**.

_(Full numeric tables, including all 14 diagnostics, appear in the Appendix. The selection-family
summary is rendered as a table in the PDF.)_

Key reads:

- **`momentum_L5d_S1d` (strongest):** HAC t = 2.403, HAC p = 0.0081 (> 0.00714), bootstrap p =
  0.0058, HLZ = suggestive, spanning alpha t = 2.206, DSR = 0.460. **Survives Bonferroni only via
  the bootstrap override, and `holds_sign = no` — DISQUALIFIED** by the §2.6 sign-stability gate.
- **`momentum_L7d_S1d`:** HAC t = 1.887, HAC p = 0.0296 — not significant.
- **`momentum_L14d_S1d`:** HAC t = 1.945, HAC p = 0.0259 — not significant.
- **`momentum_L3d_S1d`:** HAC t = 1.533, sign-unstable — not significant.
- **`momentum_L28d_S1d` (academic 4-week):** HAC t = 0.955, HAC p = 0.1698 — not significant.
- **`momentum_L56d_S1d`:** HAC t = 0.756 — not significant.
- **`momentum_L1d_S1d`:** HAC t = −1.300 (negative; short-term reversal), sign-unstable.

Grid-level overfit controls: **PBO/CSCV = 0.114** (below the 0.5 overfitting threshold but on a
correlated family — Fig 5 shows short lookbacks cluster tightly, so the effective independent-test
count is below 7).

---

## 4. Deployed-candidate characterization (Stage-4 backtest)

For completeness we characterized the strongest candidate, `momentum_L5d_S1d`, net of realistic
spot costs. **This characterizes the candidate; it does not rescue it** — the Stage-2
disqualification stands.

**Gross vs net, in-sample vs out-of-sample (primary `momentum_L5d_S1d`):**

| segment | gross Sharpe | net Sharpe | net ann ret | net ann vol | Sortino | max DD | Calmar | hit rate | ann turnover |
|---|---|---|---|---|---|---|---|---|---|
| in-sample | 0.789 | 0.664 | 0.2490 | 0.3752 | 0.756 | −0.4851 | 0.513 | 0.574 | 24.31 |
| out-of-sample | 0.897 | 0.756 | 0.2469 | 0.3266 | 1.031 | −0.2516 | 0.981 | 0.567 | 28.22 |

- **IS − OOS net Sharpe gap = −0.092.** The OOS figure exceeds in-sample, but this is a
  spent-once, ~30-observation, single-regime (2024-bull) artifact — not evidence of a stable edge.
- **Capacity ≈ $36,044,481** — the AUM at which size-scaled slippage on the actual per-rebalance
  traded order (~2.0× summed one-way turnover) erases the gross edge (per-rebalance gross edge
  0.02436). This is comfortably above a $1M book, so **capacity is not the binding constraint**.
- **Turnover** is ~24 (IS) / ~28 (OOS) annualized — high, consistent with a 5-day signal.

---

## 5. Robustness

**2× costs (in-sample, primary):** net Sharpe degrades from **0.664 → 0.552** (gross unchanged at
0.789; max DD widens to −0.5219). The edge thins materially under a doubled cost assumption.

**±50% lookback (in-sample, net):** the construction is **fragile** to its one free parameter.

| variant | net Sharpe | net ann ret | note |
|---|---|---|---|
| `momentum_L5d_S1d` (chosen) | 0.664 | 0.2490 | chosen lookback = 5d |
| `momentum_L2d_S1d` (−50%) | −0.328 | −0.1155 | lookback halved |
| `momentum_L8d_S1d` (+50%) | 0.621 | 0.2645 | lookback up 50% |

A −50% lookback flips the candidate net-negative — not robust.

**Regime breakdown (in-sample, net mean return per regime):**

| regime | n | mean net return |
|---|---|---|
| bull | 17 | −0.00624 |
| bear | 28 | 0.02195 |
| chop | 22 | 0.03453 |

> **Caveat (do not over-read):** this is a **full-sample, descriptive** partition, **not** a
> walk-forward signal — it could not have been traded ex-ante. The `chop` bucket is the
> top-|market-move| tercile, which on this sample skews toward large **up** moves, so the
> apparent "negative in bull / positive in bear" contrast is **overstated** and is an artifact of
> where the magnitude cut falls. **The disqualifying evidence is the Stage-2 §2.6 sign-instability
> itself, not this descriptive split.**

**One-shot OOS, single-regime character.** The OOS window is 30 overlapping-regime observations
spent exactly once; a single favorable stretch (the 2024 crypto bull) carries the positive OOS
Sharpe. A near-zero or negative OOS Sharpe would have been reported as overfitting; the positive
figure is reported as-is and interpreted as regime exposure, consistent with the sign-flip
disqualification.

---

## 6. Recommendation

**Do not deploy.** Target allocation to a momentum sleeve on the Artemis spot universe: **zero.**

- **Expected Sharpe (range):** for the strongest candidate, net Sharpe sits in roughly
  **0.66–0.76** in the realized sample, but with **no statistically reliable edge** and a true
  expected Sharpe indistinguishable from zero once multiple testing, sign-instability, and the
  single-regime OOS are accounted for. We do not represent this as a deployable expectation.

- **The suggestive ~5-day signal is a research lead, not a strategy.** It would warrant
  follow-up only with (a) more independent observations, (b) a cleaner, lower-turnover
  construction, and (c) a true point-in-time universe that addresses residual survivorship.

- **Key risks if (against this recommendation) deployed:** regime dependence (the return is
  regime exposure, not a stable factor); selection / multiple-testing fragility (no family
  Bonferroni survivor on its own terms); subsample **sign-instability** (the hard disqualifier);
  parameter fragility (±50% lookback breaks it); and data limitations (no funding, daily t+1-close
  execution, volume unreliability, residual survivorship).

- **Next steps:** treat momentum as closed for deployment on this universe; if revisited, pursue
  the above three conditions, and consider it only as one input to a broader, properly
  multiple-testing-controlled research program — never as a standalone sleeve.

---

## 7. Appendix — full tables and required disclosures

### 7.1 Required disclosures (spec §5.4)

- **Gross vs net side by side** (Section 4): spot costs reduce **every** reported Sharpe; net is
  never flattered above gross.
- **In-sample vs out-of-sample side by side** (Section 4): the OOS window was opened exactly once
  (single-use guard); a near-zero / negative OOS Sharpe would be reported as overfitting, not
  hidden.
- **"Works only gross" — `momentum_L28d_S1d`:** the academic 4-week canonical lookback is
  positive gross (IS gross Sharpe 0.073) but **net-negative** both in-sample (**net −0.031**) and
  out-of-sample (**net −0.270**, gross −0.158). The canonical horizon does not work net of costs.
- **No funding / spot:** Artemis exposes no perpetual funding; returns and costs carry **no
  funding term** (the guide's third cost component is N/A and stated).
- **Daily, t+1-close execution:** Artemis serves daily granularity only; fills are at the t+1
  close with slippage applied — a one-period lag, not a costless intraday fill.
- **Wrapped-token exclusion:** wrapped tokens (WBTC, WETH, stETH, …) are excluded as redundant
  price exposures of their underlyings — a deliberate, disclosed deviation from the guide.
- **Residual (as-of-today catalog) survivorship:** the Artemis catalog is an as-of-today snapshot
  with no listing/delisting dates; coins fully purged before the query are unrecoverable. This is
  an **unestimable lower bound** on survivorship bias; reported factor returns are **upper bounds**.
- **Volume reliability:** only `24H_VOLUME` is historical (`30D_VOLUME` is real-time only) and is
  unreliable cross-sectionally; liquidity is gated on a market-cap floor + trailing-30d **median**
  24H volume — a disclosed deviation from the guide's mean-ADV rule.
- **Survivorship flows into P&L:** a collapsed short-leg coin's crash books as a positive
  contribution; dead coins are **not** dropped (238/846, 28% terminal collapses carried).

### 7.2 Universe / survivorship figures (`docs/AUDIT.md`)

| Metric | Value |
|---|---|
| Artemis `/asset` catalog entries (as-of-today) | 1,013 |
| Assets with any price data (panel) | 846 |
| Date range | 2018-01-01 → 2026-05-30 |
| Panel rows (date × symbol) | 2,598,912 |
| Total terminal collapses (>90% drawdown, carried) | 238 (28%) |
| — delisted cohort | 15 |
| — zombie cohort (still printing near-zero) | 223 |
| Left-censored assets (first price = 2018-01-01) | 272 |
| Assets ever eligible | 303 |
| Assets eligible on latest date | 226 |

### 7.3 Full Stage-2 significance table (all 21 variants, including failures) — rendered in PDF.

### 7.4 Full Stage-4 tables (gross-vs-net, IS-vs-OOS, robustness, additional net metrics) — rendered in PDF.
</content>
</invoke>
