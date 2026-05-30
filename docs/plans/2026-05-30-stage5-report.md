# Stage 5 — Research Report Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Apply karpathy-guidelines. The full analytical pipeline is complete and SOUND. Stage 5 is the **deliverable memo** — an 8–12 page PDF for a mixed quant/non-quant audience. **Do not re-run or re-derive the analysis; pull every number verbatim from the committed source-of-truth docs and parquet artifacts. Invent nothing.**

**Goal:** Produce a reproducible, conclusion-first research report (PDF) that honestly presents the finding: **momentum is not a deployable factor on the Artemis spot universe** — a rigorous, regime-dependent near-null.

**Sources of truth (read; quote exactly):**
- `docs/STAGE2_RESULTS.md` — the significance table (all 21 variants incl. failures).
- `docs/STAGE4_RESULTS.md` — gross/net + IS/OOS Sharpe, capacity, regime, robustness.
- `docs/AUDIT.md` — universe + survivorship figures.
- `docs/specs/2026-05-30-artemis-momentum-design.md` — methodology, all disclosed limitations (§3, §10, Appendix B).
- `data/stats/significance.parquet`, `data/backtest/{equity,positions,trades}.parquet`.

**Authoritative headline numbers (verify against the docs; do not contradict):**
- Universe: **846 assets**, daily point-in-time grid 2018→2026, survivorship-corrected; **238/846 (28%)** terminal >90% collapses (15 delisted + 223 still-printing zombies), crashes carried into P&L.
- Significance (in-sample, n=69/variant; selection family = 7 lookbacks at skip=1; Bonferroni 0.05/7=0.00714): **no family variant is significant on the Newey-West test**; strongest L5d HAC t=2.40 (p=0.0081, *above* threshold), HLZ "suggestive", clears Bonferroni only via the bootstrap-override and is **disqualified by subsample sign-instability**; DSR(L5d)=0.46, grid PBO=0.114. The eye-catching L3d/skip=3 (t≈5.0) are **diagnostics, not in the family**.
- Backtest (net of costs): **L5d** IS Sharpe gross 0.789→net 0.664, OOS gross 0.897→net 0.756 (IS–OOS gap −0.092, but a single-regime 2024-bull, spent-once, 30-obs artifact). **L28d** (academic 4-week) gross 0.073→**net −0.031** ("works only gross"), OOS net −0.270. Capacity ≈ **$36M** (not the binding constraint). 2× costs → IS net 0.552; ±50% lookback fragile (L2d net −0.328).
- **Verdict: NO-DEPLOY.**

**Hard rules:** Never open/print `.env`. One-line commits, no Co-Authored-By. Do NOT push. The report is reproducible via a committed script. No invented numbers.

---

## Task RPT1: Charts

**Files:** Create `scripts/build_report_figures.py`, figures under `docs/report/figures/`.

- [ ] `uv add matplotlib`. Generate (from the parquet artifacts, NET and GROSS):
  1. **Cumulative P&L** — L5d gross and net on one axis (and L28d net for contrast).
  2. **Drawdown** plot (net).
  3. **Rolling 6-month Sharpe** (net).
  4. **Per-variant gross Sharpe** bar chart across the 7 selection lookbacks (skip=1), diagnostics shaded separately.
  5. **Variant correlation heatmap** (factor-return correlations across the selection family).
  Each figure saved as PNG (150+ dpi). Print a one-sentence plain-language **takeaway** for each (used as captions in RPT2).
- [ ] **Commit:** `feat: research-report figures`.

## Task RPT2: Report narrative + PDF assembly

**Files:** Create `scripts/build_report.py`, `docs/report/Artemis_Momentum_Report.md` (the source narrative), output `docs/report/Artemis_Momentum_Report.pdf`.

Structure (guide §5.1; 8–12 pp):
1. **Conclusion first** — variants tested, what held up (nothing, at the family level), combined/deployed-candidate net-of-cost OOS Sharpe, and the **recommendation: do not deploy**.
2. **Methodology (brief)** — Artemis-only data; point-in-time survivorship-free universe (846 assets, recycled-ticker splitting, crashes carried); spot returns (no funding); momentum construction (7-lookback grid, skip=1, quintile dollar-neutral, t+1-close, no look-ahead); the Stage-2 battery (Newey-West, Lo SE, spanning vs market+size control, arch bootstrap, Bonferroni/HLZ, subsample, DSR/PBO); sealed OOS; cost-aware backtest.
3. **Factor-by-factor (variant) results** — the full Stage-2 table **including the failures**; columns: coef, Newey-West t, p/stars, n, spanning alpha; note the diagnostics are excluded from selection.
4. **Deployed-candidate characterization** — the Stage-4 backtest (gross vs net, IS vs OOS side by side), capacity (~$36M, not binding), turnover.
5. **Robustness** — 2× costs, ±50% lookback (fragile), regime breakdown (with the descriptive-proxy caveat), the one-shot OOS and its single-regime character.
6. **Recommendation** — no deploy; the suggestive ~5-day signal is a research lead not a strategy; key risks (regime dependence, selection/multiple-testing, sign-instability, data limits); next steps.
7. **Appendix** — full tables; **required disclosures** (gross vs net and IS vs OOS side by side; "works only gross" called out for L28d; data limitations: no funding/spot, daily t+1-close execution, wrapped exclusion, as-of-today-catalog residual survivorship, volume reliability).

- [ ] Assemble the PDF reproducibly. Prefer **reportlab** (pure-Python: text + embedded figure PNGs + tables) to avoid system-lib dependencies; if a markdown→PDF path is cleaner and installs, that is acceptable as long as `uv run python scripts/build_report.py` deterministically regenerates the PDF. Verify the output is a valid PDF of 8–12 pages (e.g. check with pypdf/pdfinfo).
- [ ] **Commit:** `feat: Stage 5 research report (PDF) + reproducible build script`.

---

## Verification gate (must reach SOUND)

One adversarial reviewer confirms: the PDF exists, opens, is 8–12 pp; **every number matches** `STAGE2_RESULTS.md`/`STAGE4_RESULTS.md` (spot-check 5+); the conclusion leads and is **NO-DEPLOY**; gross-vs-net and IS-vs-OOS are side by side; failed variants are included; the L28d "works only gross" and all data limitations are disclosed; no invented/contradictory figures; each chart has a takeaway. SOUND ends the project (then push + final summary).

## Self-Review
- Covers guide §5.1–5.5. Numbers sourced from committed docs/parquet, not re-derived. Honest null is foregrounded, not buried.
