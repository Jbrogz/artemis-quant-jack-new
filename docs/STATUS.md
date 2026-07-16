# Artemis Momentum Factor Book — Project STATUS & Handoff

_Last updated: 2026-06-02. Read this FIRST when resuming in a new window._

## What this project is
A single-factor (**momentum only**) crypto long/short **factor book**, with **all data sourced from Artemis**, following the Project-1 Factor Book Guide methodology **strictly** (`docs/reference/factor-book-guide.md`). Python package `amom`. The competition entry that this replaces (the old `cmom` "momentum sleeve") was deemed junk.

- **Repo:** `github.com/Jbrogz/artemis-quant-jack-new` (private, personal account `Jbrogz`).
- **Local:** `/Users/jackbrogan/Desktop/artemis-quant-joint/new-artemis-work` (separate git repo, nested in the `artemis-quant-joint` working dir which gitignores it).
- **Deliverables (done):** `docs/report/Artemis_Momentum_Report.pdf` (10pp) and `docs/report/Artemis_Momentum_Findings.docx`. No pitch deck (per scope).

## Current state — COMPLETE through Stage 5 + skip≥2 widened validation (RESOLVED)
- **254 tests passing, 3 skipped.** ~3.2k LOC src + ~5k LOC tests.
- Stage 1 universe (SOUND), Stage 1.2–1.4 returns+signal+formation (SOUND), Stage 2 significance (SOUND), Stage 3–4 cost-aware backtest+OOS (SOUND after a remediation), Stage 5 report (SOUND).
- 2026-06-02 Codex remediation pass added a competition-readiness plan, stronger OOS-seal tests, a real `make reproduce` chain, report/write-up consistency tests, conservative missing-ADV costs, and missing-return visibility in the backtest equity artifact. The strategy verdict remains **NO-DEPLOY**.

## The verified finding (pre-registered analysis)
**Momentum is a rigorous, regime-dependent statistical NULL → NO-DEPLOY.** Pre-registered selection family = the **7 lookbacks at skip=1** (n=69 in-sample, Bonferroni 0.05/7=0.00714). No family variant clears Newey-West (lowest HAC p is L5d=0.0081, *above* threshold). L5d is "suggestive" (HAC t=2.40, spanning α t=2.21) but **fails subsample sign-stability** (holds_sign=False) and only "survives" via the bootstrap override → deployment-disqualified. DSR(L5d)=0.46, grid PBO=0.114. Net of costs: academic L28d works only gross (net −0.031); capacity ~$36M (not binding).

## ✅ RESOLVED (2026-05-31) — skip≥2 widened/post-hoc validation: NO deployable candidate
**The open work item is closed.** The strong `skip ∈ {2,3}` variants were promoted to a widened selection axis (full family **m=21**, Bonferroni 0.05/21=**0.0023810**), charged the full multiple-testing, run through the cost-aware backtest, and each candidate's **previously-unspent OOS window was spent EXACTLY ONCE** (the single guarded read, `open_count==1`). Plan: `docs/plans/2026-05-31-skip-variant-validation.md`. Results: `docs/STAGE2_RESULTS.md` (widened section) + `docs/STAGE4_RESULTS.md` (widened conclusion).

**Finding: no candidate is deployable.** The deployment bar is the intersection of three gates — survive costs (IS net Sharpe > 0), survive OOS net of costs, and be m=21-robust (clear under **both** HAC and bootstrap). That intersection is **empty**:

| variant | m=21 robust (HAC AND boot) | IS net Sharpe | OOS net Sharpe | ann turnover | verdict |
|---|---|---|---|---|---|
| **L3d/S3d** (primary) | yes (HAC t=5.0, DSR 0.98) | 1.542 | 0.297 | 24.27 | **fails-OOS** |
| L14d/S3d | yes (HAC t=3.95, DSR 0.80) | 0.844 | **−0.486** | 23.58 | **fails-OOS** (OOS net-negative) |
| L1d/S3d | **no — MARGINAL** (HAC clears, boot 0.0054 does not) | 1.010 | 0.455 | 22.09 | **marginal** |
| L5d/S3d | no (reported p 0.00391) | 0.908 | 0.645 | 22.03 | **marginal** |
| L5d/S2d | no (reported p 0.0108) | 0.923 | 0.714 | 20.96 | **marginal** |

- The two genuinely multiple-testing-robust survivors **both fail OOS**: the strongest in-sample variant in the whole study, **L3d/S3d** (gross Sharpe 1.91, net 1.54, HAC t=5.0, DSR 0.98, sign-stable), **collapses to OOS net 0.297** (the highest-turnover, most cost-exposed short lookback); **L14d/S3d goes net-negative OOS (−0.486)**.
- The OOS-positive specs (**L5d/S2d** best at 0.714, then L5d/S3d 0.645, then marginal L1d/S3d 0.455) **fail the multiple-testing-aware gate** — none clears m=21 under both tests; their DSRs (0.56 / 0.68 / 0.87) already reflect the widened-family deflation.
- **L1d/S3d is explicitly MARGINAL:** HAC p 0.00218 clears the widened threshold but bootstrap p 0.0054 does not; applying the bootstrap-override-on-disagreement rule consistently at m=21, the non-clearing bootstrap governs → not a qualified survivor.

**Mandatory honesty caveats (all preserved in the docs):**
1. **Post-hoc / selection-biased.** These were chosen *after* seeing they won in-sample. The widened m=21 analysis is exploratory and does **NOT** overturn the pre-registered skip=1 null — that result **remains the headline finding** (see below).
2. **OOS spent once PER VARIANT.** Each skip≥2 OOS window was opened exactly once and must never be iterated on.
3. **Cost/turnover/reversal risk realized.** Short lookbacks (L1d/L3d) rebalance into near-reversal territory → highest turnover, most cost-exposed (guide §1.3); the gross t=5 of L3d/S3d did not survive OOS net of costs.

**No deliverable regeneration warranted.** Because no candidate is genuinely deployable, the report (`docs/report/Artemis_Momentum_Report.pdf`) and findings (`Artemis_Momentum_Findings.docx`) need **no** regeneration — their headline pre-registered NULL → NO-DEPLOY is unchanged. (Had a candidate cleared all three gates, it would be a genuine post-hoc positive requiring a forward-data confirmation before any deployment, and the report would be regenerated with its OOS net Sharpe, IS→OOS gap, and the widened conclusion — but none did.)

## How to work in this repo (conventions that MUST be preserved)
- **TDD always** (real test → fail → minimal impl → pass → commit). ≥80% coverage. karpathy-guidelines (minimal, surgical, no speculative machinery).
- **No look-ahead is the cardinal rule.** Every decision for date t uses only data ≤ close t; execution at t+1 close. Tests must be *discriminating* (mutating future data must change nothing at t).
- **Honest reporting.** Naive t is never the headline (HAC is); underpowered → "inconclusive"; failures are always included; gross-vs-net and IS-vs-OOS shown side by side; never massage toward significance.
- **Gated workflows:** build via fresh Codex subagents, then a multi-lens adversarial review must return SOUND before proceeding. Use stronger reasoning on correctness-critical work and faster agents on mechanical checks.
- **Codex workflow note:** this repository is now being remediated with Codex subagents, not Claude/Opus sessions. Preserve the same discipline: TDD, small ownership slices, independent review, and explicit verification.
- **`.env` is OFF-LIMITS** — never open/cat/grep/print it. Scripts load `ARTEMIS_API_KEY` via `python-dotenv` themselves and print only `len`.
- **GitHub push gotcha:** the macOS keychain may serve a different github.com credential by default, so a plain `git push` can 404. **Always `gh auth switch -u Jbrogz` immediately before pushing.** The repo's local credential helper is pinned to `gh auth git-credential`.

## Repo map
- `src/amom/` — `config.py` (all frozen params/constants), `providers/` (Artemis client), `cache.py`, `universe/` (registry, coverage, eligibility, builder, recycle), `returns/spot.py`, `factor/` (momentum, portfolio), `stats/` (core, sharpe_se, spanning, bootstrap, subsample, dsr, pbo), `backtest/` (costs, engine, metrics).
- `scripts/` — `probe_artemis.py`, `build_universe.py`, `build_returns.py`, `build_factor_returns.py`, `run_stage2.py`, `run_backtest.py`, `build_report_figures.py`, `build_report.py`, `build_writeup_docx.py`.
- `docs/` — `reference/factor-book-guide.md` (authoritative methodology), `specs/2026-05-30-artemis-momentum-design.md` (the spec, rev 3 + Appendix B = live-Artemis facts), `plans/*.md`, `STAGE2_RESULTS.md`, `STAGE4_RESULTS.md`, `AUDIT.md`, `report/`.
- `Makefile` targets: `probe`, `universe`, `returns`, `factor`, `stage2`, `backtest`, `figures`, `report`, `writeup`, `test`, `lint`, `reproduce`.
- Data artifacts (`data/`) are **gitignored** and regenerated by the scripts — rebuild before consuming (some on-disk parquet may lag the latest schema).

## Key Artemis data facts (verified live — spec Appendix B)
`GET https://data-svc.artemisxyz.com/asset` enumerates ~1013 assets keyed on stable `artemis_id` (≠ ticker). No funding (→ spot returns, no carry). Only `24H_VOLUME` is historical (`30D_VOLUME` is real-time only) and it's noisy → liquidity uses MC + median 24H-vol. DAY granularity only. Catalog is as-of-today → purged-dead-coins unrecoverable (residual survivorship disclosed; 28% of the 846-asset universe show terminal >90% collapses, carried into P&L).

## Known minor items (non-blocking)
- `scripts/build_report.py` / `build_writeup_docx.py` still embed verified narrative and some table numbers as literals. The PDF reads the full Stage-2 table from `data/stats/significance.parquet`; `tests/test_report_source_consistency.py` checks the highest-risk DOCX Stage-2/Stage-4 tables against `docs/STAGE2_RESULTS.md` and `docs/STAGE4_RESULTS.md`. Optional future work: wire every narrative value directly to regenerated artifacts.
- A few pre-existing ruff unused-import warnings in some `tests/test_universe_*.py` / `test_cache.py` files (predate this work; left untouched per surgical-change rule).

## Handoff checklist for the new window
1. `cd new-artemis-work`; `uv sync`; `uv run pytest -q`; `uv run ruff check src tests scripts`.
2. Read this STATUS.md (the skip≥2 widened validation is now ✅ RESOLVED — no deployable candidate), then `docs/specs/2026-05-30-artemis-momentum-design.md` (§2, §3, §7) and, for the closed widened analysis, `docs/plans/2026-05-31-skip-variant-validation.md` + the widened sections of `docs/STAGE2_RESULTS.md` / `docs/STAGE4_RESULTS.md`.
3. For full live regeneration, run `make reproduce` after confirming `ARTEMIS_API_KEY` is available in the local secret environment. Do not open or inspect `.env*` files.
4. There is **no open strategy work item.** The pre-registered skip=1 NULL → NO-DEPLOY is the headline finding and the widened/post-hoc skip≥2 analysis confirms it (no candidate clears costs + OOS + m=21-robustness). Push with `gh auth switch -u Jbrogz` first.
