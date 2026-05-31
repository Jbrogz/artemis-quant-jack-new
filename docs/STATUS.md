# Artemis Momentum Factor Book — Project STATUS & Handoff

_Last updated: 2026-05-31. Read this FIRST when resuming in a new window._

## What this project is
A single-factor (**momentum only**) crypto long/short **factor book**, with **all data sourced from Artemis**, following the Project-1 Factor Book Guide methodology **strictly**. Python package `amom`. The competition entry that this replaces (the old `cmom` "momentum sleeve") was deemed junk.

- **Repo:** `github.com/Jbrogz/artemis-quant-jack-new` (private, personal account `Jbrogz`).
- **Local:** `/Users/jackbrogan/Desktop/artemis-quant-joint/new-artemis-work` (separate git repo, nested in the `artemis-quant-joint` working dir which gitignores it).
- **Deliverables (done):** `docs/report/Artemis_Momentum_Report.pdf` (10pp) and `docs/report/Artemis_Momentum_Findings.docx`. No pitch deck (per scope).

## Current state — COMPLETE through Stage 5, all stages adversarially verified SOUND
- **226 tests passing.** ~3.2k LOC src + ~5k LOC tests.
- Stage 1 universe (SOUND), Stage 1.2–1.4 returns+signal+formation (SOUND), Stage 2 significance (SOUND), Stage 3–4 cost-aware backtest+OOS (SOUND after a remediation), Stage 5 report (SOUND).
- Every stage was built by a **gated multi-agent workflow** (fresh agent per task, TDD, opus on correctness-critical work + sonnet on mechanical, karpathy-guidelines) and **adversarially verified by a 3-lens review** before the next stage began.

## The verified finding (pre-registered analysis)
**Momentum is a rigorous, regime-dependent statistical NULL → NO-DEPLOY.** Pre-registered selection family = the **7 lookbacks at skip=1** (n=69 in-sample, Bonferroni 0.05/7=0.00714). No family variant clears Newey-West (lowest HAC p is L5d=0.0081, *above* threshold). L5d is "suggestive" (HAC t=2.40, spanning α t=2.21) but **fails subsample sign-stability** (holds_sign=False) and only "survives" via the bootstrap override → deployment-disqualified. DSR(L5d)=0.46, grid PBO=0.114. Net of costs: academic L28d works only gross (net −0.031); capacity ~$36M (not binding).

## ⭐ OPEN WORK ITEM (next window) — validate the high-t-stat skip≥2 variants
**Decision (2026-05-31):** the `skip ∈ {2,3}` diagnostics were excluded from the pre-registered selection only because skip was fixed at 1 by convention. Several are genuinely strong and must be tested properly. Promote **skip to a selection axis** (full family m=21), charge the full multiple-testing, and run the strong candidates through the **cost-aware backtest + their unspent OOS**, reported honestly as a **widened / post-hoc** analysis.

The strong candidates (in-sample, from `data/stats/significance.parquet` / `docs/STAGE2_RESULTS.md`):

| variant | HAC t | HAC p | spanning α (t) | holds_sign | DSR |
|---|---|---|---|---|---|
| **L3d/S3d** | 5.01 | ~3e-7 | 0.081 (4.74) | **YES** | **0.98** |
| L14d/S3d | 3.95 | ~4e-5 | 0.060 (3.89) | YES | 0.80 |
| L1d/S3d | 2.85 | 0.0022 | 0.075 (2.65) | YES | 0.87 |
| L5d/S3d | 2.66 | 0.0039 | 0.050 (2.18) | YES | 0.68 |
| L5d/S2d | 2.59 | 0.0048 | 0.045 (2.05) | YES | 0.56 |

At the **m=21 Bonferroni threshold (0.05/21=0.00238)**: **L3d/S3d, L14d/S3d, L1d/S3d clear** (L5d/S3d at 0.0039 does not). The **DSR already deflates for the 21 trials**, and L3d/S3d (0.98) / L1d/S3d (0.87) / L14d/S3d (0.80) pass the ~0.95-ish bar or close to it. These are sign-stable, unlike the skip=1 family. **This is the strongest lead in the study.**

**Mandatory honesty caveats to carry into the validation (do NOT drop these):**
1. **Post-hoc selection.** These were chosen *after* seeing they won. Report the widened family at m=21 and label the analysis post-hoc/exploratory — it does not retroactively make the original pre-registered null wrong.
2. **OOS is spent once PER VARIANT.** The skip≥2 OOS windows are genuinely unspent (Stage 4 only touched L5d/S1d + L28d/S1d). Run each candidate's OOS exactly once; never iterate on it.
3. **Cost/turnover/reversal risk.** Short lookbacks (L1d/L3d) rebalance into near-reversal territory → **highest turnover, most cost-exposed** (guide §1.3 warns of this). A gross t=5 can shrink materially net of fees+slippage. The Stage-4 cost model (`src/amom/backtest/costs.py`) and one-shot-OOS engine already exist — reuse them.

**The execution plan is written:** `docs/plans/2026-05-31-skip-variant-validation.md`. Run it via the same gated-workflow pattern.

## How to work in this repo (conventions that MUST be preserved)
- **TDD always** (real test → fail → minimal impl → pass → commit). ≥80% coverage. karpathy-guidelines (minimal, surgical, no speculative machinery).
- **No look-ahead is the cardinal rule.** Every decision for date t uses only data ≤ close t; execution at t+1 close. Tests must be *discriminating* (mutating future data must change nothing at t).
- **Honest reporting.** Naive t is never the headline (HAC is); underpowered → "inconclusive"; failures are always included; gross-vs-net and IS-vs-OOS shown side by side; never massage toward significance.
- **Gated workflows:** build via fresh subagents, then a multi-lens adversarial review must return SOUND before proceeding. Model-delegate (opus = correctness/verification, sonnet = mechanical).
- **`.env` is OFF-LIMITS** — never open/cat/grep/print it. Scripts load `ARTEMIS_API_KEY` via `python-dotenv` themselves and print only `len`.
- **GitHub push gotcha:** the macOS keychain serves a `JB-acap` (work) credential for github.com by default, so a plain `git push` 404s. **Always `gh auth switch -u Jbrogz` immediately before pushing.** The repo's local credential helper is pinned to `gh auth git-credential`.

## Repo map
- `src/amom/` — `config.py` (all frozen params/constants), `providers/` (Artemis client), `cache.py`, `universe/` (registry, coverage, eligibility, builder, recycle), `returns/spot.py`, `factor/` (momentum, portfolio), `stats/` (core, sharpe_se, spanning, bootstrap, subsample, dsr, pbo), `backtest/` (costs, engine, metrics).
- `scripts/` — `probe_artemis.py`, `build_universe.py`, `build_returns.py`, `build_factor_returns.py`, `run_stage2.py`, `run_backtest.py`, `build_report_figures.py`, `build_report.py`, `build_writeup_docx.py`.
- `docs/` — `specs/2026-05-30-artemis-momentum-design.md` (the spec, rev 3 + Appendix B = live-Artemis facts; authoritative methodology), `plans/*.md`, `STAGE2_RESULTS.md`, `STAGE4_RESULTS.md`, `AUDIT.md`, `report/`.
- `Makefile` targets: `probe universe features sleeve` … plus `figures report writeup test`.
- Data artifacts (`data/`) are **gitignored** and regenerated by the scripts — rebuild before consuming (some on-disk parquet may lag the latest schema).

## Key Artemis data facts (verified live — spec Appendix B)
`GET https://data-svc.artemisxyz.com/asset` enumerates ~1013 assets keyed on stable `artemis_id` (≠ ticker). No funding (→ spot returns, no carry). Only `24H_VOLUME` is historical (`30D_VOLUME` is real-time only) and it's noisy → liquidity uses MC + median 24H-vol. DAY granularity only. Catalog is as-of-today → purged-dead-coins unrecoverable (residual survivorship disclosed; 28% of the 846-asset universe show terminal >90% collapses, carried into P&L).

## Known minor items (non-blocking)
- `scripts/build_report.py` / `build_writeup_docx.py` embed verified numbers as literals (every value matched the parquet/docs; PDFs/docx are byte-reproducible). Optional: wire them to read live from `data/stats/significance.parquet` + `data/backtest/*`.
- A few pre-existing ruff unused-import warnings in some `tests/test_universe_*.py` / `test_cache.py` files (predate this work; left untouched per surgical-change rule).

## Handoff checklist for the new window
1. `cd new-artemis-work`; `uv sync`; `uv run pytest -q` (expect 226 passed).
2. Read this STATUS.md, then `docs/specs/2026-05-30-artemis-momentum-design.md` (§2, §3, §7) and `docs/plans/2026-05-31-skip-variant-validation.md`.
3. Execute the skip-variant validation plan via gated workflows (same pattern), preserving all conventions above. Push with `gh auth switch -u Jbrogz` first.
