# Competition Readiness Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining competition-readiness gaps in the Artemis momentum factor book without changing the honest no-deploy conclusion unless regenerated evidence requires it.

**Architecture:** Keep the existing `amom` pipeline. Remediate the audit risks with surgical changes: make OOS access easier to reason about, expose a true reproduce target, make report/write-up generation claims honest, harden cost/data-gap assumptions, and update handoff docs. Use parallel agents only on disjoint file sets.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `ruff`, `pandas`, `numpy`, `python-docx`, ReportLab.

---

## Streams

### Stream A: OOS Sealing

**Files:**
- Modify: `scripts/run_stage2.py`
- Modify: `scripts/run_backtest.py`
- Modify: `tests/test_run_stage2.py`
- Modify: `tests/test_run_backtest.py`
- Optional docs: `docs/STAGE2_RESULTS.md`, `docs/STAGE4_RESULTS.md`

- [x] Add tests proving in-sample Stage 2 outputs are unchanged when rows on or after `OOS_START` are mutated.
- [x] Add tests proving Stage 4 OOS books, returns, and universe panels are sliced by the single guarded OOS path before OOS runs consume them.
- [x] Implement the smallest change that makes the OOS access semantics explicit and testable.
- [x] Run targeted pytest for `tests/test_run_stage2.py tests/test_run_backtest.py` through the existing project venv.

### Stream B: Reproduce And Reporting

**Files:**
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `docs/STATUS.md`
- Modify: `scripts/build_report.py`
- Modify: `scripts/build_writeup_docx.py`
- Optional tests if added.

- [x] Add `returns`, `factor`, `stage2`, `backtest`, `lint`, and `reproduce` targets.
- [x] Update `README.md` from design-phase text to a competition-ready project summary.
- [x] Update `docs/STATUS.md` with the Codex-only workflow and current verification commands.
- [x] Remove or soften false artifact-driven claims where report/write-up scripts intentionally use verified literals.
- [x] Prefer artifact-derived values for any small, low-risk table that can be read directly from existing docs or parquet.
- [x] Run `make -n reproduce` and the targeted report/write-up consistency tests.

### Stream C: Cost And Data-Gap Rigor

**Files:**
- Modify: `src/amom/backtest/costs.py`
- Modify: `src/amom/backtest/engine.py`
- Modify: `tests/test_backtest_costs.py`
- Modify: `tests/test_backtest_engine.py`
- Optional docs: `docs/STAGE4_RESULTS.md`, `docs/report/Artemis_Momentum_Report.md`

- [x] Add a test that non-positive or missing ADV does not get a free fee-only market-impact assumption.
- [x] Add a test that all-missing realized return windows are surfaced rather than silently treated as ordinary 0% returns.
- [x] Implement minimal conservative behavior and document it.
- [x] Run targeted pytest for `tests/test_backtest_costs.py tests/test_backtest_engine.py` through the existing project venv.

### Stream D: Review And Publication

**Files:**
- Modify only if needed after review.

- [x] Run full tests through the existing project venv.
- [x] Run scoped lint on touched Python files; full-repo ruff still has unrelated pre-existing warnings.
- [x] Regenerate report artifacts only if source outputs or report text changed materially.
- [ ] Compare local branch to `origin/main`; do not force push without explicit review.
- [ ] Prepare a safe publish path to `Jbrogz/artemis-quant-jack-new`.

## Non-Negotiables

- Do not read, inspect, source, copy, or edit `.env` or `.env.*`, including `.env.example`.
- Do not rewrite the no-deploy conclusion to sound positive. Report the evidence.
- Do not tune variants after seeing OOS.
- Keep edits surgical and focused on the audit gaps.
