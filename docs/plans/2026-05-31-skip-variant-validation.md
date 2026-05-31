# Skip-Variant Validation Plan (widened family, post-hoc, honest)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development; run via the same gated-workflow pattern (fresh agents + 3-lens adversarial verification to SOUND). Apply karpathy-guidelines. Read `docs/STATUS.md` and `docs/specs/2026-05-30-artemis-momentum-design.md` first.

**Why:** The pre-registered selection family fixed `skip=1` by convention, so the `skip ∈ {2,3}` variants were reported only as diagnostics. Several are genuinely strong **and sign-stable** (unlike the skip=1 family): L3d/S3d HAC t=5.01 (DSR 0.98, holds_sign=YES), L14d/S3d t=3.95 (DSR 0.80), L1d/S3d t=2.85 (DSR 0.87). The user has directed that these be tested properly. The honest way is to **promote `skip` to a selection axis (family m=21)**, charge the full multiple-testing, and run the strong candidates through the cost-aware backtest + their **unspent** OOS — reported as a **widened / post-hoc** analysis that does NOT overturn the original pre-registered null.

**Goal:** Determine whether any skip≥2 momentum variant is a genuinely deployable factor once charged the full multiple-testing correction AND validated net of (high, short-lookback) costs AND on out-of-sample data it has never seen.

**Candidates (factor-return series already exist in `data/factor/factor_returns.parquet`):** primary **L3d/S3d**; also **L14d/S3d**, **L1d/S3d** (the three that clear m=21 Bonferroni 0.05/21=0.00238); plus **L5d/S3d**, **L5d/S2d** as secondary.

**Hard rules:** TDD. No look-ahead. **OOS spent exactly once per variant** (skip≥2 OOS windows are currently unspent — keep them clean). Never open/print `.env`. Honest reporting — label everything post-hoc; keep the pre-registered null intact in the narrative. One-line commits, no Co-Authored-By. Push with `gh auth switch -u Jbrogz` first.

---

## Task V0: Widen the family to skip-as-axis (honest multiple-testing reframe)

**Files:** Modify `scripts/run_stage2.py` (or add a `--widened` mode); update `docs/STAGE2_RESULTS.md` with a clearly-labelled widened section.

- [ ] Compute Bonferroni + HLZ over the **full m=21 family** (threshold 0.05/21=0.00238); record which clear (expect L3d/S3d, L14d/S3d, L1d/S3d). Keep the original m=7 pre-registered result intact and ABOVE the widened one, with a paragraph stating the widened family is **post-hoc** (skip was originally a fixed convention) and is reported for completeness — it does not retroactively make the skip=1 null a false negative.
- [ ] Note that **DSR already deflates for the 21 trials** (so DSR is the multiple-testing-aware metric); list DSR per candidate.
- [ ] **Commit:** `feat: widened skip-as-axis multiple-testing reframe (post-hoc, m=21)`.

## Task V1: Cost-aware backtest of the skip≥2 candidates

**Files:** Modify `scripts/run_backtest.py` to accept the candidate list (add L3d/S3d, L14d/S3d, L1d/S3d, L5d/S3d); reuse `src/amom/backtest/{costs,engine,metrics}.py` unchanged.

- [ ] Backtest each candidate **in-sample** (gross vs net Sharpe, **annualized turnover** — expect it to be high for L1d/L3d, this is the key risk, **capacity**). The cost model + walk-forward vol targeting + t+1-close execution already exist — reuse them.
- [ ] Flag explicitly any candidate whose edge is killed net of costs (the short-lookback/turnover/reversal concern, guide §1.3).
- [ ] **Commit:** `feat: cost-aware backtest of skip>=2 momentum candidates`.

## Task V2: One-shot OOS per candidate

**Files:** `scripts/run_backtest.py` (extend the existing `read_oos_panels_once` single-use guard to each candidate; keep `open_count==1` per candidate run).

- [ ] For each candidate, read its OOS window **exactly once** and report the **IS-vs-OOS net Sharpe gap**. The skip≥2 OOS is unspent; do not iterate on it. Add 2×-cost and the regime breakdown (descriptive-proxy caveat) as for Stage 4.
- [ ] **Commit:** `feat: one-shot OOS validation of skip>=2 candidates`.

## Task V3: Honest results update

**Files:** Update `docs/STAGE4_RESULTS.md` (widened section), `docs/STATUS.md`, and regenerate the report/findings (`scripts/build_report.py`, `build_writeup_docx.py`) if a candidate is deployable.

- [ ] State the verdict per candidate: deployable / works-only-gross / fails-OOS. If L3d/S3d survives costs AND OOS, that is a genuine (post-hoc-discovered) positive — report it as such WITH the post-hoc + selection-bias caveat and a recommendation to confirm on forward data. If it dies net of costs/OOS, report that plainly (likely, given turnover).
- [ ] **Commit:** `docs: skip>=2 validation results + verdict`.

---

## Verification gate (must reach SOUND, per candidate)
3-lens review: (1) cost/turnover correctly applied (net<gross; turnover path-independent); (2) no look-ahead + OOS read exactly once per candidate; (3) honesty — widened family labelled post-hoc, pre-registered null intact, IS-vs-OOS and gross-vs-net side by side, short-lookback cost risk surfaced.

## Self-Review
- This is the single open work item from STATUS.md. It does not change any prior SOUND result; it adds a clearly-labelled widened/post-hoc analysis. The infrastructure (stats, costs, engine, OOS guard) all exists — this is mostly orchestration + honest framing, not new machinery.
