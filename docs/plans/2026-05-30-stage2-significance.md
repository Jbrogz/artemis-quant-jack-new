# Stage 2 — Statistical Significance Battery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Apply karpathy-guidelines. The factor-return series for 21 variants (7 lookbacks × 3 skips, 99 non-overlapping 30-day obs each) are built and SOUND at `data/factor/factor_returns.parquet`. This plan answers the project's core question — **is Artemis momentum a true positive expected return, or a sample artifact?**

**Goal:** Run the full guide §2 significance battery on the momentum variants — Newey-West HAC (at the correct bandwidth), Lo (2002) Sharpe SE, spanning regression vs {market, size control}, stationary block bootstrap, Bonferroni + Harvey-Liu-Zhu multiple-testing, subsample stability, DSR/PBO overfitting — on the **in-sample** slice only, with the **OOS window reserved and sealed**. Report whatever it shows; an honest null is a valid result.

**References (read first):** `docs/specs/2026-05-30-artemis-momentum-design.md` §2.0–§2.8, §3.2 (single-factor spanning), §7. PORT from the author's earlier private factor-eval `stats.py` (hac_tstat, ols_tstat_hac, newey_west_se, bonferroni_correction, classify_tstat_hlz, sharpe/drawdown/calmar) and `../src/cmom/overfitting/{dsr.py,pbo.py}` (bootstrap.py = cross-check only; arch is the bootstrap of record).

**Hard rules:** TDD. Never open/print `.env`. One-line commits, no Co-Authored-By. Do NOT push. **OOS DISCIPLINE: no code in this stage may read factor-return rows dated ≥ `OOS_START`; selection uses in-sample only.** No stubs.

---

## Task T0: Port stats core + overfitting; reconcile holding-return column

**Files:** Create `src/amom/stats/__init__.py`, `src/amom/stats/core.py`, `src/amom/stats/dsr.py`, `src/amom/stats/pbo.py`, `tests/test_stats_core.py`. Minor: reconcile `returns/spot.py` column name to `holding_return` if it emits `ret` (Stage-1 nice-to-have).

- [ ] Port `stats/core.py` from `factor_eval/stats.py` verbatim (fix imports): `hac_tstat`, `newey_west_se`, `ols_tstat_hac`, `bonferroni_correction`, `classify_tstat_hlz`, `sharpe_ratio`, `max_drawdown`, `calmar_ratio`, `rolling_sharpe`. Port `dsr.py`, `pbo.py` from `cmom/overfitting/`.
- [ ] **Test:** `hac_tstat(series, bandwidth=L)` equals `statsmodels.OLS(y, add_constant).fit(cov_type='HAC', cov_kwds={'maxlags':L})` intercept t-stat to ~6 dp (the guide's §2.2 equivalence). Bonferroni: m = #finite p-values, threshold α/m, survivors p ≤ threshold. HLZ tiers t≥3 / 2<t<3 / else. Watch FAIL → port → PASS.
- [ ] **Commit:** `feat: port stats core (Newey-West, Bonferroni, HLZ) + DSR/PBO into amom`.

## Task T1: HAC bandwidth rule + Lo (2002) Sharpe SE (autocorrelation-aware)

**Files:** Create `src/amom/stats/sharpe_se.py`, `tests/test_stats_sharpe_se.py`.

- [ ] `maxlags_for(n_obs, holding_obs)` = `max(holding_obs - 1, ceil(n_obs ** 0.25))` (spec §2.0 — bandwidth covers holding-period overlap). For non-overlapping 30-day obs `holding_obs=1`, so it reduces to ⌈T^¼⌉; keep the rule general.
- [ ] `lo_sharpe_se(returns, periods_per_year)`: iid SE `sqrt((1 + 0.5·SR²)/T)`; if Ljung-Box (or ACF) flags autocorrelation, use Lo's autocorrelation-corrected SE instead (spec §2.3). Return `(sharpe, se, used_autocorr_correction)`.
- [ ] `effective_n_and_power(returns, holding_obs)`: report effective non-overlapping n and approximate power; expose `MIN_EFFECTIVE_N` in config; a variant below it is labelled `"inconclusive (underpowered)"`.
- [ ] **Tests:** iid SE matches the closed form; an autocorrelated series triggers the corrected SE; the maxlags rule; underpowered labelling. Commit: `feat: Lo (2002) Sharpe SE (autocorr-aware) + HAC bandwidth + power labelling`.

## Task T2: Spanning regression vs {market, size control}

**Files:** Create `src/amom/stats/spanning.py`, `tests/test_stats_spanning.py`; helper to build market + size control.

- [ ] Build (from the holding-return + universe panels, point-in-time, in-sample): `market_return` = equal-weighted return of the eligible universe per rebalance window; `size_control` = a small-minus-big long/short return from `MC` quintiles (TEST-ONLY regressor, never deployed — spec §2, §3.2).
- [ ] `spanning_alpha(factor_returns, regressors_df, bandwidth)` -> `{alpha, alpha_tstat (HAC), betas, r2, n}` via OLS with Newey-West SE (reuse `newey_west_se`). Keep the variant only if alpha is significant; disclose this is a reduced LTW set.
- [ ] **Tests:** a factor that is purely `2×market` has alpha≈0; a factor orthogonal to the regressors keeps its mean as alpha; HAC t matches `ols`-with-HAC. Commit: `feat: spanning regression vs market + size control (test-only)`.

## Task T3: Stationary block bootstrap (arch — bootstrap of record)

**Files:** Create `src/amom/stats/bootstrap.py`, `tests/test_stats_bootstrap.py`; add `arch` dep.

- [ ] `stationary_bootstrap_pvalue(returns, *, reps, block_size, seed)` using `arch.bootstrap.StationaryBootstrap` → one-sided empirical p-value for mean>0. This is the bootstrap of record; `cmom` bootstrap may only cross-check.
- [ ] `disagrees(hac_p, boot_p, threshold)` = the two fall on opposite sides of the (Bonferroni-adjusted) threshold; on disagreement the **bootstrap** is the reported verdict (spec §2.7).
- [ ] **Tests:** a strongly-positive iid series → small bootstrap p; a zero-mean series → p≈0.5; determinism under fixed seed; the disagreement rule. Commit: `feat: arch stationary block bootstrap p-value + NW-disagreement rule`.

## Task T4: OOS reserve, subsample stability, and the battery runner

**Files:** Create `src/amom/stats/subsample.py`, `scripts/run_stage2.py`, `tests/test_stats_subsample.py`, `tests/test_run_stage2.py`; Modify `config.py` (freeze `OOS_START`, `MIN_EFFECTIVE_N`).

- [ ] **Freeze OOS_START:** compute the rebalance date at the 70th percentile of the (sorted, deduped) rebalance dates and write it as a literal constant `OOS_START` in `config.py` (sealed; spec §2.8). Add a guard helper `in_sample(df)` that returns rows with `rebalance_date < OOS_START`, and assert no Stage-2 path consumes `>= OOS_START`.
- [ ] `subsample.sign_stability(returns)`: split into halves and thirds; return per-subsample mean sign; `holds_sign` True iff all same sign as full. A sign-flip **disqualifies** the variant from deployment (spec §2.6).
- [ ] `scripts/run_stage2.py`: on the **in-sample** slice, for each of the 21 variants compute — naive t; **HAC t** at `maxlags_for`; mean & ann. return; Lo Sharpe + SE + autocorr flag; **spanning alpha + HAC t**; **bootstrap p** + disagreement verdict; subsample signs + holds_sign; HLZ tier; effective-n/power label. Apply **Bonferroni on the pre-registered selection family** (7 lookbacks at skip=1, m=7) and record the **total** test count. Run **DSR** and **PBO/CSCV** across the variant grid. Write `data/stats/significance.parquet` and a human-readable `docs/STAGE2_RESULTS.md` table (variants incl. failures, # tests, survivors, the honest verdict).
- [ ] **Tests:** subsample sign logic; `in_sample` excludes OOS rows; the runner produces one row per variant with all columns; Bonferroni m == 7 for the selection family. Run live (in-sample). Commit: `feat: Stage-2 battery — OOS reserve, subsample, multiple-testing, DSR/PBO runner`.

---

## Verification gate (must reach SOUND)

3-lens adversarial review:
- **stat-correctness:** HAC t matches statsmodels; Lo SE formula correct + autocorr-triggered; spanning alpha/HAC correct; arch bootstrap correct & deterministic; Bonferroni m and HLZ tiers correct; DSR/PBO sensible. No vacuous tests.
- **OOS/selection discipline:** `OOS_START` frozen; **no Stage-2 code reads rows ≥ OOS_START**; selection family pre-registered (m=7); total tests recorded; bootstrap-overrides-NW on disagreement.
- **honesty:** naive vs HAC distinction reported; underpowered variants labelled "inconclusive" not "insignificant"; failures included; survivorship still intact in the underlying series.

Proceed to Stage 3/4 (deploy + cost-aware backtest, where OOS is used exactly once) only when synthesis = SOUND. The verdict (significant / suggestive / inconclusive / null) is reported faithfully regardless of which it is.

## Self-Review
- Covers spec §2.0 (T1 bandwidth/power), §2.1–2.2 (T0 naive+HAC), §2.3 (T1 Lo SE), §2.4 (T2 spanning), §2.5 (T0 Bonferroni/HLZ + T4 runner records m/total), §2.6 (T4 subsample), §2.7 (T3 bootstrap), §2.8 (T4 OOS reserve) + DSR/PBO (T0 port, T4 run).
- Type names: `factor_returns`, `holding_return`, `OOS_START`, `maxlags_for`, `lo_sharpe_se`, `spanning_alpha`, `stationary_bootstrap_pvalue`, `sign_stability`, `in_sample` — consistent across tasks.
