# ArtemisComp — Project Summary

A consolidated overview of every main file in this repo, what it does, and where
the project landed. Written 2026-06-01 by reading `data_pipeline.py`,
`test_data_pipeline.py`, the four notebooks, `PROJECT_SUMMARY.md`, and `status.md`.

---

## 1. What this project is

An entry for the **Artemis Analytics Quant Research Competition — Track 1 (Crypto
Factor Rebalancing)**. It builds and then rigorously stress-tests a single factor:

> **Funding-rate mean reversion on Hyperliquid perpetual futures.** Coins with
> extremely high funding (crowded longs) are expected to underperform; coins with
> very negative funding are expected to outperform. The signal is a z-score:
> `z = (short_window_mean − long_window_mean) / long_window_std` of daily funding.

The repo has two layers, and they reach **opposite conclusions** — that tension is
the point of the project, not a mistake:

- **The discovery layer** (`PROJECT_SUMMARY.md`, `signal_analysis.ipynb`,
  `parameter_sweep.ipynb`, `universe_sweep.ipynb`) mined parameters and found an
  optimistic-looking edge: **21d / 90d / 14d**, top-35 universe, Sharpe ≈ 1.9.
- **The honesty layer** (`validation.ipynb`, `status.md`) re-tests that claim with
  formal statistics and concludes the edge is a **data-mining artifact**.

**Bottom-line verdict: NOT DEPLOYABLE (2 of 4 decision gates passed).** The
competition explicitly rewards honest critical evaluation over impressive
backtests, so the null result is the deliverable.

---

## 2. File map

| File | Role |
|------|------|
| `data_pipeline.py` | Pulls funding rates + OHLCV from the Hyperliquid API and builds the analysis panel. Source of the data. |
| `test_data_pipeline.py` | 4 pytest alignment tests for `build_panel` forward returns (values, trailing NaN, no cross-symbol leak, gap→NaN). |
| `signal_analysis.ipynb` | Primary "discovery" analysis — quintiles, IC, L/S backtest, robustness. Now carries an honest-limitations section (cell 15). |
| `parameter_sweep.ipynb` | The original mining: 78–84 window/horizon combos; crowned 21/90/14. |
| `universe_sweep.ipynb` | Sweep of universe size (top 10–50 coins); picked top-35. |
| `validation.ipynb` | **The rigorous re-test** — HAC t-stats, IS-only sweep, BH/Holm, bootstrap, Deflated Sharpe, cost model, OOS, four-gate decision rule. |
| `PROJECT_SUMMARY.md` | Thesis + research narrative + findings — the human-readable write-up (competition context, the economic idea, the three-act test, the verdict). |
| `status.md` | Exhaustive dev log of the 10-task validation overhaul (~1000 lines). The authoritative record. |
| `Start.py` | Stub (imports only). |
| `data/` | `symbols.json` + 5 parquet files (raw funding, OHLCV, features, panel, reversion sleeve). |
| `pdfs/` | Competition brief + the Betting Against Beta reference paper (the Momentum PDF is left over from a dropped idea and is not used). |

---

## 3. Data pipeline (`data_pipeline.py`)

- **Source:** Hyperliquid `info` API (no key, works in the US; Binance is geo-blocked).
  Endpoints: `metaAndAssetCtxs` (universe + 24h volume), `fundingHistory` (hourly
  funding, paginated 500/req), `candleSnapshot` (daily OHLCV).
- **Coverage:** ~1 year (May 2025 – May 2026), top-35 coins by 24h volume.
- **Build steps:** `get_top_perp_symbols` → `pull_funding_rates` →
  `pull_ohlcv` → `build_funding_features` (daily funding + rolling z-score) →
  `build_panel` (merge + forward returns `ret_1d/7d/14d/21d`).
- **Known biases documented in-code:** survivorship (live universe excludes
  delisted coins), point-in-time (ranked by *today's* volume), and an in-sample
  selection disclosure on the tuned windows.
- **Fixes applied (2026-06-01):** corrected funding cadence docs (hourly, ~24/day,
  not 8h); fixed a forward-return label-alignment bug (`build_panel` now reindexes
  each symbol onto a contiguous daily grid so a gap-spanning `ret_Nd` becomes NaN
  instead of a wrong number). The fix is a no-op on the current gap-free panel
  (rebuilt panel is byte-identical), guarded by `test_data_pipeline.py`.

---

## 4. The discovery layer (what looked good)

From `PROJECT_SUMMARY.md` and the sweep notebooks:

- **Best params:** short 21d / long 90d / horizon 14d; mean IC ≈ −0.049, Q1–Q5
  spread ≈ 2%. Thesis: funding crowding builds and unwinds over *weeks*, so the
  original fast 7d/30d windows caught noise.
- **Best universe:** top-35 coins (Sharpe ≈ 1.91); small universes fail on
  diversification, very large ones dilute the signal.
- **Regime dependence (key honest caveat even here):** worked Jul–Dec 2025
  (range-bound), then drew down hard in Jan 2026 (BTC rally) — mean reversion loses
  when crowded longs are *right*. On a standalone factor (no offsetting trend
  exposure) this regime fragility is a first-order risk.

---

## 5. The honesty layer (`validation.ipynb`, 10 tasks)

Each task is a separate stress test; Task 9 rolls them into a verdict.

| # | Test | Result |
|---|------|--------|
| 1 | Survivorship / universe audit | 7 coins listed mid-study; live-universe bias documented (unfixable without historical snapshots). |
| 2 | Seal OOS window | IS ≈ 273 days / 9,139 rows; OOS ≈ 93 days from Feb 15, 2026 (partly "burned" by the original sweep). |
| 3 | IS-only re-sweep with HAC t-stats | 56 combos; **nothing significant** (best p=0.080), all top ICs negative, the mined 21/90/14 not in the top 10. |
| 4 | Multiple-testing correction (BH + Holm) | **0 of 56 survive** by any method → statistical null. |
| 5 | Stationary block bootstrap | Traded spread faintly **positive** (+0.00224/day, naive Sharpe 1.95) but **insignificant** (boot p=0.057, HAC p=0.073, CI straddles 0). |
| 6 | Deflated Sharpe Ratio | **47%** at N=140 trials (fails at N=56/84/140) ≪ 95% bar → **FAIL**. (Fixed a σ_SR bug in the plan's code.) |
| 7 | Transaction-cost model | ~18%/day turnover ≈ 6.5%/yr cost; net stays +94.8% (and +82.7% at 2× stress). **Costs do NOT kill it.** (Fixed a 2× double-count bug.) |
| 8 | Out-of-sample test | Edge **holds up positive** OOS (+12.7% cum, 76% of IS Sharpe) when z-scores warm up on IS history. **OOS does NOT kill it.** |
| 9 | Four-gate decision rule | **Gate 1 FAIL** (significance), **Gate 2 PASS** (OOS sign), **Gate 3 FAIL** (Deflated Sharpe), **Gate 4 PASS** (costs) → **2/4 = NOT DEPLOYABLE.** |
| 10 | Honest caveats written back into `signal_analysis.ipynb` | New "Methodological Limitations" section reconciled against the verified numbers. |

**The takeaway pattern:** a large in-sample return that *survives costs and even
carries into OOS* but *flunks the significance and search-effort tests* is exactly
what data-mining looks like. The strategy dies on **significance (Gate 1) +
overfitting penalty (Gate 3)** — not on costs or OOS. The originally "winning"
21/90/14 was cherry-picked noise, carried forward only to document the null
honestly.

---

## 6. Status & logistics

- **Git:** under version control, pushed to a **private** repo
  `github.com/Dcon42/artemis-daniel` (branch `main`). Notebooks committed with
  outputs (~1.5 MB) — consider `nbstripout` if diffs get noisy.
- **Environment:** conda env `artemis`, Python 3.11; `statsmodels 0.14.6`,
  `arch 8.0.0`, `scipy 1.17.1`, pandas/numpy/matplotlib, pyarrow for parquet.
- **Tests:** `pytest test_data_pipeline.py` → 4/4 pass.
- **Remaining competition deliverables** (per `PROJECT_SUMMARY.md`): write the
  research report (PDF) and build the pitch deck. Scope is now the mean-reversion
  factor alone (the planned momentum pairing was dropped).

> **Where to look first:** `status.md` is the authoritative, detailed record
> (verdict + per-task findings). `validation.ipynb` holds the actual tests.
> `PROJECT_SUMMARY.md` is the older optimistic framing — read it knowing the
> honesty layer supersedes its conclusions.
