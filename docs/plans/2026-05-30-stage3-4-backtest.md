# Stage 3–4 — Deployment Characterization + Cost-Aware Backtest Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Apply karpathy-guidelines (no speculative machinery — the in-sample result is a NULL, so build the lean honest backtest, not an elaborate combiner). The Stage-2 verdict is in `docs/STAGE2_RESULTS.md`: **no selection-family variant survives HAC+Bonferroni on its own terms; the strongest, L5d (skip=1), is suggestive but sign-unstable → deployment-disqualified.**

**Goal:** Complete the evidence honestly. Characterize the strongest in-sample candidate net of realistic spot costs, **spend the sealed OOS window exactly once**, and report gross-vs-net and in-sample-vs-OOS — quantifying that momentum is (almost certainly) not a deployable factor on the Artemis spot universe.

**Candidates to backtest:** primary = `momentum_L5d_S1d` (strongest in-sample); comparator = `momentum_L28d_S1d` (academic 4-week canonical). No multi-factor combination (single factor, and a null) — Stage-3 reduces to volatility targeting on the candidate (guide §3.5), which we include; richer weighting is N/A and we say so.

**References:** spec §3.3–3.6, §4.1–4.6, §3.5 (spot liquidity), §3.2 (t+1-close); guide §3–§4. Reuse `src/amom/stats/core.py` (sharpe/drawdown/calmar), `src/amom/factor/portfolio.py` (formation), the universe + holding-return panels.

**Hard rules:** TDD. Never open/print `.env`. One-line commits, no Co-Authored-By. Do NOT push. **OOS is spent EXACTLY ONCE, in Task B4 — no earlier task may read rows ≥ `OOS_START`.** No stubs. Report net-of-cost and OOS honestly; do not flatter.

---

## Task B0: Fold Stage-2 reporting-precision fixes (quick)

**Files:** Modify `docs/STAGE2_RESULTS.md`, `scripts/run_stage2.py`, `src/amom/stats/sharpe_se.py`.

- [ ] In `STAGE2_RESULTS.md` headline, annotate the lone survivor: `Bonferroni survivors: momentum_L5d_S1d (DISQUALIFIED — fails §2.6 sign-stability; survives only via the bootstrap-override rule, HAC p=0.0081 > 0.00714)`. Relabel `Total tests run` → `Total mean-return tests (one per variant): 21`.
- [ ] In `run_stage2.py`, slice the holding panel to `date < OOS_START` before `build_market_return`/`build_size_control` so the last in-sample spanning window consumes no OOS-dated daily row (close the boundary nuance), OR if behavior is identical, fix the misleading comment. Add a one-line ddof-convention note in `sharpe_se.py`.
- [ ] **Commit:** `docs: honest Stage-2 survivor annotation + spanning OOS-boundary slice`.

## Task B1: Spot cost model

**Files:** Create `src/amom/backtest/__init__.py`, `src/amom/backtest/costs.py`, `tests/test_backtest_costs.py`; add cost params to `config.py`.

- [ ] Config: `TAKER_FEE_BPS=10` (per side, spot), `SLIPPAGE_TOP_BPS=5`, `SLIPPAGE_SMALL_BPS=15`, `SLIPPAGE_TOP_N=30`, `SLIPPAGE_ADV_REF=0.01` (order/ADV reference for scaling). All by convention; disclosed.
- [ ] `costs.trade_cost(traded_notional, adv, liquidity_rank, aum)` → fee (bps on notional) + size-scaled slippage (tier by `liquidity_rank` vs `SLIPPAGE_TOP_N`, scaled with order/ADV). No funding term (spot — disclosed).
- [ ] **Tests:** fee is symmetric per side; slippage higher for small/illiquid names; slippage scales up with order/ADV; zero trade → zero cost. Commit: `feat: spot taker-fee + size-scaled slippage cost model (no funding)`.

## Task B2: Backtest engine (positions, t+1-close execution, vol targeting)

**Files:** Create `src/amom/backtest/engine.py`, `tests/test_backtest_engine.py`.

- [ ] `engine.run_backtest(weights_by_rebal, holding_returns, universe_panel, *, aum, cost_model, vol_target)` →: convert candidate weights to per-coin positions with `PER_COIN_CAP` and `GROSS_LEVERAGE_CAP=2.0`; **volatility-target** the book to `ANNUAL_VOL_TARGET` using **trailing realized vol (past data only, walk-forward)**; execute at **t+1 close** with the cost model on traded notional (trade list = target − current); carry forward with price P&L; retain equity curve, position history, trade log.
- [ ] **Tests (discriminating):** dollar-neutrality each rebalance; **gross ≤ 2×** asserted every period; vol scalar uses only trailing (≤ t) data (mutating future returns doesn't change today's scalar — no look-ahead); costs reduce net vs gross; a collapsed short-leg coin's crash flows into P&L. Commit: `feat: cost-aware backtest engine with walk-forward vol targeting`.

## Task B3: Net-of-cost metrics + capacity

**Files:** Create `src/amom/backtest/metrics.py`, `tests/test_backtest_metrics.py`.

- [ ] `metrics.performance(equity, trades, returns)` → total & annualized return, ann vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, avg win vs avg loss, **annualized turnover** (all NET of costs; reuse `stats.core` for sharpe/drawdown/calmar).
- [ ] `metrics.capacity(candidate, cost_model)` → sweep AUM, recompute size-scaled slippage per coin via order/ADV, return the AUM at which **net** expected return crosses zero.
- [ ] **Tests:** turnover computed from the trade log; capacity monotonic in AUM (more AUM → more slippage → lower net); Sortino/Calmar signs sane. Commit: `feat: net-of-cost performance metrics + capacity estimate`.

## Task B4: Robustness + one-shot OOS runner (spends OOS exactly once)

**Files:** Create `scripts/run_backtest.py`, `tests/test_run_backtest.py`; Create `docs/STAGE4_RESULTS.md`.

- [ ] `scripts/run_backtest.py`: backtest `L5d_S1d` (primary) and `L28d_S1d` (comparator). Report **in-sample** net metrics (gross vs net side by side). Then **read the OOS window EXACTLY ONCE** (rows ≥ `OOS_START`), backtest there, and report the **in-sample-vs-OOS Sharpe gap** (a near-zero/negative OOS Sharpe is reported as overfitting). Robustness: **2× costs** rerun; **±50% lookback** rerun (reuses the chosen spec); **regime breakdown** (bull/bear/chop by trailing market-return sign × realized-vol terciles, config-defined). Write `data/backtest/{equity,positions,trades}.parquet` + `docs/STAGE4_RESULTS.md`.
- [ ] **Tests:** the OOS slice is read in exactly one code path (a single-use guard / counter); gross ≥ net; the robustness reruns reuse the chosen spec (not new selection). Run live. Commit: `feat: cost-aware backtest + one-shot OOS + robustness (Stage 4)`.

---

## Verification gate (must reach SOUND)

3-lens adversarial review:
- **cost-model & metrics:** fees/slippage applied correctly to traded notional; net < gross; capacity monotonic; turnover/Sharpe/Sortino/Calmar correct; no funding term.
- **no-look-ahead / OOS-once:** vol scalar and all decisions use only ≤ t data; execution at t+1 close; the **OOS window is read exactly once** and never by B0–B3; the in-sample-vs-OOS gap is reported.
- **honesty:** gross-vs-net and IS-vs-OOS shown side by side; any variant that "works only gross" stated; the conclusion does not overstate a null; survivorship still flows into P&L.

Proceed to Stage 5 (report) when synthesis = SOUND. The expected honest outcome — momentum is not deployable net of costs / OOS — is reported plainly.

## Self-Review
- Covers spec §4.1 (B2 positions/caps), §4.2 (B1 costs), §4.3 (B2 t+1-close), §4.4 (B2 loop/artifacts), §4.5 (B3 metrics+capacity), §4.6 (B4 robustness+OOS-once); §3.5 vol targeting (B2). Multi-factor combination (§3.2–3.3) is N/A for a single null factor — stated, not built (YAGNI).
- Type names: `run_backtest`, `trade_cost`, `performance`, `capacity`, `GROSS_LEVERAGE_CAP`, `ANNUAL_VOL_TARGET`, `OOS_START`, `TAKER_FEE_BPS` — consistent.
