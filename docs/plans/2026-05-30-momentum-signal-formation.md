# Momentum Signal + Portfolio Formation Plan (Stage 1.2–1.4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Apply karpathy-guidelines (minimal real code, surgical, TDD). The universe foundation is SOUND (registry, point-in-time eligibility, survivorship, death signal, spot returns module). This plan builds the momentum factor on top of it.

**Goal:** Produce a clean dollar-neutral long/short **factor-return series for each momentum variant**, with no look-ahead, on the sound Artemis universe + spot returns — ready for the Stage-2 significance battery.

**References (read first):** `docs/specs/2026-05-30-artemis-momentum-design.md` §4 Stage 1.2–1.4, §3.1–3.2, §7. Reference implementations to PORT (parent repo, strip funding): `../Project 1/factor-book-jack/factor-eval/factors/momentum.py` (signal) and `.../momentum_portfolio.py` (formation). Existing in-repo: `src/amom/universe/builder.py` (eligibility panel), `src/amom/returns/spot.py` (holding returns).

**Hard rules:** TDD (real test → fail → minimal impl → pass → commit). Never open/print `.env`. One-line commits, no Co-Authored-By. No stubs/placeholders. Match existing `amom` style.

---

## Task S0: Panel refresh + durable survivorship disclosure + holding-return panel

**Files:** Modify `scripts/build_universe.py`; Create `docs/AUDIT.md`, `scripts/build_returns.py`.

- [ ] Fold in the verified nice-to-haves: in `build_universe.py` disclosure, **report two cohorts separately** — `delisted_asof`-ever-True (stopped reporting) vs sustained >90%-drawdown-but-still-printing (zombie). Fix the `build_panel` return annotation (`-> tuple[pd.DataFrame, pd.DataFrame]`) and the stale `build_universe_history` docstring column list.
- [ ] Create `docs/AUDIT.md` and write the survivorship disclosure durably: total assets, ever-eligible, terminal->90% collapse count, delisted-vs-zombie split, and an explicit "purged-dead-coins are unrecoverable from Artemis's as-of-today catalog — unestimable lower bound" note. (Spec §3.6/§10 promise this in a versioned file.)
- [ ] **Regenerate** `data/universe/universe_history.parquet` via `build_universe.py` so the on-disk panel carries the current 8-column schema (incl. `left_censored`).
- [ ] Create `scripts/build_returns.py`: pull `PRICE` for all registry `artemis_id`s, run `returns.spot.build_holding_returns(price_panel, universe_panel)`, write `data/returns/holding_returns.parquet` (long: `date, symbol, holding_return`). Print rows/#symbols/date range; confirm a collapsed coin (terra) shows its ~-99.99% realized terminal return.
- [ ] **Commit:** `feat: cohort-split survivorship disclosure (docs/AUDIT.md) + holding-return panel`.

**Success:** durable, quantified survivorship disclosure exists; the holding-return panel is built on the sound universe with crash returns carried.

## Task S1: Momentum signal grid (Stage 1.3)

**Files:** Create `src/amom/factor/__init__.py`, `src/amom/factor/momentum.py`, `tests/test_factor_momentum.py`.

Port the signal from the reference `momentum.py` (it already uses PRICE returns — no funding to strip in the signal). Build `build_momentum_signal(price_panel, lookback_days, skip_days) -> wide signal panel (dates × symbols)` = log-sum of daily price returns over `[t-skip-lookback, t-skip]`, i.e. the trailing return skipping the most recent `skip_days`. Expose the frozen grid from config: `LOOKBACKS_DAYS=(1,3,5,7,14,28,56)`, `PRIMARY_SKIP_DAYS=1`, `ROBUSTNESS_SKIPS=(2,3)`.

- [ ] **Tests first:** (a) signal at date `t` uses only data ≤ close `t` — mutating any price after `t` leaves `signal[t]` unchanged (no look-ahead); (b) a hand-computed small series matches the log-sum-over-lookback-with-skip value; (c) the skip correctly excludes the most recent `skip_days`; (d) NaN where insufficient history. Watch FAIL.
- [ ] **Implement** `momentum.py`; add the grid constants to `config.py`. Watch PASS.
- [ ] **Commit:** `feat: momentum signal grid (lookback x skip), no look-ahead`.

**Success:** the 7-lookback signal is computed point-in-time from Artemis prices, grid frozen in config.

## Task S2: Dollar-neutral long/short formation (Stage 1.4)

**Files:** Create `src/amom/factor/portfolio.py`, `tests/test_factor_portfolio.py`; Create `scripts/build_factor_returns.py`.

Port `momentum_portfolio.py`, **stripping funding**: factor P&L uses the spot `holding_return` panel (Task S0), not a funding-adjusted return. Construction: quintile sort (`QUANTILE=0.20`), long top / short bottom, equal-weight within leg, dollar-neutral (Σweights=0), `factor_return = long_leg − short_leg`. **Lag:** signal computed through close `t` drives entry at close `t+1` (the holding window is `(t, t+HOLDING_DAYS]`), eligibility-masked per the universe panel, min-bucket gate (`MIN_BUCKET_SIZE`).

- [ ] **Tests first:** (a) **dollar-neutrality** — long and short legs have equal dollar weight, Σweights ≈ 0; (b) **no look-ahead** — the signal at `t` (not `t+1`) selects buckets and entry is at `t+1`; mutating post-`t` signal data does not change the bucket chosen for the `t→t+1` trade; (c) equal-weight within leg; (d) a collapsed coin in the short leg contributes its crash return to P&L (survivorship payoff flows through); (e) rebalance skipped when a bucket < `MIN_BUCKET_SIZE`. Watch FAIL.
- [ ] **Implement** `portfolio.py`. Watch PASS.
- [ ] Create `scripts/build_factor_returns.py`: for each of the 7 lookbacks at skip=1 (+ diagnostics {2,3}), build the signal (S1) and the long/short portfolio, compute the per-rebalance factor-return series, write `data/factor/factor_returns.parquet` (`variant, rebalance_date, factor_return, long_return, short_return, n_long, n_short`). Run live; print per-variant n_obs, mean, annualized Sharpe (gross, pre-significance).
- [ ] **Commit:** `feat: dollar-neutral momentum long/short formation + factor-return series`.

**Success:** each variant has a clean, gap-free, dollar-neutral factor-return series with the no-look-ahead lag enforced; survivorship crashes flow into short-leg P&L.

---

## Verification gate (must reach SOUND)

3-lens adversarial review on the signal + formation:
- **look-ahead:** signal(t) ≤ close t; entry at t+1; mutating future data changes nothing at t. Per-date, not just last-date.
- **construction:** dollar-neutrality, equal-weight, quintile thresholds, factor_return = long−short, min-bucket gate, eligibility integration — all correct and tested non-vacuously.
- **spot/survivorship:** no funding term anywhere (spot); collapsed coins' crash returns flow into leg P&L; conventions (simple XS, log compounding) not mixed.

Proceed to Stage 2 (significance battery) only when synthesis = SOUND.

## Self-Review
- Covers spec §4 Stage 1.2 (holding panel — S0), 1.3 (signal — S1), 1.4 (formation — S2). Nice-to-haves from the universe gate folded into S0.
- Type names consistent: `holding_return`, `factor_return`, `signal panel`, `LOOKBACKS_DAYS`, `PRIMARY_SKIP_DAYS`, `QUANTILE`, `HOLDING_DAYS`, `MIN_BUCKET_SIZE`.
- Stage 2 (stats), 3 (deploy), 4 (backtest), 5 (report) follow in subsequent plans/workflows.
