# Artemis Momentum Factor Book — Design Spec

_Date: 2026-05-30. Status: rev 3 — universe corrected after live-Artemis verification (Appendix B)._
_Authoritative methodology: a private factor-book methodology guide (the author's prior work). This rev incorporates an adversarial_
_vetting pass against that methodology — see Appendix A for the findings and their resolutions._

---

## 1. Objective

Determine, with full statistical rigor and **net of realistic trading costs**, whether a
**cross-sectional momentum factor** on an **Artemis-sourced crypto spot universe** has a
*true* positive expected return — as opposed to a sample artifact.

The null hypothesis for every test is **a true mean factor return of zero**. We build the
machinery to find the truth and report whatever it shows. If momentum is insignificant
(as in the author's prior private research), **that honest null is the deliverable.**

This project follows that private factor-book methodology guide **strictly**, including all anti-bias
and anti-look-ahead requirements. It differs from the guide only where the data source forces
it, and every such difference is documented (Section 3).

## 2. Scope

**In scope:** the **momentum factor only** — construction, the statistical significance
battery, in-sample deployment choice + vol targeting, a cost-aware backtest, and a written
research report (PDF). A reproducible code repository is a deliverable.

**Out of scope (this round):** size, short-term reversal, betting-against-beta, carry **as
deployed factors**; the pitch deck; any multi-factor *combination*. The existing "momentum
sleeve" architecture (ridge composite, 2-of-3 TS gate, sizing, risk overlay) is **not reused**.

**Note on the size control:** a small-minus-big **size control** is built from Artemis market
cap solely as a **regressor inside the Stage-2.4 spanning regression** (a significance test).
It is never formed as a portfolio, never deployed, never reported as a strategy. This is a
test instrument, not a second factor, and does not breach momentum-only scope.

## 3. The forced adaptations (Artemis-only) — all disclosed

Per "source all data from Artemis; if Artemis lacks required data, note it and build without
it," the deviations from the guide are:

### 3.1 No funding data → spot holding returns, no Carry factor
Artemis exposes price, market cap, FDV, volume, and on-chain/fundamental metrics, but **not
perpetual funding rates**. Therefore:
- **Holding return = simple spot price return** (guide §1.2's funding-adjusted return is not
  computable; we use the unfunded spot return and state so).
- **No Carry factor** (out of scope here anyway).
- The momentum **signal** is unaffected (always built from price returns, not funding).

### 3.2 Daily price granularity → execution at t+1 close
The Artemis API serves daily (`DAY`) granularity; intraday open / first-hour VWAP is not
available. The guide's "enter at the open of t+1" (§1.4, §4.3) is therefore realized as
**fill at the close of t+1, with the slippage model applied** — a clean one-period lag that
still fully honors the no-look-ahead rule. Disclosed as a convention.

### 3.3 Single factor → Stage 3 "combination" reinterpreted
With momentum only, Stage 3 ("combine surviving factors") collapses to **selecting the
momentum specification to deploy** — the single best in-sample variant, or a risk-weighted
multi-horizon momentum composite — plus volatility targeting. All selection uses **in-sample
data only** and is decided by convention/robustness, never by maximizing backtest Sharpe
(guide §3.6). Stage-5 wording uses "deployed strategy," not "combined portfolio."

### 3.4 Wrapped-token exclusion (deliberate addition)
In addition to the guide's stablecoin exclusion (§1.1), we exclude **wrapped tokens** (WBTC,
WETH, stETH, …) as redundant price exposures of their underlyings. This is a deliberate
deviation from the guide's exclusion set, disclosed here and in the report. The guide's
unlock / unverified-circulating-supply detector (§1.3) is a **Size-factor** concern and is
out of scope; we note that unlock-driven price jumps are a residual momentum-signal risk and
flag any coin whose market cap jumps mechanically, without excluding it.

### 3.5 Liquidity filter rebuilt on `24H_VOLUME` + market cap (forced by data)
A live Artemis investigation (Appendix B) found that `30D_VOLUME` is **real-time only**
(returns a sentinel on any historical pull) and `24H_VOLUME` — the only historical volume
metric — is **unreliable across the cross-section** (sub-dollar prints, flatlined series such
as `for` pinned at 75.66). A strict "$1M trailing-30d mean ADV" filter (guide §1.1) is
therefore not faithfully computable. We instead gate liquidity on **(a) a market-cap floor**
(`MC` is dense and stable) **and (b) a trailing-30d _median_ of `24H_VOLUME`** above a
threshold — the median survives broken single-day prints where a mean would not. Broken /
sub-dollar volume prints are winsorized/flagged. This deviation from the guide's mean-ADV rule
is disclosed in the report.

### 3.6 No point-in-time catalog → as-of-today enumeration, residual survivorship disclosed
The Artemis `/asset` catalog (1,013 assets, keyed by a stable `artemis_id` slug) is the
**programmatic, Artemis-native universe enumeration** — it replaces any hand-curated symbol
list. But it is an **as-of-today** snapshot with no listing/delisting dates, so it cannot give
a true point-in-time *membership* set, and coins fully purged from Artemis before query are
unrecoverable. We reconstruct per-asset listing dates from first-observed price and keep every
catalog asset that ever collapsed (with its crash carried), but the residual survivorship from
purged-dead-coins is **quantified and disclosed** (§10), not hidden. Recycled tickers (a single
ticker reused by a new project, e.g. `ust`) are split into distinct synthetic assets by a
terminal-drawdown-to-near-zero + gap detector, with `artemis_id` / `coingecko_id` used to
confirm continuity. Where Artemis already separates twins (`lunc`→`terra` classic vs
`luna`→`terra2` revival), the `artemis_id` is preferred.

## 4. Stage-by-stage requirements (mapped to the guide)

### Stage 1 — Construct the momentum portfolio

**1.1 Universe — point-in-time, dead coins included (guide §1.1).**
- **Enumerate the universe programmatically from Artemis `/asset`** (1,013 assets), persisting
  an `artemis_id`-keyed registry (`artemis_id`, `symbol`, `coingecko_id`, `title`). Everything
  downstream keys on the stable `artemis_id`, never the mutable ticker. This is the
  Artemis-native "widest available asset list" — no hand-curated symbol list (§3.6).
- **Reconstruct each asset's listing date** from its first observed Artemis price.
- **Keep every asset that later collapsed**, final return reflecting the collapse (a 90% crash
  is a −90% return, not a dropped value). **Split recycled tickers** (terminal drawdown→~0 +
  gap, confirmed via `artemis_id`/`coingecko_id`) into distinct synthetic assets so a revived
  ticker's healthy series is never spliced onto a dead asset's crash (§3.6).
- Point-in-time eligibility filters, evaluated as-of each date and **rebuilt daily**:
  ≥ `MIN_HISTORY_DAYS` (90) price history **with a minimum observation density** (not just
  calendar age); **liquidity = `MC` floor AND trailing-30d _median_ `24H_VOLUME`** above
  threshold (§3.5, not mean-ADV); a **tradeability/staleness check** (a non-NaN price within a
  short grace window of the date, so a stopped/dead coin exits eligibility promptly); exclude
  stablecoins and wrapped tokens.
- **Death signal carried in the panel:** the eligibility panel includes `price_last_date` and a
  point-in-time `delisted_asof` flag so the returns layer (§Stage 1.2) knows exactly where to
  book the terminal crash return.
- **Minimum-universe gate (point-in-time):** a rebalance date is gated if fewer than
  `MIN_ELIGIBLE_NAMES` are eligible, so each quintile has ≥ `MIN_BUCKET_SIZE` names — derived
  from `MIN_BUCKET_SIZE`, not a coincidental constant. As-of-date info only. Thin early-history
  quintiles are disclosed; effective names/quintile over time is reported.
- The eligibility panel is materialized on a **daily** grid (guide §1.1 "rebuild daily");
  rebalance formation downsamples from it.
- **Documented, quantified limitation (§10):** the `/asset` catalog is as-of-today, so assets
  fully purged from Artemis before query are unrecoverable; residual survivorship is quantified
  (e.g. count of catalog assets showing terminal collapses vs an estimate of purged names), not
  hidden.

**1.2 Returns (guide §1.2).**
- Holding return = **simple spot price return** (no funding; §3.1).
- **Simple returns** to aggregate across coins within a period; **log returns** to compound a
  series through time. Never mixed.
- A coin going to ~0 contributes its realized crash return; a coin leaving the eligible set
  mid-hold has its position closed at the last observed price.

**1.3 Momentum signal (guide §1.3).**
- Signal = trailing return over a fixed lookback, **skipping the most recent days**.
- **Lookback grid fixed in advance** (not selected from backtest): academic LTW baseline
  **14, 28, 56 days** (2/4/8 weeks) plus crypto-short **1, 3, 5, 7 days** → **7 lookbacks**.
- **Skip is a nuisance parameter fixed by convention: `PRIMARY_SKIP_DAYS = 1`.** Skips
  {2, 3} are run only as a **separately-reported robustness check**, not as a selection axis.
- Signal = log-sum compounding of daily price returns over the lookback, shifted by the skip.
  Ported from the author's earlier `momentum.py`.

**1.4 Portfolio formation (guide §1.4).**
- Sort the eligible universe by signal into **quintiles** (top/bottom 20%). Long top, short
  bottom. **Equal-weight** within each leg. **Dollar-neutral.**
- Cross-sectional ranking uses the cumulative **simple** return; log/simple ranking is
  monotone so the sort is identical — noted to avoid a convention-mix appearance.
- `factor_return = long_leg_return − short_leg_return`.
- **No look-ahead, enforced in code (guide §1.4):** signal from data through **close of t**;
  position entered at **close of t+1** (§3.2). The signal function for t must not access any
  data after t's close. The one-period gap is mandatory and unit-tested. (Reference
  `momentum_portfolio.py` enforces the lag; ported with funding removed.)

**Stage 1 completion criteria:** panel includes delisted/collapsed coins with final returns;
eligibility is point-in-time and rebuilt daily; holding returns are spot (funding noted
unavailable); the momentum factor produces a clean long/short series with no gaps; the
signal→execution gap is enforced in code.

### Stage 2 — Test for statistically significant expectancy

**Pre-registered selection grid (decided before any OOS access):** the **7 lookbacks at
`PRIMARY_SKIP_DAYS=1` and the canonical 30-day hold** → `m_select = 7`. Bonferroni operates
on this family (threshold `0.05 / 7`). Holding-period sweeps, skip {2,3}, and quantile-breadth
variants are **diagnostics**, reported for transparency and counted in the recorded *total*
test count, but they do **not** drive deployment selection and do **not** relax the selection
threshold (guide §2.5: record the total number of tests; selection family is pre-registered).

#### 2.0 Statistical conventions (resolves the overlapping-window / power risk)
- **HAC bandwidth:** `maxlags = max(holding_period_in_obs − 1, ceil(T^(1/4)))`. The holding
  period (not just the signal window) drives return-series overlap; the bandwidth must cover
  it or the HAC SE under-corrects — the exact downward-SE bias the guide forbids (§2.1).
- **Autocorrelation diagnostic:** compute Ljung-Box / ACF on each factor-return series; if
  significant autocorrelation is present, the Lo (2002) Sharpe SE uses the
  **autocorrelation-corrected** form (guide §2.3), not the iid form. The trigger is automatic,
  not optional.
- **Power / effective-n:** report each variant's effective non-overlapping `n` and approximate
  power. A variant with effective `n < MIN_EFFECTIVE_N` is labelled **"inconclusive
  (underpowered)"**, distinct from "insignificant." (Honesty over false nulls; mirrors Project
  1's explicit power≈9% framing.)

#### Battery
- **2.1 Naive t-stat** — computed and reported but **flagged biased**; never the headline.
- **2.2 Newey-West HAC t-stat (guide §2.2)** — OLS on a constant with HAC SE at the §2.0
  bandwidth. **The reported mean-return test.** (Reuse `factor_eval.stats.hac_tstat` /
  `ols_tstat_hac`, verified against statsmodels.)
- **2.3 Sharpe + Lo (2002) SE (guide §2.3)** — every Sharpe reported with SE; autocorrelation-
  corrected per §2.0. **NEW** in the stats module.
- **2.4 Spanning regression (guide §2.4)** — regress momentum on **{equal-weighted market
  return, small-minus-big size control (Artemis market cap, test-only)}**; report the
  intercept (alpha) and its HAC t-stat. Keep the variant only if alpha is significant. The
  size control is a regressor, never deployed (§2). **NEW** helper.
- **2.5 Multiple-testing correction (guide §2.5)** — Bonferroni on the pre-registered
  selection family (`m_select = 7`), Harvey-Liu-Zhu tiers (t≥3 significant, 2<t<3 suggestive),
  and **record the total number of tests run** (selection + diagnostics). (Reuse
  `bonferroni_correction`, `classify_tstat_hlz`.)
- **2.6 Subsample stability (guide §2.6)** — split into halves and thirds; a valid variant
  **holds its sign**. **A sign-flip across subsamples disqualifies the variant from
  deployment**, regardless of full-sample t-stat (a single-regime return is regime exposure,
  not a factor). **NEW** analysis with a deployment gate.
- **2.7 Stationary block bootstrap (guide §2.7)** — `arch.bootstrap.StationaryBootstrap` is
  the **sole bootstrap of record** for the empirical mean p-value; the existing
  `cmom/overfitting/bootstrap.py` may serve only as a cross-check, never the reported result.
  **Disagreement** between bootstrap and Newey-West is defined as the two falling on **opposite
  sides of the Bonferroni-adjusted threshold**; on disagreement **the bootstrap result is the
  reported verdict** (guide §2.7).
- **2.8 Out-of-sample reserve (guide §2.8)** — **before any analysis**, set aside the most
  recent ~30% of rebalance dates. The split date is frozen in config (`OOS_START`) after the
  Stage-1 panel exists and before any Stage-2 statistic is computed. The OOS window is
  **sealed and evaluated exactly once**, in Stage 4 — enforced by a single-use guard and a
  test asserting OOS metrics are produced by one code path. All selection reads the in-sample
  slice only.
- **Overfitting controls (reused):** Deflated Sharpe (DSR) and PBO/CSCV from
  `cmom/overfitting/{dsr,pbo}.py` quantify selection-induced overfit across the variant grid.

**Stage 2 completion criteria:** every variant has a Newey-West t-stat (not the naive one) at
the §2.0 bandwidth; every Sharpe has an (autocorrelation-aware) SE; the candidate has a
spanning alpha vs {market, size control}; the total tests are recorded and Bonferroni + HLZ
applied to the pre-registered family; surviving variants hold their sign across subsamples
(sign-flips disqualified); the OOS window is defined in config and sealed.

### Stage 3 — Deployed momentum strategy over time

- **3.1 Benchmark** — the equal-weight single-canonical-variant momentum portfolio; anything
  fancier must beat it net of added turnover.
- **3.2–3.3 Specification choice** — if multiple horizons survive Stage 2 in-sample, build a
  **risk-weighted (inverse-vol / risk-parity) multi-horizon momentum composite**; else deploy
  the single survivor. No naive mean-variance unless disciplined (shrinkage + constraints).
- **3.4 Walk-forward (guide §3.4)** — every time-varying input (vol scalar, composite weights)
  estimated from **data up to each rebalance date only**, re-estimated monthly; **expanding vs
  rolling** both tested and the choice justified.
- **3.5 Volatility targeting (guide §3.5)** — scale to `ANNUAL_VOL_TARGET` (pinned by
  convention in config, not tuned) using trailing realized vol; state the trade-off.
- **3.6 Meta-parameters (guide §3.6)** — vol/cov lookback windows are config constants set by
  convention, **not** by maximizing backtest Sharpe.

**Stage 3 completion criteria:** a benchmark exists; weighting uses past data only; expanding
vs rolling tested and justified; vol-target trade-off stated. (Honest note: light for one
factor family.)

### Stage 4 — Cost-aware backtest (spot-adapted)

- **4.1 Positions** — the factor's coin weights are the book; apply `PER_COIN_CAP` and
  `GROSS_LEVERAGE_CAP = 2.0` (within the guide's 2–3× band, by convention — §3.6); verify
  dollar-neutrality and that gross matches the vol target each period.
- **4.2 Cost model (guide §4.2, spot-adapted)** —
  - **Fees:** spot taker fee per side (`TAKER_FEE_BPS`) on traded notional.
  - **Slippage:** size-scaled by liquidity tier — ~5 bps top names, 15+ bps smaller — scaled
    with order size relative to ADV.
  - **Funding:** **not modeled — Artemis has no funding and this is a spot strategy; stated
    explicitly** (guide's third component N/A here).
- **4.3 Execution timing (guide §4.3)** — distinct **signal price** (close t) and **execution
  price** (close t+1, §3.2) with slippage applied to the fill; never a costless fill. Default
  is t+1 close; (no intraday alternative on Artemis).
- **4.4 Backtest loop (guide §4.4)** — per rebalance: target positions, trade list =
  target − current, fees+slippage on traded notional, carry forward with price P&L. Retain
  **equity curve, position history, trade log** as parquet.
- **4.5 Metrics net of costs (guide §4.5)** — total & annualized return, vol, Sharpe, Sortino,
  max drawdown, Calmar, hit rate, avg win vs loss, annualized turnover, and a **capacity
  estimate**: sweep AUM, recompute size-scaled slippage per coin via the order/ADV ratio, and
  report the AUM at which net expected return crosses zero. (Reuse `factor_eval.stats`.)
- **4.6 Robustness (guide §4.6)** — **2× costs** rerun; **±50% lookback** rerun (reuses the
  chosen spec — not new selection tests); **regime breakdown** with regimes defined by
  convention in config (trailing market-return sign × realized-vol terciles → bull/bear/chop);
  **OOS run once** with the in-sample-vs-OOS Sharpe gap reported. Near-zero/negative OOS Sharpe
  is reported as overfitting.

**Stage 4 completion criteria:** caps applied (`GROSS ≤ 2×` asserted every period) and
dollar-neutrality verified; fees + size-scaled slippage modeled (funding N/A noted); signal
price ≠ execution price; full net metric set incl. turnover and capacity; cost/parameter/
regime sensitivity run; OOS run exactly once with the gap reported.

### Stage 5 — Research report (PDF, 8–12 pages)

- **5.1 Structure** — open with the conclusion (variants tested, what held up, deployed-strategy
  net-of-cost OOS Sharpe, recommendation); then methodology, variant-by-variant results,
  deployed strategy, robustness, recommendation, appendix of full tables.
- **5.2 Charts** — cumulative P&L (gross + net), drawdown, rolling 6-month Sharpe, per-variant
  contribution, variant correlation heatmap; **each chart carries one plain-language takeaway.**
- **5.3 Regression tables** — rows = variants; columns = coefficient, Newey-West t-stat,
  p-value/stars; report n and R²; include the spanning alpha; **include variants that failed.**
- **5.4 Required disclosures** — gross vs net Sharpe side by side; in-sample vs OOS Sharpe side
  by side; any variant that works only gross of costs stated explicitly; the no-funding/spot,
  daily-execution, wrapped-exclusion, and Artemis-coverage survivorship limitations all stated.
- **5.5 Recommendation** — which variant(s)/composite to deploy (if any), target allocation,
  expected Sharpe with a range, key risks (regime dependence, capacity, crowding, data limits),
  next steps.

## 5. Architecture — `new-artemis-work/`

uv-managed Python package (working name `amom`), TDD, ≥80% coverage, many small focused
modules (≤400 lines typical), immutable transforms.

```
new-artemis-work/
  pyproject.toml, uv.lock, .env.example, Makefile, README.md, .gitignore
  docs/
    specs/2026-05-30-artemis-momentum-design.md     # this spec (authoritative methodology)
    plans/                                          # implementation plan (next step)
  src/amom/
    config.py            # universe params, lookback grid, skip, OOS rule, cost params, caps,
                         #   vol target, maxlags rule, regime + power thresholds
    providers/           # ported Artemis client + base protocol
    cache.py             # ported parquet cache
    universe/            # point-in-time builder, listing-date reconstruction, dead-coin
                         #   inclusion, min-universe gate, stablecoin+wrapped exclusion
    returns/             # spot holding-return panel
    factor/              # momentum signal grid + quintile dollar-neutral formation (t+1 lag)
    stats/               # ported factor_eval.stats + NEW Lo SE (autocorr-aware), spanning
                         #   (market + size control), arch StationaryBootstrap; + DSR, PBO
    backtest/            # NEW spot cost engine, t+1-close execution, capacity, regimes
    report/              # charts, regression tables, memo assembly
  scripts/               # connectivity probe, build_universe, build_returns, build_factor,
                         # run_stage2, run_backtest, build_report
  tests/                 # unit + integration; synthetic fixtures for offline TDD
  data/                  # parquet artifacts (gitignored); cache/
```

## 6. Reuse-copy manifest (everything used is *copied* in, per instruction)

| Source | File(s) | Destination | Adaptation |
|---|---|---|---|
| `src/cmom/providers/` | `artemis.py`, `base.py` | `src/amom/providers/` | verify auth vs 403 on first probe |
| `src/cmom/` | `cache.py` | `src/amom/cache.py` | none |
| `src/cmom/config.py` | Artemis URL, market metric list, `STABLECOINS`+`WRAPPED` exclusion sets | `src/amom/config.py` | momentum-only params added |
| `src/cmom/overfitting/` | `dsr.py`, `pbo.py`, `bootstrap.py` | `src/amom/stats/` | `bootstrap.py` = cross-check only; `arch` is record |
| earlier private factor-eval `src/factor_eval/` | `stats.py`, `types.py`, `evaluator.py` | `src/amom/stats/`, `src/amom/factor/` | extend stats (Lo SE, spanning, maxlags rule) |
| earlier private factor-eval `factors/` | `momentum.py`, `momentum_portfolio.py` | `src/amom/factor/` | **strip funding**; source from Artemis panels; t+1-close lag |

The "junk" sleeve modules (`src/cmom/sleeve/*`) are **not** copied. The stablecoin/wrapped
exclusion comes from the ported `STABLECOINS | WRAPPED` sets in `cmom/config.py`.

## 7. Anti-bias / look-ahead compliance matrix

| Guide rule | Where enforced | Test |
|---|---|---|
| Universe enumerated from Artemis (no hand-curated list) | `universe/registry` pulls `/asset` (1,013), keys on `artemis_id` | assert registry built from `/asset`, not a literal symbol list |
| Survivorship: keep delisted/collapsed coins + carry crash | `universe/` keeps dead coins + `price_last_date`/`delisted_asof`; `returns/` books the terminal return | fixture coin crashes ~−95% then stops → present pre-crash, and **realized return ≈ −95% asserted in `returns/`** |
| Recycled tickers split into distinct assets | `universe/` drawdown→0 + gap splitter; `artemis_id`/`coingecko_id` continuity | fixture ticker dies then a new project reuses it → two synthetic assets, crash carried on the first |
| Liquidity computable from Artemis (no `30D_VOLUME`) | `universe/` MC floor + trailing-30d **median** `24H_VOLUME` | sparse 2-print window does NOT pass on a mean artifact; median/MC gate behaves correctly |
| Tradeability / no stale-but-eligible dead coin | `universe/` requires a non-NaN price within a grace window of `as_of` | coin that stops reporting becomes ineligible within the grace window on a daily grid |
| Point-in-time eligibility, rebuilt daily | `universe/` evaluates filters as-of each date; panel materialized on a **daily** grid | assert coin ineligible before 90d history; assert grid is daily |
| Min-universe gate derived from `MIN_BUCKET_SIZE`, point-in-time | `universe/` gate uses as-of data only | assert gate decision unchanged when future data mutated; assert per-quintile floor |
| No look-ahead: signal(t) ≤ close t; enter t+1 close | `factor/` t+1 lag; signal fn forbidden post-t data | assert signal(t) unchanged when t+1.. data mutated |
| Lookback grid + skip fixed in advance | frozen constants in `config.py`; skip=1 convention | grid/skip are frozen constants |
| HAC bandwidth covers holding-period overlap | `stats/` maxlags = max(hold−1, ⌈T^¼⌉) | assert maxlags ≥ holding_period_obs − 1 |
| Multiple-testing on pre-registered family; record total | `stats/` Bonferroni on m_select=7; report records total | assert m_select == len(selection grid); total recorded |
| Subsample sign-flip disqualifies deployment | Stage 3 reads Stage 2 sign-stability gate | construct a sign-flipping variant; assert excluded |
| OOS reserved up front, sealed, used once | `config.OOS_START`; single-use guard | assert no selection code reads ≥ OOS_START; assert one OOS path |
| Bootstrap (arch) overrides NW on disagreement | Stage 2 reporting rule; disagreement = opposite sides of adj. threshold | construct disagreement; assert reported verdict = bootstrap |
| Don't mix simple/log conventions | `returns/` simple XS, log compounding; ranking monotone | documented + unit-tested |
| Caps by convention, gross ≤ 2× | `config.py` constants | assert gross ≤ 2× every backtest period |
| Meta-params by convention, not Sharpe-max | vol/cov lookbacks, vol target, regimes are config constants | constants, not optimized |

## 8. Testing strategy

TDD: tests first, ≥80% coverage. **Synthetic fixtures** (deterministic panels with a known
momentum effect and a known crash coin) build and verify the full pipeline **offline**,
independent of live Artemis. A separate **integration** suite hits the live Artemis API
(connectivity, coverage depth, dead-coin presence, history start) and is the first thing run
once the key is confirmed.

## 9. Deliverables

1. **Research report** (PDF, 8–12 pp) per Stage 5.
2. **Reproducible code repository** — private GitHub repo `artemis-quant-jack-new`, with `make`
   targets reproducing every number in the report.

(No pitch deck this round.)

## 10. Risks & open limitations

- **Artemis API access — RESOLVED.** Live probe confirmed `api_ok=true`: BTC daily history
  from 2013-04-28, and a working `/asset` enumeration of 1,013 assets (Appendix B). The prior
  403 was a stale duplicate `.env`.
- **Survivorship — partially mitigated, residual quantified & disclosed.** The universe is the
  Artemis `/asset` catalog (1,013, `artemis_id`-keyed), which includes many crashed-but-still-
  listed coins, and recycled tickers are split so real collapses (e.g. `lunc`→`terra`'s
  −99.99% LUNA crash) are carried. **Residual:** the catalog is *as-of-today* with no PIT
  membership dates, so assets fully purged from Artemis before query are unrecoverable. The
  report quantifies this (count of catalog assets showing terminal collapses; note that names
  with no surviving series anywhere cannot be included) — per "note it and build without it."
- **Volume reliability.** Only `24H_VOLUME` is historical (`30D_VOLUME` is real-time only) and
  it has broken prints; liquidity is gated on `MC` + trailing-30d *median* `24H_VOLUME` (§3.5),
  a disclosed deviation from the guide's mean-ADV rule.
- **Power / overlapping windows.** 30-day holds yield few non-overlapping observations ⇒ low
  power; handled by the §2.0 bandwidth rule, the bootstrap, the power/effective-n labelling,
  and reliance on the full grid rather than cherry-picking.
- **Spot vs perp.** Results describe a spot long/short book; perp-specific effects (funding)
  are absent by construction.
- **Daily execution.** t+1-close fills (not intraday — Artemis intraday history is stale/placeholder).

## 11. Build mechanism

After this spec is approved, the build is driven by the **Workflow tool** in phases: scaffold
→ port reuse → universe → returns → factor grid → Stage-2 stats → backtest → report →
adversarial verification, TDD throughout, with review gates. The first workflow step is a
**live Artemis connectivity + coverage probe** (scripts load the key themselves; the agent
never opens `.env`).

---

## Appendix A — Vetting record (rev 1 → rev 2)

Rev 1 was reviewed by an adversarial methodology agent against the factor-book methodology guide.
Verdict: **FAITHFUL_WITH_FIXES**. The anti-bias core (survivorship, point-in-time, look-ahead-
in-code, OOS-up-front, bootstrap-overrides-NW, reporting honesty) was confirmed faithful with
real tests. Resolutions folded into rev 2:

- **C1 (autocorrelation/power):** added §2.0 — `maxlags` covers holding-period overlap;
  automatic Ljung-Box trigger for Lo autocorrelation-corrected Sharpe SE; effective-n/power
  labelling ("inconclusive (underpowered)" vs "insignificant").
- **C2 (spanning adequacy):** §2.4 spanning set now includes a **size control** (Artemis market
  cap, test-only regressor) per user decision, making the redundancy test non-vacuous; the
  reduced-vs-full-LTW limitation is disclosed.
- **H1 (skip + test count):** skip fixed by convention (=1), {2,3} as separate robustness;
  pre-registered selection family `m_select = 7`; total test count recorded.
- **H2 (exclusions):** wrapped exclusion disclosed as deliberate (§3.4); unlock/supply noted as
  out-of-scope Size concern with residual-risk flag.
- **H3 (caps):** `GROSS_LEVERAGE_CAP = 2.0` and a concrete `PER_COIN_CAP` pinned in config;
  matrix assertion added.
- **H4 (bootstrap):** `arch.StationaryBootstrap` is sole record; "disagreement" defined.
- **M1–M6, L1–L4:** disagreement rule, sign-flip deployment gate, OOS single-use guard,
  capacity method, t+1-close execution default, regime definitions, "deployed strategy"
  wording, and exclusion-source naming all folded in.

Items flagged by the reviewer and resolved here: Artemis daily granularity → t+1-close fills
(§3.2); min-universe gate (§Stage 1.1); size control in spanning (user-approved); effective-n
power labelling (§2.0). No open items remain for the user beyond final spec approval.

## Appendix B — Live Artemis data-capability findings (rev 2 → rev 3)

The rev-2 universe build was adversarially verified and returned **NEEDS_FIXES**: it had been
built from a hand-curated 124-symbol list (the survivorship trap), its "dead coins" resolved to
zombies/revivals carrying no crash, the panel handed no death signal downstream, the crash test
was vacuous, and ADV used a mean-of-present-rows denominator. A follow-up live investigation of
the Artemis API established what the platform can actually support; rev 3 (the universe
remediation plan, `docs/plans/2026-05-30-universe-remediation.md`) is built on these facts:

- **Enumeration:** `GET https://data-svc.artemisxyz.com/asset` (no `/data/api` prefix, no key)
  returns **1,013 assets**: `artemis_id` (unique stable slug), `symbol`, `coingecko_id`
  (397/1013), `title`. This is the Artemis-native universe seed; key everything on `artemis_id`.
  (Note: any path under `/data/api/...` returns HTTP 200 with empty symbols because the first
  segment is read as a metric name — not a real endpoint.)
- **Catalog is as-of-today**, no listing/delisting dates → no true PIT membership; purged-dead
  coins unrecoverable (disclosed, §10/§3.6).
- **Volume:** only `24H_VOLUME` is historical; `30D_VOLUME` is real-time-only (sentinel on
  historical pulls — and is wrongly listed in `MARKET_METRICS`, to be fixed). `24H_VOLUME` is
  unreliable cross-sectionally (sub-dollar prints, `for` flatlined at 75.66) → liquidity uses
  `MC` + trailing-30d median `24H_VOLUME` (§3.5).
- **Recycled/zombie tickers:** death usually = price decay→~0 with the series continuing, or a
  ticker repointed to a new project (`ust`). `lunc`→`terra` holds the real −99.99% LUNA crash;
  `luna`→`terra2` is the revival. `artemis_id` separates twins kept under different tickers; a
  drawdown→0 + gap splitter handles single tickers reused over time.
- **History/granularity:** earliest 2013-04-28; **DAY only** usable (HOUR/MINUTE are
  stale/placeholder for history).

Rev-3 remediation (per the universe-remediation plan): asset-registry from `/asset`; liquidity
on MC + median-`24H_VOLUME`; death signal (`price_last_date`/`delisted_asof`) in the panel +
crash-return assertion moved to `returns/`; recycled-ticker splitter; observation-density and
tradeability/staleness filters; daily eligibility grid; ADV denominator fixed; quantified
survivorship disclosure. Re-verified by the same 3-lens adversarial gate; must reach **SOUND**
before any returns/factor/stats work begins.
