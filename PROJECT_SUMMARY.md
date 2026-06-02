# Funding-Rate Mean Reversion — Thesis, Research & Findings

A single account of this competition entry: the *idea* behind the factor, *why* it
should work in theory, *how* we tested it, and *what we learned*. For a repo file
map see `SUMMARY.md`; for the full engineering log see `status.md`.

> **Scope:** this submission is a **single-factor** study of funding-rate mean
> reversion. It stands entirely on this one signal — there is no second factor and
> no multi-factor combination.

---

## 1. Competition context

- **Competition:** Artemis Analytics Quant Research Competition — Track 1: Crypto
  Factor Rebalancing Strategy.
- **Deadline:** June 1, 2026, 11:59 PM EST.
- **Deliverables:** research report (PDF), code/analysis (GitHub repo), pitch deck
  (Google Slides shared with lindsey@artemisanalytics.xyz).
- **Judging:** Research Quality (30%), Signal/Edge Validity (30%), Critical
  Evaluation (20%), Communication (20%). The judges explicitly value **honest
  thinking over impressive backtests** — which is exactly what this entry leans on.

---

## 2. The thesis

**Perpetual futures have no expiry, so an exchange uses a funding rate to keep the
perp price tethered to spot.** When a perp trades above spot — because too many
traders are long and bidding it up — longs pay shorts a periodic funding fee
(positive funding). When it trades below spot, shorts pay longs (negative funding).
The funding rate is therefore a *direct, real-money readout of positioning
crowding*: how lopsided the leveraged crowd is, and how much it costs them to stay
there.

The economic claim follows in two steps:

1. **Extreme funding marks a crowded, over-leveraged trade.** Very positive funding
   means longs are crowded and paying a steep carry to hold on. Very negative
   funding means shorts are crowded.

2. **Crowded leveraged positioning mean-reverts.** Crowded longs are fragile: they
   are paying to hold, they are vulnerable to liquidation cascades, and the marginal
   buyer is exhausted. So coins with *extremely high* funding should **underperform**
   going forward, and coins with *extremely negative* funding should **outperform**
   — the crowd snaps back.

This is the same intuition as **Betting Against Beta** (one of the reference papers):
the over-leveraged, over-loved end of the cross-section earns a *lower* future
return than its risk would suggest, because leverage-constrained traders bid it up.
Funding is just a cleaner, dollar-denominated thermometer for that crowding than
beta is.

**The tradeable expression.** Rank the universe each day by a funding *z-score*
(how extreme today's funding is versus its own recent history), go **long the
lowest-funding coins** and **short the highest-funding coins**, equal-weighted,
market-neutral. If the thesis holds, the long leg outperforms the short leg as the
crowding unwinds.

```
z = (short_window_mean(funding) − long_window_mean(funding)) / long_window_std(funding)
```

The z-score (rather than raw funding) matters because each coin has its own baseline
funding level; we care about how *unusually* crowded a coin is relative to itself,
not its absolute rate.

---

## 3. Why it's worth testing — and why to be suspicious

The thesis is economically reasonable and grounded in published factor research, so
it deserves a real test. But three things should make any honest researcher
suspicious from the start:

- **Crypto is a momentum-heavy, regime-driven market.** Mean reversion and momentum
  are opposite bets. In a strong trend, the crowded longs are *right*, and a
  mean-reversion factor bleeds. Any edge here is likely *regime-conditional*, not
  universal.
- **Funding signals are noisy and fast-decaying.** Positioning shifts in hours; the
  "right" lookback windows are not obvious, which invites parameter mining.
- **The data is short and survivorship-prone.** One year of history on a live
  exchange universe is exactly the setting where a backtest can look great for
  reasons that won't repeat.

So the project was deliberately built in two halves: first *find* the best-looking
version of the factor, then *attack* it as if a skeptic were grading it.

---

## 4. Data & pipeline

- **Source:** Hyperliquid `info` API — free, no key, works in the US (Binance
  futures is geo-blocked). Endpoints: `metaAndAssetCtxs` (universe + 24h volume),
  `fundingHistory` (hourly funding, ~24/day, paginated 500/req), `candleSnapshot`
  (daily OHLCV).
- **Coverage:** ~1 year (May 2025 – May 2026), top-35 coins by 24h volume,
  ~246k funding records and ~10.7k daily candles.
- **Build (`data_pipeline.py`):** `get_top_perp_symbols` → `pull_funding_rates` →
  `pull_ohlcv` → `build_funding_features` (daily funding + rolling z-score) →
  `build_panel` (merge + forward returns `ret_1d/7d/14d/21d`). Output parquet files
  live in `data/`; use `panel.parquet` for analysis.
- **Known biases documented in-code:** survivorship (the live universe silently
  excludes delisted coins), point-in-time (ranked by *today's* volume), and an
  in-sample selection disclosure on the tuned windows. A forward-return
  label-alignment bug was fixed (calendar-day reindex so gap-spanning returns become
  NaN) and pinned by `test_data_pipeline.py` (4/4 pass).

---

## 5. The work, in three acts

### Act I — Discovery (find the most favorable version)

Using the data above, we built the z-score signal and searched for the parameters
that made it look best:

- A **parameter sweep** over short/long windows and forward horizons crowned
  **21-day / 90-day windows at a 14-day horizon** (mean IC ≈ −0.049, Q1–Q5 spread
  ≈ 2%). The story: crowding builds and unwinds over *weeks*, so slow windows
  capture real positioning while fast ones (the original 7d/30d) catch noise.
- A **universe sweep** landed on **top-35 coins** — enough names to diversify each
  quintile, not so many that illiquid coins dilute the signal:

  | Coins | IC | Spread | Return | Sharpe | Max DD |
  |-------|------|--------|--------|--------|--------|
  | 10 | −0.0230 | −0.69% | −8.6% | −0.13 | −56.0% |
  | 20 | −0.0212 | 1.23% | 13.9% | 0.34 | −29.0% |
  | 25 | −0.0667 | 3.54% | 65.2% | 1.68 | −24.1% |
  | **35** | **−0.0314** | **1.94%** | **63.6%** | **1.91** | **−19.2%** |
  | 50 | −0.0324 | 1.95% | 50.7% | 1.85 | −12.5% |

- Even here, honest red flags surfaced: the strategy climbed to +80% through a
  choppy Jul–Dec 2025, then **drew down hard in the Jan 2026 BTC rally** — the
  regime dependence we feared. And in the quintile sort, Q1 (lowest funding)
  outperformed as expected but Q2 often underperformed, hinting the signal is a
  *tail* effect at the extremes, not a smooth linear factor.

At the end of Act I the factor *looked* like a winner. That is precisely the moment
to distrust it.

### Act II — Interrogation (attack the result)

The discovery numbers are in-sample and were chosen *after* looking at the data, so
they are guilty until proven innocent. We ran the standard overfitting defenses in
`validation.ipynb`:

- **Honest re-search (IS-only + HAC t-stats).** Re-ran the sweep on only the
  in-sample window with Newey-West standard errors that don't get fooled by
  overlapping returns. Across 56 combos, **nothing was statistically significant**
  (best p = 0.08), and the "winning" 21/90/14 didn't make the top 10.
- **Multiple-testing correction.** Testing many combos guarantees a few look good by
  luck. Benjamini-Hochberg and Holm corrections left **0 survivors of 56**.
- **Block bootstrap of the actual traded spread.** The real long/short return is
  faintly *positive* (Sharpe ≈ 1.95) but **insignificant** (p = 0.057); the
  confidence interval straddles zero.
- **Deflated Sharpe Ratio.** Penalizing that 1.95 Sharpe for *how hard we searched*
  drops the probability it's real to **~47%** — a coin flip, far under the 95% bar.
- **Transaction costs.** A realistic Hyperliquid cost stack (~6.5%/yr on ~18%/day
  turnover) **does not kill it** — net return stays strongly positive even at 2×.
- **Out-of-sample.** On a sealed ~93-day holdout the edge **held its sign** and kept
  ~76% of its in-sample Sharpe — it did *not* collapse the way a pure fluke usually
  does.

### Act III — Verdict (the decision rule)

A four-gate rubric turns the measurements into one yes/no answer that can't be
cherry-picked:

| Gate | Question | Result |
|------|----------|--------|
| 1. Significance | Significant after multiple-testing correction? | **FAIL** |
| 2. Sign stability | Positive in-sample *and* out-of-sample? | PASS |
| 3. Overfitting | Beats the Deflated-Sharpe search penalty? | **FAIL** |
| 4. Costs | Survives real (and 2×) trading frictions? | PASS |

**2 of 4 → NOT DEPLOYABLE.**

---

## 6. What we learned

**The honest answer is "no edge," and the *way* it fails is the lesson.** This is
not a strategy that loses money in the backtest — it makes a lot, survives costs,
and even carries into out-of-sample. It fails on the two tests that specifically
detect *luck*: statistical significance after correcting for the search, and the
Deflated Sharpe penalty for how many combinations we tried.

That combination — **a big, cost-robust, OOS-positive in-sample number that still
can't clear an honest significance bar** — is the textbook fingerprint of
data-mining. Try enough parameter sets and one will print money for a while by
chance; costs and a short holdout won't expose it. Only the significance and
search-effort tests do.

A few specific takeaways:

- **The rank correlation and the traded spread can disagree.** The overall IC was
  negative, but the long-low/short-high spread came out slightly positive, because
  the worst returns sat in the *middle* funding quintiles, not the extremes. The
  signal, to the extent it exists, is a non-monotone *tail* effect — not a clean
  linear factor.
- **Regime is everything.** The factor lives in range-bound markets and dies in
  trends (it bled through the Jan 2026 BTC rally). On its own — with no offsetting
  trend exposure — that regime fragility is a first-order risk, not a footnote.
- **Rigor changed the conclusion.** The discovery layer would have shipped a
  "Sharpe 1.9 winner." The interrogation layer caught three real code bugs (a
  degenerate Deflated-Sharpe calc, a 2× cost double-count, an OOS warm-up that
  crippled the signal) *and* overturned the headline. The verdict is only
  trustworthy because the process was adversarial.

**For the competition:** the deliverable is the honest null. We found an
economically sensible thesis, gave it every chance to succeed, and showed with
formal statistics that the apparent edge does not survive contact with the tests
built to catch exactly this kind of mirage.

---

## 7. Reference: best in-sample parameters (the mined "winner")

Carried forward only to document the null honestly — not because it cleared any bar.

```
Z-Score Short Window:  21 days
Z-Score Long Window:   90 days
Forward Horizon:       14 days
Universe Size:         Top 35 coins by 24h volume on Hyperliquid
Rebalance Frequency:   Daily
```

---

## 8. Status & remaining work

- **Done:** data pipeline + bias documentation, full validation battery
  (`validation.ipynb`), four-gate verdict, honest-limitations write-back into
  `signal_analysis.ipynb`, transaction-cost model, git repo
  (`github.com/Dcon42/artemis-daniel`, private, `main`).
- **Remaining deliverables:** research report (PDF) and pitch deck (Google Slides
  to lindsey@artemisanalytics.xyz).

## 9. Technical notes

- VS Code + conda env `artemis` (Python 3.11) on a MacBook Air; project at
  `~/Desktop/ArtemisComp`. Key packages: pandas, numpy, statsmodels (HAC),
  scipy (multipletests), arch (stationary bootstrap), matplotlib, pyarrow.
- The Artemis API (`artemis-py`) covers on-chain fundamentals (price, market cap,
  TVL, fees, revenue, dev activity) but **not** per-asset funding rates — which is
  why funding is pulled from Hyperliquid directly.
