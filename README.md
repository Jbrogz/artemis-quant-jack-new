# amom — crypto momentum factor study with an honest null result

A single-factor (momentum-only) crypto long/short research study on daily
[Artemis](https://www.artemisanalytics.com/) data, built test-first in Python.
Its headline finding is a **rigorous statistical NULL — NO-DEPLOY**: after
multiple-testing correction, HAC-robust inference, and transaction costs, no
variant in the pre-registered family justifies deployment.

> **Research / paper trading only — no live capital, no live P&L.**
> Every number in this repository and its report is a backtest statistic on
> historical data. Nothing here is a realized return or an expected one.

## The finding (and why a null is the point)

The study pre-registered a selection family of 7 momentum lookbacks
(1/3/5/7/14/28/56 days, skip fixed at 1 by convention — see
`src/amom/config.py`), formed dollar-neutral quintile long/short portfolios,
and tested them under a Bonferroni-corrected Newey-West threshold. **No
pre-registered variant cleared it.** The best in-sample candidate looked
"suggestive" on raw t-stats but failed subsample sign-stability, and the one
variant that worked gross of costs did not survive net of the fee + slippage
model. Full numbers, figures, and caveats (including a disclosed residual
survivorship bias in the as-of-today asset catalog) are in the
[PDF report](docs/report/Artemis_Momentum_Report.pdf).

Most retail-grade backtests would have shipped this strategy. The point of the
project is the machinery that correctly refuses to: the pipeline reports what
survives honest statistics, and here the answer is *nothing in the
pre-registered family* — so the verdict is NO-DEPLOY, stated plainly.

## Overfitting-control toolkit

All implemented in `src/amom/stats/` and `src/amom/backtest/`, each with its
own test module:

- **Newey-West HAC t-statistics** — autocorrelation-robust inference; naive t
  is never the headline.
- **Bonferroni correction** over the pre-registered 7-variant family; the
  family was frozen in `config.py` before any backtest ran.
- **Lo (2002)** autocorrelation-adjusted Sharpe standard errors.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) — deflates for the
  number of trials and non-normal returns.
- **Probability of Backtest Overfitting** via CSCV (Bailey et al. 2015).
- **Stationary block bootstrap** (Politis & Romano 1994) confidence intervals.
- **Spanning regressions** — HAC-tested alpha vs. an equal-weighted market
  return and a size control, so momentum must add something a benchmark
  doesn't already span.
- **Subsample sign-stability** checks across time splits.
- **One-shot out-of-sample**: each variant's OOS window is spent exactly once,
  never iterated on.
- **Cost-aware backtest** — per-side taker fee plus size-scaled, tiered
  slippage; gross and net always shown side by side.
- **No look-ahead as the cardinal rule**: decisions at close *t* execute at
  close *t+1*, enforced by discriminating tests (mutating future data must not
  change anything at *t*).

## Architecture

```
src/amom/
  config.py        frozen parameters: universe gates, lookback family, cost model
  providers/       Artemis REST client (mocked in tests)
  cache.py         on-disk parquet response cache
  universe/        point-in-time eligibility: registry, coverage, liquidity, builder
  returns/spot.py  spot holding returns (daily grid, no funding/carry)
  factor/          momentum signal + dollar-neutral quintile portfolio formation
  stats/           core HAC, sharpe_se, spanning, bootstrap, subsample, dsr, pbo
  backtest/        costs, one-shot-OOS engine, metrics
scripts/           pipeline stages: probe -> universe -> returns -> factor ->
                   stage2 stats -> backtest -> figures -> report -> writeup
tests/             226 tests mirroring the module layout (unit + integration)
docs/              design spec, stage plans, stage results, report/
```

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Jbrogz/artemis-quant-jack-new.git
cd artemis-quant-jack-new
uv sync --extra dev
uv run pytest -q     # 223 passed, 3 skipped (~8s)
```

The 3 skips are live-API integration tests that self-skip without an
`ARTEMIS_API_KEY`. Everything else — the full unit suite and all statistics —
runs keyless and offline.

To rebuild the data pipeline against live Artemis data, set `ARTEMIS_API_KEY`
in a local `.env` (see `.env.example`), then run the `Makefile` stages
(`make probe universe`, then the `scripts/` stages in the order above).
Generated data lives under gitignored `data/`; the compiled outputs are
committed under `docs/report/`.

## Testing

Built strictly test-first: 226 tests (~5.0k lines of test code against ~3.2k
lines of source), red-green-refactor per stage, with each stage adversarially
reviewed before the next began. CI runs the keyless suite on every push via
`.github/workflows/ci.yml` (uv + pytest, no secrets referenced).

## Report and extended analysis

- **Report:** [docs/report/Artemis_Momentum_Report.pdf](docs/report/Artemis_Momentum_Report.pdf)
  (10 pages; also as [Markdown](docs/report/Artemis_Momentum_Report.md), with a
  [findings writeup](docs/report/Artemis_Momentum_Findings.docx)) — methodology,
  stage-by-stage results, cost sensitivity, and the NO-DEPLOY verdict.
- **Extended analysis:** the
  [`codex/competition-ready`](https://github.com/Jbrogz/artemis-quant-jack-new/tree/codex/competition-ready)
  branch validates the post-hoc skip>=2 variants as a widened multiple-testing
  family — explicitly labeled exploratory, with the pre-registered null left
  intact.

## Provenance

The build was spec-driven and AI-assisted as an engineering-practice exercise:
a frozen design spec (`docs/specs/2026-05-30-artemis-momentum-design.md`)
executed through gated TDD workflows with adversarial review at every stage.

MIT License — see [LICENSE](LICENSE).
