# Project 1: Crypto Factor Book — Implementation Guide (verbatim reference)

> Source: `Project 1/factor-book-jack/Project1_Factor_Book_Guide.docx`, extracted verbatim.
> This is the AUTHORITATIVE methodology. The Artemis Momentum project follows it strictly.
> Hyperliquid/CMC-specific 'reuse the existing X' notes are guidance from the original
> assignment; the Artemis project substitutes Artemis-sourced equivalents where noted in the spec.

---

Project 1: Building and Testing the Crypto Factor Book
Implementation guide

Objective. Build a dollar-neutral long/short portfolio of crypto factors on the Hyperliquid perpetual universe and determine, net of transaction costs, whether it has a statistically significant positive expected return.
The work is divided into five stages, completed in sequence. Do not begin a stage before completing the prior one.
Construct the factor portfolios.
Test each factor for statistically significant positive expectancy.
Combine surviving factors into a single portfolio, reweighted through time.
Backtest the combined portfolio with realistic costs.
Present the regression and backtest results.
Tools. Python; pandas or polars for data; statsmodels and linearmodels for regressions; arch for the stationary block bootstrap; numpy and scipy otherwise. All data is parquet under data/ — there is no Postgres for this project.

Starting state
The project repo Elven will share already contains the data pipelines, the point-in-time universe, the symbol-mapping layer, and factor-eval (the evaluation framework). Clone it, uv sync, work through the stages — inline notes flag specific pieces to reuse rather than rebuild.
Setup
Clone the project repo (Elven will share access). uv sync, copy .env.example to .env, set CMC_API_KEY (ask Elven), and run nightly_data_update.py once to populate data/ — then daily after that so your analysis stays on fresh data.

Stage 1 — Construct the factor portfolios
Objective. Produce a clean long/short return series for each of five factors.
1.1  Define the universe
The universe is every perpetual listed on Hyperliquid, defined point-in-time.
For each backtest date, include only coins listed and tradeable on that date. Include coins later delisted; their final return must reflect the delisting (a delisting after a 90% drop is a return of −90%, not a skipped value).
Building the universe from currently-listed coins introduces survivorship bias and inflates every factor. Reconstruct listing and delisting dates from Hyperliquid metadata and historical candles.
Apply liquidity filters, also point-in-time: minimum 90 days of history; minimum $1M average daily volume over the trailing 30 days. Exclude stablecoins.
Output: a table giving the eligible coin list for each date. Rebuild it daily.
Use the existing eligibility table. data/universe/universe_history.parquet is already date-indexed and point-in-time; layer your liquidity and history filters on top. Stablecoin exclusion is already applied via the EXCLUDED_SYMBOLS set in data_pull.py.
Hyperliquid↔CMC symbol mapping is solved. data/hl_cmc_mapping.csv resolves symbol collisions (several CMC tokens share symbols, e.g. GOAT). Use the mapping; do not rebuild it.
1.2  Compute returns
Holding return = price return adjusted for funding paid or received, signed by position direction. Implement this before building factors; the carry factor depends on it.
Use simple returns to aggregate across coins within a period. Use log returns to compound a series through time. Do not mix conventions.
Funding is in the data pipeline. Combine hourly funding from data/hyperliquid/funding/ with price returns. Re-run nightly_data_update.py to refresh it; do not write a separate puller.
1.3  Build the five signals
Each factor is a FactorSignal / FactorPortfolio (factor_eval.types) evaluated by FactorEvaluator — use the framework; do not start a parallel one.
Factor
Signal
Rebalance
Momentum (MOM)
Trailing 4-week return, skipping the most recent 1–3 days
Weekly
Size (SIZE)
Circulating market capitalization
Weekly
Short-term reversal (REV)
Trailing 1-week return
Daily
Betting-against-beta (BAB)
Trailing 60-day beta to the crypto market
Weekly
Carry (CARRY)
Trailing 7-day average funding rate
Daily / weekly

Momentum. Sort by trailing return; long winners, short losers. Test 2-, 4-, and 8-week lookbacks with a 1–3 day skip. Fix the lookback set in advance; do not select it from backtest results.
Then experiment beyond the paper. LTW's 2-, 4-, and 8-week lookbacks are the academic baseline. Crypto's characteristic timescales are shorter than equities, so also test shorter lookbacks (e.g. 1, 3, 5, 7 days). Pick what to deploy from your Stage 2 statistical results, not from the paper.
Size. Use circulating market cap, not fully-diluted. Coins with imminent unlocks show mechanical cap jumps unrelated to the size premium; flag them.
Read CMC_RESEARCH_FINDINGS.md first. CMC market caps disappear when circulating supply is unverified; the existing pipeline includes a detector that catches these supply-field errors. Use it; do not rebuild it.
Short-term reversal. Long prior-week losers, short prior-week winners. Rebalances daily; highest transaction-cost exposure of the five factors.
Betting-against-beta. Estimate each coin's beta to the equal-weighted universe return over a trailing 60-day window. Long low-beta, short high-beta, with each leg levered to beta = 1 per Frazzini-Pedersen Section 3. The leverage adjustment is required.
Carry. Long coins with low or negative funding, short coins with high funding. Use realized funding from the prior window. Funding is already netted into returns; do not double-count it.
Funding appears in two places; both are needed. Realized funding inside the holding-return series and the trailing realized funding used as a predictive signal are distinct. Neither replaces the other.
1.4  Form the portfolios
On each rebalance date, sort the eligible universe by signal into quintiles (~30 coins per bucket). The long leg is one extreme quintile, the short leg the other.
Equal-weight coins within each leg for the baseline.
Make each factor dollar-neutral: equal dollar exposure long and short. Factor return = long-leg return − short-leg return.
Timing: compute the signal from data through the close of date t; enter at the open of date t+1. Enforce this in code — the signal function for date t must not access data after t's close. Computing the signal and entering at the same close is look-ahead.
Stage 1 — completion criteria
The historical panel includes delisted coins, with final returns reflecting the delisting.
Eligibility (listing status, liquidity, history length) is evaluated point-in-time and rebuilt daily.
Holding returns net funding, correctly signed for longs and shorts.
Each of the five factors produces a clean long/short return series with no gaps.
A one-period gap between signal and execution is enforced in code.

Stage 2 — Test for statistically significant expectancy
Objective. Determine which factors have a true positive expected return rather than a sample artifact. For each factor, the null hypothesis is a true mean return of zero.
Extend, do not fork. factor_eval/stats.py already implements hac_tstat (Newey-West) and ols_tstat_hac. Add Lo (2002) Sharpe SE, a Bonferroni helper, a spanning-regression helper, and a stationary block bootstrap (arch.bootstrap.StationaryBootstrap) to the same module.
2.1  Mean-return t-test
Naive t-statistic: t = mean(f) / (std(f) / sqrt(T)), where T is the number of observations. This assumes independent, identically distributed observations. Factor returns built from overlapping windows are autocorrelated, which biases the naive standard error downward. Do not report the naive t-stat as the result.
2.2  Newey-West standard errors
Regress the factor return series on a constant using HAC standard errors:
import statsmodels.api as sm

model = sm.OLS(f, sm.add_constant(np.ones(len(f))))
res = model.fit(cov_type='HAC', cov_kwds={'maxlags': L})
# res.params[0]  -> mean factor return
# res.tvalues[0] -> corrected t-stat
The intercept t-stat is the corrected mean-return test. Set maxlags a few periods beyond any signal-window overlap (T^(1/4) is a standard default). Report this t-stat, not the naive one.
2.3  Sharpe ratio with standard error
Standard error of an estimated Sharpe for iid returns: SE(SR) ~ sqrt((1 + 0.5*SR^2) / T) (Lo, 2002). Report every Sharpe ratio with its standard error. Apply Lo's autocorrelation correction if returns are autocorrelated.
2.4  Spanning regression
Regress each candidate factor on the established factors (market, size, momentum). The intercept (alpha) is the return not explained by existing factors. Keep the factor only if its alpha is statistically significant; an insignificant alpha means the factor is redundant regardless of its standalone t-stat. This follows Liu-Tsyvinski-Wu, Table 6.
2.5  Multiple-testing correction
Five factors with several lookback variants is approximately 15 tests. At a 5% threshold, roughly one will clear by chance. Apply a Bonferroni correction (threshold divided by the number of tests). Treat results with 2 < t < 3 as suggestive, not conclusive (Harvey-Liu-Zhu, 2016). Record the total number of tests run.
2.6  Subsample stability
Split the sample into halves or thirds. A valid factor holds its sign across subsamples. A factor that earns its return in a single regime is exposure to that regime, not a factor.
2.7  Bootstrap
Crypto factor returns are fat-tailed and skewed. Run a stationary block bootstrap (Politis-Romano) for an empirical p-value of the mean return. If it disagrees with the Newey-West t-test, use the bootstrap result. Available in the arch package.
2.8  Reserve an out-of-sample window
Before any analysis, set aside the most recent ~30% of history. All factor selection, lookback choices, and weighting decisions use the in-sample portion only. The out-of-sample window is used once, at the end of Stage 4. Record the split date in a config file.
Stage 2 — completion criteria
Every factor has a Newey-West t-stat reported, not the naive one.
Every Sharpe ratio is reported with a standard error.
Each candidate factor has a spanning-regression alpha against the core factors.
The total number of tests is recorded and a multiple-testing correction is applied.
Each surviving factor holds its sign across subsamples.
The out-of-sample window is defined in config and untouched.

Stage 3 — Construct the optimal portfolio over time
Objective. Combine the surviving factors (typically two or three) into a single portfolio with weights estimated walk-forward.
3.1  Equal-weight benchmark
Build an equal-risk-weighted combination first. It is the benchmark; any other weighting scheme must beat it net of its added turnover to be used.
3.2  Risk-based weighting
Inverse-volatility: weight each factor by 1/volatility so each contributes equal risk.
Risk parity: as above, additionally accounting for inter-factor correlations.
Either typically outperforms equal weight.
3.3  Mean-variance optimization (optional)
Naive mean-variance weighting (w proportional to inv(Sigma) * mu) is unstable: estimation error in the expected returns and the covariance matrix is amplified by the matrix inversion, producing extreme, unstable weights. If used, it must be disciplined:
Shrink the covariance matrix (Ledoit-Wolf; sklearn.covariance.LedoitWolf).
Shrink expected returns toward the cross-sectional mean, or omit expected returns entirely and use risk parity.
Constrain weights: no factor shorting; per-factor caps.
3.4  Walk-forward estimation
At each rebalance date, estimate all inputs (volatilities, correlations) from data available up to that date only, then apply the resulting weights forward. Re-estimate monthly. Test both an expanding window and a rolling window. Estimating one weight set over the full history and applying it backward invalidates the backtest.
3.5  Volatility targeting
Scale the combined portfolio to a target annualized volatility (e.g. 10%) using trailing realized volatility. This stabilizes strategy risk. It modestly improves performance in trending markets and detracts in choppy markets.
3.6  Meta-parameters
Set covariance lookback windows and similar meta-parameters from convention, not by maximizing backtest Sharpe. Tuning them on the backtest is overfitting.
Stage 3 — completion criteria
An equal-weight benchmark combination exists.
Any optimizer in use applies covariance shrinkage and weight constraints.
Portfolio weights at every date are computed from past data only.
Both expanding and rolling windows are tested, and a choice is justified.
If volatility targeting is used, its trade-off is stated.

Stage 4 — Backtest the trading strategy
Objective. Convert the combined factor portfolio into Hyperliquid perp positions and measure performance net of costs.
factor-eval already backtests a single factor's long/short portfolio gross of costs (FactorEvaluator.cumulative_returns, drawdown_analysis, summary_stats). Stage 4 is the extension layer that adds what is missing:
Multi-factor position netting (4.1). Per-coin positions summed across factors, with per-coin and gross-leverage caps applied.
Cost overlay (4.2). Fees on traded notional, size-scaled slippage by liquidity tier, and funding accrued every funding interval on held positions — not only at rebalance dates.
Execution timing (4.3). A separate signal price and execution price, with slippage applied to the fill.
Artifacts (4.4). Equity curve, position history, and per-trade log retained as parquet.
Capacity estimate (4.5). The AUM at which slippage eliminates the edge.
Where it lives. factor-eval/backtest/, as a new module alongside the single-factor evaluator.
4.1  Convert factor weights to coin positions
Scale each factor's coin weights by that factor's portfolio weight; sum across factors per coin; trade the net position.
Verify: the book is dollar-neutral; gross exposure matches the volatility target; no single coin exceeds the per-coin cap.
Cap gross leverage at 2–3x.
4.2  Cost model
Model all three cost components:
Fees. Hyperliquid taker fee per side, current published schedule, applied to traded notional.
Slippage. A function of order size relative to liquidity. Tier it — approximately 5 bps for the top 30 coins, 15+ bps below — and scale it with order size relative to average daily volume.
Funding. Accrue every funding interval on held positions, not only at rebalances.
4.3  Execution timing
Signal at the close of date t; execute at t+1. Assume an achievable fill — first-hour VWAP, or the open with the slippage model applied — not the t+1 open at no cost.
4.4  Backtest loop
For each rebalance date: compute target positions; form the trade list as target minus current; apply fees and slippage to traded notional; carry positions forward, accruing funding and price P&L each period. Retain the equity curve, position history, and trade log.
4.5  Performance metrics
Report, net of all costs: total and annualized return, annualized volatility, Sharpe, Sortino, maximum drawdown, Calmar, hit rate, average win versus average loss, and annualized turnover. Include a capacity estimate — the AUM at which slippage eliminates the edge.
4.6  Robustness
Cost sensitivity. Double all cost assumptions and rerun. A real edge degrades but does not vanish.
Parameter sensitivity. Vary factor lookbacks by ±50% and rerun. Sharpe should change but not collapse.
Regime breakdown. Report performance separately for bull, bear, and chop periods.
Out-of-sample. Run once on the reserved window. Report the in-sample versus out-of-sample Sharpe gap. A near-zero or negative out-of-sample Sharpe indicates overfitting; report it as such.
Stage 4 — completion criteria
Factor weights are netted into per-coin positions, with per-coin and gross caps applied.
Fees, size-scaled slippage, and funding are all in the cost model.
Signal price and execution price are different prices, one rebalance apart.
The full metric set is reported net of costs, including turnover and a capacity estimate.
Cost, parameter, and regime sensitivity are all run.
The out-of-sample test is run once, with the in-sample gap reported.

Stage 5 — Present the results
Objective. Deliver a memo that allows the partners to make a decision. The audience is mixed: quant and non-quant.
5.1  Structure
Open with the conclusion: factors tested, factors that held up, combined out-of-sample Sharpe net of costs, and the recommendation.
Follow with: methodology (brief), factor-by-factor results, combined portfolio, robustness, recommendation, and an appendix of full tables.
Body length: 8–12 pages.
5.2  Charts
Include: cumulative P&L (gross and net on one axis); drawdown plot; rolling 6-month Sharpe; per-factor return contribution; factor correlation heatmap. Each chart carries one plain-language takeaway sentence.
5.3  Regression tables
Standard format: rows are factors; columns are the coefficient, the Newey-West t-statistic, and a p-value or significance stars. Report the number of observations and R². Include spanning-regression alphas. Include the factors that failed.
5.4  Required disclosures
Gross and net Sharpe shown side by side.
In-sample and out-of-sample Sharpe shown side by side.
Any factor that works only gross of costs stated explicitly.
5.5  Recommendation
State: which factors to deploy, the target allocation, the expected Sharpe with a range, the key risks (regime dependence, capacity, crowding), and the proposed next steps.
Stage 5 — completion criteria
The memo opens with the answer and the recommendation.
Every chart has a one-sentence plain-language takeaway.
Regression tables show Newey-West t-stats and include the factors that failed.
Gross vs. net and in-sample vs. out-of-sample Sharpe are shown side by side.
The recommendation specifies factors, allocation, and risks.

Appendix — references and tools
Factor literature
Liu, Tsyvinski & Wu (2022), Common Risk Factors in Cryptocurrency — replication target; see Table 6.
Asness, Moskowitz & Pedersen (2013), Value and Momentum Everywhere
Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers — momentum construction.
Frazzini & Pedersen (2014), Betting Against Beta — BAB construction; Section 3.
Koijen, Moskowitz, Pedersen & Vrugt (2018), Carry
BIS Working Paper 1087 (2023), Crypto Carry
Statistical methods
Harvey, Liu & Zhu (2016), …and the Cross-Section of Expected Returns — multiple-testing correction.
Lo (2002), The Statistics of Sharpe Ratios, Financial Analysts Journal — Sharpe-ratio standard error.
Ledoit & Wolf (2004), Honey, I Shrunk the Sample Covariance Matrix — covariance shrinkage.
Libraries
pandas / polars — data panel.
statsmodels — OLS with HAC (Newey-West) standard errors.
linearmodels — Fama-MacBeth cross-sectional regressions.
scikit-learn — covariance shrinkage (LedoitWolf).
arch — stationary block bootstrap.
