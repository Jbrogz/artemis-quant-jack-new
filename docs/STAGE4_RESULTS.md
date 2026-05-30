# Stage 4 — Cost-Aware Backtest Results

_Spot long/short book. NO funding term (spec §3.1 / §4.2; Artemis exposes no funding — disclosed). Costs: 10 bps taker fee per side + size-scaled tiered slippage. Execution at the t+1 close. OOS window (rebalance_date >= 2023-12-02) spent EXACTLY ONCE._

## Headline (honest)

The Stage-2 verdict was a **NULL**: no selection-family variant survives HAC + Bonferroni on its own terms, and the strongest, `momentum_L5d_S1d`, is **sign-unstable -> deployment-disqualified** (`docs/STAGE2_RESULTS.md`). This backtest characterizes that candidate net of realistic spot costs — it does not rescue it.

- **Primary `momentum_L5d_S1d`:** in-sample net Sharpe **0.664** (gross 0.789); out-of-sample net Sharpe **0.756** (gross 0.897).
- **IS - OOS net Sharpe gap (primary): -0.092.** OOS Sharpe did NOT collapse (it exceeds in-sample) — but this is a single-regime artifact on a spent-once 30-obs window, not evidence of a deployable edge (the disqualifying signal is the Stage-2 §2.6 sign-instability; the regime breakdown is a descriptive cut only).
- **Comparator `momentum_L28d_S1d`:** IS net Sharpe -0.031, OOS net Sharpe -0.270 (gap 0.239).
- **Capacity (primary, net expected return -> 0):** ~$36,044,481 (AUM at which size-scaled slippage erases the gross edge, computed on the actual per-rebalance *traded* order — ~2.0x summed one-way turnover — not the standing held book; per-rebalance gross edge 0.02436). Comfortably above a $1M book — **capacity is not the binding constraint** (see Conclusion).

## Gross vs net, in-sample vs out-of-sample

### Primary L5d_S1d

| segment | gross Sharpe | net Sharpe | net ann ret | net ann vol | Sortino | max DD | Calmar | hit rate | ann turnover |
|---|---|---|---|---|---|---|---|---|---|
| in-sample | 0.789 | 0.664 | 0.2490 | 0.3752 | 0.756 | -0.4851 | 0.513 | 0.574 | 24.31 |
| out-of-sample | 0.897 | 0.756 | 0.2469 | 0.3266 | 1.031 | -0.2516 | 0.981 | 0.567 | 28.22 |

### Comparator L28d_S1d

| segment | gross Sharpe | net Sharpe | net ann ret | net ann vol | Sortino | max DD | Calmar | hit rate | ann turnover |
|---|---|---|---|---|---|---|---|---|---|
| in-sample | 0.073 | -0.031 | -0.0134 | 0.4314 | -0.030 | -0.6809 | -0.020 | 0.559 | 23.90 |
| out-of-sample | -0.158 | -0.270 | -0.0818 | 0.3031 | -0.283 | -0.5168 | -0.158 | 0.433 | 26.57 |

## Additional §4.5 net metrics (total return, avg win / loss)

| spec | segment | total return | avg win | avg loss |
|---|---|---|---|---|
| Primary L5d_S1d | in-sample | 1.7225 | 0.08584 | -0.06745 |
| Primary L5d_S1d | out-of-sample | 0.6266 | 0.07509 | -0.05136 |
| Comparator L28d_S1d | in-sample | -0.4518 | 0.06534 | -0.08526 |
| Comparator L28d_S1d | out-of-sample | -0.2687 | 0.06283 | -0.05991 |

## Robustness (primary L5d_S1d; reruns reuse the chosen spec)

### 2x costs (in-sample)

| segment | gross Sharpe | net Sharpe | net ann ret | net ann vol | Sortino | max DD | Calmar | hit rate | ann turnover |
|---|---|---|---|---|---|---|---|---|---|
| 1x costs | 0.789 | 0.664 | 0.2490 | 0.3752 | 0.756 | -0.4851 | 0.513 | 0.574 | 24.31 |
| 2x costs | 0.789 | 0.552 | 0.2072 | 0.3753 | 0.610 | -0.5219 | 0.397 | 0.574 | 24.32 |

_2x costs is a sensitivity rerun of the deployed construction (same spec, doubled fee + slippage); it is not a re-selection._

### +/-50% lookback (in-sample, net)

| variant | net Sharpe | net ann ret | note |
|---|---|---|---|
| `momentum_L5d_S1d` (chosen) | 0.664 | 0.2490 | chosen lookback = 5d |
| `momentum_L2d_S1d` | -0.328 | -0.1155 | lookback-50% (skip/quantile unchanged) |
| `momentum_L8d_S1d` | 0.621 | 0.2645 | lookback+50% (skip/quantile unchanged) |

### Regime breakdown (in-sample, net mean return per regime)

| regime | n | mean net return |
|---|---|---|
| bull | 17 | -0.00624 |
| bear | 28 | 0.02195 |
| chop | 22 | 0.03453 |

_Regimes: high-vol (top |market-return| tercile) windows are **chop**; otherwise non-negative trailing market return is **bull**, negative is **bear** (spec §4.6 convention)._

> **Caveat (do not over-read):** this regime cut is a **full-sample, descriptive** partition of the in-sample windows, **not** a walk-forward signal — it could not have been traded ex-ante. The `chop` bucket is just the top-|market-move| tercile, which on this sample skews toward large **up** moves, so it absorbs much of the strongest bull tape; the apparent 'negative in bull / positive in bear' contrast is therefore **overstated** and is an artifact of where the magnitude cut falls, not clean evidence of a bear-only edge. The disqualifying signal is the Stage-2 §2.6 sign-instability, not this descriptive split.

## Disclosures (spec §5.4)

- **Gross vs net side by side** above: spot costs reduce every reported Sharpe; net is never flattered above gross.
- **IS vs OOS side by side** above: the OOS window was opened exactly once (single-use guard); a near-zero / negative OOS Sharpe is reported as overfitting, not hidden.
- **No funding** term in costs or returns (spot; Artemis has no funding — the guide's third cost component is N/A here, stated).
- **Survivorship** still flows into the P&L: a collapsed short-leg coin's crash books as a positive contribution; dead coins are not dropped.
- **Multi-factor combination is N/A** for a single null factor (spec §3.2-§3.3); Stage 3 reduced to volatility targeting on the candidate, included here.
- **Persisted `equity.parquet` `gross_return`** is the **net run's pre-cost** book return (the vol-scalar path fed by net returns), **not** the reported gross Sharpe series — that gross Sharpe comes from an *independent frictionless* run (its own vol-scalar path). The two gross series differ slightly by construction; the headline gross Sharpe is the frictionless one.
- **Dropped boundary window:** the holding window straddling `OOS_START` (from the last in-sample rebalance to 2023-12-02) is priced by neither segment — the in-sample run has no forward window past its last rebalance and the OOS run starts fresh at `OOS_START` — so that one straddle window is intentionally not counted (no double-count, no leak).

## Conclusion (honest, not flattering the null either way)

The primary `momentum_L5d_S1d` posts a positive net Sharpe both in-sample (0.66) and out-of-sample (0.76); the OOS figure did **not** collapse. But this is **not** evidence of a deployable edge, and the Stage-2 disqualification stands:

- The OOS window is **30 overlapping-regime observations spent once** — a single favorable stretch (the 2024 crypto bull) carries it. The return is **regime exposure, not a stable factor**, consistent with the Stage-2 §2.6 sign-flip disqualification. (The regime breakdown above is suggestive but is a full-sample *descriptive* cut, not a walk-forward signal — see its caveat; the disqualifying evidence is the §2.6 sign-instability itself.)
- The **±50% lookback rerun is fragile**: the construction is not robust to a small change in its one free parameter (the deployed lookback).
- The comparator `momentum_L28d_S1d` (academic 4-week canonical) is **net-negative both in-sample and out-of-sample** — the canonical horizon does not work at all net of costs.
- **Capacity does NOT bind** at deployable size: the net edge crosses zero only at ~$36,044,481 of AUM (recomputed on the actual per-rebalance *traded* order, ~2.0x summed one-way turnover, not the standing held book). At a $1M book the slippage drag is immaterial, so capacity is **not** what disqualifies this candidate — the no-deploy case rests entirely on the three points above.

**Net verdict:** consistent with the Stage-2 null and the sign-instability disqualification, momentum on the Artemis spot universe is **not a deployable factor**. The primary's positive OOS Sharpe is a single-regime artifact on a spent-once 30-observation window, not a repeatable edge; it is reported as-is, neither inflated nor suppressed. (Capacity is comfortable at $1M and is **not** the binding constraint.)

