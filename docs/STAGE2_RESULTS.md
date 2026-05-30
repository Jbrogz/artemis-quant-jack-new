# Stage 2 — Statistical Significance Battery Results

_In-sample slice only (rebalance_date < OOS_START = 2023-12-02; the OOS window is sealed for Stage 4)._

## Headline

- **Honest verdict (selection family): `suggestive`.**
- Pre-registered selection family: **7 tests** (7 lookbacks at skip=1); Bonferroni threshold = 0.05 / 7 = 0.00714.
- **Total mean-return tests (one per variant): 21.**
- Bonferroni survivors: `momentum_L5d_S1d` **(DISQUALIFIED — fails §2.6 sign-stability; survives only via the bootstrap-override rule, HAC p=0.0081 > 0.00714)**.
- Grid PBO/CSCV (probability of backtest overfitting): 0.114 (over 69 common in-sample dates).

The **HAC** t-stat (Newey-West, autocorrelation-robust) is the reported mean-return test; the **naive** t-stat is shown but is biased (overstates significance) and is never the headline. Underpowered variants are labelled **inconclusive (underpowered)**, distinct from insignificant. On a Newey-West / bootstrap disagreement the **bootstrap** is the reported verdict.

## Selection family (skip = 1) — the deployment candidates

| variant | n | naive t | HAC t | HLZ | ann ret | Sharpe (SE) | autocorr | span α | span α t | HAC p | boot p | disagree | holds sign | power | DSR | survives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `momentum_L14d_S1d` | 69 | 1.902 | 1.945 | not_significant | 0.5128 | 0.799 (0.425) | no | 0.02310 | 1.597 | 0.0259 | 0.0354 | no | yes | powered | 0.308 | no |
| `momentum_L1d_S1d` | 69 | -1.247 | -1.300 | not_significant | -0.3078 | -0.524 (0.422) | no | -0.02502 | -1.457 | 0.9032 | 0.9102 | no | **no** | powered | 0.000 | no |
| `momentum_L28d_S1d` | 69 | 0.911 | 0.955 | not_significant | 0.2791 | 0.383 (0.421) | no | -0.00197 | -0.104 | 0.1698 | 0.1780 | no | yes | powered | 0.060 | no |
| `momentum_L3d_S1d` | 69 | 1.606 | 1.533 | not_significant | 0.4098 | 0.674 (0.424) | no | 0.01265 | 0.811 | 0.0626 | 0.0656 | no | **no** | powered | 0.198 | no |
| `momentum_L56d_S1d` | 69 | 0.935 | 0.756 | not_significant | 0.2683 | 0.393 (0.525) | yes | 0.00777 | 0.351 | 0.2249 | 0.2024 | no | **no** | powered | 0.062 | no |
| `momentum_L5d_S1d` | 69 | 2.315 | 2.403 | suggestive | 0.6163 | 0.972 (0.428) | no | 0.04631 | 2.206 | 0.0081 | 0.0058 | yes | **no** | powered | 0.460 | **yes** |
| `momentum_L7d_S1d` | 69 | 2.068 | 1.887 | not_significant | 0.6209 | 0.868 (0.426) | no | 0.04020 | 1.614 | 0.0296 | 0.0214 | no | yes | powered | 0.352 | no |

## Diagnostics (skip {2,3}) — reported, not selected

| variant | n | naive t | HAC t | HLZ | ann ret | Sharpe (SE) | autocorr | span α | span α t | HAC p | boot p | disagree | holds sign | power | DSR | survives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `momentum_L14d_S2d` | 69 | 2.369 | 2.298 | suggestive | 0.6492 | 0.995 (0.428) | no | 0.03656 | 2.197 | 0.0108 | 0.0140 | no | yes | powered | 0.481 | no |
| `momentum_L14d_S3d` | 69 | 3.241 | 3.949 | significant | 0.8988 | 1.361 (0.436) | no | 0.06042 | 3.888 | 0.0000 | 0.0006 | no | yes | powered | 0.805 | no |
| `momentum_L1d_S2d` | 69 | 0.075 | 0.069 | not_significant | 0.0288 | 0.032 (0.420) | no | -0.03414 | -1.504 | 0.4724 | 0.4459 | no | **no** | powered | 0.010 | no |
| `momentum_L1d_S3d` | 69 | 3.338 | 2.851 | suggestive | 1.0432 | 1.402 (0.437) | no | 0.07471 | 2.654 | 0.0022 | 0.0054 | no | yes | powered | 0.870 | no |
| `momentum_L28d_S2d` | 69 | 0.872 | 0.937 | not_significant | 0.2497 | 0.366 (0.421) | no | 0.00301 | 0.167 | 0.1744 | 0.1758 | no | yes | powered | 0.060 | no |
| `momentum_L28d_S3d` | 69 | 1.416 | 1.393 | not_significant | 0.3889 | 0.595 (0.423) | no | 0.01918 | 1.061 | 0.0818 | 0.0804 | no | yes | powered | 0.142 | no |
| `momentum_L3d_S2d` | 69 | 2.766 | 2.378 | suggestive | 1.0109 | 1.161 (0.505) | yes | 0.04369 | 2.105 | 0.0087 | 0.0192 | no | **no** | powered | 0.653 | no |
| `momentum_L3d_S3d` | 69 | 4.663 | 5.015 | significant | 1.0807 | 1.958 (0.452) | no | 0.08099 | 4.735 | 0.0000 | 0.0002 | no | yes | powered | 0.985 | no |
| `momentum_L56d_S2d` | 69 | 1.217 | 0.990 | not_significant | 0.3780 | 0.511 (0.522) | yes | 0.00903 | 0.412 | 0.1610 | 0.1466 | no | **no** | powered | 0.099 | no |
| `momentum_L56d_S3d` | 69 | 1.075 | 0.891 | not_significant | 0.3159 | 0.451 (0.513) | yes | 0.01069 | 0.517 | 0.1866 | 0.1658 | no | **no** | powered | 0.081 | no |
| `momentum_L5d_S2d` | 69 | 2.508 | 2.588 | suggestive | 0.9616 | 1.053 (0.429) | no | 0.04495 | 2.048 | 0.0048 | 0.0108 | yes | yes | powered | 0.558 | no |
| `momentum_L5d_S3d` | 69 | 2.857 | 2.660 | suggestive | 0.8167 | 1.200 (0.432) | no | 0.05033 | 2.179 | 0.0039 | 0.0030 | no | yes | powered | 0.675 | no |
| `momentum_L7d_S2d` | 69 | 2.097 | 2.327 | suggestive | 0.6428 | 0.881 (0.427) | no | 0.03496 | 1.681 | 0.0100 | 0.0116 | no | yes | powered | 0.372 | no |
| `momentum_L7d_S3d` | 69 | 1.925 | 2.102 | suggestive | 0.5245 | 0.808 (0.426) | no | 0.03522 | 2.075 | 0.0178 | 0.0110 | no | yes | powered | 0.330 | no |

## Notes

- `holds_sign` is the Stage-2.6 deployment gate: a sign-flip across halves/thirds disqualifies a variant from deployment regardless of t-stat.
- The spanning alpha is the factor mean after partialling out {equal-weighted market return, small-minus-big size control}; the size control is a TEST-ONLY regressor (never deployed).
- DSR (deflated Sharpe) deflates each variant's Sharpe by the expected maximum across the full trial grid; PBO > 0.5 indicates selection overfitting.
- Survivorship: the dead/collapsed coins remain in the underlying series; this battery does not re-filter the universe.

