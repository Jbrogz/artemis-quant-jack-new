"""Static configuration: momentum-only constants for amom.

Copied from src/cmom/config.py — only the Artemis market metrics, exclusion
sets, and paths are included. On-chain/dev metrics and sleeve parameters from
cmom are NOT copied (out of scope for the momentum factor book).
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# --- Artemis API ---
ARTEMIS_BASE_URL = "https://data-svc.artemisxyz.com/data/api"

# --- Asset catalog (no API key required; spec §4 Stage 1.1, Appendix B) ---
ASSET_CATALOG_URL = "https://data-svc.artemisxyz.com/asset"

# --- Point-in-time eligibility thresholds (guide §1.1, spec §4 Stage 1.1) ---
MIN_HISTORY_DAYS = 90              # min price history, in days, to be eligible

# --- Liquidity gate (spec §3.5): MC floor + trailing-30d median 24H_VOLUME ---
# 30D_VOLUME is real-time-only (sentinel on historical pulls); not used.
MIN_MC_USD = 10_000_000           # min market cap, USD
MIN_MEDIAN_VOL_USD = 1_000_000    # min trailing-30d median 24H_VOLUME, USD
LIQUIDITY_VOL_WINDOW_DAYS = 30    # window for median volume calculation

# --- Observation density + tradeability (spec §4 Stage 1.1) ---
MIN_OBS_DENSITY = 0.5             # min fraction of trailing-90d days with a price
LISTING_STALENESS_DAYS = 7        # max days since last price to be tradeable

# --- Universe panel grid ---
UNIVERSE_GRID_FREQ = "D"          # daily grid (guide §1.1 "rebuild daily")

# --- Minimum-universe gate (point-in-time; guide §1.1, spec §4 Stage 1.1) ---
# A rebalance date with fewer than MIN_ELIGIBLE_NAMES eligible coins is gated
# (skipped). The floor is DERIVED from MIN_BUCKET_SIZE: a non-gated date must be
# able to field a full quintile sort with >= MIN_BUCKET_SIZE names per bucket,
# i.e. N_QUINTILES * MIN_BUCKET_SIZE. Not a coincidental constant. (spec §4)
MIN_BUCKET_SIZE = 3               # min names per quintile bucket
N_QUINTILES = 5                   # quintile sort -> 5 buckets
MIN_ELIGIBLE_NAMES = N_QUINTILES * MIN_BUCKET_SIZE  # min eligible for non-gated

# --- Momentum signal grid (spec §1.3; fixed in advance, NOT selected from backtest) ---
# Academic LTW baseline (14/28/56d = 2/4/8 weeks) plus crypto-short horizons.
LOOKBACKS_DAYS = (1, 3, 5, 7, 14, 28, 56)   # 7 lookbacks, frozen
# Skip is a nuisance parameter fixed by convention; {2,3} are robustness only
# (separately reported), never a selection axis (spec §1.3, Appendix A H1).
PRIMARY_SKIP_DAYS = 1
ROBUSTNESS_SKIPS = (2, 3)

# --- Portfolio formation grid (spec §1.4; fixed by convention, NOT tuned) ---
# Quintile sort: long top QUANTILE, short bottom QUANTILE, equal-weight within
# each leg, dollar-neutral (Σ weights = 0). factor_return = long_leg − short_leg.
QUANTILE = 0.20                   # top/bottom 20% (quintile sort)
# Canonical one-month hold (spec §1.4 / guide §1.4). The factor-return series is
# a per-rebalance series; each holding window spans HOLDING_DAYS grid days.
HOLDING_DAYS = 30
# No-look-ahead execution lag (spec §3.2, §7): the signal computed through close
# t drives the trade entered at close t+1, i.e. the signal used at rebalance
# date r is the one dated r − LAG_DAYS; the holding window is (r, r+HOLDING_DAYS].
LAG_DAYS = 1

# --- Stage-2 power / effective-n threshold (spec §2.0) ---
# A variant whose *effective* (non-overlapping) sample size falls below this
# floor is labelled "inconclusive (underpowered)" rather than "insignificant":
# too few independent draws to distinguish a true zero mean from a small one.
# Pinned by convention (a conventional small-sample floor), NOT tuned to a
# result. The 99 non-overlapping 30-day obs clear it; an overlapping series
# whose effective n collapses below it is flagged underpowered.
MIN_EFFECTIVE_N = 30

# Market metrics used in all provider calls for the momentum factor.
# 30D_VOLUME is real-time-only on Artemis (sentinel on historical pulls) and
# FDMC is unused; both are dropped. Only 24H_VOLUME is historical. (spec §3.5, Appendix B)
MARKET_METRICS = ("PRICE", "MC", "24H_VOLUME")

# --- Exclusions ---
STABLECOINS = frozenset({
    "usdt", "usdc", "dai", "tusd", "usds", "fdusd", "pyusd", "usde",
    "gusd", "usdp", "frax", "lusd", "usdd", "susd",
})
WRAPPED = frozenset({
    "wbtc", "weth", "steth", "wsteth", "wbeth", "reth", "cbeth",
    "rseth", "weeth", "meth", "lbtc", "cbbtc", "sweth",
})
EXCLUDED = STABLECOINS | WRAPPED
