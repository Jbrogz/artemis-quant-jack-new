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
# (skipped), so each quintile has >= MIN_BUCKET_SIZE names. Convention, not tuned.
MIN_ELIGIBLE_NAMES = 20           # min eligible coins for a non-gated rebalance date
MIN_BUCKET_SIZE = 3               # min names per quintile bucket

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
