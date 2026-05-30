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

# --- Point-in-time eligibility thresholds (guide §1.1, spec §4 Stage 1.1) ---
MIN_HISTORY_DAYS = 90              # min price history, in days, to be eligible
MIN_ADV_USD = 1_000_000           # min trailing-30d average daily volume, USD

# --- Minimum-universe gate (point-in-time; guide §1.1, spec §4 Stage 1.1) ---
# A rebalance date with fewer than MIN_ELIGIBLE_NAMES eligible coins is gated
# (skipped), so each quintile has >= MIN_BUCKET_SIZE names. Convention, not tuned.
MIN_ELIGIBLE_NAMES = 20           # min eligible coins for a non-gated rebalance date
MIN_BUCKET_SIZE = 3               # min names per quintile bucket

# Market metrics used in all provider calls for the momentum factor
MARKET_METRICS = ("PRICE", "MC", "FDMC", "24H_VOLUME", "30D_VOLUME")

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
