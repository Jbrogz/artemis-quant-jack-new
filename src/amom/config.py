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
