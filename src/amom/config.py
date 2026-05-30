"""Static configuration: momentum-only constants for amom.

Copied from src/cmom/config.py — only the Artemis market metrics, exclusion
sets, and paths are included. On-chain/dev metrics and sleeve parameters from
cmom are NOT copied (out of scope for the momentum factor book).
"""
from pathlib import Path

import pandas as pd

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

# --- Out-of-sample reserve (spec §2.8 / guide §2.8) — SEALED ---------------
# Before any Stage-2 statistic is computed, the most recent ~30% of rebalance
# dates are set aside and sealed for a single Stage-4 evaluation. OOS_START is
# the rebalance date at the 70th percentile of the (sorted, deduped) rebalance
# dates of the built factor-return series — frozen here as a LITERAL constant,
# never recomputed at runtime (so a later panel mutation cannot move the split).
#
# Derivation (recorded for provenance, not re-evaluated): the series has 99
# non-overlapping monthly rebalance dates; sorted index 69 (= floor(0.70 * 99))
# is 2023-12-02, giving 69 in-sample and 30 out-of-sample observations per
# variant — the intended ~30% reserve. No Stage-2 code may read rows dated
# >= OOS_START; that window is opened exactly once in Stage 4.
OOS_START = pd.Timestamp("2023-12-02")


def in_sample(df: pd.DataFrame, date_col: str = "rebalance_date") -> pd.DataFrame:
    """In-sample slice: rows strictly before ``OOS_START`` (spec §2.8).

    The out-of-sample window (``date_col >= OOS_START``) is sealed for the single
    Stage-4 evaluation; every Stage-2 selection/estimation path must route its
    input through this guard so no statistic can read an OOS row. Returns a
    **copy** (immutable contract: the caller cannot mutate the source frame).

    Args:
        df: a frame carrying a rebalance-date column.
        date_col: the name of that column (default ``"rebalance_date"``).

    Returns:
        A copy of ``df`` containing only rows with ``df[date_col] < OOS_START``.
    """
    return df.loc[df[date_col] < OOS_START].copy()

# --- Spot cost model (Stage 4, spec §4.2; all bps, by convention, disclosed) ---
# Fees: a spot TAKER fee charged PER SIDE on the traded notional. No maker
# rebate is assumed (the strategy crosses the spread at the t+1 close). 10 bps
# is a conservative retail spot taker rate.
TAKER_FEE_BPS = 10                 # spot taker fee, per side, on traded notional
# Slippage: size-scaled and tiered by liquidity rank. Names inside the top
# SLIPPAGE_TOP_N (rank < N) pay the lower SLIPPAGE_TOP_BPS; smaller / less liquid
# names pay SLIPPAGE_SMALL_BPS. The tier bps is the slippage of an order equal to
# SLIPPAGE_ADV_REF of the coin's ADV; larger fractions of ADV scale the bps up
# linearly in the order/ADV ratio (a market-impact proxy). No funding term —
# this is a spot strategy and Artemis has no funding (spec §3.1; N/A, disclosed).
SLIPPAGE_TOP_BPS = 5               # base slippage (bps) for top-N liquid names
SLIPPAGE_SMALL_BPS = 15            # base slippage (bps) for smaller / illiquid names
SLIPPAGE_TOP_N = 30                # liquidity-rank cutoff: rank < N is the top tier
SLIPPAGE_ADV_REF = 0.01            # order/ADV reference: base bps applies at this ratio

# --- Backtest book caps + vol target (Stage 4, spec §4.1 / §3.5; by convention) ---
# Gross leverage cap: the sum of |position weight| across coins may not exceed
# this. 2.0 (a 1x-long / 1x-short dollar-neutral book) sits at the bottom of the
# guide's 2-3x band, pinned by convention — NOT tuned to a backtest (spec §3.6).
GROSS_LEVERAGE_CAP = 2.0
# Per-coin cap: no single coin's |weight| may exceed this fraction of the book.
# A loose 20% cap that almost never binds on a quintile book (equal-weight legs
# of >=3 names cap at 1/3), present to bound concentration if a leg is thin.
PER_COIN_CAP = 0.20
# Annual vol target the book is scaled to (spec §3.5): the walk-forward vol
# scalar multiplies the raw dollar-neutral weights so the book's trailing
# realized vol annualizes to this. Pinned by convention, not tuned.
ANNUAL_VOL_TARGET = 0.20
# Trailing window (in rebalance periods) for the realized-vol estimate that
# drives the vol scalar. Walk-forward: only periods <= t enter. A conventional
# ~1y window at the 30-day cadence (~12 periods/yr); not Sharpe-maximized.
VOL_TARGET_LOOKBACK = 12

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
