"""Build the dollar-neutral momentum factor-return series for every variant.

Pipeline (spec §1.4, plan Task S2):
  1. Load the holding-return panel (data/returns/holding_returns.parquet) and the
     universe panel (data/universe/universe_history.parquet) from disk.
  2. Reconstruct a per-symbol price level from the realized daily returns (the
     panel IS the daily simple returns; a cumulative product yields a price whose
     daily log returns are byte-identical to the source prices, verified) so the
     tested ``build_momentum_signal`` (Task S1) can be reused without re-hitting
     the API. The signal is defined by price returns; the returns panel is their
     authoritative source.
  3. For each of the 7 lookbacks at the primary skip (=1) plus the robustness
     skips {2, 3} as diagnostics, build the signal and the dollar-neutral
     long/short portfolio, and compute the per-rebalance factor-return series.
  4. Write data/factor/factor_returns.parquet
     (variant, rebalance_date, factor_return, long_return, short_return,
      n_long, n_short).
  5. Print per-variant n_obs, mean, and annualized gross Sharpe (pre-significance).

This script is fully offline (no API key needed): the signal and the factor are
derived from the on-disk panels built by build_returns.py / build_universe.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amom.config import (  # noqa: E402
    DATA_DIR,
    HOLDING_DAYS,
    LOOKBACKS_DAYS,
    PRIMARY_SKIP_DAYS,
    ROBUSTNESS_SKIPS,
)
from amom.factor.momentum import build_momentum_signal  # noqa: E402
from amom.factor.portfolio import build_factor_returns, build_rebalance_dates  # noqa: E402

RETURNS_PATH = DATA_DIR / "returns" / "holding_returns.parquet"
UNIVERSE_PATH = DATA_DIR / "universe" / "universe_history.parquet"
OUTPUT_PATH = DATA_DIR / "factor" / "factor_returns.parquet"

# Calendar days per year for annualizing the per-rebalance (HOLDING_DAYS-spaced)
# factor-return series.
DAYS_PER_YEAR = 365.0

_OUTPUT_COLUMNS = [
    "variant",
    "rebalance_date",
    "factor_return",
    "long_return",
    "short_return",
    "n_long",
    "n_short",
]


def _returns_wide(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Long ``[date, symbol, holding_return]`` -> wide (dates x symbols) returns."""
    wide = returns_long.pivot(index="date", columns="symbol", values="holding_return")
    return wide.sort_index()


def _reconstructed_price_panel(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a long ``[date, symbol, price]`` panel from the daily returns.

    Per symbol the price level is ``cumprod(1 + ret)`` over its observed dates
    (base level arbitrary; the momentum signal is a log-RETURN sum, invariant to
    the base). Days with no realized return are dropped per symbol so consecutive
    prices reproduce exactly the realized daily returns — the resulting signal is
    byte-identical to one built from the source Artemis prices.
    """
    frames = []
    for sym in returns_wide.columns:
        ser = returns_wide[sym].dropna()
        if ser.empty:
            continue
        price = (1.0 + ser).cumprod()
        frames.append(
            pd.DataFrame({"date": price.index, "symbol": sym, "price": price.to_numpy()})
        )
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    return pd.concat(frames, ignore_index=True)


def _eligibility_wide(universe_long: pd.DataFrame) -> pd.DataFrame:
    """Universe panel -> wide (dates x symbols) bool eligibility mask.

    A coin is eligible only on dates where the panel both marks it ``eligible``
    AND the date is not min-universe-``gated`` (a gated date fields too few names
    for a full quintile sort, so no rebalance happens there). Missing cells are
    treated as ineligible.
    """
    elig = universe_long.copy()
    elig["effective"] = elig["eligible"].astype(bool) & (~elig["gated"].astype(bool))
    wide = elig.pivot(index="date", columns="symbol", values="effective")
    return wide.sort_index().fillna(False).astype(bool)


def _annualized_sharpe(returns: pd.Series, holding_days: int) -> float:
    """Annualized gross Sharpe of a per-rebalance factor-return series.

    Each observation spans ``holding_days`` calendar days, so the per-period
    Sharpe is scaled by ``sqrt(DAYS_PER_YEAR / holding_days)``. Returns NaN when
    fewer than 2 observations or zero dispersion (Sharpe undefined).
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    per_period = float(r.mean()) / sd
    return per_period * np.sqrt(DAYS_PER_YEAR / holding_days)


def _variant_name(lookback: int, skip: int) -> str:
    return f"momentum_L{lookback}d_S{skip}d"


def _build_variant(
    price_panel: pd.DataFrame,
    eligibility: pd.DataFrame,
    returns_wide: pd.DataFrame,
    lookback: int,
    skip: int,
) -> pd.DataFrame:
    """Build one variant's factor-return series, tagged with its variant name."""
    signal = build_momentum_signal(price_panel, lookback, skip)
    rebal = build_rebalance_dates(signal.index, holding_days=HOLDING_DAYS)
    fr = build_factor_returns(signal, eligibility, returns_wide, rebal)
    if fr.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    fr = fr.copy()
    fr.insert(0, "variant", _variant_name(lookback, skip))
    return fr[_OUTPUT_COLUMNS]


def main() -> int:
    if not RETURNS_PATH.exists():
        print(f"ERROR: {RETURNS_PATH} missing. Run scripts/build_returns.py first.")
        return 1
    if not UNIVERSE_PATH.exists():
        print(f"ERROR: {UNIVERSE_PATH} missing. Run scripts/build_universe.py first.")
        return 2

    returns_long = pd.read_parquet(RETURNS_PATH)
    universe_long = pd.read_parquet(UNIVERSE_PATH)
    print(f"Holding returns: {len(returns_long):,} rows, "
          f"{returns_long['symbol'].nunique()} symbols")
    print(f"Universe panel:  {len(universe_long):,} rows")

    returns_wide = _returns_wide(returns_long)
    price_panel = _reconstructed_price_panel(returns_wide)
    eligibility = _eligibility_wide(universe_long)
    print(f"Returns wide:    {returns_wide.shape[0]} dates x {returns_wide.shape[1]} symbols")
    print(f"Eligibility:     {eligibility.shape[0]} dates x {eligibility.shape[1]} symbols")
    print()

    # Primary skip (=1) for all 7 lookbacks; robustness skips {2,3} as diagnostics.
    skips = (PRIMARY_SKIP_DAYS,) + tuple(ROBUSTNESS_SKIPS)
    variant_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for skip in skips:
        kind = "primary" if skip == PRIMARY_SKIP_DAYS else "diagnostic"
        for lookback in LOOKBACKS_DAYS:
            fr = _build_variant(price_panel, eligibility, returns_wide, lookback, skip)
            variant_frames.append(fr)
            name = _variant_name(lookback, skip)
            n_obs = len(fr)
            mean = float(fr["factor_return"].mean()) if n_obs else float("nan")
            sharpe = _annualized_sharpe(fr["factor_return"], HOLDING_DAYS) if n_obs else float("nan")
            summary_rows.append(
                {
                    "variant": name,
                    "kind": kind,
                    "skip": skip,
                    "lookback": lookback,
                    "n_obs": n_obs,
                    "mean": mean,
                    "ann_sharpe": sharpe,
                }
            )

    factor_returns = pd.concat(variant_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    factor_returns.to_parquet(OUTPUT_PATH, index=False)

    print("=" * 78)
    print(f"  MOMENTUM FACTOR-RETURN SERIES  (Task S2)  ->  {OUTPUT_PATH}")
    print(f"  hold={HOLDING_DAYS}d  primary skip={PRIMARY_SKIP_DAYS}  "
          f"diagnostics skip={tuple(ROBUSTNESS_SKIPS)}  (gross, pre-significance)")
    print("=" * 78)
    print(f"  {'variant':<22} {'kind':<11} {'n_obs':>6} {'mean':>12} {'ann_Sharpe':>12}")
    print("  " + "-" * 70)
    last_kind = None
    for row in summary_rows:
        if last_kind is not None and row["kind"] != last_kind:
            print("  " + "-" * 70)
        last_kind = row["kind"]
        print(
            f"  {row['variant']:<22} {row['kind']:<11} {row['n_obs']:>6} "
            f"{row['mean']:>+12.6f} {row['ann_sharpe']:>12.4f}"
        )
    print()
    print(f"  total rows written: {len(factor_returns):,}  "
          f"({len(summary_rows)} variants: 7 lookbacks x {len(skips)} skips)")
    print("  factor_return = long_leg - short_leg, dollar-neutral, equal-weight,")
    print("  signal through close t drives entry at close t+1 (no look-ahead).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
