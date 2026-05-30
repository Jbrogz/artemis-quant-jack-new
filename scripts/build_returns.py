"""Build the holding-return panel from the Artemis price series.

Pipeline (spec §4 Stage 1.2, plan Task S0):
  1. Load the asset registry (1,013 artemis_ids).
  2. Pull PRICE for every artemis_id on a daily grid (cached).
  3. Run the recycled-ticker splitter (crash+gap -> distinct synthetic assets).
  4. Load the on-disk universe panel (data/universe/universe_history.parquet).
  5. Run returns.spot.build_holding_returns(price_panel, universe_panel).
  6. Write data/returns/holding_returns.parquet (long: date, symbol, holding_return).
  7. Print rows / #symbols / date range; confirm terra (LUNA crash ~-99.99%).

Conventions (spec §3.1, §4 Stage 1.2):
  - Holding return = simple spot price return (no funding — Artemis serves none).
  - Terminal crash returns are carried; no collapsed coin is silently dropped.
  - The 'ret' column from spot.build_holding_returns is renamed 'holding_return'
    on output for clarity (plan Task S0 output schema).

Security: ARTEMIS_API_KEY loaded via dotenv; NEVER printed — only len(key).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from amom.config import DATA_DIR, MARKET_METRICS  # noqa: E402
from amom.providers.artemis import ArtemisProvider  # noqa: E402
from amom.providers.base import ProviderError  # noqa: E402
from amom.returns.spot import build_holding_returns  # noqa: E402
from amom.universe.recycle import split_recycled  # noqa: E402
from amom.universe.registry import enumerate_assets  # noqa: E402

HISTORY_START = "2018-01-01"
UNIVERSE_PANEL_PATH = DATA_DIR / "universe" / "universe_history.parquet"
OUTPUT_PATH = DATA_DIR / "returns" / "holding_returns.parquet"

# Max batch size for provider calls — mirrors build_universe.py.
_BATCH_SIZE = 10

# Terra (LUNA) classic artemis_id — confirm its realized terminal return is present.
_TERRA_SYM = "lunc"


def _long_price(market: pd.DataFrame) -> pd.DataFrame:
    """Extract the PRICE metric from the long market frame."""
    sub = market[(market["metric"] == "PRICE") & market["value"].notna()].copy()
    return (
        sub[["date", "symbol", "value"]]
        .rename(columns={"value": "price"})
        .reset_index(drop=True)
    )


def _fetch_price_panel(
    artemis_ids: list[str],
    provider: ArtemisProvider,
    history_start: str,
    history_end: str,
) -> pd.DataFrame:
    """Pull PRICE for all artemis_ids, batch-fetching with cache, then split recycled."""
    frames: list[pd.DataFrame] = []
    failed_batches: list[list[str]] = []
    total_batches = (len(artemis_ids) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(artemis_ids), _BATCH_SIZE)):
        batch = artemis_ids[i : i + _BATCH_SIZE]
        try:
            if hasattr(provider, "_cached_get"):
                chunk = provider._cached_get(
                    MARKET_METRICS, batch, history_start, history_end, "DAY"
                )
            else:
                chunk = provider._get(
                    list(MARKET_METRICS), batch, history_start, history_end, "DAY"
                )
            frames.append(chunk)
        except ProviderError as exc:
            failed_batches.append(batch)
            print(f"  [SKIP] batch {batch_idx + 1} failed: {exc!s:.120}", flush=True)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(
                f"  Fetched {batch_idx + 1}/{total_batches} batches "
                f"(failures so far: {len(failed_batches)}) ...",
                flush=True,
            )

    if not frames:
        raise ProviderError("No data returned from Artemis for any registry asset.")
    if failed_batches:
        n_failed = sum(len(b) for b in failed_batches)
        print(
            f"  WARNING: {len(failed_batches)} batches ({n_failed} ids) failed "
            f"and were skipped."
        )

    market = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "symbol", "metric"])
        .reset_index(drop=True)
    )
    price_panel = _long_price(market)
    price_panel["date"] = price_panel["date"].dt.normalize()

    # Recycled-ticker splitter: crash+gap -> distinct synthetic assets.
    price_panel = split_recycled(price_panel)
    return price_panel


def main() -> None:
    api_key = os.environ.get("ARTEMIS_API_KEY", "")
    print(f"ARTEMIS_API_KEY present: {bool(api_key)}  len={len(api_key)}")
    if not api_key:
        print("ERROR: ARTEMIS_API_KEY not set; cannot build holding returns.")
        sys.exit(1)

    # --- Guard: universe panel must exist. ---
    if not UNIVERSE_PANEL_PATH.exists():
        print(
            f"ERROR: universe panel not found at {UNIVERSE_PANEL_PATH}. "
            "Run scripts/build_universe.py first."
        )
        sys.exit(2)

    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    # --- Load registry. ---
    registry = enumerate_assets()
    registry = registry.dropna(subset=["artemis_id"]).reset_index(drop=True)
    artemis_ids = registry["artemis_id"].tolist()
    print(f"Registry: {len(artemis_ids)} artemis_ids. Pulling PRICE {HISTORY_START}..{today}")

    # --- Fetch price panel (cached from prior build_universe run). ---
    provider = ArtemisProvider(api_key=api_key)
    try:
        price_panel = _fetch_price_panel(artemis_ids, provider, HISTORY_START, today)
    except ProviderError as exc:
        print(f"ProviderError: {exc}")
        sys.exit(3)

    print(f"Price panel: {len(price_panel):,} rows, {price_panel['symbol'].nunique()} symbols")

    # --- Load universe panel. ---
    universe_panel = pd.read_parquet(UNIVERSE_PANEL_PATH)
    print(f"Universe panel loaded: {len(universe_panel):,} rows")

    # --- Build holding returns. ---
    raw = build_holding_returns(price_panel, universe_panel)

    # Rename 'ret' -> 'holding_return' (plan Task S0 output schema).
    holding = raw.rename(columns={"ret": "holding_return"})

    # --- Write output. ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    holding.to_parquet(OUTPUT_PATH, index=False)

    # --- Stats. ---
    n_rows = len(holding)
    n_syms = holding["symbol"].nunique()
    date_min = holding["date"].min().date()
    date_max = holding["date"].max().date()

    print()
    print("=" * 60)
    print("  HOLDING-RETURN PANEL STATS  (Task S0)")
    print("=" * 60)
    print(f"  rows            : {n_rows:,}")
    print(f"  #symbols        : {n_syms:,}")
    print(f"  date range      : {date_min}  to  {date_max}")
    print(f"  written         : {OUTPUT_PATH}")
    print()

    # --- Terra (LUNA) crash confirmation. ---
    # The Artemis registry maps artemis_id='terra' -> symbol='lunc' (Luna Classic).
    # In the panel the symbol column carries the artemis_id, so we search for
    # 'terra' as well as 'lunc' and any recycled-ticker segments.
    terra_candidates = ["terra", "lunc"]
    terra_syms = [
        s for s in holding["symbol"].unique()
        if any(s.lower().startswith(c) for c in terra_candidates)
    ]
    if terra_syms:
        for tsym in sorted(terra_syms):
            tseries = holding.loc[holding["symbol"] == tsym, "holding_return"]
            worst = float(tseries.min())
            print(f"  Terra ({tsym}) worst daily return: {worst:.6f}  ({worst*100:.2f}%)")
            if worst > -0.90:
                print(f"  WARNING: {tsym} worst return {worst:.4f} is above -90% threshold.")
    else:
        print("  WARNING: terra/lunc symbol not found in holding-return panel.")

    print()
    print("  Convention: holding_return = simple spot price return (p_t/p_{t-1} - 1).")
    print("  No funding term; Artemis serves spot prices only (spec §3.1).")
    print("  Terminal crash returns are carried; no collapsed coin is silently dropped.")


if __name__ == "__main__":
    main()
