"""Build the point-in-time universe eligibility panel from live Artemis data.

Pipeline (guide §1.1, spec §4 Stage 1.1):
  1. Pull PRICE + 24H_VOLUME for the broadest Artemis asset list.
  2. Pivot to long price / volume panels (every symbol ever seen, incl. dead).
  3. Build a monthly rebalance grid spanning the data.
  4. ``build_universe_history`` -> point-in-time eligibility + trailing-30d ADV
     + min-universe gate.
  5. Write ``data/universe/universe_history.parquet`` and print:
     rows, #symbols, #ever-eligible, #delisted-ever, #eligible-on-latest-date.

Artemis serves daily 24H_VOLUME (not 30D_VOLUME); the builder takes the
trailing-30d mean of 24H_VOLUME as the ADV liquidity measure (plan Task 5,
spec §3.2 daily granularity).

Usage:
    uv run python scripts/build_universe.py

Security: loads the key via dotenv; NEVER prints the API key — only len(key).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from amom.config import DATA_DIR, EXCLUDED  # noqa: E402
from amom.providers.artemis import ArtemisProvider  # noqa: E402
from amom.providers.base import ProviderError  # noqa: E402
from amom.universe.builder import build_universe_history  # noqa: E402
from amom.universe.coverage import first_seen_dates  # noqa: E402

# The broadest asset list Artemis serves (no public supported-assets endpoint;
# reuse the curated candidate set the probe validated).
from probe_artemis import BROAD_CANDIDATES  # noqa: E402

HISTORY_START = "2018-01-01"
OUTPUT_PATH = DATA_DIR / "universe" / "universe_history.parquet"
# Survivorship grace: a coin is "delisted-ever" if its last price predates the
# latest grid date by more than this many days (it stopped reporting).
DELISTED_GRACE_DAYS = 30


def _long_metric(market: pd.DataFrame, metric: str, value_name: str) -> pd.DataFrame:
    """Slice one metric out of the long market frame, renaming value -> value_name."""
    sub = market[(market["metric"] == metric) & market["value"].notna()]
    out = sub[["date", "symbol", "value"]].rename(columns={"value": value_name})
    return out.reset_index(drop=True)


def main() -> None:
    api_key = os.environ.get("ARTEMIS_API_KEY", "")
    print(f"ARTEMIS_API_KEY present: {bool(api_key)}  len={len(api_key)}")
    if not api_key:
        print("ERROR: ARTEMIS_API_KEY not set; cannot build a sound universe.")
        sys.exit(1)

    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    symbols = sorted(set(BROAD_CANDIDATES))
    print(f"Pulling PRICE + 24H_VOLUME for {len(symbols)} symbols, {HISTORY_START}..{today}")

    provider = ArtemisProvider(api_key=api_key)
    try:
        market = provider._get(["PRICE", "24H_VOLUME"], symbols, HISTORY_START, today)
    except ProviderError as exc:
        print(f"ProviderError: {exc}")
        sys.exit(2)

    price_panel = _long_metric(market, "PRICE", "price")
    volume_panel = _long_metric(market, "24H_VOLUME", "volume")
    if price_panel.empty:
        print("ERROR: no PRICE data returned from Artemis.")
        sys.exit(3)

    # Monthly rebalance grid spanning the observed price history.
    grid_start = price_panel["date"].min().normalize()
    grid_end = price_panel["date"].max().normalize()
    dates = pd.date_range(grid_start, grid_end, freq="MS")
    print(f"Grid: {len(dates)} monthly dates {dates.min().date()}..{dates.max().date()}")

    panel = build_universe_history(
        price_panel, volume_panel, dates=dates, excluded=EXCLUDED
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)

    # --- Stats ---
    n_rows = len(panel)
    n_symbols = panel["symbol"].nunique()
    ever_eligible = panel.loc[panel["eligible"], "symbol"].nunique()

    coverage = first_seen_dates(price_panel)
    latest = dates.max()
    delisted_cutoff = latest - pd.Timedelta(days=DELISTED_GRACE_DAYS)
    delisted_ever = int((coverage["price_last_date"] < delisted_cutoff).sum())

    latest_rows = panel[panel["date"] == latest]
    eligible_latest = int(latest_rows["eligible"].sum())
    gated_latest = bool(latest_rows["gated"].iloc[0]) if not latest_rows.empty else None

    print("\n" + "=" * 60)
    print("  UNIVERSE PANEL STATS")
    print("=" * 60)
    print(f"  rows                    : {n_rows}")
    print(f"  #symbols                : {n_symbols}")
    print(f"  #ever-eligible          : {ever_eligible}")
    print(f"  #delisted-ever          : {delisted_ever}")
    print(f"  #eligible-on-latest-date: {eligible_latest}  (latest={latest.date()})")
    print(f"  latest-date gated?      : {gated_latest}")
    print(f"  written                 : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
