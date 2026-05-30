"""Build the point-in-time universe eligibility panel from the Artemis registry.

Pipeline (guide §1.1, spec §4 Stage 1.1, plan Task R5):
  1. Load the Artemis /asset registry -> 1,013 artemis_ids (no hand-curated list).
  2. Pull PRICE + 24H_VOLUME + MC for every artemis_id on a daily grid.
  3. Run the recycled-ticker splitter (crash+gap -> distinct synthetic assets).
  4. Build the panel on a DAILY grid via build_universe_history.
  5. Write data/universe/universe_history.parquet.
  6. Print:
       rows, #assets, #ever-eligible, #eligible-on-latest,
       #assets showing a terminal >90% collapse (survivorship figure).

Artemis serves daily 24H_VOLUME (not 30D_VOLUME); liquidity gate uses
trailing-30d median 24H_VOLUME + MC floor (plan Task R2, spec §3.5).

Security: loads ARTEMIS_API_KEY via dotenv; NEVER prints the key — only len(key).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from amom.config import DATA_DIR, EXCLUDED, MARKET_METRICS  # noqa: E402
from amom.providers.artemis import ArtemisProvider  # noqa: E402
from amom.providers.base import ProviderError  # noqa: E402
from amom.universe.builder import build_universe_history  # noqa: E402
from amom.universe.recycle import split_recycled  # noqa: E402
from amom.universe.registry import enumerate_assets  # noqa: E402

HISTORY_START = "2018-01-01"
OUTPUT_PATH = DATA_DIR / "universe" / "universe_history.parquet"

# Survivorship threshold: >90% drawdown from a symbol's running peak
# that is sustained (price never recovers) counts as a terminal collapse.
_TERMINAL_DRAWDOWN_THRESH = 0.90

# Max batch size for provider calls — chunk the 1,013 registry assets
# so each request is within the provider's _MAX_SYMBOLS_PER_CALL limit.
_BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# Helpers (testable pure functions)
# ---------------------------------------------------------------------------

def _long_metric(
    market: pd.DataFrame, metric: str, value_name: str
) -> pd.DataFrame:
    """Extract one metric from the long market frame, renaming value -> value_name."""
    sub = market[(market["metric"] == metric) & market["value"].notna()].copy()
    return (
        sub[["date", "symbol", "value"]]
        .rename(columns={"value": value_name})
        .reset_index(drop=True)
    )


def compute_terminal_collapses(
    price_panel: pd.DataFrame,
    *,
    drawdown_thresh: float = _TERMINAL_DRAWDOWN_THRESH,
) -> int:
    """Count symbols with a terminal drawdown exceeding drawdown_thresh.

    A symbol counts as a "terminal collapse" iff its last observed price is
    <= (1 - drawdown_thresh) * its all-time peak price. This is a conservative,
    point-in-time-safe measure of survivorship using the realized series.

    Args:
        price_panel: long DataFrame [date, symbol, price] (already filtered to
            non-NaN prices).
        drawdown_thresh: fraction of peak that counts as a collapse (default 0.9).

    Returns:
        Count of symbols exhibiting a terminal >drawdown_thresh collapse.
    """
    if price_panel.empty:
        return 0

    priced = price_panel.dropna(subset=["price"])
    if priced.empty:
        return 0

    count = 0
    for sym, group in priced.groupby("symbol", sort=False):
        prices = group["price"].to_numpy(dtype=float)
        if prices.size == 0:
            continue
        peak = float(np.max(prices))
        last = float(prices[-1]) if not group.empty else float("nan")
        if peak > 0 and last <= (1.0 - drawdown_thresh) * peak:
            count += 1
    return count


def cohort_split_collapses(
    price_panel: pd.DataFrame,
    universe_panel: pd.DataFrame,
    *,
    drawdown_thresh: float = _TERMINAL_DRAWDOWN_THRESH,
) -> tuple[int, int]:
    """Split terminal collapses into two cohorts: delisted vs zombie.

    - **Delisted**: symbols that eventually fired ``delisted_asof=True`` in
      the universe panel (i.e. their price reporting lapsed past the
      ``LISTING_STALENESS_DAYS`` grace). These are dead — they stopped
      printing.
    - **Zombie**: symbols with a terminal >``drawdown_thresh`` collapse that
      are *still printing* today (delisted_asof never fires, but they trade
      at near-zero permanently, e.g. a coin on life-support with micro
      volume).

    Only symbols that actually show a terminal collapse (per the same peak-
    last rule as ``compute_terminal_collapses``) are counted in either cohort.

    Args:
        price_panel: long ``[date, symbol, price]`` after recycled-ticker split.
        universe_panel: the 8-column eligibility panel from ``build_universe_history``.
        drawdown_thresh: collapse threshold (default 0.90, i.e. >90% drawdown).

    Returns:
        Tuple of (n_delisted, n_zombie) — counts of the two collapse cohorts.
    """
    if price_panel.empty or universe_panel.empty:
        return (0, 0)

    priced = price_panel.dropna(subset=["price"])
    if priced.empty:
        return (0, 0)

    # Set of symbols that ever fired delisted_asof=True (stopped reporting).
    ever_delisted: set[str] = set(
        universe_panel.loc[universe_panel["delisted_asof"], "symbol"]
    )

    n_delisted = 0
    n_zombie = 0
    for sym, group in priced.groupby("symbol", sort=False):
        prices = group.sort_values("date")["price"].to_numpy(dtype=float)
        if prices.size == 0:
            continue
        peak = float(np.max(prices))
        last = float(prices[-1])
        collapsed = peak > 0 and last <= (1.0 - drawdown_thresh) * peak
        if not collapsed:
            continue
        if sym in ever_delisted:
            n_delisted += 1
        else:
            n_zombie += 1

    return (n_delisted, n_zombie)


def build_panel(
    registry: pd.DataFrame,
    provider: ArtemisProvider,
    history_start: str,
    history_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull data for all registry artemis_ids and build the universe panel.

    This is the testable core extracted from main() so tests can patch it
    without needing filesystem I/O or a real API key.

    Args:
        registry: the asset registry DataFrame [artemis_id, symbol, ...].
        provider: an ArtemisProvider (or compatible) instance.
        history_start: ISO date string for the start of the pull.
        history_end: ISO date string for the end of the pull.

    Returns:
        Tuple of (universe_panel, price_panel). ``universe_panel`` is the
        8-column long panel ``[date, symbol, eligible, adv_30d,
        price_last_date, delisted_asof, left_censored, gated]`` from
        ``build_universe_history``. ``price_panel`` is the long
        ``[date, symbol, price]`` frame after recycled-ticker splitting —
        used by callers to compute holding returns and survivorship stats.
    """
    artemis_ids = registry["artemis_id"].dropna().tolist()

    # Pull PRICE + 24H_VOLUME + MC for all registry artemis_ids.
    # The Artemis data API uses artemis_id as the symbol identifier.
    # Use _cached_get per batch so subsequent runs are fast.
    # Resilient: HTTP 500 on a batch is skipped (logged), not fatal — some
    # obscure artemis_ids trigger server errors; we keep the rest of the data.
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
            # Skip batches that trigger server errors (some artemis_ids are
            # equity tickers or otherwise unsupported by the data endpoint).
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
        n_failed_ids = sum(len(b) for b in failed_batches)
        print(
            f"  WARNING: {len(failed_batches)} batches ({n_failed_ids} ids) failed "
            f"and were skipped. These assets are excluded from the panel."
        )

    market = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "symbol", "metric"])
        .reset_index(drop=True)
    )

    price_panel = _long_metric(market, "PRICE", "price")
    volume_panel = _long_metric(market, "24H_VOLUME", "volume")
    mc_panel = _long_metric(market, "MC", "mc")

    if price_panel.empty:
        raise ProviderError("No PRICE data returned from Artemis.")

    # --- Normalize dates to midnight so all <= / > comparisons are stable. ---
    price_panel["date"] = price_panel["date"].dt.normalize()
    volume_panel["date"] = volume_panel["date"].dt.normalize()
    mc_panel["date"] = mc_panel["date"].dt.normalize()

    # --- Recycled-ticker splitter: crash+gap -> distinct synthetic assets. ---
    price_panel = split_recycled(price_panel)

    # --- Daily grid spanning the observed price history. ---
    grid_start = price_panel["date"].min().normalize()
    grid_end = price_panel["date"].max().normalize()
    dates = pd.date_range(grid_start, grid_end, freq="D")

    # --- Build the point-in-time universe eligibility panel. ---
    panel = build_universe_history(
        price_panel,
        volume_panel,
        mc_panel,
        dates=dates,
        excluded=EXCLUDED,
    )
    return panel, price_panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ARTEMIS_API_KEY", "")
    print(f"ARTEMIS_API_KEY present: {bool(api_key)}  len={len(api_key)}")
    if not api_key:
        print("ERROR: ARTEMIS_API_KEY not set; cannot build a sound universe.")
        sys.exit(1)

    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    # --- Load the registry (enumerates all 1,013 Artemis assets). ---
    registry = enumerate_assets()
    print(f"Registry loaded: {len(registry)} assets, artemis_id unique={registry['artemis_id'].is_unique}")

    # Filter: drop rows with null artemis_id (edge case).
    registry = registry.dropna(subset=["artemis_id"]).reset_index(drop=True)

    print(f"Pulling PRICE + 24H_VOLUME + MC for {len(registry)} artemis_ids, {HISTORY_START}..{today}")

    provider = ArtemisProvider(api_key=api_key)
    try:
        panel, price_panel = build_panel(registry, provider, HISTORY_START, today)
    except ProviderError as exc:
        print(f"ProviderError: {exc}")
        sys.exit(2)

    # --- Write output. ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)

    # --- Stats. ---
    n_rows = len(panel)
    n_assets = panel["symbol"].nunique()
    ever_eligible = panel.loc[panel["eligible"], "symbol"].nunique()

    latest = panel["date"].max()
    latest_rows = panel[panel["date"] == latest]
    eligible_latest = int(latest_rows["eligible"].sum())

    # Survivorship figure: #assets with a terminal >90% collapse, split by cohort.
    # Uses the actual price panel returned by build_panel (includes the
    # recycled-ticker-split series) to count realized collapses from peak.
    n_terminal_collapse = compute_terminal_collapses(
        price_panel, drawdown_thresh=_TERMINAL_DRAWDOWN_THRESH
    )
    n_delisted, n_zombie = cohort_split_collapses(
        price_panel, panel, drawdown_thresh=_TERMINAL_DRAWDOWN_THRESH
    )

    print()
    print("=" * 60)
    print("  UNIVERSE PANEL STATS  (Task S0)")
    print("=" * 60)
    print(f"  rows                    : {n_rows:,}")
    print(f"  #assets                 : {n_assets:,}")
    print(f"  #ever-eligible          : {ever_eligible:,}")
    print(f"  #eligible-on-latest-date: {eligible_latest:,}  (latest={latest.date()})")
    print(f"  #assets with terminal >90% collapse: {n_terminal_collapse:,}")
    print(f"    delisted cohort (stopped reporting)  : {n_delisted:,}")
    print(f"    zombie cohort (still printing, ~zero): {n_zombie:,}")
    print(f"  written                 : {OUTPUT_PATH}")
    print()
    print("  Survivorship note:")
    print("  Terminal collapse = last observed price is <=10% of the all-time")
    print("  peak price for that asset. lunc/terra is included (LUNA crash).")
    print("  Delisted cohort: stopped reporting (delisted_asof flag fired).")
    print("  Zombie cohort: still printing at near-zero (never delisted_asof).")
    print("  Collapses are carried as realized crash returns in the panel.")
    print("  Residual survivorship from assets purged by Artemis is disclosed")
    print("  in docs/AUDIT.md (spec §3.6 / §10) but cannot be recovered.")


if __name__ == "__main__":
    main()
