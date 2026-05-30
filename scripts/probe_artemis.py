"""Artemis live connectivity and coverage probe.

Probes four aspects of the Artemis API:
1. BTC price history depth (first/last date, point count).
2. Multi-metric coverage for ~15 large-cap assets (PRICE, MC, 30D_VOLUME).
3. Dead/collapsed coin presence (luna, lunc, ftt, ust).
4. Broadest available asset list estimate.

Prints a structured summary line:
    API_OK=<bool> HISTORY_START=<date> N_ASSETS=<int> DEAD_COIN_COVERAGE=<list>

Usage:
    uv run python scripts/probe_artemis.py

Security: NEVER prints the API key — only len(key).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as a script without editable install being resolved.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from amom.providers.artemis import ArtemisProvider
from amom.providers.base import ProviderError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TODAY = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
BTC_START = "2013-01-01"

LARGE_CAPS = [
    "btc", "eth", "bnb", "sol", "xrp",
    "ada", "doge", "avax", "dot", "link",
    "matic", "ltc", "uni", "atom", "icp",
]

DEAD_COINS = ["luna", "lunc", "ftt", "ust"]

# Broad candidate set to estimate universe coverage when no supported-assets
# endpoint is available.
BROAD_CANDIDATES = [
    "btc", "eth", "bnb", "sol", "xrp", "ada", "doge", "avax", "dot", "link",
    "matic", "ltc", "uni", "atom", "icp", "shib", "trx", "dai", "wbtc",
    "xlm", "near", "algo", "bch", "etc", "fil", "vet", "hbar", "sand",
    "mana", "axs", "theta", "egld", "xtz", "eos", "aave", "mkr", "comp",
    "snx", "crv", "yfi", "sushi", "1inch", "bat", "zrx", "enj", "omg",
    "lrc", "imx", "grt", "flow", "iota", "neo", "ont", "qtum", "icx",
    "zil", "ren", "storj", "sc", "dcr", "kcs", "ht", "okb", "ftm",
    "one", "celo", "ksm", "rune", "osmo", "juno", "luna", "lunc",
    "ftt", "ust", "waves", "rose", "kava", "band", "ankr", "coti",
    "ctx", "reef", "pha", "vite", "sun", "win", "nft", "alice",
    "raca", "tlm", "chr", "alpha", "bake", "burger", "chess",
    "dusk", "for", "hard", "lazio", "nuls", "porto", "pols",
    "vai", "xvs", "high", "agix", "fet", "ocean", "rndr",
    "gala", "gmx", "arb", "op", "pepe", "floki", "bonk",
    "wld", "sei", "blur", "ordi", "sats", "inj", "apt", "sui",
]


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def probe_btc_history(provider: ArtemisProvider) -> tuple[str, str, int]:
    """Fetch BTC daily prices from 2013-01-01 to today."""
    _section("1. BTC Price History Depth")
    df = provider._get(["PRICE"], ["btc"], BTC_START, TODAY)
    price = df[(df["symbol"] == "btc") & (df["metric"] == "PRICE")].copy()
    price = price.sort_values("date").dropna(subset=["value"])

    if price.empty:
        print("  ERROR: no BTC price data returned")
        return "n/a", "n/a", 0

    first_date = price["date"].min().strftime("%Y-%m-%d")
    last_date = price["date"].max().strftime("%Y-%m-%d")
    n_points = len(price)
    print(f"  First date : {first_date}")
    print(f"  Last date  : {last_date}")
    print(f"  N points   : {n_points}")
    return first_date, last_date, n_points


def probe_large_cap_metrics(
    provider: ArtemisProvider,
) -> dict[str, list[str]]:
    """Fetch PRICE, MC, 30D_VOLUME for large-cap assets."""
    _section("2. Large-Cap Multi-Metric Coverage")
    start = (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    df = provider._get(["PRICE", "MC", "30D_VOLUME"], LARGE_CAPS, start, TODAY)

    metrics_by_symbol: dict[str, list[str]] = {}
    for sym in LARGE_CAPS:
        available = sorted(
            df[(df["symbol"] == sym) & df["value"].notna()]["metric"].unique().tolist()
        )
        metrics_by_symbol[sym] = available
        status = ", ".join(available) if available else "NONE"
        print(f"  {sym:10s}: {status}")

    return metrics_by_symbol


def probe_dead_coins(provider: ArtemisProvider) -> dict[str, bool]:
    """Check if collapsed coins return a price series."""
    _section("3. Dead / Collapsed Coin Presence")
    df = provider._get(["PRICE"], DEAD_COINS, "2018-01-01", TODAY)

    coverage: dict[str, bool] = {}
    for coin in DEAD_COINS:
        series = df[
            (df["symbol"] == coin) & (df["metric"] == "PRICE")
        ].sort_values("date").dropna(subset=["value"])

        if series.empty:
            coverage[coin] = False
            print(f"  {coin:6s}: NO DATA")
        else:
            last_val = series["value"].iloc[-1]
            last_dt = series["date"].iloc[-1].strftime("%Y-%m-%d")
            n = len(series)
            coverage[coin] = True
            print(f"  {coin:6s}: {n} points, last={last_dt}, last_price={last_val:.6g}")

    return coverage


def probe_asset_coverage(provider: ArtemisProvider) -> int:
    """Estimate the breadth of the Artemis universe."""
    _section("4. Broadest Asset List Estimate")

    # Try a supported-assets endpoint first (Artemis does not expose a public
    # /assets list in the documented API; skip gracefully if 403/404).
    n_from_endpoint = _try_assets_endpoint(provider)
    if n_from_endpoint is not None:
        print(f"  Supported-assets endpoint returned {n_from_endpoint} assets")
        return n_from_endpoint

    # Fall back: probe a large candidate set over a recent 30-day window.
    print(f"  No supported-assets endpoint found; probing {len(BROAD_CANDIDATES)} candidates …")
    start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    df = provider._get(["PRICE"], BROAD_CANDIDATES, start, TODAY)

    responding = sorted(
        df[df["metric"] == "PRICE"]["symbol"].unique().tolist()
    )
    n = len(responding)
    print(f"  {n}/{len(BROAD_CANDIDATES)} candidates returned PRICE data")
    print(f"  Responding: {responding[:20]}{'...' if len(responding) > 20 else ''}")
    return n


def _try_assets_endpoint(provider: ArtemisProvider) -> int | None:
    """Attempt a supported-assets discovery endpoint; return count or None."""
    import requests

    candidate_paths = [
        "/assets",
        "/supported-assets",
        "/assets/list",
    ]
    for path in candidate_paths:
        url = f"{provider._base_url.rstrip('/')}{path}"
        try:
            resp = requests.get(
                url,
                params={"APIKey": provider._api_key},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict):
                    # Try common wrapping keys
                    for key in ("assets", "symbols", "data", "items"):
                        if isinstance(data.get(key), list):
                            return len(data[key])
        except Exception:
            pass
    return None


def main() -> None:
    api_key = os.environ.get("ARTEMIS_API_KEY", "")
    print(f"ARTEMIS_API_KEY present: {bool(api_key)}  len={len(api_key)}")

    if not api_key:
        print("\nERROR: ARTEMIS_API_KEY not set. Load it via .env in the project root.")
        print("API_OK=False HISTORY_START=n/a N_ASSETS=0 DEAD_COIN_COVERAGE=[]")
        sys.exit(1)

    api_ok = False
    history_start = "n/a"
    n_assets = 0
    dead_coin_coverage: list[str] = []
    metrics_available: list[str] = []

    try:
        provider = ArtemisProvider(api_key=api_key)

        # 1. BTC history
        first_date, _last_date, n_points = probe_btc_history(provider)
        if n_points > 0:
            api_ok = True
            history_start = first_date

        # 2. Large-cap metrics
        metrics_by_sym = probe_large_cap_metrics(provider)
        all_metrics: set[str] = set()
        for m_list in metrics_by_sym.values():
            all_metrics.update(m_list)
        metrics_available = sorted(all_metrics)

        # 3. Dead coins
        dead_coverage = probe_dead_coins(provider)
        dead_coin_coverage = [c for c, ok in dead_coverage.items() if ok]

        # 4. Asset coverage
        n_assets = probe_asset_coverage(provider)

    except ProviderError as exc:
        print(f"\nProviderError: {exc}")
        api_ok = False

    except Exception as exc:
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}")
        api_ok = False

    # Structured summary line (required by the plan)
    _section("SUMMARY")
    print(
        f"API_OK={api_ok} "
        f"HISTORY_START={history_start} "
        f"N_ASSETS={n_assets} "
        f"DEAD_COIN_COVERAGE={dead_coin_coverage}"
    )
    print(f"METRICS_AVAILABLE={metrics_available}")


if __name__ == "__main__":
    main()
