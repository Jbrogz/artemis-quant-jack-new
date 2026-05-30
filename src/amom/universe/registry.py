"""Artemis asset registry: enumerate all assets from the /asset catalog.

The /asset endpoint returns the full Artemis asset list (1,013 assets as of
2026-05-30) keyed on a stable `artemis_id` slug.  No API key is required.
Everything downstream must key on `artemis_id`, never on the mutable symbol.

Spec §4 Stage 1.1 and Appendix B:
  GET https://data-svc.artemisxyz.com/asset
  → list of {artemis_id, symbol, coingecko_id (partial), title}

The result is cached to data/universe/asset_registry.parquet so subsequent
calls in a session re-read from disk.  Pass force_refresh=True to re-fetch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import ASSET_CATALOG_URL, DATA_DIR

_REQUIRED_COLUMNS = ("artemis_id", "symbol", "coingecko_id", "title")
_TIMEOUT_SECONDS = 30

# Module-level path constant — monkeypatched in tests to redirect to tmp_path
REGISTRY_CACHE_PATH: Path = DATA_DIR / "universe" / "asset_registry.parquet"


def enumerate_assets(
    url: str = ASSET_CATALOG_URL,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch the full Artemis asset catalog and return an artemis_id-keyed DataFrame.

    Parameters
    ----------
    url:
        Catalog endpoint (defaults to ASSET_CATALOG_URL from config).
    force_refresh:
        If True, bypass the on-disk cache and re-fetch from the API.

    Returns
    -------
    pd.DataFrame with columns: artemis_id, symbol, coingecko_id, title.
        coingecko_id is NaN (float) for assets without a CoinGecko mapping.
        artemis_id is unique; the frame is indexed by default integer index.
    """
    cache_path = REGISTRY_CACHE_PATH  # allow monkeypatching

    if not force_refresh and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    resp = requests.get(url, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()

    # The /asset endpoint wraps the list under an "assets" key:
    #   {"assets": [{artemis_id, symbol, coingecko_id, title, ...}, ...]}
    # Fall back to a bare list for forward-compat if the structure ever changes.
    if isinstance(payload, dict):
        raw: list[dict] = payload.get("assets", [])
    else:
        raw = payload

    # Build rows, filling absent/None coingecko_id with NaN
    rows = []
    for item in raw:
        rows.append({
            "artemis_id": item.get("artemis_id"),
            "symbol": item.get("symbol"),
            "coingecko_id": item.get("coingecko_id") or None,  # None stays None → NaN
            "title": item.get("title"),
        })

    df = pd.DataFrame(rows, columns=list(_REQUIRED_COLUMNS))

    # Ensure coingecko_id missing values are NaN (not the string "None")
    df["coingecko_id"] = df["coingecko_id"].where(df["coingecko_id"].notna(), other=float("nan"))

    # Validate uniqueness
    if not df["artemis_id"].is_unique:
        dupes = df.loc[df["artemis_id"].duplicated(keep=False), "artemis_id"].tolist()
        raise ValueError(f"Non-unique artemis_id values in /asset response: {dupes[:10]}")

    # Persist to disk
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
