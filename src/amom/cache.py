"""Content-hashed Parquet cache for provider responses.

The cache key is a hash of the canonical request, so identical requests
hit the cache regardless of argument order, and any change misses cleanly.

Ported verbatim from src/cmom/cache.py; import path updated to amom package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


def cache_key(
    metrics: list[str], symbols: list[str], start: str, end: str, granularity: str
) -> str:
    """Return a short deterministic key for a provider request.

    Metrics and symbols are sorted so argument order does not affect the key.
    The request is JSON-encoded before hashing so a value containing a
    delimiter character cannot collide with a differently-split request.
    """
    canonical = json.dumps(
        [sorted(metrics), sorted(symbols), str(start), str(end), str(granularity)],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def is_cached(key: str) -> bool:
    return _path(key).exists()


def write_frame(key: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final = _path(key)
    tmp = final.with_name(final.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)


def read_frame(key: str) -> pd.DataFrame:
    return pd.read_parquet(_path(key))
