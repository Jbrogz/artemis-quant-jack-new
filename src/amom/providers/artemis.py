"""Artemis REST API client.

Endpoint: GET {base}/{metricNames}/?symbols=&startDate=&endDate=&granularity=&APIKey=
Response: {"data": {"symbols": {"<sym>": {"<METRIC>": [{"date","val"}]}}}}

Ported from src/cmom/providers/artemis.py; imports updated to amom package.
On-chain and dev-metric methods are not included (out of scope for momentum).
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import requests

from ..cache import cache_key, is_cached, read_frame, write_frame
from ..config import ARTEMIS_BASE_URL, MARKET_METRICS
from .base import METRIC_COLUMNS, ProviderError

_MAX_SYMBOLS_PER_CALL = 10
_MAX_DAYS_PER_CALL = 900
_TIMEOUT_SECONDS = 30


def _chunks(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _date_windows(start: str, end: str, max_days: int) -> list[tuple[str, str]]:
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    windows: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        win_end = min(cur + timedelta(days=max_days - 1), e)
        windows.append((cur.isoformat(), win_end.isoformat()))
        cur = win_end + timedelta(days=1)
    return windows


def _parse_response(payload: dict) -> pd.DataFrame:
    rows = []
    data = payload.get("data") or {}
    symbols = data.get("symbols") or {}
    for symbol, metrics in symbols.items():
        for metric, points in (metrics or {}).items():
            if not isinstance(points, list):
                # Artemis returns a string sentinel (e.g. "Metric not
                # available for asset.") instead of a point list when a
                # metric is unavailable for a coin; skip it.
                continue
            for pt in points:
                rows.append({
                    "date": pt.get("date"),
                    "symbol": symbol,
                    "metric": metric,
                    "value": pt.get("val"),
                })
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    # Pin to ns resolution: pandas 3.x infers `us` from string dates, but the
    # metric-frame contract and downstream joins expect `ns`. Coerce
    # unconditionally so empty frames also satisfy the contract (an object
    # `date` would collapse a later concat with a populated frame).
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


class ArtemisProvider:
    """Implements MarketDataProvider against the Artemis API."""

    def __init__(self, api_key: str, base_url: str = ARTEMIS_BASE_URL):
        if not api_key:
            raise ProviderError("Artemis API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _redact(self, text: object) -> str:
        """Strip the API key from text before it surfaces in errors or logs.

        The key travels as a URL query param, so a `RequestException` string
        embeds the full URL (incl. `APIKey=...`). Redact it everywhere an
        external message is built.
        """
        return str(text).replace(self._api_key, "***")

    def _request(
        self, metrics: list[str], symbols: list[str],
        start: str, end: str, granularity: str,
    ) -> pd.DataFrame:
        url = f"{self._base_url}/{','.join(metrics)}/"
        params = {
            "symbols": ",".join(symbols),
            "startDate": start,
            "endDate": end,
            "granularity": granularity,
            "APIKey": self._api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise ProviderError(
                f"Artemis request failed: {self._redact(exc)}"
            ) from exc
        if resp.status_code != 200:
            raise ProviderError(
                f"Artemis HTTP {resp.status_code}: {self._redact(resp.text[:200])}"
            )
        return _parse_response(resp.json())

    def _get(
        self, metrics: list[str], symbols: list[str],
        start: str, end: str, granularity: str = "DAY",
    ) -> pd.DataFrame:
        frames = []
        for sym_batch in _chunks(list(symbols), _MAX_SYMBOLS_PER_CALL):
            for win_start, win_end in _date_windows(start, end, _MAX_DAYS_PER_CALL):
                frames.append(
                    self._request(metrics, sym_batch, win_start, win_end, granularity)
                )
        if not frames:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["date", "symbol", "metric"])
            .reset_index(drop=True)
        )

    def _cached_get(
        self, metrics: tuple[str, ...], symbols: list[str],
        start: str, end: str, granularity: str,
    ) -> pd.DataFrame:
        key = cache_key(list(metrics), list(symbols), start, end, granularity)
        if is_cached(key):
            return read_frame(key)
        df = self._get(list(metrics), list(symbols), start, end, granularity)
        write_frame(key, df)
        return df

    def fetch_market(
        self, symbols: list[str], start: str, end: str, granularity: str = "DAY"
    ) -> pd.DataFrame:
        return self._cached_get(MARKET_METRICS, symbols, start, end, granularity)
