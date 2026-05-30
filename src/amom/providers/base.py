"""Provider interfaces and the shared metric-frame data contract.

Every provider returns a long-format DataFrame with columns:
    date (datetime64), symbol (str), metric (str), value (float)

Ported verbatim from src/cmom/providers/base.py; no import changes required
(this module has no internal imports).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

METRIC_COLUMNS = ["date", "symbol", "metric", "value"]


class ProviderError(RuntimeError):
    """Raised when an external data provider fails or returns bad data."""


def validate_metric_frame(df: pd.DataFrame) -> None:
    """Raise ProviderError unless df satisfies the metric-frame contract."""
    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ProviderError(f"metric frame missing columns: {missing}")


@runtime_checkable
class MarketDataProvider(Protocol):
    """Price and market-cap data."""

    def fetch_market(
        self, symbols: list[str], start: str, end: str, granularity: str = "DAY"
    ) -> pd.DataFrame: ...


@runtime_checkable
class OnChainProvider(Protocol):
    """On-chain activity metrics."""

    def fetch_onchain(
        self, symbols: list[str], start: str, end: str, granularity: str = "DAY"
    ) -> pd.DataFrame: ...
