"""Tests for universe/registry.py — enumerate_assets() from /asset endpoint.

TDD: tests written first against the public contract; implementation follows.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Canned /asset payload — realistic subset (artemis_id is the stable slug)
# ---------------------------------------------------------------------------
_CANNED_PAYLOAD = [
    {
        "artemis_id": "bitcoin",
        "symbol": "BTC",
        "coingecko_id": "bitcoin",
        "title": "Bitcoin",
    },
    {
        "artemis_id": "ethereum",
        "symbol": "ETH",
        "coingecko_id": "ethereum",
        "title": "Ethereum",
    },
    {
        "artemis_id": "terra-classic",
        "symbol": "LUNC",
        "coingecko_id": None,          # missing coingecko_id — must become NaN, not crash
        "title": "Terra Classic",
    },
    {
        "artemis_id": "optimism",
        "symbol": "OP",
        "coingecko_id": "optimism",
        "title": "Optimism",
    },
    {
        "artemis_id": "for-protocol",
        "symbol": "FOR",
        # coingecko_id key entirely absent — must become NaN
        "title": "For Protocol",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal requests.Response stand-in.

    The live /asset endpoint returns {"assets": [...]}, so we mirror that
    structure here so the mock exercises the same code path.
    """

    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"assets": self._payload}


@pytest.fixture()
def mock_get_canned(monkeypatch):
    """Monkeypatch requests.get to return _CANNED_PAYLOAD without hitting the network."""
    def _fake_get(url, **kwargs):  # noqa: ANN001, ARG001
        return _FakeResponse(_CANNED_PAYLOAD)

    monkeypatch.setattr("requests.get", _fake_get)


# ---------------------------------------------------------------------------
# Contract tests (all run against the monkeypatched HTTP layer)
# ---------------------------------------------------------------------------

def test_enumerate_assets_returns_dataframe(mock_get_canned, tmp_path, monkeypatch):
    """enumerate_assets() returns a DataFrame with the required columns."""
    # Redirect cache writes to tmp_path so tests don't pollute data/
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()
    assert isinstance(df, pd.DataFrame)


def test_enumerate_assets_required_columns(mock_get_canned, tmp_path, monkeypatch):
    """Result has artemis_id, symbol, coingecko_id, title columns."""
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()
    for col in ("artemis_id", "symbol", "coingecko_id", "title"):
        assert col in df.columns, f"missing column: {col}"


def test_enumerate_assets_artemis_id_unique(mock_get_canned, tmp_path, monkeypatch):
    """artemis_id must be unique across the result."""
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()
    assert df["artemis_id"].is_unique, "artemis_id is not unique"


def test_enumerate_assets_row_count(mock_get_canned, tmp_path, monkeypatch):
    """Row count matches the canned payload length."""
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()
    assert len(df) == len(_CANNED_PAYLOAD)


def test_enumerate_assets_missing_coingecko_id_is_nan(mock_get_canned, tmp_path, monkeypatch):
    """A None or absent coingecko_id becomes NaN — not a crash and not 'None' string."""
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()

    # terra-classic has explicit None; for-protocol has the key absent entirely
    for aid in ("terra-classic", "for-protocol"):
        row = df.loc[df["artemis_id"] == aid, "coingecko_id"]
        assert len(row) == 1, f"expected exactly one row for {aid}"
        val = row.iloc[0]
        assert val is None or (isinstance(val, float) and math.isnan(val)), (
            f"{aid} coingecko_id should be NaN, got {val!r}"
        )


def test_enumerate_assets_known_values(mock_get_canned, tmp_path, monkeypatch):
    """Spot-check that known values from the payload are faithfully returned."""
    monkeypatch.setattr(
        "amom.universe.registry.REGISTRY_CACHE_PATH",
        tmp_path / "asset_registry.parquet",
    )
    from amom.universe.registry import enumerate_assets

    df = enumerate_assets()
    btc = df.loc[df["artemis_id"] == "bitcoin"].iloc[0]
    assert btc["symbol"] == "BTC"
    assert btc["coingecko_id"] == "bitcoin"
    assert btc["title"] == "Bitcoin"


def test_config_market_metrics_no_30d_volume():
    """MARKET_METRICS must not contain 30D_VOLUME (real-time-only sentinel)."""
    from amom.config import MARKET_METRICS

    assert "30D_VOLUME" not in MARKET_METRICS, (
        "30D_VOLUME is real-time-only and must be excluded from MARKET_METRICS"
    )


def test_config_market_metrics_has_required():
    """MARKET_METRICS must include PRICE, MC, and 24H_VOLUME."""
    from amom.config import MARKET_METRICS

    for m in ("PRICE", "MC", "24H_VOLUME"):
        assert m in MARKET_METRICS, f"MARKET_METRICS missing required metric: {m}"


def test_config_liquidity_constants_present():
    """All new liquidity/grid config constants must be importable with sane values."""
    from amom.config import (
        ASSET_CATALOG_URL,
        LIQUIDITY_VOL_WINDOW_DAYS,
        LISTING_STALENESS_DAYS,
        MIN_MC_USD,
        MIN_MEDIAN_VOL_USD,
        MIN_OBS_DENSITY,
        UNIVERSE_GRID_FREQ,
    )

    assert ASSET_CATALOG_URL.startswith("https://")
    assert "artemis" in ASSET_CATALOG_URL
    assert MIN_MC_USD > 0
    assert MIN_MEDIAN_VOL_USD > 0
    assert LIQUIDITY_VOL_WINDOW_DAYS > 0
    assert 0 < MIN_OBS_DENSITY < 1
    assert LISTING_STALENESS_DAYS > 0
    assert UNIVERSE_GRID_FREQ == "D"
