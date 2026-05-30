"""Live Artemis API integration tests.

Marked @pytest.mark.integration. Skipped when ARTEMIS_API_KEY is absent.
These tests hit the real Artemis API and verify:
- BTC price history returns > 1000 daily points.
- At least one of the probed dead/collapsed coins (luna, lunc, ftt, ust)
  returns a non-empty price series.

Run explicitly with:
    uv run pytest tests/integration/ -v -m integration
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

INTEGRATION = pytest.mark.integration

_SKIP = pytest.mark.skipif(
    not os.environ.get("ARTEMIS_API_KEY"),
    reason="ARTEMIS_API_KEY not set — skipping live Artemis tests",
)


@INTEGRATION
@_SKIP
def test_btc_price_history_depth() -> None:
    """BTC daily price series must contain > 1000 observations."""
    from amom.providers.artemis import ArtemisProvider

    api_key = os.environ["ARTEMIS_API_KEY"]
    provider = ArtemisProvider(api_key=api_key)

    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    df = provider._get(["PRICE"], ["btc"], "2013-01-01", today)

    btc_price = df[(df["symbol"] == "btc") & (df["metric"] == "PRICE")]
    assert len(btc_price) > 1000, (
        f"Expected > 1000 BTC price points, got {len(btc_price)}"
    )


@INTEGRATION
@_SKIP
def test_dead_coins_return_price_series() -> None:
    """At least one collapsed coin (luna/lunc/ftt/ust) must return a price series."""
    from amom.providers.artemis import ArtemisProvider

    api_key = os.environ["ARTEMIS_API_KEY"]
    provider = ArtemisProvider(api_key=api_key)

    dead_coins = ["luna", "lunc", "ftt", "ust"]
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    df = provider._get(["PRICE"], dead_coins, "2019-01-01", today)

    returned = []
    for coin in dead_coins:
        series = df[(df["symbol"] == coin) & (df["metric"] == "PRICE")]
        if len(series) > 0:
            returned.append(coin)

    assert len(returned) >= 1, (
        f"Expected at least one dead coin with price data; got none from {dead_coins}"
    )


@INTEGRATION
@_SKIP
def test_large_caps_return_multiple_metrics() -> None:
    """Sample large caps must return at least PRICE data (MC and volume may be partial)."""
    from amom.providers.artemis import ArtemisProvider

    api_key = os.environ["ARTEMIS_API_KEY"]
    provider = ArtemisProvider(api_key=api_key)

    large_caps = ["btc", "eth", "bnb", "sol", "xrp"]
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    df = provider._get(["PRICE", "MC", "30D_VOLUME"], large_caps, start, today)

    for coin in large_caps:
        price_rows = df[(df["symbol"] == coin) & (df["metric"] == "PRICE")]
        assert len(price_rows) > 0, f"Expected PRICE data for {coin}, got none"
