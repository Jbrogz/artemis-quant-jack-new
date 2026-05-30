"""Tests for the Artemis provider: parse, redact, sentinel-skip."""
from __future__ import annotations

import pytest
import pandas as pd

from amom.providers.artemis import _parse_response, ArtemisProvider
from amom.providers.base import METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANNED_PAYLOAD = {
    "data": {
        "symbols": {
            "btc": {
                "PRICE": [
                    {"date": "2024-01-01", "val": 42_000.0},
                    {"date": "2024-01-02", "val": 43_500.0},
                ]
            }
        }
    }
}

SENTINEL_PAYLOAD = {
    "data": {
        "symbols": {
            "btc": {
                "PRICE": [{"date": "2024-01-01", "val": 42_000.0}],
                "MC": "Metric not available for asset.",  # string sentinel
            }
        }
    }
}

MULTI_SYMBOL_PAYLOAD = {
    "data": {
        "symbols": {
            "btc": {
                "PRICE": [{"date": "2024-01-01", "val": 42_000.0}],
            },
            "eth": {
                "PRICE": [{"date": "2024-01-01", "val": 2_400.0}],
            },
        }
    }
}


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


def test_parse_returns_correct_columns():
    """Parsed frame has exactly the metric-frame columns."""
    df = _parse_response(CANNED_PAYLOAD)
    assert list(df.columns) == METRIC_COLUMNS


def test_parse_date_dtype_is_datetime64_ns():
    """date column must be datetime64[ns] (not object or us)."""
    df = _parse_response(CANNED_PAYLOAD)
    assert df["date"].dtype == "datetime64[ns]"


def test_parse_value_is_numeric():
    """value column must be numeric (float64)."""
    df = _parse_response(CANNED_PAYLOAD)
    assert pd.api.types.is_numeric_dtype(df["value"])


def test_parse_long_format():
    """Parsed frame is long: rows = #symbols × #metrics × #dates."""
    df = _parse_response(CANNED_PAYLOAD)
    assert len(df) == 2  # 1 symbol, 1 metric, 2 dates


def test_parse_correct_values():
    """Values and symbols are preserved correctly."""
    df = _parse_response(CANNED_PAYLOAD)
    row0 = df.iloc[0]
    assert row0["symbol"] == "btc"
    assert row0["metric"] == "PRICE"
    assert row0["value"] == pytest.approx(42_000.0)
    assert row0["date"] == pd.Timestamp("2024-01-01")


def test_parse_string_sentinel_skipped():
    """A string sentinel (e.g. 'Metric not available') must be skipped silently."""
    df = _parse_response(SENTINEL_PAYLOAD)
    # Only PRICE rows — MC sentinel row must not appear
    assert set(df["metric"]) == {"PRICE"}
    assert len(df) == 1


def test_parse_multiple_symbols():
    """Multi-symbol payloads produce one row per (symbol, metric, date)."""
    df = _parse_response(MULTI_SYMBOL_PAYLOAD)
    assert set(df["symbol"]) == {"btc", "eth"}
    assert len(df) == 2


def test_parse_empty_payload():
    """Empty payload returns empty frame with correct columns and date dtype."""
    df = _parse_response({})
    assert list(df.columns) == METRIC_COLUMNS
    assert df["date"].dtype == "datetime64[ns]"
    assert len(df) == 0


# ---------------------------------------------------------------------------
# ArtemisProvider._redact
# ---------------------------------------------------------------------------


def test_redact_removes_api_key():
    """_redact must replace the API key with *** in an error string."""
    provider = ArtemisProvider(api_key="super_secret_key_123")
    error_text = "Request failed: https://api.example.com?APIKey=super_secret_key_123"
    redacted = provider._redact(error_text)
    assert "super_secret_key_123" not in redacted
    assert "***" in redacted


def test_redact_leaves_other_text_intact():
    """_redact must not alter text that does not contain the key."""
    provider = ArtemisProvider(api_key="my_secret_key")
    text = "Something went wrong at the server side."
    assert provider._redact(text) == text


# ---------------------------------------------------------------------------
# ArtemisProvider.__init__ validation
# ---------------------------------------------------------------------------


def test_provider_requires_non_empty_key():
    """Empty API key raises ProviderError."""
    from amom.providers.base import ProviderError

    with pytest.raises(ProviderError):
        ArtemisProvider(api_key="")


# ---------------------------------------------------------------------------
# ArtemisProvider.fetch_market (monkeypatched _request)
# ---------------------------------------------------------------------------


def test_fetch_market_returns_parsed_frame(tmp_path, monkeypatch):
    """fetch_market calls _request and returns a parsed long frame."""
    import amom.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)

    provider = ArtemisProvider(api_key="test_key_xyz")

    def fake_request(metrics, symbols, start, end, granularity):
        return _parse_response(CANNED_PAYLOAD)

    monkeypatch.setattr(provider, "_request", fake_request)

    df = provider.fetch_market(["btc"], "2024-01-01", "2024-01-02")
    assert list(df.columns) == METRIC_COLUMNS
    assert len(df) > 0
    assert df["date"].dtype == "datetime64[ns]"
