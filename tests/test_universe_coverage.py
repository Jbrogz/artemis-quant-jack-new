"""Tests for listing-date (first-seen) reconstruction — Task 3.

Fixtures are fully synthetic and offline; no API calls.
"""

import numpy as np
import pandas as pd
import pytest

from amom.universe.coverage import first_seen_dates


def _make_long(symbol_ranges: dict) -> pd.DataFrame:
    """Build a long price frame with one row per (date, symbol).

    Args:
        symbol_ranges: mapping of symbol -> (start, end, price_or_none_mask).
            Each value is either:
              (start_str, end_str)  -> daily prices 1.0, NaN-free
              (start_str, end_str, nan_dates)  -> prices 1.0, NaN on nan_dates
    """
    rows = []
    for sym, spec in symbol_ranges.items():
        start, end = spec[0], spec[1]
        nan_dates = set(spec[2]) if len(spec) > 2 else set()
        dates = pd.date_range(start, end, freq="D")
        for d in dates:
            price = np.nan if d in nan_dates else 1.0
            rows.append({"date": d, "symbol": sym, "price": price})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: basic first/last dates and n_obs for two coins of different ages
# ---------------------------------------------------------------------------

class TestFirstSeenDates:
    def test_btc_starts_earlier_than_newcoin(self):
        df = _make_long({
            "btc": ("2020-01-01", "2026-01-01"),
            "newcoin": ("2025-01-01", "2026-01-01"),
        })
        result = first_seen_dates(df)

        btc = result.set_index("symbol").loc["btc"]
        newcoin = result.set_index("symbol").loc["newcoin"]

        assert btc["price_first_date"] == pd.Timestamp("2020-01-01")
        assert newcoin["price_first_date"] == pd.Timestamp("2025-01-01")

    def test_price_last_date(self):
        df = _make_long({
            "btc": ("2020-01-01", "2026-01-01"),
            "newcoin": ("2025-01-01", "2026-01-01"),
        })
        result = first_seen_dates(df).set_index("symbol")

        assert result.loc["btc", "price_last_date"] == pd.Timestamp("2026-01-01")
        assert result.loc["newcoin", "price_last_date"] == pd.Timestamp("2026-01-01")

    def test_n_obs_excludes_nan_rows(self):
        """n_obs must count only non-NaN price observations."""
        nan_dates = {pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-02")}
        df = _make_long({
            "gappy": ("2025-01-01", "2025-12-31", list(nan_dates)),
        })
        result = first_seen_dates(df).set_index("symbol")

        total_days = len(pd.date_range("2025-01-01", "2025-12-31", freq="D"))
        assert result.loc["gappy", "n_obs"] == total_days - len(nan_dates)

    def test_internal_nan_gap_does_not_shift_first_or_last_date(self):
        """An internal NaN gap should NOT change first/last — only boundary NaNs matter."""
        nan_dates = [
            pd.Timestamp("2025-03-01"),
            pd.Timestamp("2025-03-02"),
            pd.Timestamp("2025-03-03"),
        ]
        df = _make_long({
            "gappy": ("2025-01-01", "2025-12-31", nan_dates),
        })
        result = first_seen_dates(df).set_index("symbol")

        # Internal NaN gap — first/last should be the boundary dates
        assert result.loc["gappy", "price_first_date"] == pd.Timestamp("2025-01-01")
        assert result.loc["gappy", "price_last_date"] == pd.Timestamp("2025-12-31")

    def test_returns_required_columns(self):
        df = _make_long({"btc": ("2020-01-01", "2020-12-31")})
        result = first_seen_dates(df)
        assert set(result.columns) == {"symbol", "price_first_date", "price_last_date", "n_obs"}

    def test_one_row_per_symbol(self):
        df = _make_long({
            "btc": ("2020-01-01", "2026-01-01"),
            "eth": ("2021-01-01", "2026-01-01"),
            "newcoin": ("2025-01-01", "2026-01-01"),
        })
        result = first_seen_dates(df)
        assert result["symbol"].nunique() == 3
        assert len(result) == 3

    def test_all_nan_symbol_excluded(self):
        """A symbol with no non-NaN prices should not appear in the result."""
        rows = [
            {"date": pd.Timestamp("2025-01-01"), "symbol": "ghost", "price": np.nan},
            {"date": pd.Timestamp("2025-01-02"), "symbol": "ghost", "price": np.nan},
            {"date": pd.Timestamp("2025-01-01"), "symbol": "btc", "price": 1.0},
        ]
        df = pd.DataFrame(rows)
        result = first_seen_dates(df)
        assert "ghost" not in result["symbol"].values
        assert "btc" in result["symbol"].values

    def test_n_obs_correct_for_btc_and_newcoin(self):
        df = _make_long({
            "btc": ("2020-01-01", "2026-01-01"),
            "newcoin": ("2025-01-01", "2026-01-01"),
        })
        result = first_seen_dates(df).set_index("symbol")

        btc_days = len(pd.date_range("2020-01-01", "2026-01-01", freq="D"))
        newcoin_days = len(pd.date_range("2025-01-01", "2026-01-01", freq="D"))

        assert result.loc["btc", "n_obs"] == btc_days
        assert result.loc["newcoin", "n_obs"] == newcoin_days
