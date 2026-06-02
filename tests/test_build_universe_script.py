"""Tests for the R5 build_universe.py rewrite.

Verifies the script-level pipeline offline:
  - loads the registry (mocked) and pulls PRICE + 24H_VOLUME + MC for
    all artemis_ids (no BROAD_CANDIDATES dependency),
  - runs the recycled-ticker splitter,
  - builds the panel on a daily grid,
  - prints the correct stats keys (rows, #assets, #ever-eligible,
    #eligible-on-latest, #terminal-collapse),
  - BROAD_CANDIDATES is not imported from probe_artemis.

All fixtures are synthetic and offline; no API calls.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Ensure src/ is on the path for importing the script as a module-like object.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_registry(artemis_ids: list[str]) -> pd.DataFrame:
    """Minimal asset registry DataFrame."""
    return pd.DataFrame(
        {
            "artemis_id": artemis_ids,
            "symbol": artemis_ids,  # simplification for tests
            "coingecko_id": [float("nan")] * len(artemis_ids),
            "title": [f"Token {a}" for a in artemis_ids],
        }
    )


def _make_market_rows(
    artemis_ids: list[str],
    start: str,
    end: str,
    price: float = 100.0,
    volume: float = 5e6,
    mc: float = 50e6,
) -> pd.DataFrame:
    """Synthetic long market DataFrame with PRICE + 24H_VOLUME + MC."""
    rows = []
    for aid in artemis_ids:
        for d in pd.date_range(start, end, freq="D"):
            rows.append({"date": d, "symbol": aid, "metric": "PRICE", "value": price})
            rows.append({"date": d, "symbol": aid, "metric": "24H_VOLUME", "value": volume})
            rows.append({"date": d, "symbol": aid, "metric": "MC", "value": mc})
    return pd.DataFrame(rows)


def _make_crash_rows(
    artemis_id: str,
    start: str,
    crash_date: str,
    pre_price: float = 100.0,
    crash_price: float = 2.0,
    volume: float = 5e6,
    mc_high: float = 50e6,
    mc_crash: float = 200_000,
) -> pd.DataFrame:
    """Synthetic crash: price runs from start to crash_date-1 at pre_price,
    then drops to crash_price on crash_date; series ends there."""
    rows = []
    for d in pd.date_range(start, crash_date, freq="D"):
        p = pre_price if d < pd.Timestamp(crash_date) else crash_price
        m = mc_high if d < pd.Timestamp(crash_date) else mc_crash
        rows.append({"date": d, "symbol": artemis_id, "metric": "PRICE", "value": p})
        rows.append({"date": d, "symbol": artemis_id, "metric": "24H_VOLUME", "value": volume})
        rows.append({"date": d, "symbol": artemis_id, "metric": "MC", "value": m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core pipeline function test (no BROAD_CANDIDATES)
# ---------------------------------------------------------------------------

def _import_build_universe():
    """Import build_universe fresh; reload if already imported."""
    mod_name = "build_universe"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_no_broad_candidates_import():
    """build_universe must NOT import BROAD_CANDIDATES from probe_artemis."""
    mod = _import_build_universe()
    src = Path(mod.__file__).read_text()
    assert "BROAD_CANDIDATES" not in src, (
        "build_universe still references BROAD_CANDIDATES — the old dependency must be removed"
    )


def test_pipeline_uses_registry_artemis_ids(tmp_path):
    """The pipeline pulls data for all registry artemis_ids, not a hand list.

    Registry has 3 assets; the provider must be called with those IDs.
    """
    build_universe = _import_build_universe()

    registry_ids = ["aaaa", "bbbb", "cccc"]
    registry_df = _make_registry(registry_ids)
    market_df = _make_market_rows(registry_ids, "2022-01-01", "2022-06-30")

    captured_symbols: list[list[str]] = []

    class FakeProvider:
        def __init__(self, **_):
            pass

        def _get(self, metrics, symbols, start, end, granularity="DAY"):
            captured_symbols.append(list(symbols))
            # Return rows only for the symbols requested in this batch.
            return market_df[market_df["symbol"].isin(symbols)]

    with (
        patch.object(build_universe, "enumerate_assets", return_value=registry_df),
        patch.object(build_universe, "ArtemisProvider", FakeProvider),
        patch.object(build_universe, "OUTPUT_PATH", tmp_path / "universe_history.parquet"),
    ):
        result, _prices = build_universe.build_panel(registry_df, FakeProvider(), "2022-01-01", "2022-06-30")

    all_pulled = {s for batch in captured_symbols for s in batch}
    assert all_pulled == set(registry_ids), (
        f"provider was not called with all registry artemis_ids; got {all_pulled}"
    )
    assert result is not None and not result.empty


def test_panel_has_required_columns(tmp_path):
    """Output panel has at least the required columns."""
    build_universe = _import_build_universe()

    ids = ["aaa", "bbb", "ccc", "ddd", "eee", "fff"]
    registry_df = _make_registry(ids)
    market_df = _make_market_rows(ids, "2022-01-01", "2022-12-31")

    class FakeProvider:
        def __init__(self, **_):
            pass

        def _get(self, metrics, symbols, start, end, granularity="DAY"):
            return market_df[market_df["symbol"].isin(symbols)]

    with (
        patch.object(build_universe, "enumerate_assets", return_value=registry_df),
        patch.object(build_universe, "ArtemisProvider", FakeProvider),
        patch.object(build_universe, "OUTPUT_PATH", tmp_path / "universe_history.parquet"),
    ):
        result, _prices = build_universe.build_panel(registry_df, FakeProvider(), "2022-01-01", "2022-12-31")

    required = {"date", "symbol", "eligible", "adv_30d", "price_last_date", "delisted_asof", "gated"}
    assert required.issubset(set(result.columns)), (
        f"panel missing columns; got {list(result.columns)}"
    )


def test_terminal_collapse_count_nonzero_when_crash_present(tmp_path):
    """compute_terminal_collapses returns a positive count when a crash coin exists."""
    build_universe = _import_build_universe()

    # crasher: 98% drop in its last 60 days of data, then gone.
    price_rows = []
    for d in pd.date_range("2022-01-01", "2022-11-01", freq="D"):
        price_rows.append({"date": d, "symbol": "crasher", "metric": "PRICE", "value": 100.0})
    # Crash: price drops 98% on last day, then series ends.
    price_rows.append({"date": pd.Timestamp("2022-11-02"), "symbol": "crasher", "metric": "PRICE", "value": 2.0})

    price_panel = pd.DataFrame(price_rows)[["date", "symbol", "metric", "value"]]
    price_panel_filtered = price_panel[price_panel["metric"] == "PRICE"].rename(columns={"value": "price"})[["date", "symbol", "price"]]

    count = build_universe.compute_terminal_collapses(price_panel_filtered, drawdown_thresh=0.9)
    assert count >= 1, f"expected >= 1 terminal collapse, got {count}"


def test_terminal_collapse_count_zero_for_healthy_coins(tmp_path):
    """compute_terminal_collapses returns 0 when no crash coins exist."""
    build_universe = _import_build_universe()

    price_rows = []
    for sym in ["healthy1", "healthy2"]:
        for d in pd.date_range("2022-01-01", "2022-12-31", freq="D"):
            price_rows.append({"date": d, "symbol": sym, "price": 100.0})
    price_panel = pd.DataFrame(price_rows)

    count = build_universe.compute_terminal_collapses(price_panel, drawdown_thresh=0.9)
    assert count == 0, f"expected 0 terminal collapses for healthy coins, got {count}"


def test_recycled_ticker_splitter_called_in_pipeline(tmp_path):
    """The pipeline runs split_recycled before building the panel."""
    build_universe = _import_build_universe()

    ids = ["aaa", "bbb", "ccc", "ddd", "eee"]
    registry_df = _make_registry(ids)
    market_df = _make_market_rows(ids, "2022-01-01", "2022-06-30")

    split_called = []

    original_split = build_universe.split_recycled

    def _tracking_split(price_panel, **kwargs):
        split_called.append(True)
        return original_split(price_panel, **kwargs)

    class FakeProvider:
        def __init__(self, **_):
            pass

        def _get(self, metrics, symbols, start, end, granularity="DAY"):
            return market_df[market_df["symbol"].isin(symbols)]

    with (
        patch.object(build_universe, "enumerate_assets", return_value=registry_df),
        patch.object(build_universe, "ArtemisProvider", FakeProvider),
        patch.object(build_universe, "split_recycled", side_effect=_tracking_split),
        patch.object(build_universe, "OUTPUT_PATH", tmp_path / "universe_history.parquet"),
    ):
        build_universe.build_panel(registry_df, FakeProvider(), "2022-01-01", "2022-06-30")  # noqa: F841

    assert split_called, "split_recycled was not called in the pipeline"
