"""Tests for the parquet cache (round-trip, is_cached, unknown key)."""
import pandas as pd

from amom.cache import cache_key, is_cached, read_frame, write_frame


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).astype("datetime64[ns]"),
            "symbol": ["btc", "btc"],
            "metric": ["PRICE", "PRICE"],
            "value": [42_000.0, 43_000.0],
        }
    )


def test_round_trip(tmp_path, monkeypatch):
    """write_frame then read_frame returns an equal DataFrame."""
    import amom.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)

    df = _sample_df()
    key = cache_key(["PRICE"], ["btc"], "2024-01-01", "2024-01-02", "DAY")
    write_frame(key, df)
    result = read_frame(key)

    pd.testing.assert_frame_equal(df.reset_index(drop=True), result.reset_index(drop=True))


def test_is_cached_true_after_write(tmp_path, monkeypatch):
    """is_cached returns True for a key that was written."""
    import amom.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)

    key = cache_key(["PRICE"], ["btc"], "2024-01-01", "2024-01-02", "DAY")
    assert not is_cached(key), "should be False before write"
    write_frame(key, _sample_df())
    assert is_cached(key), "should be True after write"


def test_is_cached_false_for_unknown_key(tmp_path, monkeypatch):
    """is_cached returns False for a key that was never written."""
    import amom.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)

    key = cache_key(["MC"], ["eth"], "2020-01-01", "2020-12-31", "DAY")
    assert not is_cached(key)


def test_cache_key_is_order_independent():
    """cache_key must not depend on argument order (metrics and symbols are sorted)."""
    key1 = cache_key(["MC", "PRICE"], ["btc", "eth"], "2024-01-01", "2024-12-31", "DAY")
    key2 = cache_key(["PRICE", "MC"], ["eth", "btc"], "2024-01-01", "2024-12-31", "DAY")
    assert key1 == key2


def test_cache_key_differs_on_different_args():
    """Distinct requests produce distinct keys."""
    key1 = cache_key(["PRICE"], ["btc"], "2024-01-01", "2024-12-31", "DAY")
    key2 = cache_key(["PRICE"], ["eth"], "2024-01-01", "2024-12-31", "DAY")
    assert key1 != key2
