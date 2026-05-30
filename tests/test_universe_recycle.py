"""Tests for the recycled-ticker splitter (Task R1, spec §3.6).

A single Artemis ticker is sometimes reused by a *new* project after the
original collapses (e.g. ``ust``). If we leave the series as one asset, a
healthy post-revival run is spliced onto a dead project's crash — fabricating a
recovery that never happened to the original holder and erasing the realized
loss. The splitter detects a **terminal drawdown to near-zero** followed by a
**multi-month gap / new regime** and relabels the post-revival rows as a new
synthetic asset (``sym__seg1``), leaving the dead segment (``sym__seg0``)
ending at its crash.

The detector is pure and point-in-time-safe: the split decision uses only the
realized series (a crash + gap that already happened), never future-relative
information beyond the observed prices themselves.

Every fixture is synthetic and offline; no API calls.
"""

import pandas as pd

from amom.universe.recycle import split_recycled


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _price_rows(symbol: str, start: str, end: str, value: float = 100.0):
    dates = pd.date_range(start, end, freq="D")
    return [{"date": d, "symbol": symbol, "price": float(value)} for d in dates]


def _panel(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", "price"])


# ---------------------------------------------------------------------------
# CENTERPIECE: death (>=90% drawdown to ~0) + gap + revival -> TWO segments
# ---------------------------------------------------------------------------

def test_death_then_gap_then_revival_splits_into_two_segments():
    """A ticker that decays ~99% to near-zero, then after a multi-month gap
    resumes at a brand-new regime, is split into two synthetic assets.

    seg0 holds the dead project ending at its crash; seg1 holds the revival.
    """
    # Original project: price decays from 100 -> ~0.5 (>=90% drawdown to near
    # zero) over the first life, last reporting 2022-05-31.
    life1 = (
        _price_rows("ust", "2022-01-01", "2022-05-01", value=100.0)
        + _price_rows("ust", "2022-05-02", "2022-05-31", value=0.5)
    )
    # GAP: no rows from 2022-06-01 .. 2022-10-14 (> gap_days=45).
    # New project reuses the ticker, resuming at a fresh regime on 2022-10-15.
    life2 = _price_rows("ust", "2022-10-15", "2023-03-31", value=40.0)

    panel = _panel(life1 + life2)

    out = split_recycled(panel)

    symbols = set(out["symbol"].unique())
    assert symbols == {"ust__seg0", "ust__seg1"}, symbols

    seg0 = out.query("symbol == 'ust__seg0'").sort_values("date")
    seg1 = out.query("symbol == 'ust__seg1'").sort_values("date")

    # seg0 (the dead project) ends at the crash, NOT after the revival.
    assert seg0["date"].max() == ts("2022-05-31")
    assert seg0["price"].iloc[-1] == 0.5
    # seg1 (the revival) starts after the gap.
    assert seg1["date"].min() == ts("2022-10-15")
    assert seg1["price"].iloc[0] == 40.0

    # No row is lost or duplicated by the relabel.
    assert len(out) == len(panel)


# ---------------------------------------------------------------------------
# A continuous, healthy series is NOT split (one segment, unchanged)
# ---------------------------------------------------------------------------

def test_continuous_healthy_series_is_one_segment_unchanged():
    """A coin that never crashes to ~0 stays a single asset (no __segN suffix
    proliferation), and its rows are returned unchanged."""
    rows = _price_rows("btc", "2022-01-01", "2023-12-31", value=30000.0)
    panel = _panel(rows)

    out = split_recycled(panel)

    assert set(out["symbol"].unique()) == {"btc"}, "healthy series must not be split"
    # Same dates, same prices.
    merged = out.sort_values("date").reset_index(drop=True)
    base = panel.sort_values("date").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged[["date", "price"]], base[["date", "price"]])


def test_drawdown_below_threshold_with_gap_is_not_split():
    """A series that drops only ~50% (below drawdown_thresh) and then gaps is
    NOT a recycled-ticker death; it stays a single asset."""
    life1 = _price_rows("alt", "2022-01-01", "2022-05-31", value=100.0)
    # only 50% drawdown -> not a terminal-collapse-to-near-zero
    dip = _price_rows("alt", "2022-05-01", "2022-05-31", value=50.0)
    # gap then resume
    life2 = _price_rows("alt", "2022-10-15", "2023-03-31", value=60.0)
    panel = _panel(life1[: -31] + dip + life2)

    out = split_recycled(panel)
    assert set(out["symbol"].unique()) == {"alt"}, "shallow drawdown must not split"


# ---------------------------------------------------------------------------
# Crash with NO revival -> ONE segment ending at the crash (not split)
# ---------------------------------------------------------------------------

def test_crash_with_no_revival_is_single_segment_ending_at_crash():
    """A coin that collapses >=90% to ~0 and then simply stops reporting (no
    later resumption) is a genuine death, NOT a recycle: it stays ONE asset and
    keeps its crash as the final value (survivorship: the crash is preserved)."""
    pre = _price_rows("dead", "2022-01-01", "2022-05-01", value=100.0)
    crash = _price_rows("dead", "2022-05-02", "2022-05-31", value=0.3)  # ~99.7% down
    panel = _panel(pre + crash)

    out = split_recycled(panel)

    assert set(out["symbol"].unique()) == {"dead"}, "no revival -> no split"
    dead = out.sort_values("date")
    assert dead["date"].max() == ts("2022-05-31")
    assert dead["price"].iloc[-1] == 0.3  # crash carried, not dropped


def test_crash_then_short_gap_within_threshold_is_not_split():
    """Death-to-~0 followed by a resume after a gap SHORTER than gap_days is
    treated as continuation (e.g. a brief data outage), not a recycle."""
    life1 = (
        _price_rows("x", "2022-01-01", "2022-05-01", value=100.0)
        + _price_rows("x", "2022-05-02", "2022-05-31", value=0.4)
    )
    # 10-day gap (< gap_days=45) then resume at a new regime.
    life2 = _price_rows("x", "2022-06-11", "2022-09-30", value=20.0)
    panel = _panel(life1 + life2)

    out = split_recycled(panel)
    assert set(out["symbol"].unique()) == {"x"}, "short gap must not split"


# ---------------------------------------------------------------------------
# Point-in-time safety + multi-symbol behavior
# ---------------------------------------------------------------------------

def test_multiple_symbols_only_recycled_one_is_split():
    """A panel with a healthy coin and a recycled coin splits only the latter;
    the healthy coin is untouched."""
    healthy = _price_rows("eth", "2022-01-01", "2023-03-31", value=2000.0)
    life1 = (
        _price_rows("ust", "2022-01-01", "2022-05-01", value=100.0)
        + _price_rows("ust", "2022-05-02", "2022-05-31", value=0.5)
    )
    life2 = _price_rows("ust", "2022-10-15", "2023-03-31", value=40.0)
    panel = _panel(healthy + life1 + life2)

    out = split_recycled(panel)

    assert set(out["symbol"].unique()) == {"eth", "ust__seg0", "ust__seg1"}
    assert len(out) == len(panel)


def test_split_is_point_in_time_only_uses_realized_series():
    """Appending FUTURE rows to seg1 (after the split has occurred) must not
    move where seg0 ends — the dead segment's terminal date is determined only
    by the realized crash + gap, never relabeled by later data."""
    life1 = (
        _price_rows("ust", "2022-01-01", "2022-05-01", value=100.0)
        + _price_rows("ust", "2022-05-02", "2022-05-31", value=0.5)
    )
    life2 = _price_rows("ust", "2022-10-15", "2023-03-31", value=40.0)
    base = _panel(life1 + life2)
    base_out = split_recycled(base)
    base_seg0_end = base_out.query("symbol == 'ust__seg0'")["date"].max()

    # Extend the revival further into the future.
    extra = _price_rows("ust", "2023-04-01", "2024-01-01", value=55.0)
    extended = _panel(life1 + life2 + extra)
    ext_out = split_recycled(extended)
    ext_seg0_end = ext_out.query("symbol == 'ust__seg0'")["date"].max()

    assert ext_seg0_end == base_seg0_end == ts("2022-05-31")
