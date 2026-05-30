"""Tests for Stage-2.6 subsample sign stability + the OOS in_sample guard.

Task T4 (plan §T4, spec §2.6 / §2.8). Two units under test:

* ``subsample.sign_stability(returns)`` -- split the series into halves and
  thirds and report each subsample's mean sign; ``holds_sign`` is True iff every
  subsample's mean has the same sign as the full-sample mean. A sign-flip across
  any subsample makes ``holds_sign`` False, which **disqualifies the variant from
  deployment** (spec §2.6): a single-regime return is regime exposure, not a
  factor.

* ``config.in_sample(df)`` -- returns only the rows with
  ``rebalance_date < OOS_START``; the OOS window (``>= OOS_START``) is sealed for
  Stage 4 and no Stage-2 path may consume it (spec §2.8).
"""

import numpy as np
import pandas as pd

from amom.config import OOS_START, in_sample
from amom.stats.subsample import sign_stability


# ---------------------------------------------------------------------------
# sign_stability: halves + thirds, sign-flip disqualifies
# ---------------------------------------------------------------------------

def test_consistently_positive_series_holds_sign():
    # Every observation positive -> every half and third is positive -> holds.
    r = pd.Series([0.01, 0.02, 0.015, 0.03, 0.012, 0.02])
    out = sign_stability(r)
    assert out["full_sign"] == 1
    assert out["holds_sign"] is True
    assert all(s == 1 for s in out["half_signs"])
    assert all(s == 1 for s in out["third_signs"])


def test_consistently_negative_series_holds_sign():
    r = pd.Series([-0.01, -0.02, -0.015, -0.03, -0.012, -0.02])
    out = sign_stability(r)
    assert out["full_sign"] == -1
    assert out["holds_sign"] is True


def test_sign_flip_in_a_half_disqualifies():
    # Positive full-sample mean, but the first half is net negative -> flips.
    # first half = [-0.10,-0.10,-0.10] (mean<0); second half = [0.30,0.20,0.20]
    r = pd.Series([-0.10, -0.10, -0.10, 0.30, 0.20, 0.20])
    assert r.mean() > 0  # full sample is positive
    out = sign_stability(r)
    assert out["full_sign"] == 1
    assert out["holds_sign"] is False  # a half flipped -> disqualified


def test_sign_flip_in_a_third_disqualifies():
    # Full mean positive; the middle third is net negative -> flips on thirds.
    r = pd.Series([0.20, 0.20, -0.30, -0.30, 0.25, 0.25])
    assert r.mean() > 0
    out = sign_stability(r)
    # the halves may hold but a third flips -> overall disqualified
    assert out["holds_sign"] is False
    assert -1 in out["third_signs"]


def test_signs_reported_for_each_subsample():
    r = pd.Series(np.arange(1, 13, dtype=float) / 100.0)  # 12 positive obs
    out = sign_stability(r)
    assert len(out["half_signs"]) == 2
    assert len(out["third_signs"]) == 3


def test_drops_nans_before_splitting():
    r = pd.Series([0.01, np.nan, 0.02, 0.015, np.nan, 0.03])
    out = sign_stability(r)
    assert out["n"] == 4  # NaNs dropped
    assert out["holds_sign"] is True


def test_too_few_obs_returns_inconclusive_not_holds():
    # Fewer than 3 obs cannot be split into thirds -> holds_sign is False
    # (cannot establish stability), with the full sign still reported.
    out = sign_stability(pd.Series([0.01, 0.02]))
    assert out["holds_sign"] is False


# ---------------------------------------------------------------------------
# in_sample: excludes rows >= OOS_START (spec §2.8 -- OOS sealed for Stage 4)
# ---------------------------------------------------------------------------

def test_oos_start_is_a_frozen_timestamp():
    # The split is a literal sealed constant, not recomputed at runtime.
    assert isinstance(OOS_START, pd.Timestamp)


def test_in_sample_excludes_oos_rows():
    df = pd.DataFrame(
        {
            "rebalance_date": [
                OOS_START - pd.Timedelta(days=60),
                OOS_START - pd.Timedelta(days=1),
                OOS_START,                          # exactly the boundary -> OOS
                OOS_START + pd.Timedelta(days=30),  # OOS
            ],
            "factor_return": [0.01, 0.02, 0.03, 0.04],
        }
    )
    ins = in_sample(df)
    assert len(ins) == 2  # only the two strictly-before-OOS rows
    assert (ins["rebalance_date"] < OOS_START).all()
    # the boundary date itself is sealed (>= OOS_START is OOS)
    assert OOS_START not in set(ins["rebalance_date"])


def test_in_sample_is_a_copy_not_a_view():
    df = pd.DataFrame(
        {"rebalance_date": [OOS_START - pd.Timedelta(days=1)], "factor_return": [0.5]}
    )
    out = in_sample(df)
    out.loc[out.index[0], "factor_return"] = 999.0
    assert df.loc[0, "factor_return"] == 0.5  # original untouched (immutability)
