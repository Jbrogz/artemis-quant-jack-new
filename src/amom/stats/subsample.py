"""Stage-2.6 subsample sign stability + deployment gate (spec §2.6 / guide §2.6).

A valid factor **holds its sign** across subsamples: split the per-rebalance
factor-return series into halves and thirds and require every subsample's mean to
carry the same sign as the full-sample mean. A sign-flip across any subsample
**disqualifies the variant from deployment**, regardless of its full-sample
t-stat — a return that is positive only in one regime is regime exposure, not a
factor (spec §2.6).

``sign_stability`` is pure (no I/O); the deployment gate is the boolean
``holds_sign``, which the Stage-3 deployment selection reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sign(x: float) -> int:
    """Sign of a mean: +1 if > 0, -1 if < 0, 0 if exactly zero / non-finite."""
    if not np.isfinite(x) or x == 0.0:
        return 0
    return 1 if x > 0.0 else -1


def _contiguous_splits(r: np.ndarray, k: int) -> list[np.ndarray]:
    """Split ``r`` into ``k`` contiguous, near-equal time-ordered chunks."""
    return [chunk for chunk in np.array_split(r, k) if chunk.size > 0]


def sign_stability(returns: pd.Series) -> dict:
    """Subsample sign stability of a factor-return series (spec §2.6).

    Splits the (NaN-dropped, time-ordered) series into 2 halves and 3 thirds and
    reports each subsample's mean sign. ``holds_sign`` is True iff there are
    enough observations to form all subsamples (>= 3) AND every half- and
    third-mean shares the full-sample mean's sign. A sign-flip in any subsample
    makes ``holds_sign`` False — the deployment-disqualifying condition.

    Args:
        returns: per-rebalance factor returns (NaNs dropped before splitting).

    Returns:
        Dict with:
          ``full_sign``   sign of the full-sample mean (+1 / -1 / 0),
          ``half_signs``  list[int] signs of the 2 half-sample means,
          ``third_signs`` list[int] signs of the 3 third-sample means,
          ``holds_sign``  bool deployment gate (all subsamples match full sign),
          ``n``           number of finite observations used.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = int(r.size)
    full_sign = _sign(float(r.mean())) if n >= 1 else 0

    # Need at least 3 obs to populate all three thirds; below that, stability
    # cannot be established, so the variant does not hold its sign (inconclusive
    # is treated as "does not qualify" for the deployment gate).
    if n < 3:
        return {
            "full_sign": full_sign,
            "half_signs": [_sign(float(c.mean())) for c in _contiguous_splits(r, 2)],
            "third_signs": [_sign(float(c.mean())) for c in _contiguous_splits(r, 3)],
            "holds_sign": False,
            "n": n,
        }

    half_signs = [_sign(float(c.mean())) for c in _contiguous_splits(r, 2)]
    third_signs = [_sign(float(c.mean())) for c in _contiguous_splits(r, 3)]

    holds_sign = (
        full_sign != 0
        and all(s == full_sign for s in half_signs)
        and all(s == full_sign for s in third_signs)
    )

    return {
        "full_sign": full_sign,
        "half_signs": half_signs,
        "third_signs": third_signs,
        "holds_sign": bool(holds_sign),
        "n": n,
    }
