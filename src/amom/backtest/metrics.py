"""Net-of-cost performance metrics + capacity estimate (Task B3, spec §4.5).

``performance(equity, trades, returns)`` reduces an engine run (the three frames
``run_backtest`` emits) to the §4.5 metric set, all **net of costs**:

  * total & annualized return, annualized vol,
  * Sharpe, Sortino, max drawdown, Calmar,
  * hit rate, avg win vs avg loss,
  * **annualized turnover** (from the trade log).

Sharpe, max drawdown, and Calmar are delegated to ``amom.stats.core`` so the
backtest reports the *same* numerics the Stage-2 battery used — no re-derivation.
Sortino is the only ratio added here (downside-deviation denominator); it is not
in ``stats.core``.

``capacity(candidate, cost_model)`` answers the §4.5 capacity question: as the
book grows, the size-scaled slippage (∝ order/ADV) eats more of the gross expected
return per dollar, so the *net* expected return falls monotonically with AUM. The
capacity ceiling is the AUM at which that net expected return crosses zero —
found by a monotone bisection over a per-coin order/ADV recomputation. A
frictionless cost model never crosses zero, so capacity is ``+inf`` (unbounded).

Pure; no I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from amom.config import HOLDING_DAYS
from amom.stats import core

# Annualization default: the per-rebalance return series has ~365 / HOLDING_DAYS
# periods per year (matches the engine's convention). Callers (the B4 runner)
# pass the same constant explicitly; the default keeps the function self-contained.
_PERIODS_PER_YEAR = 365.0 / HOLDING_DAYS


def _sortino(returns: pd.Series, periods_per_year: float) -> float:
    """Annualized Sortino ratio: annual mean / annualized downside deviation.

    Downside deviation is the root-mean-square of the *negative* returns (zero
    counts as no downside), so only losing periods inflate the denominator. A
    series with no losers has zero downside deviation -> NaN (undefined).
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0.0]
    if downside.empty:
        return float("nan")
    dd_period = math.sqrt(float((downside ** 2).mean()))
    if dd_period <= 0.0:
        return float("nan")
    dd_annual = dd_period * math.sqrt(periods_per_year)
    return float(r.mean() * periods_per_year / dd_annual)


def _annual_turnover(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    n_periods: int,
    periods_per_year: float,
) -> float:
    """Annualized one-way turnover = Σ|traded notional| / book / period * ppy.

    Turnover is read straight from the trade log: the total traded notional over
    the run, normalized by the book size (the opening equity) and the number of
    realized periods, then annualized. With no priced periods turnover is 0.0.
    """
    if trades.empty or n_periods <= 0 or equity.empty:
        return 0.0
    book = float(equity["equity"].iloc[0])
    if book <= 0.0:
        return 0.0
    total_traded = float(trades["traded_notional"].abs().sum())
    return total_traded / book / n_periods * periods_per_year


def performance(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    returns: pd.Series,
    *,
    periods_per_year: float = _PERIODS_PER_YEAR,
) -> dict:
    """Net-of-cost performance summary for a backtest run (spec §4.5).

    Args:
        equity: the engine's equity frame ``[date, equity, gross_return,
            net_return, cost]`` — the opening row plus one per priced window.
        trades: the engine's trade log ``[..., traded_notional, cost]`` — drives
            the annualized turnover.
        returns: the per-period **net** return series (the engine's ``net_return``
            column, opening row excluded).
        periods_per_year: annualization factor (default 365 / HOLDING_DAYS).

    Returns:
        Dict of metrics; ratios that are undefined on the input are NaN.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)

    total_return = (
        float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0)
        if len(equity) >= 2 and equity["equity"].iloc[0] != 0.0
        else float("nan")
    )
    annual_return = float(r.mean() * periods_per_year) if n >= 1 else float("nan")
    annual_vol = (
        float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n >= 2 else float("nan")
    )

    cum = (1 + r).cumprod() - 1
    winners = r[r > 0.0]
    losers = r[r < 0.0]

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": core.sharpe_ratio(r, periods_per_year),
        "sortino": _sortino(r, periods_per_year),
        "max_drawdown": core.max_drawdown(cum) if n >= 1 else float("nan"),
        "calmar": core.calmar_ratio(r, periods_per_year),
        "hit_rate": float((r > 0.0).mean()) if n >= 1 else float("nan"),
        "avg_win": float(winners.mean()) if not winners.empty else float("nan"),
        "avg_loss": float(losers.mean()) if not losers.empty else float("nan"),
        "annual_turnover": _annual_turnover(trades, equity, n, periods_per_year),
    }


# --- Capacity estimate (spec §4.5) -----------------------------------------

# AUM sweep grid for the capacity report: log-spaced from $10k to $10B. Reported
# verbatim alongside the bisected crossing AUM; by convention (not tuned).
_CAPACITY_AUM_GRID = np.logspace(4, 10, num=13)


def _net_expected_return(candidate: dict, cost_model, aum: float) -> float:
    """Net expected per-period return of ``candidate`` at a given book size.

    For each coin the traded notional is ``traded_fraction * aum``; the cost model
    charges fee + size-scaled slippage on it (using the coin's ADV + liquidity
    rank). The total cost expressed as a fraction of the book is the per-period
    cost drag, subtracted from the gross expected return.
    """
    if aum <= 0.0:
        return candidate["gross_expected_return"]
    cost = 0.0
    for frac, adv, rank in candidate["coins"]:
        notional = float(frac) * aum
        cost += float(cost_model(notional, adv, int(rank), aum))
    return float(candidate["gross_expected_return"] - cost / aum)


def capacity(
    candidate: dict,
    cost_model,
    *,
    aum_grid=_CAPACITY_AUM_GRID,
    max_aum: float = 1e15,
) -> dict:
    """AUM at which the candidate's NET expected return crosses zero (spec §4.5).

    Sweeps ``aum_grid`` recording the net expected return at each book size (the
    size-scaled slippage is recomputed per coin from order/ADV), then bisects for
    the exact crossing. Because the slippage drag rises monotonically with AUM
    while fees are constant per dollar, net expected return is monotone-decreasing
    in AUM, so the crossing — if any — is unique.

    Args:
        candidate: ``{"gross_expected_return": float, "coins": [(traded_fraction,
            adv, liquidity_rank), ...]}`` — the per-period gross edge and the
            per-coin order profile (fraction of the book traded each rebalance).
        cost_model: the B1 ``trade_cost`` signature ``f(notional, adv, rank, aum)``.
        aum_grid: book sizes to report in the sweep (by convention).
        max_aum: upper bisection bound; if net is still positive there, the
            strategy is treated as effectively unbounded (``+inf``).

    Returns:
        ``{"capacity_aum": float, "sweep": [{"aum", "net_expected_return"}, ...]}``.
        ``capacity_aum`` is ``+inf`` when net never turns negative (e.g. a
        frictionless cost model).
    """
    sweep = [
        {"aum": float(a), "net_expected_return": _net_expected_return(candidate, cost_model, a)}
        for a in aum_grid
    ]

    # Net at AUM->0+ is the gross expected return (no traded notional, no cost).
    net_low = candidate["gross_expected_return"]
    if net_low <= 0.0:
        # Even costless the edge is non-positive: zero capacity.
        return {"capacity_aum": 0.0, "sweep": sweep}

    net_high = _net_expected_return(candidate, cost_model, max_aum)
    if net_high > 0.0:
        # Still net-positive at the upper bound -> unbounded (e.g. no slippage).
        return {"capacity_aum": float("inf"), "sweep": sweep}

    # Monotone decreasing from net_low>0 to net_high<=0: bisect the crossing.
    lo, hi = 0.0, max_aum
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _net_expected_return(candidate, cost_model, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return {"capacity_aum": float(0.5 * (lo + hi)), "sweep": sweep}
