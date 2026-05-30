"""Cost-aware backtest engine: positions, t+1-close execution, vol targeting.

``run_backtest`` walks a per-rebalance dollar-neutral coin-weight schedule
through time (Task B2, spec §4.1/§4.3/§4.4/§3.5). For each rebalance date ``r``:

  1. **Cap** each coin's raw weight at ``PER_COIN_CAP`` (concentration bound).
  2. **Vol-target** the book to ``vol_target`` with a *walk-forward* scalar built
     from the trailing realized book vol — only book returns from windows that
     ended **at or before ``r``** enter, so the scalar at ``r`` cannot see any
     return dated after ``r`` (the no-look-ahead cardinal rule, spec §7).
  3. **Cap gross** at ``GROSS_LEVERAGE_CAP`` (the 2x dollar-neutral band, §4.1).
  4. **Execute at the t+1 close**: the trade list is ``target - current`` where
     ``current`` is the *drifted* prior book (prior target weights grown by their
     realized window P&L). ``cost_model`` charges fee + slippage on each coin's
     traded notional (= |Δweight| * aum), using the coin's ADV and liquidity rank
     from the universe panel. Costs flow out of equity; price P&L flows in.

The book is priced one holding window at a time: the realized return of coin
``s`` held over ``(r, next_r]`` is the compounded ``holding_return`` over that
window (a collapsed short-leg coin's crash therefore books into the P&L — a
short of a −90% coin earns +90% * |weight|; survivorship is honored, spec §1.2).

Outputs (a dict of three frames, all NET of costs unless the cost model is
frictionless):
  * ``equity``    — ``[date, equity, gross_return, net_return, cost]`` per window
                    plus the opening row at ``aum``;
  * ``positions`` — ``[rebalance_date, symbol, weight]`` the executed book;
  * ``trades``    — ``[rebalance_date, symbol, target_weight, prior_weight,
                    traded_weight, traded_notional, cost]``.

Pure / walk-forward; performs no I/O and reads no OOS-dated rows (the OOS-once
discipline lives in ``scripts/run_backtest.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amom.config import (
    ANNUAL_VOL_TARGET,
    GROSS_LEVERAGE_CAP,
    HOLDING_DAYS,
    PER_COIN_CAP,
    VOL_TARGET_LOOKBACK,
)

# Annualization: each holding window spans HOLDING_DAYS calendar days (spec §1.4),
# so the book return series has ~365 / HOLDING_DAYS periods per year. Matches the
# Stage-2 runner's convention.
_DAYS_PER_YEAR = 365.0
_PERIODS_PER_YEAR = _DAYS_PER_YEAR / HOLDING_DAYS

_EQUITY_COLUMNS = ["date", "equity", "gross_return", "net_return", "cost"]
_POSITION_COLUMNS = ["rebalance_date", "symbol", "weight"]
_TRADE_COLUMNS = [
    "rebalance_date",
    "symbol",
    "target_weight",
    "prior_weight",
    "traded_weight",
    "traded_notional",
    "cost",
]


def _wide_returns(holding_returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form holding-return panel to wide (dates x symbols)."""
    return (
        holding_returns.pivot_table(
            index="date", columns="symbol", values="holding_return"
        )
        .sort_index()
    )


def _window_return(returns_wide: pd.DataFrame, symbol: str, start, end) -> float:
    """Compounded simple return for ``symbol`` over ``(start, end]``.

    NaN cells inside the window compound as 0% (held, no price observed) — the
    same convention as ``factor.portfolio`` (a coin delisting mid-hold closes at
    its last observed price; spec §1.2). All-NaN -> 0.0 (no realized move).
    """
    if symbol not in returns_wide.columns:
        return 0.0
    mask = (returns_wide.index > start) & (returns_wide.index <= end)
    window = returns_wide.loc[mask, symbol]
    if window.dropna().empty:
        return 0.0
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def _liquidity_rank_asof(universe_panel: pd.DataFrame, rebalance_date) -> dict:
    """ADV and 0-based liquidity rank per symbol, as-of ``rebalance_date``.

    Uses the latest universe row dated ``<= rebalance_date`` (point-in-time, no
    look-ahead). Rank 0 = most liquid (largest ADV). Returns ``{symbol: (adv,
    rank)}``; symbols absent as-of the date get ``(0.0, large_rank)`` (no ADV
    reference -> the cost model charges fee only).
    """
    asof = universe_panel[universe_panel["date"] <= rebalance_date]
    if asof.empty:
        return {}
    latest_date = asof["date"].max()
    row = asof[asof["date"] == latest_date]
    ranked = row.sort_values("adv_30d", ascending=False).reset_index(drop=True)
    out = {}
    for rank, rec in ranked.iterrows():
        out[rec["symbol"]] = (float(rec["adv_30d"]), int(rank))
    return out


def _cap_weights(raw: pd.Series, scalar: float) -> pd.Series:
    """Scale raw weights, clip per-coin, then clip gross — preserving neutrality.

    ``raw`` is the dollar-neutral target (Σ ~= 0). The vol ``scalar`` levers the
    book; ``PER_COIN_CAP`` bounds each |weight|; ``GROSS_LEVERAGE_CAP`` bounds the
    book gross (Σ|weight|). Both caps clip uniformly per leg so the long/short
    balance — hence dollar-neutrality — is preserved.
    """
    w = raw * float(scalar)
    # Per-coin cap (uniform clip preserves the sign pattern, not neutrality if it
    # bit unevenly — but legs are equal-weight so it clips both legs identically).
    w = w.clip(lower=-PER_COIN_CAP, upper=PER_COIN_CAP)
    gross = w.abs().sum()
    if gross > GROSS_LEVERAGE_CAP and gross > 0:
        w = w * (GROSS_LEVERAGE_CAP / gross)
    return w


def run_backtest(
    weights_by_rebal: pd.DataFrame,
    holding_returns: pd.DataFrame,
    universe_panel: pd.DataFrame,
    *,
    aum: float,
    cost_model,
    vol_target: float = ANNUAL_VOL_TARGET,
) -> dict:
    """Walk per-rebalance dollar-neutral weights through time, net of costs.

    Args:
        weights_by_rebal: long-form ``[rebalance_date, symbol, weight]`` — the raw
            dollar-neutral candidate book (Σ signed weight ~= 0 per rebalance).
        holding_returns: long-form ``[date, symbol, holding_return]`` spot panel.
        universe_panel: long-form universe history with at least ``[date, symbol,
            adv_30d]`` (the ADV + liquidity rank source for the cost model).
        aum: book size in the cost currency (constant across the run).
        cost_model: ``f(traded_notional, adv, liquidity_rank, aum) -> cost`` (the
            B1 ``trade_cost``; a frictionless model gives the gross curve).
        vol_target: annualized vol the book is scaled to (default
            ``ANNUAL_VOL_TARGET``; the B4 runner passes the config constant).

    Returns:
        ``{"equity": DataFrame, "positions": DataFrame, "trades": DataFrame}``.
    """
    returns_wide = _wide_returns(holding_returns)
    rebal_dates = sorted(pd.to_datetime(weights_by_rebal["rebalance_date"]).unique())

    equity = float(aum)
    equity_rows = [
        {"date": rebal_dates[0], "equity": equity, "gross_return": 0.0,
         "net_return": 0.0, "cost": 0.0}
    ] if rebal_dates else []
    position_rows: list[dict] = []
    trade_rows: list[dict] = []

    book_returns: list[float] = []  # realized NET book returns, one per closed window
    prior_weights = pd.Series(dtype=float)  # drifted prior book at the next rebalance

    # Each rebalance r (except the last) has a forward window (r, next_r] to price.
    for i in range(len(rebal_dates) - 1):
        r = rebal_dates[i]
        next_r = rebal_dates[i + 1]

        raw = (
            weights_by_rebal[weights_by_rebal["rebalance_date"] == r]
            .set_index("symbol")["weight"]
            .astype(float)
        )

        # --- Walk-forward vol scalar: only trailing (<= r) book returns. -------
        scalar = _vol_scalar(book_returns, vol_target)
        target = _cap_weights(raw, scalar)

        # --- Execute at t+1 close: trade list = target - drifted prior book. ---
        liq = _liquidity_rank_asof(universe_panel, r)
        symbols = sorted(set(target.index) | set(prior_weights.index))
        period_cost = 0.0
        for s in symbols:
            tw = float(target.get(s, 0.0))
            pw = float(prior_weights.get(s, 0.0))
            traded = tw - pw
            traded_notional = traded * equity
            adv, rank = liq.get(s, (0.0, 10**9))
            cost = float(cost_model(traded_notional, adv, rank, aum))
            period_cost += cost
            trade_rows.append({
                "rebalance_date": r, "symbol": s, "target_weight": tw,
                "prior_weight": pw, "traded_weight": traded,
                "traded_notional": traded_notional, "cost": cost,
            })

        for s, w in target.items():
            position_rows.append({"rebalance_date": r, "symbol": s, "weight": float(w)})

        # --- Price the window (r, next_r]; book return is the weighted P&L. ----
        coin_rets = {s: _window_return(returns_wide, s, r, next_r) for s in target.index}
        gross_return = float(sum(target[s] * coin_rets[s] for s in target.index))

        cost_drag = period_cost / equity if equity != 0 else 0.0
        net_return = gross_return - cost_drag
        equity = equity * (1.0 + net_return)
        book_returns.append(net_return)

        equity_rows.append({
            "date": next_r, "equity": equity, "gross_return": gross_return,
            "net_return": net_return, "cost": period_cost,
        })

        # --- Drift the executed book by realized P&L -> prior book at next_r. --
        prior_weights = pd.Series(
            {s: target[s] * (1.0 + coin_rets[s]) for s in target.index}, dtype=float
        )

    return {
        "equity": pd.DataFrame(equity_rows, columns=_EQUITY_COLUMNS),
        "positions": pd.DataFrame(position_rows, columns=_POSITION_COLUMNS),
        "trades": pd.DataFrame(trade_rows, columns=_TRADE_COLUMNS),
    }


def _vol_scalar(book_returns: list[float], vol_target: float) -> float:
    """Walk-forward vol scalar from trailing book returns (no look-ahead).

    Uses the most recent ``VOL_TARGET_LOOKBACK`` realized (closed-window) book
    returns — all dated at or before the current rebalance — to estimate the
    annualized realized vol, and returns ``vol_target / realized_vol``. Before
    enough history exists (or a zero-vol estimate), the scalar is 1.0 (the raw,
    un-levered book), so the very first rebalances are un-targeted by construction
    rather than by reading future data.
    """
    if len(book_returns) < 2:
        return 1.0
    trailing = np.asarray(book_returns[-VOL_TARGET_LOOKBACK:], dtype=float)
    period_vol = float(trailing.std(ddof=1))
    if period_vol <= 0.0 or not np.isfinite(period_vol):
        return 1.0
    realized_annual_vol = period_vol * np.sqrt(_PERIODS_PER_YEAR)
    return float(vol_target / realized_annual_vol)
