"""Spot holding returns + terminal crash-return imputation (Task R4, spec §1.2).

The whole survivorship argument is realized here. The universe panel keeps every
coin ever seen (incl. ones that collapsed) and carries a point-in-time
``delisted_asof`` death signal. ``build_holding_returns`` produces the realized
**simple spot** return of holding each coin, and — crucially — it keeps the
final realized leg of a coin that crashes and then stops reporting. A coin that
falls ~95% on its last print and goes silent contributes a realized terminal
return ≈ −0.95 on that print's date; it is **not** dropped because it later
becomes ineligible. Dropping it is exactly the survivorship bias guide §1.1
forbids (it would silently erase the loss).

Conventions (spec §3.1, §4 Stage 1.2):
  - Holding return = **simple** price return ``p_t / p_{t-1} − 1`` (aggregatable
    across coins within a period). Log compounding is used elsewhere to chain a
    series through time; the two are never mixed.
  - **No funding term.** Artemis serves no funding and this is a spot book; the
    holding return is pure price.
  - **Point-in-time.** Each symbol's returns come from its own realized price
    series; a return dated ``d`` uses only that symbol's prices dated ``<= d``.
  - **Terminal close.** A coin leaving the eligible set mid-hold has its position
    closed at the **last observed price** (spec §1.2); the realized return into
    that last price is preserved. No phantom return is fabricated on the later
    delisting date, where the coin has no price.

The function is pure and performs no I/O.
"""
from __future__ import annotations

import pandas as pd

_OUTPUT_COLUMNS = ["date", "symbol", "ret"]


def build_holding_returns(
    price_panel: pd.DataFrame, universe_panel: pd.DataFrame
) -> pd.DataFrame:
    """Build the per-symbol simple spot holding-return series.

    Args:
        price_panel: long DataFrame ``[date, symbol, price]`` covering every
            symbol ever seen, including coins that crashed and then stopped
            reporting. Each symbol's realized price series is used as-is.
        universe_panel: the R3 universe panel ``[date, symbol, ..., price_last_date,
            delisted_asof, ...]``. The death signal it carries is consistent with
            "close at the last observed price": once a coin's reporting lapses it
            has no further price, so no return is booked past its final print —
            the realized crash leg is the terminal return.

    Returns:
        Long DataFrame ``[date, symbol, ret]`` with one row per realized holding
        return. The first observation of each symbol yields no return (no prior
        price to hold from). ``ret`` is the simple return ``p_t / p_{t-1} − 1``;
        the final leg of a coin that crashes then delists is booked, never NaN or
        dropped. Sorted by ``(date, symbol)``.
    """
    if price_panel.empty:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"),
             "symbol": pd.Series([], dtype=object),
             "ret": pd.Series([], dtype=float)}
        )

    priced = price_panel.dropna(subset=["price"]).copy()
    priced["date"] = pd.to_datetime(priced["date"])
    priced = priced.sort_values(["symbol", "date"])

    # Simple spot return per symbol from its own realized series; the crash leg
    # (e.g. 5/100 − 1 = −0.95) is captured here and survives even though the
    # coin delists right after — it is the realized terminal return.
    priced["ret"] = (
        priced.groupby("symbol", sort=False)["price"].pct_change()
    )

    # The first observation of each symbol has no prior price -> drop that row
    # (there is no realized holding return yet), but keep every later return,
    # including the terminal crash. We never drop rows on eligibility/delisting,
    # which is what preserves the survivorship signal.
    returns = priced.dropna(subset=["ret"])

    out = (
        returns[["date", "symbol", "ret"]]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    out["ret"] = out["ret"].astype(float)
    return out[_OUTPUT_COLUMNS]
