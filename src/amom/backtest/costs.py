"""Spot cost model: per-side taker fee + size-scaled, tiered slippage (spec §4.2).

The cost of executing one side of a fill is

    cost = fee + slippage
    fee      = |traded_notional| * TAKER_FEE_BPS / 1e4
    slippage = |traded_notional| * base_bps(liquidity_rank)
               * (order_adv_ratio / SLIPPAGE_ADV_REF) / 1e4
    order_adv_ratio = |traded_notional| / adv
    base_bps = SLIPPAGE_TOP_BPS  if liquidity_rank < SLIPPAGE_TOP_N  else SLIPPAGE_SMALL_BPS

Both components are charged on the *magnitude* of the traded notional, so the
cost is symmetric per side (a buy and a sell of the same size cost the same) and
never negative. There is **no funding term**: this is a spot strategy and
Artemis exposes no funding rate (spec §3.1 / §4.2), so the guide's third cost
component is N/A here — stated, not silently dropped.

The slippage is a deliberately simple market-impact proxy: it is tiered by
liquidity rank (top-N liquid names slip less than smaller / illiquid names) and
scales *linearly* in the order/ADV ratio, normalized so that an order equal to
``SLIPPAGE_ADV_REF`` of the coin's ADV pays exactly the tier's base bps. A bigger
fraction of ADV therefore costs proportionally more bps — total cost is
super-linear in order size, which is what drives the capacity estimate (§4.5).
This function is pure and performs no I/O.
"""

import math

from amom.config import (
    SLIPPAGE_ADV_REF,
    SLIPPAGE_SMALL_BPS,
    SLIPPAGE_TOP_BPS,
    SLIPPAGE_TOP_N,
    TAKER_FEE_BPS,
)

_BPS = 1e4


def _base_slippage_bps(liquidity_rank: int) -> float:
    """Tier the base slippage bps by liquidity rank vs the top-N cutoff.

    ``rank < SLIPPAGE_TOP_N`` is the liquid top tier (``SLIPPAGE_TOP_BPS``);
    everything at or beyond the cutoff is the smaller / illiquid tier
    (``SLIPPAGE_SMALL_BPS``). A two-tier step, by convention (spec §4.2).
    """
    if liquidity_rank < SLIPPAGE_TOP_N:
        return float(SLIPPAGE_TOP_BPS)
    return float(SLIPPAGE_SMALL_BPS)


def trade_cost(
    traded_notional: float,
    adv: float,
    liquidity_rank: int,
    aum: float,
) -> float:
    """Spot execution cost of one side of a fill (fee + size-scaled slippage).

    Args:
        traded_notional: signed traded notional for this leg (target − current,
            in the same currency as ``adv``); only its magnitude enters the cost.
        adv: the coin's average daily volume, same currency as the notional. Used
            as the slippage market-impact denominator. A missing, non-finite, or
            non-positive ADV uses a conservative "unknown liquidity" assumption
            rather than receiving free market-impact.
        liquidity_rank: 0-based cross-sectional liquidity rank of the coin (0 =
            most liquid). ``rank < SLIPPAGE_TOP_N`` slips at the top tier.
        aum: book size; carried through for the capacity sweep (§4.5) where the
            traded notional is derived from ``aum``. Not used directly here — the
            size scaling is fully captured by ``traded_notional / adv``.

    Returns:
        Non-negative cost in the currency of ``traded_notional``. A zero trade
        costs exactly zero. There is no funding term (spot; spec §3.1).
    """
    notional = abs(float(traded_notional))
    if notional == 0.0:
        return 0.0

    fee = notional * TAKER_FEE_BPS / _BPS

    adv_value = None if adv is None else float(adv)
    if adv_value is None or not math.isfinite(adv_value) or adv_value <= 0.0:
        # No ADV reference is a data-quality problem, not a free fill. Treat the
        # order as consuming 100% of unknown ADV in the illiquid tier, so the
        # backtest/capacity estimate is conservative and the gap is visible.
        order_adv_ratio = 1.0
        base_bps = float(SLIPPAGE_SMALL_BPS)
    else:
        order_adv_ratio = notional / adv_value
        base_bps = _base_slippage_bps(int(liquidity_rank))

    # Linear in the order/ADV ratio, normalized at SLIPPAGE_ADV_REF: at that
    # reference ratio the bps equals base_bps; larger orders scale it up.
    slippage_bps = base_bps * (order_adv_ratio / SLIPPAGE_ADV_REF)
    slippage = notional * slippage_bps / _BPS

    return fee + slippage
