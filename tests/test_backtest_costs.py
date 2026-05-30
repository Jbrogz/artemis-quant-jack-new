"""Tests for the spot cost model (Task B1, spec §4.2 / plan §B1).

``trade_cost(traded_notional, adv, liquidity_rank, aum)`` is the per-trade
cost of one side of a fill: a spot taker fee on the traded notional plus a
size-scaled slippage tiered by liquidity rank and scaled with the order/ADV
ratio. There is NO funding term — this is a spot strategy and Artemis exposes
no funding (spec §3.1 / §4.2; disclosed). The discriminating behaviours:

* fee is symmetric per side (same cost to buy or sell a given |notional|);
* slippage is higher for small / illiquid names (rank outside the top-N) than
  for top names, for the same order/ADV ratio;
* slippage scales up with order size relative to ADV (a bigger fraction of ADV
  costs proportionally more bps);
* a zero trade costs exactly zero (no fee, no slippage).
"""

import math

from amom.backtest.costs import trade_cost
from amom.config import (
    SLIPPAGE_ADV_REF,
    SLIPPAGE_SMALL_BPS,
    SLIPPAGE_TOP_BPS,
    SLIPPAGE_TOP_N,
    TAKER_FEE_BPS,
)

# A liquid top-name order at exactly the reference order/ADV ratio: slippage
# should equal the top-tier base bps, so the total is just fee + base slippage.
_ADV = 1_000_000.0
_REF_NOTIONAL = SLIPPAGE_ADV_REF * _ADV  # order == SLIPPAGE_ADV_REF of ADV


def _fee(notional: float) -> float:
    return abs(notional) * TAKER_FEE_BPS / 1e4


# ---------------------------------------------------------------------------
# zero trade -> zero cost
# ---------------------------------------------------------------------------

def test_zero_trade_costs_nothing():
    assert trade_cost(0.0, adv=_ADV, liquidity_rank=0, aum=1e6) == 0.0


# ---------------------------------------------------------------------------
# fee is symmetric per side (buy vs sell of the same magnitude)
# ---------------------------------------------------------------------------

def test_fee_symmetric_per_side():
    buy = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    sell = trade_cost(-_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    # cost depends on |notional| only — identical for the two sides.
    assert math.isclose(buy, sell, rel_tol=0.0, abs_tol=1e-12)
    assert buy > 0.0


def test_fee_component_is_taker_bps_on_notional():
    # At the reference order/ADV ratio the top-tier slippage equals the base
    # top bps, so total = fee + (SLIPPAGE_TOP_BPS bps on notional).
    cost = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    expected_slip = _REF_NOTIONAL * SLIPPAGE_TOP_BPS / 1e4
    assert math.isclose(cost, _fee(_REF_NOTIONAL) + expected_slip, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# slippage higher for small / illiquid names (rank outside top-N)
# ---------------------------------------------------------------------------

def test_slippage_higher_for_small_illiquid_names():
    top = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    small = trade_cost(
        _REF_NOTIONAL, adv=_ADV, liquidity_rank=SLIPPAGE_TOP_N + 5, aum=1e6
    )
    # same fee, same order/ADV ratio -> the only difference is the slippage tier
    assert small > top
    # the gap equals the bps spread of the two tiers on the reference notional
    expected_gap = _REF_NOTIONAL * (SLIPPAGE_SMALL_BPS - SLIPPAGE_TOP_BPS) / 1e4
    assert math.isclose(small - top, expected_gap, rel_tol=1e-12)


def test_tier_boundary_is_top_n():
    # rank < SLIPPAGE_TOP_N is the top tier; rank >= SLIPPAGE_TOP_N is small.
    last_top = trade_cost(
        _REF_NOTIONAL, adv=_ADV, liquidity_rank=SLIPPAGE_TOP_N - 1, aum=1e6
    )
    first_small = trade_cost(
        _REF_NOTIONAL, adv=_ADV, liquidity_rank=SLIPPAGE_TOP_N, aum=1e6
    )
    assert first_small > last_top


# ---------------------------------------------------------------------------
# slippage scales up with order size relative to ADV
# ---------------------------------------------------------------------------

def test_slippage_scales_with_order_over_adv():
    small_order = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    big_order = trade_cost(4 * _REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    # 4x the notional -> 4x the fee AND a larger order/ADV ratio -> >4x slippage,
    # so the per-dollar cost rises with size (super-linear total cost).
    assert big_order > 4 * small_order


def test_slippage_scales_with_inverse_adv():
    # Same order, thinner book (smaller ADV) -> higher order/ADV ratio -> more bps.
    deep = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    thin = trade_cost(_REF_NOTIONAL, adv=_ADV / 4, liquidity_rank=0, aum=1e6)
    assert thin > deep


def test_slippage_at_reference_ratio_equals_base_bps():
    # By construction the order/ADV scaling is normalized at SLIPPAGE_ADV_REF:
    # an order that is exactly SLIPPAGE_ADV_REF of ADV pays the base tier bps.
    cost = trade_cost(_REF_NOTIONAL, adv=_ADV, liquidity_rank=0, aum=1e6)
    slip = cost - _fee(_REF_NOTIONAL)
    assert math.isclose(slip, _REF_NOTIONAL * SLIPPAGE_TOP_BPS / 1e4, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# no funding term (spot) — cost is purely fee + slippage, both >= 0
# ---------------------------------------------------------------------------

def test_cost_is_nonnegative_and_has_no_funding_term():
    # Whatever the sign of the trade, there is no funding credit/debit that could
    # make the cost negative — a spot taker only ever pays fee + slippage.
    for notional in (_REF_NOTIONAL, -_REF_NOTIONAL, 5 * _REF_NOTIONAL):
        assert trade_cost(notional, adv=_ADV, liquidity_rank=10, aum=1e6) >= 0.0
