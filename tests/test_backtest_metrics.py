"""Tests for net-of-cost performance metrics + capacity (Task B3, spec §4.5).

``performance(equity, trades, returns)`` summarizes a backtest run NET of costs:
total & annualized return, annualized vol, Sharpe, Sortino, max drawdown, Calmar,
hit rate, avg win vs avg loss, and **annualized turnover** computed from the trade
log. Sharpe / max-drawdown / Calmar reuse ``amom.stats.core`` (no re-derivation).

``capacity(candidate, cost_model)`` sweeps AUM, recomputes the size-scaled
slippage per coin via the order/ADV ratio, and returns the AUM at which the
strategy's **net** expected return crosses zero (the capacity ceiling, §4.5).

Discriminating behaviours (plan §B3):
  * turnover is computed from the trade log (Σ|traded notional| / equity, annualized);
  * capacity is monotone in AUM (more AUM -> more slippage -> lower net return);
  * Sortino / Calmar signs are sane (positive for a winning series, negative for a
    losing one) and Sortino penalizes only downside.

All fixtures are synthetic and offline; no API calls, no disk reads.
"""

import numpy as np
import pandas as pd

from amom.backtest.costs import trade_cost
from amom.backtest.metrics import capacity, performance


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# A small NET equity curve / return series with both winning and losing windows
# so hit-rate, avg-win/avg-loss, and downside (Sortino) deviation are all exercised.
_RETURNS = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])


def _equity_from_returns(returns: pd.Series, aum: float = 1_000_000.0) -> pd.DataFrame:
    """Build an engine-shaped equity frame from a net-return series."""
    equity = aum
    rows = [{"date": ts("2020-01-01"), "equity": equity,
             "gross_return": 0.0, "net_return": 0.0, "cost": 0.0}]
    for i, r in enumerate(returns):
        equity *= (1.0 + r)
        rows.append({"date": ts("2020-01-01") + pd.Timedelta(days=30 * (i + 1)),
                     "equity": equity, "gross_return": r, "net_return": r, "cost": 0.0})
    return pd.DataFrame(rows)


def _trades(notionals: list[float], aum: float = 1_000_000.0) -> pd.DataFrame:
    """A trade log whose |traded_notional| sums to a known turnover."""
    rows = []
    for i, n in enumerate(notionals):
        rows.append({
            "rebalance_date": ts("2020-01-01") + pd.Timedelta(days=30 * i),
            "symbol": f"C{i}", "target_weight": 0.0, "prior_weight": 0.0,
            "traded_weight": n / aum, "traded_notional": n, "cost": 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# performance() returns the full net metric set
# ---------------------------------------------------------------------------

def test_performance_returns_full_metric_set():
    equity = _equity_from_returns(_RETURNS)
    trades = _trades([100_000.0, 50_000.0])
    perf = performance(equity, trades, _RETURNS)
    expected = {
        "total_return", "annual_return", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "avg_win", "avg_loss",
        "annual_turnover",
    }
    assert expected.issubset(set(perf.keys()))


def test_total_return_matches_equity_curve():
    equity = _equity_from_returns(_RETURNS)
    perf = performance(equity, _trades([1.0]), _RETURNS)
    expected_total = equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0
    assert np.isclose(perf["total_return"], expected_total)


def test_hit_rate_and_avg_win_loss():
    perf = performance(_equity_from_returns(_RETURNS), _trades([1.0]), _RETURNS)
    # 3 winners / 2 losers out of 5.
    assert np.isclose(perf["hit_rate"], 3 / 5)
    assert np.isclose(perf["avg_win"], np.mean([0.05, 0.03, 0.04]))
    assert np.isclose(perf["avg_loss"], np.mean([-0.02, -0.01]))


# ---------------------------------------------------------------------------
# Sharpe / max-drawdown / Calmar reuse stats.core (consistency check)
# ---------------------------------------------------------------------------

def test_sharpe_drawdown_calmar_reuse_stats_core():
    from amom.stats import core
    ppy = 12.0
    perf = performance(
        _equity_from_returns(_RETURNS), _trades([1.0]), _RETURNS,
        periods_per_year=ppy,
    )
    assert np.isclose(perf["sharpe"], core.sharpe_ratio(_RETURNS, ppy))
    cum = (1 + _RETURNS).cumprod() - 1
    assert np.isclose(perf["max_drawdown"], core.max_drawdown(cum))
    assert np.isclose(perf["calmar"], core.calmar_ratio(_RETURNS, ppy))


# ---------------------------------------------------------------------------
# Sortino: penalizes only downside; sign sane
# ---------------------------------------------------------------------------

def test_sortino_uses_only_downside_deviation():
    ppy = 12.0
    perf = performance(_equity_from_returns(_RETURNS), _trades([1.0]), _RETURNS,
                       periods_per_year=ppy)
    downside = _RETURNS[_RETURNS < 0]
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(ppy)
    expected = _RETURNS.mean() * ppy / dd
    assert np.isclose(perf["sortino"], expected)
    # A winning series has a positive Sortino.
    assert perf["sortino"] > 0


def test_sortino_negative_for_losing_series():
    losing = pd.Series([-0.05, 0.01, -0.04, -0.02, 0.01])
    perf = performance(_equity_from_returns(losing), _trades([1.0]), losing,
                       periods_per_year=12.0)
    assert perf["sortino"] < 0
    assert perf["calmar"] < 0


# ---------------------------------------------------------------------------
# annualized turnover from the trade log
# ---------------------------------------------------------------------------

def test_annual_turnover_from_trade_log():
    aum = 1_000_000.0
    returns = pd.Series([0.0, 0.0])  # flat: equity stays at aum
    equity = _equity_from_returns(returns, aum=aum)
    # Two rebalances trading 100k and 50k of a 1M book -> 0.15 one-way turnover
    # over the run; annualized by periods/yr over the realized windows.
    trades = _trades([100_000.0, 50_000.0], aum=aum)
    ppy = 365.0 / 30.0
    perf = performance(equity, trades, returns, periods_per_year=ppy)
    n_periods = len(returns)
    expected = (150_000.0 / aum) / n_periods * ppy
    assert np.isclose(perf["annual_turnover"], expected)


def test_annual_turnover_is_path_independent():
    # Turnover must normalize each period's traded notional by THAT period's
    # equity (not the constant opening book), so it is path-independent: a 1x and
    # a 2x cost run trade the SAME weight fractions but on different equity paths
    # (costs drag equity down), so the per-period notionals differ — yet turnover
    # must come out ~identical. Here two trades execute at different equity levels
    # but the SAME 10% fraction each; turnover must read 0.10 per period regardless.
    ppy = 365.0 / 30.0
    # Equity path: opens at 1.0M, drifts to 0.8M before the 2nd rebalance.
    equity = pd.DataFrame(
        [
            {"date": ts("2020-01-01"), "equity": 1_000_000.0, "gross_return": 0.0,
             "net_return": 0.0, "cost": 0.0},
            {"date": ts("2020-02-01"), "equity": 800_000.0, "gross_return": -0.2,
             "net_return": -0.2, "cost": 0.0},
            {"date": ts("2020-03-01"), "equity": 800_000.0, "gross_return": 0.0,
             "net_return": 0.0, "cost": 0.0},
        ]
    )
    # Rebalance 0 trades 10% of the 1.0M book (100k); rebalance 1 trades 10% of
    # the drifted 0.8M book (80k). Each period's turnover fraction is exactly 0.10.
    trades = pd.DataFrame(
        [
            {"rebalance_date": ts("2020-01-01"), "symbol": "A",
             "traded_weight": 0.10, "traded_notional": 100_000.0, "cost": 0.0},
            {"rebalance_date": ts("2020-02-01"), "symbol": "A",
             "traded_weight": 0.10, "traded_notional": 80_000.0, "cost": 0.0},
        ]
    )
    returns = pd.Series([-0.2, 0.0])
    perf = performance(equity, trades, returns, periods_per_year=ppy)
    # Mean per-period one-way turnover fraction = 0.10, annualized by ppy.
    assert np.isclose(perf["annual_turnover"], 0.10 * ppy)
    # The buggy constant-book0 formula would give (180k / 1.0M) / 2 * ppy =
    # 0.09 * ppy — strictly different, so this pins the path-independent fix.
    assert not np.isclose(perf["annual_turnover"], (180_000.0 / 1_000_000.0) / 2 * ppy)


# ---------------------------------------------------------------------------
# capacity: monotone in AUM and crosses zero
# ---------------------------------------------------------------------------

def _candidate(gross_expected: float = 0.01) -> dict:
    """A one-period candidate: per-coin traded fractions, ADV, liquidity rank,
    and the gross expected per-period return that costs eat into."""
    return {
        "gross_expected_return": gross_expected,
        "coins": [
            # (traded_fraction_of_aum, adv, liquidity_rank)
            (0.5, 50_000_000.0, 0),
            (0.5, 5_000_000.0, 100),
        ],
    }


def test_capacity_net_return_decreasing_in_aum():
    cand = _candidate()
    # Reuse the real cost model; net expected return must fall as AUM grows
    # (larger orders -> larger order/ADV ratio -> more slippage).
    res = capacity(cand, trade_cost)
    sweep = res["sweep"]
    nets = [row["net_expected_return"] for row in sweep]
    assert all(nets[i] >= nets[i + 1] for i in range(len(nets) - 1))


def test_capacity_aum_is_where_net_crosses_zero():
    cand = _candidate(gross_expected=0.01)
    res = capacity(cand, trade_cost)
    cap_aum = res["capacity_aum"]
    assert cap_aum > 0
    # Just below the ceiling the strategy is net-positive; just above, net-negative.
    below = _net_expected(cand, trade_cost, cap_aum * 0.5)
    above = _net_expected(cand, trade_cost, cap_aum * 2.0)
    assert below > 0
    assert above < 0


def _net_expected(candidate: dict, cost_model, aum: float) -> float:
    """Mirror of the capacity net-return computation, for the crossing test."""
    cost = 0.0
    for frac, adv, rank in candidate["coins"]:
        notional = frac * aum
        cost += cost_model(notional, adv, rank, aum)
    return candidate["gross_expected_return"] - cost / aum


def test_capacity_handles_costless_candidate_as_unbounded():
    # A frictionless cost model never eats the gross return -> no finite ceiling.
    def _free(*args, **kwargs):
        return 0.0

    res = capacity(_candidate(gross_expected=0.01), _free)
    assert np.isinf(res["capacity_aum"])
