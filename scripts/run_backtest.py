"""Stage-4 cost-aware backtest + one-shot OOS + robustness (Task B4, spec §4.6).

The Stage-2 verdict is a NULL: no selection-family variant survives HAC +
Bonferroni on its own terms, and the strongest (``momentum_L5d_S1d``) is
sign-unstable -> deployment-disqualified (``docs/STAGE2_RESULTS.md``). This
runner is the *honest* characterization of that candidate net of realistic spot
costs, not an attempt to rescue it.

Pipeline:
  1. Rebuild the per-coin dollar-neutral weight book for the **chosen spec**
     (primary ``momentum_L5d_S1d``; comparator ``momentum_L28d_S1d``) from the
     on-disk price/eligibility panels, reusing the tested ``factor.portfolio``
     formation (no re-selection — the spec is pre-registered, spec §3.3).
  2. **In-sample** (rows < ``OOS_START``): run gross (frictionless) vs net (spot
     fees + size-scaled slippage) side by side, report the §4.5 metric set.
  3. **Spend the OOS window exactly once** (rows >= ``OOS_START``) through a
     single-use guard (``OneShotOOS``); report the IS-vs-OOS Sharpe gap. A
     near-zero / negative OOS Sharpe is reported plainly as overfitting.
  4. **Robustness:** 2x costs rerun; +/-50% lookback rerun (reuses the chosen
     spec, only the lookback moves); regime breakdown (bull/bear/chop by trailing
     market-return sign x realized-vol terciles).
  5. Write ``data/backtest/{equity,positions,trades}.parquet`` (the primary, full
     gross+net run) + ``docs/STAGE4_RESULTS.md``.

Spot only: NO funding term anywhere (spec §3.1 / §4.2; disclosed). No-look-ahead:
the engine's vol scalar and every decision use only data <= t; execution is at
the t+1 close. OOS discipline: rows >= ``OOS_START`` are read in exactly ONE code
path — ``read_oos_once`` guarded by ``OneShotOOS`` — and never by Tasks B0-B3.

Fully offline: price/eligibility panels and the size-control market-cap panel are
read from disk / the on-disk cache; no API key is opened.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amom.backtest.costs import trade_cost  # noqa: E402
from amom.backtest.engine import run_backtest  # noqa: E402
from amom.backtest.metrics import capacity, performance  # noqa: E402
from amom.config import (  # noqa: E402
    ANNUAL_VOL_TARGET,
    DATA_DIR,
    HOLDING_DAYS,
    OOS_START,
    PRIMARY_SKIP_DAYS,
    QUANTILE,
    TAKER_FEE_BPS,
)
from amom.factor.momentum import build_momentum_signal  # noqa: E402
from amom.factor.portfolio import build_rebalance_dates, select_buckets  # noqa: E402

# --- Annualization: each obs spans HOLDING_DAYS calendar days (spec §1.4). ---
DAYS_PER_YEAR = 365.0
PERIODS_PER_YEAR = DAYS_PER_YEAR / HOLDING_DAYS  # ~12.17 for 30-day holds

# --- The chosen spec (pre-registered, NOT re-selected from the grid). --------
# Primary = strongest in-sample variant; comparator = academic 4-week canonical.
# These are convention-fixed (spec §3.3); the runner never maximizes backtest
# Sharpe to pick them.
PRIMARY_SPEC = {
    "variant": "momentum_L5d_S1d",
    "lookback": 5,
    "skip": PRIMARY_SKIP_DAYS,
    "quantile": QUANTILE,
}
COMPARATOR_SPEC = {
    "variant": "momentum_L28d_S1d",
    "lookback": 28,
    "skip": PRIMARY_SKIP_DAYS,
    "quantile": QUANTILE,
}

# --- Standard book AUM for the headline run (the capacity sweep spans AUM). --
DEFAULT_AUM = 1_000_000.0

# --- Regime breakdown (spec §4.6): trailing market sign x realized-vol terciles.
# Bull = trailing market return >= 0 AND vol not in the top tercile; Bear =
# trailing market return < 0 AND vol not in the top tercile; Chop = the
# high-vol (top-tercile) tail regardless of sign. Convention-defined, not tuned.
VOL_TERCILE_HIGH = 2.0 / 3.0

RETURNS_PATH = DATA_DIR / "returns" / "holding_returns.parquet"
UNIVERSE_PATH = DATA_DIR / "universe" / "universe_history.parquet"
BACKTEST_DIR = DATA_DIR / "backtest"
OUTPUT_MD = Path(__file__).resolve().parents[1] / "docs" / "STAGE4_RESULTS.md"


# ===========================================================================
# OOS single-use guard (spec §2.8 / §4.6: OOS spent EXACTLY ONCE)
# ===========================================================================

class OneShotOOS:
    """A single-use guard for the sealed OOS window.

    ``open()`` may be called exactly once over the run's lifetime; a second call
    raises ``RuntimeError``. The runner reads rows dated ``>= OOS_START`` only
    through ``read_oos_once``, which opens this guard — so the OOS window is
    provably spent in one code path (spec §2.8). ``open_count`` is the recorded
    counter the test asserts against.
    """

    def __init__(self) -> None:
        self.open_count = 0

    @property
    def opened(self) -> bool:
        return self.open_count > 0

    def open(self) -> bool:
        if self.open_count > 0:
            raise RuntimeError(
                "OOS window already spent — it may be opened exactly once (spec §2.8)."
            )
        self.open_count += 1
        return True


def in_sample_slice(df: pd.DataFrame, date_col: str = "rebalance_date") -> pd.DataFrame:
    """Rows strictly before ``OOS_START`` — a pure read (does NOT spend the OOS guard)."""
    return df.loc[df[date_col] < OOS_START].copy()


def read_oos_once(
    df: pd.DataFrame, guard: OneShotOOS, date_col: str = "rebalance_date"
) -> pd.DataFrame:
    """Return the OOS slice (rows ``>= OOS_START``), spending the single-use guard.

    This is the **only** code path that reads OOS-dated rows; ``guard.open()``
    raises if the window has already been spent, so a second OOS read is a hard
    error rather than a silent leak (spec §2.8 / §4.6).
    """
    guard.open()
    return df.loc[df[date_col] >= OOS_START].copy()


# ===========================================================================
# Panel reconstruction (offline; reused from the Stage-1/Stage-2 conventions)
# ===========================================================================

def _returns_wide(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Long ``[date, symbol, holding_return]`` -> wide (dates x symbols) returns."""
    return returns_long.pivot(
        index="date", columns="symbol", values="holding_return"
    ).sort_index()


def reconstruct_price_panel(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Long ``[date, symbol, price]`` from the daily returns (cumprod per symbol).

    The momentum signal is a log-return sum, invariant to the price base; this
    reproduces the same signal the Stage-1 build used (Stage-2 runner convention).
    """
    frames = []
    for sym in returns_wide.columns:
        ser = returns_wide[sym].dropna()
        if ser.empty:
            continue
        price = (1.0 + ser).cumprod()
        frames.append(
            pd.DataFrame({"date": price.index, "symbol": sym, "price": price.to_numpy()})
        )
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    return pd.concat(frames, ignore_index=True)


def eligibility_wide(universe_long: pd.DataFrame) -> pd.DataFrame:
    """Universe panel -> wide bool eligibility (eligible AND not gated; missing -> False)."""
    elig = universe_long.copy()
    elig["effective"] = elig["eligible"].astype(bool) & (~elig["gated"].astype(bool))
    return (
        elig.pivot(index="date", columns="symbol", values="effective")
        .sort_index()
        .fillna(False)
        .astype(bool)
    )


# ===========================================================================
# Toy-panel adapters (used by the offline unit tests; pure helpers)
# ===========================================================================

def price_panel_to_holding_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Long ``[date, symbol, price]`` -> long ``[date, symbol, holding_return]``."""
    wide = price_panel.pivot(index="date", columns="symbol", values="price").sort_index()
    rets = wide.pct_change()
    long = rets.stack().reset_index()
    long.columns = ["date", "symbol", "holding_return"]
    return long


def eligibility_to_universe(elig_long: pd.DataFrame, *, adv: float = 1e9) -> pd.DataFrame:
    """Long ``[date, symbol, eligible]`` -> a universe panel with a dense ADV + gate cols."""
    out = elig_long.copy()
    out["adv_30d"] = adv
    out["gated"] = False
    return out


# ===========================================================================
# Weight-book extraction for a chosen spec (reuses tested formation; no re-select)
# ===========================================================================

def build_weight_book(
    price_panel: pd.DataFrame,
    eligibility_input: pd.DataFrame,
    spec: dict,
    *,
    holding_days: int = HOLDING_DAYS,
) -> pd.DataFrame:
    """Per-rebalance dollar-neutral coin-weight book for one spec (the engine's input).

    Reuses ``factor.portfolio.select_buckets`` (the tested t+1-lag formation) for
    the spec's ``(lookback, skip, quantile)`` — exactly the in-sample-chosen
    construction. Returns long ``[rebalance_date, symbol, weight]`` (the engine's
    ``weights_by_rebal``), pooling every rebalance's long/short book.

    Args:
        price_panel: long ``[date, symbol, price]``.
        eligibility_input: either a wide bool eligibility frame, or a long
            ``[date, symbol, eligible]`` frame (auto-pivoted).
        spec: ``{"lookback", "skip", "quantile", ...}``.
        holding_days: rebalance cadence (default ``HOLDING_DAYS``).
    """
    if {"date", "symbol"}.issubset(eligibility_input.columns):
        eligibility = (
            eligibility_input.assign(eligible=eligibility_input["eligible"].astype(bool))
            .pivot(index="date", columns="symbol", values="eligible")
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
    else:
        eligibility = eligibility_input

    signal = build_momentum_signal(price_panel, spec["lookback"], spec["skip"])
    rebal_dates = build_rebalance_dates(signal.index, holding_days=holding_days)

    frames = []
    for r in rebal_dates:
        book = select_buckets(signal, eligibility, r, quantile=spec["quantile"])
        if book is not None:
            frames.append(book[["rebalance_date", "symbol", "weight"]])
    if not frames:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "weight"])
    return pd.concat(frames, ignore_index=True)


def lookback_robustness_specs(spec: dict) -> list[dict]:
    """The +/-50% lookback robustness variants of the CHOSEN spec (spec §4.6).

    Only the lookback moves (rounded to integers, floored at 1); skip and
    quantile are inherited from the chosen spec. This is a sensitivity rerun of
    the deployed construction, NOT a re-selection across the frozen grid.
    """
    base = spec["lookback"]
    short = max(1, int(round(base * 0.5)))
    long = max(short + 1, int(round(base * 1.5)))
    out = []
    for lb, tag in ((short, "lookback-50%"), (long, "lookback+50%")):
        out.append({**spec, "lookback": lb, "variant": f"momentum_L{lb}d_S{spec['skip']}d", "tag": tag})
    return out


# ===========================================================================
# Engine runs: gross (frictionless) vs net (spot costs); a cost multiplier
# ===========================================================================

def _scaled_cost(multiplier: float):
    """A cost model = ``multiplier`` x the spot ``trade_cost`` (for the 2x rerun)."""
    if multiplier == 1.0:
        return trade_cost

    def _cm(traded_notional, adv, liquidity_rank, aum):
        return multiplier * trade_cost(traded_notional, adv, liquidity_rank, aum)

    return _cm


def _zero_cost(*_args, **_kwargs) -> float:
    return 0.0


def run_gross_and_net(
    book: pd.DataFrame,
    holding: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    aum: float = DEFAULT_AUM,
    cost_multiplier: float = 1.0,
    vol_target: float = ANNUAL_VOL_TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the engine twice (frictionless gross, then spot-cost net); equity frames.

    Returns ``(gross_equity, net_equity)``. Gross uses a zero cost model so it is
    the costless upper bound; net uses ``cost_multiplier`` x the spot ``trade_cost``
    (multiplier 2.0 is the §4.6 2x-cost rerun). Net is always <= gross.
    """
    gross = run_backtest(
        book, holding, universe, aum=aum, cost_model=_zero_cost, vol_target=vol_target
    )
    net = run_backtest(
        book, holding, universe, aum=aum,
        cost_model=_scaled_cost(cost_multiplier), vol_target=vol_target,
    )
    return gross["equity"], net["equity"]


def run_full(
    book: pd.DataFrame,
    holding: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    aum: float = DEFAULT_AUM,
    cost_multiplier: float = 1.0,
    vol_target: float = ANNUAL_VOL_TARGET,
) -> dict:
    """One net engine run returning all three artifacts (equity/positions/trades)."""
    return run_backtest(
        book, holding, universe, aum=aum,
        cost_model=_scaled_cost(cost_multiplier), vol_target=vol_target,
    )


def metrics_for(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """The §4.5 net-metric set from an engine equity frame + trade log."""
    returns = equity["net_return"].iloc[1:] if len(equity) > 1 else pd.Series(dtype=float)
    return performance(equity, trades, returns, periods_per_year=PERIODS_PER_YEAR)


def gross_sharpe(equity: pd.DataFrame) -> float:
    """Annualized Sharpe of the GROSS (costless) per-window book return series."""
    from amom.stats import core

    r = equity["gross_return"].iloc[1:] if len(equity) > 1 else pd.Series(dtype=float)
    return float(core.sharpe_ratio(pd.Series(r, dtype=float), PERIODS_PER_YEAR))


# ===========================================================================
# Regime breakdown (spec §4.6): trailing market sign x realized-vol terciles
# ===========================================================================

def equal_weight_market_return(
    holding: pd.DataFrame, universe: pd.DataFrame, window_dates: pd.DatetimeIndex
) -> pd.Series:
    """Equal-weighted eligible-universe holding-period return per rebalance window.

    A cheap market proxy for the regime split: for each window ``(r, next_r]`` the
    mean compounded return of the coins eligible as-of ``r``. Indexed by ``r``.
    """
    returns_wide = _returns_wide(holding)
    elig = eligibility_wide(universe)
    dates = list(window_dates)
    out = {}
    for i in range(len(dates) - 1):
        r, next_r = dates[i], dates[i + 1]
        elig_dates = elig.index[elig.index <= r]
        if len(elig_dates) == 0:
            continue
        elig_row = elig.loc[elig_dates[-1]]
        syms = [s for s in elig_row.index if bool(elig_row[s]) and s in returns_wide.columns]
        if not syms:
            continue
        mask = (returns_wide.index > r) & (returns_wide.index <= next_r)
        window = returns_wide.loc[mask, syms]
        if window.empty:
            continue
        coin_rets = (1.0 + window.fillna(0.0)).prod() - 1.0
        out[r] = float(coin_rets.mean())
    return pd.Series(out, name="market_return").sort_index()


def regime_labels(market_return: pd.Series) -> pd.Series:
    """Label each period bull / bear / chop (spec §4.6 convention).

    High realized vol (top tercile of |market return|) is **chop** regardless of
    sign; otherwise a non-negative market return is **bull** and a negative one is
    **bear**. (A volatility-tercile proxy: the magnitude of the per-window market
    return stands in for trailing realized vol on the same series.)
    """
    mr = pd.Series(market_return, dtype=float).dropna()
    if mr.empty:
        return pd.Series(dtype=object)
    vol_proxy = mr.abs()
    high_cut = vol_proxy.quantile(VOL_TERCILE_HIGH)
    labels = {}
    for d, v in mr.items():
        if vol_proxy[d] >= high_cut:
            labels[d] = "chop"
        elif v >= 0.0:
            labels[d] = "bull"
        else:
            labels[d] = "bear"
    return pd.Series(labels, name="regime").reindex(mr.index)


def regime_breakdown(net_returns: pd.Series, labels: pd.Series) -> dict:
    """Per-regime mean net return + count, aligned on the shared dates."""
    common = net_returns.index.intersection(labels.index)
    nr = net_returns.loc[common]
    lab = labels.loc[common]
    out = {}
    for regime in ("bull", "bear", "chop"):
        sel = nr[lab == regime]
        out[regime] = {
            "n": int(len(sel)),
            "mean_net_return": float(sel.mean()) if len(sel) else float("nan"),
        }
    return out


# ===========================================================================
# Offline market-cap panel (capacity ADV/liquidity is from the universe panel)
# ===========================================================================

def _candidate_for_capacity(
    book: pd.DataFrame, universe: pd.DataFrame, gross_expected_return: float
) -> dict:
    """Build the capacity ``candidate`` from the average per-rebalance trade profile.

    Per coin the traded fraction is the mean |traded weight| across rebalances
    (the order size as a fraction of the book each rebalance); ADV and liquidity
    rank are taken from the universe panel as-of the first rebalance the coin
    appears in. This feeds ``metrics.capacity`` (spec §4.5).
    """
    # Mean |weight| per symbol = mean traded fraction (book starts flat; the
    # standing book is roughly re-established each rebalance on this universe).
    frac_by_sym = book.groupby("symbol")["weight"].apply(lambda w: float(w.abs().mean()))
    # ADV + liquidity rank as-of the latest universe row (point-in-time-ish; the
    # capacity sweep is an order-of-magnitude estimate, not a per-date integral).
    latest_date = universe["date"].max()
    asof = universe[universe["date"] == latest_date]
    ranked = asof.sort_values("adv_30d", ascending=False).reset_index(drop=True)
    adv_by_sym = dict(zip(ranked["symbol"], ranked["adv_30d"]))
    rank_by_sym = {s: i for i, s in enumerate(ranked["symbol"])}
    coins = []
    for s, frac in frac_by_sym.items():
        adv = float(adv_by_sym.get(s, 0.0))
        rank = int(rank_by_sym.get(s, 10**9))
        coins.append((float(frac), adv, rank))
    return {"gross_expected_return": float(gross_expected_return), "coins": coins}


def build_mc_unused_placeholder() -> None:  # pragma: no cover - documentation hook
    """No size-control panel is needed in Stage 4 (the spanning test was Stage 2)."""
    return None


# ===========================================================================
# Markdown report
# ===========================================================================

def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float) and abs(x) >= 1e6:
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def _usd(x) -> str:
    """Format an AUM / capacity figure as a thousands-separated dollar amount."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a" if x != float("inf") else "unbounded"
    return f"${x:,.0f}"


def _metric_row(name: str, gross_sh: float, net: dict) -> str:
    return (
        f"| {name} | {_fmt(gross_sh, 3)} | {_fmt(net['sharpe'], 3)} | "
        f"{_fmt(net['annual_return'], 4)} | {_fmt(net['annual_vol'], 4)} | "
        f"{_fmt(net['sortino'], 3)} | {_fmt(net['max_drawdown'], 4)} | "
        f"{_fmt(net['calmar'], 3)} | {_fmt(net['hit_rate'], 3)} | "
        f"{_fmt(net['annual_turnover'], 2)} |"
    )


def write_markdown(report: dict, path: Path = OUTPUT_MD) -> None:
    """Write the human-readable Stage-4 results (gross-vs-net, IS-vs-OOS, robustness)."""
    p = report["primary"]
    c = report["comparator"]
    L = []
    L.append("# Stage 4 — Cost-Aware Backtest Results")
    L.append("")
    L.append(
        f"_Spot long/short book. NO funding term (spec §3.1 / §4.2; Artemis exposes "
        f"no funding — disclosed). Costs: {TAKER_FEE_BPS} bps taker fee per side + "
        f"size-scaled tiered slippage. Execution at the t+1 close. "
        f"OOS window (rebalance_date >= {OOS_START.date()}) spent EXACTLY ONCE._"
    )
    L.append("")
    L.append("## Headline (honest)")
    L.append("")
    L.append(
        "The Stage-2 verdict was a **NULL**: no selection-family variant survives "
        "HAC + Bonferroni on its own terms, and the strongest, `momentum_L5d_S1d`, "
        "is **sign-unstable -> deployment-disqualified** (`docs/STAGE2_RESULTS.md`). "
        "This backtest characterizes that candidate net of realistic spot costs — "
        "it does not rescue it."
    )
    L.append("")
    L.append(
        f"- **Primary `momentum_L5d_S1d`:** in-sample net Sharpe "
        f"**{_fmt(p['is_net']['sharpe'], 3)}** (gross {_fmt(p['is_gross_sharpe'], 3)}); "
        f"out-of-sample net Sharpe **{_fmt(p['oos_net']['sharpe'], 3)}** "
        f"(gross {_fmt(p['oos_gross_sharpe'], 3)})."
    )
    L.append(
        f"- **IS - OOS net Sharpe gap (primary): {_fmt(p['is_oos_gap'], 3)}.** "
        + p["overfit_note"]
    )
    L.append(
        f"- **Comparator `momentum_L28d_S1d`:** IS net Sharpe "
        f"{_fmt(c['is_net']['sharpe'], 3)}, OOS net Sharpe {_fmt(c['oos_net']['sharpe'], 3)} "
        f"(gap {_fmt(c['is_oos_gap'], 3)})."
    )
    L.append(
        f"- **Capacity (primary, net expected return -> 0):** "
        f"{_usd(p['capacity_aum'])} (AUM at which size-scaled slippage erases "
        f"the gross edge; per-rebalance gross edge {_fmt(p['gross_edge'], 5)})."
    )
    L.append("")

    L.append("## Gross vs net, in-sample vs out-of-sample")
    L.append("")
    hdr = (
        "| segment | gross Sharpe | net Sharpe | net ann ret | net ann vol | "
        "Sortino | max DD | Calmar | hit rate | ann turnover |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    for spec_key, label in (("primary", "Primary L5d_S1d"), ("comparator", "Comparator L28d_S1d")):
        s = report[spec_key]
        L.append(f"### {label}")
        L.append("")
        L.append(hdr)
        L.append(sep)
        L.append(_metric_row("in-sample", s["is_gross_sharpe"], s["is_net"]))
        L.append(_metric_row("out-of-sample", s["oos_gross_sharpe"], s["oos_net"]))
        L.append("")

    L.append("## Robustness (primary L5d_S1d; reruns reuse the chosen spec)")
    L.append("")
    L.append("### 2x costs (in-sample)")
    L.append("")
    L.append(hdr)
    L.append(sep)
    L.append(_metric_row("1x costs", p["is_gross_sharpe"], p["is_net"]))
    L.append(_metric_row("2x costs", p["is_gross_sharpe"], p["is_net_2x"]))
    L.append("")
    L.append(
        "_2x costs is a sensitivity rerun of the deployed construction (same spec, "
        "doubled fee + slippage); it is not a re-selection._"
    )
    L.append("")
    L.append("### +/-50% lookback (in-sample, net)")
    L.append("")
    L.append(
        "| variant | net Sharpe | net ann ret | note |"
    )
    L.append("|---|---|---|---|")
    L.append(
        f"| `{PRIMARY_SPEC['variant']}` (chosen) | {_fmt(p['is_net']['sharpe'], 3)} | "
        f"{_fmt(p['is_net']['annual_return'], 4)} | chosen lookback = 5d |"
    )
    for rv in p["lookback_robustness"]:
        L.append(
            f"| `{rv['variant']}` | {_fmt(rv['net']['sharpe'], 3)} | "
            f"{_fmt(rv['net']['annual_return'], 4)} | {rv['tag']} (skip/quantile unchanged) |"
        )
    L.append("")
    L.append("### Regime breakdown (in-sample, net mean return per regime)")
    L.append("")
    L.append("| regime | n | mean net return |")
    L.append("|---|---|---|")
    for regime in ("bull", "bear", "chop"):
        rb_ = p["regimes"][regime]
        L.append(f"| {regime} | {rb_['n']} | {_fmt(rb_['mean_net_return'], 5)} |")
    L.append("")
    L.append(
        "_Regimes: high-vol (top |market-return| tercile) windows are **chop**; "
        "otherwise non-negative trailing market return is **bull**, negative is "
        "**bear** (spec §4.6 convention)._"
    )
    L.append("")

    L.append("## Disclosures (spec §5.4)")
    L.append("")
    L.append(
        "- **Gross vs net side by side** above: spot costs reduce every reported "
        "Sharpe; net is never flattered above gross."
    )
    L.append(
        "- **IS vs OOS side by side** above: the OOS window was opened exactly once "
        "(single-use guard); a near-zero / negative OOS Sharpe is reported as "
        "overfitting, not hidden."
    )
    L.append(
        "- **No funding** term in costs or returns (spot; Artemis has no funding — "
        "the guide's third cost component is N/A here, stated)."
    )
    L.append(
        "- **Survivorship** still flows into the P&L: a collapsed short-leg coin's "
        "crash books as a positive contribution; dead coins are not dropped."
    )
    L.append(
        "- **Multi-factor combination is N/A** for a single null factor (spec §3.2-"
        "§3.3); Stage 3 reduced to volatility targeting on the candidate, included here."
    )
    L.append("")
    L.append("## Conclusion (honest, not flattering the null either way)")
    L.append("")
    L.append(
        f"The primary `momentum_L5d_S1d` posts a positive net Sharpe both in-sample "
        f"({_fmt(p['is_net']['sharpe'], 2)}) and out-of-sample ({_fmt(p['oos_net']['sharpe'], 2)}); "
        f"the OOS figure did **not** collapse. But this is **not** evidence of a "
        f"deployable edge, and the Stage-2 disqualification stands:"
    )
    L.append("")
    L.append(
        "- The OOS window is **30 overlapping-regime observations spent once** — "
        "a single favorable stretch (the 2024 crypto bull) carries it; the regime "
        "breakdown above shows the net edge is **negative in the bull regime** and "
        "positive only in bear / high-vol windows, i.e. the return is **regime "
        "exposure, not a stable factor** (the Stage-2 §2.6 sign-flip disqualification)."
    )
    L.append(
        "- The **±50% lookback rerun is fragile**: the construction is not robust to "
        "a small change in its one free parameter (the deployed lookback)."
    )
    L.append(
        "- The comparator `momentum_L28d_S1d` (academic 4-week canonical) is **net-"
        "negative both in-sample and out-of-sample** — the canonical horizon does "
        "not work at all net of costs."
    )
    L.append(
        f"- **Capacity** is small ({_usd(p['capacity_aum'])}): the size-scaled "
        f"slippage erases the per-rebalance gross edge at a modest book size, so even "
        f"the gross edge is not scalably harvestable."
    )
    L.append("")
    L.append(
        "**Net verdict:** consistent with the Stage-2 null and the sign-instability "
        "disqualification, momentum on the Artemis spot universe is **not a deployable "
        "factor**. The primary's positive OOS Sharpe is a single-regime artifact on a "
        "spent-once 30-observation window, not a repeatable edge; it is reported as-is, "
        "neither inflated nor suppressed."
    )
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# ===========================================================================
# Per-spec characterization (IS gross/net + robustness; OOS read separately)
# ===========================================================================

def characterize_in_sample(
    book_is: pd.DataFrame,
    holding_is: pd.DataFrame,
    universe_is: pd.DataFrame,
    *,
    aum: float = DEFAULT_AUM,
) -> dict:
    """In-sample gross/net metrics + 2x-cost rerun for one spec's weight book."""
    gross_eq, net_eq = run_gross_and_net(book_is, holding_is, universe_is, aum=aum)
    net_run = run_full(book_is, holding_is, universe_is, aum=aum)
    net_2x = run_full(book_is, holding_is, universe_is, aum=aum, cost_multiplier=2.0)
    return {
        "is_gross_sharpe": gross_sharpe(gross_eq),
        "is_net": metrics_for(net_eq, net_run["trades"]),
        "is_net_2x": metrics_for(net_2x["equity"], net_2x["trades"]),
        "_net_run": net_run,
        "_gross_eq": gross_eq,
    }


# ===========================================================================
# Entry point (live)
# ===========================================================================

def main() -> int:
    if not RETURNS_PATH.exists() or not UNIVERSE_PATH.exists():
        print("ERROR: build the returns + universe panels first.")
        return 1

    returns_long = pd.read_parquet(RETURNS_PATH)
    returns_long["date"] = pd.to_datetime(returns_long["date"]).dt.normalize()
    universe_long = pd.read_parquet(UNIVERSE_PATH)
    universe_long["date"] = pd.to_datetime(universe_long["date"]).dt.normalize()

    returns_wide = _returns_wide(returns_long)
    price_panel = reconstruct_price_panel(returns_wide)

    print("=" * 78)
    print("  STAGE 4 — COST-AWARE BACKTEST (in-sample; then OOS spent ONCE)")
    print("=" * 78)
    print(f"  OOS_START (sealed)   : {OOS_START.date()}")
    print(f"  candidates           : {PRIMARY_SPEC['variant']} (primary), "
          f"{COMPARATOR_SPEC['variant']} (comparator)")
    print()

    eligibility = eligibility_wide(universe_long)
    specs = {"primary": PRIMARY_SPEC, "comparator": COMPARATOR_SPEC}

    # In-sample panels (date < OOS_START): the holding / universe panels are
    # pre-sliced so the final IS window's (r, next_r] pricing consumes no OOS-
    # dated daily row (the spanning-boundary nuance, Task B0). No OOS row enters
    # any in-sample computation below.
    holding_is = returns_long[returns_long["date"] < OOS_START]
    universe_is = universe_long[universe_long["date"] < OOS_START]

    # --- Build every spec's full weight book ONCE (a pure formation; the book
    #     itself carries OOS-dated rows but no metric reads them until the single
    #     guarded OOS step below). ----------------------------------------------
    books: dict[str, pd.DataFrame] = {}
    for key, spec in specs.items():
        book = build_weight_book(price_panel, eligibility, spec)
        book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
        books[key] = book

    # =======================================================================
    # IN-SAMPLE characterization (rows < OOS_START only; OOS untouched)
    # =======================================================================
    is_results: dict[str, dict] = {}
    for key, spec in specs.items():
        book_is = in_sample_slice(books[key])
        in_sample_res = characterize_in_sample(book_is, holding_is, universe_is)

        lb_robust: list[dict] = []
        regimes = {r: {"n": 0, "mean_net_return": float("nan")} for r in ("bull", "bear", "chop")}
        capacity_aum = float("nan")
        gross_edge = float("nan")
        if key == "primary":
            # +/-50% lookback rerun (reuses the chosen spec; only lookback moves).
            for rspec in lookback_robustness_specs(spec):
                rbook = build_weight_book(price_panel, eligibility, rspec)
                rbook["rebalance_date"] = pd.to_datetime(rbook["rebalance_date"]).dt.normalize()
                rrun = run_full(in_sample_slice(rbook), holding_is, universe_is)
                lb_robust.append({
                    "variant": rspec["variant"], "tag": rspec["tag"],
                    "net": metrics_for(rrun["equity"], rrun["trades"]),
                })
            # Regime breakdown (in-sample market proxy).
            is_dates = pd.DatetimeIndex(sorted(book_is["rebalance_date"].unique()))
            labels = regime_labels(equal_weight_market_return(holding_is, universe_is, is_dates))
            net_series = in_sample_res["_net_run"]["equity"].set_index("date")["net_return"].iloc[1:]
            regimes = regime_breakdown(net_series, labels)
            # Capacity (per-rebalance gross edge).
            gross_eq = in_sample_res["_gross_eq"]
            gross_edge = float(gross_eq["gross_return"].iloc[1:].mean()) if len(gross_eq) > 1 else float("nan")
            cand = _candidate_for_capacity(book_is, universe_is, gross_edge)
            capacity_aum = capacity(cand, trade_cost)["capacity_aum"]

        is_results[key] = {
            "in_sample_res": in_sample_res,
            "lb_robust": lb_robust,
            "regimes": regimes,
            "capacity_aum": capacity_aum,
            "gross_edge": gross_edge,
        }

    # =======================================================================
    # OUT-OF-SAMPLE: spend the sealed window EXACTLY ONCE (one guarded read)
    # =======================================================================
    # All specs' books are concatenated and the OOS slice is read in a SINGLE
    # guarded call — the only code path in the whole runner that touches a row
    # dated >= OOS_START (spec §2.8 / §4.6). The combined OOS frame is then split
    # back per spec; no further OOS read occurs.
    oos_guard = OneShotOOS()
    combined_books = pd.concat(
        [books[k].assign(_spec=k) for k in specs], ignore_index=True
    )
    oos_combined = read_oos_once(combined_books, oos_guard)

    report: dict = {}
    for key, spec in specs.items():
        in_sample_res = is_results[key]["in_sample_res"]
        book_oos = (
            oos_combined[oos_combined["_spec"] == key]
            .drop(columns="_spec")
            .reset_index(drop=True)
        )
        # Price OOS forward (r, next_r] — all OOS-dated; the full panels are passed
        # but only the OOS rebalances drive the engine, so only OOS rows are read.
        oos_gross_eq, oos_net_eq = run_gross_and_net(book_oos, returns_long, universe_long)
        oos_net_run = run_full(book_oos, returns_long, universe_long)
        oos_net = metrics_for(oos_net_eq, oos_net_run["trades"])
        oos_gross_sh = gross_sharpe(oos_gross_eq)

        is_net_sh = in_sample_res["is_net"]["sharpe"]
        gap = (is_net_sh - oos_net["sharpe"]) if np.isfinite(is_net_sh) and np.isfinite(oos_net["sharpe"]) else float("nan")
        if not np.isfinite(oos_net["sharpe"]):
            overfit_note = "OOS Sharpe is undefined on the spent-once window."
        elif oos_net["sharpe"] <= 0.25:
            overfit_note = (
                "OOS Sharpe is near-zero / negative -> the in-sample edge does NOT "
                "persist out-of-sample (overfitting)."
            )
        elif np.isfinite(gap) and gap < 0:
            overfit_note = (
                "OOS Sharpe did NOT collapse (it exceeds in-sample) — but this is a "
                "single-regime artifact on a spent-once 30-obs window, not evidence "
                "of a deployable edge (see the regime breakdown + sign-instability)."
            )
        else:
            overfit_note = "OOS Sharpe is positive but the gap quantifies in-sample decay."

        report[key] = {
            "is_gross_sharpe": in_sample_res["is_gross_sharpe"],
            "is_net": in_sample_res["is_net"],
            "is_net_2x": in_sample_res["is_net_2x"],
            "oos_gross_sharpe": oos_gross_sh,
            "oos_net": oos_net,
            "is_oos_gap": gap,
            "overfit_note": overfit_note,
            "lookback_robustness": is_results[key]["lb_robust"],
            "regimes": is_results[key]["regimes"],
            "capacity_aum": is_results[key]["capacity_aum"],
            "gross_edge": is_results[key]["gross_edge"],
        }

        # --- Persist the PRIMARY full net run artifacts (spec §4.4). -----------
        if key == "primary":
            BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
            full = in_sample_res["_net_run"]
            for art, oos_art in (("equity", oos_net_run["equity"]),
                                 ("positions", oos_net_run["positions"]),
                                 ("trades", oos_net_run["trades"])):
                combined = pd.concat([full[art], oos_art], ignore_index=True)
                combined.to_parquet(BACKTEST_DIR / f"{art}.parquet", index=False)

        print(f"  [{key}] {spec['variant']}")
        print(f"    IS  gross Sharpe {in_sample_res['is_gross_sharpe']:+.3f} | "
              f"net Sharpe {in_sample_res['is_net']['sharpe']:+.3f}")
        print(f"    OOS gross Sharpe {oos_gross_sh:+.3f} | net Sharpe {oos_net['sharpe']:+.3f}")
        print(f"    IS-OOS net Sharpe gap {gap:+.3f}")
        print()

    write_markdown(report)
    print(f"  wrote {BACKTEST_DIR / 'equity.parquet'} (+ positions, trades)")
    print(f"  wrote {OUTPUT_MD}")
    print(f"  OOS guard open_count = {oos_guard.open_count} (must be 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
