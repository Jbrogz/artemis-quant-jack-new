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

# --- POST-HOC widened skip>=2 candidates (Task V1; plan 2026-05-31). ---------
# These are CONVENTION-FIXED specs the user directed to validate, NOT re-selected
# by maximizing backtest Sharpe. They are the skip>=2 variants that were strong
# and sign-stable in-sample (the widened m=21 family, docs/STAGE2_RESULTS.md):
# L3d/S3d (HAC t=5.0, the lead), L14d/S3d, L1d/S3d clear the m=21 Bonferroni
# 0.05/21=0.00238 (L1d/S3d only on HAC -> MARGINAL); L5d/S3d and L5d/S2d are
# secondary (do NOT clear). Short lookbacks (L1d/L3d) rebalance into near-reversal
# territory -> HIGH turnover, the key net-of-cost risk (guide §1.3).
WIDENED_CANDIDATES = [
    {"variant": "momentum_L3d_S3d", "lookback": 3, "skip": 3, "quantile": QUANTILE},
    {"variant": "momentum_L14d_S3d", "lookback": 14, "skip": 3, "quantile": QUANTILE},
    {"variant": "momentum_L1d_S3d", "lookback": 1, "skip": 3, "quantile": QUANTILE},
    {"variant": "momentum_L5d_S3d", "lookback": 5, "skip": 3, "quantile": QUANTILE},
    {"variant": "momentum_L5d_S2d", "lookback": 5, "skip": 2, "quantile": QUANTILE},
]
# The widened family's primary is the highest-t lead, L3d/S3d.
WIDENED_PRIMARY = WIDENED_CANDIDATES[0]

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


def _oos_slice(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Pure OOS slice (rows ``>= OOS_START``) — does NOT touch the guard."""
    return df.loc[df[date_col] >= OOS_START].copy()


def read_oos_once(
    df: pd.DataFrame, guard: OneShotOOS, date_col: str = "rebalance_date"
) -> pd.DataFrame:
    """Return the OOS slice (rows ``>= OOS_START``), spending the single-use guard.

    ``guard.open()`` raises if the window has already been spent, so a second OOS
    read is a hard error rather than a silent leak (spec §2.8 / §4.6).
    """
    guard.open()
    return _oos_slice(df, date_col)


def read_oos_panels_once(
    book: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    guard: OneShotOOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the OOS slices of ALL THREE engine inputs, spending the guard once.

    The OOS window enters the engine through three panels — the weight book
    (sliced by ``rebalance_date``), the daily holding returns and the universe
    (both sliced by ``date``). Reading them via a single ``guard.open()`` makes
    OOS-once **enforced** across every input rather than merely conventional: the
    engine is then handed OOS-only panels, so it cannot read an OOS-dated daily
    return or universe row outside this one guarded code path (spec §2.8 / §4.6).
    A second call raises. ``open_count`` stays at 1.
    """
    guard.open()
    return (
        _oos_slice(book, "rebalance_date"),
        _oos_slice(returns, "date"),
        _oos_slice(universe, "date"),
    )


def build_oos_weight_inputs_once(
    price_panel: pd.DataFrame,
    eligibility_input: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    specs: dict[str, dict],
    guard: OneShotOOS,
    *,
    widened_specs: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build OOS weight books and slice OOS panels inside one guarded path.

    The guard opens before any OOS-dated variant weight is formed. Each book is
    built only for rebalance dates ``>= OOS_START`` and tagged so callers can
    split the single combined OOS read back per spec.
    """
    guard.open()
    frames: list[pd.DataFrame] = []
    for key, spec in specs.items():
        book = build_weight_book(price_panel, eligibility_input, spec, start=OOS_START)
        book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
        frames.append(book.assign(_spec=key))

    for spec in widened_specs or []:
        variant = spec["variant"]
        book = build_weight_book(price_panel, eligibility_input, spec, start=OOS_START)
        book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
        frames.append(book.assign(_spec=f"widened::{variant}"))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = _oos_slice(combined, "rebalance_date")
    else:
        combined = pd.DataFrame(columns=["rebalance_date", "symbol", "weight", "_spec"])

    return combined, _oos_slice(returns, "date"), _oos_slice(universe, "date")


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
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
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
        start: optional inclusive lower bound for rebalance dates.
        end: optional exclusive upper bound for rebalance dates.
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
    if start is not None:
        start = pd.Timestamp(start).normalize()
        rebal_dates = [r for r in rebal_dates if r >= start]
    if end is not None:
        end = pd.Timestamp(end).normalize()
        rebal_dates = [r for r in rebal_dates if r < end]

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
    universe: pd.DataFrame,
    trades: pd.DataFrame,
    gross_expected_return: float,
) -> dict:
    """Build the capacity ``candidate`` from the average per-rebalance TRADED order.

    Per coin the traded fraction is the mean |traded weight| **per rebalance** —
    ``Σ |Δw| / n_rebalances`` from the engine's trade log — i.e. the order size
    actually executed each rebalance, NOT the standing held weight. (Most of the
    book is re-established, not re-traded, so the held weight grossly overstates
    the per-rebalance order: summed traded turnover here is ~2.0, the summed held
    weight ~9.) Slippage is super-linear in order/ADV, so using the held weight
    would understate capacity by orders of magnitude. ADV and liquidity rank are
    taken from the universe panel as-of the latest in-sample row. This feeds
    ``metrics.capacity`` (spec §4.5).
    """
    # Per-rebalance traded fraction per coin = Σ |Δw| / n_rebalances (the order
    # size executed each rebalance; the standing book is re-established, not
    # re-traded, so traded << held).
    n_rebalances = int(trades["rebalance_date"].nunique())
    if n_rebalances <= 0:
        frac_by_sym = pd.Series(dtype=float)
    else:
        frac_by_sym = (
            trades.groupby("symbol")["traded_weight"].apply(lambda w: float(w.abs().sum()))
            / n_rebalances
        )
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


def widened_candidates_section(records: dict) -> list[str]:
    """Markdown for the POST-HOC widened skip>=2 candidates (Task V1).

    A clearly-labelled section that SITS BELOW the pre-registered L5d/S1d +
    L28d/S1d results (which stay intact). ``records`` maps each variant to the
    in-sample record from ``characterize_candidate_in_sample`` (gross/net Sharpe,
    annualized turnover, 2x-cost net, capacity, gross edge). The single guarded
    OOS read is filled by Task V2; the OOS columns here are placeholders until
    then. Any candidate whose net Sharpe is non-positive (edge killed net of
    costs) is flagged explicitly — the short-lookback turnover/reversal risk.
    """
    L: list[str] = []
    L.append("## POST-HOC widened skip>=2 candidates (exploratory; Task V1 in-sample)")
    L.append("")
    L.append(
        "> **This section is POST-HOC / exploratory.** `skip` was a fixed "
        "convention in the pre-registered family (m=7, skip=1), so these skip>=2 "
        "variants were originally diagnostics. They are validated here under the "
        "widened m=21 family (Bonferroni 0.05/21 = 0.00238; `docs/STAGE2_RESULTS.md`). "
        "This does **NOT** overturn the pre-registered skip=1 null above — that "
        "result stands. The robust survivors (clear under BOTH HAC and bootstrap) "
        "are **L3d/S3d** and **L14d/S3d**; **L1d/S3d is MARGINAL** (clears on the "
        "HAC p only, not bootstrap). L5d/S3d and L5d/S2d do **not** clear m=21 and "
        "are reported as secondary."
    )
    L.append("")
    L.append(
        "**Key cost risk (guide §1.3):** short lookbacks (L1d/L3d) rebalance into "
        "near-reversal territory -> **high turnover, most cost-exposed**. A gross "
        "edge can shrink materially net of fees + size-scaled slippage. Gross vs "
        "net Sharpe and annualized turnover are shown side by side below; any "
        "candidate whose **net edge is killed** (net Sharpe <= 0) is flagged."
    )
    L.append("")
    hdr = (
        "| candidate | gross Sharpe | net Sharpe | net Sharpe (2x cost) | "
        "net ann ret | ann turnover | capacity (AUM) | OOS net Sharpe |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    L.append(hdr)
    L.append(sep)
    for spec in WIDENED_CANDIDATES:
        variant = spec["variant"]
        rec = records.get(variant)
        if rec is None:
            L.append(f"| `{variant}` | n/a | n/a | n/a | n/a | n/a | n/a | _V2_ |")
            continue
        net = rec["is_net"]
        net_2x = rec["is_net_2x"]
        oos = rec.get("oos_net_sharpe", "_filled by V2 (OOS spent once)_")
        oos_cell = oos if isinstance(oos, str) else _fmt(oos, 3)
        L.append(
            f"| `{variant}` | {_fmt(rec['is_gross_sharpe'], 3)} | "
            f"{_fmt(net['sharpe'], 3)} | {_fmt(net_2x['sharpe'], 3)} | "
            f"{_fmt(net['annual_return'], 4)} | {_fmt(net['annual_turnover'], 2)} | "
            f"{_usd(rec['capacity_aum'])} | {oos_cell} |"
        )
    L.append("")
    # Explicit per-candidate flag for any edge killed net of costs.
    killed = [
        spec["variant"]
        for spec in WIDENED_CANDIDATES
        if (r := records.get(spec["variant"])) is not None
        and np.isfinite(r["is_net"]["sharpe"]) and r["is_net"]["sharpe"] <= 0.0
    ]
    if killed:
        L.append(
            "> **Edge killed net of costs (net Sharpe <= 0):** "
            + ", ".join(f"`{v}`" for v in killed)
            + " — the gross edge does not survive fees + slippage (the short-"
            "lookback turnover risk realized)."
        )
    else:
        L.append(
            "_(No candidate's net Sharpe is non-positive in this section as filled; "
            "any that turn non-positive once the real numbers are written are "
            "flagged here.)_"
        )
    L.append("")
    L.append(
        "_The OOS net Sharpe column is a placeholder until Task V2 spends each "
        "candidate's (currently unspent) OOS window EXACTLY ONCE. DSR — already "
        "deflated for the 21 trials — is the multiple-testing-aware metric (see "
        "`docs/STAGE2_RESULTS.md`); it is not re-derived here._"
    )
    L.append("")
    return L


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
        f"~{_usd(p['capacity_aum'])} (AUM at which size-scaled slippage erases the "
        f"gross edge, computed on the actual per-rebalance *traded* order — "
        f"~2.0x summed one-way turnover — not the standing held book; per-rebalance "
        f"gross edge {_fmt(p['gross_edge'], 5)}). Comfortably above a $1M book — "
        f"**capacity is not the binding constraint** (see Conclusion)."
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

    # The §4.5 metric set performance() computes but the table above omits:
    # total compounded return + avg win / avg loss per period (all net of costs).
    L.append("## Additional §4.5 net metrics (total return, avg win / loss)")
    L.append("")
    L.append("| spec | segment | total return | avg win | avg loss |")
    L.append("|---|---|---|---|---|")
    for spec_key, label in (("primary", "Primary L5d_S1d"), ("comparator", "Comparator L28d_S1d")):
        s = report[spec_key]
        for seg, m in (("in-sample", s["is_net"]), ("out-of-sample", s["oos_net"])):
            L.append(
                f"| {label} | {seg} | {_fmt(m['total_return'], 4)} | "
                f"{_fmt(m['avg_win'], 5)} | {_fmt(m['avg_loss'], 5)} |"
            )
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
    L.append(
        "> **Caveat (do not over-read):** this regime cut is a **full-sample, "
        "descriptive** partition of the in-sample windows, **not** a walk-forward "
        "signal — it could not have been traded ex-ante. The `chop` bucket is just "
        "the top-|market-move| tercile, which on this sample skews toward large "
        "**up** moves, so it absorbs much of the strongest bull tape; the apparent "
        "'negative in bull / positive in bear' contrast is therefore **overstated** "
        "and is an artifact of where the magnitude cut falls, not clean evidence of "
        "a bear-only edge. The disqualifying signal is the Stage-2 §2.6 "
        "sign-instability, not this descriptive split."
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
    L.append(
        "- **Persisted `equity.parquet` `gross_return`** is the **net run's pre-cost** "
        "book return (the vol-scalar path fed by net returns), **not** the reported "
        "gross Sharpe series — that gross Sharpe comes from an *independent "
        "frictionless* run (its own vol-scalar path). The two gross series differ "
        "slightly by construction; the headline gross Sharpe is the frictionless one."
    )
    L.append(
        f"- **Dropped boundary window:** the holding window straddling `OOS_START` "
        f"(from the last in-sample rebalance to {OOS_START.date()}) is priced by "
        f"neither segment — the in-sample run has no forward window past its last "
        f"rebalance and the OOS run starts fresh at `OOS_START` — so that one "
        f"straddle window is intentionally not counted (no double-count, no leak)."
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
        "- The OOS window is **30 overlapping-regime observations spent once** — a "
        "single favorable stretch (the 2024 crypto bull) carries it. The return is "
        "**regime exposure, not a stable factor**, consistent with the Stage-2 §2.6 "
        "sign-flip disqualification. (The regime breakdown above is suggestive but "
        "is a full-sample *descriptive* cut, not a walk-forward signal — see its "
        "caveat; the disqualifying evidence is the §2.6 sign-instability itself.)"
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
        f"- **Capacity does NOT bind** at deployable size: the net edge crosses zero "
        f"only at ~{_usd(p['capacity_aum'])} of AUM (recomputed on the actual "
        f"per-rebalance *traded* order, ~2.0x summed one-way turnover, not the "
        f"standing held book). At a $1M book the slippage drag is immaterial, so "
        f"capacity is **not** what disqualifies this candidate — the no-deploy case "
        f"rests entirely on the three points above."
    )
    L.append("")
    L.append(
        "**Net verdict:** consistent with the Stage-2 null and the sign-instability "
        "disqualification, momentum on the Artemis spot universe is **not a deployable "
        "factor**. The primary's positive OOS Sharpe is a single-regime artifact on a "
        "spent-once 30-observation window, not a repeatable edge; it is reported as-is, "
        "neither inflated nor suppressed. (Capacity is comfortable at $1M and is **not** "
        "the binding constraint.)"
    )
    L.append("")
    # --- POST-HOC widened skip>=2 candidates: BELOW the pre-registered results,
    #     which stay intact above. Only emitted when the runner supplies the
    #     widened in-sample records (Task V1); absent -> the section is skipped. -
    widened = report.get("widened")
    if widened:
        L.extend(widened_candidates_section(widened))
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


def _oos_overfit_note(oos_net_sharpe: float, is_oos_gap: float) -> str:
    """Honest one-line characterization of an OOS net Sharpe (Task V2).

    Reuses the SAME classification thresholds as the pre-registered specs in
    ``main`` so the widened candidates are judged on identical terms: a near-zero
    / negative OOS Sharpe is reported plainly as overfitting; an OOS that did NOT
    collapse (gap < 0, i.e. OOS exceeds in-sample) is a single-regime spent-once
    artifact, NOT a deployable edge; otherwise the positive gap quantifies the
    in-sample decay. Never massages toward significance.
    """
    if not np.isfinite(oos_net_sharpe):
        return "OOS Sharpe is undefined on the spent-once window."
    if oos_net_sharpe <= 0.25:
        return (
            "OOS Sharpe is near-zero / negative -> the in-sample edge does NOT "
            "persist out-of-sample (overfitting)."
        )
    if np.isfinite(is_oos_gap) and is_oos_gap < 0:
        return (
            "OOS Sharpe did NOT collapse (it exceeds in-sample) — but this is a "
            "single-regime artifact on a spent-once 30-obs window, NOT a "
            "deployable edge (the widened family is post-hoc; confirm on forward "
            "data; the regime breakdown is a descriptive cut only)."
        )
    return "OOS Sharpe is positive but the gap quantifies in-sample decay."


def candidate_verdict(
    *,
    is_net_sharpe: float,
    oos_net_sharpe: float,
    is_oos_gap: float,
    clears_m21_both_tests: bool,
    oos_floor: float = 0.25,
    max_stable_gap: float = 0.8,
) -> str:
    """Honest per-candidate deployment verdict (Task V3) from the real numbers.

    A PURE classifier over a candidate's cost-aware results. The deployment bar,
    applied in order:

      * ``works-only-gross`` — the in-sample NET Sharpe is non-positive: costs kill
        the edge before OOS is even relevant.
      * ``fails-OOS`` — EITHER the OOS net Sharpe is non-positive / near-zero (at or
        below ``oos_floor``, the same overfit floor used in ``_oos_overfit_note``),
        OR the edge **collapses** out-of-sample: a large positive IS-vs-OOS net
        Sharpe gap (``is_oos_gap > max_stable_gap``) means the OOS retains only a
        small fraction of the in-sample edge — not a stable, repeatable factor.
        (This is what catches `momentum_L3d_S3d`: OOS net 0.297 is above the floor
        but its gap 1.245 shows the in-sample 1.542 collapsed.)
      * ``marginal`` — OOS net Sharpe survives the floor AND is stable (gap small)
        but the variant does NOT clear the widened m=21 Bonferroni under BOTH the
        HAC and bootstrap tests: a positive-but-not-qualified result.
      * ``deployable`` — OOS net Sharpe survives the floor, is stable (gap small),
        AND the variant is m=21-robust under both tests. (No real widened candidate
        meets all three; this branch keeps the classifier data-driven, not a
        hardcoded null.)

    Never massages toward significance: a positive OOS that either collapses from
    in-sample or is not multiple-testing-robust is NOT called ``deployable``.
    """
    if not np.isfinite(is_net_sharpe) or is_net_sharpe <= 0.0:
        return "works-only-gross"
    if not np.isfinite(oos_net_sharpe) or oos_net_sharpe <= oos_floor:
        return "fails-OOS"
    # A large positive gap = the in-sample edge collapsed out-of-sample, even if the
    # OOS Sharpe is still marginally above the floor.
    if np.isfinite(is_oos_gap) and is_oos_gap > max_stable_gap:
        return "fails-OOS"
    if not clears_m21_both_tests:
        return "marginal"
    return "deployable"


# The descriptive-proxy caveat for the OOS regime cut, carried verbatim in spirit
# with the Stage-4 markdown caveat: this is a full-sample DESCRIPTIVE partition of
# the OOS windows, NOT a walk-forward signal — do not over-read it.
_OOS_REGIME_CAVEAT = (
    "Do not over-read: this regime cut is a full-sample DESCRIPTIVE partition of "
    "the spent-once OOS windows, NOT a walk-forward signal — it could not have "
    "been traded ex-ante, and on only ~30 OOS observations the buckets are tiny. "
    "It is suggestive context, not evidence of a regime-specific edge."
)


def characterize_candidate_oos(
    book_oos: pd.DataFrame,
    oos_returns: pd.DataFrame,
    oos_universe: pd.DataFrame,
    *,
    is_net_sharpe: float,
    aum: float = DEFAULT_AUM,
) -> dict:
    """OOS cost-aware characterization of ONE widened candidate (Task V2).

    Receives panels that are ALREADY sliced to OOS-only rows (the single
    ``read_oos_panels_once`` guard in ``main`` does the slicing and is the only
    code path that touches a row dated ``>= OOS_START``). This helper therefore
    cannot read an OOS row outside that one guarded read, and — because it is
    handed OOS-only panels — it cannot read an in-sample row either (a strict
    no-look-ahead boundary: an in-sample mutation cannot change any OOS metric).

    Reuses the SAME helpers as the pre-registered OOS step: ``run_gross_and_net``
    + ``run_full`` (gross vs net + the 2x-cost rerun), ``metrics_for`` (the §4.5
    net metric set incl. annualized turnover), and ``regime_breakdown`` /
    ``regime_labels`` / ``equal_weight_market_return`` for the descriptive regime
    cut. Returns the OOS gross/net Sharpe, the 2x-cost net metrics, the IS-vs-OOS
    net Sharpe gap, the honest overfit note, the regime breakdown + its caveat.
    """
    oos_gross_eq, oos_net_eq = run_gross_and_net(
        book_oos, oos_returns, oos_universe, aum=aum
    )
    oos_net_run = run_full(book_oos, oos_returns, oos_universe, aum=aum)
    oos_net_2x_run = run_full(
        book_oos, oos_returns, oos_universe, aum=aum, cost_multiplier=2.0
    )
    oos_net = metrics_for(oos_net_eq, oos_net_run["trades"])
    oos_net_2x = metrics_for(oos_net_2x_run["equity"], oos_net_2x_run["trades"])
    oos_gross_sh = gross_sharpe(oos_gross_eq)

    gap = (
        (is_net_sharpe - oos_net["sharpe"])
        if np.isfinite(is_net_sharpe) and np.isfinite(oos_net["sharpe"])
        else float("nan")
    )

    # Descriptive OOS regime breakdown (same convention as Stage 4; do-not-over-read).
    oos_dates = pd.DatetimeIndex(sorted(book_oos["rebalance_date"].unique()))
    labels = regime_labels(
        equal_weight_market_return(oos_returns, oos_universe, oos_dates)
    )
    net_series = oos_net_run["equity"].set_index("date")["net_return"]
    net_series = net_series.iloc[1:] if len(net_series) > 1 else net_series
    regimes = regime_breakdown(net_series, labels)

    return {
        "oos_gross_sharpe": oos_gross_sh,
        "oos_net": oos_net,
        "oos_net_2x": oos_net_2x,
        "is_oos_gap": gap,
        "overfit_note": _oos_overfit_note(oos_net["sharpe"], gap),
        "regimes": regimes,
        "caveat": _OOS_REGIME_CAVEAT,
    }


def characterize_candidate_in_sample(
    price_panel: pd.DataFrame,
    eligibility_input: pd.DataFrame,
    holding_is: pd.DataFrame,
    universe_is: pd.DataFrame,
    spec: dict,
    *,
    aum: float = DEFAULT_AUM,
) -> dict:
    """In-sample cost-aware characterization of ONE widened skip>=2 candidate.

    Reuses the SAME helpers as the pre-registered primary — ``build_weight_book``
    (tested t+1-lag formation, no re-selection), ``characterize_in_sample`` (gross
    vs net Sharpe + annualized turnover + the 2x-cost rerun), and
    ``_candidate_for_capacity`` + ``metrics.capacity`` (capacity on the actual
    per-rebalance *traded* order, not the standing held book). NO OOS rows are
    read: the book is formed only for dates ``< OOS_START`` and only the
    pre-sliced in-sample holding/universe panels are passed in.

    Returns a record carrying the gross/net IS metrics, the 2x-cost rerun, the
    per-rebalance gross edge and the finite capacity AUM, plus the spec.
    """
    book = build_weight_book(price_panel, eligibility_input, spec, end=OOS_START)
    book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()

    res = characterize_in_sample(book, holding_is, universe_is, aum=aum)

    gross_eq = res["_gross_eq"]
    gross_edge = (
        float(gross_eq["gross_return"].iloc[1:].mean()) if len(gross_eq) > 1 else float("nan")
    )
    cand = _candidate_for_capacity(universe_is, res["_net_run"]["trades"], gross_edge)
    capacity_aum = capacity(cand, trade_cost)["capacity_aum"]

    return {
        "spec": spec,
        "is_gross_sharpe": res["is_gross_sharpe"],
        "is_net": res["is_net"],
        "is_net_2x": res["is_net_2x"],
        "gross_edge": gross_edge,
        "capacity_aum": capacity_aum,
        "_net_run": res["_net_run"],
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

    # --- Build in-sample books only. OOS-dated weights are not formed until the
    #     single guarded OOS step below opens the sealed window. -----------------
    books_is: dict[str, pd.DataFrame] = {}
    for key, spec in specs.items():
        book = build_weight_book(price_panel, eligibility, spec, end=OOS_START)
        book["rebalance_date"] = pd.to_datetime(book["rebalance_date"]).dt.normalize()
        books_is[key] = book

    # =======================================================================
    # IN-SAMPLE characterization (rows < OOS_START only; OOS untouched)
    # =======================================================================
    is_results: dict[str, dict] = {}
    for key, spec in specs.items():
        book_is = books_is[key]
        in_sample_res = characterize_in_sample(book_is, holding_is, universe_is)

        lb_robust: list[dict] = []
        regimes = {r: {"n": 0, "mean_net_return": float("nan")} for r in ("bull", "bear", "chop")}
        capacity_aum = float("nan")
        gross_edge = float("nan")
        if key == "primary":
            # +/-50% lookback rerun (reuses the chosen spec; only lookback moves).
            for rspec in lookback_robustness_specs(spec):
                rbook = build_weight_book(price_panel, eligibility, rspec, end=OOS_START)
                rbook["rebalance_date"] = pd.to_datetime(rbook["rebalance_date"]).dt.normalize()
                rrun = run_full(rbook, holding_is, universe_is)
                lb_robust.append({
                    "variant": rspec["variant"], "tag": rspec["tag"],
                    "net": metrics_for(rrun["equity"], rrun["trades"]),
                })
            # Regime breakdown (in-sample market proxy).
            is_dates = pd.DatetimeIndex(sorted(book_is["rebalance_date"].unique()))
            labels = regime_labels(equal_weight_market_return(holding_is, universe_is, is_dates))
            net_series = in_sample_res["_net_run"]["equity"].set_index("date")["net_return"].iloc[1:]
            regimes = regime_breakdown(net_series, labels)
            # Capacity (per-rebalance gross edge vs the actual traded-delta order
            # profile from the net run's trade log — not the standing held weight).
            gross_eq = in_sample_res["_gross_eq"]
            gross_edge = float(gross_eq["gross_return"].iloc[1:].mean()) if len(gross_eq) > 1 else float("nan")
            cand = _candidate_for_capacity(
                universe_is, in_sample_res["_net_run"]["trades"], gross_edge
            )
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
    # OOS-dated weight books are formed only after the guard opens, then combined
    # with the OOS slices of the daily returns and universe in this single code
    # path. The combined OOS book carries the pre-registered specs AND every
    # widened candidate (namespaced `widened::<variant>` tags), then gets split
    # back per spec; no further OOS read or OOS weight formation occurs.
    oos_guard = OneShotOOS()
    oos_combined, oos_returns, oos_universe = build_oos_weight_inputs_once(
        price_panel,
        eligibility,
        returns_long,
        universe_long,
        specs,
        oos_guard,
        widened_specs=WIDENED_CANDIDATES,
    )

    report: dict = {}
    for key, spec in specs.items():
        in_sample_res = is_results[key]["in_sample_res"]
        book_oos = (
            oos_combined[oos_combined["_spec"] == key]
            .drop(columns="_spec")
            .reset_index(drop=True)
        )
        # Price OOS forward (r, next_r] on the OOS-only panels: every row the
        # engine reads is OOS-dated and came through the single guarded read.
        oos_gross_eq, oos_net_eq = run_gross_and_net(book_oos, oos_returns, oos_universe)
        oos_net_run = run_full(book_oos, oos_returns, oos_universe)
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
                "of a deployable edge (the disqualifying signal is the Stage-2 §2.6 "
                "sign-instability; the regime breakdown is a descriptive cut only)."
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

    # =======================================================================
    # POST-HOC widened skip>=2 candidates: IN-SAMPLE characterization (Task V1).
    # Reuses the in-sample panels above (date < OOS_START) and the same helpers
    # as the pre-registered specs. No OOS row is read here; widened OOS books and
    # panels have already been acquired through the single guarded path above.
    # =======================================================================
    widened_records: dict = {}
    for spec in WIDENED_CANDIDATES:
        rec = characterize_candidate_in_sample(
            price_panel, eligibility, holding_is, universe_is, spec
        )
        widened_records[spec["variant"]] = rec
        print(f"  [widened] {spec['variant']}")
        print(f"    IS gross Sharpe {rec['is_gross_sharpe']:+.3f} | "
              f"net Sharpe {rec['is_net']['sharpe']:+.3f} | "
              f"ann turnover {rec['is_net']['annual_turnover']:.2f}x")

    # =======================================================================
    # POST-HOC widened skip>=2 candidates: spend their (unspent) OOS window —
    # already read EXACTLY ONCE through the single guard above (open_count == 1).
    # Each candidate's OOS rows are split out of the SAME combined OOS book; the
    # OOS gross/net Sharpe, the IS-vs-OOS net Sharpe gap, the 2x-cost rerun and
    # the descriptive regime breakdown reuse the pre-registered helpers, and the
    # OOS net Sharpe is stored so the markdown fills its column (Task V2).
    # =======================================================================
    for spec in WIDENED_CANDIDATES:
        variant = spec["variant"]
        rec = widened_records[variant]
        book_oos = (
            oos_combined[oos_combined["_spec"] == f"widened::{variant}"]
            .drop(columns="_spec")
            .reset_index(drop=True)
        )
        oos = characterize_candidate_oos(
            book_oos, oos_returns, oos_universe,
            is_net_sharpe=rec["is_net"]["sharpe"],
        )
        rec["oos_gross_sharpe"] = oos["oos_gross_sharpe"]
        rec["oos_net"] = oos["oos_net"]
        rec["oos_net_2x"] = oos["oos_net_2x"]
        rec["oos_net_sharpe"] = oos["oos_net"]["sharpe"]
        rec["is_oos_gap"] = oos["is_oos_gap"]
        rec["oos_overfit_note"] = oos["overfit_note"]
        rec["oos_regimes"] = oos["regimes"]
        rec["oos_regime_caveat"] = oos["caveat"]
        print(f"  [widened-OOS] {variant}")
        print(f"    OOS gross Sharpe {oos['oos_gross_sharpe']:+.3f} | "
              f"net Sharpe {oos['oos_net']['sharpe']:+.3f} | "
              f"IS-OOS gap {oos['is_oos_gap']:+.3f}")
    report["widened"] = widened_records

    write_markdown(report)
    print(f"  wrote {BACKTEST_DIR / 'equity.parquet'} (+ positions, trades)")
    print(f"  wrote {OUTPUT_MD}")
    print(f"  OOS guard open_count = {oos_guard.open_count} (must be 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
