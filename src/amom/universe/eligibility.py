"""Point-in-time eligibility — the anti-bias core (guide §1.1, spec §4 Stage 1.1).

A coin is eligible *as of* a date when, using only information available at or
before that date, it satisfies all three filters:

  - history:   ``(as_of - first_date).days >= MIN_HISTORY_DAYS`` (90 days),
  - liquidity: ``adv_30d >= MIN_ADV_USD`` ($1M trailing-30d average daily volume),
  - exclusion: ``symbol not in EXCLUDED`` (stablecoins ∪ wrapped tokens).

No look-ahead by construction: ``is_eligible`` consumes only as-of scalars —
there is no parameter through which a future-dated series could enter the
decision. ``eligible_mask`` reads, per symbol, the most recent ADV observation
dated at or before ``as_of`` and never any later row. Both functions are pure
and perform no I/O.
"""

import pandas as pd

from amom.config import MIN_ADV_USD, MIN_HISTORY_DAYS


def is_eligible(
    symbol: str,
    as_of: pd.Timestamp,
    *,
    first_date: pd.Timestamp,
    adv_30d: float,
    excluded,
) -> bool:
    """Return whether ``symbol`` is eligible as of ``as_of``.

    Args:
        symbol: the coin symbol (case-sensitive; compared against ``excluded``).
        as_of: the evaluation date; only this date's information is consulted.
        first_date: the coin's first observed price date (listing proxy).
        adv_30d: the trailing-30d average daily USD volume as of ``as_of``.
            A NaN value fails the liquidity filter.
        excluded: a set/collection of excluded symbols (stablecoins ∪ wrapped).

    Returns:
        True iff the history, liquidity, and exclusion filters all pass.
    """
    if symbol in excluded:
        return False

    if pd.isna(first_date) or (as_of - first_date).days < MIN_HISTORY_DAYS:
        return False

    if pd.isna(adv_30d) or adv_30d < MIN_ADV_USD:
        return False

    return True


def eligible_mask(
    as_of: pd.Timestamp,
    coverage_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    excluded,
) -> set:
    """Return the set of symbols eligible as of ``as_of``.

    Args:
        as_of: the evaluation date.
        coverage_df: DataFrame with columns ``[symbol, price_first_date]``
            (the listing-date reconstruction from ``coverage.first_seen_dates``).
        adv_df: long DataFrame with columns ``[date, symbol, adv_30d]``. For
            each symbol the most recent observation dated ``<= as_of`` is used;
            rows dated after ``as_of`` are ignored entirely (no look-ahead). A
            symbol with no row at or before ``as_of`` fails the liquidity filter.
        excluded: a set/collection of excluded symbols.

    Returns:
        The set of eligible symbols.
    """
    # As-of ADV: latest observation at or before as_of, per symbol.
    past = adv_df[adv_df["date"] <= as_of]
    latest = past.sort_values("date").drop_duplicates("symbol", keep="last")
    adv_as_of = latest.set_index("symbol")["adv_30d"].to_dict()

    eligible = set()
    for row in coverage_df.itertuples(index=False):
        # Missing ADV -> NaN -> fails liquidity inside is_eligible.
        adv = adv_as_of.get(row.symbol, float("nan"))
        if is_eligible(
            row.symbol,
            as_of,
            first_date=row.price_first_date,
            adv_30d=adv,
            excluded=excluded,
        ):
            eligible.add(row.symbol)

    return eligible
