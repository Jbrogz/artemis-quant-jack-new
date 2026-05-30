"""Listing-date (first-seen) reconstruction from price history.

Exposes first_seen_dates(price_panel) which computes, per symbol, the
min/max non-NaN price date and the count of non-NaN observations.  This
gives the point-in-time listing proxy required by the methodology guide §1.1
("reconstruct listing and delisting dates from first/last observed price").
"""

import pandas as pd


def first_seen_dates(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-symbol price coverage stats derived from non-NaN prices.

    Args:
        price_panel: Long DataFrame with columns [date, symbol, price].
            Rows where price is NaN are ignored in the min/max/count.

    Returns:
        DataFrame with columns [symbol, price_first_date, price_last_date, n_obs],
        one row per symbol that has at least one non-NaN price observation.
        Symbols where every price is NaN are excluded from the result.
    """
    observed = price_panel.dropna(subset=["price"])

    agg = (
        observed
        .groupby("symbol", sort=False)["date"]
        .agg(price_first_date="min", price_last_date="max", n_obs="count")
        .reset_index()
    )

    # Ensure date columns have datetime dtype regardless of input dtype.
    agg["price_first_date"] = pd.to_datetime(agg["price_first_date"])
    agg["price_last_date"] = pd.to_datetime(agg["price_last_date"])

    return agg[["symbol", "price_first_date", "price_last_date", "n_obs"]]
