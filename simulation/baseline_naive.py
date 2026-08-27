"""
simulation/baseline_naive.py
-----------------------------
The Naive Baseline allocation model, used purely as a comparison benchmark
against the Smart Engine (engine/allocation.py).

Rule: allocate inventory strictly in proportion to last month's raw sales
volume, with no regard for margin, stockout penalty, or capital efficiency,
subject only to the same ₹4 Crore total capital cap (so the comparison is
fair on the capital dimension).
"""

import numpy as np
import pandas as pd

DEFAULT_CAPITAL_CAP_INR = 40_000_000


def run_naive_allocation(
    sales_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    capital_cap_inr: float = DEFAULT_CAPITAL_CAP_INR,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Computes last-30-day raw sales volume per (store, sku), then allocates
    warehouse capital strictly proportional to that raw volume share --
    i.e. it does NOT distinguish margin-rich SKUs from low-margin ones, and
    does NOT apply any stockout-penalty weighting. This deliberately mirrors
    how many real-world retail chains allocate stock by "just reorder what
    sold last month."
    """
    as_of_date = sales_df["date"].max()
    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    window_df = sales_df[(sales_df["date"] > window_start) & (sales_df["date"] <= as_of_date)]

    volume = (
        window_df.groupby(["store_id", "sku_id"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "last_month_units"})
    )
    volume = volume.merge(skus_df[["sku_id", "mrp_inr", "margin_pct", "model_name", "price_tier"]],
                           on="sku_id", how="left")
    volume["unit_cost_inr"] = volume["mrp_inr"] * (1 - volume["margin_pct"])

    total_volume = volume["last_month_units"].sum()
    if total_volume <= 0:
        return pd.DataFrame()

    # Proportional share of the capital cap based purely on raw volume share
    volume["capital_share"] = volume["last_month_units"] / total_volume
    volume["capital_allocated_inr"] = volume["capital_share"] * capital_cap_inr
    volume["recommended_units"] = np.floor(
        volume["capital_allocated_inr"] / volume["unit_cost_inr"].replace(0, np.nan)
    ).fillna(0).clip(lower=0)

    volume["actual_capital_required_inr"] = volume["recommended_units"] * volume["unit_cost_inr"]
    volume["expected_net_profit_inr"] = volume["recommended_units"] * volume["mrp_inr"] * volume["margin_pct"]

    result = volume[volume["recommended_units"] > 0][[
        "store_id", "sku_id", "model_name", "price_tier",
        "recommended_units", "actual_capital_required_inr", "expected_net_profit_inr",
    ]].rename(columns={
        "store_id": "Store_ID",
        "sku_id": "SKU_ID",
        "model_name": "Model_Name",
        "price_tier": "Price_Tier",
        "recommended_units": "Recommended_Units",
        "actual_capital_required_inr": "Capital_Required_INR",
        "expected_net_profit_inr": "Expected_Net_Profit_INR",
    })

    return result.reset_index(drop=True)


if __name__ == "__main__":
    from engine.profiling import load_sales_history, load_skus

    sales = load_sales_history()
    skus = load_skus()
    naive_result = run_naive_allocation(sales, skus)
    print(f"Naive baseline allocated {len(naive_result)} lines, "
          f"₹{naive_result['Capital_Required_INR'].sum():,.0f} capital used.")
