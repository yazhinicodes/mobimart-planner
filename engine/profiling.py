"""
engine/profiling.py
--------------------
Shared analytics utilities used by both the allocation engine and the EOL
lifecycle engine: 4-week moving average demand, velocity trend detection,
and current SKU age-in-weeks.
"""

import json
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def format_inr(amount) -> str:
    """Formats a rupee amount the way an Indian owner actually reads money --
    lakh/crore, not western comma-grouping -- matching the brief's own
    language ('₹4 crore', '₹15 lakh'). Shared by the engine's own generated
    rupee-justification text and the dashboard display layer, so a number
    reads the same way everywhere in the app."""
    amount = float(amount)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_00_00_000:
        return f"{sign}₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"{sign}₹{amount / 1_00_000:.2f} L"
    else:
        return f"{sign}₹{amount:,.0f}"

def load_sales_history() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "sales_history_12m.csv")
    df = pd.read_csv(path, parse_dates=["date"], dtype={"festival_name": "string"})
    return df


def load_stores() -> pd.DataFrame:
    with open(os.path.join(DATA_DIR, "stores_config.json")) as f:
        cfg = json.load(f)
    return pd.DataFrame(cfg["stores"])


def load_skus() -> pd.DataFrame:
    with open(os.path.join(DATA_DIR, "skus_config.json")) as f:
        cfg = json.load(f)
    return pd.DataFrame(cfg["skus"])


def four_week_moving_average(sales_df: pd.DataFrame, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Computes 4-week (28-day) trailing average DAILY demand per store_id/sku_id,
    as of a given reference date (defaults to the max date in the dataset).

    Returns a DataFrame: store_id, sku_id, avg_daily_units_4wk, weekly_units_4wk
    """
    if as_of_date is None:
        as_of_date = sales_df["date"].max()
    window_start = as_of_date - pd.Timedelta(days=28)

    window_df = sales_df[(sales_df["date"] > window_start) & (sales_df["date"] <= as_of_date)]

    grouped = (
        window_df.groupby(["store_id", "sku_id"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "total_units_4wk"})
    )
    grouped["avg_daily_units_4wk"] = grouped["total_units_4wk"] / 28.0
    grouped["weekly_units_4wk"] = grouped["avg_daily_units_4wk"] * 7.0
    return grouped


def velocity_trend(sales_df: pd.DataFrame, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compares the most recent 2-week velocity against the prior 2-week velocity
    (within the trailing 4-week window) to flag declining SKUs.

    Returns: store_id, sku_id, recent_2wk_units, prior_2wk_units, velocity_change_pct
    """
    if as_of_date is None:
        as_of_date = sales_df["date"].max()

    recent_start = as_of_date - pd.Timedelta(days=14)
    prior_start = as_of_date - pd.Timedelta(days=28)

    recent = sales_df[(sales_df["date"] > recent_start) & (sales_df["date"] <= as_of_date)]
    prior = sales_df[(sales_df["date"] > prior_start) & (sales_df["date"] <= recent_start)]

    recent_g = recent.groupby("sku_id")["units_sold"].sum().rename("recent_2wk_units")
    prior_g = prior.groupby("sku_id")["units_sold"].sum().rename("prior_2wk_units")

    merged = pd.concat([recent_g, prior_g], axis=1).fillna(0).reset_index()
    merged["velocity_change_pct"] = np.where(
        merged["prior_2wk_units"] > 0,
        (merged["recent_2wk_units"] - merged["prior_2wk_units"]) / merged["prior_2wk_units"] * 100.0,
        0.0,
    )
    return merged


def sku_age_weeks(sales_df: pd.DataFrame, skus_df: pd.DataFrame, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Computes each SKU's age in weeks since its first observed sale date,
    used as a proxy for time-since-launch.
    """
    if as_of_date is None:
        as_of_date = sales_df["date"].max()

    first_sale = sales_df.groupby("sku_id")["date"].min().rename("first_sale_date").reset_index()
    first_sale["age_weeks"] = (as_of_date - first_sale["first_sale_date"]).dt.days / 7.0

    result = skus_df.merge(first_sale, on="sku_id", how="left")
    result["age_weeks"] = result["age_weeks"].fillna(0)
    return result
