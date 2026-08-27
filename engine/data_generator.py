"""
engine/data_generator.py
------------------------
Generates 365 days of realistic daily sales history for the MobiMart chain:
25 stores x 60 SKUs, with store-profile demand skew, festive multipliers,
and new-SKU cannibalization of older SKUs in the same price tier.

Output: data/sales_history_12m.csv
Columns: date, store_id, store_profile, sku_id, price_tier, mrp_inr,
         units_sold, revenue_inr, is_festive_day, festival_name
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
DAYS_IN_HISTORY = 365
BASE_DATE = datetime(2025, 1, 1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_configs():
    with open(os.path.join(DATA_DIR, "stores_config.json")) as f:
        stores_cfg = json.load(f)
    with open(os.path.join(DATA_DIR, "skus_config.json")) as f:
        skus_cfg = json.load(f)
    return stores_cfg, skus_cfg


def _profile_affinity(store_profile: str, price_tier: str) -> float:
    """
    Returns a demand affinity multiplier for a given store profile / SKU price tier pairing.
    Rule (a): Flagship stores skew heavily to Flagship SKUs; Tier-3 stores skew
    90%+ of demand to Budget (<15k) SKUs.
    """
    table = {
        ("Flagship", "Flagship"): 3.2,
        ("Flagship", "Mid"): 1.1,
        ("Flagship", "Budget"): 0.25,
        ("Tier2_Hub", "Flagship"): 1.0,
        ("Tier2_Hub", "Mid"): 1.6,
        ("Tier2_Hub", "Budget"): 1.1,
        ("Tier3_Volume", "Flagship"): 0.05,
        ("Tier3_Volume", "Mid"): 0.35,
        ("Tier3_Volume", "Budget"): 3.6,
    }
    return table[(store_profile, price_tier)]


def _festive_multiplier(day_of_year: int, festive_windows: dict, rng: np.random.Generator):
    """Rule (b): 3.0x-4.0x multiplier during festive windows."""
    for name, window in festive_windows.items():
        if window["start_day_of_year"] <= day_of_year <= window["end_day_of_year"]:
            lo, hi = window["multiplier_range"]
            return rng.uniform(lo, hi), name
    return 1.0, None


def generate_sales_history(save_csv: bool = True) -> pd.DataFrame:
    """
    Main entry point. Builds the full daily sales history DataFrame and
    optionally writes it to data/sales_history_12m.csv.
    """
    rng = np.random.default_rng(RNG_SEED)
    stores_cfg, skus_cfg = load_configs()
    stores = stores_cfg["stores"]
    skus = skus_cfg["skus"]
    festive_windows = stores_cfg["festive_windows"]

    # Base daily demand (units) per SKU price tier, before affinity/festive adjustment
    base_demand_by_tier = {"Flagship": 0.35, "Mid": 0.9, "Budget": 1.6}

    records = []

    # Precompute successor map for cannibalization: old_sku -> successor's launch day.
    # Cannibalization must trigger when the SUCCESSOR launches, not when the old
    # SKU itself launched -- so we look up the successor's own launch_day_of_year.
    sku_by_id = {s["sku_id"]: s for s in skus}
    successor_effective_day = {}
    for s in skus:
        successor_id = s.get("successor_sku_id")
        if successor_id:
            successor_effective_day[s["sku_id"]] = sku_by_id[successor_id]["launch_day_of_year"]

    for day_offset in range(DAYS_IN_HISTORY):
        current_date = BASE_DATE + timedelta(days=day_offset)
        day_of_year = current_date.timetuple().tm_yday
        festive_mult, festival_name = _festive_multiplier(day_of_year, festive_windows, rng)
        is_festive = festival_name is not None

        for store in stores:
            store_profile = store["profile"]
            for sku in skus:
                price_tier = sku["price_tier"]

                # SKU must have "launched" to have any sales
                if day_of_year < sku["launch_day_of_year"]:
                    continue

                affinity = _profile_affinity(store_profile, price_tier)
                base = base_demand_by_tier[price_tier]

                # Rule (c): Cannibalization - if this SKU has a successor and the
                # successor has launched, apply 60-80% velocity drop.
                cannibalization_factor = 1.0
                if sku["sku_id"] in successor_effective_day:
                    succ_launch_day = successor_effective_day[sku["sku_id"]]
                    if day_of_year >= succ_launch_day:
                        cannibalization_factor = 1.0 - rng.uniform(0.60, 0.80)

                noise = rng.gamma(shape=4.0, scale=0.25)  # mean ~1.0, right-skewed realism
                expected_units = base * affinity * festive_mult * cannibalization_factor * noise

                units_sold = rng.poisson(lam=max(expected_units, 0.01))
                if units_sold <= 0:
                    continue

                revenue = units_sold * sku["mrp_inr"]

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "store_profile": store_profile,
                    "sku_id": sku["sku_id"],
                    "price_tier": price_tier,
                    "mrp_inr": sku["mrp_inr"],
                    "units_sold": int(units_sold),
                    "revenue_inr": float(revenue),
                    "is_festive_day": is_festive,
                    "festival_name": festival_name if festival_name else "",
                })

    df = pd.DataFrame.from_records(records)

    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, "sales_history_12m.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df):,} rows to {out_path}")

    return df


if __name__ == "__main__":
    generate_sales_history()
