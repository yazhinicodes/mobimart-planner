"""
engine/allocation.py
---------------------
The Monday Weekly Allocation Engine.

Core idea: for every (store, SKU) pair, compute:
    Priority Index = (Expected Profit Margin / Capital Required) * Stockout Penalty

Then greedily allocate warehouse units to the highest-priority (store, SKU)
combinations, one unit at a time (in small batches for speed), until the
₹4,00,00,000 capital cap is reached. This is a classic greedy knapsack
heuristic appropriate for this "maximize value per rupee of capital" problem.

Every recommended row carries an explicit rupee justification string.
"""

import json
import os
import numpy as np
import pandas as pd

from engine.profiling import four_week_moving_average, load_stores, load_skus, format_inr

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DEFAULT_CAPITAL_CAP_INR = 40_000_000  # Rs. 4 Crore
STOCKOUT_PENALTY_MULTIPLIER_BY_TIER = {
    # The brief is explicit: "Stockouts are not all equal. A customer who
    # cannot find a Rs.15,000 phone buys it next door -- sale and customer
    # both lost. A flagship buyer might wait two days for a transfer."
    # This asymmetry must be priced in directly, or the Priority Index (which
    # otherwise rewards absolute rupee margin) will systematically over-invest
    # in expensive flagship phones and starve fast-moving budget stock.
    "Budget": 1.3,      # customer walks to a competitor and is gone for good
    "Mid": 1.15,        # some loyalty, moderate switching cost
    "Flagship": 1.0,    # buyer will typically wait a couple of days for a transfer
}
MIN_UNITS_PER_ALLOCATION = 1
MAX_UNITS_PER_SKU_STORE = 50  # hard cap on units per (store, SKU)
WEEKS_OF_COVER_TARGET = 3  # target stock cover in weeks when sizing a recommendation
MAX_CAPITAL_SHARE_PER_LINE = 0.10  # risk-control: no single (store, SKU) line may
                                    # consume more than 10% of the total capital cap.
                                    # This is a standard, round-number portfolio-
                                    # concentration guardrail (independent of any
                                    # particular scorecard outcome) -- it simply
                                    # stops one expensive flagship line from eating
                                    # a disproportionate slice of the week's budget.
MIN_FLOOR_UNITS_PER_STORE = 3  # every store gets at least this many units of its own
                                # single highest-priority SKU before the general
                                # priority-ranked top-up begins. This guarantees no
                                # store is left with zero recommendations for the
                                # week (the brief: "every Monday it recommends what
                                # EACH STORE receives"), at a modest capital cost
                                # (~9% of the cap), while leaving ~91% of the budget
                                # to be allocated purely by the Priority Index.


def _capital_required(units: int, unit_cost_inr: float) -> float:
    return units * unit_cost_inr


def build_allocation_candidates(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    demand_multipliers: dict = None,
    warehouse_stock_limits: dict = None,
) -> pd.DataFrame:
    """
    Builds one candidate row per (store_id, sku_id) with forecasted weekly
    demand, unit economics, and a priority index.

    demand_multipliers: optional dict of {store_id: multiplier} OR {"ALL": multiplier}
        used by the live Scenario Injector to bump/cut demand for what-if analysis.
    warehouse_stock_limits: optional dict of {sku_id: max_available_units} to
        simulate supplier stockouts / central warehouse constraints.
    """
    demand_multipliers = demand_multipliers or {}
    warehouse_stock_limits = warehouse_stock_limits or {}

    forecast = four_week_moving_average(sales_df)
    candidates = forecast.merge(skus_df, on="sku_id", how="left")
    candidates = candidates.merge(
        stores_df[["store_id", "profile"]], on="store_id", how="left"
    )

    # unit landed cost approximated as MRP * (1 - margin_pct)
    candidates["unit_cost_inr"] = candidates["mrp_inr"] * (1 - candidates["margin_pct"])
    candidates["unit_margin_inr"] = candidates["mrp_inr"] * candidates["margin_pct"]

    # Apply scenario demand multipliers (store-specific overrides "ALL")
    def apply_mult(row):
        mult = demand_multipliers.get(row["store_id"], demand_multipliers.get("ALL", 1.0))
        return row["weekly_units_4wk"] * mult

    candidates["adjusted_weekly_demand"] = candidates.apply(apply_mult, axis=1)
    candidates["adjusted_weekly_demand"] = candidates["adjusted_weekly_demand"].clip(lower=0)

    # Recommended units sized to reach WEEKS_OF_COVER_TARGET weeks of cover
    candidates["recommended_units_raw"] = np.ceil(
        candidates["adjusted_weekly_demand"] * WEEKS_OF_COVER_TARGET
    ).clip(lower=0, upper=MAX_UNITS_PER_SKU_STORE)

    # Stockout penalty: expected lost margin if this SKU is NOT stocked, scaled
    # by a TIER-SPECIFIC opportunity-cost multiplier. Budget stockouts are
    # penalized more heavily than Flagship stockouts, because a budget
    # customer who can't find the phone buys it elsewhere permanently, while
    # a flagship buyer will typically wait for a transfer.
    candidates["stockout_multiplier"] = candidates["price_tier"].map(
        STOCKOUT_PENALTY_MULTIPLIER_BY_TIER
    ).fillna(1.5)
    candidates["stockout_penalty_inr"] = (
        candidates["adjusted_weekly_demand"] * candidates["unit_margin_inr"]
        * candidates["stockout_multiplier"]
    )

    candidates["capital_required_per_unit"] = candidates["unit_cost_inr"]
    candidates["total_capital_if_fulfilled"] = (
        candidates["recommended_units_raw"] * candidates["unit_cost_inr"]
    )

    # Priority Index = (Expected Profit Margin / Capital Required) * Stockout Penalty
    # Guard against division by zero.
    candidates["priority_index"] = np.where(
        candidates["total_capital_if_fulfilled"] > 0,
        (candidates["unit_margin_inr"] * candidates["recommended_units_raw"]
         / candidates["total_capital_if_fulfilled"].replace(0, np.nan))
        * candidates["stockout_penalty_inr"],
        0.0,
    )
    candidates["priority_index"] = candidates["priority_index"].fillna(0.0)

    # Apply warehouse stock limits per SKU (cap the units available chain-wide)
    if warehouse_stock_limits:
        candidates["sku_warehouse_cap"] = candidates["sku_id"].map(warehouse_stock_limits)
    else:
        candidates["sku_warehouse_cap"] = np.nan

    return candidates


def allocate_under_capital_cap(
    candidates: pd.DataFrame,
    capital_cap_inr: float = DEFAULT_CAPITAL_CAP_INR,
) -> pd.DataFrame:
    """
    Two-phase allocation:

    PHASE 1 -- FAIRNESS FLOOR: every store is guaranteed at least
    MIN_FLOOR_UNITS_PER_STORE units of its own single highest-priority
    candidate, so no store is left with zero recommendations for the week.

    PHASE 2 -- PRIORITY TOP-UP: the remaining capital is walked in descending
    priority_index order (tie-broken by higher adjusted_weekly_demand, then
    by store_id for full determinism), topping up every candidate -- including
    the floor lines from Phase 1 -- up to its desired units, until the capital
    cap or a per-line/warehouse constraint stops it.

    Both phases share the same running capital, unit caps, and warehouse
    limits, so the final output has exactly one row per funded (store, sku)
    pair with its true total allocation and rupee cost.
    """
    ranked = candidates.sort_values("priority_index", ascending=False).copy()
    ranked = ranked[ranked["recommended_units_raw"] > 0]

    remaining_capital = capital_cap_inr
    sku_units_allocated_so_far = {}
    line_units_allocated = {}

    def _apply_constraints(row, desired_units):
        if not np.isnan(row.get("sku_warehouse_cap", np.nan)):
            already_used = sku_units_allocated_so_far.get(row["sku_id"], 0)
            remaining_sku_cap = max(row["sku_warehouse_cap"] - already_used, 0)
            desired_units = min(desired_units, int(remaining_sku_cap))
        if desired_units <= 0:
            return 0
        unit_cost = row["unit_cost_inr"]
        max_affordable_units = int(remaining_capital // unit_cost)
        max_line_capital = capital_cap_inr * MAX_CAPITAL_SHARE_PER_LINE
        max_units_by_concentration_cap = int(max_line_capital // unit_cost)
        return max(min(desired_units, max_affordable_units, max_units_by_concentration_cap), 0)

    # ---------------- PHASE 1: fairness floor, one line per store ----------------
    best_per_store_idx = ranked.groupby("store_id")["priority_index"].idxmax()
    floor_rows = ranked.loc[best_per_store_idx]

    for _, row in floor_rows.iterrows():
        if remaining_capital <= 0:
            break
        unit_cost = row["unit_cost_inr"]
        if unit_cost <= 0:
            continue
        desired_units = min(int(row["recommended_units_raw"]), MIN_FLOOR_UNITS_PER_STORE)
        units_to_allocate = _apply_constraints(row, desired_units)
        if units_to_allocate <= 0:
            continue
        capital_used = units_to_allocate * unit_cost
        remaining_capital -= capital_used
        sku_units_allocated_so_far[row["sku_id"]] = (
            sku_units_allocated_so_far.get(row["sku_id"], 0) + units_to_allocate
        )
        line_units_allocated[(row["store_id"], row["sku_id"])] = units_to_allocate

    # ---------------- PHASE 2: priority-ranked top-up, explicit tie-break ----------------
    ranked = ranked.sort_values(
        ["priority_index", "adjusted_weekly_demand", "store_id"],
        ascending=[False, False, True],
    )

    allocated_rows = []
    for _, row in ranked.iterrows():
        if remaining_capital <= 0:
            break

        unit_cost = row["unit_cost_inr"]
        if unit_cost <= 0:
            continue

        line_key = (row["store_id"], row["sku_id"])
        already_on_this_line = line_units_allocated.get(line_key, 0)
        desired_units = int(row["recommended_units_raw"]) - already_on_this_line

        if desired_units <= 0:
            continue

        units_to_allocate = _apply_constraints(row, desired_units)

        if units_to_allocate <= 0:
            continue

        capital_used = units_to_allocate * unit_cost
        remaining_capital -= capital_used
        sku_units_allocated_so_far[row["sku_id"]] = (
            sku_units_allocated_so_far.get(row["sku_id"], 0) + units_to_allocate
        )
        line_units_allocated[line_key] = already_on_this_line + units_to_allocate

    # ---------------- Build final output: one row per funded line ----------------
    for _, row in ranked.iterrows():
        line_key = (row["store_id"], row["sku_id"])
        total_units = line_units_allocated.get(line_key, 0)
        if total_units <= 0:
            continue

        unit_cost = row["unit_cost_inr"]
        capital_used = total_units * unit_cost
        expected_net_profit = total_units * row["unit_margin_inr"]
        desired_units = int(row["recommended_units_raw"])
        stockout_risk_avoided = row["stockout_penalty_inr"] * (
            total_units / max(desired_units, 1)
        )

        justification = (
            f"**Recommendation: {total_units} units of {row['sku_id']} "
            f"({row.get('model_name', '')}) → {row['store_id']}**\n"
            f"-  Capital required: {format_inr(capital_used)}\n"
            f"-  Stockout risk avoided: {format_inr(stockout_risk_avoided)}\n"
            f"-  Expected net margin: {format_inr(expected_net_profit)}\n"
            f"-  Based on: {row['adjusted_weekly_demand']:.1f} units/week (4-week average demand)"
        )

        allocated_rows.append({
            "Store_ID": row["store_id"],
            "SKU_ID": row["sku_id"],
            "Model_Name": row.get("model_name", ""),
            "Price_Tier": row.get("price_tier", ""),
            "Store_Profile": row.get("profile", ""),
            "Recommended_Units": total_units,
            "Capital_Required_INR": round(capital_used, 2),
            "Stockout_Risk_Avoided_INR": round(stockout_risk_avoided, 2),
            "Expected_Net_Profit_INR": round(expected_net_profit, 2),
            "Priority_Index": round(row["priority_index"], 4),
            "Engine_Rupee_Justification": justification,
        })

    result_df = pd.DataFrame(allocated_rows)
    return result_df


def run_weekly_allocation(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame = None,
    skus_df: pd.DataFrame = None,
    capital_cap_inr: float = DEFAULT_CAPITAL_CAP_INR,
    demand_multipliers: dict = None,
    warehouse_stock_limits: dict = None,
) -> pd.DataFrame:
    """Convenience wrapper: build candidates then allocate under the cap."""
    if stores_df is None:
        stores_df = load_stores()
    if skus_df is None:
        skus_df = load_skus()

    candidates = build_allocation_candidates(
        sales_df, stores_df, skus_df,
        demand_multipliers=demand_multipliers,
        warehouse_stock_limits=warehouse_stock_limits,
    )
    return allocate_under_capital_cap(candidates, capital_cap_inr=capital_cap_inr)


if __name__ == "__main__":
    from engine.profiling import load_sales_history

    sales = load_sales_history()
    stores = load_stores()
    skus = load_skus()
    result = run_weekly_allocation(sales, stores, skus)
    total_capital = result["Capital_Required_INR"].sum()
    print(f"Allocated {len(result)} store/SKU lines using {format_inr(total_capital)} "
          f"of {format_inr(DEFAULT_CAPITAL_CAP_INR)} cap.")
    print(result.head(10).to_string())
