"""
engine/lifecycle.py
--------------------
End-of-Life (EOL) Risk Handling Engine.

Detects SKUs at EOL risk (age >= 8 weeks with declining velocity, OR a
successor launch confirmed within 10-14 days) and evaluates 3 explicit
financial options for each at-risk (store, SKU) holding:

    OPTION A (HOLD):              loss from forced post-launch markdown (30%)
    OPTION B (LOCAL MARKDOWN):    net outcome of an immediate 15% discount
    OPTION C (INTER-STORE TRANSFER): net outcome of shipping to a flagship
                                    store, net of ₹/unit logistics fee

The engine recommends whichever option maximizes net capital recovery.
"""

import os
import numpy as np
import pandas as pd

from engine.profiling import (
    load_sales_history, load_stores, load_skus,
    velocity_trend, sku_age_weeks, four_week_moving_average, format_inr,
)

DEFAULT_TRANSFER_FEE_PER_UNIT_INR = 600
POST_LAUNCH_MARKDOWN_PCT = 0.30       # Option A: forced markdown after successor fully launches
LOCAL_MARKDOWN_PCT = 0.15             # Option B: proactive local discount
EOL_AGE_WEEKS_THRESHOLD = 8
DECLINING_VELOCITY_THRESHOLD_PCT = -20.0  # velocity_change_pct below this = "declining"
SUCCESSOR_WINDOW_MIN_DAYS = 10
SUCCESSOR_WINDOW_MAX_DAYS = 14


def estimate_current_holding_units(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy for "units currently held" at a store for a SKU: since we don't model
    live inventory ledgers, we approximate held stock as ~3 weeks of the
    trailing 4-week average demand (a typical reorder-point assumption).
    This is intentionally simple and can be swapped for real inventory data.
    """
    demand = four_week_moving_average(sales_df)
    demand["estimated_units_held"] = np.ceil(demand["weekly_units_4wk"] * 3).clip(lower=1)
    return demand[["store_id", "sku_id", "estimated_units_held", "weekly_units_4wk"]]


def _normalize_successor_override(successor_launch_days_override: dict):
    """
    Normalizes the override dict into two separate maps: CONFIRMED launches
    and RUMOURED launches. Accepts either:
      - {sku_id: days}                          -> treated as CONFIRMED (back-compat)
      - {sku_id: {"days": int, "confidence": "confirmed"|"rumoured"}}

    Design decision (explicitly required by the brief -- "decide how your
    system treats a rumour versus a confirmed date"):

        Only a CONFIRMED successor date can trigger an EOL-risk action
        (markdown / transfer / hold recommendation with real capital
        consequences). A RUMOURED date is surfaced on a separate WATCHLIST
        for human awareness only -- no capital action is auto-recommended
        against it.

    Rationale: rumours leak from distributors/forums well before an official
    launch and are frequently wrong on both timing and existence. Acting on
    them (e.g. markdowns or inter-store transfers) risks destroying margin
    on a SKU that may not actually be succeeded for months, or at all. A
    confirmed date (official distributor communication) is a much stronger
    signal and is the only thing allowed to move capital.
    """
    confirmed, rumoured = {}, {}
    if not successor_launch_days_override:
        return confirmed, rumoured

    for sku_id, value in successor_launch_days_override.items():
        if isinstance(value, dict):
            days = value.get("days")
            confidence = value.get("confidence", "confirmed")
        else:
            days = value
            confidence = "confirmed"

        if confidence == "rumoured":
            rumoured[sku_id] = days
        else:
            confirmed[sku_id] = days

    return confirmed, rumoured


def detect_eol_risk_skus(
    sales_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    successor_launch_days_override: dict = None,
) -> pd.DataFrame:
    """
    Flags SKUs as EOL-risk (actionable) if:
      (a) age_weeks >= 8 AND recent velocity has declined by more than
          DECLINING_VELOCITY_THRESHOLD_PCT, OR
      (b) the SKU has a successor whose launch is CONFIRMED within a
          10-14 day countdown window.

    A RUMOURED successor date never triggers is_eol_risk on its own -- see
    _normalize_successor_override for the reasoning. Rumoured SKUs are still
    surfaced separately via build_rumour_watchlist() so nothing is silently
    dropped, but no capital action is auto-recommended for them.

    successor_launch_days_override: optional
        {sku_id: days} (treated as confirmed) OR
        {sku_id: {"days": int, "confidence": "confirmed"|"rumoured"}}
        used by the Scenario Injector to simulate a sudden successor announcement.
    """
    confirmed_override, _rumoured_override = _normalize_successor_override(successor_launch_days_override)

    as_of_date = sales_df["date"].max()
    age_df = sku_age_weeks(sales_df, skus_df, as_of_date)
    velocity_df = velocity_trend(sales_df, as_of_date)

    merged = age_df.merge(velocity_df, on="sku_id", how="left")
    merged["velocity_change_pct"] = merged["velocity_change_pct"].fillna(0.0)

    merged["days_to_successor_launch"] = np.nan
    for sku_id, days in confirmed_override.items():
        merged.loc[merged["sku_id"] == sku_id, "days_to_successor_launch"] = days

    condition_age_decline = (
        (merged["age_weeks"] >= EOL_AGE_WEEKS_THRESHOLD)
        & (merged["velocity_change_pct"] <= DECLINING_VELOCITY_THRESHOLD_PCT)
    )
    condition_successor_window = (
        (merged["days_to_successor_launch"] >= SUCCESSOR_WINDOW_MIN_DAYS)
        & (merged["days_to_successor_launch"] <= SUCCESSOR_WINDOW_MAX_DAYS)
    )

    merged["is_eol_risk"] = condition_age_decline | condition_successor_window
    merged["eol_reason"] = np.select(
        [condition_successor_window, condition_age_decline],
        ["Successor launch CONFIRMED (10-14 day window)", "Aged >=8wks with declining velocity"],
        default="",
    )

    return merged[merged["is_eol_risk"]].copy()


def build_rumour_watchlist(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    successor_launch_days_override: dict = None,
) -> pd.DataFrame:
    """
    Surfaces SKUs whose successor launch is only RUMOURED (not confirmed).
    These are monitoring-only: no Hold/Markdown/Transfer action is computed
    or recommended, because acting on an unconfirmed rumour risks needlessly
    destroying margin on stock that may not actually face a successor for
    months (or ever). This is the explicit "rumour vs confirmed" policy the
    brief asks for -- rumours change what we WATCH, not what we DO.
    """
    _confirmed, rumoured_override = _normalize_successor_override(successor_launch_days_override)
    if not rumoured_override:
        return pd.DataFrame()

    holdings = estimate_current_holding_units(sales_df)
    rows = []
    for sku_id, days in rumoured_override.items():
        sku_row = skus_df[skus_df["sku_id"] == sku_id]
        if sku_row.empty:
            continue
        sku_row = sku_row.iloc[0]
        sku_holdings = holdings[holdings["sku_id"] == sku_id]
        total_units_held = int(sku_holdings["estimated_units_held"].sum())
        holding_value = total_units_held * sku_row["mrp_inr"] * (1 - sku_row["margin_pct"])

        rows.append({
            "SKU_ID": sku_id,
            "Model_Name": sku_row.get("model_name", ""),
            "Price_Tier": sku_row.get("price_tier", ""),
            "Rumoured_Days_To_Launch": days,
            "Total_Units_Held_Chainwide": total_units_held,
            "Holding_Value_At_Cost_INR": round(holding_value, 2),
            "Status": "WATCHLIST ONLY — no action recommended",
            "Note": (
                f"A successor to {sku_id} is rumoured (not confirmed) to launch in "
                f"~{days} days. {format_inr(holding_value)} of chainwide holding value is being "
                f"monitored, but no markdown or transfer is recommended until the "
                f"launch is officially confirmed, to avoid destroying margin on a "
                f"rumour that may not materialize."
            ),
        })

    return pd.DataFrame(rows)


def evaluate_eol_options(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    successor_launch_days_override: dict = None,
    transfer_fee_per_unit_inr: float = DEFAULT_TRANSFER_FEE_PER_UNIT_INR,
) -> pd.DataFrame:
    """
    For every (store, CONFIRMED at-risk SKU) holding, computes the net
    capital outcome of Hold vs. Local Markdown vs. Inter-Store Transfer, and
    recommends the option with the highest net recovery. Returns one row per
    (store, sku). Rumoured-only SKUs are excluded here -- see
    build_rumour_watchlist() for how those are surfaced instead.
    """
    at_risk_skus = detect_eol_risk_skus(sales_df, skus_df, successor_launch_days_override)
    if at_risk_skus.empty:
        return pd.DataFrame()

    holdings = estimate_current_holding_units(sales_df)
    flagship_stores = stores_df[stores_df["profile"] == "Flagship"]["store_id"].tolist()

    rows = []
    for _, sku_row in at_risk_skus.iterrows():
        sku_id = sku_row["sku_id"]
        mrp = sku_row["mrp_inr"]
        margin_pct = sku_row["margin_pct"]
        unit_cost = mrp * (1 - margin_pct)

        sku_holdings = holdings[holdings["sku_id"] == sku_id]
        for _, h in sku_holdings.iterrows():
            store_id = h["store_id"]
            units_held = int(h["estimated_units_held"])
            if units_held <= 0:
                continue

            holding_value_at_cost = units_held * unit_cost

            # OPTION A: HOLD -> eventually forced into a steeper post-launch markdown
            post_markdown_price = mrp * (1 - POST_LAUNCH_MARKDOWN_PCT)
            option_a_revenue = units_held * post_markdown_price
            option_a_net = option_a_revenue - holding_value_at_cost

            # OPTION B: LOCAL MARKDOWN NOW (smaller, proactive discount)
            local_markdown_price = mrp * (1 - LOCAL_MARKDOWN_PCT)
            option_b_revenue = units_held * local_markdown_price
            option_b_net = option_b_revenue - holding_value_at_cost

            # OPTION C: INTER-STORE TRANSFER to a flagship store (assume full
            # MRP sell-through there, net of transit fee), only viable if this
            # store is NOT itself a flagship and at least one flagship exists.
            if store_id not in flagship_stores and flagship_stores:
                transfer_cost = units_held * transfer_fee_per_unit_inr
                option_c_revenue = units_held * mrp - transfer_cost
                option_c_net = option_c_revenue - holding_value_at_cost
                option_c_available = True
            else:
                option_c_net = -np.inf
                option_c_available = False

            outcomes = {
                "HOLD": option_a_net,
                "LOCAL_MARKDOWN": option_b_net,
                "INTER_STORE_TRANSFER": option_c_net,
            }
            best_option = max(outcomes, key=outcomes.get)
            best_value = outcomes[best_option]

            transfer_line = (
                f"- Transfer: {format_inr(option_c_net)} net (₹{transfer_fee_per_unit_inr}/unit logistics fee to flagship)"
                if option_c_available else
                "- Transfer: not available (no eligible flagship destination)"
            )
            justification = (
                f"**{units_held} units of {sku_id} at {store_id} — EOL Risk**\n"
                f"- Hold: {format_inr(option_a_net)} net (forced {int(POST_LAUNCH_MARKDOWN_PCT*100)}% markdown post-launch)\n"
                f"- Local markdown: {format_inr(option_b_net)} net ({int(LOCAL_MARKDOWN_PCT*100)}% discount now)\n"
                f"{transfer_line}\n"
                f"- Recommended: **{best_option.replace('_', ' ').title()}**, recovering {format_inr(best_value)}"
            )

            rows.append({
                "Store_ID": store_id,
                "SKU_ID": sku_id,
                "Model_Name": sku_row.get("model_name", ""),
                "Price_Tier": sku_row.get("price_tier", ""),
                "Units_Held": units_held,
                "EOL_Reason": sku_row["eol_reason"],
                "Age_Weeks": round(sku_row["age_weeks"], 1),
                "Holding_Value_At_Cost_INR": round(holding_value_at_cost, 2),
                "Option_A_Hold_Net_INR": round(option_a_net, 2),
                "Option_B_Local_Markdown_Net_INR": round(option_b_net, 2),
                "Option_C_Transfer_Net_INR": round(option_c_net, 2) if option_c_available else None,
                "Recommended_Action": best_option.replace("_", " ").title(),
                "Recommended_Net_Recovery_INR": round(best_value, 2),
                "Engine_Rupee_Justification": justification,
            })

    return pd.DataFrame(rows).sort_values("Recommended_Net_Recovery_INR", ascending=True)


if __name__ == "__main__":
    sales = load_sales_history()
    stores = load_stores()
    skus = load_skus()
    eol_df = evaluate_eol_options(sales, stores, skus)
    print(f"Found {len(eol_df)} at-risk (store, SKU) holdings (confirmed successor / aged+declining).")
    if not eol_df.empty:
        print(eol_df.head(10).to_string())

    # Demo: rumour vs confirmed handling
    sample_sku = skus["sku_id"].iloc[0]
    rumoured_override = {sample_sku: {"days": 12, "confidence": "rumoured"}}
    watchlist = build_rumour_watchlist(sales, stores, skus, rumoured_override)
    print(f"\nRumoured-only watchlist (no capital action taken): {len(watchlist)} entries")
    if not watchlist.empty:
        print(watchlist.to_string())
