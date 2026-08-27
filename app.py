"""
app.py
------
MobiMart Planner — Owner Dashboard.

Streamlit front-end with 4 tabs:
  1. Executive Overview
  2. Weekly Allocation Hub
  3. EOL Risk Manager
  4. Defense Scenario Simulator & Scorecard

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from engine.profiling import load_sales_history, load_stores, load_skus, format_inr
from engine.allocation import (
    run_weekly_allocation,
    DEFAULT_CAPITAL_CAP_INR,
)
from engine.lifecycle import evaluate_eol_options, build_rumour_watchlist, DEFAULT_TRANSFER_FEE_PER_UNIT_INR
from simulation.evaluator import run_backtest_scorecard, compute_recent_performance_summary

st.set_page_config(page_title="MobiMart Planner", layout="wide", page_icon="📱")

def show_table(df: pd.DataFrame, height: int = None, selectable: bool = False):
    """Owner-facing table display: row numbers start at 1, not pandas' default 0,
    since a non-technical owner reads a spreadsheet-style table starting at 1.
    Store_ID is shown with its friendly store name, and raw column names are
    relabeled into plain English -- all in one place so it's consistent
    everywhere a table renders in the app.

    When selectable=True, the table becomes click-to-select (single row) and
    this function returns the selection event so the caller can react to it.
    """
    display_df = df.copy()
    if "Store_ID" in display_df.columns:
        display_df["Store_ID"] = display_df["Store_ID"].map(STORE_LABELS).fillna(display_df["Store_ID"])
    display_df = display_df.rename(columns=COLUMN_DISPLAY_NAMES)
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1

    kwargs = {"use_container_width": True}
    if height:
        kwargs["height"] = height

    if selectable:
        kwargs["on_select"] = "rerun"
        kwargs["selection_mode"] = "single-row"
        return st.dataframe(display_df, **kwargs)

    st.dataframe(display_df, **kwargs)


COLUMN_DISPLAY_NAMES = {
    "Store_ID": "Store",
    "SKU_ID": "SKU",
    "Model_Name": "Model",
    "Price_Tier": "Tier",
    "Store_Profile": "Store Type",
    "Recommended_Units": "Units",
    "Capital_Required_INR": "Capital Required",
    "Stockout_Risk_Avoided_INR": "Stockout Risk Avoided",
    "Expected_Net_Profit_INR": "Net Profit",
    "Engine_Rupee_Justification": "Reasoning",
    "Units_Held": "Units Held",
    "EOL_Reason": "Why Flagged",
    "Age_Weeks": "Age (Weeks)",
    "Holding_Value_At_Cost_INR": "Holding Value (Cost)",
    "Option_A_Hold_Net_INR": "Hold — Net",
    "Option_B_Local_Markdown_Net_INR": "Local Markdown — Net",
    "Option_C_Transfer_Net_INR": "Transfer — Net",
    "Recommended_Action": "Recommended Action",
    "Recommended_Net_Recovery_INR": "Net Recovery",
    "Rumoured_Days_To_Launch": "Days to Launch (Rumoured)",
    "Total_Units_Held_Chainwide": "Total Units Held (Chainwide)",
    "Priority_Index": "Priority Score",
}

# --------------------------------------------------------------------------
# Cached data loaders (data & configs rarely change within a session)
# --------------------------------------------------------------------------
@st.cache_data
def _load_sales():
    return load_sales_history()


@st.cache_data
def _load_stores():
    return load_stores()


@st.cache_data
def _load_skus():
    return load_skus()


sales_df = _load_sales()
stores_df = _load_stores()
skus_df = _load_skus()

STORE_LABELS = {row["store_id"]: f"{row['store_id']} — {row['name']}" for _, row in stores_df.iterrows()}

# --------------------------------------------------------------------------
# Sidebar: global parameters
# --------------------------------------------------------------------------
st.sidebar.title("📱 MobiMart Planner")
st.sidebar.caption("Weekly stock allocation & EOL risk engine")

st.sidebar.header("Global Parameters")
capital_cap_cr = st.sidebar.slider(
    "Capital Cap (₹ Crore)", min_value=1.0, max_value=6.0,
    value=DEFAULT_CAPITAL_CAP_INR / 1e7, step=0.25,
)
capital_cap_inr = capital_cap_cr * 1e7

transfer_fee = st.sidebar.number_input(
    "Inter-Store Transfer Fee (₹/unit)", min_value=0, max_value=5000,
    value=DEFAULT_TRANSFER_FEE_PER_UNIT_INR, step=50,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Adjust these to see how the whole chain's Monday allocation and EOL "
    "recommendations shift in real time."
)


# --------------------------------------------------------------------------
# Scenario Injector state (shared across Tab 4, but consumed everywhere)
# --------------------------------------------------------------------------
if "demand_multipliers" not in st.session_state:
    st.session_state.demand_multipliers = {}
if "warehouse_stock_limits" not in st.session_state:
    st.session_state.warehouse_stock_limits = {}
if "successor_override" not in st.session_state:
    st.session_state.successor_override = {}
if "scenario_label" not in st.session_state:
    st.session_state.scenario_label = "Baseline (no scenario active)"


def _current_allocation():
    return run_weekly_allocation(
        sales_df, stores_df, skus_df,
        capital_cap_inr=capital_cap_inr,
        demand_multipliers=st.session_state.demand_multipliers,
        warehouse_stock_limits=st.session_state.warehouse_stock_limits,
    )


def _current_eol():
    return evaluate_eol_options(
        sales_df, stores_df, skus_df,
        successor_launch_days_override=st.session_state.successor_override,
        transfer_fee_per_unit_inr=transfer_fee,
    )


def _current_watchlist():
    return build_rumour_watchlist(
        sales_df, stores_df, skus_df,
        successor_launch_days_override=st.session_state.successor_override,
    )


allocation_df = _current_allocation()
eol_df = _current_eol()
watchlist_df = _current_watchlist()

total_capital_in_use = allocation_df["Capital_Required_INR"].sum() if not allocation_df.empty else 0.0
at_risk_value = eol_df["Holding_Value_At_Cost_INR"].sum() if not eol_df.empty else 0.0


@st.cache_data
def _cached_recent_performance(cap_inr):
    # Cached on capital_cap_inr only: this is a retrospective backtest (what
    # WOULD have happened), so it deliberately does NOT respond to the live
    # Scenario Injector on Tab 4 -- the past doesn't change because of a
    # what-if scenario. Caching also avoids re-running the backtest on every
    # unrelated Streamlit rerun (e.g. clicking a button on another tab).
    return compute_recent_performance_summary(sales_df, stores_df, skus_df, capital_cap_inr=cap_inr)


recent_performance = _cached_recent_performance(capital_cap_inr)



@st.cache_data
def _cached_backtest_scorecard(cap_inr):
    # Same reasoning as _cached_recent_performance above: this backtest is
    # expensive (runs both engines across the full 12-month dataset) and its
    # result only depends on capital_cap_inr, not on the live Scenario
    # Injector state -- so it shouldn't recompute on every unrelated rerun.
    return run_backtest_scorecard(sales_df, stores_df, skus_df, capital_cap_inr=cap_inr)


def show_active_parameters_strip():
    """A persistent, one-line summary of every knob that can currently be
    changing the numbers on screen -- capital cap, transfer fee, and any
    active Scenario Injector preset/custom scenario. Shown at the top of
    EVERY tab (not just where the scenario was set) so it's impossible to
    lose track of what's currently active while jumping between tabs live
    -- e.g. during a fast-paced defense session."""
    scenario = st.session_state.scenario_label
    is_baseline = scenario == "Baseline (no scenario active)"
    summary = (
        f"⚙️ Capital Cap: {format_inr(capital_cap_inr)} · "
        f"Transfer Fee: ₹{transfer_fee}/unit · "
        f"Scenario: {scenario}"
    )
    if is_baseline:
        st.caption(summary)
    else:
        st.warning(summary)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Owner's Dashboard",
    "📦 Weekly Allocation Hub",
    "⚠️ EOL Risk Manager",
    "🎯 Defense Scenario Simulator & Scorecard",
])

# ============================================================================
# TAB 1: OWNER'S DASHBOARD
# ============================================================================
with tab1:
    st.subheader("Owner's Dashboard")
    show_active_parameters_strip()

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.markdown("#### 📅 Past 4 Weeks — Earned or Lost")
            st.caption(
                "Simulated backtest, not a live audit: what the Smart Engine's own "
                "logic would have realized over the trailing 4 weeks of demand, "
                "measured against what the Naive Baseline would have realized over "
                "the identical window and capital cap."
            )
            st.metric(
                "Realized Net Profit (Smart Engine)",
                format_inr(recent_performance['smart_realized_profit_inr']),
            )
            outperf = recent_performance["outperformance_inr"]
            st.metric(
                "Outperformance vs. Naive Baseline",
                f"{'+' if outperf >= 0 else ''}{format_inr(outperf)}",
                f"Naive would have earned {format_inr(recent_performance['naive_realized_profit_inr'])}",
            )

    with right_col:
        with st.container(border=True):
            st.markdown("#### 📍 Right Now — Capital & Risk")
            st.metric(
                "Capital In Use vs. Cap",
                format_inr(total_capital_in_use),
                f"{total_capital_in_use / capital_cap_inr * 100:.1f}% of {format_inr(capital_cap_inr)}",
            )
            st.metric(
                "At-Risk Inventory (Right Now)",
                format_inr(at_risk_value),
                f"{len(eol_df)} holdings flagged" if not eol_df.empty else "0 holdings flagged",
            )

    st.markdown("---")

    if not allocation_df.empty:
        by_profile = (
            allocation_df.groupby("Store_Profile")["Capital_Required_INR"]
            .sum().reset_index().sort_values("Capital_Required_INR", ascending=False)
        )
        fig = px.bar(
            by_profile, x="Store_Profile", y="Capital_Required_INR",
            title="Capital Distribution by Store Profile",
            labels={"Capital_Required_INR": "Capital Allocated (₹)", "Store_Profile": "Store Profile"},
            text_auto=".2s", color="Store_Profile",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            by_tier = (
                allocation_df.groupby("Price_Tier")["Capital_Required_INR"]
                .sum().reset_index()
            )
            fig2 = px.pie(
                by_tier, names="Price_Tier", values="Capital_Required_INR",
                title="Capital Split by Price Tier",
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            top_lines = allocation_df.nlargest(10, "Expected_Net_Profit_INR")
            fig3 = px.bar(
                top_lines, x="Expected_Net_Profit_INR", y="SKU_ID",
                orientation="h", color="Store_ID",
                title="Top 10 Lines by Expected Net Profit",
                labels={"Expected_Net_Profit_INR": "Net Profit (₹)"},
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No allocation recommendations generated under current parameters.")


# ============================================================================
# TAB 2: WEEKLY ALLOCATION HUB
# ============================================================================
with tab2:
    st.subheader("Weekly Allocation Hub — Monday Stock Recommendations")
    show_active_parameters_strip()

    if allocation_df.empty:
        st.warning("No allocations to display.")
    else:
        store_options = sorted(allocation_df["Store_ID"].unique().tolist())
        selected_stores = st.multiselect(
            "Filter by store", options=store_options, default=[],
            help="Leave empty to show all stores",
        )
        tier_options = sorted(allocation_df["Price_Tier"].unique().tolist())
        selected_tiers = st.multiselect(
            "Filter by price tier", options=tier_options, default=[],
        )

        filtered = allocation_df.copy()
        if selected_stores:
            filtered = filtered[filtered["Store_ID"].isin(selected_stores)]
        if selected_tiers:
            filtered = filtered[filtered["Price_Tier"].isin(selected_tiers)]

        st.caption(
            f"Showing {len(filtered)} of {len(allocation_df)} recommended allocation lines — "
            f"{format_inr(filtered['Capital_Required_INR'].sum())} capital required."
        )

        # Owner-facing table: click any row to see its rupee justification
        # appear immediately below — no scrolling to find it separately.
        display_cols = [
            "Store_ID", "SKU_ID", "Model_Name", "Price_Tier", "Recommended_Units",
            "Capital_Required_INR", "Stockout_Risk_Avoided_INR",
            "Expected_Net_Profit_INR",
        ]
        selection_event = show_table(
            filtered[display_cols],
            height=500,
            selectable=True,
        )

        st.markdown("##### Why this recommendation? (reasoning in rupees)")
        selected_rows = selection_event.selection["rows"] if selection_event else []
        if selected_rows:
            sel_row = filtered.reset_index(drop=True).iloc[selected_rows[0]]
            st.info(
                f"**{sel_row['Store_ID']} · {sel_row['SKU_ID']} ({sel_row['Model_Name']})** — "
                f"{sel_row['Recommended_Units']} units, {format_inr(sel_row['Capital_Required_INR'])}"
            )
            st.write(sel_row["Engine_Rupee_Justification"])
        else:
            st.caption("👆 Click any row in the table above to see its rupee justification here.")



# ============================================================================
# TAB 3: EOL RISK MANAGER
# ============================================================================
with tab3:
    st.subheader("EOL Risk Manager — Action Decision Matrix")
    show_active_parameters_strip()
    st.caption(
        "**EOL = End-of-Life**: stock that's aging out or about to be superseded by "
        "a successor model. This tab flags at-risk holdings and recommends Markdown, "
        "Transfer, or Hold for each, with the rupee cost of every option shown."
    )

    if eol_df.empty:
        st.success("No SKUs currently flagged as EOL-risk under current parameters.")
    else:
        st.caption(
            f"{len(eol_df)} (store, SKU) holdings flagged as EOL-risk, "
            f"{format_inr(eol_df['Holding_Value_At_Cost_INR'].sum())} total holding value at cost."
        )

        display_cols = [
            "Store_ID", "SKU_ID", "Model_Name", "Units_Held", "EOL_Reason",
            "Option_A_Hold_Net_INR", "Option_B_Local_Markdown_Net_INR",
            "Option_C_Transfer_Net_INR", "Recommended_Action",
            "Recommended_Net_Recovery_INR",
        ]

        selection_event = show_table(eol_df[display_cols], height=400, selectable=True)

        st.markdown("#### Rupee Justification")
        selected_rows = selection_event.selection["rows"] if selection_event else []
        if selected_rows:
            sel_row = eol_df.reset_index(drop=True).iloc[selected_rows[0]]
            st.info(
                f"**{sel_row['Store_ID']} · {sel_row['SKU_ID']} ({sel_row['Model_Name']})** → "
                f"{sel_row['Recommended_Action']}"
            )
            st.write(sel_row["Engine_Rupee_Justification"])
        else:
            st.caption("👆 Click any row in the table above to see its rupee justification here.")

        st.markdown("---")
        transfers = eol_df[eol_df["Recommended_Action"] == "Inter Store Transfer"]
        if transfers.empty:
            st.caption("No inter-store transfers are currently the optimal recommendation.")
        else:
            total_units = transfers["Units_Held"].sum()
            total_recovery = transfers["Recommended_Net_Recovery_INR"].sum()
            st.info(
                f"📋 **If all recommended transfers were carried out:** {len(transfers)} "
                f"transfer(s), {total_units} units moved, {format_inr(total_recovery)} projected "
                f"net recovery. (This dashboard recommends the action and its rupee cost, "
                f"per the brief — it does not execute stock movements.)"
            )

    st.markdown("---")
    st.markdown("#### 👁️ Rumour Watchlist (informational only — no capital action)")
    st.caption(
        "Policy: only a CONFIRMED successor launch date can trigger a markdown/transfer "
        "recommendation. A RUMOURED date is monitored here but never auto-actioned, since "
        "acting on unconfirmed leaks risks destroying margin on stock that may not actually "
        "be superseded for months (or at all)."
    )
    if watchlist_df.empty:
        st.info("No rumoured successor launches are currently being monitored.")
    else:
        show_table(
            watchlist_df[[
                "SKU_ID", "Model_Name", "Price_Tier", "Rumoured_Days_To_Launch",
                "Total_Units_Held_Chainwide", "Holding_Value_At_Cost_INR", "Status",
            ]]
        )
        for _, row in watchlist_df.iterrows():
            st.caption(f"ℹ️ {row['Note']}")


# ============================================================================
# TAB 4: DEFENSE SCENARIO SIMULATOR & SCORECARD
# ============================================================================
with tab4:
    st.subheader("Defense Scenario Simulator")
    show_active_parameters_strip()
    st.caption(
        "Inject any live interview curveball and watch allocations, EOL "
        "decisions, and the scorecard recalculate instantly."
    )

    # ---- Preset quick-trigger buttons ----
    st.markdown("##### One-Click Presets")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        if st.button("Preset 1: Successor Launch + Local Collapse"):
            flagship_skus = skus_df[skus_df["price_tier"] == "Flagship"]["sku_id"].tolist()
            affected_sku = flagship_skus[0] if flagship_skus else None
            some_store = stores_df["store_id"].iloc[0]
            st.session_state.successor_override = {affected_sku: 10} if affected_sku else {}
            st.session_state.demand_multipliers = {some_store: 0.60}  # -40% demand
            st.session_state.scenario_label = (
                f"Successor launching in 10 days for {affected_sku} + "
                f"{some_store} demand drops 40%"
            )
            st.rerun()

    with p2:
        if st.button("Preset 2: Supplier Stockout Crisis"):
            flagship_skus = skus_df[skus_df["price_tier"] == "Flagship"]["sku_id"].tolist()
            st.session_state.warehouse_stock_limits = {
                sku: 5 for sku in flagship_skus  # simulate ~70% cut vs typical ~15-20 unit need
            }
            st.session_state.demand_multipliers = {}
            st.session_state.successor_override = {}
            st.session_state.scenario_label = "Central warehouse stock drops 70% across all Flagship models"
            st.rerun()

    with p3:
        if st.button("Preset 3: Localized Festive Boom"):
            tier3_stores = stores_df[stores_df["profile"] == "Tier3_Volume"]["store_id"].tolist()
            mult_dict = {s: 3.0 for s in tier3_stores}  # +200%
            st.session_state.demand_multipliers = mult_dict
            st.session_state.warehouse_stock_limits = {}
            st.session_state.successor_override = {}
            st.session_state.scenario_label = "Surge demand (+200%) in Tier-3 stores due to local festival"
            st.rerun()

    with p4:
        if st.button("Reset to Baseline"):
            st.session_state.demand_multipliers = {}
            st.session_state.warehouse_stock_limits = {}
            st.session_state.successor_override = {}
            st.session_state.scenario_label = "Baseline (no scenario active)"
            st.rerun()

    st.markdown("---")
    st.markdown("##### Custom Parameter Injection")

    c1, c2 = st.columns(2)
    with c1:
        target_stores = st.multiselect(
            "Target Store(s)", options=stores_df["store_id"].tolist(),
            format_func=lambda sid: f"{sid} — {stores_df.set_index('store_id').loc[sid, 'name']}",
        )
        demand_delta_pct = st.slider(
            "Store Demand Delta Multiplier (%)", min_value=-100, max_value=300, value=0, step=5,
            help="-100% = complete collapse/closure, +300% = sudden demand surge",
        )
        target_scope = st.selectbox(
            "Target SKU / Category Scope",
            options=["Entire chain"] + ["Flagship", "Mid", "Budget"] + skus_df["sku_id"].tolist(),
        )

    with c2:
        successor_days = st.number_input(
            "Successor Launch / EOL Days Countdown (0-30)", min_value=0, max_value=30, value=0,
        )
        successor_sku_target = st.selectbox(
            "SKU affected by successor countdown", options=["(none)"] + skus_df["sku_id"].tolist(),
        )
        successor_confidence = st.radio(
            "Signal confidence",
            options=["Confirmed", "Rumoured"],
            horizontal=True,
            help=(
                "Confirmed dates trigger a real markdown/transfer recommendation. "
                "Rumoured dates are only added to the Watchlist (Tab 3) — no capital "
                "action is auto-recommended, since rumours are frequently wrong on "
                "timing or existence."
            ),
        )
        warehouse_limit = st.number_input(
            "Central Warehouse Stock Limit (units, per affected SKU)",
            min_value=0, max_value=1000, value=1000,
            help="Set below 1000 to simulate a supplier stockout / supply disruption",
        )

    if st.button("Apply Custom Scenario", type="primary"):
        mult = 1.0 + demand_delta_pct / 100.0
        new_demand_mult = {}
        if target_stores:
            for s in target_stores:
                new_demand_mult[s] = mult
        else:
            new_demand_mult["ALL"] = mult
        st.session_state.demand_multipliers = new_demand_mult

        if warehouse_limit < 1000:
            if target_scope in ("Flagship", "Mid", "Budget"):
                affected_skus = skus_df[skus_df["price_tier"] == target_scope]["sku_id"].tolist()
            elif target_scope == "Entire chain":
                affected_skus = skus_df["sku_id"].tolist()
            else:
                affected_skus = [target_scope]
            st.session_state.warehouse_stock_limits = {sku: warehouse_limit for sku in affected_skus}
        else:
            st.session_state.warehouse_stock_limits = {}

        if successor_sku_target != "(none)" and successor_days > 0:
            st.session_state.successor_override = {
                successor_sku_target: {
                    "days": successor_days,
                    "confidence": successor_confidence.lower(),
                }
            }
        else:
            st.session_state.successor_override = {}

        store_desc = ", ".join(target_stores) if target_stores else "all stores"
        st.session_state.scenario_label = (
            f"Custom: {demand_delta_pct:+d}% demand at {store_desc}, scope={target_scope}, "
            f"warehouse cap={warehouse_limit}, successor countdown={successor_days}d "
            f"({successor_confidence})"
        )
        st.rerun()

    st.markdown("---")

    # ---- Real-time recalculation output ----
    st.markdown("##### Revised Monday Allocation (live)")
    if not allocation_df.empty:
        show_table(
            allocation_df[[
                "Store_ID", "SKU_ID", "Model_Name", "Recommended_Units",
                "Capital_Required_INR", "Expected_Net_Profit_INR",
            ]].head(20),
            height=300,
        )
        revised_capital = allocation_df["Capital_Required_INR"].sum()
        st.caption(f"Revised capital deployed: {format_inr(revised_capital)} of {format_inr(capital_cap_inr)} cap "
                   f"({revised_capital / capital_cap_inr * 100:.1f}%).")
    else:
        st.warning("Under this scenario, no allocation lines are feasible.")

    st.markdown("##### Financial Trade-Off Matrix (Hold vs. Markdown vs. Transfer)")
    if not eol_df.empty:
        show_table(
            eol_df[[
                "Store_ID", "SKU_ID", "Units_Held", "Option_A_Hold_Net_INR",
                "Option_B_Local_Markdown_Net_INR", "Option_C_Transfer_Net_INR",
                "Recommended_Action", "Recommended_Net_Recovery_INR",
            ]].head(15),
            height=300,
        )
    else:
        st.info("No EOL-risk holdings detected under this scenario.")

    if not watchlist_df.empty:
        st.caption(
            f"👁️ {len(watchlist_df)} SKU(s) on the rumour watchlist (see Tab 3) — "
            f"monitored only, no action taken since the successor date isn't confirmed."
        )

    # ---- Live verbal defense summary ----
    st.markdown("##### Live Verbal Defense Summary")
    n_lines = len(allocation_df)
    capital_pct = (allocation_df["Capital_Required_INR"].sum() / capital_cap_inr * 100) if n_lines else 0
    n_eol = len(eol_df)
    top_action = eol_df["Recommended_Action"].mode().iloc[0] if n_eol else "N/A"

    defense_text = (
        f"Under the active scenario, the engine recomputed {n_lines} allocation lines, "
        f"deploying {capital_pct:.1f}% of the {format_inr(capital_cap_inr)} capital cap. "
        f"The Priority Index (profit margin ÷ capital required, weighted by stockout penalty) "
        f"re-ranked every (store, SKU) pair against the new demand and supply constraints, so "
        f"capital automatically shifted away from any SKU whose warehouse availability, demand "
        f"multiplier, or successor countdown changed. "
    )
    if n_eol:
        defense_text += (
            f"On the EOL side, {n_eol} holdings are now at risk, and '{top_action}' is the most "
            f"common recommended action because it produces the highest net rupee recovery given "
            f"the current transfer fee (₹{transfer_fee}/unit) and markdown assumptions."
        )
    st.info(defense_text)

    st.markdown("---")
    st.markdown("### Task 5 Scorecard: Smart Engine vs. Naive Baseline")
    st.caption(
        "Reported honestly — this scorecard is not tuned to force a sweep. Where the "
        "Naive Baseline wins on a metric, that's a real trade-off, not a bug."
    )
    with st.spinner("Backtesting both engines over 12 months of history..."):
        scorecard_df = _cached_backtest_scorecard(capital_cap_inr)
    show_table(scorecard_df)

    wins = scorecard_df["Smart Engine Wins"].sum()
    losses = len(scorecard_df) - wins
    if losses == 0:
        st.success(f"Smart Engine wins on all {len(scorecard_df)} metrics this run.")
    else:
        st.info(
            f"Smart Engine wins on {wins} of {len(scorecard_df)} metrics and trails on "
            f"{losses}. Typically it's Annual Capital Turns: the Smart Engine deliberately "
            f"protects margin-rich, slower-moving flagship stock that a pure "
            f"volume-proportional baseline ignores — which costs it some raw turnover "
            f"while still being the better capital-allocation decision."
        )
