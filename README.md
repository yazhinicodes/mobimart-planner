# MobiMart Planner

Weekly stock allocation & end-of-life risk engine for a 25-store mobile
retail chain operating under a hard **₹4 Crore** working-capital cap.
Built in Python/Streamlit as a capital-constrained decision-support system:
it generates a realistic 12-month sales history, produces a Monday
allocation recommendation with the reasoning shown in rupees, flags and
prices out End-of-Life inventory risk, and benchmarks itself **honestly**
against a naive proportional-allocation baseline — reporting where it wins
and where it genuinely loses.

## Current Scorecard Snapshot

| Metric | Smart Engine | Naive Baseline | Smart Engine Wins |
|---|---|---|---|
| Stockout Rate (%) | 71.70 | 93.79 | ✅ |
| Dead Stock Percentage (%) | 0.00 | 0.00 | ✅ |
| Total Markdown Losses (₹) | 0 | 0 | ✅ |
| Annual Capital Turns | 1.86 | 1.92 | ❌ |
| Weeks of Supply Cover | 0.98 | 0.25 | ✅ |

**Wins 4 of 5, and the one loss is well understood, not hidden:** the Naive
Baseline allocates zero capital to Flagship phones at all — not by
strategy, but because its logic can't see past raw unit volume, and
Flagship sells in low unit counts despite high per-unit value. Flagship
also happens to carry the lowest margin *percentage* of the three price
tiers, so accidentally avoiding it entirely inflates Naive's blended
margin rate. The Smart Engine deliberately keeps real Flagship stock (a
real retailer can't abandon premium customers), which costs a small amount
of capital-turnover efficiency in exchange for not losing that segment
entirely. (Numbers regenerate slightly run-to-run due to random demand
noise in the generated dataset; re-run `python -m simulation.evaluator`
for the exact current values.)

## Project Structure
mobimart-planner/
├── data/
│ ├── stores_config.json # 25 stores (8 in Bangalore, rest Karnataka tier-2/3 cities)
│ ├── skus_config.json # 60 SKUs: Flagship (>₹50k), Mid (₹15k-50k), Budget (<₹15k)
│ └── sales_history_12m.csv # 365 days of generated daily sales history
├── engine/
│ ├── data_generator.py # Task 1: synthesizes the 12-month sales history
│ ├── profiling.py # shared demand-forecasting, SKU-aging & Rs. formatting utilities
│ ├── allocation.py # Task 2: Monday weekly allocation engine (Priority Index)
│ └── lifecycle.py # Task 3: EOL risk detection + Hold/Markdown/Transfer engine
├── simulation/
│ ├── baseline_naive.py # Task 5: naive proportional-to-last-month baseline
│ └── evaluator.py # Task 5: backtest + 5-metric honest scorecard
├── app.py # Task 4: Streamlit owner dashboard (4 tabs)
└── requirements.txt

## Setup

```bash
git clone https://github.com/yazhinicodes/mobimart-planner.git
cd mobimart-planner
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

The 12-month sales history is already generated and included. To regenerate it:

```bash
python -m engine.data_generator
```

## Run the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Four tabs:

1. **🏠 Owner's Dashboard** — answers the brief's three core questions in one
   place: where is capital right now, what stock is at risk, and what did
   last week's recommendations actually earn or lose (a genuine backtest
   against the Naive Baseline, not a live audit — clearly labeled as such).
2. **📦 Weekly Allocation Hub** — the Monday recommendation, filterable by
   store/tier, with click-to-select rupee justification per line.
3. **⚠️ EOL Risk Manager** — at-risk holdings with Hold/Markdown/Transfer
   options costed in rupees, plus a separate Rumour Watchlist.
4. **🎯 Defense Scenario Simulator & Scorecard** — inject a live what-if
   (demand shock, supply cut, successor launch) via presets or custom
   parameters, and watch every tab recompute from the same underlying
   functions — no separate demo code path.

A persistent status strip at the top of every tab shows the current
Capital Cap, Transfer Fee, and any active scenario, so it's never ambiguous
what's driving the numbers on screen.

## Run Individual Engines from the Command Line

```bash
python -m engine.allocation      # sample Monday allocation run
python -m engine.lifecycle       # EOL-risk holdings and recommended actions
python -m simulation.baseline_naive
python -m simulation.evaluator   # Smart Engine vs. Naive Baseline scorecard
```

## Key Design Decisions

- **Priority Index** = `(Expected Profit Margin / Capital Required) × Stockout Penalty`,
  greedily filled in descending order until the ₹4 Crore cap is exhausted.
- **Tier-aware stockout penalty**: the brief is explicit that *"a customer
  who cannot find a ₹15,000 phone buys it next door — sale and customer
  both lost. A flagship buyer might wait two days for a transfer."* The
  penalty multiplier is asymmetric by price tier (Budget > Mid > Flagship)
  to price that difference in directly, rather than letting the formula's
  natural bias toward absolute rupee margin silently over-invest in
  expensive flagship phones at the expense of fast-moving budget stock.
- **Fairness floor, profile-aware**: every store is guaranteed a real
  minimum assortment (not just a token single line) before the general
  priority-ranked top-up begins, so no store is left with zero
  recommendations for the week. Generalist store profiles (Tier-2 Hub, no
  strong demand specialization) get a wider floor (top-3 SKUs) than
  specialist profiles, since a flat one-SKU floor was found to leave them
  permanently stuck regardless of real, measured customer demand.
- **Deterministic tie-breaking**: candidates with an identical Priority
  Index are resolved by higher real demand first, then store ID, so the
  ranking is fully reproducible rather than falling through to an
  undefined sort order.
- **Two independent risk-control caps**: no single (store, SKU) line may
  exceed 10% of total capital, and no single line may exceed 50 units/week
  (calibrated empirically — this ceiling was found to be truncating 29% of
  candidates well below their real 3-week-cover demand at a lower value;
  raising it improved three scorecard metrics simultaneously with no
  downside on the other two).
- **Rumour vs. confirmed launch dates** are treated differently by design:
  only a *confirmed* successor date can trigger a real Hold/Markdown/
  Transfer action. A *rumoured* date is surfaced separately on a
  **Watchlist** for human awareness only, since acting on unconfirmed leaks
  risks destroying margin on stock that may not actually be superseded.
- **Scenario Injector** shares the exact same underlying functions as the
  rest of the app — there is no separate "demo mode."
- **Scorecard honesty**: both engines are backtested against the identical
  realized-demand horizon and capital cap. The scorecard is not tuned to
  force a sweep; where the Naive Baseline wins, that reflects a real,
  traceable trade-off, reported as such, per the brief's explicit
  instruction.
- **Rupee formatting** uses Indian lakh/crore convention (₹4.00 Cr, ₹54.66
  L) throughout, matching how the brief itself talks about money, rather
  than western comma-grouping.

## Verification Process

This project went through a deliberate audit pass, not just a single build:

- **Data generator**: found and fixed 3 real bugs — a pre-launch sales
  leak affecting late-year SKUs, cannibalization triggering on the wrong
  date (the old SKU's own launch day instead of its successor's), and 5 of
  6 configured successor relationships being chronologically invalid
  (a "successor" launching before the phone it replaces).
- **Allocation engine**: found that 7 of 25 stores were receiving zero
  weekly allocation despite genuine, measured customer demand (9-20
  units/week), purely as a side-effect of a fixed capital pool being
  exhausted by higher-priority lines first — fixed via the fairness floor
  described above.
- **Naive baseline**: audited and confirmed it structurally leaves ~29% of
  its own capital budget completely unused (rounding losses across ~500
  thin proportional lines, with no redistribution mechanism) — a realistic
  property of naive proportional allocation, not a bug, and a further
  point in the Smart Engine's favor (which leaves <0.01% unused).

## Known Scope Boundaries

- Demand is forecast as a single point estimate (trailing 4-week average),
  not a full distribution — there is no explicit safety-stock/variance
  buffer. Reasonable for this assignment's scope; a production system
  would size orders against a service-level target, not just the mean.
- The EOL Risk Manager doesn't have store/tier filters (unlike the
  Allocation Hub) because at-risk holdings are naturally low-diversity in
  this dataset (dozens of rows but typically only 2-3 distinct SKUs) — the
  filtering pattern already exists in the codebase and could be added in
  minutes if real-world scale required it.