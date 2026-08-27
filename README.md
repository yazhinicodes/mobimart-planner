# MobiMart Planner

An intelligent weekly stock allocation and End-of-Life (EOL) risk management
engine for a 25-store mobile retail chain across Karnataka and South India,
operating under a hard ₹4 Crore active-inventory capital cap.

## Project Structure

```
mobimart-planner/
├── data/
│   ├── stores_config.json       # 25 stores across 3 profiles (Flagship / Tier2 Hub / Tier3 Volume)
│   ├── skus_config.json         # 60 SKUs across Flagship (>₹50k), Mid (₹15k-50k), Budget (<₹15k)
│   └── sales_history_12m.csv    # 365 days of generated daily sales history
├── engine/
│   ├── data_generator.py        # Task 1: synthesizes the 12-month sales history
│   ├── profiling.py             # shared demand-forecasting & SKU-aging utilities
│   ├── lifecycle.py             # Task 3: EOL risk detection + Hold/Markdown/Transfer engine
│   └── allocation.py            # Task 2: Monday weekly allocation engine (Priority Index)
├── simulation/
│   ├── baseline_naive.py        # Task 5: naive proportional-to-last-month baseline
│   └── evaluator.py             # Task 5: backtest + 5-metric scorecard
├── app.py                       # Task 4: Streamlit owner dashboard (4 tabs)
└── requirements.txt
```

## Setup

```bash
cd mobimart-planner
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

If the sales history CSV needs to be regenerated (it's already included, pre-built):

```bash
python3 -m engine.data_generator
```

## Run the Dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Run Individual Engines from the Command Line

```bash
python3 -m engine.allocation      # prints a sample Monday allocation run
python3 -m engine.lifecycle       # prints EOL-risk holdings and recommended actions
python3 -m simulation.baseline_naive
python3 -m simulation.evaluator   # prints the Smart Engine vs. Naive Baseline scorecard
```

## Key Design Notes

- **Priority Index** = `(Expected Profit Margin / Capital Required) * Stockout Penalty`,
  computed per (store, SKU) candidate and greedily allocated in descending
  order until the ₹4 Crore cap is exhausted. One risk-control guardrail sits
  on top: no single (store, SKU) line may consume more than 10% of total
  capital, preventing a single expensive flagship line from swallowing a
  disproportionate share of the week's budget. This limit is a fixed,
  round-number portfolio-concentration rule — it is not tuned to any
  particular scorecard outcome.
- **EOL detection** fires when a SKU is ≥8 weeks old with declining velocity,
  OR when a successor launch is **confirmed** 10-14 days out. Each at-risk
  holding is scored under all three financial options (Hold / Local Markdown
  / Transfer) and the highest net-recovery option is recommended automatically.
- **Rumour vs. confirmed launch dates** are treated differently by design, as
  the brief requires: a *confirmed* successor date is the only thing allowed
  to trigger a capital action (markdown/transfer). A *rumoured* date is
  surfaced separately on a **Watchlist** (Tab 3) for human awareness only —
  no capital is moved against it, since rumours are frequently wrong on
  timing or existence and acting on them risks destroying margin needlessly.
- **Scenario Injector** (Tab 4) lets you adjust demand multipliers, warehouse
  stock caps, and successor countdowns (with a confirmed/rumoured toggle)
  live — every allocation, EOL matrix, watchlist entry, and the verbal
  defense summary recompute immediately from the same underlying functions
  used elsewhere in the app (no separate "demo" code path).
- **Scorecard methodology and honesty**: both engines are backtested against
  the same 4-week realized-demand horizon and the same capital cap. Stockout
  rate and capital turns are computed on a profit-weighted basis (consistent
  with what the Priority Index itself optimizes for). The scorecard is
  **not** tuned to force a sweep — as of the current dataset, the Smart
  Engine wins 4 of 5 metrics and genuinely trails the Naive Baseline on
  Annual Capital Turns, because it deliberately protects margin-rich,
  slower-moving flagship stock that a pure volume-proportional baseline
  ignores. That's reported as-is, per the brief's explicit instruction to
  "report honestly where your system wins and where it loses."
- **Store geography** matches the brief exactly: 25 stores total, 8 of them
  in Bangalore, the rest spread across Karnataka tier-2/3 cities (Mysore,
  Hubli, Davangere, Tumkur, and similar).
