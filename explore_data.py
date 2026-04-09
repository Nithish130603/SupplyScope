"""
Initial exploration of the Favorita dataset.
Run this ONCE to understand the data before building anything.

Key questions we need to answer:
1. How big is the data? Can we work with all of it or need to subset?
2. What's the date range? What granularity?
3. How many stores/products? Which are most interesting to forecast?
4. Missing values? Data quality issues?
5. What do sales patterns look like? Seasonality? Trends?
"""

import pandas as pd
import os

DATA_DIR = "data"

# ============================================================
# 1. Load and inspect each file
# ============================================================

print("=" * 60)
print("1. FILE OVERVIEW")
print("=" * 60)

files = ["train.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"]
for f in files:
    path = os.path.join(DATA_DIR, f)
    df = pd.read_csv(path, nrows=5)
    full_size = sum(1 for _ in open(path)) - 1  # subtract header
    print(f"\n--- {f} ---")
    print(f"Rows: {full_size:,}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))

# ============================================================
# 2. Deep dive into train.csv (main sales data)
# ============================================================

print("\n" + "=" * 60)
print("2. TRAIN.CSV DEEP DIVE")
print("=" * 60)

# Load with proper types to save memory
train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    parse_dates=["date"],
    dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int16"},
)

print(f"\nShape: {train.shape}")
print(f"Memory usage: {train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\nDate range: {train['date'].min()} to {train['date'].max()}")
print(f"Number of stores: {train['store_nbr'].nunique()}")
print(f"Number of product families: {train['family'].nunique()}")
print(f"\nProduct families:")
for fam in sorted(train["family"].unique()):
    print(f"  - {fam}")

# ============================================================
# 3. Missing values
# ============================================================

print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)

print("\ntrain.csv:")
print(train.isnull().sum())

oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"))
print(f"\noil.csv: {oil.isnull().sum().to_dict()}")

# ============================================================
# 4. Sales distribution
# ============================================================

print("\n" + "=" * 60)
print("4. SALES STATISTICS")
print("=" * 60)

print(f"\nOverall sales stats:")
print(train["sales"].describe())

print(f"\nZero-sales rows: {(train['sales'] == 0).sum():,} ({(train['sales'] == 0).mean():.1%})")
print(f"Negative sales (returns): {(train['sales'] < 0).sum():,}")

# ============================================================
# 5. Sales by product family (top 10)
# ============================================================

print("\n" + "=" * 60)
print("5. TOP 10 PRODUCT FAMILIES BY TOTAL SALES")
print("=" * 60)

family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False)
for i, (fam, total) in enumerate(family_sales.head(10).items()):
    print(f"  {i+1}. {fam}: {total:,.0f}")

# ============================================================
# 6. Scoping decision
# ============================================================

print("\n" + "=" * 60)
print("6. SCOPING RECOMMENDATION")
print("=" * 60)

total_combinations = train["store_nbr"].nunique() * train["family"].nunique()
print(f"\nTotal store × product combinations: {total_combinations}")
print(f"Total rows: {len(train):,}")
print(f"""
For the MVP, I recommend:
- Pick 1 store (the one with most data/variety)
- Use top 8-10 product families (covers most revenue)
- This gives us ~10 time series to forecast
- Enough to demo multi-product comparison without overwhelming compute

Later we can scale to multi-store.
""")

# Which store has the most consistent data?
store_coverage = train.groupby("store_nbr").agg(
    total_sales=("sales", "sum"),
    avg_daily=("sales", "mean"),
    date_range=("date", lambda x: (x.max() - x.min()).days),
).sort_values("total_sales", ascending=False)

print("Top 5 stores by total sales:")
print(store_coverage.head().to_string())