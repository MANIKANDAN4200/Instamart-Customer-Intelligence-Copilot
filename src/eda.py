
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as conn:
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT * FROM products", conn)

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

completed = orders.loc[orders["status"] == "completed"].copy()

print("\n1. PIVOT TABLE: REVENUE BY CITY AND CATEGORY")
revenue = completed.merge(
    customers[["customer_id", "city"]], on="customer_id", how="left"
).merge(
    products[["product_id", "category"]], on="product_id", how="left"
)

pivot = pd.pivot_table(
    revenue,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All",
)
print(pivot.round(2).to_string())
print("Pivot grand total INR:", round(pivot.loc["All", "All"], 2))

print("\n2. CITY-LEVEL COMPLETED ORDER STATISTICS")
city_stats = revenue.groupby("city").agg(
    revenue_inr=("order_amount", "sum"),
    mean_order_amount=("order_amount", "mean"),
    completed_order_count=("order_id", "count"),
)
print(city_stats.round(2).to_string())

print("\n3. COMPLETED ORDER AMOUNT DISTRIBUTION")
amount = completed["order_amount"].dropna()
q1 = amount.quantile(0.25)
median = amount.median()
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("Count:", len(amount))
print("Mean INR:", round(amount.mean(), 2))
print("Median INR:", round(median, 2))
print("Skew:", round(amount.skew(), 2))
print("Q1 INR:", round(q1, 2))
print("Q3 INR:", round(q3, 2))
print("IQR INR:", round(iqr, 2))
print("Lower fence INR:", round(lower_fence, 2))
print("Upper fence INR:", round(upper_fence, 2))
print("Outliers below lower fence:", int((amount < lower_fence).sum()))
print("Outliers above upper fence:", int((amount > upper_fence).sum()))

print("\n4. CORRELATION MATRIX")
corr_cols = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]
correlation = completed[corr_cols].corr()
print(correlation.round(3).to_string())

pairs = []
for i, col1 in enumerate(corr_cols):
    for col2 in corr_cols[i + 1:]:
        value = correlation.loc[col1, col2]
        if pd.notna(value):
            pairs.append((col1, col2, value))

strongest = max(pairs, key=lambda item: abs(item[2]))
print(
    "Largest absolute correlation:",
    strongest[0],
    "vs",
    strongest[1],
    "=",
    round(strongest[2], 3),
)

print("\nEDA COMPLETED")
