
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load the raw data without modifying the original files.
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# --------------------------------------------------
# 1. Clean customers
# --------------------------------------------------
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# Standardize gender values.
gender_map = {
    "M": "male",
    "MALE": "male",
    "MAN": "male",
    "F": "female",
    "FEMALE": "female",
    "WOMAN": "female",
}

gender_values = (
    customers["gender"].astype("string").str.strip().str.upper()
)
customers["gender"] = gender_values.map(gender_map).fillna("unknown")

# Convert invalid ages to missing, then fill using each city's median.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
customers["age"] = customers["age"].round().astype("int64")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# --------------------------------------------------
# 2. Clean products
# --------------------------------------------------
products["product_id"] = products["product_id"].astype("string").str.strip()
products["category"] = products["category"].astype("string").str.strip()

# --------------------------------------------------
# 3. Clean orders
# --------------------------------------------------
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

# Remove exact duplicate rows first.
orders = orders.drop_duplicates().copy()

# Keep the latest record for each order_id.
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset="order_id", keep="last").copy()

orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Join product details to calculate missing order amounts.
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

# Negative delivery times are invalid.
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Impute missing delivery time only for completed orders.
completed = orders["status"].str.lower().eq("completed")
category_medians = orders.loc[completed].groupby("category")[
    "delivery_minutes"
].transform("median")

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_medians)
)

# Cancelled orders must have no delivery time.
cancelled = orders["status"].str.lower().eq("cancelled")
orders.loc[cancelled, "delivery_minutes"] = pd.NA

# Remove helper column and normalize status.
orders = orders.drop(columns=["price_inr", "category"])
orders["status"] = orders["status"].astype("string").str.strip().str.lower()

# --------------------------------------------------
# 4. Save cleaned CSV files
# --------------------------------------------------
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)

# Store timestamps in the required SQLite-friendly format.
orders_to_save = orders.copy()
orders_to_save["order_ts"] = orders_to_save["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
orders_to_save.to_csv(CLEAN_DIR / "orders.csv", index=False)

# --------------------------------------------------
# 5. Create SQLite database
# --------------------------------------------------
if DB_PATH.exists():
    DB_PATH.unlink()

with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders_to_save.to_sql("orders", conn, if_exists="replace", index=False)

# --------------------------------------------------
# 6. Print validation summary
# --------------------------------------------------
print("\nCUSTOMER CHECKS")
print("Rows:", len(customers))
print("Cities:\n", customers["city"].value_counts().sort_index().to_string())
print("Genders:\n", customers["gender"].value_counts().sort_index().to_string())
print("Missing ages:", customers["age"].isna().sum())
print("Age range:", customers["age"].min(), "to", customers["age"].max())

print("\nORDER CHECKS")
print("Rows:", len(orders))
print("Statuses:\n", orders["status"].value_counts().sort_index().to_string())
print("Unique order IDs:", orders["order_id"].nunique())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times for completed orders:",
    orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum(),
)
print(
    "Missing delivery times for cancelled orders:",
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum(),
)
print(
    "Completed order amount total:",
    round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2),
)

print("\nSaved cleaned CSV files and SQLite database.")
