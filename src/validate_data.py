
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"

checks = []


def check(description, condition):
    passed = bool(condition)
    checks.append((description, passed))
    print(f"{'PASS' if passed else 'FAIL'}: {description}")


# Check 1–4: raw data counts
customers_raw = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
products_raw = pd.read_csv(RAW / "products.csv")
orders_raw = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
tickets_raw = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

check("Raw customers = 1,200", len(customers_raw) == 1200)
check("Raw products = 40", len(products_raw) == 40)
check("Raw orders = 13,084", len(orders_raw) == 13084)
check("Raw tickets = 164", len(tickets_raw) == 164)

# Check 5–8: expected raw data issues
check(
    "Raw orders contain 119 exact duplicate rows",
    int(orders_raw.duplicated().sum()) == 119,
)
check(
    "Raw orders contain 656 pending rows",
    int((orders_raw["status"].astype(str).str.strip().str.lower() == "pending").sum()) == 656,
)
check(
    "Raw orders contain 120 non-numeric order amounts",
    int(pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum()) == 120,
)
delivery_raw = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
check(
    "Raw orders contain 106 negative delivery times",
    int((delivery_raw < 0).sum()) == 106,
)

# Check 9–12: cleaned CSV files
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

check("Cleaned customers = 1,200", len(customers) == 1200)
check("Cleaned products = 40", len(products) == 40)
check("Cleaned orders = 12,309", len(orders) == 12309)
check("Cleaned order IDs are unique", orders["order_id"].is_unique)

# Check 13–17: cleaned customer data
check("No missing cleaned customer ages", customers["age"].notna().all())
check("All cleaned ages are between 18 and 60", customers["age"].between(18, 60).all())
check(
    "Unknown gender count = 30",
    int((customers["gender"].astype(str).str.lower() == "unknown").sum()) == 30,
)
check(
    "Six expected cities",
    set(customers["city"].dropna().unique())
    == {"Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"},
)
check("No missing cleaned customer cities", customers["city"].notna().all())

# Check 18–22: cleaned order data
status_counts = orders["status"].value_counts().to_dict()
check("Completed orders = 11,110", status_counts.get("completed", 0) == 11110)
check("Cancelled orders = 1,199", status_counts.get("cancelled", 0) == 1199)
check("No pending orders remain", status_counts.get("pending", 0) == 0)
check("No missing order amounts", orders["order_amount"].notna().all())
completed = orders.loc[orders["status"] == "completed"]
cancelled = orders.loc[orders["status"] == "cancelled"]
check(
    "No completed orders have missing delivery times",
    completed["delivery_minutes"].notna().all(),
)
check(
    "All cancelled orders have missing delivery times",
    cancelled["delivery_minutes"].isna().all(),
)

# Check 23: total completed revenue
revenue = round(pd.to_numeric(completed["order_amount"]).sum(), 2)
check("Completed revenue = INR 2,998,668.80", revenue == 2998668.80)

# Check 24–26: SQLite database and tables
check("SQLite database exists", DB.exists())

if DB.exists():
    with sqlite3.connect(DB) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        check(
            "SQLite has customers, products and orders tables",
            {"customers", "products", "orders"}.issubset(tables),
        )

        db_revenue = conn.execute(
            "SELECT ROUND(SUM(order_amount), 2) "
            "FROM orders WHERE status = 'completed'"
        ).fetchone()[0]

        check(
            "SQLite completed revenue matches cleaned CSV",
            db_revenue is not None and round(db_revenue, 2) == revenue,
        )

print(f"\nPassed: {sum(passed for _, passed in checks)}/{len(checks)} checkpoints")

if all(passed for _, passed in checks):
    print("ALL CHECKPOINTS PASSED")
else:
    raise SystemExit("Validation failed. Review the FAIL messages above.")
