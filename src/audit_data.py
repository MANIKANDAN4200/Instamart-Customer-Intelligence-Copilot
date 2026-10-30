
from pathlib import Path
import pandas as pd

# Find the project root directory
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

files = {
    "customers": RAW_DIR / "customers.csv",
    "products": RAW_DIR / "products.csv",
    "orders": RAW_DIR / "orders.csv",
    "tickets": RAW_DIR / "tickets.csv",
}

for name, path in files.items():
    print("\n" + "=" * 60)
    print(f"TABLE: {name.upper()}")
    print("=" * 60)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    # Preserve empty ticket text as an empty string
    if name == "tickets":
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_csv(path)

    print(f"\nShape: {df.shape}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

print("\nRAW DATA AUDIT COMPLETED")
