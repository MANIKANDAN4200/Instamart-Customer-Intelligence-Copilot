
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split the file into seven queries using the Q comments.
queries = []
current = []

for line in sql_text.splitlines():
    if line.strip().startswith("-- Q") and current:
        query = "\n".join(current).strip()
        if query:
            queries.append(query)
        current = []
    if not line.strip().startswith("--"):
        current.append(line)

query = "\n".join(current).strip()
if query:
    queries.append(query)

print(f"Found {len(queries)} SQL queries.")

if len(queries) != 7:
    raise ValueError("Expected exactly 7 SQL queries in sql/queries.sql")

with sqlite3.connect(DB_PATH) as conn:
    for i, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, conn)

        print("\n" + "=" * 70)
        print(f"Q{i} RESULTS")
        print("=" * 70)
        print(f"Rows returned: {len(result)}")
        print(result.head(10).to_string(index=False))

        if i == 2:
            print(
                "Revenue grand total INR:",
                round(result["revenue_inr"].sum(), 2),
            )
        elif i == 5:
            print(
              "Average customer spend INR:",
              round(result["average_customer_spend"].iloc[0], 2)
            )
        elif i == 6:
            print(
                "Rows where RANK differs from ROW_NUMBER:",
                int((result["spend_rank"] != result["row_num"]).sum()),
            )
        elif i == 7:
            print(
                "Final cumulative revenue INR:",
                result["cumulative_revenue_inr"].iloc[-1],
            )

print("\nSQL QUERY EXECUTION COMPLETED")
