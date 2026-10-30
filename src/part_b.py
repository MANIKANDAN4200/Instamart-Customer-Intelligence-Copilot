
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

# --------------------------------------------------
# 2. Load Part A cleaned data
# --------------------------------------------------
orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Use completed orders only for customer behaviour.
completed = orders.loc[
    orders["status"].eq("completed")
    & orders["order_ts"].notna()
].copy()

# --------------------------------------------------
# 3. Build the RFM table
# R = recency, F = frequency, M = monetary
# --------------------------------------------------
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

print("\n========== RFM SUMMARY ==========")
print("Customers with completed orders:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# Check the reference statistics.
assert len(rfm) == 1175, "Unexpected RFM customer count"
assert rfm["recency"].median() == 19, "Unexpected median recency"
assert rfm["recency"].min() == 0, "Unexpected minimum recency"
assert rfm["recency"].max() == 180, "Unexpected maximum recency"
assert rfm["frequency"].max() == 33, "Unexpected maximum frequency"

# --------------------------------------------------
# 4. Scale RFM and compare K-means models
# --------------------------------------------------
rfm_columns = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[rfm_columns])

results = []

print("\n========== K-MEANS METRICS ==========")

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette,
    })

metrics = pd.DataFrame(results)
print(metrics.round(4).to_string(index=False))

# Save the comparison table for documentation.
metrics.to_csv(OUTPUT_DIR / "kmeans_metrics.csv", index=False)

# K=4 is selected based on its strong silhouette score,
# the elbow pattern and useful business segmentation.
print("\nSelected K:", FINAL_K)
print(
    "K=8 has lower inertia because more clusters let K-means "
    "fit the data more closely. However, inertia always tends "
    "to decrease as K increases, so it must be considered with "
    "silhouette scores and business interpretability."
)

# --------------------------------------------------
# 5. Fit final K=4 clustering
# --------------------------------------------------
final_model = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42,
)

rfm["segment_id"] = final_model.fit_predict(X_scaled)

profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .round(2)
)

print("\n========== SEGMENT PROFILES ==========")
print(profiles.to_string())

# --------------------------------------------------
# 6. Give clusters business-friendly names
# Cluster IDs are arbitrary labels, not rankings.
# --------------------------------------------------
premium_id = profiles["mean_monetary"].idxmax()
remaining = [sid for sid in profiles.index if sid != premium_id]

frequent_id = profiles.loc[remaining, "mean_frequency"].idxmax()
remaining.remove(frequent_id)

at_risk_id = profiles.loc[remaining, "mean_recency"].idxmax()
remaining.remove(at_risk_id)

steady_id = remaining[0]

segment_names = {
    premium_id: "Premium High-Value Customers",
    frequent_id: "Frequent Regulars",
    at_risk_id: "At-Risk Occasional Customers",
    steady_id: "Steady Shoppers",
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

profiles["segment_name"] = profiles.index.map(segment_names)
profiles["is_premium"] = profiles.index == premium_id

print("\n========== NAMED SEGMENTS ==========")
print(profiles.to_string())
print("\nPremium segment ID:", premium_id)
print("Premium means the segment with the highest mean monetary value.")
print("Cluster IDs are labels only; they have no natural ordering.")

# --------------------------------------------------
# 7. PCA on scaled RFM
# --------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X_scaled)

variance = pca.explained_variance_ratio_

print("\n========== PCA ==========")
print("PC1 explained variance:", round(variance[0], 4))
print("PC2 explained variance:", round(variance[1], 4))
print("Total explained variance:", round(variance.sum(), 4))

# --------------------------------------------------
# 8. Save required customer segments file
# --------------------------------------------------
output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

segments_path = OUTPUT_DIR / "customer_segments.csv"
rfm[output_columns].to_csv(segments_path, index=False)

print("\nSaved:", segments_path)
print("Segment rows:", len(rfm))
print("\nPART B SEGMENTATION COMPLETED")
