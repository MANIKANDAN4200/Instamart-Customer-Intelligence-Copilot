

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. File paths and cutoff dates
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
SEPTEMBER_END = pd.Timestamp("2025-10-01")

# --------------------------------------------------
# 2. Load the cleaned data
# --------------------------------------------------
orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

print("Clean orders:", len(orders))
print("Clean customers:", len(customers))

# --------------------------------------------------
# 3. Define the training cohort and September label
# --------------------------------------------------
completed = orders.loc[
    orders["status"].eq("completed")
    & orders["order_ts"].notna()
].copy()

# Use only completed orders strictly before September 1
history = completed.loc[
    completed["order_ts"] < CUTOFF
].copy()

# Customers who had at least one completed order
# before the cutoff are eligible for the cohort.
cohort_ids = history["customer_id"].unique()

# September orders are used only to create the label.
september_completed = completed.loc[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < SEPTEMBER_END)
]

september_customer_ids = set(
    september_completed["customer_id"].unique()
)

cohort = customers.loc[
    customers["customer_id"].isin(cohort_ids)
].copy()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_customer_ids)
).astype(int)

print("\n========== LAPSE COHORT ==========")
print("Customers in cohort:", len(cohort))
print("Lapsed customers:", int(cohort["lapsed"].sum()))
print("Not-lapsed customers:", int((cohort["lapsed"] == 0).sum()))
print("Lapse rate (%):", round(cohort["lapsed"].mean() * 100, 2))

# --------------------------------------------------
# 4. Build features from pre-cutoff history only
# --------------------------------------------------
rfm_features = (
    history.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)

rfm_features["recency_days"] = (
    CUTOFF - rfm_features["last_order_ts"]
).dt.days

# Cancellation rate: cancelled orders divided by all orders
# with a final status before the cutoff.
all_pre_cutoff = orders.loc[
    orders["order_ts"].notna()
    & (orders["order_ts"] < CUTOFF)
].copy()

cancel_stats = (
    all_pre_cutoff.assign(
        is_cancelled=all_pre_cutoff["status"].eq("cancelled").astype(int)
    )
    .groupby("customer_id")
    .agg(
        total_pre_cutoff_orders=("order_id", "nunique"),
        cancelled_pre_cutoff_orders=("is_cancelled", "sum"),
    )
    .reset_index()
)

cancel_stats["cancellation_rate"] = (
    cancel_stats["cancelled_pre_cutoff_orders"]
    / cancel_stats["total_pre_cutoff_orders"]
)

# Most frequently ordered product category before the cutoff.
history_with_products = history.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)

category_counts = (
    history_with_products.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="category_order_count")
    .sort_values(
        ["customer_id", "category_order_count", "category"],
        ascending=[True, False, True],
    )
)

favorite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favorite_category"})
)

# Merge only pre-cutoff behavioural features with customer attributes.
features = (
    cohort[["customer_id", "lapsed"]]
    .merge(rfm_features, on="customer_id", how="left")
    .merge(cancel_stats[["customer_id", "cancellation_rate"]],
           on="customer_id", how="left")
    .merge(favorite_category, on="customer_id", how="left")
    .merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]],
        on="customer_id",
        how="left",
        suffixes=("", "_customer"),
    )
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

features = features.drop(columns=["last_order_ts", "signup_date"])

# --------------------------------------------------
# 5. Check dataset and split into train/test
# --------------------------------------------------
target = features["lapsed"].copy()
X = features.drop(columns=["customer_id", "lapsed"]).copy()

numeric_features = [
    "recency_days",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]

categorical_features = [
    "favorite_category",
    "city",
    "gender",
]

print("\n========== FEATURE CHECKS ==========")
print("Feature rows:", len(X))
print("Feature columns:", list(X.columns))
print("Missing values per feature:")
print(X.isna().sum().to_string())
print("\nLapse label counts:")
print(target.value_counts().sort_index().to_string())

# Stop if a feature is missing; don't silently fill values.
assert not X.isna().any().any(), (
    "Missing feature values found. Review the report above."
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    target,
    test_size=0.2,
    stratify=target,
    random_state=42,
)

print("\n========== TRAIN/TEST SPLIT ==========")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training lapsed customers:", int(y_train.sum()))
print("Test lapsed customers:", int(y_test.sum()))

print("\nPART B LAPSE DATA PREPARATION COMPLETED")

# --------------------------------------------------
# 6. Baseline: predict every customer is not lapsed
# --------------------------------------------------
baseline_predictions = [0] * len(y_test)

print("\n========== BASELINE MODEL ==========")
print("Baseline accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
print(
    "Baseline lapsed-class F1:",
    round(f1_score(y_test, baseline_predictions, zero_division=0), 4),
)

# --------------------------------------------------
# 7. Preprocessing pipeline
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    remainder="drop",
)

# Show the full-cohort frequency mean for comparison.
# The scaler will be fitted on training rows only.
print("\n========== SCALER CHECK ==========")
print(
    "Full-cohort frequency mean:",
    round(X["frequency"].mean(), 4),
)

# --------------------------------------------------
# 8. Logistic Regression hyperparameter tuning
# --------------------------------------------------
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

lr_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

print("\nTraining Logistic Regression...")
lr_search.fit(X_train, y_train)

print("\n========== LOGISTIC REGRESSION ==========")
print("Best parameters:", lr_search.best_params_)
print("Best cross-validation F1:", round(lr_search.best_score_, 4))

lr_best = lr_search.best_estimator_
lr_predictions = lr_best.predict(X_test)

print("Test confusion matrix:")
print(confusion_matrix(y_test, lr_predictions, labels=[0, 1]))
print(
    classification_report(
        y_test,
        lr_predictions,
        labels=[0, 1],
        target_names=["Not lapsed", "Lapsed"],
        zero_division=0,
    ),
)

# The fitted numeric scaler's mean is learned from training data.
fitted_preprocessor = lr_best.named_steps["preprocessor"]
numeric_scaler = fitted_preprocessor.named_transformers_["numeric"]

frequency_position = numeric_features.index("frequency")
print(
    "Training-only scaler mean for frequency:",
    round(numeric_scaler.mean_[frequency_position], 4),
)
print(
    "The scaler mean can differ from the full-cohort mean because "
    "the scaler is fitted only on the training split.",
)

# --------------------------------------------------
# 9. Random Forest hyperparameter tuning
# --------------------------------------------------
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

print("\nTraining Random Forest (45 fits)...")
rf_search.fit(X_train, y_train)

print("\n========== RANDOM FOREST ==========")
print("Best parameters:", rf_search.best_params_)
print("Best cross-validation F1:", round(rf_search.best_score_, 4))

rf_best = rf_search.best_estimator_
rf_predictions = rf_best.predict(X_test)

print("Test confusion matrix:")
print(confusion_matrix(y_test, rf_predictions, labels=[0, 1]))
print(
    classification_report(
        y_test,
        rf_predictions,
        labels=[0, 1],
        target_names=["Not lapsed", "Lapsed"],
        zero_division=0,
    ),
)

print("\nPART B MODEL TRAINING COMPLETED")


# --------------------------------------------------
# 10. Oversample the minority class in training data
# --------------------------------------------------
print("\n========== CLASS BALANCE EXPERIMENT ==========")

train_data = X_train.copy()
train_data["lapsed"] = y_train.to_numpy()

majority_count = int(train_data["lapsed"].value_counts().max())
minority_rows = train_data.loc[train_data["lapsed"] == 1]

# Sample existing lapsed rows with replacement until both
# classes have equal counts. The test data is not touched.
oversampled_lapsed = minority_rows.sample(
    n=majority_count - len(minority_rows),
    replace=True,
    random_state=42,
)

balanced_train = pd.concat(
    [train_data, oversampled_lapsed],
    ignore_index=True,
)

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

print("Original training class counts:")
print(y_train.value_counts().sort_index().to_dict())

print("Balanced training class counts:")
print(y_balanced.value_counts().sort_index().to_dict())
print("Balanced training rows:", len(X_balanced))

# Refit the selected Logistic Regression pipeline.
# The scaler and encoder are refitted using balanced training rows only.
lr_balanced = lr_search.best_estimator_
lr_balanced.fit(X_balanced, y_balanced)

balanced_predictions = lr_balanced.predict(X_test)

print("\n========== BALANCED LOGISTIC REGRESSION ==========")
print(
    "Precision:",
    round(precision_score(y_test, balanced_predictions, zero_division=0), 4),
)
print(
    "Recall:",
    round(recall_score(y_test, balanced_predictions, zero_division=0), 4),
)
print(
    "F1-score:",
    round(f1_score(y_test, balanced_predictions, zero_division=0), 4),
)
print("Confusion matrix:")
print(confusion_matrix(y_test, balanced_predictions, labels=[0, 1]))

print("\n========== BALANCING COMPARISON ==========")
print(
    f"Unbalanced LR: precision="
    f"{precision_score(y_test, lr_predictions, zero_division=0):.4f}, "
    f"recall={recall_score(y_test, lr_predictions, zero_division=0):.4f}, "
    f"F1={f1_score(y_test, lr_predictions, zero_division=0):.4f}"
)
print(
    f"Balanced LR:   precision="
    f"{precision_score(y_test, balanced_predictions, zero_division=0):.4f}, "
    f"recall={recall_score(y_test, balanced_predictions, zero_division=0):.4f}, "
    f"F1={f1_score(y_test, balanced_predictions, zero_division=0):.4f}"
)


# --------------------------------------------------
# 11. Interpret the Random Forest and Logistic Regression
# --------------------------------------------------
print("\n========== FEATURE IMPORTANCE ==========")

# Random Forest importance from the unbalanced, tuned RF model.
rf_preprocessor = rf_best.named_steps["preprocessor"]
rf_model = rf_best.named_steps["model"]

rf_feature_names = rf_preprocessor.get_feature_names_out()
importance_table = (
    pd.DataFrame({
        "feature": rf_feature_names,
        "importance": rf_model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .head(5)
)

print("Top 5 Random Forest features:")
print(importance_table.to_string(index=False))

# Logistic Regression coefficients from the balanced model.
lr_preprocessor_balanced = lr_balanced.named_steps["preprocessor"]
lr_model_balanced = lr_balanced.named_steps["model"]

lr_feature_names = lr_preprocessor_balanced.get_feature_names_out()
coefficient_table = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_model_balanced.coef_[0],
})

print("\nLogistic Regression coefficient examples:")
for feature_prefix in [
    "numeric__recency_days",
    "numeric__frequency",
    "numeric__monetary",
]:
    matches = coefficient_table[
        coefficient_table["feature"] == feature_prefix
    ]
    if not matches.empty:
        row = matches.iloc[0]
        direction = "increases" if row["coefficient"] > 0 else "decreases"
        print(
            f"{row['feature']}: {row['coefficient']:.4f} "
            f"({direction} predicted lapse odds, holding other features fixed)"
        )

# --------------------------------------------------
# 12. Save and reload the selected model
# --------------------------------------------------
MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(lr_balanced, MODEL_PATH)
reloaded_model = joblib.load(MODEL_PATH)

reloaded_predictions = reloaded_model.predict(X_test)
same_predictions = (
    reloaded_predictions == balanced_predictions
).all()

print("\n========== MODEL PERSISTENCE ==========")
print("Saved model:", MODEL_PATH)
print("Reloaded predictions match:", same_predictions)

assert same_predictions, "Reloaded model predictions do not match!"

# --------------------------------------------------
# 13. Build the Premium retention list
# --------------------------------------------------
SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
segments = pd.read_csv(SEGMENTS_PATH)

# Use predicted probabilities from the selected balanced model.
lapse_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

test_results = pd.DataFrame({
    "customer_id": cohort.loc[
        X_test.index, "customer_id"
    ].values if X_test.index.isin(cohort.index).all() else [],
})

# X_test retains the original DataFrame index from the feature table.
# Use the matching customer IDs directly from the feature dataset.
test_results = pd.DataFrame({
    "customer_id": features.loc[X_test.index, "customer_id"].values,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.loc[X_test.index].values,
})

# Premium segment labels are used only after model training, to filter
# the illustrative retention list. They are not model input features.
premium_customers = set(
    segments.loc[segments["is_premium"].astype(bool), "customer_id"]
)

retention_list = (
    test_results[
        test_results["customer_id"].isin(premium_customers)
    ]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    .copy()
)

retention_list["lapse_probability"] = (
    retention_list["lapse_probability"].round(4)
)

RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(RETENTION_PATH, index=False)

print("\n========== PREMIUM RETENTION LIST ==========")
print(retention_list.to_string(index=False))
print("Saved:", RETENTION_PATH)
print("Premium customers listed:", len(retention_list))
print(
    "Actually lapsed among listed customers:",
    int(retention_list["actually_lapsed"].sum()),
)

print("\nPART B MODEL, INTERPRETATION AND RETENTION LIST COMPLETED")

