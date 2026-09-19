"""
Separate ML training script for the Assam + Bihar + Odisha flood-risk project.

Run:
    python train_model.py
"""

import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

DATA_FILE = "assam_bihar_odisha_500k_weather_flood_dataset.csv"
MODEL_FILE = "flood_risk_catboost.cbm"
TARGET = "flood_binary"

DROP_COLUMNS = [
    "timestamp", "flood_binary", "flood_depth_m",
    "weather_source", "rainfall_source", "soil_source",
    "terrain_source", "flood_label_source", "data_quality_flag"
]

print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nStates:", sorted(df["state"].dropna().unique()))

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

FEATURES = [c for c in df.columns if c not in DROP_COLUMNS]
CATEGORICAL_FEATURES = ["state", "district"]

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype(str)

for col in FEATURES:
    if col not in CATEGORICAL_FEATURES and X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    thread_count=-1,
    l2_leaf_reg=5,
    verbose=50
)

print("\nStarting training...")
model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

probability = model.predict_proba(X_val)[:, 1]
predicted = (probability >= 0.50).astype(int)

metrics = {
    "accuracy": accuracy_score(y_val, predicted),
    "precision": precision_score(y_val, predicted, zero_division=0),
    "recall": recall_score(y_val, predicted, zero_division=0),
    "f1_score": f1_score(y_val, predicted, zero_division=0),
    "roc_auc": roc_auc_score(y_val, probability),
    "best_iteration": model.get_best_iteration()
}

print("\n================ MODEL PERFORMANCE ================")
for key, value in metrics.items():
    print(f"{key:16s}: {value}")
print("====================================================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predicted))

print("\nClassification Report:")
print(classification_report(
    y_val, predicted,
    target_names=["No Flood", "Flood"],
    zero_division=0
))

model.save_model(MODEL_FILE)
pd.DataFrame([metrics]).to_csv("validation_metrics.csv", index=False)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)
importance.to_csv("feature_importance.csv", index=False)

metadata = {
    "model": "CatBoostClassifier",
    "target": TARGET,
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "risk_thresholds": {
        "green": "< 0.33",
        "yellow": "0.33 to < 0.66",
        "red": ">= 0.66"
    },
    "data_warning": "Current supplied dataset is a synthetic prototype."
}

with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nCreated:")
print("  flood_risk_catboost.cbm")
print("  validation_metrics.csv")
print("  feature_importance.csv")
print("  model_metadata.json")
print("\nTraining complete.")
