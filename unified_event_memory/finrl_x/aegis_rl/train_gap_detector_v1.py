import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
)
from joblib import dump

DATA = Path("results/aegis_rl_v1/gap_dataset_pit_v2")
OUT = Path("results/aegis_rl_v1/gap_detector_v1")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "head_degree",
    "tail_degree",
    "head_relation_degree",
    "tail_relation_degree",
    "pair_degree",
    "relation_count",
    "head_age_days",
    "tail_age_days",
    "head_relation_seen",
    "tail_relation_seen",
]

def load(split):
    df = pd.read_csv(DATA / f"{split}.csv")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError(f"{split}: nonfinite features")

    return df, X, y

print("=" * 65)
print("AEGIS-RL BLIND-SPOT DETECTOR V1")
print("=" * 65)

train_df, X_train, y_train = load("train")
val_df, X_val, y_val = load("val")
test_df, X_test, y_test = load("test")

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

def evaluate(y, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)

    return {
        "AUC": float(roc_auc_score(y, probability)),
        "AP": float(average_precision_score(y, probability)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "Recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "LogLoss": float(
            log_loss(y, probability, labels=[0, 1])
        ),
    }

# Select hyperparameters using validation only.
configs = [
    {"max_iter": 80, "max_leaf_nodes": 15},
    {"max_iter": 120, "max_leaf_nodes": 31},
    {"max_iter": 160, "max_leaf_nodes": 31},
]

best_model = None
best_score = -1
best_config = None

for config in configs:
    print("\nTraining:", config, flush=True)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=config["max_iter"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_probability = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, val_probability)

    print(
        "Validation AUC:",
        round(metrics["AUC"], 6),
        "AP:",
        round(metrics["AP"], 6),
        "F1:",
        round(metrics["F1"], 6),
    )

    if metrics["AP"] > best_score:
        best_score = metrics["AP"]
        best_model = model
        best_config = config

print("\n" + "=" * 65)
print("BEST VALIDATION CONFIG:", best_config)
print("=" * 65)

# Freeze chosen configuration before evaluating test.
val_probability = best_model.predict_proba(X_val)[:, 1]
test_probability = best_model.predict_proba(X_test)[:, 1]

val_metrics = evaluate(y_val, val_probability)
test_metrics = evaluate(y_test, test_probability)

print("\nVALIDATION")
for key, value in val_metrics.items():
    print(f"{key:12s}: {value:.6f}")

print("\nTEST")
for key, value in test_metrics.items():
    print(f"{key:12s}: {value:.6f}")

# Export per-triple predictions for later error analysis.
predictions = test_df[
    [
        "head_id",
        "relation_id",
        "tail_id",
        "available_at",
        "label",
    ]
].copy()

predictions["probability"] = test_probability
predictions["prediction"] = (
    test_probability >= 0.5
).astype(int)

predictions.to_csv(
    OUT / "test_predictions.csv",
    index=False,
)

dump(best_model, OUT / "model.joblib")

results = {
    "model": "HistGradientBoostingClassifier",
    "best_config": best_config,
    "features": FEATURES,
    "train_years": [2018, 2020],
    "validation_year": 2021,
    "test_years": [2022, 2023],
    "validation": val_metrics,
    "test": test_metrics,
}

with (OUT / "metrics.json").open("w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT.resolve())
print("Model: model.joblib")
print("Metrics: metrics.json")
print("Predictions: test_predictions.csv")
print("\nIMPORTANT:")
print("This evaluates sampled triple reconstruction.")
print("It does not measure portfolio performance.")
