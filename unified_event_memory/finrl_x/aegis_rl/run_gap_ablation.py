import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path("results/aegis_rl_v1/gap_dataset_pit_v2")
OUT = Path("results/aegis_rl_v1/gap_ablation")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
test = pd.read_csv(DATA / "test.csv")

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

GROUPS = {
    "Full": FEATURES,

    "Degree_Only": [
        "head_degree",
        "tail_degree",
        "head_relation_degree",
        "tail_relation_degree",
        "pair_degree",
        "relation_count",
    ],

    "Age_Only": [
        "head_age_days",
        "tail_age_days",
    ],

    "Relation_Only": [
        "head_relation_degree",
        "tail_relation_degree",
        "relation_count",
        "head_relation_seen",
        "tail_relation_seen",
    ],

    "No_Degree": [
        "head_age_days",
        "tail_age_days",
        "head_relation_seen",
        "tail_relation_seen",
    ],

    "No_Relation": [
        "head_degree",
        "tail_degree",
        "pair_degree",
        "head_age_days",
        "tail_age_days",
    ],
}

results = []

for name, cols in GROUPS.items():

    print("Training:", name, flush=True)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    model.fit(train[cols], train["label"])

    for split_name, df in [
        ("Validation", val),
        ("Test", test),
    ]:
        p = model.predict_proba(df[cols])[:, 1]

        results.append({
            "model": name,
            "split": split_name,
            "AUC": roc_auc_score(df["label"], p),
            "AP": average_precision_score(df["label"], p),
        })

result = pd.DataFrame(results)

result.to_csv(
    OUT / "ablation_results.csv",
    index=False,
)

print("\n===== ABLATION RESULTS =====")

print(
    result.pivot(
        index="model",
        columns="split",
        values="AUC"
    ).round(6).to_string()
)

print("\nSaved:", OUT)
