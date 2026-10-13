#!/usr/bin/env python3
"""
V23 Graph-to-FinRL bridge.

Source:
  V20 static fusion
  V23 temporal event graph features

Destination:
  outputs/v23/finrl/

Keeps V21 observation dimensions and schema.
Does not modify V18, V20, V21 or V22.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/v20/fusion"
OUT = ROOT / "outputs/v23/finrl"

TIERS = ["conservative", "inclusive"]

GRAPH_FEATURES = [
    "tkg_edge_count",
    "tkg_reobserved_edges",
    "tkg_new_since_previous_market_day",
    "tkg_reobserved_since_previous_market_day",
    "tkg_company_edges",
    "tkg_supplies_to",
    "tkg_customer_of",
    "tkg_competes_with",
    "tkg_same_sector",
    "tkg_partners_with",
    "tkg_other",
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    schema_path = SRC / "feature_sets.json"

    schema = json.loads(
        schema_path.read_text(encoding="utf-8")
    )

    static = schema["FULL_STATIC"]
    dynamic = schema["FULL_DYNAMIC"]

    if dynamic[:len(static)] != static:
        raise ValueError(
            "V20 feature schema is not prefix aligned"
        )

    dyn_columns = dynamic[len(static):]

    if len(dyn_columns) != len(GRAPH_FEATURES):
        raise ValueError(
            "Expected 11 V21 dynamic columns"
        )

    report = {}

    for tier in TIERS:
        old_path = SRC / f"dataset_{tier}.csv"

        graph_path = (
            ROOT / "outputs/v23" / tier
            / "daily_graph_features.csv"
        )

        if not old_path.exists():
            raise FileNotFoundError(old_path)
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)

        old = pd.read_csv(old_path)
        graph = pd.read_csv(graph_path)

        for frame in [old, graph]:
            frame["date"] = pd.to_datetime(
                frame["date"], errors="raise"
            ).dt.normalize()
            frame["ticker"] = (
                frame["ticker"]
                .astype(str).str.upper()
            )

            if frame.duplicated(
                ["date", "ticker"]
            ).any():
                raise ValueError(
                    "Duplicate date/ticker"
                )

        missing = [
            c for c in GRAPH_FEATURES
            if c not in graph
        ]
        if missing:
            raise ValueError(
                f"Missing graph features: {missing}"
            )

        old = old.drop(
            columns=dyn_columns
        )

        combined = old.merge(
            graph[
                ["date", "ticker"] + GRAPH_FEATURES
            ],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
            indicator=True
        )

        if not combined["_merge"].eq("both").all():
            raise ValueError(
                f"{tier}: incomplete V23 market panel"
            )

        combined.drop(
            columns=["_merge"], inplace=True
        )

        scaler = {}
        train = combined["split"].eq("train")

        # The data source is a candidate-level historical
        # graph; no future backfilling.
        for raw_col, output_col in zip(
            GRAPH_FEATURES, dyn_columns
        ):
            x = pd.to_numeric(
                combined[raw_col],
                errors="raise"
            ).astype(float)

            if not np.isfinite(x).all():
                raise ValueError(
                    f"Nonfinite graph input {raw_col}"
                )
            if (x < 0).any():
                raise ValueError(
                    f"Negative count {raw_col}"
                )

            transformed = np.log1p(x)
            fitted = transformed.loc[train]

            if fitted.empty:
                raise ValueError(
                    "Empty train split"
                )

            mu = float(fitted.mean())
            sigma = float(fitted.std(ddof=0))

            if sigma <= 1e-12 or not np.isfinite(sigma):
                sigma = 1.0

            combined[output_col] = (
                (transformed - mu) / sigma
            ).astype(float)

            scaler[raw_col] = {
                "model_column": output_col,
                "train_mean": mu,
                "train_std": sigma,
            }

        # Every previous feature set remains supported.
        for condition, cols in schema.items():
            missing = [
                c for c in cols
                if c not in combined
            ]
            if missing:
                raise ValueError(
                    f"{tier}/{condition}: missing {missing}"
                )

            arr = combined[cols].to_numpy(
                dtype=float
            )
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{tier}/{condition}: nonfinite"
                )

        combined.sort_values(
            ["date", "ticker"],
            inplace=True
        )

        combined.to_csv(
            OUT / f"dataset_{tier}.csv",
            index=False
        )

        (OUT / f"graph_scaler_{tier}.json").write_text(
            json.dumps(
                scaler, indent=2
            ),
            encoding="utf-8"
        )

        report[tier] = {
            "rows": len(combined),
            "static_feature_dims": len(static),
            "dynamic_feature_dims": len(dynamic),
            "graph_source_columns": GRAPH_FEATURES,
            "reused_model_columns": dyn_columns,
            "strict_pit_approved": False,
        }

    (OUT / "feature_sets.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8"
    )

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n=== V23 FINRL BRIDGE READY ===")
    print(json.dumps(report, indent=2))
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
