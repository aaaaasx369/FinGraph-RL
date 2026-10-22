#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C, PPO, DDPG
from finrl.meta.preprocessor.preprocessors import data_split

# Import the custom projection class so SB3 can reconstruct
# the KG-trained policies saved by the original experiment.
from unified_event_memory.finrl_x import run_dynamic_kg_drl as runner

GROUPS = ["clean", "static", "dynamic"]
ALGORITHMS = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
}

def build_panels(embeddings):
    exp = runner.exp

    market = runner.prepare_panel(
        exp.build_market_data()
    )

    static = runner.load_kg(
        embeddings / "daily_static_kgt.csv"
    )
    dynamic = runner.load_kg(
        embeddings / "daily_dynamic_kgt.csv"
    )

    panels = {
        "clean": market,
        "static": runner.attach_kg(market, static),
        "dynamic": runner.attach_kg(market, dynamic),
    }

    features = {
        "clean": list(exp.BASE_FEATURES),
        "static": list(exp.BASE_FEATURES) + runner.KG_COLS,
        "dynamic": list(exp.BASE_FEATURES) + runner.KG_COLS,
    }

    reference = market[["date", "tic"]].reset_index(drop=True)

    for name in GROUPS:
        x = panels[name].sort_values(
            ["date", "tic"]
        ).reset_index(drop=True)

        assert x[["date", "tic"]].equals(reference), (
            f"Market/KG rows mismatch: {name}"
        )

    return panels, features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trained",
        default="results/unified_dynamic_kgt_finrl_x_v1",
    )
    parser.add_argument(
        "--embeddings",
        default="outputs/unified_kgt_finrl_x_v1",
    )
    parser.add_argument(
        "--out",
        default="results/unified_dynamic_kgt_nine_models",
    )

    args = parser.parse_args()

    trained = ROOT / args.trained
    embeddings = ROOT / args.embeddings
    out = ROOT / args.out

    out.mkdir(parents=True, exist_ok=True)

    panels, features = build_panels(embeddings)

    results = []

    for group in GROUPS:
        panel = panels[group]
        feature_list = features[group]

        val_df = data_split(
            panel, "2022-01-01", "2023-01-01"
        )
        test_df = data_split(
            panel, "2023-01-01", "2024-01-01"
        )

        group_out = out / group
        group_out.mkdir(parents=True, exist_ok=True)

        for algorithm, model_class in ALGORITHMS.items():
            print("\n" + "=" * 75)
            print("EVALUATE", group, algorithm)
            print("=" * 75)

            model_path = (
                trained
                / group
                / f"{algorithm.lower()}_model.zip"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Trained model missing: {model_path}"
                )

            model = model_class.load(
                str(model_path),
                device="cuda" if torch.cuda.is_available() else "cpu",
                custom_objects={
                    "features_extractor_class":
                        runner.KGProjectionExtractor,
                },
            )

            # Validate observation compatibility.
            e = runner.environment(val_df, feature_list)
            obs, _ = e.reset()

            if tuple(obs.shape) != tuple(
                model.observation_space.shape
            ):
                raise RuntimeError(
                    f"Observation shape mismatch: "
                    f"{group}/{algorithm}, "
                    f"env={obs.shape}, "
                    f"model={model.observation_space.shape}"
                )

            for split_name, df in (
                ("validation", val_df),
                ("test", test_df),
            ):
                returns, weights = runner.evaluate(
                    model, df, feature_list
                )

                m = runner.exp.calculate_metrics(returns)

                row = {
                    "group": group,
                    "algorithm": algorithm,
                    "split": split_name,
                    **m,
                }

                results.append(row)

                returns.to_csv(
                    group_out
                    / f"{algorithm.lower()}_{split_name}_returns.csv",
                    index=False,
                )

                weights.to_csv(
                    group_out
                    / f"{algorithm.lower()}_{split_name}_weights.csv",
                )

                print(
                    split_name,
                    "value=", round(m["final_value"], 2),
                    "multiple=", round(m["final_multiple"], 6),
                    "sharpe=", round(m["sharpe"], 6),
                    "mdd=", round(m["max_drawdown"], 6),
                )

            pd.DataFrame(results).to_csv(
                out / "all_model_validation_test.csv",
                index=False,
            )

    full = pd.DataFrame(results)

    val = full[
        full["split"] == "validation"
    ].copy()

    test = full[
        full["split"] == "test"
    ].copy()

    val.to_csv(
        out / "nine_models_2022_validation.csv",
        index=False,
    )

    test.to_csv(
        out / "nine_models_2023_test.csv",
        index=False,
    )

    comparison = val.merge(
        test,
        on=["group", "algorithm"],
        suffixes=("_2022", "_2023"),
        validate="one_to_one",
    )

    comparison = comparison[
        [
            "group",
            "algorithm",
            "final_multiple_2022",
            "sharpe_2022",
            "max_drawdown_2022",
            "final_multiple_2023",
            "sharpe_2023",
            "max_drawdown_2023",
        ]
    ]

    comparison.to_csv(
        out / "nine_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 85)
    print("ALL NINE MODELS: 2022 VALIDATION / 2023 TEST")
    print("=" * 85)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Same-algorithm differences in the 2023 test.
    pivot = test.pivot(
        index="algorithm",
        columns="group",
        values=["final_multiple", "sharpe"],
    )

    delta = pd.DataFrame({
        "algorithm": pivot.index,
        "dynamic_minus_clean_multiple":
            pivot[("final_multiple", "dynamic")].values
            - pivot[("final_multiple", "clean")].values,
        "dynamic_minus_static_multiple":
            pivot[("final_multiple", "dynamic")].values
            - pivot[("final_multiple", "static")].values,
        "dynamic_minus_clean_sharpe":
            pivot[("sharpe", "dynamic")].values
            - pivot[("sharpe", "clean")].values,
        "dynamic_minus_static_sharpe":
            pivot[("sharpe", "dynamic")].values
            - pivot[("sharpe", "static")].values,
    })

    delta.to_csv(
        out / "dynamic_kg_improvement_by_algorithm.csv",
        index=False,
    )

    print("\n" + "=" * 85)
    print("SAME-ALGORITHM DYNAMIC KG IMPROVEMENTS")
    print("=" * 85)
    print(delta.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    print("\nNINE MODEL EVALUATION PASS")
    print("Results:", out)


if __name__ == "__main__":
    main()
