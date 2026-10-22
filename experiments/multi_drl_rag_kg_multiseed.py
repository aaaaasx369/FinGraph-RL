#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Prevent argparse in imported experiment from seeing our args.
sys.argv = [
    "multi_drl_rag_kg_compare.py"
]

import experiments.multi_drl_rag_kg_compare as exp


# ============================================================
# Configuration
# ============================================================

SEEDS = [
    42,
    123,
    2026,
    7,
    999,
]

OUT_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_kg_multiseed"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


METRICS = [
    "final_multiple",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
]


# ============================================================
# Build data ONCE
# ============================================================

print("=" * 80)
print("BUILDING COMMON DATA")
print("=" * 80)


# Make audit output valid during data preparation.
exp.RESULT_ROOT = OUT_ROOT

market = exp.build_market_data()

rag_panel, common_start = (
    exp.merge_rag_features(
        market
    )
)

rag_kg_panel = (
    exp.build_rag_kg_panel(
        rag_panel
    )
)


# ============================================================
# Hard consistency checks
# ============================================================

exp.validate_panel(
    rag_panel,
    "RAG",
)

exp.validate_panel(
    rag_kg_panel,
    "RAG_KG",
)


rag_dates = (
    pd.to_datetime(
        rag_panel["date"]
    )
    .dt.strftime("%Y-%m-%d")
    .tolist()
)

kg_dates = (
    pd.to_datetime(
        rag_kg_panel["date"]
    )
    .dt.strftime("%Y-%m-%d")
    .tolist()
)


if rag_dates != kg_dates:
    raise RuntimeError(
        "RAG and RAG_KG dates differ."
    )


rag_feature_list = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
)

rag_kg_feature_list = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
    + exp.KG_FEATURES
)


print("\nCommon start:", common_start)
print(
    "RAG feature count:",
    len(rag_feature_list),
)
print(
    "RAG+KG feature count:",
    len(rag_kg_feature_list),
)


# ============================================================
# Multi-seed experiment
#
# Resume behavior:
# - if best_model_test_summary.csv exists -> reuse it
# - otherwise run that pipeline
# - save combined progress after every completed/reused pipeline
# ============================================================

rows = []


def load_existing_result(
    seed,
    feature_set,
    pipeline_name,
    summary_path,
):
    """
    Load one completed pipeline result from disk.
    """

    df = pd.read_csv(
        summary_path
    )

    if len(df) != 1:
        raise RuntimeError(
            f"Unexpected summary rows in {summary_path}: {len(df)}"
        )

    row = df.iloc[0].to_dict()

    return {
        "seed":
            seed,

        "feature_set":
            feature_set,

        **row,
    }


def save_progress(
    rows,
):
    """
    Save current progress immediately so interruption is safe.
    """

    pd.DataFrame(
        rows
    ).to_csv(
        OUT_ROOT
        / "all_seed_results.csv",
        index=False,
    )


for run_seed in SEEDS:

    print("\n")
    print("#" * 80)
    print(
        f"PAIRED SEED = {run_seed}"
    )
    print("#" * 80)


    exp.MODEL_SEEDS = {
        "A2C":
            run_seed,

        "PPO":
            run_seed,

        "DDPG":
            run_seed,
    }


    seed_root = (
        OUT_ROOT
        / f"seed_{run_seed}"
    )

    seed_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # RAG
    # ========================================================

    rag_summary_path = (
        seed_root
        / "rag"
        / "best_model_test_summary.csv"
    )


    if rag_summary_path.exists():

        print(
            f"\n[RESUME] seed {run_seed} RAG already DONE -> skipping training"
        )

        rag_row = load_existing_result(
            seed=run_seed,
            feature_set="RAG",
            pipeline_name="rag",
            summary_path=rag_summary_path,
        )

    else:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"SEED {run_seed}: RAG"
        )

        print(
            "=" * 80
        )


        exp.RESULT_ROOT = (
            seed_root
        )


        rag_result = (
            exp.run_pipeline(
                name="rag",
                panel=rag_panel,
                feature_list=rag_feature_list,
                common_start=common_start,
            )
        )


        rag_row = {
            "seed":
                run_seed,

            "feature_set":
                "RAG",

            **rag_result,
        }


    rows.append(
        rag_row
    )

    save_progress(
        rows
    )


    # ========================================================
    # RAG + KG
    # ========================================================

    kg_summary_path = (
        seed_root
        / "rag_kg"
        / "best_model_test_summary.csv"
    )


    if kg_summary_path.exists():

        print(
            f"\n[RESUME] seed {run_seed} RAG_KG already DONE -> skipping training"
        )

        kg_row = load_existing_result(
            seed=run_seed,
            feature_set="RAG_KG",
            pipeline_name="rag_kg",
            summary_path=kg_summary_path,
        )

    else:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"SEED {run_seed}: RAG_KG"
        )

        print(
            "=" * 80
        )


        exp.RESULT_ROOT = (
            seed_root
        )


        kg_result = (
            exp.run_pipeline(
                name="rag_kg",
                panel=rag_kg_panel,
                feature_list=rag_kg_feature_list,
                common_start=common_start,
            )
        )


        kg_row = {
            "seed":
                run_seed,

            "feature_set":
                "RAG_KG",

            **kg_result,
        }


    rows.append(
        kg_row
    )

    save_progress(
        rows
    )


# ============================================================
# Full results
# ============================================================

results = pd.DataFrame(
    rows
)

results.to_csv(
    OUT_ROOT
    / "all_seed_results.csv",
    index=False,
)


# ============================================================
# Mean ± Std
# ============================================================

summary_rows = []

for feature_set in [
    "RAG",
    "RAG_KG",
]:

    sub = results[
        results[
            "feature_set"
        ]
        == feature_set
    ]

    row = {
        "feature_set":
            feature_set,

        "n_seeds":
            len(sub),
    }

    for metric in METRICS:

        row[
            metric + "_mean"
        ] = float(
            sub[
                metric
            ].mean()
        )

        row[
            metric + "_std"
        ] = float(
            sub[
                metric
            ].std(
                ddof=1
            )
        )

    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_ROOT
    / "summary_mean_std.csv",
    index=False,
)


# ============================================================
# Paired RAG_KG - RAG deltas
# ============================================================

rag = (
    results[
        results[
            "feature_set"
        ]
        == "RAG"
    ]
    .set_index(
        "seed"
    )
)

kg = (
    results[
        results[
            "feature_set"
        ]
        == "RAG_KG"
    ]
    .set_index(
        "seed"
    )
)


delta_rows = []


for seed in SEEDS:

    row = {
        "seed":
            seed,

        "rag_selected_model":
            rag.loc[
                seed,
                "selected_model",
            ],

        "rag_kg_selected_model":
            kg.loc[
                seed,
                "selected_model",
            ],
    }

    for metric in METRICS:

        row[
            "delta_" + metric
        ] = float(
            kg.loc[
                seed,
                metric,
            ]
            -
            rag.loc[
                seed,
                metric,
            ]
        )

    delta_rows.append(
        row
    )


deltas = pd.DataFrame(
    delta_rows
)

deltas.to_csv(
    OUT_ROOT
    / "paired_deltas.csv",
    index=False,
)


# ============================================================
# Delta mean / std / positive count
# ============================================================

delta_summary = {
    "n_seeds":
        len(SEEDS),
}


for metric in METRICS:

    c = (
        "delta_"
        + metric
    )

    values = deltas[
        c
    ]


    delta_summary[
        c + "_mean"
    ] = float(
        values.mean()
    )

    delta_summary[
        c + "_std"
    ] = float(
        values.std(
            ddof=1
        )
    )


    # For max drawdown:
    # positive delta means less-negative drawdown = improvement.
    delta_summary[
        c + "_positive_count"
    ] = int(
        (
            values > 0
        ).sum()
    )


pd.DataFrame(
    [
        delta_summary
    ]
).to_csv(
    OUT_ROOT
    / "paired_delta_summary.csv",
    index=False,
)


# ============================================================
# Print meeting-friendly tables
# ============================================================

print("\n")
print("=" * 100)
print("5-SEED MEAN ± STD")
print("=" * 100)


for _, row in summary.iterrows():

    print(
        f"\n{row['feature_set']}"
    )

    print(
        "  Final multiple : "
        f"{row['final_multiple_mean']:.6f}"
        " ± "
        f"{row['final_multiple_std']:.6f}"
    )

    print(
        "  Annual return  : "
        f"{row['annual_return_mean']:.2%}"
        " ± "
        f"{row['annual_return_std']:.2%}"
    )

    print(
        "  Annual vol     : "
        f"{row['annual_volatility_mean']:.2%}"
        " ± "
        f"{row['annual_volatility_std']:.2%}"
    )

    print(
        "  Sharpe         : "
        f"{row['sharpe_mean']:.4f}"
        " ± "
        f"{row['sharpe_std']:.4f}"
    )

    print(
        "  Max drawdown   : "
        f"{row['max_drawdown_mean']:.2%}"
        " ± "
        f"{row['max_drawdown_std']:.2%}"
    )


print("\n")
print("=" * 100)
print("PAIRED DELTAS: RAG_KG - RAG")
print("=" * 100)

print(
    deltas[
        [
            "seed",
            "rag_selected_model",
            "rag_kg_selected_model",
            "delta_final_multiple",
            "delta_annual_return",
            "delta_annual_volatility",
            "delta_sharpe",
            "delta_max_drawdown",
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("MEAN PAIRED DELTA")
print("=" * 100)

for metric in METRICS:

    c = (
        "delta_"
        + metric
    )

    print(
        f"{metric:20s}: "
        f"{deltas[c].mean():+.6f}"
        " ± "
        f"{deltas[c].std(ddof=1):.6f}"
        " | positive seeds = "
        f"{(deltas[c] > 0).sum()}"
        f"/{len(SEEDS)}"
    )


print(
    "\nSaved to:",
    OUT_ROOT
)
