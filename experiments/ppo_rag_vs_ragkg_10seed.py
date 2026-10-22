#!/usr/bin/env python3

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# prevent argparse in imported experiment from reading this script args
sys.argv = ["multi_drl_rag_kg_compare.py"]

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
    314,
    2718,
    808,
    2024,
    77,
]

OUT_ROOT = (
    ROOT
    / "results"
    / "ppo_rag_vs_ragkg_10seed"
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
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Train PPO only
# ============================================================

def train_ppo_only(
    train_df,
    feature_list,
    seed,
):

    set_seed(seed)

    e_train = exp.StockPortfolioEnv(
        df=train_df,
        **exp.make_env_kwargs(
            feature_list
        ),
    )

    env_train, _ = (
        e_train.get_sb_env()
    )

    agent = exp.DRLAgent(
        env=env_train
    )

    model = exp.train_ppo(
        agent,
        exp.USE_GPU,
    )

    return model


# ============================================================
# Run one PPO experiment
# ============================================================

def run_ppo_pipeline(
    name,
    panel,
    feature_list,
    common_start,
    seed,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_string = (
        common_start
        .strftime(
            "%Y-%m-%d"
        )
    )

    train_df = exp.data_split(
        panel,
        start_string,
        exp.TRAIN_END,
    )

    validation_df = exp.data_split(
        panel,
        exp.TRAIN_END,
        exp.VALIDATION_END,
    )

    test_df = exp.data_split(
        panel,
        exp.VALIDATION_END,
        exp.TEST_END,
    )


    print("\n" + "=" * 80)
    print(
        f"{name} | PPO | seed={seed}"
    )
    print("=" * 80)

    print(
        "TRAIN:",
        train_df["date"].min(),
        "->",
        train_df["date"].max(),
    )

    print(
        "VALIDATION:",
        validation_df["date"].min(),
        "->",
        validation_df["date"].max(),
    )

    print(
        "TEST:",
        test_df["date"].min(),
        "->",
        test_df["date"].max(),
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_ppo_only(
        train_df,
        feature_list,
        seed,
    )

    model.save(
        str(
            output_dir
            / "ppo_model"
        )
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_returns, val_weights = (
        exp.evaluate_model(
            model,
            validation_df,
            feature_list,
        )
    )

    val_metrics = exp.calculate_metrics(
        val_returns
    )

    val_returns.to_csv(
        output_dir
        / "validation_returns.csv",
        index=False,
    )

    val_weights.to_csv(
        output_dir
        / "validation_weights.csv",
    )

    pd.DataFrame(
        [
            val_metrics
        ]
    ).to_csv(
        output_dir
        / "validation_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_returns, test_weights = (
        exp.evaluate_model(
            model,
            test_df,
            feature_list,
        )
    )

    test_metrics = (
        exp.calculate_metrics(
            test_returns
        )
    )

    test_returns.to_csv(
        output_dir
        / "test_returns.csv",
        index=False,
    )

    test_weights.to_csv(
        output_dir
        / "test_weights.csv",
    )

    result = {
        "seed":
            seed,

        "feature_set":
            name,

        "model":
            "PPO",

        **test_metrics,
    }

    pd.DataFrame(
        [
            result
        ]
    ).to_csv(
        output_dir
        / "test_summary.csv",
        index=False,
    )


    print(
        "Final multiple:",
        f"{test_metrics['final_multiple']:.6f}"
    )

    print(
        "Sharpe:",
        f"{test_metrics['sharpe']:.4f}"
    )

    print(
        "Max drawdown:",
        f"{test_metrics['max_drawdown']:.2%}"
    )


    return result


# ============================================================
# Build common data once
# ============================================================

print("=" * 80)
print("BUILD COMMON DATA")
print("=" * 80)

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


rag_features = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
)

rag_kg_features = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
    + exp.KG_FEATURES
)


# ============================================================
# Hard consistency checks
# ============================================================

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


print(
    "Common start:",
    common_start
)

print(
    "RAG feature count:",
    len(
        rag_features
    )
)

print(
    "RAG_KG feature count:",
    len(
        rag_kg_features
    )
)


# ============================================================
# Resume-aware paired loop
# ============================================================

rows = []


for seed in SEEDS:

    print("\n")
    print("#" * 100)
    print(
        f"PAIRED PPO SEED = {seed}"
    )
    print("#" * 100)


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_dir = (
        OUT_ROOT
        / f"seed_{seed}"
        / "rag"
    )

    rag_summary = (
        rag_dir
        / "test_summary.csv"
    )


    if rag_summary.exists():

        print(
            f"[RESUME] seed={seed} RAG DONE -> load"
        )

        rag_row = (
            pd.read_csv(
                rag_summary
            )
            .iloc[0]
            .to_dict()
        )

    else:

        rag_row = run_ppo_pipeline(
            name="RAG",
            panel=rag_panel,
            feature_list=rag_features,
            common_start=common_start,
            seed=seed,
            output_dir=rag_dir,
        )


    rows.append(
        rag_row
    )


    # --------------------------------------------------------
    # RAG + KG
    # --------------------------------------------------------

    kg_dir = (
        OUT_ROOT
        / f"seed_{seed}"
        / "rag_kg"
    )

    kg_summary = (
        kg_dir
        / "test_summary.csv"
    )


    if kg_summary.exists():

        print(
            f"[RESUME] seed={seed} RAG_KG DONE -> load"
        )

        kg_row = (
            pd.read_csv(
                kg_summary
            )
            .iloc[0]
            .to_dict()
        )

    else:

        kg_row = run_ppo_pipeline(
            name="RAG_KG",
            panel=rag_kg_panel,
            feature_list=rag_kg_features,
            common_start=common_start,
            seed=seed,
            output_dir=kg_dir,
        )


    rows.append(
        kg_row
    )


    pd.DataFrame(
        rows
    ).to_csv(
        OUT_ROOT
        / "all_results.csv",
        index=False,
    )


# ============================================================
# Results
# ============================================================

results = pd.DataFrame(
    rows
)

results.to_csv(
    OUT_ROOT
    / "all_results.csv",
    index=False,
)


# ============================================================
# Mean ± std
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
# Paired deltas
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
# Paired summary
# ============================================================

paired_summary = {
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

    paired_summary[
        c + "_mean"
    ] = float(
        values.mean()
    )

    paired_summary[
        c + "_std"
    ] = float(
        values.std(
            ddof=1
        )
    )

    paired_summary[
        c + "_median"
    ] = float(
        values.median()
    )

    paired_summary[
        c + "_positive_count"
    ] = int(
        (
            values > 0
        ).sum()
    )


pd.DataFrame(
    [
        paired_summary
    ]
).to_csv(
    OUT_ROOT
    / "paired_summary.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\n")
print("=" * 100)
print("PPO 10-SEED MEAN ± STD")
print("=" * 100)

for _, row in summary.iterrows():

    print(
        f"\n{row['feature_set']}"
    )

    print(
        "Final multiple : "
        f"{row['final_multiple_mean']:.6f}"
        " ± "
        f"{row['final_multiple_std']:.6f}"
    )

    print(
        "Annual return  : "
        f"{row['annual_return_mean']:.2%}"
        " ± "
        f"{row['annual_return_std']:.2%}"
    )

    print(
        "Annual vol     : "
        f"{row['annual_volatility_mean']:.2%}"
        " ± "
        f"{row['annual_volatility_std']:.2%}"
    )

    print(
        "Sharpe         : "
        f"{row['sharpe_mean']:.4f}"
        " ± "
        f"{row['sharpe_std']:.4f}"
    )

    print(
        "Max drawdown   : "
        f"{row['max_drawdown_mean']:.2%}"
        " ± "
        f"{row['max_drawdown_std']:.2%}"
    )


print("\n")
print("=" * 100)
print("PAIRED DELTAS: RAG_KG - RAG")
print("=" * 100)

print(
    deltas.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("PAIRED SUMMARY")
print("=" * 100)

for metric in METRICS:

    c = (
        "delta_"
        + metric
    )

    print(
        f"{metric:20s}: "
        f"mean={deltas[c].mean():+.6f} | "
        f"median={deltas[c].median():+.6f} | "
        f"std={deltas[c].std(ddof=1):.6f} | "
        f"positive={int((deltas[c] > 0).sum())}/{len(SEEDS)}"
    )


print(
    "\nSaved to:",
    OUT_ROOT
)
