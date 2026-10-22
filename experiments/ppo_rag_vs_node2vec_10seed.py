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

sys.argv = ["multi_drl_rag_kg_compare.py"]

import experiments.multi_drl_rag_kg_compare as exp


SEEDS = [
    42, 123, 2026, 7, 999,
    314, 2718, 808, 2024, 77,
]

OUT_ROOT = (
    ROOT
    / "results"
    / "ppo_rag_vs_node2vec_10seed"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

N2V_PATH = (
    ROOT
    / "kg"
    / "output"
    / "finreflectkg_company_node2vec_32d.csv"
)

N2V_FEATURES = [
    f"n2v_{i:02d}"
    for i in range(32)
]

METRICS = [
    "final_multiple",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
]


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CRITICAL:
    # train_ppo() reads this value internally.
    exp.MODEL_SEEDS["PPO"] = seed


# ============================================================
# Build Node2Vec panel
# ============================================================

def build_node2vec_panel(rag_panel):

    n2v = pd.read_csv(
        N2V_PATH
    )

    # KG year Y will be used for trading year Y+1.
    n2v["trading_year"] = (
        n2v["year"].astype(int)
        + 1
    )

    n2v = n2v.rename(
        columns={
            "ticker": "tic",
            "year": "n2v_source_year",
        }
    )

    keep = (
        [
            "tic",
            "trading_year",
            "n2v_source_year",
            "n2v_available",
        ]
        + N2V_FEATURES
    )

    n2v = n2v[
        keep
    ].copy()

    n2v["tic"] = (
        n2v["tic"]
        .astype(str)
        .str.upper()
    )


    out = rag_panel.copy()

    out["date_dt"] = pd.to_datetime(
        out["date"]
    )

    out["tic"] = (
        out["tic"]
        .astype(str)
        .str.upper()
    )

    out["trading_year"] = (
        out["date_dt"].dt.year
    )


    before_rows = len(out)


    out = out.merge(
        n2v,
        on=[
            "tic",
            "trading_year",
        ],
        how="left",
        validate="many_to_one",
    )


    if len(out) != before_rows:
        raise RuntimeError(
            "Node2Vec merge changed row count."
        )


    out["n2v_available"] = (
        out["n2v_available"]
        .fillna(0)
        .astype(float)
    )


    for c in N2V_FEATURES:
        out[c] = (
            pd.to_numeric(
                out[c],
                errors="coerce",
            )
            .fillna(0.0)
        )


    # ========================================================
    # Training-only normalization
    #
    # Fit using only unique AVAILABLE ticker/source-year
    # embeddings observed before TRAIN_END.
    #
    # This avoids:
    # 1. test leakage
    # 2. 2018 missing-zero rows contaminating scaler
    # 3. weighting an annual embedding hundreds of times
    # ========================================================

    fit = out[
        (out["date_dt"] < pd.Timestamp(exp.TRAIN_END))
        &
        (out["n2v_available"] > 0.5)
    ].copy()


    fit = (
        fit[
            [
                "tic",
                "n2v_source_year",
            ]
            + N2V_FEATURES
        ]
        .drop_duplicates(
            subset=[
                "tic",
                "n2v_source_year",
            ]
        )
    )


    print(
        "Node2Vec scaler fit ticker-years:",
        len(fit),
    )


    for c in N2V_FEATURES:

        mu = float(
            fit[c].mean()
        )

        sd = float(
            fit[c].std(
                ddof=0
            )
        )

        if (
            not np.isfinite(sd)
            or sd < 1e-12
        ):
            sd = 1.0


        out[c] = (
            out[c] - mu
        ) / sd


    # Missing embedding must remain neutral zero AFTER scaling.
    missing = (
        out["n2v_available"]
        < 0.5
    )

    out.loc[
        missing,
        N2V_FEATURES,
    ] = 0.0


    # ========================================================
    # Leakage checks
    # ========================================================

    available = (
        out["n2v_available"]
        > 0.5
    )

    if available.any():

        source_year = (
            out.loc[
                available,
                "n2v_source_year",
            ]
            .astype(int)
        )

        trading_year = (
            out.loc[
                available,
                "trading_year",
            ]
            .astype(int)
        )

        if not (
            source_year
            ==
            trading_year - 1
        ).all():

            raise RuntimeError(
                "Node2Vec year leakage detected."
            )


    # 2018 must have no embedding.
    y2018 = (
        out["trading_year"]
        == 2018
    )

    if y2018.any():

        if (
            out.loc[
                y2018,
                "n2v_available",
            ].max()
            != 0
        ):
            raise RuntimeError(
                "2018 unexpectedly has Node2Vec features."
            )


    print("\nNode2Vec availability by trading year:")

    print(
        out.groupby(
            "trading_year"
        )[
            "n2v_available"
        ].mean()
    )


    print(
        "\nSource-year mapping:"
    )

    print(
        out.loc[
            available,
            [
                "trading_year",
                "n2v_source_year",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "trading_year"
        )
        .to_string(
            index=False
        )
    )


    # Restore original date format expected downstream.
    out["date"] = (
        out["date_dt"]
        .dt.strftime("%Y-%m-%d")
    )


    return out


# ============================================================
# PPO only
# ============================================================

def train_ppo_only(
    train_df,
    feature_list,
    seed,
):

    set_seed(seed)

    print(
        f"PPO effective seed = "
        f"{exp.MODEL_SEEDS['PPO']}"
    )


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
# Run one PPO pipeline
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
        common_start.strftime(
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


    print("\n" + "=" * 100)
    print(
        f"{name} | PPO | seed={seed}"
    )
    print("=" * 100)


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


    # Validation retained for audit.
    val_returns, val_weights = (
        exp.evaluate_model(
            model,
            validation_df,
            feature_list,
        )
    )


    val_metrics = (
        exp.calculate_metrics(
            val_returns
        )
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
        [val_metrics]
    ).to_csv(
        output_dir
        / "validation_summary.csv",
        index=False,
    )


    # Untouched 2023 test.
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
        [result]
    ).to_csv(
        output_dir
        / "test_summary.csv",
        index=False,
    )


    print(
        f"Final multiple : "
        f"{test_metrics['final_multiple']:.6f}"
    )

    print(
        f"Sharpe         : "
        f"{test_metrics['sharpe']:.4f}"
    )

    print(
        f"Max drawdown   : "
        f"{test_metrics['max_drawdown']:.2%}"
    )


    return result


# ============================================================
# Build common data
# ============================================================

print("=" * 100)
print("BUILD COMMON DATA")
print("=" * 100)

exp.RESULT_ROOT = OUT_ROOT


market = exp.build_market_data()


rag_panel, common_start = (
    exp.merge_rag_features(
        market
    )
)


node2vec_panel = (
    build_node2vec_panel(
        rag_panel
    )
)


RAG_FEATURES = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
)


NODE2VEC_FEATURES = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
    + N2V_FEATURES
    + ["n2v_available"]
)


print(
    "\nRAG feature count:",
    len(RAG_FEATURES),
)

print(
    "RAG+Node2Vec feature count:",
    len(NODE2VEC_FEATURES),
)


# ============================================================
# Paired 10-seed experiment
# ============================================================

rows = []


for seed in SEEDS:

    print("\n")
    print("#" * 100)
    print(
        f"PAIRED PPO SEED = {seed}"
    )
    print("#" * 100)


    seed_root = (
        OUT_ROOT
        / f"seed_{seed}"
    )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_dir = (
        seed_root
        / "rag"
    )

    rag_summary = (
        rag_dir
        / "test_summary.csv"
    )


    if rag_summary.exists():

        print(
            f"[RESUME] seed={seed} "
            f"RAG DONE"
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
            feature_list=RAG_FEATURES,
            common_start=common_start,
            seed=seed,
            output_dir=rag_dir,
        )


    rows.append(
        rag_row
    )


    # --------------------------------------------------------
    # RAG + Node2Vec
    # --------------------------------------------------------

    n2v_dir = (
        seed_root
        / "rag_node2vec"
    )

    n2v_summary = (
        n2v_dir
        / "test_summary.csv"
    )


    if n2v_summary.exists():

        print(
            f"[RESUME] seed={seed} "
            f"RAG_NODE2VEC DONE"
        )

        n2v_row = (
            pd.read_csv(
                n2v_summary
            )
            .iloc[0]
            .to_dict()
        )

    else:

        n2v_row = run_ppo_pipeline(
            name="RAG_NODE2VEC",
            panel=node2vec_panel,
            feature_list=NODE2VEC_FEATURES,
            common_start=common_start,
            seed=seed,
            output_dir=n2v_dir,
        )


    rows.append(
        n2v_row
    )


    pd.DataFrame(
        rows
    ).to_csv(
        OUT_ROOT
        / "all_results.csv",
        index=False,
    )


# ============================================================
# Aggregate
# ============================================================

results = pd.DataFrame(
    rows
)


summary_rows = []


for fs in [
    "RAG",
    "RAG_NODE2VEC",
]:

    sub = results[
        results["feature_set"]
        == fs
    ]

    row = {
        "feature_set": fs,
        "n_seeds": len(sub),
    }

    for m in METRICS:

        row[
            m + "_mean"
        ] = sub[m].mean()

        row[
            m + "_std"
        ] = sub[m].std(
            ddof=1
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


rag = (
    results[
        results["feature_set"]
        == "RAG"
    ]
    .set_index("seed")
)

n2v = (
    results[
        results["feature_set"]
        == "RAG_NODE2VEC"
    ]
    .set_index("seed")
)


delta_rows = []


for seed in SEEDS:

    row = {
        "seed": seed,
    }

    for m in METRICS:

        row[
            "delta_" + m
        ] = (
            n2v.loc[
                seed,
                m,
            ]
            -
            rag.loc[
                seed,
                m,
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


paired_summary = {
    "n_seeds":
        len(SEEDS)
}


for m in METRICS:

    c = (
        "delta_"
        + m
    )

    paired_summary[
        c + "_mean"
    ] = deltas[c].mean()

    paired_summary[
        c + "_std"
    ] = deltas[c].std(
        ddof=1
    )

    paired_summary[
        c + "_median"
    ] = deltas[c].median()

    paired_summary[
        c + "_positive_count"
    ] = int(
        (
            deltas[c] > 0
        ).sum()
    )


pd.DataFrame(
    [paired_summary]
).to_csv(
    OUT_ROOT
    / "paired_summary.csv",
    index=False,
)


print("\n")
print("=" * 100)
print("PPO 10-SEED MEAN ± STD")
print("=" * 100)


for _, r in summary.iterrows():

    print(
        f"\n{r['feature_set']}"
    )

    print(
        "Final multiple : "
        f"{r['final_multiple_mean']:.6f}"
        " ± "
        f"{r['final_multiple_std']:.6f}"
    )

    print(
        "Annual return  : "
        f"{r['annual_return_mean']:.2%}"
        " ± "
        f"{r['annual_return_std']:.2%}"
    )

    print(
        "Annual vol     : "
        f"{r['annual_volatility_mean']:.2%}"
        " ± "
        f"{r['annual_volatility_std']:.2%}"
    )

    print(
        "Sharpe         : "
        f"{r['sharpe_mean']:.4f}"
        " ± "
        f"{r['sharpe_std']:.4f}"
    )

    print(
        "Max drawdown   : "
        f"{r['max_drawdown_mean']:.2%}"
        " ± "
        f"{r['max_drawdown_std']:.2%}"
    )


print("\n")
print("=" * 100)
print("PAIRED DELTAS: RAG_NODE2VEC - RAG")
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

for m in METRICS:

    c = (
        "delta_"
        + m
    )

    print(
        f"{m:20s}: "
        f"mean={deltas[c].mean():+.6f} | "
        f"median={deltas[c].median():+.6f} | "
        f"std={deltas[c].std(ddof=1):.6f} | "
        f"positive={int((deltas[c] > 0).sum())}/10"
    )


print(
    "\nSaved to:",
    OUT_ROOT
)
