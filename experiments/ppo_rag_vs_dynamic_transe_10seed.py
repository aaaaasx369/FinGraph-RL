#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    / "ppo_rag_vs_dynamic_transe_10seed"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

DYNAMIC_PATH = (
    ROOT
    / "kg"
    / "output"
    / "finreflectkg_dynamic_transe_32d.csv"
)

# We can reuse the already-computed fixed-PPO RAG baselines.
OLD_RAG_ROOT = (
    ROOT
    / "results"
    / "ppo_rag_vs_transe_10seed"
)

TRANSE_FEATURES = [
    f"transe_{i:02d}"
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
# Build leakage-free dynamic TransE panel
# ============================================================

def build_dynamic_panel(rag_panel):

    snapshots = pd.read_csv(
        DYNAMIC_PATH
    )

    snapshots["snapshot_date"] = pd.to_datetime(
        snapshots["snapshot_date"]
    )

    snapshots["ticker"] = (
        snapshots["ticker"]
        .astype(str)
        .str.upper()
    )

    keep = (
        [
            "snapshot_date",
            "ticker",
            "transe_available",
        ]
        + TRANSE_FEATURES
    )

    snapshots = snapshots[
        keep
    ].copy()


    left = rag_panel.copy()

    left["date_dt"] = pd.to_datetime(
        left["date"]
    )

    left["tic"] = (
        left["tic"]
        .astype(str)
        .str.upper()
    )


    pieces = []


    # ========================================================
    # Strict backward merge separately for each ticker
    #
    # allow_exact_matches=False:
    # filing published on date t cannot be used on date t.
    # ========================================================

    for ticker in sorted(
        left["tic"].unique()
    ):

        l = (
            left[
                left["tic"] == ticker
            ]
            .copy()
            .sort_values("date_dt")
        )

        r = (
            snapshots[
                snapshots["ticker"] == ticker
            ]
            .copy()
            .sort_values("snapshot_date")
        )


        if len(r) == 0:

            l["snapshot_date"] = pd.NaT
            l["transe_available"] = 0.0

            for c in TRANSE_FEATURES:
                l[c] = 0.0

        else:

            r = r.drop(
                columns=["ticker"]
            )

            l = pd.merge_asof(
                l,
                r,
                left_on="date_dt",
                right_on="snapshot_date",
                direction="backward",
                allow_exact_matches=False,
            )


        pieces.append(l)


    out = pd.concat(
        pieces,
        ignore_index=True,
    )


    out = out.sort_values(
        [
            "date_dt",
            "tic",
        ]
    ).reset_index(drop=True)


    out["transe_available"] = (
        out["transe_available"]
        .fillna(0.0)
        .astype(float)
    )


    for c in TRANSE_FEATURES:

        out[c] = (
            pd.to_numeric(
                out[c],
                errors="coerce",
            )
            .fillna(0.0)
        )


    # ========================================================
    # HARD leakage check
    # ========================================================

    matched = out["snapshot_date"].notna()

    if matched.any():

        bad = (
            out.loc[
                matched,
                "snapshot_date",
            ]
            >=
            out.loc[
                matched,
                "date_dt",
            ]
        )

        if bad.any():

            example = out.loc[
                matched
            ].loc[
                bad.values,
                [
                    "date_dt",
                    "tic",
                    "snapshot_date",
                ],
            ]

            raise RuntimeError(
                "Dynamic KG leakage detected:\n"
                + example.head().to_string(
                    index=False
                )
            )


    # ========================================================
    # Training-only normalization
    #
    # Do not count the same snapshot hundreds of times.
    # Fit each ticker/snapshot once.
    # ========================================================

    fit = out[
        (out["date_dt"] < pd.Timestamp(exp.TRAIN_END))
        &
        (out["transe_available"] > 0.5)
        &
        (out["snapshot_date"].notna())
    ].copy()


    fit = (
        fit[
            [
                "tic",
                "snapshot_date",
            ]
            + TRANSE_FEATURES
        ]
        .drop_duplicates(
            [
                "tic",
                "snapshot_date",
            ]
        )
    )


    print(
        "\nDynamic TransE scaler fit ticker-snapshots:",
        len(fit),
    )


    for c in TRANSE_FEATURES:

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


    # unavailable company embeddings remain neutral zero
    unavailable = (
        out["transe_available"]
        < 0.5
    )

    out.loc[
        unavailable,
        TRANSE_FEATURES,
    ] = 0.0


    # ========================================================
    # Audit
    # ========================================================

    print("\nDynamic KG audit:")

    print(
        "Trading rows:",
        len(out)
    )

    print(
        "Unique matched snapshots:",
        out["snapshot_date"]
        .dropna()
        .nunique()
    )

    print(
        "Earliest trading date:",
        out["date_dt"].min()
    )

    print(
        "Earliest used snapshot:",
        out["snapshot_date"].min()
    )

    print(
        "Latest used snapshot:",
        out["snapshot_date"].max()
    )


    print(
        "\nAvailability by trading year:"
    )

    print(
        out.assign(
            trading_year=
                out["date_dt"].dt.year
        )
        .groupby(
            "trading_year"
        )[
            "transe_available"
        ]
        .mean()
    )


    # sample to prove dynamics actually changes within year
    print(
        "\n2023 snapshot dates actually used:"
    )

    print(
        out.loc[
            out["date_dt"].dt.year == 2023,
            "snapshot_date",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d")
        .to_list()
    )


    out["date"] = (
        out["date_dt"]
        .dt.strftime("%Y-%m-%d")
    )


    return out


# ============================================================
# Fixed PPO run
# ============================================================

def run_ppo(
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


    exp.MODEL_SEEDS["PPO"] = seed


    print("\n" + "=" * 100)

    print(
        f"{name} | PPO | "
        f"effective seed={exp.MODEL_SEEDS['PPO']}"
    )

    print("=" * 100)


    start = common_start.strftime(
        "%Y-%m-%d"
    )


    train_df = exp.data_split(
        panel,
        start,
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


    model = exp.train_algorithm(
        "PPO",
        train_df,
        feature_list,
    )


    model.save(
        str(
            output_dir
            / "ppo_model"
        )
    )


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
        [val_metrics]
    ).to_csv(
        output_dir
        / "validation_summary.csv",
        index=False,
    )


    test_returns, test_weights = (
        exp.evaluate_model(
            model,
            test_df,
            feature_list,
        )
    )

    test_metrics = exp.calculate_metrics(
        test_returns
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
        "Final multiple : "
        f"{test_metrics['final_multiple']:.6f}"
    )

    print(
        "Sharpe         : "
        f"{test_metrics['sharpe']:.4f}"
    )


    return result


# ============================================================
# Data
# ============================================================

print("=" * 100)
print("BUILD DATA")
print("=" * 100)

exp.RESULT_ROOT = OUT_ROOT


market = exp.build_market_data()


rag_panel, common_start = (
    exp.merge_rag_features(
        market
    )
)


dynamic_panel = (
    build_dynamic_panel(
        rag_panel
    )
)


RAG_FEATURES = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
)


DYNAMIC_FEATURES = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
    + TRANSE_FEATURES
    + ["transe_available"]
)


# Ensure exact same observations.
rag_keys = (
    rag_panel[
        [
            "date",
            "tic",
        ]
    ]
    .astype(str)
    .reset_index(drop=True)
)

dynamic_keys = (
    dynamic_panel[
        [
            "date",
            "tic",
        ]
    ]
    .astype(str)
    .reset_index(drop=True)
)


rag_keys = rag_keys.sort_values(
    ["date", "tic"]
).reset_index(drop=True)

dynamic_keys = dynamic_keys.sort_values(
    ["date", "tic"]
).reset_index(drop=True)


if not rag_keys.equals(
    dynamic_keys
):
    raise RuntimeError(
        "RAG and dynamic panels differ in date/ticker rows."
    )


print(
    "\nRAG features:",
    len(RAG_FEATURES)
)

print(
    "Dynamic features:",
    len(DYNAMIC_FEATURES)
)


# ============================================================
# 10 paired seeds
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


    # ========================================================
    # RAG baseline
    #
    # Reuse prior identical fixed-PPO baseline if available.
    # ========================================================

    rag_dir = (
        seed_root
        / "rag"
    )

    rag_summary = (
        rag_dir
        / "test_summary.csv"
    )


    old_summary = (
        OLD_RAG_ROOT
        / f"seed_{seed}"
        / "rag"
        / "test_summary.csv"
    )


    if rag_summary.exists():

        print(
            f"[RESUME] seed={seed} RAG DONE"
        )

        rag_row = (
            pd.read_csv(
                rag_summary
            )
            .iloc[0]
            .to_dict()
        )


    elif old_summary.exists():

        print(
            f"[REUSE] seed={seed} "
            f"RAG baseline from prior fixed-PPO experiment"
        )

        rag_row = (
            pd.read_csv(
                old_summary
            )
            .iloc[0]
            .to_dict()
        )

        rag_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            [rag_row]
        ).to_csv(
            rag_summary,
            index=False,
        )


    else:

        rag_row = run_ppo(
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


    # ========================================================
    # Dynamic TransE
    # ========================================================

    dyn_dir = (
        seed_root
        / "rag_dynamic_transe"
    )

    dyn_summary = (
        dyn_dir
        / "test_summary.csv"
    )


    if dyn_summary.exists():

        print(
            f"[RESUME] seed={seed} "
            f"RAG_DYNAMIC_TRANSE DONE"
        )

        dyn_row = (
            pd.read_csv(
                dyn_summary
            )
            .iloc[0]
            .to_dict()
        )

    else:

        dyn_row = run_ppo(
            name="RAG_DYNAMIC_TRANSE",
            panel=dynamic_panel,
            feature_list=DYNAMIC_FEATURES,
            common_start=common_start,
            seed=seed,
            output_dir=dyn_dir,
        )


    rows.append(
        dyn_row
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
    "RAG_DYNAMIC_TRANSE",
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

dyn = (
    results[
        results["feature_set"]
        == "RAG_DYNAMIC_TRANSE"
    ]
    .set_index("seed")
)


delta_rows = []


for seed in SEEDS:

    row = {
        "seed": seed
    }


    for m in METRICS:

        row[
            "delta_" + m
        ] = (
            dyn.loc[
                seed,
                m,
            ]
            -
            rag.loc[
                seed,
                m,
            ]
        )


    delta_rows.append(row)


deltas = pd.DataFrame(
    delta_rows
)


deltas.to_csv(
    OUT_ROOT
    / "paired_deltas.csv",
    index=False,
)


# ============================================================
# Paired statistical tests
# ============================================================

stats_rows = []


for m in METRICS:

    c = (
        "delta_"
        + m
    )

    x = deltas[c].to_numpy(
        dtype=float
    )

    mean = float(
        x.mean()
    )

    sd = float(
        x.std(
            ddof=1
        )
    )

    t = stats.ttest_1samp(
        x,
        popmean=0.0,
    )

    sem = stats.sem(x)

    ci = stats.t.interval(
        confidence=0.95,
        df=len(x) - 1,
        loc=mean,
        scale=sem,
    )


    stats_rows.append(
        {
            "metric":
                m,

            "mean_delta":
                mean,

            "median_delta":
                float(
                    np.median(x)
                ),

            "std_delta":
                sd,

            "positive_count":
                int(
                    (x > 0).sum()
                ),

            "n":
                len(x),

            "p_value":
                float(
                    t.pvalue
                ),

            "ci95_low":
                float(
                    ci[0]
                ),

            "ci95_high":
                float(
                    ci[1]
                ),
        }
    )


stats_df = pd.DataFrame(
    stats_rows
)

stats_df.to_csv(
    OUT_ROOT
    / "paired_statistics.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\n")
print("=" * 100)
print("DYNAMIC TRANSE: PPO 10-SEED MEAN ± STD")
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
print("PAIRED: DYNAMIC TRANSE - RAG")
print("=" * 100)

print(
    deltas.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("PAIRED STATISTICS")
print("=" * 100)


for _, r in stats_df.iterrows():

    print(
        f"{r['metric']:20s}: "
        f"mean={r['mean_delta']:+.6f} | "
        f"median={r['median_delta']:+.6f} | "
        f"positive={int(r['positive_count'])}/10 | "
        f"p={r['p_value']:.4f} | "
        f"95% CI=["
        f"{r['ci95_low']:+.6f}, "
        f"{r['ci95_high']:+.6f}]"
    )


print(
    "\nSaved to:",
    OUT_ROOT
)
