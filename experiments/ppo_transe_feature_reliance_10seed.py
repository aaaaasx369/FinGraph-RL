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


# ============================================================
# Config
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
    / "ppo_transe_feature_reliance_10seed"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

TRANSE_PATH = (
    ROOT
    / "kg"
    / "output"
    / "finreflectkg_company_transe_32d.csv"
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

SHUFFLE_SEED = 20261004


# ============================================================
# Build normalized TransE panel
# ============================================================

def build_transe_panel(rag_panel):

    transe = pd.read_csv(
        TRANSE_PATH
    )

    # KG year Y -> trading year Y+1
    transe["trading_year"] = (
        transe["year"].astype(int)
        + 1
    )

    transe = transe.rename(
        columns={
            "ticker": "tic",
            "year": "transe_source_year",
        }
    )

    transe = transe[
        [
            "tic",
            "trading_year",
            "transe_source_year",
            "transe_available",
        ]
        + TRANSE_FEATURES
    ].copy()

    transe["tic"] = (
        transe["tic"]
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


    before = len(out)

    out = out.merge(
        transe,
        on=[
            "tic",
            "trading_year",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(out) != before:
        raise RuntimeError(
            "TransE merge changed row count."
        )


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
    # Fit normalization using TRAINING-period available
    # unique ticker/source-year embeddings only.
    # ========================================================

    fit = out[
        (out["date_dt"] < pd.Timestamp(exp.TRAIN_END))
        &
        (out["transe_available"] > 0.5)
    ].copy()

    fit = (
        fit[
            [
                "tic",
                "transe_source_year",
            ]
            + TRANSE_FEATURES
        ]
        .drop_duplicates(
            [
                "tic",
                "transe_source_year",
            ]
        )
    )

    print(
        "Scaler fit ticker-years:",
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


    # unavailable rows remain neutral
    missing = (
        out["transe_available"]
        < 0.5
    )

    out.loc[
        missing,
        TRANSE_FEATURES,
    ] = 0.0


    # ========================================================
    # Leakage checks
    # ========================================================

    available = (
        out["transe_available"]
        > 0.5
    )

    if available.any():

        source = (
            out.loc[
                available,
                "transe_source_year",
            ]
            .astype(int)
        )

        trading = (
            out.loc[
                available,
                "trading_year",
            ]
            .astype(int)
        )

        if not (
            source
            ==
            trading - 1
        ).all():

            raise RuntimeError(
                "TransE leakage detected."
            )


    # 2018 must be unavailable
    y2018 = (
        out["trading_year"]
        == 2018
    )

    if (
        y2018.any()
        and
        out.loc[
            y2018,
            "transe_available",
        ].max() != 0
    ):
        raise RuntimeError(
            "2018 unexpectedly has TransE."
        )


    out["date"] = (
        out["date_dt"]
        .dt.strftime("%Y-%m-%d")
    )

    return out


# ============================================================
# ZERO ablation
#
# IMPORTANT:
# Keep transe_available unchanged.
#
# Therefore NORMAL and ZERO have:
# - same dimensions
# - same availability feature
# - same dates
#
# Only embedding information disappears.
# ============================================================

def make_zero_panel(normal_panel):

    out = normal_panel.copy()

    out.loc[
        :,
        TRANSE_FEATURES,
    ] = 0.0

    return out


# ============================================================
# SHUFFLED ablation
#
# Shuffle company identity WITHIN each year.
#
# Example:
#
# AAPL  <- NVDA embedding
# MSFT  <- AMZN embedding
# ...
#
# We require a derangement:
# no company receives its own embedding.
#
# This preserves:
# - embedding distribution
# - yearly timing
# - availability
#
# but destroys semantic company identity.
# ============================================================

def make_shuffled_panel(normal_panel):

    rng = np.random.default_rng(
        SHUFFLE_SEED
    )

    out = normal_panel.copy()


    unique = (
        out[
            [
                "trading_year",
                "tic",
                "transe_available",
            ]
            + TRANSE_FEATURES
        ]
        .drop_duplicates(
            [
                "trading_year",
                "tic",
            ]
        )
        .sort_values(
            [
                "trading_year",
                "tic",
            ]
        )
        .reset_index(drop=True)
    )


    mapping_rows = []


    print("\nSHUFFLE MAPPING")
    print("=" * 80)


    for year in sorted(
        unique["trading_year"].unique()
    ):

        u = (
            unique[
                unique["trading_year"]
                == year
            ]
            .copy()
            .reset_index(drop=True)
        )


        # 2018 has no KG; leave zero.
        available_mask = (
            u["transe_available"]
            > 0.5
        )


        if available_mask.sum() < 2:

            for _, row in u.iterrows():

                d = {
                    "trading_year":
                        year,

                    "tic":
                        row["tic"],

                    "donor_tic":
                        row["tic"],
                }

                for c in TRANSE_FEATURES:
                    d[c] = row[c]

                mapping_rows.append(d)

            continue


        avail = (
            u[
                available_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        n = len(avail)


        # Find a permutation with no fixed points.
        while True:

            perm = rng.permutation(n)

            if np.all(
                perm
                != np.arange(n)
            ):
                break


        donors = (
            avail.iloc[
                perm
            ]
            .reset_index(drop=True)
        )


        for i in range(n):

            recipient = (
                avail.iloc[i]["tic"]
            )

            donor = (
                donors.iloc[i]["tic"]
            )

            print(
                f"{year}: "
                f"{recipient:5s} <- {donor:5s}"
            )


            d = {
                "trading_year":
                    year,

                "tic":
                    recipient,

                "donor_tic":
                    donor,
            }

            for c in TRANSE_FEATURES:
                d[c] = donors.iloc[i][c]

            mapping_rows.append(d)


        # unavailable rows, if any
        unavailable = (
            u[
                ~available_mask
            ]
        )

        for _, row in unavailable.iterrows():

            d = {
                "trading_year":
                    year,

                "tic":
                    row["tic"],

                "donor_tic":
                    row["tic"],
            }

            for c in TRANSE_FEATURES:
                d[c] = row[c]

            mapping_rows.append(d)


    mapping = pd.DataFrame(
        mapping_rows
    )


    # Store mapping for audit.
    mapping[
        [
            "trading_year",
            "tic",
            "donor_tic",
        ]
    ].to_csv(
        OUT_ROOT
        / "shuffle_mapping.csv",
        index=False,
    )


    replacement = mapping[
        [
            "trading_year",
            "tic",
        ]
        + TRANSE_FEATURES
    ].copy()


    out = out.drop(
        columns=TRANSE_FEATURES
    )


    out = out.merge(
        replacement,
        on=[
            "trading_year",
            "tic",
        ],
        how="left",
        validate="many_to_one",
    )


    if out[
        TRANSE_FEATURES
    ].isna().any().any():

        raise RuntimeError(
            "NaN after shuffled merge."
        )


    return out


# ============================================================
# One fixed-PPO run
# ============================================================

def run_pipeline(
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


    # Explicitly set seed used by original train_algorithm().
    exp.MODEL_SEEDS["PPO"] = seed


    print("\n" + "=" * 100)

    print(
        f"{name} | PPO | "
        f"effective seed={exp.MODEL_SEEDS['PPO']}"
    )

    print("=" * 100)


    start = (
        common_start.strftime(
            "%Y-%m-%d"
        )
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


    # Validation saved for audit.
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


    # Untouched test.
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
        "Final multiple : "
        f"{test_metrics['final_multiple']:.6f}"
    )

    print(
        "Sharpe         : "
        f"{test_metrics['sharpe']:.4f}"
    )

    print(
        "Max drawdown   : "
        f"{test_metrics['max_drawdown']:.2%}"
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


normal_panel = (
    build_transe_panel(
        rag_panel
    )
)


zero_panel = (
    make_zero_panel(
        normal_panel
    )
)


shuffled_panel = (
    make_shuffled_panel(
        normal_panel
    )
)


FEATURES = (
    exp.BASE_FEATURES
    + exp.RAG_FEATURES
    + TRANSE_FEATURES
    + ["transe_available"]
)


# ============================================================
# Sanity checks
# ============================================================

if not (
    len(normal_panel)
    ==
    len(zero_panel)
    ==
    len(shuffled_panel)
):
    raise RuntimeError(
        "Panel row counts differ."
    )


if (
    zero_panel[
        TRANSE_FEATURES
    ]
    .abs()
    .to_numpy()
    .max()
    != 0
):
    raise RuntimeError(
        "ZERO panel contains nonzero TransE."
    )


print(
    "\nFeature count:",
    len(FEATURES),
)

print(
    "Normal rows:",
    len(normal_panel),
)

print(
    "Zero rows:",
    len(zero_panel),
)

print(
    "Shuffled rows:",
    len(shuffled_panel),
)


# ============================================================
# Run 10 paired seeds
# ============================================================

variants = {
    "NORMAL":
        normal_panel,

    "ZERO":
        zero_panel,

    "SHUFFLED":
        shuffled_panel,
}


rows = []


for seed in SEEDS:

    print("\n")
    print("#" * 100)
    print(
        f"PAIRED PPO SEED = {seed}"
    )
    print("#" * 100)


    for name, panel in variants.items():

        output_dir = (
            OUT_ROOT
            / f"seed_{seed}"
            / name.lower()
        )

        summary_path = (
            output_dir
            / "test_summary.csv"
        )


        if summary_path.exists():

            print(
                f"[RESUME] "
                f"seed={seed} "
                f"{name} DONE"
            )

            row = (
                pd.read_csv(
                    summary_path
                )
                .iloc[0]
                .to_dict()
            )

        else:

            row = run_pipeline(
                name=name,
                panel=panel,
                feature_list=FEATURES,
                common_start=common_start,
                seed=seed,
                output_dir=output_dir,
            )


        rows.append(row)


        # Save after every single completed run.
        pd.DataFrame(
            rows
        ).to_csv(
            OUT_ROOT
            / "all_results.csv",
            index=False,
        )


# ============================================================
# Aggregate mean ± std
# ============================================================

results = pd.DataFrame(
    rows
)


summary_rows = []


for name in [
    "NORMAL",
    "ZERO",
    "SHUFFLED",
]:

    sub = results[
        results["feature_set"]
        == name
    ]


    row = {
        "feature_set":
            name,

        "n_seeds":
            len(sub),
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


    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUT_ROOT
    / "summary_mean_std.csv",
    index=False,
)


# ============================================================
# Paired comparisons
#
# NORMAL - ZERO
# NORMAL - SHUFFLED
# ============================================================

indexed = {
    name:
        (
            results[
                results["feature_set"]
                == name
            ]
            .set_index("seed")
        )

    for name in variants
}


comparison_rows = []


for comparison, other in [
    (
        "NORMAL_MINUS_ZERO",
        "ZERO",
    ),
    (
        "NORMAL_MINUS_SHUFFLED",
        "SHUFFLED",
    ),
]:

    for seed in SEEDS:

        row = {
            "comparison":
                comparison,

            "seed":
                seed,
        }


        for m in METRICS:

            row[
                "delta_" + m
            ] = (
                indexed[
                    "NORMAL"
                ].loc[
                    seed,
                    m,
                ]
                -
                indexed[
                    other
                ].loc[
                    seed,
                    m,
                ]
            )


        comparison_rows.append(row)


paired = pd.DataFrame(
    comparison_rows
)


paired.to_csv(
    OUT_ROOT
    / "paired_deltas.csv",
    index=False,
)


# ============================================================
# Statistical summary
# ============================================================

stat_rows = []


for comparison in paired[
    "comparison"
].unique():

    x = paired[
        paired["comparison"]
        == comparison
    ]


    for m in METRICS:

        c = (
            "delta_"
            + m
        )

        values = (
            x[c]
            .astype(float)
            .to_numpy()
        )


        mean = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        sd = float(
            np.std(
                values,
                ddof=1,
            )
        )


        t_result = stats.ttest_1samp(
            values,
            popmean=0.0,
        )


        sem = stats.sem(
            values
        )


        ci = stats.t.interval(
            confidence=0.95,
            df=len(values) - 1,
            loc=mean,
            scale=sem,
        )


        stat_rows.append(
            {
                "comparison":
                    comparison,

                "metric":
                    m,

                "mean_delta":
                    mean,

                "median_delta":
                    median,

                "std_delta":
                    sd,

                "positive_count":
                    int(
                        (
                            values > 0
                        ).sum()
                    ),

                "n":
                    len(values),

                "t_stat":
                    float(
                        t_result.statistic
                    ),

                "p_value":
                    float(
                        t_result.pvalue
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
    stat_rows
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
print("FEATURE RELIANCE: 10-SEED MEAN ± STD")
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
print("PAIRED RELIANCE TEST")
print("=" * 100)


for comparison in stats_df[
    "comparison"
].unique():

    print(
        f"\n{comparison}"
    )

    sub = stats_df[
        stats_df["comparison"]
        == comparison
    ]


    for _, r in sub.iterrows():

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
