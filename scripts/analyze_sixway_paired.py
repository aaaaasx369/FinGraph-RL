from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


INPUT = Path(
    "results/sixway_train2020_final/test_runs.csv"
)

OUTPUT = Path(
    "results/sixway_train2020_final/paired_comparison.csv"
)


df = pd.read_csv(INPUT)

# 先聚焦 A2C，因為正式結果中 A2C 最能利用 KG features
df = df[
    df["agent"] == "A2C"
].copy()


comparisons = [
    (
        "SEC_FINDKG_vs_SEC",
        "BASE_SEC_FINDKG",
        "BASE_SEC",
    ),
    (
        "SEC_REFLECTKG_vs_SEC",
        "BASE_SEC_REFLECTKG",
        "BASE_SEC",
    ),
    (
        "REFLECTKG_vs_BASE",
        "BASE_REFLECTKG",
        "BASE",
    ),
    (
        "FINDKG_vs_BASE",
        "BASE_FINDKG",
        "BASE",
    ),
    (
        "SEC_REFLECTKG_vs_SEC_FINDKG",
        "BASE_SEC_REFLECTKG",
        "BASE_SEC_FINDKG",
    ),
]


metrics = [
    "total_return",
    "sharpe",
    "max_drawdown",
]


rows = []


for comparison_name, treatment, baseline in comparisons:

    a = (
        df[
            df["feature_set"] == treatment
        ][
            [
                "seed",
                *metrics,
            ]
        ]
        .copy()
        .set_index("seed")
        .sort_index()
    )

    b = (
        df[
            df["feature_set"] == baseline
        ][
            [
                "seed",
                *metrics,
            ]
        ]
        .copy()
        .set_index("seed")
        .sort_index()
    )


    common = sorted(
        set(a.index)
        &
        set(b.index)
    )

    a = a.loc[common]
    b = b.loc[common]


    for metric in metrics:

        diff = (
            a[metric]
            -
            b[metric]
        ).astype(float)

        n = len(diff)

        mean_diff = float(
            diff.mean()
        )

        std_diff = float(
            diff.std(
                ddof=1
            )
        ) if n > 1 else 0.0


        # 95% CI for paired mean difference
        if n > 1:

            sem = (
                std_diff
                /
                np.sqrt(n)
            )

            tcrit = stats.t.ppf(
                0.975,
                df=n - 1,
            )

            ci_low = (
                mean_diff
                -
                tcrit
                *
                sem
            )

            ci_high = (
                mean_diff
                +
                tcrit
                *
                sem
            )

        else:

            ci_low = np.nan
            ci_high = np.nan


        # Paired t-test
        if n > 1:

            t_stat, t_p = (
                stats.ttest_rel(
                    a[metric],
                    b[metric],
                )
            )

        else:

            t_stat = np.nan
            t_p = np.nan


        # Wilcoxon signed-rank test
        try:

            w_stat, w_p = (
                stats.wilcoxon(
                    diff
                )
            )

        except ValueError:

            w_stat = np.nan
            w_p = np.nan


        rows.append(
            {
                "comparison":
                    comparison_name,

                "treatment":
                    treatment,

                "baseline":
                    baseline,

                "metric":
                    metric,

                "n_seeds":
                    n,

                "mean_treatment":
                    float(
                        a[metric].mean()
                    ),

                "mean_baseline":
                    float(
                        b[metric].mean()
                    ),

                "mean_paired_diff":
                    mean_diff,

                "std_paired_diff":
                    std_diff,

                "ci95_low":
                    ci_low,

                "ci95_high":
                    ci_high,

                "paired_t_stat":
                    t_stat,

                "paired_t_p":
                    t_p,

                "wilcoxon_stat":
                    w_stat,

                "wilcoxon_p":
                    w_p,
            }
        )


out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False,
)


print("=" * 110)
print("PAIRED A2C COMPARISON")
print("=" * 110)


show = out[
    out["metric"]
    ==
    "total_return"
][
    [
        "comparison",
        "mean_treatment",
        "mean_baseline",
        "mean_paired_diff",
        "ci95_low",
        "ci95_high",
        "paired_t_p",
        "wilcoxon_p",
    ]
].copy()


for c in [
    "mean_treatment",
    "mean_baseline",
    "mean_paired_diff",
    "ci95_low",
    "ci95_high",
]:

    show[c] = (
        show[c]
        *
        100
    )


print(
    show.to_string(
        index=False
    )
)


print()
print("Saved:")
print(OUTPUT)


print()
print("=" * 110)
print("SEED-LEVEL A2C RETURNS")
print("=" * 110)


pivot = (
    df.pivot(
        index="seed",
        columns="feature_set",
        values="total_return",
    )
    *
    100
)


cols = [
    "BASE",
    "BASE_SEC",
    "BASE_FINDKG",
    "BASE_SEC_FINDKG",
    "BASE_REFLECTKG",
    "BASE_SEC_REFLECTKG",
]


print(
    pivot[
        cols
    ].round(
        2
    ).to_string()
)
