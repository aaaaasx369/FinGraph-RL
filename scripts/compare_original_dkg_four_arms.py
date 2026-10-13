from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.cwd()
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "research_audit"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "CLEAN": ("ORIGINAL_KGTRANSFORMER", "CLEAN"),
    "KGTRANSFORMER": ("ORIGINAL_KGTRANSFORMER", "FULL_DYNAMIC"),
    "PLACEBO": ("ORIGINAL_KG_PLACEBO", "FULL_DYNAMIC"),
    "FROZEN": ("ORIGINAL_KG_FROZEN", "FULL_DYNAMIC"),
}

KEYS = ["tier", "agent", "seed", "split"]
METRICS = ["total_return", "sharpe", "max_drawdown"]

frames = []

for name, (folder, condition) in SOURCES.items():
    path = BASE / "runs_multiseed" / folder / "runs.csv"

    if not path.exists():
        raise SystemExit(f"找不到檔案：{path}")

    df = pd.read_csv(path)
    df = df[
        df["condition"].eq(condition)
        & df["split"].eq("test")
    ].copy()

    if df.duplicated(KEYS).any():
        raise RuntimeError(f"{name} 存在重複紀錄")

    if len(df) != 30:
        raise RuntimeError(
            f"{name} 應有 30 筆 2023 Test，實際 {len(df)}"
        )

    df["experiment"] = name
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

summary = (
    data.groupby(["tier", "agent", "experiment"])
    .agg(
        n=("seed", "size"),
        return_mean=("total_return", "mean"),
        return_std=("total_return", "std"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        mdd_mean=("max_drawdown", "mean"),
        mdd_std=("max_drawdown", "std"),
    )
    .reset_index()
)

summary.to_csv(
    OUT / "four_arm_performance.csv",
    index=False,
)

comparisons = [
    ("KGTRANSFORMER", "FROZEN"),
    ("KGTRANSFORMER", "PLACEBO"),
    ("KGTRANSFORMER", "CLEAN"),
    ("FROZEN", "CLEAN"),
]

records = []

for (tier, agent), group in data.groupby(["tier", "agent"]):
    for metric in METRICS:
        pivot = group.pivot(
            index="seed",
            columns="experiment",
            values=metric,
        )

        if set(pivot.index) != set(range(5)):
            raise RuntimeError("五個 Seeds 不完整")

        for a, b in comparisons:
            diff = (pivot[a] - pivot[b]).to_numpy(float)

            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            n = len(diff)

            if sd > 0:
                t, p = stats.ttest_1samp(diff, 0)
                margin = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            else:
                t = 0 if mean == 0 else np.inf
                p = 1.0 if mean == 0 else 0.0
                margin = 0.0

            records.append({
                "tier": tier,
                "agent": agent,
                "metric": metric,
                "comparison": f"{a}-{b}",
                "mean_difference": mean,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "p_value": float(p),
            })

tests = pd.DataFrame(records)

# Holm correction within each tier and metric.
tests["p_holm"] = np.nan

for _, indexes in tests.groupby(
    ["tier", "metric"]
).groups.items():
    ids = list(indexes)
    p = tests.loc[ids, "p_value"].to_numpy(float)
    order = np.argsort(p)

    adjusted = np.zeros(len(p))
    previous = 0.0

    for rank, j in enumerate(order):
        previous = max(previous, (len(p)-rank)*p[j])
        adjusted[j] = min(1.0, previous)

    tests.loc[ids, "p_holm"] = adjusted

tests.to_csv(
    OUT / "four_arm_paired_tests.csv",
    index=False,
)

print("\n=== 2023 TEST：四組平均績效 ===")
print(summary.round(5).to_string(index=False))

print("\n=== KGTRANSFORMER vs FROZEN ===")
print(
    tests[
        tests["comparison"].eq("KGTRANSFORMER-FROZEN")
    ][
        [
            "tier", "agent", "metric",
            "mean_difference", "ci95_low",
            "ci95_high", "p_value", "p_holm"
        ]
    ].round(5).to_string(index=False)
)

print("\nFOUR-ARM COMPARISON COMPLETE")
