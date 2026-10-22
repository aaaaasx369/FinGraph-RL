#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse the verified FinRL-X implementation.
#
# It already handles:
# - A2C / PPO / DDPG
# - KGProjectionExtractor
# - Original StockPortfolioEnv
# - 2022 validation
# - 2023 testing
# - All-algorithm test summaries

from unified_event_memory.finrl_x import (
    run_dynamic_kg_drl as runner
)

GROUPS = [
    "clean",
    "real_dynamic",
    "frozen_dynamic",
    "time_shuffled",
]

ALGORITHMS = [
    "A2C",
    "PPO",
    "DDPG",
]

DEFAULT_SEEDS = [
    42,
    123,
    2026,
    7,
    999,
]

METRICS = [
    "final_value",
    "final_multiple",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
]

SPLITS = [
    ("train", "2018-01-01", "2022-01-01"),
    ("validation", "2022-01-01", "2023-01-01"),
    ("test", "2023-01-01", "2024-01-01"),
]


def get_split(date_value):
    d = str(date_value)[:10]

    for name, start, end in SPLITS:
        if start <= d < end:
            return name

    raise RuntimeError(
        f"Date outside experimental splits: {d}"
    )


def load_market_and_dynamic(embedding_dir):
    print()
    print("=" * 78)
    print("LOADING MARKET AND DYNAMIC KG")
    print("=" * 78)

    market = runner.prepare_panel(
        runner.exp.build_market_data()
    )

    kg = runner.load_kg(
        embedding_dir / "daily_dynamic_kgt.csv"
    )

    market_keys = (
        market[["date", "tic"]]
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    # Only market trading dates are retained.
    kg = kg.merge(
        market_keys,
        on=["date", "tic"],
        how="inner",
        validate="one_to_one",
    )

    if len(kg) != len(market_keys):
        raise RuntimeError(
            "Missing KG features on market trading dates"
        )

    if not (
        kg[["date", "tic"]]
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
        .equals(market_keys)
    ):
        raise RuntimeError(
            "KG/market date-ticker mismatch"
        )

    kg = kg.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    for ticker in runner.TICKERS:
        n = int(
            (kg["tic"] == ticker).sum()
        )

        if n != market["date"].nunique():
            raise RuntimeError(
                f"Incomplete KG dates for {ticker}"
            )

    print("Market dates:", market["date"].nunique())
    print("Market rows:", len(market))
    print("KG rows:", len(kg))

    return market, kg


def make_frozen_dynamic(kg):
    """
    Fix each company's representation at the FIRST
    available trading date in the experiment.

    No future KGT state is used to backfill
    earlier training observations.
    """

    dates = sorted(
        kg["date"].unique()
    )

    first_date = dates[0]

    reference = (
        kg[kg["date"] == first_date]
        .set_index("tic")
    )

    if set(reference.index) != set(runner.TICKERS):
        raise RuntimeError(
            "First date missing portfolio companies"
        )

    out = kg[["date", "tic"]].copy()

    for column in runner.KG_COLS:
        mapping = reference[column].to_dict()

        out[column] = (
            out["tic"].map(mapping).astype("float32")
        )

    # Hard check: fixed for every ticker.
    variation = (
        out.groupby("tic")[runner.KG_COLS]
        .nunique()
        .max(axis=1)
    )

    if not (variation == 1).all():
        raise RuntimeError(
            "Frozen Dynamic KG is not constant"
        )

    print(
        "Frozen reference date:",
        first_date,
    )

    return out, first_date


def make_past_only_shuffle(
    kg,
    seed=2026,
    max_lag=60,
):
    """
    Temporal negative control.

    For each split independently:
      use current/past KG snapshots only.

    At day t, randomly choose from:
      max(split_start, t-max_lag) ... t

    The same sampled SOURCE DATE is used for all
    five companies, preserving cross-company
    relationships of each historical snapshot.

    The assignment is fixed across all DRL
    training seeds, so DRL randomness is the
    only variable across the five repetitions.

    This is a past-only temporal reassignment,
    NOT an unrestricted date permutation.
    """

    if max_lag < 1:
        raise ValueError(
            "max_lag must be >= 1"
        )

    rng = np.random.default_rng(seed)

    dates = sorted(
        kg["date"].unique()
    )

    date_to_source = {}
    mapping_rows = []

    for split_name, start, end in SPLITS:
        split_dates = [
            d for d in dates
            if start <= d < end
        ]

        if not split_dates:
            raise RuntimeError(
                f"No dates for {split_name}"
            )

        for i, target_day in enumerate(split_dates):
            earliest = max(
                0,
                i - max_lag,
            )

            # Strictly past where possible.
            # At split start, use that day's already
            # as-of-safe feature vector.
            latest = i - 1 if i > 0 else 0

            source_index = int(
                rng.integers(
                    earliest,
                    latest + 1,
                )
            )

            source_day = split_dates[source_index]

            if source_day > target_day:
                raise RuntimeError(
                    "Future KG feature leakage"
                )

            date_to_source[target_day] = source_day

            mapping_rows.append({
                "split": split_name,
                "target_date": target_day,
                "source_date": source_day,
                "lag_trading_days": i - source_index,
            })

    source_rows = kg.rename(
        columns={"date": "source_date"}
    )

    keys = kg[["date", "tic"]].copy()
    keys["source_date"] = (
        keys["date"].map(date_to_source)
    )

    if keys["source_date"].isna().any():
        raise RuntimeError(
            "Unmapped temporal shuffle dates"
        )

    out = keys.merge(
        source_rows,
        on=["source_date", "tic"],
        how="left",
        validate="many_to_one",
    )

    if out[runner.KG_COLS].isna().any().any():
        raise RuntimeError(
            "Missing shuffled KG features"
        )

    if not (
        out["source_date"] <= out["date"]
    ).all():
        raise RuntimeError(
            "Temporal shuffle used future features"
        )

    out = (
        out[["date", "tic"] + runner.KG_COLS]
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    audit = pd.DataFrame(mapping_rows)

    print(
        "Time shuffle seed:", seed,
        "maximum lag:", max_lag,
    )

    print(
        "Mean sampled lag:",
        float(audit["lag_trading_days"].mean()),
    )

    return out, audit


def build_panels(
    market,
    real_kg,
    frozen_kg,
    shuffled_kg,
):
    panels = {
        "clean": market.copy(),

        "real_dynamic": runner.attach_kg(
            market, real_kg
        ),

        "frozen_dynamic": runner.attach_kg(
            market, frozen_kg
        ),

        "time_shuffled": runner.attach_kg(
            market, shuffled_kg
        ),
    }

    features = {
        "clean": list(runner.exp.BASE_FEATURES)
    }

    for group in GROUPS[1:]:
        features[group] = (
            list(runner.exp.BASE_FEATURES)
            + list(runner.KG_COLS)
        )

    expected = (
        market[["date", "tic"]]
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    for name, panel in panels.items():
        actual = (
            panel[["date", "tic"]]
            .sort_values(["date", "tic"])
            .reset_index(drop=True)
        )

        if not actual.equals(expected):
            raise RuntimeError(
                f"Misaligned market panel: {name}"
            )

        runner.exp.validate_panel(
            panel, name
        )

        arr = (
            panel[runner.KG_COLS].to_numpy(
                dtype=np.float32
            )
            if name != "clean"
            else np.zeros((1, 1))
        )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Nonfinite values: {name}"
            )

        print(
            name,
            "features:",
            len(features[name]),
        )

    return panels, features


def read_complete_group(
    out_dir,
    seed,
    group,
):
    path = (
        out_dir
        / f"seed_{seed}"
        / group
        / "all_algorithms_test_summary.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if len(df) != len(ALGORITHMS):
        raise RuntimeError(
            f"Unexpected number of results: {path}"
        )

    if set(df["algorithm"]) != set(ALGORITHMS):
        raise RuntimeError(
            f"Wrong algorithms in {path}"
        )

    if set(df["group"]) != {group}:
        raise RuntimeError(
            f"Wrong group in {path}"
        )

    for metric in METRICS:
        if metric not in df.columns:
            raise RuntimeError(
                f"Missing metric {metric}: {path}"
            )

        if not np.isfinite(
            df[metric].to_numpy(
                dtype=np.float64
            )
        ).all():
            raise RuntimeError(
                f"Nonfinite results: {path}"
            )

    df["seed"] = seed

    return df


def collect_results(out, seeds):
    frames = []

    for seed in seeds:
        for group in GROUPS:
            df = read_complete_group(
                out, seed, group
            )

            if df is not None:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def summarize(out, seeds):
    df = collect_results(out, seeds)

    if df.empty:
        return

    df.to_csv(
        out / "all_models_all_seeds.csv",
        index=False,
    )

    summary = (
        df.groupby(["algorithm", "group"])[METRICS]
        .agg(["mean", "std", "count"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        out / "mean_std_summary.csv",
        index=False,
    )

    paired_rows = []

    for algorithm in ALGORITHMS:
        alg_df = df[
            df["algorithm"] == algorithm
        ]

        real = alg_df[
            alg_df["group"] == "real_dynamic"
        ].set_index("seed")

        for benchmark in (
            "clean",
            "frozen_dynamic",
            "time_shuffled",
        ):
            baseline = alg_df[
                alg_df["group"] == benchmark
            ].set_index("seed")

            common_seeds = sorted(
                set(real.index)
                & set(baseline.index)
            )

            for seed in common_seeds:
                record = {
                    "algorithm": algorithm,
                    "benchmark": benchmark,
                    "seed": seed,
                }

                for metric in METRICS:
                    record[f"delta_{metric}"] = (
                        float(real.loc[seed, metric])
                        - float(baseline.loc[seed, metric])
                    )

                paired_rows.append(record)

    if paired_rows:
        paired = pd.DataFrame(
            paired_rows
        )

        paired.to_csv(
            out / "paired_seed_differences.csv",
            index=False,
        )

        comparisons = (
            paired.groupby(
                ["algorithm", "benchmark"]
            )
            .agg(
                n=("seed", "count"),
                mean_delta_multiple=(
                    "delta_final_multiple", "mean"
                ),
                std_delta_multiple=(
                    "delta_final_multiple", "std"
                ),
                mean_delta_sharpe=(
                    "delta_sharpe", "mean"
                ),
                std_delta_sharpe=(
                    "delta_sharpe", "std"
                ),
                positive_sharpe_seeds=(
                    "delta_sharpe",
                    lambda x: int((x > 0).sum()),
                ),
            )
            .reset_index()
        )

        comparisons.to_csv(
            out / "temporal_ablation_comparison.csv",
            index=False,
        )

    print()
    print("=" * 88)
    print("TEMPORAL ABLATION: MEAN +/- STD")
    print("=" * 88)

    for _, row in summary.iterrows():
        print(
            f"{row['algorithm']:4s} "
            f"{row['group']:15s} "
            f"n={int(row['sharpe_count'])} "
            f"Multiple={row['final_multiple_mean']:.5f} "
            f"+/- {row['final_multiple_std']:.5f} "
            f"Sharpe={row['sharpe_mean']:.5f} "
            f"+/- {row['sharpe_std']:.5f}"
        )

    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=80000,
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--max-lag",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--embeddings",
        default="outputs/unified_kgt_finrl_x_v1",
    )

    parser.add_argument(
        "--out",
        default="results/unified_temporal_ablation_v1",
    )

    args = parser.parse_args()

    if args.timesteps < 1:
        raise RuntimeError(
            "Invalid timesteps"
        )

    if len(args.seeds) != len(set(args.seeds)):
        raise RuntimeError(
            "Duplicate seeds"
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # Safety:
    # Do not silently resume results generated
    # with different experimental settings.
    config_path = out / "experiment_config.json"

    configuration = {
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "shuffle_seed": args.shuffle_seed,
        "max_lag": args.max_lag,
        "embeddings": args.embeddings,
        "groups": GROUPS,
        "algorithms": ALGORITHMS,
        "train": "2018-2021",
        "validation": "2022",
        "test": "2023",
    }

    if config_path.exists():
        previous = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        if previous != configuration:
            raise RuntimeError(
                "Existing output contains a different "
                "experiment configuration. "
                "Choose another --out directory."
            )
    else:
        config_path.write_text(
            json.dumps(
                configuration,
                indent=2,
            ),
            encoding="utf-8",
        )

    embedding_dir = ROOT / args.embeddings

    market, real_kg = (
        load_market_and_dynamic(
            embedding_dir
        )
    )

    frozen_kg, frozen_date = (
        make_frozen_dynamic(
            real_kg
        )
    )

    shuffled_kg, shuffle_audit = (
        make_past_only_shuffle(
            real_kg,
            seed=args.shuffle_seed,
            max_lag=args.max_lag,
        )
    )

    shuffle_audit.to_csv(
        out / "temporal_shuffle_audit.csv",
        index=False,
    )

    metadata = {
        **configuration,
        "frozen_date": frozen_date,
        "shuffle_type": (
            "past_only_random_temporal_reassignment"
        ),
        "no_future_source_dates": True,
        "kg_weights_walk_forward": False,
        "strict_pit_certified": False,
    }

    (out / "ablation_metadata.json").write_text(
        json.dumps(
            metadata, indent=2
        ),
        encoding="utf-8",
    )

    panels, feature_lists = (
        build_panels(
            market,
            real_kg,
            frozen_kg,
            shuffled_kg,
        )
    )

    # Confirm that previously installed patch
    # saves all three test algorithm results.
    source = Path(
        runner.__file__
    ).read_text(
        encoding="utf-8"
    )

    if (
        "all_algorithms_test_summary.csv"
        not in source
    ):
        raise RuntimeError(
            "The nine-model evaluation patch "
            "is missing from run_dynamic_kg_drl.py"
        )

    for seed in args.seeds:
        print()
        print("#" * 88)
        print("TEMPORAL ABLATION SEED:", seed)
        print("#" * 88)

        for group in GROUPS:
            existing = (
                read_complete_group(
                    out, seed, group
                )
            )

            if existing is not None:
                print(
                    "SKIP COMPLETED:",
                    seed, group,
                )
                continue

            group_out = (
                out
                / f"seed_{seed}"
                / group
            )

            print(
                "START",
                seed,
                group,
                flush=True,
            )

            runner.run_one_group(
                group=group,
                panel=panels[group],
                features=feature_lists[group],
                out=group_out,
                algorithms=ALGORITHMS,
                seed=seed,
                timesteps=args.timesteps,
            )

            # Verify the group really completed.
            verified = read_complete_group(
                out, seed, group
            )

            if verified is None:
                raise RuntimeError(
                    f"Incomplete output: {seed}/{group}"
                )

            # Save progress after each group.
            summarize(out, args.seeds)

    completed = collect_results(
        out, args.seeds
    )

    expected = len(args.seeds) * len(
        GROUPS
    ) * len(ALGORITHMS)

    if len(completed) != expected:
        raise RuntimeError(
            f"Expected {expected} rows, "
            f"found {len(completed)}"
        )

    summarize(out, args.seeds)

    print()
    print("=" * 88)
    print("TEMPORAL ABLATION EXPERIMENT COMPLETED")
    print("=" * 88)
    print("DRL models:", expected)
    print("Results:", out)


if __name__ == "__main__":
    main()
