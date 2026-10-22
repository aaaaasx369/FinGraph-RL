#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/doralin/projects/FinRL-Trading")

DEFAULT_KG = (
    ROOT
    / "outputs/unified_kgt_finrl_x_v1"
    / "daily_dynamic_kgt.csv"
)

DEFAULT_EXPERIMENT = (
    ROOT
    / "results/unified_temporal_ablation_v1"
)

DEFAULT_OUTPUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/kg_feature_diagnostics_v1"
)

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]

SEEDS = [42, 123, 2026, 7, 999]

ALGORITHMS = ["A2C", "PPO", "DDPG"]

EPS = 1e-12


def log(*args):
    print(*args, flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("SAVED:", path)


def find_column(df, candidates):
    mapping = {
        str(c).lower(): c for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    return None


def load_embeddings(path):
    log("\n===== LOAD KG EMBEDDINGS =====")
    log("FILE:", path)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    date_col = find_column(
        df,
        ["date", "datetime", "day", "trade_date"]
    )

    tic_col = find_column(
        df,
        ["tic", "ticker", "symbol", "company"]
    )

    if date_col is None or tic_col is None:
        raise ValueError(
            "Cannot identify date/ticker columns. "
            f"Actual columns: {list(df.columns)[:30]}"
        )

    df = df.rename(
        columns={
            date_col: "date",
            tic_col: "tic",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["tic"] = df["tic"].astype(str).str.upper()

    df = df[
        df["tic"].isin(TICKERS)
    ].copy()

    metadata_cols = {
        "date", "tic", "ticker", "symbol",
        "company", "datetime", "day",
        "trade_date",
    }

    numeric_candidates = [
        c for c in df.columns
        if c not in metadata_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Prefer names indicating embedding dimensions.
    preferred = [
        c for c in numeric_candidates
        if re.search(
            r"(?:emb|kgt|kg|dim|feature)[_-]?\d+$",
            str(c),
            re.IGNORECASE,
        )
    ]

    if len(preferred) == 200:
        feature_cols = preferred
    elif len(numeric_candidates) == 200:
        feature_cols = numeric_candidates
    else:
        # Handle wide CSV with unnamed numeric dimension columns.
        excluded = {
            "year", "month", "week", "time_id",
            "entity_id", "company_id",
        }

        feature_cols = [
            c for c in numeric_candidates
            if str(c).lower() not in excluded
        ]

        if len(feature_cols) != 200:
            raise ValueError(
                "Expected 200 KG features, found "
                f"{len(feature_cols)}. "
                f"Columns: {feature_cols[:30]}"
            )

    df = df.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    if df.duplicated(["date", "tic"]).any():
        raise ValueError(
            "Duplicate (date, ticker) rows found."
        )

    if df[feature_cols].isna().any().any():
        raise ValueError(
            "KG embedding contains NaN."
        )

    values = df[feature_cols].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "KG embedding contains Inf."
        )

    log("ROWS:", len(df))
    log("DIMENSIONS:", len(feature_cols))
    log("DATE RANGE:", df.date.min(), df.date.max())
    log("TICKERS:", sorted(df.tic.unique()))

    return df, feature_cols


def temporal_variation(df, features, out):
    log("\n===== TEMPORAL VARIATION =====")

    all_rows = []

    xcols = features

    for ticker, part in df.groupby("tic"):
        part = part.sort_values("date")

        dates = part["date"].to_numpy()
        x = part[xcols].to_numpy(np.float64)

        if len(x) < 2:
            continue

        delta = x[1:] - x[:-1]

        l2 = np.linalg.norm(delta, axis=1)

        norms_before = np.linalg.norm(
            x[:-1], axis=1
        )

        norms_after = np.linalg.norm(
            x[1:], axis=1
        )

        cosine = np.sum(
            x[1:] * x[:-1], axis=1
        ) / np.maximum(
            norms_after * norms_before, EPS
        )

        cosine = np.clip(cosine, -1, 1)

        cosine_distance = 1 - cosine

        relative_l2 = l2 / np.maximum(
            norms_before, EPS
        )

        changed_dimensions = np.sum(
            np.abs(delta) > 1e-8,
            axis=1,
        )

        for i in range(len(delta)):
            all_rows.append({
                "tic": ticker,
                "date": str(dates[i + 1])[:10],
                "l2_change": float(l2[i]),
                "relative_l2_change": float(
                    relative_l2[i]
                ),
                "cosine_distance": float(
                    cosine_distance[i]
                ),
                "changed_dimensions": int(
                    changed_dimensions[i]
                ),
                "changed": bool(l2[i] > 1e-8),
            })

    detail = pd.DataFrame(all_rows)

    save_csv(
        detail,
        out / "temporal_change_daily.csv"
    )

    summary = (
        detail.groupby("tic")
        .agg(
            transitions=("changed", "size"),
            changed_days=("changed", "sum"),
            mean_l2=("l2_change", "mean"),
            median_l2=("l2_change", "median"),
            max_l2=("l2_change", "max"),
            mean_relative_l2=(
                "relative_l2_change", "mean"
            ),
            mean_cosine_distance=(
                "cosine_distance", "mean"
            ),
            mean_changed_dimensions=(
                "changed_dimensions", "mean"
            ),
        )
        .reset_index()
    )

    summary["change_rate"] = (
        summary["changed_days"]
        / summary["transitions"]
    )

    save_csv(
        summary,
        out / "temporal_change_summary.csv"
    )

    log(summary.to_string(index=False))

    return summary


def frozen_audit(df, features, out):
    log("\n===== FROZEN KG AUDIT =====")

    train = df[
        (df.date >= "2018-01-01")
        & (df.date < "2022-01-01")
    ].copy()

    if train.empty:
        raise ValueError(
            "No training data in 2018-2021."
        )

    rows = []
    processed = {}

    for ticker in TICKERS:
        all_part = df[
            df.tic == ticker
        ].sort_values("date")

        tr = train[
            train.tic == ticker
        ].sort_values("date")

        if tr.empty or all_part.empty:
            continue

        train_x = tr[features].to_numpy(
            np.float64
        )

        all_x = all_part[features].to_numpy(
            np.float64
        )

        # Reconstruct the train-fitted z-score.
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)

        safe_std = np.where(
            std < EPS, 1.0, std
        )

        real_z = (all_x - mean) / safe_std

        # Frozen at the earliest observation.
        frozen_raw = np.repeat(
            all_x[0:1],
            len(all_x),
            axis=0,
        )

        # Normalize using REAL training statistics:
        frozen_real_scaler = (
            frozen_raw - mean
        ) / safe_std

        # Normalize using FROZEN training statistics:
        frozen_train = np.repeat(
            train_x[0:1],
            len(train_x),
            axis=0,
        )

        frozen_mean = frozen_train.mean(axis=0)
        frozen_std = frozen_train.std(axis=0)

        frozen_safe_std = np.where(
            frozen_std < EPS,
            1.0,
            frozen_std,
        )

        frozen_own_scaler = (
            frozen_raw - frozen_mean
        ) / frozen_safe_std

        rows.append({
            "tic": ticker,
            "n_dates": len(all_x),
            "n_features": len(features),
            "raw_frozen_std_max": float(
                frozen_raw.std(axis=0).max()
            ),
            "real_z_nonzero_fraction": float(
                np.mean(np.abs(real_z) > 1e-8)
            ),
            "frozen_real_scaler_nonzero_fraction":
                float(
                    np.mean(
                        np.abs(frozen_real_scaler)
                        > 1e-8
                    )
                ),
            "frozen_own_scaler_nonzero_fraction":
                float(
                    np.mean(
                        np.abs(frozen_own_scaler)
                        > 1e-8
                    )
                ),
            "frozen_own_scaler_max_abs": float(
                np.max(
                    np.abs(frozen_own_scaler)
                )
            ),
            "train_zero_variance_dimensions":
                int(np.sum(std < EPS)),
            "mean_real_z_norm": float(
                np.linalg.norm(
                    real_z, axis=1
                ).mean()
            ),
            "mean_frozen_own_z_norm": float(
                np.linalg.norm(
                    frozen_own_scaler,
                    axis=1,
                ).mean()
            ),
        })

        processed[ticker] = {
            "dates": all_part.date.to_numpy(),
            "raw": all_x,
            "real_z": real_z,
            "frozen_real_scaler":
                frozen_real_scaler,
            "frozen_own_scaler":
                frozen_own_scaler,
        }

    summary = pd.DataFrame(rows)

    save_csv(
        summary,
        out / "frozen_normalization_audit.csv"
    )

    log(summary.to_string(index=False))

    return processed


def find_models(experiment):
    log("\n===== SEARCH SAVED DRL MODELS =====")

    models = []

    for p in experiment.rglob("*.zip"):
        path_lower = str(p).lower()

        if "real_dynamic" not in path_lower:
            continue

        seed = None

        for s in SEEDS:
            if f"seed_{s}" in path_lower:
                seed = s
                break

        if seed is None:
            continue

        algorithm = None

        for alg in ALGORITHMS:
            if re.search(
                rf"(?<![a-z]){alg.lower()}(?![a-z])",
                path_lower,
            ):
                algorithm = alg
                break

        if algorithm is None:
            continue

        models.append({
            "seed": seed,
            "algorithm": algorithm,
            "path": p,
        })

    models.sort(
        key=lambda x: (
            x["seed"],
            x["algorithm"],
            str(x["path"]),
        )
    )

    for m in models:
        log(
            m["seed"],
            m["algorithm"],
            m["path"],
        )

    return models


def make_kg_matrix(
    processed,
    date,
    mode,
):
    columns = []

    for ticker in TICKERS:
        part = processed[ticker]

        dates = part["dates"]

        valid = np.flatnonzero(
            dates <= np.datetime64(date)
        )

        if len(valid) == 0:
            raise ValueError(
                f"No KG data for {ticker} {date}"
            )

        i = valid[-1]

        if mode == "real":
            vec = part["real_z"][i]
        elif mode == "frozen":
            vec = part["frozen_own_scaler"][i]
        elif mode == "zero":
            vec = np.zeros(
                part["real_z"].shape[1]
            )
        elif mode == "lagged":
            j = max(0, i - 20)
            vec = part["real_z"][j]
        else:
            raise ValueError(mode)

        columns.append(vec)

    # [200 KG dimensions, 5 stocks]
    return np.stack(columns, axis=1)


def get_test_dates(df, max_dates):
    dates = sorted(
        pd.to_datetime(
            df.loc[
                (df.date >= "2023-01-01")
                & (df.date < "2024-01-01"),
                "date",
            ].unique()
        )
    )

    if not dates:
        raise ValueError(
            "No 2023 KG dates."
        )

    n = min(max_dates, len(dates))

    indices = np.unique(
        np.linspace(
            0,
            len(dates) - 1,
            n,
        ).astype(int)
    )

    return [dates[i] for i in indices]


def load_observations(path, n_dates):
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        observations = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if "observations" not in data.files:
            raise ValueError(
                "NPZ must contain 'observations'."
            )
        observations = data["observations"]
    else:
        raise ValueError(
            "Use .npy or .npz observations."
        )

    if len(observations) < n_dates:
        raise ValueError(
            "Not enough observations."
        )

    return observations


def load_sb3_model(path, algorithm):
    from stable_baselines3 import A2C, PPO, DDPG

    # Register custom feature extractor classes used
    # when the trained models were originally saved.
    import unified_event_memory.finrl_x.run_dynamic_kg_drl

    classes = {
        "A2C": A2C,
        "PPO": PPO,
        "DDPG": DDPG,
    }

    return classes[algorithm].load(
        str(path),
        device="cpu",
    )


def action_sensitivity(
    df,
    processed,
    experiment,
    out,
    observation_file,
    max_dates,
):
    log("\n===== ACTION SENSITIVITY =====")

    models = find_models(experiment)

    if not models:
        log(
            "WARNING: No real_dynamic model ZIP found."
        )
        log(
            "Action test skipped; "
            "embedding audits are still available."
        )
        return

    dates = get_test_dates(
        df, max_dates
    )

    observations = load_observations(
        observation_file,
        len(dates),
    )

    mode_name = (
        "real_test_observations"
        if observations is not None
        else "zero_market_proxy"
    )

    log("MARKET INPUT MODE:", mode_name)

    rows = []
    errors = []

    for info in models:
        log(
            "\nMODEL:",
            info["seed"],
            info["algorithm"],
        )

        try:
            model = load_sb3_model(
                info["path"],
                info["algorithm"],
            )

            shape = tuple(
                model.observation_space.shape
            )

            if len(shape) != 2:
                raise ValueError(
                    f"Unsupported observation shape: {shape}"
                )

            if shape[-1] == len(TICKERS):
                kg_axis = 0
                nrows = shape[0]
                if nrows < 200:
                    raise ValueError(shape)
            elif shape[0] == len(TICKERS):
                kg_axis = 1
                nrows = shape[1]
                if nrows < 200:
                    raise ValueError(shape)
            else:
                raise ValueError(
                    "Cannot locate ticker axis. "
                    f"Shape={shape}"
                )

            expected_market_rows = nrows - 200

            log(
                "Observation:",
                shape,
                "Market rows:",
                expected_market_rows,
            )

            for k, date in enumerate(dates):
                if observations is None:
                    baseline = np.zeros(
                        shape, dtype=np.float32
                    )
                else:
                    baseline = np.asarray(
                        observations[k],
                        dtype=np.float32,
                    ).copy()

                    if baseline.shape != shape:
                        raise ValueError(
                            f"Observation mismatch: "
                            f"{baseline.shape} vs {shape}"
                        )

                acts = {}

                for kg_mode in [
                    "real",
                    "frozen",
                    "lagged",
                    "zero",
                ]:
                    obs = baseline.copy()

                    kg = make_kg_matrix(
                        processed,
                        date,
                        kg_mode,
                    ).astype(np.float32)

                    if kg_axis == 0:
                        obs[-200:, :] = kg
                    else:
                        obs[:, -200:] = kg.T

                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                    action = np.asarray(
                        action,
                        dtype=np.float64,
                    ).reshape(-1)

                    if action.size != len(TICKERS):
                        raise ValueError(
                            f"Expected 5 actions, "
                            f"got {action.size}"
                        )

                    acts[kg_mode] = action

                reference = acts["real"]

                for kg_mode in [
                    "frozen",
                    "lagged",
                    "zero",
                ]:
                    other = acts[kg_mode]
                    delta = other - reference

                    row = {
                        "seed": info["seed"],
                        "algorithm": info[
                            "algorithm"
                        ],
                        "date": str(date)[:10],
                        "comparison": (
                            "real_vs_" + kg_mode
                        ),
                        "observation_mode": mode_name,
                        "l1_action_change": float(
                            np.abs(delta).sum()
                        ),
                        "max_weight_change": float(
                            np.abs(delta).max()
                        ),
                        "l2_action_change": float(
                            np.linalg.norm(delta)
                        ),
                    }

                    for i, ticker in enumerate(TICKERS):
                        row[
                            f"delta_{ticker}"
                        ] = float(delta[i])

                    rows.append(row)

            del model

        except Exception as e:
            errors.append({
                "seed": info["seed"],
                "algorithm": info[
                    "algorithm"
                ],
                "path": str(info["path"]),
                "error": repr(e),
            })

            log(
                "MODEL FAILED:",
                repr(e)
            )

    if errors:
        save_csv(
            pd.DataFrame(errors),
            out / "action_model_errors.csv"
        )

    if not rows:
        log(
            "WARNING: No action results generated."
        )
        return

    detail = pd.DataFrame(rows)

    save_csv(
        detail,
        out / "action_sensitivity_detail.csv"
    )

    summary = (
        detail.groupby(
            ["algorithm", "comparison"]
        )
        .agg(
            n_samples=("l1_action_change", "size"),
            mean_l1=("l1_action_change", "mean"),
            median_l1=("l1_action_change", "median"),
            max_l1=("l1_action_change", "max"),
            mean_max_weight_change=(
                "max_weight_change", "mean"
            ),
            mean_l2=("l2_action_change", "mean"),
        )
        .reset_index()
    )

    save_csv(
        summary,
        out / "action_sensitivity_summary.csv"
    )

    log(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kg",
        type=Path,
        default=DEFAULT_KG,
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help=(
            "Optional real test observation NPY/NPZ. "
            "Must be aligned with sampled dates."
        ),
    )

    parser.add_argument(
        "--max-dates",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, features = load_embeddings(
        args.kg
    )

    temporal_variation(
        df,
        features,
        args.out,
    )

    processed = frozen_audit(
        df,
        features,
        args.out,
    )

    action_sensitivity(
        df=df,
        processed=processed,
        experiment=args.experiment,
        out=args.out,
        observation_file=args.observations,
        max_dates=args.max_dates,
    )

    metadata = {
        "kg_path": str(args.kg),
        "experiment_path": str(args.experiment),
        "output_path": str(args.out),
        "kg_dimensions": len(features),
        "tickers": TICKERS,
        "max_dates": args.max_dates,
        "observation_mode": (
            "real_test_observations"
            if args.observations
            else "zero_market_proxy"
        ),
        "important_limitations": [
            "Frozen z-score is reconstructed, not verified against training runner.",
            "Zero-market action probes are not actual backtest observations.",
            "Real observations require exact temporal alignment with KG dates.",
            "Embedding sensitivity is not equivalent to portfolio performance gain.",
        ],
    }

    with (
        args.out / "diagnostic_metadata.json"
    ).open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    log("\n===== DIAGNOSTICS FINISHED =====")
    log("OUTPUT:", args.out)


if __name__ == "__main__":
    main()
