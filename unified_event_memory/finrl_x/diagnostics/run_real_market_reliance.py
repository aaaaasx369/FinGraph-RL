#!/usr/bin/env python3

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/doralin/projects/FinRL-Trading")
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C, PPO, DDPG
from unified_event_memory.finrl_x import run_dynamic_kg_drl as runner
from unified_event_memory.finrl_x import run_temporal_ablation as ablation

TICKERS = list(runner.TICKERS)
KG_COLS = list(runner.KG_COLS)
ALGORITHMS = {"A2C": A2C, "PPO": PPO, "DDPG": DDPG}

OUT_DEFAULT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/results/"
    "kg_feature_diagnostics_v2"
)


def log(*args):
    print(*args, flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("SAVED:", path)


def load_panels():
    log("\n========== BUILD ORIGINAL MARKET/KG PANELS ==========")

    market, kg = ablation.load_market_and_dynamic(
        ROOT / "outputs/unified_kgt_finrl_x_v1"
    )

    frozen_kg, reference_date = ablation.make_frozen_dynamic(kg)

    real_panel = runner.attach_kg(market, kg)
    frozen_panel = runner.attach_kg(market, frozen_kg)

    features = list(runner.exp.BASE_FEATURES) + KG_COLS

    real_panel = runner.prepare_panel(real_panel)
    frozen_panel = runner.prepare_panel(frozen_panel)

    keys = ["date", "tic"]

    if not real_panel[keys].equals(frozen_panel[keys]):
        raise RuntimeError("Real and Frozen panels are misaligned")

    log("Market rows:", len(market))
    log("Real KG rows:", len(real_panel))
    log("Frozen reference:", reference_date)
    log("KG dimensions:", len(KG_COLS))
    log("Tickers:", TICKERS)

    return real_panel, frozen_panel, features, reference_date


def frozen_audit(real, frozen, reference_date, out):
    log("\n========== EXACT FROZEN KG AUDIT ==========")

    records = []

    for ticker in TICKERS:
        r = real.loc[real.tic == ticker, KG_COLS].to_numpy(
            dtype=np.float64
        )
        f = frozen.loc[frozen.tic == ticker, KG_COLS].to_numpy(
            dtype=np.float64
        )

        if r.shape != f.shape:
            raise RuntimeError(f"Shape mismatch for {ticker}")

        records.append({
            "ticker": ticker,
            "frozen_reference_date": reference_date,
            "dates": len(r),
            "real_mean_l2_norm": np.linalg.norm(r, axis=1).mean(),
            "frozen_mean_l2_norm": np.linalg.norm(f, axis=1).mean(),
            "frozen_max_temporal_std": f.std(axis=0).max(),
            "frozen_nonzero_fraction": np.mean(np.abs(f) > 1e-8),
            "real_nonzero_fraction": np.mean(np.abs(r) > 1e-8),
            "real_frozen_mean_l2": np.linalg.norm(
                r - f, axis=1
            ).mean(),
        })

    summary = pd.DataFrame(records)

    save_csv(summary, out / "exact_frozen_audit.csv")
    log(summary.to_string(index=False))

    company_vectors = []

    for ticker in TICKERS:
        vector = (
            frozen.loc[frozen.tic == ticker, KG_COLS]
            .iloc[0]
            .to_numpy(np.float64)
        )
        company_vectors.append(vector)

    company_vectors = np.stack(company_vectors)

    differences = []

    for i in range(len(TICKERS)):
        for j in range(i + 1, len(TICKERS)):
            differences.append({
                "company_a": TICKERS[i],
                "company_b": TICKERS[j],
                "frozen_l2_distance": float(
                    np.linalg.norm(
                        company_vectors[i] - company_vectors[j]
                    )
                ),
            })

    save_csv(
        pd.DataFrame(differences),
        out / "frozen_company_distances.csv",
    )


def reset_obs(env):
    result = env.reset()

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], dict)
    ):
        return result[0]

    return result


def step_obs(env, action):
    result = env.step(action)

    if len(result) == 4:
        obs, reward, done, info = result
        return obs, bool(done)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, bool(terminated or truncated)

    raise RuntimeError("Unsupported environment.step return")


def extract_test_observations(real, features):
    log("\n========== COLLECT 2023 MARKET OBSERVATIONS ==========")

    test = real.loc[
        (real.date >= "2023-01-01")
        & (real.date < "2024-01-01")
    ].copy()

    test = runner.prepare_panel(test)

    dates = sorted(test.date.unique())

    if not dates:
        raise RuntimeError("No 2023 market dates")

    # FinRL StockPortfolioEnv requires one integer index
    # per trading date, repeated for all five tickers.
    test = test.copy()

    test.index = test["date"].factorize(sort=True)[0]

    # Validate each date maps to one environment step.
    counts = test.groupby(level=0)["tic"].nunique()

    if not (counts == len(TICKERS)).all():
        raise RuntimeError(
            "Invalid FinRL date-indexed portfolio panel"
        )

    env = runner.environment(test, features)

    obs = reset_obs(env)

    observations = []

    for i, date in enumerate(dates):
        arr = np.asarray(obs, dtype=np.float32)

        expected_kg = (
            test.loc[test.date == date]
            .set_index("tic")
            .loc[TICKERS, KG_COLS]
            .to_numpy(dtype=np.float32)
            .T
        )

        if arr.ndim != 2:
            raise RuntimeError(
                f"Expected 2D observation, got {arr.shape}"
            )

        if arr.shape[1] != len(TICKERS):
            raise RuntimeError(
                f"Unexpected stock dimension: {arr.shape}"
            )

        if arr.shape[0] < len(KG_COLS):
            raise RuntimeError(
                f"Observation does not contain KG: {arr.shape}"
            )

        actual_kg = arr[-len(KG_COLS):, :]

        if not np.allclose(
            actual_kg, expected_kg,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError(
                f"KG observation mismatch on {date}. "
                "Check ticker ordering and environment indexing."
            )

        observations.append((date, arr.copy()))

        if i + 1 < len(dates):
            action = np.full(
                len(TICKERS),
                1.0 / len(TICKERS),
                dtype=np.float32,
            )

            obs, done = step_obs(env, action)

            if done:
                raise RuntimeError(
                    f"Environment terminated early at {date}"
                )

    log("Validated observation dates:", len(observations))
    log("Observation shape:", observations[0][1].shape)

    return observations


def get_kg(panel, date):
    part = (
        panel.loc[panel.date == date]
        .set_index("tic")
        .loc[TICKERS, KG_COLS]
    )

    return part.to_numpy(dtype=np.float32).T


def discover_models(root):
    models = []

    for path in root.rglob("*.zip"):
        path_parts = [p.lower() for p in path.parts]
        full_text = "/".join(path_parts)

        if "real_dynamic" not in full_text:
            continue

        seed = None

        for s in [42, 123, 2026, 7, 999]:
            if f"seed_{s}" in path_parts:
                seed = s
                break

        if seed is None:
            continue

        alg = None

        for name in ALGORITHMS:
            pattern = rf"(?<![a-z]){name.lower()}(?![a-z])"

            if re.search(pattern, full_text):
                alg = name
                break

        if alg is None:
            continue

        models.append((seed, alg, path))

    return sorted(
        models,
        key=lambda item: (
            item[0],
            item[1],
            str(item[2]),
        ),
    )


def action_sensitivity(
    real,
    frozen,
    observations,
    experiment,
    out,
    n_dates,
    lag,
):
    log("\n========== REAL-MARKET ACTION SENSITIVITY ==========")

    models = discover_models(experiment)

    log("Model files discovered:", len(models))

    for seed, algorithm, path in models:
        log(seed, algorithm, path)

    if not models:
        raise RuntimeError(
            "No real_dynamic SB3 ZIP files discovered. "
            "Please inspect model directory layout."
        )

    sample_indices = np.unique(
        np.linspace(
            0,
            len(observations) - 1,
            min(n_dates, len(observations)),
        ).astype(int)
    )

    selected = [observations[i] for i in sample_indices]

    dates = [date for date, _ in observations]
    date_positions = {date: i for i, date in enumerate(dates)}

    real_kg = {date: get_kg(real, date) for date in dates}
    frozen_kg = {date: get_kg(frozen, date) for date in dates}

    records = []
    errors = []
    successes = 0

    for seed, alg, path in models:
        log("\nLOADING:", seed, alg, path)

        try:
            model = ALGORITHMS[alg].load(
                str(path),
                device="cpu",
            )

            expected_shape = tuple(
                model.observation_space.shape
            )

            model_rows = []

            for date, original_obs in selected:
                if original_obs.shape != expected_shape:
                    raise RuntimeError(
                        f"Observation shape mismatch: "
                        f"{original_obs.shape} vs {expected_shape}"
                    )

                date_index = date_positions[date]
                lag_date = dates[max(0, date_index - lag)]

                variants = {
                    "real": real_kg[date],
                    "frozen": frozen_kg[date],
                    "lagged": real_kg[lag_date],
                    "zero": np.zeros_like(real_kg[date]),
                }

                actions = {}

                for name, kg in variants.items():
                    modified = original_obs.copy()

                    # Critical:
                    # Only last 200 KG rows are changed.
                    # Covariance/technical market features
                    # remain exactly identical.
                    modified[-len(KG_COLS):, :] = kg

                    action, _ = model.predict(
                        modified,
                        deterministic=True,
                    )

                    actions[name] = np.asarray(
                        action,
                        dtype=np.float64,
                    ).reshape(-1)

                    if len(actions[name]) != len(TICKERS):
                        raise RuntimeError(
                            "Unexpected action dimensions"
                        )

                reference = actions["real"]

                for name in ("frozen", "lagged", "zero"):
                    comparison = actions[name]
                    delta = comparison - reference

                    row = {
                        "seed": seed,
                        "algorithm": alg,
                        "date": date,
                        "comparison": f"real_vs_{name}",
                        "l1_action_change": float(
                            np.abs(delta).sum()
                        ),
                        "l2_action_change": float(
                            np.linalg.norm(delta)
                        ),
                        "max_weight_change": float(
                            np.abs(delta).max()
                        ),
                        "kg_l2_change": float(
                            np.linalg.norm(
                                variants[name] - variants["real"]
                            )
                        ),
                    }

                    for i, ticker in enumerate(TICKERS):
                        row[f"real_{ticker}"] = float(reference[i])
                        row[f"comparison_{ticker}"] = float(
                            comparison[i]
                        )
                        row[f"delta_{ticker}"] = float(delta[i])

                    model_rows.append(row)

            records.extend(model_rows)
            successes += 1

            log("MODEL OK:", seed, alg)

        except Exception as exc:
            log("MODEL FAILED:", repr(exc))

            errors.append({
                "seed": seed,
                "algorithm": alg,
                "path": str(path),
                "error": repr(exc),
            })

    if errors:
        save_csv(
            pd.DataFrame(errors),
            out / "model_errors.csv",
        )

    if not records:
        raise RuntimeError(
            "No successful model evaluations. "
            "Inspect model_errors.csv"
        )

    detail = pd.DataFrame(records)

    save_csv(
        detail,
        out / "real_market_action_detail.csv",
    )

    summary = (
        detail.groupby(["algorithm", "comparison"])
        .agg(
            n=("l1_action_change", "size"),
            mean_l1=("l1_action_change", "mean"),
            median_l1=("l1_action_change", "median"),
            max_l1=("l1_action_change", "max"),
            mean_l2=("l2_action_change", "mean"),
            mean_max_weight_change=(
                "max_weight_change", "mean"
            ),
            mean_kg_l2_change=("kg_l2_change", "mean"),
        )
        .reset_index()
    )

    save_csv(
        summary,
        out / "real_market_action_summary.csv",
    )

    seed_summary = (
        detail.groupby(["seed", "algorithm", "comparison"])
        .agg(
            mean_l1=("l1_action_change", "mean"),
            mean_max_weight_change=(
                "max_weight_change", "mean"
            ),
        )
        .reset_index()
    )

    save_csv(
        seed_summary,
        out / "action_by_seed.csv",
    )

    log("\n===== ACTION SUMMARY =====")
    log(summary.to_string(index=False))

    return {
        "model_files_found": len(models),
        "models_succeeded": successes,
        "models_failed": len(errors),
        "dates_sampled": len(selected),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "results/unified_temporal_ablation_v1",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DEFAULT,
    )

    parser.add_argument(
        "--max-dates",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=20,
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    real, frozen, features, reference_date = load_panels()

    frozen_audit(
        real,
        frozen,
        reference_date,
        args.out,
    )

    observations = extract_test_observations(
        real,
        features,
    )

    result = action_sensitivity(
        real=real,
        frozen=frozen,
        observations=observations,
        experiment=args.experiment,
        out=args.out,
        n_dates=args.max_dates,
        lag=args.lag,
    )

    metadata = {
        "market": "2023 StockPortfolioEnv observations",
        "normalization": (
            "Original runner.attach_kg / "
            "runner.standardize_kg, pooled train "
            "2018-2021, ddof=0, clip[-5,5]"
        ),
        "frozen_reference_date": reference_date,
        "lag_trading_days": args.lag,
        "limitations": [
            "Market data may be re-downloaded.",
            "Equal-weight actions advance market environment.",
            "Intervention measures action sensitivity, not returns.",
            "Counterfactual KG input may be out of distribution.",
        ],
        **result,
    }

    with (args.out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    log("\n========================================")
    log("REAL-MARKET KG DIAGNOSTICS COMPLETED")
    log("========================================")
    log("OUTPUT:", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
