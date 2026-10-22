#!/usr/bin/env python3

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from stable_baselines3 import DDPG

ROOT = Path("/home/doralin/projects/FinRL-Trading")
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.diagnostics import (
    run_real_market_reliance as base
)

TICKERS = list(base.TICKERS)
KG_COLS = list(base.KG_COLS)

DEFAULT_OUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/ddpg_actor_saturation_audit_v1"
)

EPS = 1e-12


def log(*args):
    print(*args, flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("SAVED:", path)


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def entropy(weights):
    w = np.asarray(weights, dtype=np.float64)
    return float(
        -np.sum(w * np.log(np.maximum(w, EPS)))
    )


def select_ddpg_models(experiment):
    discovered = base.discover_models(experiment)

    found = {}

    for seed, alg, path in discovered:
        if alg != "DDPG":
            continue

        found.setdefault(seed, []).append(path)

    if not found:
        raise RuntimeError(
            "No Real Dynamic DDPG model files found."
        )

    selected = []

    for seed in sorted(found):
        paths = found[seed]

        if len(paths) != 1:
            raise RuntimeError(
                f"Seed {seed} has {len(paths)} DDPG "
                f"model candidates: {paths}. "
                "Resolve duplicate models before auditing."
            )

        selected.append((seed, paths[0]))

    log("DDPG MODELS:", len(selected))

    for seed, path in selected:
        log(" ", seed, path)

    return selected


class ActorInspector:
    """
    Capture:
      - KG projector output
      - last Linear pre-Tanh activations
      - actor output after Tanh

    All captures correspond to model.predict(...)
    using the saved trained DDPG actor.
    """

    def __init__(self, model):
        self.model = model
        self.actor = model.policy.actor

        extractor = self.actor.features_extractor

        if not hasattr(extractor, "projector"):
            raise RuntimeError(
                "Saved actor has no KG projector."
            )

        self.projector = extractor.projector

        if not hasattr(self.actor, "mu"):
            raise RuntimeError(
                "DDPG actor has no mu network."
            )

        modules = list(self.actor.mu.children())

        last_linear_idx = None
        last_tanh_idx = None

        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                last_linear_idx = i
            if isinstance(module, nn.Tanh):
                last_tanh_idx = i

        if last_linear_idx is None:
            raise RuntimeError(
                "No Linear layer in Actor.mu."
            )

        if (
            last_tanh_idx is None
            or last_tanh_idx <= last_linear_idx
        ):
            raise RuntimeError(
                "Cannot locate final Actor Tanh."
            )

        self.pre_tanh_layer = modules[last_linear_idx]
        self.final_tanh_layer = modules[last_tanh_idx]

        if last_tanh_idx != len(modules) - 1:
            raise RuntimeError(
                "Final Tanh is not last Actor.mu layer."
            )

        self.captured = {}

        self.handles = [
            self.projector.register_forward_hook(
                self._make_hook("projected_kg")
            ),
            self.pre_tanh_layer.register_forward_hook(
                self._make_hook("pre_tanh")
            ),
            self.final_tanh_layer.register_forward_hook(
                self._make_hook("post_tanh")
            ),
        ]

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"{name}: non-tensor output"
                )

            self.captured[name] = (
                output.detach()
                .float()
                .cpu()
                .numpy()
                .copy()
            )

        return hook

    def inspect(self, obs):
        self.captured = {}

        with torch.no_grad():
            action, _ = self.model.predict(
                obs.astype(np.float32),
                deterministic=True,
            )

        required = [
            "projected_kg",
            "pre_tanh",
            "post_tanh",
        ]

        missing = [
            name
            for name in required
            if name not in self.captured
        ]

        if missing:
            raise RuntimeError(
                f"Actor hooks not triggered: {missing}"
            )

        action = np.asarray(
            action, dtype=np.float64
        ).reshape(-1)

        projected = self.captured[
            "projected_kg"
        ].reshape(-1)

        pre_tanh = self.captured[
            "pre_tanh"
        ].reshape(-1)

        post_tanh = self.captured[
            "post_tanh"
        ].reshape(-1)

        if len(action) != len(TICKERS):
            raise RuntimeError(
                f"Unexpected action length {len(action)}"
            )

        if len(pre_tanh) != len(TICKERS):
            raise RuntimeError(
                "Pre-Tanh output has wrong dimension."
            )

        if len(post_tanh) != len(TICKERS):
            raise RuntimeError(
                "Post-Tanh output has wrong dimension."
            )

        # SB3 unscales Actor Tanh [-1,1]
        # into the environment Box [0,1].
        expected_action = (
            post_tanh + 1.0
        ) / 2.0

        if not np.allclose(
            action,
            expected_action,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Actor Tanh / environment action "
                "scaling mismatch."
            )

        if not np.isfinite(projected).all():
            raise RuntimeError(
                "Nonfinite KG projector output."
            )

        if not np.isfinite(pre_tanh).all():
            raise RuntimeError(
                "Nonfinite pre-Tanh activations."
            )

        weights = stable_softmax(action)

        return {
            "projected": projected,
            "pre_tanh": pre_tanh,
            "post_tanh": post_tanh,
            "action": action,
            "weights": weights,
        }

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_kg_variants(
    date,
    index,
    dates,
    real_kg,
    frozen_kg,
    lag,
):
    lag_date = dates[max(0, index - lag)]

    return {
        "real": real_kg[date],
        "frozen": frozen_kg[date],
        "lagged": real_kg[lag_date],
        "zero": np.zeros_like(real_kg[date]),
    }


def append_vector_fields(row, prefix, vector):
    for i, value in enumerate(vector):
        row[f"{prefix}_{i}"] = float(value)


def audit_model(
    seed,
    path,
    observations,
    real_kg,
    frozen_kg,
    lag,
):
    log("\n===== AUDIT SEED", seed, "=====")

    model = DDPG.load(
        str(path),
        device="cpu",
    )

    inspector = ActorInspector(model)

    details = []
    comparisons = []
    allocations = []

    dates = [
        date for date, _ in observations
    ]

    previous_weights = {
        name: None
        for name in [
            "real", "frozen", "lagged", "zero"
        ]
    }

    try:
        for index, (date, observation) in enumerate(
            observations
        ):
            expected_shape = tuple(
                model.observation_space.shape
            )

            if observation.shape != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch on {date}: "
                    f"{observation.shape} vs "
                    f"{expected_shape}"
                )

            variants = build_kg_variants(
                date,
                index,
                dates,
                real_kg,
                frozen_kg,
                lag,
            )

            results = {}

            for name, kg in variants.items():
                modified = observation.copy()

                # Preserve real market features.
                # Replace only the KG rows.
                modified[-len(KG_COLS):, :] = kg

                result = inspector.inspect(modified)

                results[name] = result

                pre = result["pre_tanh"]
                post = result["post_tanh"]
                action = result["action"]
                weights = result["weights"]

                saturated_095 = np.mean(
                    np.abs(post) >= 0.95
                )

                saturated_099 = np.mean(
                    np.abs(post) >= 0.99
                )

                saturated_0999 = np.mean(
                    np.abs(post) >= 0.999
                )

                clipped_fraction = np.mean(
                    (action <= 1e-7)
                    | (action >= 1 - 1e-7)
                )

                previous = previous_weights[name]

                turnover = (
                    float("nan")
                    if previous is None
                    else 0.5 * float(
                        np.abs(weights - previous).sum()
                    )
                )

                previous_weights[name] = (
                    weights.copy()
                )

                row = {
                    "seed": seed,
                    "date": date,
                    "variant": name,
                    "pre_tanh_mean_abs": float(
                        np.abs(pre).mean()
                    ),
                    "pre_tanh_max_abs": float(
                        np.abs(pre).max()
                    ),
                    "pre_tanh_norm": float(
                        np.linalg.norm(pre)
                    ),
                    "post_tanh_abs_mean": float(
                        np.abs(post).mean()
                    ),
                    "tanh_saturation_095": float(
                        saturated_095
                    ),
                    "tanh_saturation_099": float(
                        saturated_099
                    ),
                    "tanh_saturation_0999": float(
                        saturated_0999
                    ),
                    "action_boundary_fraction":
                        float(clipped_fraction),
                    "action_sum": float(
                        action.sum()
                    ),
                    "weight_sum": float(
                        weights.sum()
                    ),
                    "weight_entropy": entropy(
                        weights
                    ),
                    "weight_max": float(
                        weights.max()
                    ),
                    "weight_min": float(
                        weights.min()
                    ),
                    "portfolio_turnover": turnover,
                    "projected_kg_norm": float(
                        np.linalg.norm(
                            result["projected"]
                        )
                    ),
                }

                for i, tic in enumerate(TICKERS):
                    row[f"action_{tic}"] = float(
                        action[i]
                    )
                    row[f"weight_{tic}"] = float(
                        weights[i]
                    )
                    row[f"pre_tanh_{tic}"] = float(
                        pre[i]
                    )

                details.append(row)

                allocations.append({
                    "seed": seed,
                    "date": date,
                    "variant": name,
                    **{
                        tic: float(weights[i])
                        for i, tic in enumerate(TICKERS)
                    },
                })

            reference = results["real"]

            for name in (
                "frozen", "lagged", "zero"
            ):
                target = results[name]

                def distance(key):
                    a = reference[key]
                    b = target[key]

                    return float(
                        np.linalg.norm(a - b)
                    )

                action_delta = (
                    reference["action"]
                    - target["action"]
                )

                weights_delta = (
                    reference["weights"]
                    - target["weights"]
                )

                comparisons.append({
                    "seed": seed,
                    "date": date,
                    "comparison": "real_vs_" + name,
                    "input_kg_l2": float(
                        np.linalg.norm(
                            variants["real"]
                            - variants[name]
                        )
                    ),
                    "projector_l2": distance(
                        "projected"
                    ),
                    "pre_tanh_l2": distance(
                        "pre_tanh"
                    ),
                    "post_tanh_l2": distance(
                        "post_tanh"
                    ),
                    "action_l1": float(
                        np.abs(
                            action_delta
                        ).sum()
                    ),
                    "action_l2": float(
                        np.linalg.norm(
                            action_delta
                        )
                    ),
                    "weight_l1": float(
                        np.abs(
                            weights_delta
                        ).sum()
                    ),
                    "weight_l2": float(
                        np.linalg.norm(
                            weights_delta
                        )
                    ),
                    "max_weight_change": float(
                        np.abs(
                            weights_delta
                        ).max()
                    ),
                })

    finally:
        inspector.close()
        del model

    return details, comparisons, allocations


def summarize_actions(detail):
    summary = (
        detail.groupby(
            ["seed", "variant"]
        )
        .agg(
            n_dates=("date", "size"),
            pre_tanh_abs_mean=(
                "pre_tanh_mean_abs", "mean"
            ),
            pre_tanh_abs_max=(
                "pre_tanh_max_abs", "max"
            ),
            saturation_095=(
                "tanh_saturation_095", "mean"
            ),
            saturation_099=(
                "tanh_saturation_099", "mean"
            ),
            saturation_0999=(
                "tanh_saturation_0999", "mean"
            ),
            action_boundary_fraction=(
                "action_boundary_fraction",
                "mean"
            ),
            mean_entropy=(
                "weight_entropy", "mean"
            ),
            mean_turnover=(
                "portfolio_turnover", "mean"
            ),
            mean_max_weight=(
                "weight_max", "mean"
            ),
            mean_projected_kg_norm=(
                "projected_kg_norm", "mean"
            ),
        )
        .reset_index()
    )

    return summary


def summarize_comparisons(comparisons):
    return (
        comparisons.groupby(
            "comparison"
        )
        .agg(
            n=("seed", "size"),
            mean_input_kg_l2=(
                "input_kg_l2", "mean"
            ),
            mean_projector_l2=(
                "projector_l2", "mean"
            ),
            max_projector_l2=(
                "projector_l2", "max"
            ),
            mean_pre_tanh_l2=(
                "pre_tanh_l2", "mean"
            ),
            max_pre_tanh_l2=(
                "pre_tanh_l2", "max"
            ),
            mean_post_tanh_l2=(
                "post_tanh_l2", "mean"
            ),
            mean_action_l1=(
                "action_l1", "mean"
            ),
            mean_weight_l1=(
                "weight_l1", "mean"
            ),
            max_weight_l1=(
                "weight_l1", "max"
            ),
        )
        .reset_index()
    )


def allocation_diversity(allocations):
    rows = []

    for (seed, variant), group in (
        allocations.groupby(
            ["seed", "variant"]
        )
    ):
        group = group.sort_values("date")

        w = group[TICKERS].to_numpy(
            dtype=np.float64
        )

        rounded = np.round(w, 8)

        n_unique = len(
            np.unique(
                rounded, axis=0
            )
        )

        daily_change = (
            0.5 * np.abs(
                np.diff(w, axis=0)
            ).sum(axis=1)
        )

        rows.append({
            "seed": seed,
            "variant": variant,
            "n_dates": len(group),
            "unique_allocations_8dp": n_unique,
            "mean_daily_weight_change": (
                float(daily_change.mean())
                if len(daily_change)
                else 0.0
            ),
            "max_daily_weight_change": (
                float(daily_change.max())
                if len(daily_change)
                else 0.0
            ),
            "mean_weight_entropy": float(
                np.mean([
                    entropy(x) for x in w
                ])
            ),
            "allocation_variation_sum": float(
                w.std(axis=0).sum()
            ),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=Path,
        default=(
            ROOT
            / "results/unified_temporal_ablation_v1"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("\n========================================")
    log("DDPG ACTOR SATURATION AUDIT")
    log("========================================")

    models = select_ddpg_models(
        args.experiment
    )

    real, frozen, features, frozen_date = (
        base.load_panels()
    )

    observations = (
        base.extract_test_observations(
            real, features
        )
    )

    dates = [
        date for date, _ in observations
    ]

    real_kg = {
        date: base.get_kg(real, date)
        for date in dates
    }

    frozen_kg = {
        date: base.get_kg(frozen, date)
        for date in dates
    }

    details = []
    comparisons = []
    allocations = []
    errors = []

    for seed, path in models:
        try:
            a, b, c = audit_model(
                seed=seed,
                path=path,
                observations=observations,
                real_kg=real_kg,
                frozen_kg=frozen_kg,
                lag=args.lag,
            )

            details.extend(a)
            comparisons.extend(b)
            allocations.extend(c)

            log("SUCCESS SEED:", seed)

        except Exception as exc:
            errors.append({
                "seed": seed,
                "model": str(path),
                "error": repr(exc),
            })

            log("FAILED SEED:", seed)
            traceback.print_exc()

    if errors:
        save_csv(
            pd.DataFrame(errors),
            args.out / "model_errors.csv"
        )

    if not details:
        raise RuntimeError(
            "No DDPG models successfully audited."
        )

    detail_df = pd.DataFrame(details)
    comparison_df = pd.DataFrame(comparisons)
    allocation_df = pd.DataFrame(allocations)

    save_csv(
        detail_df,
        args.out / "actor_daily_detail.csv"
    )

    save_csv(
        comparison_df,
        args.out / "actor_layer_comparison.csv"
    )

    save_csv(
        allocation_df,
        args.out / "portfolio_allocations.csv"
    )

    sat_summary = summarize_actions(
        detail_df
    )

    layer_summary = summarize_comparisons(
        comparison_df
    )

    diversity = allocation_diversity(
        allocation_df
    )

    save_csv(
        sat_summary,
        args.out / "saturation_by_seed.csv"
    )

    save_csv(
        layer_summary,
        args.out / "layer_sensitivity_summary.csv"
    )

    save_csv(
        diversity,
        args.out / "allocation_diversity.csv"
    )

    metadata = {
        "models_discovered": len(models),
        "models_succeeded": (
            len(models) - len(errors)
        ),
        "models_failed": len(errors),
        "observation_count_per_seed":
            len(observations),
        "test_period": "2023",
        "frozen_reference_date": frozen_date,
        "lag_trading_days": args.lag,
        "normalization":
            "Original runner.attach_kg",
        "actor_model": "SB3 DDPG",
        "action_range": [0, 1],
        "portfolio_weights":
            "StockPortfolioEnv softmax(action)",
        "turnover_definition":
            "0.5 * L1 change in successive "
            "target portfolio weights; "
            "does not include market drift",
        "limitations": [
            "Market data is regenerated.",
            "Equal-weight actions advance the "
            "environment for observation collection.",
            "Counterfactual KG features may be OOD.",
            "Weight-change turnover is not actual "
            "transaction turnover.",
            "Layer sensitivity alone does not "
            "establish causal performance benefit.",
        ],
    }

    with (
        args.out / "metadata.json"
    ).open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    log("\n===== SATURATION SUMMARY =====")
    log(
        sat_summary.to_string(index=False)
    )

    log("\n===== LAYER SENSITIVITY =====")
    log(
        layer_summary.to_string(index=False)
    )

    log("\n===== ALLOCATION DIVERSITY =====")
    log(
        diversity.to_string(index=False)
    )

    log("\n========================================")
    log("DDPG SATURATION AUDIT FINISHED")
    log("========================================")
    log("OUTPUT:", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
