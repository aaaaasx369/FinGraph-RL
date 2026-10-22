#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X Utility-RAG V1

Inspired by:
  - When Should Active RAG Retrieve? (2026)
  - Adaptive-RAG (NAACL 2024)
  - R3AG (2026)
  - Cost-Aware Retrieval (2025/2026)

Commands:
    inspect    Inspect original environment interfaces
    audit      Build daily evidence-utility diagnostics
    frontier   Build oracle and budget-matched frontiers

IMPORTANT
---------
REAL vs ZERO is evidence-content intervention,
NOT true STOP vs RETRIEVE.

Daily differences between full trajectories are
descriptive path-level contrasts, not independent
one-step causal treatment effects.

2022-2023 test labels MUST NOT be used to fit
a deployable router.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASETS = {
    "V4": "evidence_reliance_v4_test",
    "V5_FULL": "evidence_reliance_v5",
    "V5_MARKET": "evidence_reliance_v5_market",
}

MODES = (
    "REAL",
    "ZERO",
    "SHUFFLED",
    "LAGGED",
)

VALUE_CANDIDATES = (
    "equity",
    "value",
    "portfolio_value",
    "account_value",
    "total_asset",
    "total_assets",
    "asset_value",
    "portfolio_asset",
    "end_value",
)

RETURN_CANDIDATES = (
    "net_return",
    "daily_return",
    "portfolio_return",
    "step_return",
    "period_return",
)

DATE_CANDIDATES = (
    "date",
    "datetime",
    "timestamp",
    "day",
)


def log(*args):
    print(*args, flush=True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def first_column(df, candidates):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in candidates:
        if name in lookup:
            return lookup[name]

    return None


def load_mode(folder, mode, split):
    path = (
        folder
        / f"{split}_{mode.lower()}_portfolio.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"Empty replay history: {path}"
        )

    date_col = first_column(
        df, DATE_CANDIDATES
    )

    if date_col is None:
        raise RuntimeError(
            f"Missing date column in {path}\n"
            f"Available: {list(df.columns)}"
        )

    dates = pd.to_datetime(
        df[date_col],
        errors="raise",
    ).dt.normalize()

    if dates.isna().any():
        raise RuntimeError(
            f"Invalid dates: {path}"
        )

    if dates.duplicated().any():
        raise RuntimeError(
            f"Duplicated dates: {path}"
        )

    if not dates.is_monotonic_increasing:
        raise RuntimeError(
            f"Unsorted dates: {path}"
        )

    df = df.copy()
    df["__date"] = dates

    return df.set_index("__date")


def extract_wealth(df, model, mode):
    col = first_column(
        df, VALUE_CANDIDATES
    )

    if col is None:
        raise RuntimeError(
            f"{model}/{mode}: Cannot identify "
            "portfolio wealth column.\n"
            f"Available columns:\n"
            f"{list(df.columns)}\n"
            "Inspect the original replay history; "
            "do not guess a financial return column."
        )

    value = pd.to_numeric(
        df[col],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(value).all():
        raise RuntimeError(
            f"{model}/{mode}: nonfinite wealth"
        )

    if not (value > 0).all():
        raise RuntimeError(
            f"{model}/{mode}: wealth must be positive"
        )

    return value, str(col)


def extract_period_log_returns(
    df, value, model, mode
):
    """
    Prefer actual daily return column when available.

    Otherwise, only calculate the wealth ratio between
    adjacent saved rows. The first saved day's return
    is unknown and is NOT assumed to be zero.
    """
    col = first_column(
        df, RETURN_CANDIDATES
    )

    if col is not None:
        r = pd.to_numeric(
            df[col],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(r).all():
            raise RuntimeError(
                f"{model}/{mode}: nonfinite returns"
            )

        if (r <= -1).any():
            raise RuntimeError(
                "Daily return <= -100%"
            )

        return np.log1p(r), "reported_daily_return"

    out = np.full(
        len(value),
        np.nan,
        dtype=float,
    )

    out[1:] = np.diff(
        np.log(value)
    )

    return out, "adjacent_wealth_ratios"


def fetch_summary(base, directory, split):
    path = (
        base / directory
        / f"{split}_reliance_summary.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    summary = pd.read_csv(path)

    if set(summary["mode"]) != set(MODES):
        raise RuntimeError(
            f"Incomplete modes in {path}"
        )

    return summary.set_index("mode")


def audit(args):
    base = args.base.resolve()
    output = args.out.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_days = []
    all_summary = []
    metadata = {}

    for model, directory in DATASETS.items():
        folder = base / directory

        summary = fetch_summary(
            base, directory, args.split
        )

        histories = {}

        for mode in MODES:
            histories[mode] = load_mode(
                folder,
                mode,
                args.split,
            )

        real_dates = histories["REAL"].index

        for mode in MODES:
            if not real_dates.equals(
                histories[mode].index
            ):
                raise RuntimeError(
                    f"{model}: date alignment "
                    f"failure for {mode}"
                )

        if args.split == "test":
            years = set(real_dates.year)

            if not years.issubset({2022, 2023}):
                raise RuntimeError(
                    f"{model}: unexpected test years {years}"
                )

        wealth = {}
        log_returns = {}
        wealth_columns = {}
        return_methods = {}

        for mode in MODES:
            df = histories[mode]

            w, col = extract_wealth(
                df, model, mode
            )

            lr, method = (
                extract_period_log_returns(
                    df, w, model, mode
                )
            )

            wealth[mode] = w
            log_returns[mode] = lr
            wealth_columns[mode] = col
            return_methods[mode] = method

        # Return consistency is checked against
        # the source summary.
        #
        # A wealth series can omit its initial
        # baseline value, so compare if its
        # saved first value is the baseline.
        #
        # Otherwise record the discrepancy.
        consistency = {}

        for mode in MODES:
            w = wealth[mode]

            computed = (
                w[-1] / w[0] - 1
            )

            expected = float(
                summary.loc[
                    mode, "total_return"
                ]
            )

            difference = (
                computed - expected
            )

            consistency[mode] = {
                "saved_wealth_ratio_return": computed,
                "original_reported_return": expected,
                "difference": difference,
            }

        # Only the daily return effect of REAL
        # vs ZERO is the requested diagnostic.
        #
        # This is not STOP counterfactual.
        daily = pd.DataFrame({
            "date": real_dates.strftime(
                "%Y-%m-%d"
            ),
            "model": model,
            "real_log_return": (
                log_returns["REAL"]
            ),
            "zero_log_return": (
                log_returns["ZERO"]
            ),
            "shuffled_log_return": (
                log_returns["SHUFFLED"]
            ),
            "lagged_log_return": (
                log_returns["LAGGED"]
            ),
        })

        daily["evidence_log_effect"] = (
            daily["real_log_return"]
            - daily["zero_log_return"]
        )

        daily["retrieval_harm_proxy"] = (
            daily["evidence_log_effect"] < 0
        )

        daily["retrieval_help_proxy"] = (
            daily["evidence_log_effect"] > 0
        )

        daily["cost_normalized_proxy"] = (
            daily["evidence_log_effect"]
            - args.cost_per_query
            * (
                float(
                    summary.loc[
                        "REAL", "total_queries"
                    ]
                )
                / len(daily)
            )
        )

        daily["effect_type"] = (
            "trajectory_evidence_contrast"
        )

        all_days.append(daily)

        for mode in MODES:
            all_summary.append({
                "model": model,
                "mode": mode,
                "total_return": float(
                    summary.loc[
                        mode, "total_return"
                    ]
                ),
                "sharpe": float(
                    summary.loc[
                        mode, "sharpe"
                    ]
                ),
                "max_drawdown": float(
                    summary.loc[
                        mode, "max_drawdown"
                    ]
                ),
                "retrieval_days": int(
                    summary.loc[
                        mode, "retrieval_days"
                    ]
                ),
                "total_queries": int(
                    summary.loc[
                        mode, "total_queries"
                    ]
                ),
            })

        valid_effects = (
            daily["evidence_log_effect"]
            .dropna()
        )

        metadata[model] = {
            "wealth_columns": wealth_columns,
            "return_methods": return_methods,
            "consistency": consistency,
            "days": len(daily),
            "effect_days": len(valid_effects),
            "mean_evidence_log_effect": (
                float(valid_effects.mean())
                if len(valid_effects)
                else None
            ),
            "harm_fraction": (
                float(
                    (valid_effects < 0).mean()
                )
                if len(valid_effects)
                else None
            ),
            "WARNING": (
                "REAL/ZERO is evidence-content "
                "intervention, not STOP. "
                "Daily contrasts use potentially "
                "different portfolio trajectories."
            ),
        }

        log(
            f"[AUDIT] {model}:",
            f"{len(daily)} dates,",
            f"{len(valid_effects)} daily contrasts",
        )

    merged = pd.concat(
        all_days,
        ignore_index=True,
    )

    merged.to_csv(
        output / "daily_evidence_utility.csv",
        index=False,
    )

    pd.DataFrame(
        all_summary
    ).to_csv(
        output / "model_summary.csv",
        index=False,
    )

    save_json(
        output / "audit_metadata.json",
        metadata,
    )

    log()
    log("[SAVED] daily_evidence_utility.csv")
    log("[SAVED] model_summary.csv")
    log("[SAVED] audit_metadata.json")

    log()
    log("EVIDENCE UTILITY AUDIT: COMPLETE")


def frontier(args):
    folder = args.out.resolve()

    path = (
        folder / "daily_evidence_utility.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Run 'audit' first."
        )

    df = pd.read_csv(path)

    rng = np.random.default_rng(
        args.seed
    )

    output = []

    for model, raw in df.groupby(
        "model", sort=True
    ):
        raw = raw.copy()

        # First wealth-ratio return may not
        # be observable. Exclude unavailable rows.
        raw = raw[
            np.isfinite(
                raw["evidence_log_effect"]
            )
        ]

        effects = raw[
            "evidence_log_effect"
        ].to_numpy(dtype=float)

        n = len(effects)

        if n < 10:
            raise RuntimeError(
                f"Too few daily effects: {model}"
            )

        # Historical oracle:
        # chooses dates using realized effects.
        # NOT deployable; not valid for training
        # or strategy test performance claims.
        oracle_order = np.argsort(
            -effects,
            kind="stable",
        )

        # Lagged volatility-style uncertainty
        # baseline, using only prior returns.
        prev = np.abs(
            raw["zero_log_return"]
            .shift(1)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        uncertainty_order = np.argsort(
            -prev,
            kind="stable",
        )

        for budget in np.linspace(
            0, 1, 11
        ):
            k = int(
                np.floor(
                    n * budget + 1e-9
                )
            )

            mask_oracle = np.zeros(
                n,
                dtype=bool,
            )

            mask_uncertainty = np.zeros(
                n,
                dtype=bool,
            )

            mask_oracle[
                oracle_order[:k]
            ] = True

            mask_uncertainty[
                uncertainty_order[:k]
            ] = True

            # Random budget-matched baseline.
            random_utilities = []

            for _ in range(
                args.random_repeats
            ):
                if k == 0:
                    chosen = np.array(
                        [],
                        dtype=int,
                    )
                else:
                    chosen = rng.choice(
                        n, size=k,
                        replace=False,
                    )

                random_utilities.append(
                    float(
                        effects[chosen].sum()
                    )
                )

            for policy, chosen, utility in [
                (
                    "ORACLE_IN_SAMPLE",
                    mask_oracle,
                    float(
                        effects[
                            mask_oracle
                        ].sum()
                    ),
                ),
                (
                    "LAGGED_VOLATILITY_HEURISTIC",
                    mask_uncertainty,
                    float(
                        effects[
                            mask_uncertainty
                        ].sum()
                    ),
                ),
                (
                    "RANDOM_MEAN",
                    None,
                    float(
                        np.mean(
                            random_utilities
                        )
                    ),
                ),
            ]:
                cost = (
                    k * args.cost_per_query
                )

                output.append({
                    "model": model,
                    "policy": policy,
                    "budget_fraction": k/n,
                    "selected_days": k,
                    "gross_log_utility_proxy":
                        utility,
                    "query_cost_proxy": cost,
                    "net_log_utility_proxy":
                        utility - cost,
                    "NOT_ACTUAL_BACKTEST": True,
                })

    result = pd.DataFrame(output)

    output_path = (
        folder / "budget_frontier.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    log()
    log("===== DIAGNOSTIC BUDGET FRONTIER =====")

    print(
        result[
            result["budget_fraction"]
            .between(0.39, 0.61)
        ].to_string(index=False)
    )

    log()
    log("[SAVED]", output_path)
    log(
        "WARNING: ORACLE_IN_SAMPLE uses "
        "future realized outcomes."
    )
    log(
        "WARNING: This is an additive "
        "diagnostic, NOT a portfolio backtest."
    )
    log("BUDGET FRONTIER: COMPLETE")


def inspect_environment():
    from unified_event_memory.finrl_x.aegis_rl import (
        run_active_rag_v2 as v2,
        run_active_rag_v3 as v3,
        run_active_rag_v4 as v4,
        run_flare_raggym_ollama as old,
    )

    components = [
        ("V2 active env init",
         v2.ActiveRAGEnvironment.__init__),
        ("V2 active env _search",
         v2.ActiveRAGEnvironment._search),
        ("V3 router state",
         v3.ContextualActiveRAGEnv._router_state),
        ("V4 FLARE search",
         v4.flare_selfrag_search),
        ("Original environment step",
         old.MultiStepEnv.step),
        ("Original environment _obs",
         old.MultiStepEnv._obs),
        ("Original environment reset",
         old.MultiStepEnv.reset),
    ]

    output = (
        Path(
            "results/aegis_rl_v1/"
            "utility_rag_v1"
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output /
        "environment_source_audit.txt"
    )

    with path.open(
        "w", encoding="utf-8"
    ) as f:
        for name, func in components:
            title = (
                "\n"
                + "=" * 78
                + "\n"
                + name
                + "\n"
                + "=" * 78
                + "\n"
            )

            f.write(title)

            try:
                f.write(
                    inspect.getsource(func)
                )
            except Exception as exc:
                f.write(
                    f"Source unavailable: {exc}\n"
                )

    log(
        "[SAVED] Environment source audit:",
        path,
    )

    log(
        "This audit is required before "
        "implementing a real STOP intervention."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "audit",
            "frontier",
        ],
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "results/aegis_rl_v1"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "utility_rag_v1"
        ),
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--cost-per-query",
        type=float,
        default=0.0,
        help=(
            "Illustrative log-utility cost "
            "per one query. Zero by default."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_environment()

    elif args.command == "audit":
        audit(args)

    elif args.command == "frontier":
        frontier(args)


if __name__ == "__main__":
    main()
