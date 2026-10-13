#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.data import prepare_data
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    save_evaluation,
)


DEFAULT_MODES = [
    "technical",
    "fusion",
    "fusion_kg",
    "hybrid_soft",
    "hybrid_soft_kg",
    "hybrid_veto",
    "hybrid_veto_kg",
]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=str(
            ROOT / "config.yaml"
        ),
    )

    ap.add_argument(
        "--agents",
        nargs="+",
        default=["PPO"],
    )

    ap.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
    )

    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            42,
            123,
            2026,
        ],
    )

    ap.add_argument(
        "--out",
        default=str(
            ROOT / "outputs"
        ),
    )

    args = ap.parse_args()


    # ========================================================
    # Load config
    # ========================================================

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)


    # ========================================================
    # Make paths absolute relative to sec_rag_drl/
    # ========================================================

    for k in [
        "market_csv",
        "rag_csv",
        "kg_csv",
    ]:

        if k not in cfg["data"]:
            continue

        p = Path(
            cfg["data"][k]
        )

        if not p.is_absolute():

            cfg["data"][k] = str(
                ROOT / p
            )


    # ========================================================
    # Prepare data once
    # ========================================================

    (
        train_data,
        test_data,
        scaler,
        panel,
    ) = prepare_data(cfg)


    print("\n" + "=" * 80)
    print("DATA READY")
    print("=" * 80)

    print(
        "Train dates:",
        train_data.dates[0],
        "->",
        train_data.dates[-1],
    )

    print(
        "Test dates:",
        test_data.dates[0],
        "->",
        test_data.dates[-1],
    )

    print(
        "Tickers:",
        train_data.tickers,
    )

    print(
        "Technical shape:",
        train_data.tech.shape,
    )

    print(
        "RAG shape:",
        train_data.rag_state.shape,
    )

    print(
        "KG shape:",
        train_data.kg_state.shape,
    )

    print(
        "KG feature count:",
        len(
            train_data.kg_names
        ),
    )


    # ========================================================
    # Output directory
    # ========================================================

    out_root = Path(
        args.out
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    all_metrics = []


    # ========================================================
    # Experiments
    # ========================================================

    for agent in args.agents:

        for mode in args.modes:

            for seed in args.seeds:

                run_name = (
                    f"{agent.upper()}"
                    f"__{mode}"
                    f"__seed{seed}"
                )

                run_dir = (
                    out_root
                    / run_name
                )


                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    run_name
                )

                print(
                    "=" * 80
                )


                # --------------------------------------------
                # Train
                # --------------------------------------------

                model = train_model(
                    agent,
                    train_data,
                    cfg,
                    mode,
                    seed,
                    run_dir,
                )


                # --------------------------------------------
                # Evaluate
                # --------------------------------------------

                (
                    hist,
                    W,
                    metrics,
                ) = evaluate_model(
                    model,
                    test_data,
                    cfg,
                    mode,
                )


                # --------------------------------------------
                # Save evaluation
                # --------------------------------------------

                save_evaluation(
                    hist,
                    W,
                    metrics,
                    test_data.tickers,
                    run_dir,
                )


                # --------------------------------------------
                # Collect summary row
                # --------------------------------------------

                row = {
                    "agent":
                        agent.upper(),

                    "mode":
                        mode,

                    "seed":
                        seed,

                    **metrics,
                }


                all_metrics.append(
                    row
                )


                # save incrementally
                pd.DataFrame(
                    all_metrics
                ).to_csv(
                    out_root
                    / "all_metrics.csv",
                    index=False,
                )


    # ========================================================
    # Aggregate
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )


    metrics_df.to_csv(
        out_root
        / "all_metrics.csv",
        index=False,
    )


    numeric = (
        metrics_df
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    numeric = [
        c
        for c in numeric
        if c != "seed"
    ]


    summary = (
        metrics_df
        .groupby(
            [
                "agent",
                "mode",
            ]
        )[numeric]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )


    summary.to_csv(
        out_root
        / "summary_by_seed.csv"
    )


    print("\nSaved:")

    print(
        out_root
        / "all_metrics.csv"
    )

    print(
        out_root
        / "summary_by_seed.csv"
    )


if __name__ == "__main__":
    main()
