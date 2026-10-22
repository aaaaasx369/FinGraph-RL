#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V8 Phase 2B
Semantic RAG + Auxiliary Encoder + A2C Portfolio RL

Arms:
    MARKET
    SEMANTIC_RAG
    AUX_RAG

Train:      2018-2020
Validation: 2021
Test:       DISABLED

Features:
    Existing FinRL-X Market + Dynamic KGT: 1265D
    Semantic SEC RAG: 64D
    Auxiliary pretrained representation: 32D
    Portfolio holdings: 5D

Common observation:
    1265 + 64 + 32 + 5 = 1366D

All arms use the same observation dimensions,
policy architecture, A2C parameters and seeds.

Other features:
    - historical lookback across train/val boundary
    - conservative SEC point-in-time availability
    - frozen pretrained auxiliary trunk
    - three seed experiments
    - validation portfolio backtest
    - REAL vs ZERO evidence reliance
    - source audit
    - no future-return features

This is a research adaptation of FinAgent-inspired
auxiliary financial representation with Semantic RAG.
Not an exact reproduction of FinAgent.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import A2C

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b,
    run_v8_replug_finagent as v8,
)

TICKERS = list(v7b.TICKERS)

MARKET_DIM = 1265
DOC_DIM = int(v8.DOC_DIM)
AUX_DIM = 32
WEIGHT_DIM = 5

ARMS = [
    "MARKET",
    "SEMANTIC_RAG",
    "AUX_RAG",
]


def log(*items):
    print(*items, flush=True)


def check_finite(name, arr):
    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"Nonfinite values: {name}"
        )


def normalize_dates(dates):
    return pd.DatetimeIndex(
        pd.to_datetime(
            dates,
            utc=True,
        )
    ).tz_convert(None).normalize()


def load_data(args):
    log("===== LOAD TRAIN =====")
    train = v7b.load_split(args, "train")

    log("===== LOAD VALIDATION =====")
    val = v7b.load_split(args, "val")

    for name, data in [
        ("train", train),
        ("val", val),
    ]:
        if data["market"].shape[1] != MARKET_DIM:
            raise RuntimeError(
                f"{name}: market dimension "
                f"{data['market'].shape}"
            )

        if (
            len(data["market"])
            != len(data["prices"])
            or len(data["dates"])
            != len(data["prices"])
        ):
            raise RuntimeError(
                f"{name}: inconsistent lengths"
            )

        check_finite(
            name + " market",
            data["market"],
        )

    train_dates = normalize_dates(
        train["dates"]
    )

    val_dates = normalize_dates(
        val["dates"]
    )

    if train_dates.max() >= val_dates.min():
        raise RuntimeError(
            "Train / validation overlap"
        )

    if val_dates.min().year != 2021:
        raise RuntimeError(
            "Validation year mismatch"
        )

    all_dates = train_dates.append(
        val_dates
    )

    all_prices = np.concatenate(
        [
            train["prices"],
            val["prices"],
        ],
        axis=0,
    )

    if not all_dates.is_unique:
        raise RuntimeError(
            "Duplicate global market dates"
        )

    if not all_dates.is_monotonic_increasing:
        raise RuntimeError(
            "Global dates out of order"
        )

    # Historical rolling features are computed
    # on concatenated chronological prices.
    # Every day t accesses only prices <= t.
    #
    # This fixes the 2021 January warm-up problem.
    all_market_context = v8.market_features(
        all_prices
    )

    if all_market_context.shape != (
        len(all_dates), 5, 5
    ):
        raise RuntimeError(
            "Market context shape mismatch"
        )

    n_train = len(train_dates)

    return (
        train,
        val,
        all_dates,
        all_market_context,
        n_train,
    )


def build_asof_sec_retrieval(
    dates,
    market_context,
    documents,
    document_vectors,
    vectorizer,
    top_k,
):
    """
    Retrieve at each trading date.

    All candidate documents must have:
        usable_from <= current decision date

    Existing V8 load_documents applies:
        usable_from = available_from + 1 day

    This conservative convention does not
    independently verify SEC acceptance time.

    Returns:
        pooled [n_dates, 5, 64]
        audit DataFrame
    """

    usable = documents[
        "usable_from"
    ].to_numpy(dtype="datetime64[ns]")

    tickers = documents[
        "ticker"
    ].astype(str).to_numpy()

    ids_by_ticker = {}

    for ticker in TICKERS:
        ids = np.flatnonzero(
            tickers == ticker
        )

        ids = ids[
            np.argsort(
                usable[ids],
                kind="stable",
            )
        ]

        ids_by_ticker[ticker] = ids

    # Query vectors are reused for each
    # ticker and historical market regime.
    query_cache = {}

    for ticker in TICKERS:
        for negative in [False, True]:
            regime = (
                "decline loss risk"
                if negative
                else "growth earnings revenue"
            )

            question = (
                f"{ticker} financial filing "
                f"business performance revenue "
                f"risk cash flow {regime}"
            )

            query_cache[
                (ticker, negative)
            ] = (
                vectorizer.transform(
                    [question]
                ).toarray()[0].astype(np.float32)
            )

    n = len(dates)

    pooled = np.zeros(
        (n, 5, DOC_DIM),
        dtype=np.float32,
    )

    records = []

    # All documents come from a single SEC corpus.
    # Retrieval is timestamp-filtered before ranking.
    for t, date in enumerate(dates):
        day = np.datetime64(
            pd.Timestamp(date)
        )

        for j, ticker in enumerate(TICKERS):
            ids = ids_by_ticker[ticker]

            valid_count = np.searchsorted(
                usable[ids],
                day,
                side="right",
            )

            valid_ids = ids[:valid_count]

            if not len(valid_ids):
                records.append({
                    "date": str(date.date()),
                    "ticker": ticker,
                    "n_candidates": 0,
                    "n_selected": 0,
                    "document_ids": "[]",
                    "latest_usable_from": "",
                })
                continue

            negative = bool(
                market_context[t, j, 2] < 0.0
            )

            query = query_cache[
                (ticker, negative)
            ]

            scores = (
                document_vectors[valid_ids]
                @ query
            )

            # Deterministic top-k.
            order = np.lexsort(
                (valid_ids, -scores)
            )[:top_k]

            selected = valid_ids[order]
            selected_scores = scores[order]

            # Same weighting as V8 semantic RAG.
            weights = v8.softmax_np(
                selected_scores[None, :],
                temperature=0.2,
            )[0].astype(np.float32)

            pooled[t, j] = (
                weights[:, None]
                * document_vectors[selected]
            ).sum(axis=0)

            if np.any(
                usable[selected] > day
            ):
                raise RuntimeError(
                    "SEC point-in-time violation"
                )

            selected_docs = (
                documents.iloc[selected]
            )

            records.append({
                "date": str(date.date()),
                "ticker": ticker,
                "n_candidates": len(valid_ids),
                "n_selected": len(selected),
                "document_ids": json.dumps(
                    selected_docs[
                        "evidence_id"
                    ].astype(str).tolist()
                ),
                "latest_usable_from": str(
                    selected_docs[
                        "usable_from"
                    ].max().date()
                ),
            })

        if (
            t == 0
            or (t + 1) % 100 == 0
            or t + 1 == n
        ):
            log(
                f"[SEC RETRIEVAL] {t+1}/{n}"
            )

    check_finite(
        "SEC pooled embeddings",
        pooled,
    )

    return pooled, pd.DataFrame(records)


def load_aux_encoder(
    path,
    device,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pretrained encoder: {path}"
        )

    # Phase 2A checkpoint was created by
    # this project's trusted training script.
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    model = v8.AuxiliaryPredictor()

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def extract_aux_representations(
    model,
    market_context,
    pooled_documents,
    device,
):
    """
    Reconstructs V8 auxiliary input:
        5D market context
        5D ticker identity
        64D SEC embedding

    Returns 32D representation from trunk.
    """

    n = len(market_context)

    market = np.concatenate(
        [
            market_context,
            np.broadcast_to(
                np.eye(5, dtype=np.float32),
                (n, 5, 5),
            ),
        ],
        axis=2,
    ).reshape(-1, 10)

    evidence = pooled_documents.reshape(
        -1,
        DOC_DIM,
    )

    if market.shape[0] != evidence.shape[0]:
        raise RuntimeError(
            "Auxiliary input length mismatch"
        )

    features = []

    for start in range(
        0,
        len(market),
        2048,
    ):
        end = start + 2048

        x = torch.from_numpy(
            np.concatenate(
                [
                    market[start:end],
                    evidence[start:end],
                ],
                axis=1,
            ).astype(np.float32)
        ).to(device)

        z = model.trunk(x)

        features.append(
            z.detach().cpu().numpy()
        )

    result = np.concatenate(
        features,
        axis=0,
    ).reshape(n, 5, AUX_DIM)

    result = result.astype(
        np.float32
    )

    check_finite(
        "Auxiliary representations",
        result,
    )

    return result


def create_observation_features(
    market,
    pooled,
    auxiliary,
    arm,
):
    n = len(market)

    docs_flat = pooled.reshape(
        n,
        -1,
    )

    aux_flat = auxiliary.reshape(
        n,
        -1,
    )

    # Five stocks each have 64D / 32D.
    # All arms use the exact same dimensionality.
    zero_docs = np.zeros_like(
        docs_flat
    )

    zero_aux = np.zeros_like(
        aux_flat
    )

    if arm == "MARKET":
        doc_input = zero_docs
        aux_input = zero_aux

    elif arm == "SEMANTIC_RAG":
        doc_input = docs_flat
        aux_input = zero_aux

    elif arm == "AUX_RAG":
        doc_input = zero_docs
        aux_input = aux_flat

    else:
        raise ValueError(arm)

    features = np.concatenate(
        [
            market.astype(np.float32),
            doc_input.astype(np.float32),
            aux_input.astype(np.float32),
        ],
        axis=1,
    )

    expected = (
        MARKET_DIM
        + 5 * DOC_DIM
        + 5 * AUX_DIM
    )

    if features.shape != (
        n,
        expected,
    ):
        raise RuntimeError(
            f"Wrong features: {features.shape}"
        )

    check_finite(
        f"{arm} features",
        features,
    )

    return features


def run_validation(
    model,
    data,
    arm,
    pooled,
    auxiliary,
    cost,
    zero_evidence=False,
):
    """
    Closed-loop portfolio validation.

    The trained actor stays frozen.

    Evidence ZERO intervention is used
    only for SEMANTIC_RAG and AUX_RAG.
    """

    p = (
        np.zeros_like(pooled)
        if zero_evidence
        else pooled
    )

    a = (
        np.zeros_like(auxiliary)
        if zero_evidence
        else auxiliary
    )

    obs_features = create_observation_features(
        data["market"],
        p,
        a,
        arm,
    )

    env_data = {
        "market": obs_features,
        "structured": np.zeros(
            (len(obs_features), 25),
            dtype=np.float32,
        ),
        "prices": data["prices"],
        "dates": data["dates"],
    }

    # PortfolioEnv(MARKET) uses all features
    # from env_data["market"] plus holdings.
    env = v7b.PortfolioEnv(
        env_data,
        "MARKET",
        transaction_cost=cost,
        reward_scale=100.0,
        max_days=0,
    )

    if (
        tuple(model.observation_space.shape)
        != tuple(env.observation_space.shape)
    ):
        raise RuntimeError(
            "Actor / observation mismatch"
        )

    history, metrics = v7b.evaluate(
        model,
        env,
    )

    return history, metrics


def train_actor(
    data,
    arm,
    pooled,
    auxiliary,
    seed,
    timesteps,
    learning_rate,
    cost,
    device,
):
    features = create_observation_features(
        data["market"],
        pooled,
        auxiliary,
        arm,
    )

    env_data = {
        "market": features,
        "structured": np.zeros(
            (len(features), 25),
            dtype=np.float32,
        ),
        "prices": data["prices"],
        "dates": data["dates"],
    }

    env = v7b.PortfolioEnv(
        env_data,
        "MARKET",
        transaction_cost=cost,
        reward_scale=100.0,
        max_days=0,
    )

    # All arms have the same dimensions and
    # the same neural-network architecture.
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=10,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [128, 128],
        },
        seed=seed,
        device=device,
        verbose=0,
    )

    model.learn(
        total_timesteps=timesteps
    )

    return model


def calculate_action_sensitivity(
    model,
    data,
    arm,
    pooled,
    auxiliary,
    cost,
):
    """
    Compare REAL and ZERO actor outputs under
    the SAME dates and SAME pretrade holdings.

    This isolates evidence response from
    accumulated portfolio state differences.
    """

    real_features = create_observation_features(
        data["market"],
        pooled,
        auxiliary,
        arm,
    )

    zero_features = create_observation_features(
        data["market"],
        np.zeros_like(pooled),
        np.zeros_like(auxiliary),
        arm,
    )

    real_env_data = {
        "market": real_features,
        "structured": np.zeros(
            (len(real_features), 25),
            dtype=np.float32,
        ),
        "prices": data["prices"],
        "dates": data["dates"],
    }

    env = v7b.PortfolioEnv(
        real_env_data,
        "MARKET",
        transaction_cost=cost,
        reward_scale=100.0,
    )

    obs, _ = env.reset()

    rows = []

    done = False

    while not done:
        day = env.day
        prev = env.weights.astype(
            np.float32
        )

        obs_real = np.concatenate(
            [
                real_features[day],
                prev,
            ]
        ).astype(np.float32)

        obs_zero = np.concatenate(
            [
                zero_features[day],
                prev,
            ]
        ).astype(np.float32)

        action_real, _ = model.predict(
            obs_real,
            deterministic=True,
        )

        action_zero, _ = model.predict(
            obs_zero,
            deterministic=True,
        )

        w_real = env.softmax(
            action_real
        )

        w_zero = env.softmax(
            action_zero
        )

        l1 = float(
            np.abs(
                w_real - w_zero
            ).sum()
        )

        rows.append({
            "date": str(
                env.dates[day].date()
            ),
            "weight_l1": l1,
            "changed_gt_1pct": int(
                l1 > 0.01
            ),
        })

        obs, _, done, _, _ = env.step(
            action_real
        )

    df = pd.DataFrame(rows)

    return {
        "mean_action_l1": float(
            df["weight_l1"].mean()
        ),
        "max_action_l1": float(
            df["weight_l1"].max()
        ),
        "fraction_gt_1pct": float(
            df["changed_gt_1pct"].mean()
        ),
    }, df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v8_phase2b"
        ),
    )

    parser.add_argument(
        "--phase2a",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v8_phase2a"
        ),
    )

    parser.add_argument(
        "--documents",
        default=(
            "outputs/v31/"
            "evidence_alignment.parquet"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    parser.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )

    parser.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    parser.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    parser.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    parser.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    parser.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    parser.add_argument(
        "--cost-scale",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tier",
        default="conservative",
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--algo",
        default="A2C",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.max_days != 0:
        raise RuntimeError(
            "This study requires full splits"
        )

    if args.timesteps < 1:
        raise RuntimeError(
            "Timesteps must be positive"
        )

    if not args.seeds:
        raise RuntimeError(
            "Provide at least one seed"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(2)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train, val, dates, context, n_train = (
        load_data(args)
    )

    log()
    log("===== LOAD SEC DOCUMENT CORPUS =====")

    documents = v8.load_documents(
        args.documents
    )

    vectorizer, document_vectors = (
        v8.make_document_vectors(
            documents
        )
    )

    log(
        "[DOCUMENTS]",
        len(documents)
    )

    log()
    log("===== POINT-IN-TIME SEC RETRIEVAL =====")

    pooled, audit = build_asof_sec_retrieval(
        dates,
        context,
        documents,
        document_vectors,
        vectorizer,
        args.top_k,
    )

    audit.to_csv(
        args.out / "sec_retrieval_audit.csv",
        index=False,
    )

    log(
        "[RETRIEVAL COVERAGE]",
        float(
            np.mean(
                audit["n_selected"] > 0
            )
        )
    )

    # Date slicing retains continuous warm-up.
    train_pooled = pooled[:n_train]
    val_pooled = pooled[n_train:]

    train_context = context[:n_train]
    val_context = context[n_train:]

    results = []
    sensitivity_rows = []

    for seed in args.seeds:
        log()
        log("=" * 85)
        log("SEED", seed)
        log("=" * 85)

        encoder_path = (
            args.phase2a
            / "models"
            / f"SEMANTIC_RAG_seed{seed}.pt"
        )

        encoder = load_aux_encoder(
            encoder_path,
            args.device,
        )

        aux = extract_aux_representations(
            encoder,
            context,
            pooled,
            args.device,
        )

        train_aux = aux[:n_train]
        val_aux = aux[n_train:]

        if (
            len(train_aux) != len(train["market"])
            or len(val_aux) != len(val["market"])
        ):
            raise RuntimeError(
                "Train / val representation mismatch"
            )

        # Note: 2021 auxiliary representations
        # use only current and historical market
        # features + as-of documents.
        #
        # No future realized returns are inputs.

        for arm in ARMS:
            log()
            log(
                "[TRAIN A2C]",
                arm,
                "seed",
                seed
            )

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            actor = train_actor(
                train,
                arm,
                train_pooled,
                train_aux,
                seed,
                args.timesteps,
                args.learning_rate,
                args.transaction_cost,
                args.device,
            )

            run_dir = (
                args.out
                / f"{arm}_seed{seed}"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            actor.save(
                str(run_dir / "model.zip")
            )

            log("[EVALUATE] REAL")

            history, metrics = run_validation(
                actor,
                val,
                arm,
                val_pooled,
                val_aux,
                args.transaction_cost,
                zero_evidence=False,
            )

            history.to_csv(
                run_dir / "validation_REAL.csv",
                index=False,
            )

            row = {
                "seed": seed,
                "arm": arm,
                "mode": "REAL",
                **metrics,
            }

            results.append(row)

            if arm != "MARKET":
                log("[EVALUATE] ZERO EVIDENCE")

                zero_history, zero_metrics = (
                    run_validation(
                        actor,
                        val,
                        arm,
                        val_pooled,
                        val_aux,
                        args.transaction_cost,
                        zero_evidence=True,
                    )
                )

                zero_history.to_csv(
                    run_dir / "validation_ZERO.csv",
                    index=False,
                )

                results.append({
                    "seed": seed,
                    "arm": arm,
                    "mode": "ZERO",
                    **zero_metrics,
                })

                reliance, daily = (
                    calculate_action_sensitivity(
                        actor,
                        val,
                        arm,
                        val_pooled,
                        val_aux,
                        args.transaction_cost,
                    )
                )

                daily.to_csv(
                    run_dir / "action_sensitivity.csv",
                    index=False,
                )

                sensitivity_rows.append({
                    "seed": seed,
                    "arm": arm,
                    **reliance,
                    "real_minus_zero_return_pp": (
                        metrics["total_return"]
                        - zero_metrics["total_return"]
                    ) * 100.0,
                })

            log("[VALIDATION]", row)

            pd.DataFrame(
                results
            ).to_csv(
                args.out / "partial_results.csv",
                index=False,
            )

    df = pd.DataFrame(results)

    df.to_csv(
        args.out / "validation_results.csv",
        index=False,
    )

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    )

    sensitivity_df.to_csv(
        args.out / "evidence_reliance.csv",
        index=False,
    )

    real = df[
        df["mode"] == "REAL"
    ].copy()

    wide = real.pivot(
        index="seed",
        columns="arm",
        values="total_return",
    )

    for arm in [
        "SEMANTIC_RAG",
        "AUX_RAG",
    ]:
        wide[
            f"{arm}_minus_MARKET_pp"
        ] = (
            wide[arm] - wide["MARKET"]
        ) * 100.0

    wide.to_csv(
        args.out / "paired_seed_returns.csv"
    )

    log()
    log("=" * 100)
    log("V8 PHASE 2B VALIDATION")
    log("=" * 100)
    log(df.to_string(index=False))

    log()
    log("===== PAIRED RETURNS =====")
    log(wide.to_string())

    log()
    log("===== EVIDENCE RELIANCE =====")
    log(sensitivity_df.to_string(index=False))

    manifest = {
        "train_period": "2018-2020",
        "validation_period": "2021",
        "test_period_used": False,
        "market_dim": MARKET_DIM,
        "per_company_doc_dim": DOC_DIM,
        "per_company_aux_dim": AUX_DIM,
        "observation_dim": (
            MARKET_DIM
            + 5 * DOC_DIM
            + 5 * AUX_DIM
            + WEIGHT_DIM
        ),
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "top_k": args.top_k,
        "transaction_cost": (
            args.transaction_cost
        ),
        "note": (
            "Frozen pretrained auxiliary encoder; "
            "A2C actor retrained for each arm; "
            "point-in-time semantic SEC retrieval."
        ),
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("V8 PHASE 2B: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
