#!/usr/bin/env python3
"""
V42: Trainable Temporal Financial KG -> FinRL-X

Main path:
  V35 SEC silver events
  -> chronological online link-prediction training
  -> relation-aware temporal graph embeddings
  -> V21 FinRL-X input
  -> A2C / PPO / DDPG

Also exports Financial-TKG's 5-column event format.

This is a trainable temporal relational embedding baseline,
not the original FinDKG KGTransformer architecture.

Commands:
  prepare
  train
  check
  finrl
  report
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v42"
V20 = ROOT / "outputs/v20/fusion"
V35 = ROOT / "outputs/v35"
RUNNER = ROOT / "scripts/run_v21_dynamic_ablation.py"

sys.path.insert(0, str(ROOT / "scripts"))
import v36_finrl_kg_ablation as v36

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

ISSUERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

DIM = 11
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")


def fail(message):
    raise RuntimeError("V42 STOP: " + str(message))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data, indent=2, ensure_ascii=False,
            default=str
        ),
        encoding="utf-8",
    )


def market(tier):
    path = V20 / f"dataset_{tier}.csv"

    if not path.exists():
        fail("Missing V20 market: " + str(path))

    original = pd.read_csv(path, low_memory=False)

    date_col, ticker_col = v36.identify_keys(
        original
    )

    panel = original.copy()

    panel["_date"] = pd.to_datetime(
        panel[date_col], errors="raise"
    ).dt.normalize()

    panel["_tic"] = (
        panel[ticker_col].astype(str).str.upper()
    )

    if set(panel["_tic"]) != set(TICKERS):
        fail("Wrong trading universe")

    if panel.duplicated(
        ["_date", "_tic"]
    ).any():
        fail("Duplicate market ticker-day")

    if not panel.groupby(
        "_date"
    )["_tic"].nunique().eq(5).all():
        fail("Incomplete five-stock market panel")

    return original, panel


def load_events():
    path = V35 / "evidence_ledger.csv"

    if not path.exists():
        fail("Missing historical SEC ledger")

    df = pd.read_csv(
        path, dtype={"accession": str},
        low_memory=False
    )

    # V35 uses ticker to identify the SEC reporting issuer.
    # Some V35 ledger versions do not contain an issuer column.
    # Derive it from the fixed, audited five-company mapping.
    if "ticker" not in df.columns:
        fail("V35 ledger missing ticker")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    unknown_tickers = set(df["ticker"]) - set(ISSUERS)

    if unknown_tickers:
        fail(f"Unknown SEC issuer tickers: {unknown_tickers}")

    expected_issuer = df["ticker"].map(ISSUERS)

    if "issuer" not in df.columns:
        df["issuer"] = expected_issuer
        print(
            "V35 schema compatibility: derived issuer from ticker",
            flush=True,
        )
    else:
        existing = df["issuer"].fillna("").astype(str).str.strip()
        missing = existing.eq("")

        df.loc[missing, "issuer"] = expected_issuer.loc[missing]

        mismatched = df["issuer"] != expected_issuer

        if mismatched.any():
            fail(
                "V35 issuer conflicts with ticker: "
                + str(
                    df.loc[
                        mismatched,
                        ["ticker", "issuer"]
                    ].head(10).to_dict("records")
                )
            )

    required = {
        "ticker", "issuer", "subject",
        "relation", "object",
        "available_from", "evidence_id",
    }

    if not required.issubset(df.columns):
        fail(
            "V35 ledger missing: "
            + str(required - set(df.columns))
        )

    if df.empty:
        fail("No SEC evidence events")

    if not set(df["relation"]).issubset(RELATIONS):
        fail("Unknown relation type")

    df["available_from"] = pd.to_datetime(
        df["available_from"], errors="raise"
    ).dt.normalize()

    if "filing_date" in df:
        dates = pd.to_datetime(
            df["filing_date"], errors="raise"
        ).dt.normalize()

        if not (df["available_from"] > dates).all():
            fail("Disclosure time does not follow filing date")

    for row in df.itertuples(index=False):
        if row.ticker not in ISSUERS:
            fail("Issuer outside five-stock universe")

        if row.issuer != ISSUERS[row.ticker]:
            fail("Issuer identity mismatch")

        if row.subject == row.object:
            fail("Self-edge")

    df = df.drop_duplicates(
        ["evidence_id", "subject", "relation", "object"]
    ).sort_values(
        ["available_from", "evidence_id"]
    ).reset_index(drop=True)

    return df


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)

    raw = load_events()

    entities = sorted(
        set(raw["subject"])
        | set(raw["object"])
        | set(ISSUERS.values())
    )

    ent2id = {
        name: i for i, name in enumerate(entities)
    }

    rel2id = {
        name: i for i, name in enumerate(RELATIONS)
    }

    events = raw.copy()
    events["head"] = events["subject"].map(ent2id)
    events["rel"] = events["relation"].map(rel2id)
    events["tail"] = events["object"].map(ent2id)

    events["time"] = (
        events["available_from"]
        - pd.Timestamp("1970-01-01")
    ).dt.days.astype(int)

    if events[
        ["head", "rel", "tail", "time"]
    ].isna().any().any():
        fail("Invalid graph ID mapping")

    output = OUT / "original_format"
    output.mkdir(parents=True, exist_ok=True)

    # Exactly five columns as in Financial-TKG's loader.
    for split, mask in {
        "train": events["available_from"] < "2022-01-01",
        "valid": events["available_from"].between(
            "2022-01-01", "2022-12-31"
        ),
        "test": events["available_from"].between(
            "2023-01-01", "2023-12-31"
        ),
    }.items():
        sub = events.loc[
            mask, ["head", "rel", "tail", "time"]
        ].copy()

        sub["_"] = 1

        sub.to_csv(
            output / f"{split}.txt",
            sep="\t", index=False, header=False
        )

        print(
            split,
            "events:", len(sub),
            "unique times:", sub["time"].nunique(),
        )

    with (output / "stat.txt").open("w") as f:
        f.write(
            f"{len(entities)}\t{len(RELATIONS)}\t0\n"
        )

    pd.DataFrame([
        {
            "entity": name,
            "entity_id": idx,
            "ntype": "company",
            "ntype_id": 0,
        }
        for name, idx in ent2id.items()
    ]).to_csv(
        output / "entity2id.txt",
        sep="\t", index=False, header=False
    )

    pd.DataFrame([
        {
            "entity": name,
            "entity_id": idx,
        }
        for name, idx in ent2id.items()
    ]).to_csv(
        OUT / "entities.csv", index=False
    )

    pd.DataFrame([
        {
            "relation": name,
            "relation_id": idx,
        }
        for name, idx in rel2id.items()
    ]).to_csv(
        OUT / "relations.csv", index=False
    )

    events.to_csv(
        OUT / "events.csv", index=False
    )

    count_by_split = {
        split: int(mask.sum())
        for split, mask in {
            "train": events["available_from"] < "2022-01-01",
            "valid": events["available_from"].between(
                "2022-01-01", "2022-12-31"
            ),
            "test": events["available_from"].between(
                "2023-01-01", "2023-12-31"
            ),
        }.items()
    }

    write_json(
        OUT / "data_manifest.json",
        {
            "source": str(V35 / "evidence_ledger.csv"),
            "source_sha256": sha(
                V35 / "evidence_ledger.csv"
            ),
            "entities": len(entities),
            "relation_vocabulary": RELATIONS,
            "observed_relations": sorted(
                events["relation"].unique().tolist()
            ),
            "event_mentions": len(events),
            "unique_edges": len(
                events[
                    ["head", "rel", "tail"]
                ].drop_duplicates()
            ),
            "split_counts": count_by_split,
            "human_verified": False,
            "strict_pit_approved": False,
        }
    )

    print("\n=== V42 PREPARE PASS ===")
    print("Entities:", len(entities))
    print("Events:", len(events))
    print("Historical format:", output)


class TemporalRelationalModel(torch.nn.Module):
    """
    Trainable relational node embeddings with
    a link-prediction objective.

    Historical directed adjacency is separately used
    for relation-aware propagation at each timestamp.
    """

    def __init__(self, entities, relations, dim=11):
        super().__init__()

        self.entity = torch.nn.Embedding(
            entities, dim
        )

        self.relation = torch.nn.Embedding(
            relations, dim
        )

        torch.nn.init.normal_(
            self.entity.weight, std=0.1
        )

        torch.nn.init.normal_(
            self.relation.weight, std=0.1
        )

    def score(self, head, relation, tail):
        h = self.entity(head)
        r = self.relation(relation)
        t = self.entity(tail)

        return (h * r * t).sum(-1)


def train(args):
    manifest_path = OUT / "data_manifest.json"

    if not manifest_path.exists():
        fail("Run prepare first")

    data_manifest = json.loads(
        manifest_path.read_text()
    )

    if sha(V35 / "evidence_ledger.csv") != (
        data_manifest["source_sha256"]
    ):
        fail("V35 source changed after prepare")

    events = pd.read_csv(
        OUT / "events.csv"
    )

    entities = pd.read_csv(
        OUT / "entities.csv"
    )

    relations = pd.read_csv(
        OUT / "relations.csv"
    )

    n = len(entities)
    n_rel = len(relations)

    if n < 3:
        fail("Too few graph entities")

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = TemporalRelationalModel(
        n, n_rel, DIM
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Historical cumulative adjacency, separate by relation.
    graphs = np.zeros(
        (n_rel, n, n), dtype=np.float32
    )

    # One ticker-date observation per market trading day.
    original, panel = market("conservative")
    dates = sorted(panel["_date"].unique())

    events["available_from"] = pd.to_datetime(
        events["available_from"]
    ).dt.normalize()

    grouped = events.groupby(
        "available_from", sort=True
    )

    batches = [
        (date, group.copy())
        for date, group in grouped
    ]

    pos = 0
    daily = []
    training_log = []

    ticker_to_id = {
        row["entity"]: int(row["entity_id"])
        for row in entities.to_dict("records")
    }

    def process_event_batch(day, batch):
        nonlocal graphs

        # Train once when each event becomes public.
        # No fitting on 2022 or 2023 events.
        if day < TRAIN_CUTOFF:
            model.train()

            head = torch.tensor(
                batch["head"].to_numpy(dtype=int),
                dtype=torch.long,
            )

            rel = torch.tensor(
                batch["rel"].to_numpy(dtype=int),
                dtype=torch.long,
            )

            tail = torch.tensor(
                batch["tail"].to_numpy(dtype=int),
                dtype=torch.long,
            )

            negative = rng.integers(
                0, n, size=len(batch)
            )

            # Never use the observed tail as its own negative.
            true_tails = tail.numpy()
            negative = np.where(
                negative == true_tails,
                (negative + 1) % n,
                negative,
            )

            negative = torch.tensor(
                negative, dtype=torch.long
            )

            for step in range(args.event_steps):
                optimizer.zero_grad()

                positive_score = model.score(
                    head, rel, tail
                )

                negative_score = model.score(
                    head, rel, negative
                )

                loss = (
                    F.softplus(-positive_score).mean()
                    + F.softplus(negative_score).mean()
                )

                loss.backward()
                optimizer.step()

            training_log.append({
                "available_from": str(day.date()),
                "events": len(batch),
                "loss": float(loss.item()),
            })

        # In all periods, incorporate newly disclosed
        # events into the current historical graph.
        for row in batch.itertuples(index=False):
            graphs[
                int(row.rel),
                int(row.head),
                int(row.tail),
            ] = 1.0

    for date in dates:
        date = pd.Timestamp(date).normalize()

        # STRICT: only disclosures before trading date.
        while (
            pos < len(batches)
            and batches[pos][0] < date
        ):
            event_date, batch = batches[pos]
            process_event_batch(
                pd.Timestamp(event_date),
                batch,
            )
            pos += 1

        with torch.no_grad():
            base = model.entity.weight.detach().numpy()
            relation = (
                model.relation.weight.detach().numpy()
            )

        # Learned embeddings + historical relation-aware
        # message passing. Both depend on real graph events.
        propagated = np.zeros_like(base)

        for rid in range(n_rel):
            matrix = graphs[rid]

            degree = matrix.sum(axis=1, keepdims=True)

            normalized = np.divide(
                matrix, degree,
                out=np.zeros_like(matrix),
                where=degree > 0,
            )

            # Relation-specific transformation.
            propagated += (
                normalized @ base
            ) * np.tanh(relation[rid])

        graph_activity = (
            graphs.sum(axis=(0, 2))
            + graphs.sum(axis=(0, 1))
        )

        representation = np.tanh(
            base + propagated
        )

        # Companies with no historical graph interaction
        # must not receive random KG features.
        representation[
            graph_activity == 0
        ] = 0.0

        for ticker in TICKERS:
            idx = ticker_to_id[ISSUERS[ticker]]

            row = {
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
            }

            for k in range(DIM):
                row[f"dkg_emb_{k:02d}"] = float(
                    representation[idx, k]
                )

            daily.append(row)

    feature_columns = [
        f"dkg_emb_{k:02d}"
        for k in range(DIM)
    ]

    result = pd.DataFrame(daily)

    if len(result) != len(original):
        fail("Daily graph/market panel mismatch")

    if result.duplicated(["date", "tic"]).any():
        fail("Duplicate embedding ticker-date")

    if not np.isfinite(
        result[feature_columns].to_numpy()
    ).all():
        fail("Embedding contains NaN/Inf")

    result.to_csv(
        OUT / "daily_embeddings.csv",
        index=False
    )

    pd.DataFrame(training_log).to_csv(
        OUT / "online_training_log.csv",
        index=False
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "entities": entities.to_dict("records"),
            "relations": relations.to_dict("records"),
            "dim": DIM,
            "seed": args.seed,
            "training_cutoff": str(TRAIN_CUTOFF.date()),
        },
        OUT / "trained_temporal_model.pt"
    )

    # Input is exactly the V21 11 dynamic feature slots.
    config = json.loads(
        (V20 / "feature_sets.json").read_text()
    )

    slots = v36.discover_dynamic_slots(
        config,
        set(original.columns),
        "",
    )

    if len(slots) != DIM:
        fail("V21 dynamic-slot mismatch")

    # Explicitly separate primary and placebo.
    datasets = {
        "TRAINED_DKG": result,
    }

    placebo = result.copy()

    # Controls use shared date permutations within splits.
    # This is a negative control, not deployable PIT data.
    for start, end in [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
    ]:
        scoped_dates = sorted(
            d for d in result["date"].unique()
            if start <= d <= end
        )

        perm = rng.permutation(len(scoped_dates))

        for ticker in TICKERS:
            source = (
                result[result["tic"] == ticker]
                .set_index("date")
                .loc[scoped_dates, feature_columns]
                .to_numpy()
            )

            values = source[perm]

            mask = (
                placebo["tic"].eq(ticker)
                & placebo["date"].isin(scoped_dates)
            )

            positions = (
                placebo.loc[mask]
                .sort_values("date").index
            )

            placebo.loc[
                positions, feature_columns
            ] = values

    datasets["TRAINED_DKG_PLACEBO"] = placebo

    for name, features in datasets.items():
        target = OUT / "inputs" / name
        target.mkdir(
            parents=True, exist_ok=True
        )

        signal = features.copy()
        signal["_date"] = pd.to_datetime(
            signal["date"]
        ).dt.normalize()
        signal["_tic"] = signal["tic"]

        for tier in ["conservative", "inclusive"]:
            market_original, market_panel = market(tier)

            joined = market_panel.merge(
                signal[
                    ["_date", "_tic"]
                    + feature_columns
                ],
                on=["_date", "_tic"],
                how="left",
                validate="one_to_one",
            )

            if len(joined) != len(market_original):
                fail("FinRL merge changed row count")

            if joined[
                feature_columns
            ].isna().any().any():
                fail("Missing FinRL graph feature")

            for slot, feature in zip(
                slots, feature_columns
            ):
                joined[slot] = joined[
                    feature
                ].astype(float)

            result_panel = joined[
                list(market_original.columns)
            ].copy()

            untouched = [
                c for c in market_original.columns
                if c not in slots
            ]

            if not market_original[untouched].equals(
                result_panel[untouched]
            ):
                fail("Original market/SEC fields changed")

            result_panel.to_csv(
                target / f"dataset_{tier}.csv",
                index=False,
            )

        shutil.copy2(
            V20 / "feature_sets.json",
            target / "feature_sets.json",
        )

        print(
            "FINRL INPUT PREPARED:", name,
            flush=True,
        )

    write_json(
        OUT / "training_manifest.json",
        {
            "model": "OnlineTemporalRelationalEmbedding",
            "original_FinDKG_KGTransformer": False,
            "torch_model_trained": True,
            "trainable_entity_embedding": True,
            "trainable_relation_embedding": True,
            "link_prediction_loss": (
                "softplus positive + negative"
            ),
            "training_updates": len(training_log),
            "online_train_until": "2021-12-31",
            "validation_model_updates": False,
            "test_model_updates": False,
            "graph_updates_after_cutoff": True,
            "graph_feature_columns": feature_columns,
            "finrl_dynamic_slots": slots,
            "event_steps": args.event_steps,
            "strict_pit_approved": False,
            "human_verified": False,
            "note": (
                "Features depend on prior known SEC events. "
                "SEC acceptance timestamps and execution "
                "timing still need independent audit."
            ),
        }
    )

    print("\n=== V42 TRAINED DKG PASS ===")
    print("Graph training updates:", len(training_log))
    print("Daily rows:", len(result))
    print("Embedding dimension:", DIM)
    print("Five trading tickers: PASS")
    print("FinRL input construction: PASS")


def launch(args, check_only=False):
    if not (
        OUT / "training_manifest.json"
    ).exists():
        fail("Run train first")

    for name in [
        "TRAINED_DKG",
        "TRAINED_DKG_PLACEBO",
    ]:
        command = [
            sys.executable,
            str(RUNNER),
            "--input-dir",
            str(OUT / "inputs" / name),
            "--tiers",
            "conservative",
            "inclusive",
            "--agents",
            "A2C",
            "PPO",
            "DDPG",
            "--seeds",
            args.seeds,
            "--timesteps",
            str(args.timesteps),
            "--output-dir",
            str(OUT / "runs" / name),
        ]

        if check_only:
            command.append("--check-only")

        print("\nRUN:", " ".join(command), flush=True)

        if args.dry_run:
            continue

        log_dir = OUT / "logs"
        log_dir.mkdir(
            parents=True, exist_ok=True
        )

        mode = "check" if check_only else "finrl"
        log = log_dir / f"{mode}_{name}.log"

        with log.open("w") as f:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        print(name, "exit:", proc.returncode)
        print("Log:", log, flush=True)

        if proc.returncode != 0:
            print(
                "\n".join(
                    log.read_text(
                        errors="replace"
                    ).splitlines()[-70:]
                )
            )
            fail("FinRL runner failed: " + name)

    print(
        "\n=== V42 FINRL CHECK PASS ==="
        if check_only
        else "\n=== V42 FINRL RUN COMPLETE ==="
    )


def report():
    for name in [
        "TRAINED_DKG",
        "TRAINED_DKG_PLACEBO",
    ]:
        path = OUT / "runs" / name / "summary.csv"

        if not path.exists():
            print(name, "not yet trained")
            continue

        df = pd.read_csv(path)

        df = df[
            df["tier"].eq("conservative")
            & df["split"].eq("test")
            & df["condition"].isin(
                ["CLEAN", "FULL_DYNAMIC"]
            )
        ].copy()

        print("\n=== ", name, " ===")
        print(
            df[[
                "agent",
                "condition",
                "total_return_mean",
                "sharpe_mean",
                "max_drawdown_mean",
            ]].round(5).to_string(index=False)
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "prepare", "train",
            "check", "finrl", "report"
        ],
    )

    p.add_argument(
        "--seed", type=int, default=42
    )

    p.add_argument(
        "--event-steps", type=int, default=20
    )

    p.add_argument(
        "--lr", type=float, default=0.01
    )

    p.add_argument(
        "--seeds", default="0"
    )

    p.add_argument(
        "--timesteps", type=int, default=2000
    )

    p.add_argument(
        "--dry-run", action="store_true"
    )

    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(args)
    elif args.command == "check":
        launch(args, check_only=True)
    elif args.command == "finrl":
        launch(args, check_only=False)
    else:
        report()


if __name__ == "__main__":
    main()
