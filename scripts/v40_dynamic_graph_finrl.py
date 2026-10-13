#!/usr/bin/env python3
"""
V40: Dynamic Financial KG -> FinRL-X

Methods borrowed conceptually from FinDKG:
  - timestamped (head, relation, tail, time) events
  - historical graph snapshots
  - directed relation-aware aggregation
  - chronological train/validation/test data

This implementation:
  - uses existing V35 SEC silver evidence
  - implements deterministic temporal graph features
  - reuses existing V21 FinRL runner
  - needs pandas/numpy, no new DGL dependency

NOT the original trained FinDKG KGTransformer.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v40"
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

FEATURES = [
    "dkg_out_supplies",
    "dkg_in_supplies",
    "dkg_out_customer",
    "dkg_in_customer",
    "dkg_out_competes",
    "dkg_in_competes",
    "dkg_out_sector",
    "dkg_in_sector",
    "dkg_neighbors_2hop",
    "dkg_recent_activity",
    "dkg_evidence_coverage",
]


def fail(message):
    raise RuntimeError("V40 STOP: " + str(message))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def read_market(tier):
    path = V20 / f"dataset_{tier}.csv"
    if not path.exists():
        fail(f"Missing market file {path}")

    raw = pd.read_csv(path)
    date_col, ticker_col = v36.identify_keys(raw)

    df = raw.copy()
    df["_date"] = pd.to_datetime(
        df[date_col], errors="raise"
    ).dt.normalize()
    df["_tic"] = (
        df[ticker_col].astype(str).str.upper()
    )

    if df.duplicated(["_date", "_tic"]).any():
        fail("Duplicate market ticker-date")

    if set(df["_tic"]) != set(TICKERS):
        fail("Wrong tradeable ticker universe")

    if not df.groupby("_date")["_tic"].nunique().eq(5).all():
        fail("Market panel missing stocks")

    return raw, df


def load_events():
    path = V35 / "evidence_ledger.csv"

    if not path.exists():
        fail("Missing V35 SEC evidence ledger")

    raw = pd.read_csv(path, low_memory=False)

    required = [
        "ticker", "issuer", "subject", "relation",
        "object", "available_from",
        "evidence_id",
    ]

    missing = set(required) - set(raw.columns)

    if missing:
        fail(f"Missing ledger columns: {missing}")

    raw = raw.copy()

    raw["available_from"] = pd.to_datetime(
        raw["available_from"],
        errors="raise",
    ).dt.normalize()

    if "filing_date" in raw.columns:
        filing = pd.to_datetime(
            raw["filing_date"],
            errors="raise",
        ).dt.normalize()

        if not (raw["available_from"] > filing).all():
            fail("SEC availability date <= filing date")

    if set(raw["ticker"]) - set(TICKERS):
        fail("SEC ticker outside portfolio issuer universe")

    if not set(raw["relation"]).issubset(RELATIONS):
        fail("Unknown KG relation")

    for r in raw.itertuples(index=False):
        if r.issuer != ISSUERS[r.ticker]:
            fail("Issuer identity mismatch")

        if r.subject == r.object:
            fail("SEC graph self-edge")

    if raw.empty:
        fail("No verified silver events available")

    if raw["evidence_id"].isna().any():
        fail("Missing SEC evidence identifiers")

    raw = raw.drop_duplicates(
        ["evidence_id", "subject", "relation", "object"]
    ).sort_values(
        ["available_from", "evidence_id"]
    ).reset_index(drop=True)

    return raw


def create_events(raw):
    entities = sorted(
        set(raw["subject"])
        | set(raw["object"])
        | set(ISSUERS.values())
    )

    entity_id = {
        entity: i for i, entity in enumerate(entities)
    }

    relation_id = {
        relation: i for i, relation in enumerate(RELATIONS)
    }

    events = raw.copy()

    events["head"] = events["subject"].map(entity_id)
    events["rel"] = events["relation"].map(relation_id)
    events["tail"] = events["object"].map(entity_id)

    # Days since Unix epoch.
    events["time"] = (
        events["available_from"]
        - pd.Timestamp("1970-01-01")
    ).dt.days.astype(int)

    if events[["head", "rel", "tail", "time"]].isna().any().any():
        fail("Invalid temporal KG IDs")

    output = OUT / "findkg_format"
    output.mkdir(parents=True, exist_ok=True)

    entity_table = pd.DataFrame({
        "entity": entities,
        "entity_id": list(range(len(entities))),
        "ntype": ["company"] * len(entities),
        "ntype_id": [0] * len(entities),
    })

    relation_table = pd.DataFrame({
        "relation": list(relation_id),
        "relation_id": list(relation_id.values()),
    })

    entity_table.to_csv(
        output / "entity2id.txt",
        sep="\t", index=False, header=False,
    )

    relation_table.to_csv(
        output / "relation2id.txt",
        sep="\t", index=False, header=False,
    )

    events.to_csv(
        OUT / "temporal_events.csv",
        index=False,
    )

    # FinDKG-style five-column event tables.
    # This is an interchange format, not automatic
    # drop-in compatibility with every FinDKG setting.
    for name, mask in {
        "train": events["available_from"] <
                 pd.Timestamp("2022-01-01"),
        "valid": events["available_from"].between(
            "2022-01-01", "2022-12-31"
        ),
        "test": events["available_from"].between(
            "2023-01-01", "2023-12-31"
        ),
    }.items():
        part = events.loc[
            mask, ["head", "rel", "tail", "time"]
        ].copy()

        part["label"] = 1

        part.to_csv(
            output / f"{name}.txt",
            sep="\t", header=False, index=False,
        )

    with (output / "stat.txt").open("w") as f:
        f.write(
            f"{len(entities)}\t{len(RELATIONS)}\t0\n"
        )

    return events, entity_id, relation_id


def make_graph_features(events, entity_id, market_dates):
    n = len(entity_id)
    r_count = len(RELATIONS)

    dates = sorted(pd.to_datetime(market_dates).unique())

    edges = events.sort_values(
        "available_from"
    ).to_dict("records")

    # Known directed edges, no fabricated reverse relation.
    adjacency = [
        np.zeros((n, n), dtype=np.float64)
        for _ in range(r_count)
    ]

    first_seen = set()
    recent_event_days = []
    seen_issuers = set()

    pointer = 0
    output = []

    for date in dates:
        date = pd.Timestamp(date).normalize()

        # Strictly earlier disclosure date.
        while (
            pointer < len(edges)
            and pd.Timestamp(
                edges[pointer]["available_from"]
            ) < date
        ):
            e = edges[pointer]

            h = int(e["head"])
            t = int(e["tail"])
            relation = int(e["rel"])

            edge_key = (h, relation, t)

            if edge_key not in first_seen:
                first_seen.add(edge_key)

            # Weight remains fixed for distinct known edges.
            adjacency[relation][h, t] = 1.0

            recent_event_days.append(
                pd.Timestamp(
                    e["available_from"]
                ).normalize()
            )

            seen_issuers.add(
                str(e["issuer"])
            )

            pointer += 1

        # Distinct outgoing and incoming typed degree.
        outgoing = np.column_stack([
            graph.sum(axis=1)
            for graph in adjacency
        ])

        incoming = np.column_stack([
            graph.sum(axis=0)
            for graph in adjacency
        ])

        # Relation-aware message passing:
        # Each relation contributes independently.
        #
        # This uses historical graph structure only.
        # It is NOT a trainable KGTransformer.
        neighbor = np.zeros(n)

        combined = sum(adjacency)
        combined = (combined > 0).astype(float)

        # One-hop activity signal.
        degree = (
            combined.sum(axis=0)
            + combined.sum(axis=1)
        )

        # Two-hop aggregation via directed neighbors.
        # Avoid adding future edges, since adjacency was
        # updated exclusively from older SEC events.
        row_sum = combined.sum(axis=1, keepdims=True)
        transition = np.divide(
            combined,
            row_sum,
            out=np.zeros_like(combined),
            where=row_sum > 0,
        )

        neighbor = transition @ np.log1p(degree)

        recent_count = sum(
            0 <= (date - d).days <= 90
            for d in recent_event_days
        )

        for ticker in TICKERS:
            issuer = ISSUERS[ticker]
            index = entity_id[issuer]

            values = []

            # [SUPPLIES, CUSTOMER, COMPETES, SECTOR]
            for rel in range(r_count):
                values.append(outgoing[index, rel])
                values.append(incoming[index, rel])

            values.extend([
                float(neighbor[index]),
                float(recent_count),
                float(issuer in seen_issuers),
            ])

            if len(values) != len(FEATURES):
                fail("Incorrect dynamic graph vector size")

            record = {
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
            }

            record.update(dict(zip(FEATURES, values)))

            output.append(record)

    result = pd.DataFrame(output)

    if result.duplicated(["date", "tic"]).any():
        fail("Duplicate graph ticker-date")

    if not np.isfinite(
        result[FEATURES].to_numpy(dtype=float)
    ).all():
        fail("Non-finite dynamic graph features")

    return result


def normalize_graph_features(df):
    result = df.copy()

    result["_date"] = pd.to_datetime(
        result["date"]
    )

    train = result["_date"].between(
        "2018-01-01", "2021-12-31"
    )

    if not train.any():
        fail("No training interval for normalization")

    transformed = np.log1p(result[FEATURES])

    train_values = transformed.loc[train]
    mean = train_values.mean()
    std = train_values.std(ddof=0).replace(0, 1)

    norm = (
        (transformed - mean) / std
    ).clip(-5, 5)

    # Preserve zero when no relation evidence exists.
    norm = norm.where(
        result[FEATURES].ne(0),
        0,
    )

    result[FEATURES] = norm
    result = result.drop(columns="_date")

    save_json(
        OUT / "normalization.json",
        {
            "fit_period": "2018-2021",
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "clip": 5,
        },
    )

    return result


def make_placebo(df, seed):
    rng = np.random.default_rng(seed)

    # Preserve all five stocks together.
    # Shuffle within train / validation / test separately.
    # Negative control only; not PIT-deployable.
    pivot = {}

    for ticker in TICKERS:
        x = df[df["tic"] == ticker]
        pivot[ticker] = x.set_index("date")[FEATURES]

    dates = sorted(df["date"].unique())

    shuffled = df.copy().set_index(["date", "tic"])

    for start, end in [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
    ]:
        scope = [
            d for d in dates if start <= d <= end
        ]

        if not scope:
            continue

        perm = rng.permutation(len(scope))

        for ticker in TICKERS:
            old_values = (
                pivot[ticker]
                .loc[scope, FEATURES]
                .to_numpy()
            )

            new_values = old_values[perm]

            shuffled.loc[
                pd.IndexSlice[scope, ticker],
                FEATURES,
            ] = new_values

    return shuffled.reset_index()


def dynamic_slots():
    config_path = V20 / "feature_sets.json"

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    original, _ = read_market("conservative")

    return v36.discover_dynamic_slots(
        config,
        set(original.columns),
        "",
    )


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)

    raw_events = load_events()

    events, entity_id, relation_id = create_events(
        raw_events
    )

    conservative, panel = read_market("conservative")
    dates = panel["_date"].unique()

    graph = make_graph_features(
        events, entity_id, dates
    )

    if len(graph) != len(conservative):
        fail("Graph and market row counts differ")

    graph.to_csv(
        OUT / "daily_graph_raw.csv",
        index=False,
    )

    normalized = normalize_graph_features(graph)

    normalized.to_csv(
        OUT / "daily_graph_features.csv",
        index=False,
    )

    placebo = make_placebo(
        normalized, args.seed
    )

    placebo.to_csv(
        OUT / "daily_graph_placebo.csv",
        index=False,
    )

    slots = dynamic_slots()

    if len(slots) != 11:
        fail("V21 requires exactly 11 dynamic slots")

    for name, signal in {
        "DYNAMIC_GRAPH": normalized,
        "GRAPH_PLACEBO": placebo,
    }.items():

        target = OUT / "inputs" / name
        target.mkdir(parents=True, exist_ok=True)

        signal = signal.copy()
        signal["_date"] = pd.to_datetime(
            signal["date"]
        ).dt.normalize()
        signal["_tic"] = signal["tic"]

        for tier in ["conservative", "inclusive"]:
            original, market = read_market(tier)

            merged = market.merge(
                signal[
                    ["_date", "_tic"] + FEATURES
                ],
                on=["_date", "_tic"],
                how="left",
                validate="one_to_one",
            )

            if len(merged) != len(original):
                fail("Graph merge changed market rows")

            if merged[FEATURES].isna().any().any():
                fail("Missing graph signals")

            for slot, feature in zip(slots, FEATURES):
                merged[slot] = merged[feature].astype(float)

            # Exact original schema and order.
            result = merged[list(original.columns)]

            if not result[
                slots
            ].apply(pd.to_numeric).notna().all().all():
                fail("Missing V21 dynamic slots")

            result.to_csv(
                target / f"dataset_{tier}.csv",
                index=False,
            )

        import shutil

        shutil.copy2(
            V20 / "feature_sets.json",
            target / "feature_sets.json",
        )

        print("PREPARED:", name, flush=True)

    counts = events.groupby(
        "relation"
    )[["head"]].count()["head"].to_dict()

    nonzero = graph[
        FEATURES
    ].ne(0).mean().to_dict()

    manifest = {
        "source": "V35 SEC silver evidence",
        "method": (
            "FinDKG-style temporal events "
            "and deterministic relation-aware "
            "historical graph aggregation"
        ),
        "trainable_kgtransformer": False,
        "entities": len(entity_id),
        "event_mentions": len(events),
        "distinct_edges": len(set(
            zip(
                events["head"],
                events["rel"],
                events["tail"],
            )
        )),
        "relation_counts": counts,
        "relation_types": RELATIONS,
        "graph_features": FEATURES,
        "dynamic_slots": slots,
        "feature_nonzero_fraction": nonzero,
        "human_verified": False,
        "strict_pit_approved": False,
        "note": (
            "SEC availability uses a provisional "
            "filing-date proxy; preserve this limitation."
        ),
    }

    save_json(
        OUT / "manifest.json", manifest
    )

    print("\n=== V40 PREPARE PASS ===")
    print(json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ))


def check_or_train(args, check_only):
    manifest_file = OUT / "manifest.json"

    if not manifest_file.exists():
        fail("Run prepare first")

    for name in ["DYNAMIC_GRAPH", "GRAPH_PLACEBO"]:
        input_dir = OUT / "inputs" / name

        cmd = [
            sys.executable,
            str(RUNNER),
            "--input-dir",
            str(input_dir),
            "--tiers",
            "conservative",
            "inclusive",
            "--agents",
            "A2C", "PPO", "DDPG",
            "--seeds",
            args.seeds,
            "--timesteps",
            str(args.timesteps),
            "--output-dir",
            str(OUT / "runs" / name),
        ]

        if check_only:
            cmd.append("--check-only")

        print("\n=== RUN", name, "===")
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            continue

        log_dir = OUT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        mode = "check" if check_only else "train"
        log = log_dir / f"{mode}_{name}.log"

        with log.open("w") as f:
            process = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        print("Exit:", process.returncode)
        print("Log:", log)

        if process.returncode != 0:
            lines = log.read_text(
                errors="replace"
            ).splitlines()

            print("\n".join(lines[-60:]))

            fail(f"{name} failed")

    print(
        "\n=== V40 CHECK PASS ==="
        if check_only
        else "\n=== V40 TRAIN COMPLETE ==="
    )


def report():
    path = OUT / "manifest.json"

    if not path.exists():
        fail("No V40 manifest")

    manifest = json.loads(path.read_text())

    print("\n=== V40 DYNAMIC KG ===")
    print("Entities:", manifest["entities"])
    print("Event mentions:", manifest["event_mentions"])
    print("Distinct edges:", manifest["distinct_edges"])
    print("Relations:", manifest["relation_counts"])

    result_rows = []

    for experiment in [
        "DYNAMIC_GRAPH",
        "GRAPH_PLACEBO",
    ]:
        p = OUT / "runs" / experiment / "summary.csv"

        if not p.exists():
            print("Not trained:", experiment)
            continue

        data = pd.read_csv(p)

        data = data[
            (data["tier"] == "conservative")
            & (data["split"] == "test")
            & (data["condition"].isin([
                "CLEAN", "FULL_DYNAMIC"
            ]))
        ].copy()

        data["method"] = data["condition"].map({
            "CLEAN": "CLEAN",
            "FULL_DYNAMIC": experiment,
        })

        result_rows.append(data)

    if not result_rows:
        return

    result = pd.concat(
        result_rows,
        ignore_index=True,
    )

    result = result.drop_duplicates(
        ["method", "agent", "tier", "split"]
    )

    cols = [
        "method",
        "agent",
        "total_return_mean",
        "sharpe_mean",
        "max_drawdown_mean",
    ]

    result[cols].to_csv(
        OUT / "comparison.csv",
        index=False,
    )

    print("\n=== TEST PERFORMANCE ===")
    print(
        result[cols]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUT / "comparison.csv"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "prepare",
            "check",
            "train",
            "report",
        ],
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--seeds",
        default="0",
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = p.parse_args()

    if args.command == "prepare":
        prepare(args)

    elif args.command == "check":
        check_or_train(args, True)

    elif args.command == "train":
        check_or_train(args, False)

    elif args.command == "report":
        report()


if __name__ == "__main__":
    main()
