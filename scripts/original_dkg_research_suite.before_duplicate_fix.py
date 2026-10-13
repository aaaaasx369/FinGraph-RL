#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "research_audit"
RUNS = BASE / "runs_multiseed"
V20 = ROOT / "outputs/v20/fusion"

ARMS = ["ORIGINAL_KGTRANSFORMER", "ORIGINAL_KG_PLACEBO"]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
FEATURES = [f"dkg_emb_{i:02d}" for i in range(11)]
RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]


def die(msg):
    raise RuntimeError("RESEARCH AUDIT STOP: " + str(msg))


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print("SAVED:", path)
    return path


def load_runs():
    parts = []
    for arm in ARMS:
        path = RUNS / arm / "runs.csv"
        if not path.exists():
            die(f"Missing: {path}")
        frame = pd.read_csv(path)
        needed = {
            "tier", "condition", "agent", "seed",
            "split", "total_return", "sharpe",
            "max_drawdown",
        }
        if not needed.issubset(frame.columns):
            die(f"Wrong V21 columns: {path}")
        frame = frame.copy()
        frame["experiment"] = np.where(
            frame["condition"].eq("CLEAN"),
            "CLEAN",
            "KGTRANSFORMER"
            if arm == "ORIGINAL_KGTRANSFORMER"
            else "PLACEBO",
        )
        if arm == "ORIGINAL_KG_PLACEBO":
            frame = frame[
                frame["experiment"].eq("PLACEBO")
            ]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def paired():
    print("\n=== 1. PAIRED STATISTICS ===")
    raw = load_runs()
    raw = raw[raw["split"].eq("test")].copy()
    keys = ["tier", "agent", "seed"]

    if raw.duplicated(keys + ["experiment"]).any():
        die("Duplicate experimental runs")

    comparisons = [
        ("KGTRANSFORMER", "CLEAN"),
        ("KGTRANSFORMER", "PLACEBO"),
        ("PLACEBO", "CLEAN"),
    ]

    output = []
    detailed = []

    for (tier, agent), sub in raw.groupby(
        ["tier", "agent"], sort=True
    ):
        for metric in [
            "total_return", "sharpe", "max_drawdown"
        ]:
            pivot = sub.pivot(
                index="seed",
                columns="experiment",
                values=metric,
            )

            for left, right in comparisons:
                if left not in pivot or right not in pivot:
                    die(f"Missing arm {tier} {agent} {left}/{right}")

                paired_df = pivot[[left, right]].dropna()
                if len(paired_df) != 5:
                    die(
                        f"Expected 5 paired seeds; got "
                        f"{len(paired_df)}: {tier} {agent}"
                    )
                d = paired_df[left] - paired_df[right]
                values = d.to_numpy(dtype=float)
                n = len(values)
                mean = float(values.mean())
                sd = float(values.std(ddof=1))
                se = sd / np.sqrt(n)

                if sd == 0:
                    p = 1.0 if mean == 0 else 0.0
                    ci_low = ci_high = mean
                    t = 0.0 if mean == 0 else np.inf
                    dz = 0.0 if mean == 0 else np.inf
                else:
                    t, p = stats.ttest_1samp(values, 0)
                    critical = stats.t.ppf(0.975, n-1)
                    ci_low = mean-critical*se
                    ci_high = mean+critical*se
                    dz = mean/sd

                output.append({
                    "tier": tier,
                    "agent": agent,
                    "metric": metric,
                    "comparison": f"{left}-{right}",
                    "n": n,
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "t_stat": float(t),
                    "p_uncorrected": float(p),
                    "cohens_dz": dz,
                })

                for seed, difference in d.items():
                    detailed.append({
                        "tier": tier,
                        "agent": agent,
                        "metric": metric,
                        "comparison": f"{left}-{right}",
                        "seed": seed,
                        "difference": difference,
                    })

    result = pd.DataFrame(output)

    # Holm adjustment within each metric/tier,
    # across agent x comparison hypotheses.
    result["p_holm"] = np.nan

    for (_, _), idx in result.groupby(
        ["tier", "metric"]
    ).groups.items():
        ids = list(idx)
        pvals = result.loc[
            ids, "p_uncorrected"
        ].to_numpy(dtype=float)
        order = np.argsort(pvals)
        m = len(pvals)
        adjusted = np.empty(m)
        running = 0.0
        for rank, j in enumerate(order):
            running = max(
                running, (m-rank)*pvals[j]
            )
            adjusted[j] = min(1.0, running)
        result.loc[ids, "p_holm"] = adjusted

    save(result, "paired_tests.csv")
    save(pd.DataFrame(detailed), "paired_seed_differences.csv")

    show = result[
        result["metric"].eq("total_return")
        & result["comparison"].isin([
            "KGTRANSFORMER-CLEAN",
            "KGTRANSFORMER-PLACEBO",
        ])
    ]
    print(show[
        [
            "tier", "agent", "comparison",
            "mean_difference", "ci95_low",
            "ci95_high", "p_uncorrected", "p_holm",
        ]
    ].round(5).to_string(index=False))


def audit():
    print("\n=== 2. STRICT PIT AUDIT ===")
    findings = []

    def add(item, result, note):
        findings.append({
            "check": item,
            "status": result,
            "detail": note,
        })
        print(f"[{result}] {item}: {note}")

    manifest_path = BASE / "original_embedding_manifest.json"
    daily_path = BASE / "original_daily_embeddings.csv"
    weekly_path = BASE / "weekly_embeddings_64d.csv"
    pca_path = BASE / "train_only_pca.npz"

    if not manifest_path.exists():
        die("Missing original embedding manifest")

    meta = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    add(
        "Original checkpoint",
        "PASS" if meta.get(
            "original_model_loaded_strictly"
        ) else "FAIL",
        str(meta.get("checkpoint_path")),
    )

    add(
        "Train-only PCA declared",
        "PASS" if meta.get(
            "pca_fit_period"
        ) == "2018-2021" else "FAIL",
        str(meta.get("pca_fit_period")),
    )

    add(
        "PCA transform artifact",
        "PASS" if pca_path.exists() else "FAIL",
        str(pca_path),
    )

    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        daily["date"] = pd.to_datetime(
            daily["date"], errors="coerce"
        )

        duplicate = daily.duplicated(
            ["date", "tic"]
        ).sum()

        add(
            "Unique daily ticker-date",
            "PASS" if duplicate == 0 else "FAIL",
            f"duplicate={duplicate}",
        )

        add(
            "Daily universe",
            "PASS" if set(daily["tic"]) == set(
                TICKERS
            ) else "FAIL",
            str(sorted(daily["tic"].dropna().unique())),
        )

        add(
            "11 finite KG features",
            "PASS" if set(FEATURES).issubset(
                daily.columns
            ) and np.isfinite(
                daily[FEATURES].to_numpy(float)
            ).all() else "FAIL",
            "all features checked",
        )
    else:
        add("Daily embeddings", "FAIL", "file missing")

    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path)
        weekly["week_date"] = pd.to_datetime(
            weekly["week_date"], errors="coerce"
        )
        add(
            "Weekly unique snapshots",
            "PASS" if not weekly.duplicated(
                ["time_id", "tic"]
            ).any() else "FAIL",
            f"rows={len(weekly)}",
        )

        add(
            "2023 KG coverage",
            "PASS" if (
                weekly["week_date"].dt.year.eq(2023)
            ).any() else "WARN",
            "The source graph has very sparse 2023 events; "
            "missing weeks must use the last known state",
        )
    else:
        add("Weekly embeddings", "FAIL", "file missing")

    original = pd.read_csv(
        V20 / "dataset_conservative.csv",
        low_memory=False,
    )

    add(
        "Market rows",
        "PASS" if len(original) == 7545 else "WARN",
        f"rows={len(original)}",
    )

    # Critical methodological checks:
    add(
        "Train-period historical walk-forward",
        "NOT CERTIFIED",
        "2018-2021 states replayed with a checkpoint "
        "trained using the full train period and "
        "selected using 2022 validation",
    )

    add(
        "Original KG publication timestamp",
        "NOT CERTIFIED",
        "Original FinDKG-full week timestamps do not "
        "prove real-world event publication times",
    )

    add(
        "SEC true publication availability",
        "NOT CERTIFIED",
        "Filing date +1 day is a proxy; actual acceptance "
        "timestamp and market cutoff are not established",
    )

    add(
        "RL trade execution timing",
        "NOT CERTIFIED",
        "Requires inspection of FinRL-X observation, "
        "action and execution date alignment",
    )

    add(
        "2022 validation model selection",
        "DISCLOSED",
        "Original best checkpoint selected using validation",
    )

    add(
        "Four requested relations",
        "NOT CERTIFIED",
        "Native FinDKG-full uses 15 different relation types",
    )

    result = pd.DataFrame(findings)
    save(result, "pit_audit.csv")

    blocking = result["status"].isin([
        "FAIL", "WARN", "NOT CERTIFIED",
    ]).sum()

    print(
        "\nSTRICT PIT APPROVAL:",
        "NOT COMPLETE" if blocking else "APPROVED"
    )


def frozen_build():
    """
    Extra-feature control, NOT a graph-relation ablation:
    freeze each stock's 11D features at its earliest
    available pre-2022 observation.
    """
    print("\n=== 3. FROZEN FEATURE CONTROL ===")

    sys.path.insert(0, str(ROOT / "scripts"))
    import v36_finrl_kg_ablation as v36

    source = BASE / "inputs/ORIGINAL_KGTRANSFORMER"
    target = BASE / "inputs/ORIGINAL_KG_FROZEN"

    target.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (V20 / "feature_sets.json").read_text()
    )

    for tier in ["conservative", "inclusive"]:
        path = source / f"dataset_{tier}.csv"
        original = V20 / f"dataset_{tier}.csv"

        source_df = pd.read_csv(path, low_memory=False)
        clean_df = pd.read_csv(original, low_memory=False)

        if len(source_df) != len(clean_df):
            die("Mismatched clean/dynamic market data")

        date_col, ticker_col = v36.identify_keys(
            source_df
        )

        slots = v36.discover_dynamic_slots(
            config, set(source_df.columns), ""
        )

        if len(slots) != 11:
            die("Cannot identify 11 dynamic slots")

        source_df[date_col] = pd.to_datetime(
            source_df[date_col]
        )
        clean_df[date_col] = pd.to_datetime(
            clean_df[date_col]
        )

        if not source_df[
            [date_col, ticker_col]
        ].equals(clean_df[[date_col, ticker_col]]):
            die("Market rows are not aligned")

        frozen = source_df.copy()

        for ticker in TICKERS:
            mask = (
                frozen[ticker_col]
                .astype(str)
                .str.upper()
                .eq(ticker)
            )

            train_rows = source_df.loc[
                mask & source_df[date_col].lt(
                    pd.Timestamp("2022-01-01")
                )
            ].sort_values(date_col)

            if train_rows.empty:
                die(f"No train rows: {ticker}")

            # Earliest nonzero signal, else zero.
            nonzero = train_rows[
                train_rows[slots].abs().sum(axis=1) > 0
            ]

            if nonzero.empty:
                vector = np.zeros(len(slots))
            else:
                vector = nonzero.iloc[0][slots].to_numpy(
                    dtype=float
                )

            frozen.loc[mask, slots] = vector

        for col in clean_df.columns:
            if col not in slots and not source_df[
                col
            ].equals(frozen[col]):
                die("Unexpected modified field: " + col)

        frozen[date_col] = frozen[
            date_col
        ].dt.strftime("%Y-%m-%d")

        frozen.to_csv(
            target / f"dataset_{tier}.csv",
            index=False,
        )
        print("FROZEN READY:", tier)

    shutil.copy2(
        V20 / "feature_sets.json",
        target / "feature_sets.json",
    )

    print("FROZEN INPUT:", target)


def frozen_run(seeds, timesteps):
    source = BASE / "inputs/ORIGINAL_KG_FROZEN"

    if not (
        source / "dataset_conservative.csv"
    ).exists():
        die("Run frozen-build first")

    command = [
        sys.executable, "-u",
        str(ROOT / "scripts/run_v21_dynamic_ablation.py"),
        "--input-dir", str(source),
        "--tiers", "conservative", "inclusive",
        "--agents", "A2C", "PPO", "DDPG",
        "--seeds", *[str(s) for s in seeds],
        "--timesteps", str(timesteps),
        "--output-dir",
        str(BASE / "runs_multiseed/ORIGINAL_KG_FROZEN"),
    ]

    print("COMMAND:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def relation_data():
    """
    Prepare relation-removal datasets, preserving
    original numeric relation IDs.

    This does NOT retrain original KGTransformer.
    """
    print("\n=== 4. RELATIONAL ABLATION DATA ===")

    original = (
        ROOT
        / "references/Financial-TKG/FinDKG_dataset/"
        "FinDKG-full"
    )

    split_dir = RUNS.parents[0] / (
        "original_run/FinDKG_dataset/FinDKG-full"
    )

    # Prefer actual chronological re-split files.
    split_dir = BASE / (
        "original_run/FinDKG_dataset/FinDKG-full"
    )

    if not split_dir.exists():
        die(f"Missing chronological KG dataset: {split_dir}")

    relation_path = original / "relation2id.txt"

    if not relation_path.exists():
        die("Missing original relation2id.txt")

    relation_map = []

    for line in relation_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                relation_map.append({
                    "relation": parts[0],
                    "relation_id": int(parts[1]),
                })
            except ValueError:
                continue

    rel = pd.DataFrame(relation_map)

    if len(rel) != 15:
        die(
            "Expected original 15 relation definitions; "
            f"found {len(rel)}"
        )

    save(rel, "original_relation_ids.csv")

    destination = OUT / "relation_ablations"
    destination.mkdir(exist_ok=True)

    # One complete graph for each leave-one-relation-out
    # experiment, preserving original node/relation IDs.
    variants = [
        (f"DROP_{r.relation}", int(r.relation_id))
        for r in rel.itertuples(index=False)
    ]

    counts = []

    for name, excluded_id in variants:
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)

        for filename in [
            "entity2id.txt", "relation2id.txt",
            "time2id.txt", "stat.txt",
        ]:
            shutil.copy2(
                split_dir / filename,
                folder / filename,
            )

        for split in [
            "train.txt", "valid.txt", "test.txt"
        ]:
            data = pd.read_csv(
                split_dir / split,
                sep=r"\s+",
                header=None,
            )

            if data.shape[1] < 5:
                die(f"Invalid KG split: {split}")

            relation_ids = pd.to_numeric(
                data.iloc[:, 1],
                errors="raise",
            ).astype(int)

            selected = data.loc[
                relation_ids.ne(excluded_id)
            ]

            selected.to_csv(
                folder / split,
                sep="\t",
                index=False,
                header=False,
            )

            counts.append({
                "variant": name,
                "excluded_relation_id": excluded_id,
                "split": split,
                "original_rows": len(data),
                "remaining_rows": len(selected),
            })

    save(
        pd.DataFrame(counts),
        "relation_ablation_event_counts.csv",
    )

    print(
        "\nCreated 15 relation-removal graph datasets."
    )
    print(
        "IMPORTANT: Each graph still requires "
        "a separately trained original KGTransformer."
    )


def four_relations():
    print("\n=== 5. FOUR REQUESTED RELATIONS ===")

    source = ROOT / "outputs/v35/evidence_ledger.csv"

    if not source.exists():
        die("V35 evidence ledger missing")

    ledger = pd.read_csv(source, low_memory=False)

    # Preserve source data instead of inventing labels.
    print("V35 columns:", list(ledger.columns))

    possible = [
        c for c in [
            "relation", "relation_type",
            "predicate", "edge_type",
        ]
        if c in ledger.columns
    ]

    found = set()

    for col in possible:
        found.update(
            ledger[col].dropna().astype(str).str.upper()
        )

    report = []

    for rel in RELATIONS:
        present = rel in found

        report.append({
            "requested_relation": rel,
            "evidence_in_v35": present,
            "status": (
                "CANDIDATE_EVIDENCE_PRESENT"
                if present
                else "MISSING_VERIFIED_EVIDENCE"
            ),
            "ready_for_retraining": False,
        })

    save(
        pd.DataFrame(report),
        "four_relations_coverage.csv",
    )

    # Deliver an evidence-first input schema,
    # but DO NOT generate fake edges.
    evidence_file = OUT / "four_relations_verified_edges.csv"

    if not evidence_file.exists():
        pd.DataFrame(columns=[
            "head_company",
            "relation",
            "tail_company",
            "valid_from",
            "valid_until",
            "source_url",
            "source_document",
            "evidence_quote",
            "review_status",
        ]).to_csv(
            evidence_file, index=False
        )

    print(
        "\nVerified edge input schema:",
        evidence_file
    )
    print(
        "Coverage is not completion. "
        "Supplier/customer/sector edges require "
        "historical verifiable sources."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "paired",
            "audit",
            "frozen-build",
            "frozen-run",
            "relation-data",
            "four-relations",
            "all-prep",
        ],
    )
    parser.add_argument(
        "--seeds", default="0,1,2,3,4"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2000
    )
    args = parser.parse_args()

    if args.task == "paired":
        paired()
    elif args.task == "audit":
        audit()
    elif args.task == "frozen-build":
        frozen_build()
    elif args.task == "frozen-run":
        frozen_run(
            [int(x) for x in args.seeds.split(",")],
            args.timesteps,
        )
    elif args.task == "relation-data":
        relation_data()
    elif args.task == "four-relations":
        four_relations()
    elif args.task == "all-prep":
        paired()
        audit()
        frozen_build()
        relation_data()
        four_relations()


if __name__ == "__main__":
    main()
