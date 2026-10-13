#!/usr/bin/env python3
"""
V32: Human Validation Benchmark + Verified Relation Gate

Research companies:
  AAPL MSFT NVDA AMZN GOOGL

Input:
  outputs/v31/evidence_alignment.parquet

Commands:
  prepare
  evaluate
  graph
  audit

Workflow:
  prepare -> independently annotate two CSVs
          -> evaluate -> graph -> audit

Dependencies:
  numpy pandas pyarrow scikit-learn

IMPORTANT:
  - No automatic ground truth generation.
  - No automatic economic-event verification.
  - No relation edges without two agreeing reviewers.
  - No future SEC evidence used to classify historical text.
  - Static label agreement != fully certified point-in-time
    financial data or trading execution.
"""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v32"
SOURCE = (
    ROOT / "outputs/v31/"
    "evidence_alignment.parquet"
)

STOCKS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
}

TARGET_LABELS = [
    "NEW_TEXT_CANDIDATE",
    "RELATED_WORDING_CHANGE",
    "HISTORICAL_NEAR_REPEAT",
    "PREVIOUS_EXACT_REPEAT",
]

# Precommitted sampling design:
# up to three pairs per ticker/topic/classification.
N_PER_STRATUM = 3
SEED = 32

NOVELTY_LABELS = [
    "NEW",
    "REPEAT",
    "UNCERTAIN",
]

EVENT_LABELS = [
    "TRUE_EVENT",
    "TEXT_ONLY",
    "UNCERTAIN",
]

RELATION_LABELS = [
    "VERIFIED",
    "NOT_RELATION",
    "UNCERTAIN",
]

RELATION_TYPES = {
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
    "PARTNERS_WITH",
    "DEPENDS_ON",
    "MANUFACTURED_BY",
}

ANNOTATION_COLUMNS = [
    "novelty_label",
    "event_label",
    "relation_label",
    "subject_company",
    "relation_type",
    "object_entity",
    "review_notes",
]


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def make_id(*parts):
    source = "\x1f".join(map(str, parts))
    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:24]


def get_source():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Run V31 before V32: " + str(SOURCE)
        )

    df = pd.read_parquet(SOURCE)

    required = [
        "pair_id",
        "ticker",
        "topic",
        "form",
        "v31_label",
        "current_excerpt",
        "best_prior_excerpt",
        "current_filing_date",
        "best_prior_filing_date",
        "current_accession",
        "best_prior_accession",
        "current_offset",
        "best_prior_offset",
        "current_document_sha256",
        "best_prior_document_sha256",
        "source_span_verified_v29",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"V31 missing columns: {missing}"
        )

    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs")

    if set(df["ticker"]) != STOCKS:
        raise ValueError(
            "Five-stock universe violated"
        )

    if not df[
        "source_span_verified_v29"
    ].astype(bool).all():
        raise ValueError(
            "Unverified source evidence"
        )

    current = pd.to_datetime(
        df["current_filing_date"]
    )

    past = pd.to_datetime(
        df["best_prior_filing_date"]
    )

    has_past = past.notna()

    if not (past[has_past] < current[has_past]).all():
        raise ValueError(
            "Future or same-date prior evidence"
        )

    return df


def prepare():
    """
    Probability sampling within each
    company/topic/classification stratum.

    Both reviewers get exactly the same sample.

    Population inference is restricted to the four
    included V31 classes, not all 21,016 snippets.
    """
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    source = get_source()

    subset = source[
        source["v31_label"].isin(
            TARGET_LABELS
        )
    ].copy()

    rng = np.random.default_rng(SEED)

    parts = []
    design = []

    for key, group in subset.groupby(
        ["ticker", "topic", "v31_label"],
        sort=True
    ):
        N = len(group)
        n = min(N_PER_STRATUM, N)

        selected = rng.choice(
            group.index.to_numpy(),
            size=n,
            replace=False,
        )

        sample = group.loc[selected].copy()

        sample["stratum_id"] = "|".join(key)
        sample["population_N"] = N
        sample["sample_n"] = n
        sample["sample_weight"] = N / n

        parts.append(sample)

        design.append({
            "stratum_id": "|".join(key),
            "ticker": key[0],
            "topic": key[1],
            "v31_label": key[2],
            "population_N": N,
            "sample_n": n,
            "inclusion_probability": n / N,
        })

    if not parts:
        raise RuntimeError(
            "No rows eligible for sampling"
        )

    sample = pd.concat(
        parts,
        ignore_index=True
    )

    sample = sample.sort_values(
        ["ticker", "topic", "v31_label", "pair_id"]
    ).reset_index(drop=True)

    design = pd.DataFrame(design)

    selected_columns = [
        "pair_id",
        "ticker",
        "topic",
        "form",
        "v31_label",
        "v29_label",
        "v30_label",
        "current_filing_date",
        "best_prior_filing_date",
        "current_accession",
        "best_prior_accession",
        "current_offset",
        "best_prior_offset",
        "current_document_sha256",
        "best_prior_document_sha256",
        "current_excerpt",
        "best_prior_excerpt",
        "current_core",
        "best_prior_core",
        "similarity",
        "match_method",
        "population_N",
        "sample_n",
        "sample_weight",
        "stratum_id",
    ]

    selected_columns = [
        c for c in selected_columns
        if c in sample.columns
    ]

    sample = sample[selected_columns].copy()

    sample.to_csv(
        OUT / "sample_master.csv",
        index=False
    )

    design.to_csv(
        OUT / "sampling_design.csv",
        index=False
    )

    # Never overwrite completed annotation work.
    for reviewer in ["R1", "R2"]:
        path = OUT / f"review_{reviewer}.csv"

        if path.exists():
            raise FileExistsError(
                f"{path} already exists. "
                "Move it elsewhere before preparing "
                "a new sample."
            )

        sheet = sample.copy()

        sheet["reviewer_id"] = reviewer

        for col in ANNOTATION_COLUMNS:
            sheet[col] = ""

        sheet.to_csv(
            path,
            index=False
        )

    report = {
        "source_total": len(source),
        "target_population": len(subset),
        "sample_size": len(sample),
        "strata": len(design),
        "sampling_seed": SEED,
        "labels": TARGET_LABELS,
        "two_independent_review_sheets": True,
        "ground_truth_exists": False,
        "strict_pit_approved": False,
    }

    write_json(
        OUT / "prepare_report.json",
        report
    )

    print("\n=== V32 SAMPLE PREPARED ===")
    print(json.dumps(
        report,
        indent=2
    ))

    print("\nAnnotation files:")
    print(OUT / "review_R1.csv")
    print(OUT / "review_R2.csv")


def read_reviews():
    master_path = OUT / "sample_master.csv"

    if not master_path.exists():
        raise FileNotFoundError(
            "Run prepare first"
        )

    master = pd.read_csv(
        master_path,
        dtype={"pair_id": str}
    )

    sheets = []

    for reviewer in ["R1", "R2"]:
        path = OUT / f"review_{reviewer}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False
        )

        if "pair_id" not in df.columns:
            raise ValueError(
                f"{reviewer}: missing pair_id"
            )

        if df["pair_id"].duplicated().any():
            raise ValueError(
                f"{reviewer}: duplicate pair IDs"
            )

        if set(df["pair_id"]) != set(master["pair_id"]):
            raise ValueError(
                f"{reviewer}: sample IDs changed"
            )

        for col in ANNOTATION_COLUMNS:
            if col not in df:
                raise ValueError(
                    f"{reviewer}: missing {col}"
                )

        # Only annotation columns are trusted
        # from reviewer sheets; all source and
        # sampling data come from master.
        df = df[
            ["pair_id"] + ANNOTATION_COLUMNS
        ].copy()

        for col in ANNOTATION_COLUMNS:
            df[col] = (
                df[col].astype(str).str.strip()
            )

        for col in [
            "novelty_label",
            "event_label",
            "relation_label",
            "relation_type",
            "subject_company",
        ]:
            df[col] = df[col].str.upper()

        df["reviewer_id"] = reviewer

        sheets.append(df)

    all_labels = pd.concat(
        sheets,
        ignore_index=True
    )

    return master, all_labels


def validate_labels(master, annotations):
    allowed = {
        "novelty_label": set(NOVELTY_LABELS),
        "event_label": set(EVENT_LABELS),
        "relation_label": set(RELATION_LABELS),
    }

    for col, choices in allowed.items():
        invalid = annotations[
            ~annotations[col].isin(choices)
        ]

        if len(invalid):
            raise ValueError(
                f"Invalid/uncompleted {col}: "
                f"{len(invalid)} rows. "
                "Finish both reviewer sheets first."
            )

    verified = annotations[
        annotations["relation_label"].eq(
            "VERIFIED"
        )
    ]

    for r in verified.itertuples(index=False):
        if r.relation_type not in RELATION_TYPES:
            raise ValueError(
                f"Invalid verified relation type: "
                f"{r.pair_id}"
            )

        if r.subject_company not in STOCKS:
            raise ValueError(
                "Only five research companies "
                "can be relation subjects: "
                f"{r.pair_id}"
            )

        if not str(r.object_entity).strip():
            raise ValueError(
                f"Missing object: {r.pair_id}"
            )

        if r.event_label != "TRUE_EVENT":
            raise ValueError(
                "Verified relation requires "
                "TRUE_EVENT label: "
                f"{r.pair_id}"
            )

    return True


def binary_metrics(y_true, y_pred, weights=None):
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)

    return {
        "precision": float(
            precision_score(
                true, pred,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true, pred,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true, pred,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "confusion_matrix_TN_FP_FN_TP": (
            confusion_matrix(
                true,
                pred,
                labels=[0, 1],
                sample_weight=weights,
            ).ravel().tolist()
        ),
    }


def evaluate():
    master, annotations = read_reviews()

    validate_labels(master, annotations)

    wide = master.copy()

    for reviewer in ["R1", "R2"]:
        small = annotations[
            annotations["reviewer_id"].eq(
                reviewer
            )
        ].drop(
            columns=["reviewer_id"]
        )

        small = small.rename(
            columns={
                c: c + "_" + reviewer
                for c in ANNOTATION_COLUMNS
            }
        )

        wide = wide.merge(
            small,
            on="pair_id",
            how="left",
            validate="one_to_one",
        )

    report = {
        "sample_size": len(wide),
        "population_scope": (
            "Four selected V31 categories only"
        ),
        "inter_annotator": {},
        "adjudicated_agreement_subset": {},
        "unresolved_rows": {},
        "strict_pit_approved": False,
    }

    for kind, column in [
        ("novelty", "novelty_label"),
        ("economic_event", "event_label"),
        ("directed_relation", "relation_label"),
    ]:
        r1 = wide[column + "_R1"]
        r2 = wide[column + "_R2"]

        agreement = float(
            (r1 == r2).mean()
        )

        kappa = cohen_kappa_score(
            r1, r2
        )

        agreed = r1.eq(r2)
        certain = agreed & r1.ne(
            "UNCERTAIN"
        )

        report["inter_annotator"][kind] = {
            "raw_agreement": agreement,
            "cohens_kappa": (
                float(kappa)
                if np.isfinite(kappa)
                else None
            ),
            "agreed_rows": int(
                agreed.sum()
            ),
            "agreed_certain_rows": int(
                certain.sum()
            ),
        }

        report["unresolved_rows"][kind] = int(
            (~certain).sum()
        )

    # Conservative pseudo-consensus:
    # use only exact two-reviewer agreement.
    # Disagreements are NOT auto-adjudicated.
    agreed_novelty = (
        wide["novelty_label_R1"]
        == wide["novelty_label_R2"]
    ) & wide["novelty_label_R1"].ne(
        "UNCERTAIN"
    )

    evaluation = wide.loc[
        agreed_novelty
    ].copy()

    if len(evaluation):
        truth = evaluation[
            "novelty_label_R1"
        ].eq("NEW").astype(int)

        prediction = evaluation[
            "v31_label"
        ].eq(
            "NEW_TEXT_CANDIDATE"
        ).astype(int)

        report[
            "adjudicated_agreement_subset"
        ]["novelty_unweighted"] = (
            binary_metrics(
                truth, prediction
            )
        )

        report[
            "adjudicated_agreement_subset"
        ]["novelty_design_weighted"] = (
            binary_metrics(
                truth,
                prediction,
                evaluation[
                    "sample_weight"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        report[
            "adjudicated_agreement_subset"
        ]["n"] = len(evaluation)

    # Separately evaluate how often V31 novel
    # text corresponds to an agreed TRUE_EVENT.
    event_certain = (
        wide["event_label_R1"]
        == wide["event_label_R2"]
    ) & wide["event_label_R1"].ne(
        "UNCERTAIN"
    )

    event_eval = wide.loc[
        event_certain
    ].copy()

    if len(event_eval):
        report[
            "adjudicated_agreement_subset"
        ]["economic_event_detection"] = (
            binary_metrics(
                event_eval[
                    "event_label_R1"
                ].eq("TRUE_EVENT"),
                event_eval[
                    "v31_label"
                ].eq("NEW_TEXT_CANDIDATE"),
                event_eval[
                    "sample_weight"
                ].to_numpy(dtype=float),
            )
        )

    wide["novelty_agreed"] = agreed_novelty
    wide["event_agreed"] = event_certain

    wide.to_csv(
        OUT / "review_comparison.csv",
        index=False
    )

    write_json(
        OUT / "evaluation_report.json",
        report
    )

    print("\n=== V32 HUMAN EVALUATION ===")
    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ))

    print(
        "\nNOTE: Agreement-only metrics may "
        "be biased by excluded disagreements."
    )


def build_graph():
    """
    Only two reviewers agreeing on:
      TRUE_EVENT + VERIFIED + same subject,
      relation type, object
    can create a candidate verified edge.

    These are human-confirmed research labels;
    not automatically certified financial facts.

    Output preserves the exact SEC evidence.
    """
    master, ann = read_reviews()
    validate_labels(master, ann)

    r1 = ann[
        ann["reviewer_id"].eq("R1")
    ].set_index("pair_id")

    r2 = ann[
        ann["reviewer_id"].eq("R2")
    ].set_index("pair_id")

    nodes = {}
    edges = []

    for ticker in STOCKS:
        nodes["COMPANY:" + ticker] = {
            "node_id": "COMPANY:" + ticker,
            "node_type": "COMPANY",
            "label": ticker,
        }

    source = get_source().set_index("pair_id")

    for pair_id in master["pair_id"]:
        a = r1.loc[pair_id]
        b = r2.loc[pair_id]

        both = (
            a["event_label"] == "TRUE_EVENT"
            and b["event_label"] == "TRUE_EVENT"
            and a["relation_label"] == "VERIFIED"
            and b["relation_label"] == "VERIFIED"
        )

        if not both:
            continue

        subject_agrees = (
            a["subject_company"]
            == b["subject_company"]
        )

        relation_agrees = (
            a["relation_type"]
            == b["relation_type"]
        )

        object_agrees = (
            a["object_entity"].casefold()
            == b["object_entity"].casefold()
        )

        if not (
            subject_agrees
            and relation_agrees
            and object_agrees
        ):
            continue

        row = source.loc[pair_id]

        subject = a["subject_company"]
        relation = a["relation_type"]
        obj = a["object_entity"].strip()

        # External organizations are evidence
        # entities, not portfolio assets.
        obj_id = (
            "COMPANY:" + obj
            if obj in STOCKS
            else "EXTERNAL_ENTITY:"
            + make_id(obj.casefold())
        )

        if obj_id.startswith("EXTERNAL_ENTITY:"):
            nodes[obj_id] = {
                "node_id": obj_id,
                "node_type": "EXTERNAL_ENTITY",
                "label": obj,
            }

        document_id = (
            "DOCUMENT:"
            + str(row["ticker"])
            + ":"
            + str(row["current_accession"])
        )

        nodes[document_id] = {
            "node_id": document_id,
            "node_type": "DOCUMENT",
            "label": str(
                row["current_accession"]
            ),
        }

        evidence_id = (
            "EVIDENCE:"
            + str(pair_id)
        )

        nodes[evidence_id] = {
            "node_id": evidence_id,
            "node_type": "HUMAN_VERIFIED_EVIDENCE",
            "label": pair_id,
        }

        edge = {
            "edge_id": make_id(
                pair_id, subject, relation, obj
            ),
            "pair_id": pair_id,
            "ticker": str(row["ticker"]),
            "subject": subject,
            "relation": relation,
            "object": obj,
            "source_node": "COMPANY:" + subject,
            "target_node": obj_id,
            "document_node": document_id,
            "evidence_node": evidence_id,
            "current_accession": str(
                row["current_accession"]
            ),
            "filing_date": str(
                row["current_filing_date"]
            ),
            "available_from": str(
                pd.Timestamp(
                    row["current_filing_date"]
                ).normalize()
                + pd.Timedelta(days=1)
            ),
            "source_excerpt": str(
                row["current_excerpt"]
            ),
            "source_sha256": str(
                row["current_document_sha256"]
            ),
            "reviewer_agreement": True,
            "human_verified": True,
            "external_portfolio_asset": False,
            "strict_pit_verified": False,
        }

        edges.append(edge)

    edge_columns = [
        "edge_id", "pair_id", "ticker",
        "subject", "relation", "object",
        "source_node", "target_node",
        "document_node", "evidence_node",
        "current_accession", "filing_date",
        "available_from", "source_excerpt",
        "source_sha256", "reviewer_agreement",
        "human_verified", "external_portfolio_asset",
        "strict_pit_verified",
    ]

    edge_frame = pd.DataFrame(
        edges,
        columns=edge_columns
    )

    nodes_frame = pd.DataFrame(
        list(nodes.values())
    )

    nodes_frame.to_parquet(
        OUT / "verified_graph_nodes.parquet",
        index=False
    )

    edge_frame.to_parquet(
        OUT / "verified_relation_edges.parquet",
        index=False
    )

    edge_frame.to_csv(
        OUT / "verified_relation_edges.csv",
        index=False
    )

    report = {
        "portfolio_universe": sorted(STOCKS),
        "double_confirmed_relation_edges": int(
            len(edge_frame)
        ),
        "all_edges_preserve_source": True,
        "strict_pit_approved": False,
        "note": (
            "Human double-confirmed labels; "
            "not independently certified facts. "
            "No full trading timing verification."
        ),
    }

    write_json(
        OUT / "graph_report.json",
        report
    )

    print("\n=== V32 VERIFIED GRAPH GATE ===")
    print(json.dumps(
        report,
        indent=2
    ))


def audit():
    source = get_source()

    master, annotations = read_reviews()

    if master["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample pairs"
        )

    if not set(
        master["pair_id"]
    ).issubset(
        set(source["pair_id"])
    ):
        raise ValueError(
            "Unknown sample pair"
        )

    if not np.allclose(
        master["sample_weight"],
        master["population_N"]
        / master["sample_n"]
    ):
        raise ValueError(
            "Invalid sampling weights"
        )

    for reviewer in ["R1", "R2"]:
        count = int(
            annotations[
                "reviewer_id"
            ].eq(reviewer).sum()
        )

        if count != len(master):
            raise ValueError(
                f"Incomplete reviewer file: {reviewer}"
            )

    print("\n=== V32 AUDIT ===")
    print("Five-company input: PASS")
    print("Historical source dates: PASS")
    print("Evidence sample identity: PASS")
    print("Stratified sampling weights: PASS")
    print("Two reviewer sheets: PASS")

    print(
        "Human verification complete:",
        bool(
            annotations[
                "event_label"
            ].ne("").all()
        )
    )

    print("Strict PIT: NOT COMPLETE")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "evaluate",
            "graph",
            "audit",
        ]
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "graph":
        build_graph()
    elif args.command == "audit":
        audit()


if __name__ == "__main__":
    main()
