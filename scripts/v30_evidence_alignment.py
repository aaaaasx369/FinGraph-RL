#!/usr/bin/env python3
"""
V30 - SEC Previous/Current Evidence Alignment

Five study companies only:
AAPL MSFT NVDA AMZN GOOGL

Read-only inputs:
  outputs/v28/filing_features.csv
  outputs/v28/filing_evidence.parquet
  outputs/v29/evidence_changes.parquet
  outputs/v20/sec_filings_manifest.csv

Outputs:
  outputs/v30/evidence_pairs.parquet
  outputs/v30/evidence_pairs.csv
  outputs/v30/event_candidates.parquet
  outputs/v30/event_graph_nodes.parquet
  outputs/v30/event_graph_edges.parquet
  outputs/v30/human_review.csv
  outputs/v30/report.json

Commands:
  build
  audit
  evaluate --labels FILE
  all

No API key. No automatic verified economic events.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v30"

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
]

TOPICS = [
    "supply_chain",
    "manufacturing",
    "cloud",
    "competition",
    "regulation",
    "litigation",
    "finance",
    "research",
]

# Fixed before looking at test performance.
SHINGLE_N = 4
BOILERPLATE_THRESHOLD = 0.88
MINOR_THRESHOLD = 0.55
CANDIDATE_THRESHOLD = 0.35
MAX_REVIEW_PER_GROUP = 3
RANDOM_SEED = 30

LABELS = [
    "EXACT_REPEAT",
    "NUMERIC_CHANGE",
    "BOILERPLATE",
    "MINOR_WORDING",
    "NEW_DISCLOSURE_CANDIDATE",
    "UNMATCHED_NO_PRIOR",
    "LOW_INFORMATION",
]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def hash_id(*parts):
    payload = "\x1f".join(map(str, parts))
    return hashlib.sha256(
        payload.encode()
    ).hexdigest()[:24]


def dates(column):
    return pd.to_datetime(
        column, errors="raise"
    ).dt.normalize()


def normalize(text):
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower()
        )
    )


def normalize_numbers(text):
    # Replace numeric sequences including currency,
    # percentages and comma-separated quantities.
    text = str(text).lower()
    text = re.sub(
        r"(?<![a-z])[$€£]?\s*"
        r"[-+]?\d[\d,]*(?:\.\d+)?%?",
        " NUMBER ",
        text
    )
    return normalize(text)


def numeric_tokens(text):
    return re.findall(
        r"(?<![a-z])[$€£]?\s*"
        r"[-+]?\d[\d,]*(?:\.\d+)?%?",
        str(text).lower()
    )


def shingles(text, n=SHINGLE_N):
    words = normalize(text).split()

    if len(words) < n:
        return set(words)

    return {
        hash_id(*words[i:i+n])
        for i in range(
            len(words) - n + 1
        )
    }


def similarity(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def load_data():
    filenames = [
        ROOT / "outputs/v28/filing_features.csv",
        ROOT / "outputs/v28/filing_evidence.parquet",
        ROOT / "outputs/v29/evidence_changes.parquet",
    ]

    for path in filenames:
        if not path.exists():
            raise FileNotFoundError(path)

    filings = pd.read_csv(
        filenames[0], dtype={"accession": str}
    )
    evidence = pd.read_parquet(
        filenames[1]
    )
    v29 = pd.read_parquet(
        filenames[2]
    )

    required_f = {
        "ticker", "accession", "form",
        "filing_date", "available_from",
    }
    required_e = {
        "ticker", "accession", "form",
        "topic", "evidence_id",
        "excerpt", "character_offset",
    }

    if not required_f.issubset(filings.columns):
        raise ValueError("Missing V28 filing columns")

    if not required_e.issubset(evidence.columns):
        raise ValueError("Missing V28 evidence columns")

    for df in (filings, evidence, v29):
        df["ticker"] = (
            df["ticker"].astype(str).str.upper()
        )
        df["accession"] = (
            df["accession"].astype(str)
        )

        if not set(df["ticker"]).issubset(
            set(STOCKS)
        ):
            raise ValueError("Outside five-study universe")

    filings["filing_date"] = dates(
        filings["filing_date"]
    )
    filings["available_from"] = dates(
        filings["available_from"]
    )

    if len(filings) != 140:
        raise ValueError(
            f"Expected 140 filings, got {len(filings)}"
        )

    if filings.duplicated(
        ["ticker", "accession"]
    ).any():
        raise ValueError("Duplicate filing accession")

    if evidence["evidence_id"].duplicated().any():
        raise ValueError("Duplicate evidence ID")

    if v29["evidence_id"].duplicated().any():
        raise ValueError("Duplicate V29 evidence ID")

    # V29 has already performed the source excerpt
    # verification. Require every input to pass.
    if (
        "source_span_verified" not in v29
        or not v29["source_span_verified"]
        .astype(bool).all()
    ):
        raise ValueError(
            "V29 source span verification incomplete"
        )

    verified_ids = set(v29["evidence_id"])

    if set(evidence["evidence_id"]) != verified_ids:
        raise ValueError(
            "V28/V29 evidence IDs differ"
        )

    if not (filings["filing_date"] <
            filings["available_from"]).all():
        raise ValueError(
            "Unsafe filing availability"
        )

    if not set(evidence["topic"]).issubset(
        set(TOPICS)
    ):
        raise ValueError("Unexpected SEC topic")

    return filings, evidence, v29


def previous_filing_map(filings):
    """
    Each filing references the immediately previous
    same-company and same-form filing.

    Never choose future filings as comparisons.
    """
    mapping = {}

    for (ticker, form), group in filings.groupby(
        ["ticker", "form"]
    ):
        group = group.sort_values(
            ["filing_date", "accession"]
        )

        previous = None

        for _, current in group.iterrows():
            key = (
                ticker,
                str(current["accession"])
            )

            mapping[key] = (
                previous
                if previous is not None
                else None
            )

            if previous is not None:
                if not (
                    previous["filing_date"]
                    < current["filing_date"]
                ):
                    raise ValueError(
                        "Previous filing is not earlier"
                    )

            previous = current.to_dict()

    return mapping


def classify(current, previous, score):
    """
    Text alignment labels only.
    Not economic-event ground truth.
    """
    if previous is None:
        return "UNMATCHED_NO_PRIOR"

    a = normalize(current)
    b = normalize(previous)

    if len(a.split()) < 12:
        return "LOW_INFORMATION"

    if a == b:
        return "EXACT_REPEAT"

    na = normalize_numbers(current)
    nb = normalize_numbers(previous)

    if na == nb:
        if numeric_tokens(current) != numeric_tokens(previous):
            return "NUMERIC_CHANGE"
        return "MINOR_WORDING"

    if score >= BOILERPLATE_THRESHOLD:
        return "BOILERPLATE"

    if score >= MINOR_THRESHOLD:
        return "MINOR_WORDING"

    return "NEW_DISCLOSURE_CANDIDATE"


def align(filings, evidence, v29):
    prior_map = previous_filing_map(filings)

    v29_labels = dict(
        zip(
            v29["evidence_id"],
            v29["label"]
        )
    )

    grouped = {}

    for key, group in evidence.groupby(
        ["ticker", "accession", "topic"]
    ):
        rows = group.to_dict("records")

        grouped[
            (
                str(key[0]),
                str(key[1]),
                str(key[2])
            )
        ] = rows

    results = []

    for filing in filings.sort_values(
        ["ticker", "form", "filing_date", "accession"]
    ).to_dict("records"):

        ticker = filing["ticker"]
        accession = str(filing["accession"])
        form = filing["form"]

        previous = prior_map[
            (ticker, accession)
        ]

        previous_accession = (
            str(previous["accession"])
            if previous is not None
            else ""
        )

        for topic in TOPICS:
            current_items = grouped.get(
                (ticker, accession, topic),
                []
            )

            previous_items = grouped.get(
                (
                    ticker,
                    previous_accession,
                    topic
                ),
                []
            ) if previous is not None else []

            prior_signatures = [
                shingles(r["excerpt"])
                for r in previous_items
            ]

            for row in current_items:
                curr_text = str(
                    row["excerpt"]
                )
                curr_shingles = shingles(
                    curr_text
                )

                if previous_items:
                    scores = [
                        similarity(
                            curr_shingles, sig
                        )
                        for sig in prior_signatures
                    ]

                    best_index = int(
                        np.argmax(scores)
                    )
                    best = previous_items[
                        best_index
                    ]
                    best_score = float(
                        scores[best_index]
                    )

                    prior_excerpt = str(
                        best["excerpt"]
                    )

                    prior_id = str(
                        best["evidence_id"]
                    )

                    prior_offset = int(
                        best["character_offset"]
                    )

                else:
                    prior_excerpt = ""
                    prior_id = ""
                    prior_offset = -1
                    best_score = 0.0

                label = classify(
                    curr_text,
                    prior_excerpt if previous_items else None,
                    best_score
                )

                result = {
                    "pair_id": hash_id(
                        ticker,
                        accession,
                        row["evidence_id"]
                    ),
                    "ticker": ticker,
                    "form": form,
                    "topic": topic,
                    "current_accession": accession,
                    "previous_accession": previous_accession,
                    "current_filing_date": filing[
                        "filing_date"
                    ],
                    "previous_filing_date": (
                        previous["filing_date"]
                        if previous is not None
                        else pd.NaT
                    ),
                    "available_from": filing[
                        "available_from"
                    ],
                    "current_evidence_id": str(
                        row["evidence_id"]
                    ),
                    "previous_evidence_id": prior_id,
                    "current_offset": int(
                        row["character_offset"]
                    ),
                    "previous_offset": prior_offset,
                    "current_excerpt": curr_text,
                    "previous_excerpt": prior_excerpt,
                    "similarity": best_score,
                    "novelty": 1.0 - best_score,
                    "change_type": label,
                    "v29_label": v29_labels[
                        row["evidence_id"]
                    ],
                    "numbers_changed": (
                        bool(
                            numeric_tokens(curr_text)
                            != numeric_tokens(prior_excerpt)
                        )
                        if previous_items else False
                    ),
                    "source_span_verified": True,
                    "economic_event_verified": False,
                    "directed_relation_verified": False,
                    "human_reviewed": False,
                }

                results.append(result)

    pairs = pd.DataFrame(results)

    if len(pairs) != len(evidence):
        raise ValueError(
            "Not every evidence span was aligned"
        )

    if pairs["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs")

    return pairs


def make_review_sample(pairs):
    """
    Deterministic stratified sample:
    company x topic x change type.

    This is a balanced audit sample, NOT an unbiased
    estimate of overall event precision.
    """
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    pieces = []

    for _, group in pairs.groupby(
        ["ticker", "topic", "change_type"]
    ):
        n = min(
            MAX_REVIEW_PER_GROUP,
            len(group)
        )

        selected = rng.choice(
            group.index.to_numpy(),
            size=n,
            replace=False
        )

        pieces.append(
            pairs.loc[selected]
        )

    review = pd.concat(
        pieces,
        ignore_index=True
    )

    review = review.sort_values(
        ["ticker", "topic", "current_filing_date"]
    ).reset_index(drop=True)

    review["reviewer_id"] = ""
    review["human_event_label"] = ""
    review["human_relation_label"] = ""
    review["subject_company"] = ""
    review["relation_type"] = ""
    review["object_entity"] = ""
    review["review_notes"] = ""

    return review


def build_event_graph(pairs):
    """
    Evidence graph, not a verified supply-chain KG.
    """
    nodes = {}
    edges = []

    def node(node_id, kind, label):
        nodes[node_id] = {
            "node_id": node_id,
            "node_type": kind,
            "label": str(label),
        }

    for ticker in STOCKS:
        node(
            "COMPANY:" + ticker,
            "COMPANY",
            ticker
        )

    for r in pairs.itertuples(index=False):

        c = "COMPANY:" + r.ticker
        d = (
            "DOCUMENT:"
            + r.ticker
            + ":"
            + r.current_accession
        )
        t = "TOPIC:" + r.topic
        e = "EVENT_CANDIDATE:" + r.pair_id

        node(d, "DOCUMENT", r.current_accession)
        node(t, "TOPIC", r.topic)
        node(e, "EVENT_CANDIDATE", r.change_type)

        for source, target, relation in [
            (c, e, "HAS_CANDIDATE"),
            (e, d, "SUPPORTED_BY_DOCUMENT"),
            (e, t, "CONCERNS_TOPIC")
        ]:
            edges.append({
                "edge_id": hash_id(
                    r.pair_id,
                    source,
                    relation,
                    target
                ),
                "source_id": source,
                "target_id": target,
                "relation": relation,
                "ticker": r.ticker,
                "available_from": r.available_from,
                "pair_id": r.pair_id,
                "verified_relation": False,
            })

    return (
        pd.DataFrame(list(nodes.values())),
        pd.DataFrame(edges)
    )


def build():
    OUT.mkdir(parents=True, exist_ok=True)

    filings, evidence, v29 = load_data()

    pairs = align(
        filings, evidence, v29
    )

    review = make_review_sample(
        pairs
    )

    nodes, edges = build_event_graph(
        pairs
    )

    pairs.to_parquet(
        OUT / "evidence_pairs.parquet",
        index=False
    )

    pairs.to_csv(
        OUT / "evidence_pairs.csv",
        index=False
    )

    pairs[
        pairs["change_type"].eq(
            "NEW_DISCLOSURE_CANDIDATE"
        )
    ].to_parquet(
        OUT / "event_candidates.parquet",
        index=False
    )

    review.to_csv(
        OUT / "human_review.csv",
        index=False
    )

    nodes.to_parquet(
        OUT / "event_graph_nodes.parquet",
        index=False
    )

    edges.to_parquet(
        OUT / "event_graph_edges.parquet",
        index=False
    )

    report = {
        "stocks": STOCKS,
        "SEC_filings": len(filings),
        "evidence_pairs": len(pairs),
        "review_sample_size": len(review),
        "change_types": {
            str(k): int(v)
            for k, v in pairs[
                "change_type"
            ].value_counts().items()
        },
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "verified_economic_events": 0,
        "verified_directed_relations": 0,
        "strict_pit_approved": False,
    }

    write_json(
        OUT / "report.json",
        report
    )

    print("\n=== V30 BUILD ===")
    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ))


def audit():
    pairs = pd.read_parquet(
        OUT / "evidence_pairs.parquet"
    )

    nodes = pd.read_parquet(
        OUT / "event_graph_nodes.parquet"
    )

    edges = pd.read_parquet(
        OUT / "event_graph_edges.parquet"
    )

    assert len(pairs) == 21016, (
        "Unexpected evidence count"
    )

    assert set(pairs["ticker"]) == set(STOCKS)
    assert not pairs["pair_id"].duplicated().any()

    assert pairs["source_span_verified"].all()
    assert not pairs["economic_event_verified"].any()
    assert not pairs["directed_relation_verified"].any()

    current = dates(
        pairs["current_filing_date"]
    )
    available = dates(
        pairs["available_from"]
    )

    assert (current < available).all()

    has_prior = pairs[
        "previous_accession"
    ].fillna("").ne("")

    prior_dates = dates(
        pairs.loc[
            has_prior,
            "previous_filing_date"
        ]
    )

    assert (
        prior_dates
        < current.loc[has_prior]
    ).all()

    known_nodes = set(nodes["node_id"])

    assert set(edges["source_id"]).issubset(
        known_nodes
    )

    assert set(edges["target_id"]).issubset(
        known_nodes
    )

    company_nodes = set(
        nodes.loc[
            nodes["node_type"].eq("COMPANY"),
            "node_id"
        ]
    )

    assert company_nodes == {
        "COMPANY:" + ticker
        for ticker in STOCKS
    }

    print("\n=== V30 AUDIT ===")
    print("Evidence count: PASS")
    print("Five-stock universe: PASS")
    print("Prior filing chronology: PASS")
    print("Source evidence lineage: PASS")
    print("Graph integrity: PASS")
    print("Unverified relation protection: PASS")
    print("Strict PIT: NOT COMPLETE")


def evaluate(labels_path):
    """
    Evaluate independent human labels.

    Human event labels:
      TRUE_EVENT
      TEXT_CHANGE_ONLY
      BOILERPLATE
      UNCERTAIN

    Does not automatically approve KG relations.
    """
    labels = pd.read_csv(
        labels_path,
        dtype=str
    ).fillna("")

    required = {
        "pair_id",
        "human_event_label",
        "reviewer_id",
    }

    if not required.issubset(
        labels.columns
    ):
        raise ValueError(
            "Labels missing required columns"
        )

    valid = {
        "TRUE_EVENT",
        "TEXT_CHANGE_ONLY",
        "BOILERPLATE",
        "UNCERTAIN",
    }

    labels["human_event_label"] = (
        labels["human_event_label"]
        .str.upper()
        .str.strip()
    )

    labels = labels[
        labels["human_event_label"].isin(valid)
    ].copy()

    if labels.empty:
        print("No completed human labels yet.")
        return

    pairs = pd.read_parquet(
        OUT / "evidence_pairs.parquet"
    )[
        ["pair_id", "ticker", "topic", "change_type"]
    ]

    if not set(labels["pair_id"]).issubset(
        set(pairs["pair_id"])
    ):
        raise ValueError("Unknown pair_id in labels")

    # Each reviewer labels a pair at most once.
    if labels.duplicated(
        ["pair_id", "reviewer_id"]
    ).any():
        raise ValueError(
            "Duplicate reviewer/pair label"
        )

    joined = labels.merge(
        pairs,
        on="pair_id",
        how="inner",
        validate="many_to_one"
    )

    print("\n=== V30 HUMAN EVALUATION ===")
    print("Labeled rows:", len(joined))
    print(
        joined[
            "human_event_label"
        ].value_counts().to_string()
    )

    candidate = joined[
        joined["change_type"].eq(
            "NEW_DISCLOSURE_CANDIDATE"
        )
        & ~joined["human_event_label"].eq(
            "UNCERTAIN"
        )
    ]

    if len(candidate):
        precision = (
            candidate[
                "human_event_label"
            ].eq("TRUE_EVENT").mean()
        )

        print(
            "Candidate precision in labeled sample:",
            round(float(precision), 4)
        )
        print(
            "This is sample precision, "
            "not population precision."
        )

    # Agreement only where two reviewers
    # labeled the same pair.
    pivot = joined.pivot_table(
        index="pair_id",
        columns="reviewer_id",
        values="human_event_label",
        aggfunc="first"
    )

    if pivot.shape[1] >= 2:
        cols = list(pivot.columns)[:2]
        common = pivot[cols].dropna()

        if len(common):
            agreement = (
                common[cols[0]]
                == common[cols[1]]
            ).mean()
            print(
                "Two-reviewer raw agreement:",
                round(float(agreement), 4)
            )
            print(
                "Shared pairs:",
                len(common)
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "build", "audit",
            "evaluate", "all"
        ]
    )

    parser.add_argument(
        "--labels",
        default=str(
            OUT / "human_review.csv"
        )
    )

    args = parser.parse_args()

    if args.command in ("build", "all"):
        build()

    if args.command in ("audit", "all"):
        audit()

    if args.command == "evaluate":
        evaluate(args.labels)

    if args.command == "all":
        print("\n=== V30 COMPLETE ===")
        print("Portfolio:", STOCKS)
        print("Human verification: REQUIRED")
        print("Strict PIT: NOT COMPLETE")


if __name__ == "__main__":
    main()
