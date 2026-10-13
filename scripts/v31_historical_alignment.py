#!/usr/bin/env python3
"""
V31 Historical SEC Evidence Alignment.

Fixed companies:
 AAPL, MSFT, NVDA, AMZN, GOOGL

Input:
 outputs/v28/filing_features.csv
 outputs/v28/filing_evidence.parquet
 outputs/v29/evidence_changes.parquet
 outputs/v30/evidence_pairs.parquet

Output:
 outputs/v31/

Commands:
 build
 audit
 report
 evaluate --labels path.csv
 selftest
 all

No LLM, no API key, no newly invented events.

V31 is a lexical evidence-alignment system.
It does NOT prove economic events or company relations.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v31"

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

# Parameters fixed before further test analysis.
NEAR_REPEAT = 0.88
RELATED_TEXT = 0.62
WEAK_MATCH = 0.35

MAX_REVIEW_PER_CELL = 3
RANDOM_SEED = 31

VECTORIZER = HashingVectorizer(
    n_features=2**16,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",
)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?<![A-Za-z0-9])"
    r"[$€£]?\s*[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?%?"
    r"(?![A-Za-z])"
)


def write_json(path, data):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def stable_id(*parts):
    raw = "\x1f".join(
        map(str, parts)
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def normalize(text):
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(text).casefold()
        )
    )


def mask_numbers(text):
    return normalize(
        NUMBER_RE.sub(
            " numericvalue ",
            str(text)
        )
    )


def numbers(text):
    return [
        re.sub(r"\s+", "", x)
        for x in NUMBER_RE.findall(
            str(text)
        )
    ]


def core_text(excerpt, matched_term):
    """
    Choose the sentence-like span containing
    the matched topic term.

    V28 excerpt windows remain the provenance
    source. This function only determines which
    part should drive text similarity.
    """
    excerpt = " ".join(
        str(excerpt).split()
    )

    term = str(matched_term)
    lower = excerpt.casefold()
    pos = lower.find(
        term.casefold()
    )

    if pos < 0:
        center = len(excerpt) // 2
    else:
        center = pos

    # Look for nearby sentence boundaries.
    boundaries = [
        m.start()
        for m in re.finditer(
            r"[.;!?]\s+",
            excerpt
        )
    ]

    left = max(
        [
            x + 1
            for x in boundaries
            if x < center
        ],
        default=0
    )

    right = min(
        [
            x + 1
            for x in boundaries
            if x > center
        ],
        default=len(excerpt)
    )

    core = excerpt[left:right].strip()

    # Very short or very long SEC sentence:
    # use a local evidence window instead.
    if (
        len(core.split()) < 10
        or len(core) > 440
    ):
        left = max(
            0, center - 165
        )
        right = min(
            len(excerpt),
            center + 245
        )
        core = excerpt[left:right].strip()

    return core


def load_data():
    fpath = (
        ROOT /
        "outputs/v28/filing_features.csv"
    )

    epath = (
        ROOT /
        "outputs/v28/filing_evidence.parquet"
    )

    p29 = (
        ROOT /
        "outputs/v29/evidence_changes.parquet"
    )

    p30 = (
        ROOT /
        "outputs/v30/evidence_pairs.parquet"
    )

    for path in [
        fpath, epath, p29, p30
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    filings = pd.read_csv(
        fpath,
        dtype={"accession": str}
    )

    evidence = pd.read_parquet(
        epath
    )

    old29 = pd.read_parquet(
        p29
    )

    old30 = pd.read_parquet(
        p30
    )

    for frame in [
        filings, evidence, old29, old30
    ]:
        frame["ticker"] = (
            frame["ticker"]
            .astype(str)
            .str.upper()
        )

    for frame in [
        filings, evidence, old29
    ]:
        frame["accession"] = (
            frame["accession"]
            .astype(str)
        )

    filings["filing_date"] = pd.to_datetime(
        filings["filing_date"],
        errors="raise"
    ).dt.normalize()

    filings["available_from"] = pd.to_datetime(
        filings["available_from"],
        errors="raise"
    ).dt.normalize()

    evidence["available_from"] = pd.to_datetime(
        evidence["available_from"],
        errors="raise"
    ).dt.normalize()

    if len(filings) != 140:
        raise ValueError(
            "Expected 140 SEC filings"
        )

    if set(
        filings["ticker"]
    ) != set(STOCKS):
        raise ValueError(
            "Unexpected research tickers"
        )

    if evidence[
        "evidence_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate evidence IDs"
        )

    if old29[
        "evidence_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate V29 IDs"
        )

    if old30[
        "current_evidence_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate V30 IDs"
        )

    if not set(
        evidence["evidence_id"]
    ) == set(
        old29["evidence_id"]
    ) == set(
        old30["current_evidence_id"]
    ):
        raise ValueError(
            "V28/V29/V30 evidence ID mismatch"
        )

    if not old29[
        "source_span_verified"
    ].astype(bool).all():
        raise ValueError(
            "V29 has unverified SEC excerpts"
        )

    if not (
        filings["filing_date"]
        < filings["available_from"]
    ).all():
        raise ValueError(
            "Invalid available-from date"
        )

    required = [
        "evidence_id",
        "ticker",
        "accession",
        "form",
        "topic",
        "excerpt",
        "matched_term",
        "character_offset",
        "document_sha256",
    ]

    missing = [
        col for col in required
        if col not in evidence
    ]

    if missing:
        raise ValueError(
            f"Missing evidence columns: {missing}"
        )

    meta = filings[
        [
            "ticker",
            "accession",
            "filing_date",
            "available_from"
        ]
    ]

    evidence = evidence.drop(
        columns=[
            "filing_date",
            "available_from"
        ],
        errors="ignore"
    ).merge(
        meta,
        on=[
            "ticker",
            "accession"
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not evidence[
        "_merge"
    ].eq("both").all():
        raise ValueError(
            "Evidence references unknown filing"
        )

    evidence.drop(
        columns="_merge",
        inplace=True
    )

    v29_labels = dict(
        zip(
            old29["evidence_id"],
            old29["label"]
        )
    )

    v30_labels = dict(
        zip(
            old30["current_evidence_id"],
            old30["change_type"]
        )
    )

    return (
        filings,
        evidence,
        v29_labels,
        v30_labels
    )


def prepare_evidence(evidence):
    rows = []

    for r in evidence.itertuples(
        index=False
    ):
        core = core_text(
            r.excerpt,
            r.matched_term
        )

        rows.append({
            "evidence_id": str(
                r.evidence_id
            ),
            "ticker": r.ticker,
            "accession": str(
                r.accession
            ),
            "form": r.form,
            "topic": r.topic,
            "filing_date": r.filing_date,
            "available_from": r.available_from,
            "character_offset": int(
                r.character_offset
            ),
            "document_sha256": str(
                r.document_sha256
            ),
            "matched_term": str(
                r.matched_term
            ),
            "current_excerpt": str(
                r.excerpt
            ),
            "current_core": core,
            "core_normalized": normalize(core),
            "core_number_masked": mask_numbers(core),
            "core_numbers": numbers(core),
        })

    return pd.DataFrame(rows)


def classification(
    current,
    prior,
    sim,
    is_immediate,
    has_history
):
    """
    Classification is conservative.
    Historical matches override naive novelty.

    Each match is a textual match, not proof
    of a real-world relationship.
    """

    if not has_history:
        return "NO_PRIOR_TOPIC_EVIDENCE"

    if prior is None:
        return "NO_COMPARABLE_TEXT"

    exact = (
        current["core_normalized"]
        == prior["core_normalized"]
    )

    number_mask_exact = (
        current["core_number_masked"]
        == prior["core_number_masked"]
    )

    if exact:
        return (
            "PREVIOUS_EXACT_REPEAT"
            if is_immediate
            else "HISTORICAL_EXACT_REPEAT"
        )

    if number_mask_exact:
        return (
            "PREVIOUS_NUMERIC_OR_FORMAT_CHANGE"
            if is_immediate
            else "HISTORICAL_NUMERIC_OR_FORMAT_CHANGE"
        )

    if sim >= NEAR_REPEAT:
        return (
            "PREVIOUS_NEAR_REPEAT"
            if is_immediate
            else "HISTORICAL_NEAR_REPEAT"
        )

    if sim >= RELATED_TEXT:
        return "RELATED_WORDING_CHANGE"

    if sim >= WEAK_MATCH:
        return "WEAK_HISTORICAL_MATCH"

    return "NEW_TEXT_CANDIDATE"


def align_historical(prepared):
    """
    Within each ticker/form/topic:
      - Process filings in chronological order.
      - Only earlier filing evidence is available.
      - Compare to immediately preceding filing.
      - Compare to ALL older same-form filings.
      - Never compare snippets inside same filing
        against one another as history.

    HashingVectorizer is stateless:
      no future-fitted vocabulary / IDF.
    """

    results = []

    for (ticker, form, topic), group in (
        prepared.groupby(
            ["ticker", "form", "topic"],
            sort=True
        )
    ):

        group = group.sort_values(
            [
                "filing_date",
                "accession",
                "character_offset",
            ]
        )

        history = []
        history_vectors = None

        # Exact canonical maps allow earlier
        # matches across all historical filings.
        exact_index = defaultdict(list)
        masked_index = defaultdict(list)

        # Group current snippets by their filing.
        for accession, batch in group.groupby(
            "accession",
            sort=False
        ):

            batch_rows = batch.to_dict(
                "records"
            )

            date = batch_rows[0][
                "filing_date"
            ]

            # Reject duplicate/same-date historical
            # sources when ordering is ambiguous.
            accessible_indices = [
                i for i, past in enumerate(
                    history
                )
                if past["filing_date"] < date
            ]

            immediate_accession = ""

            if accessible_indices:
                prior_filing_date = max(
                    history[i]["filing_date"]
                    for i in accessible_indices
                )

                immediate = [
                    history[i]
                    for i in accessible_indices
                    if history[i]["filing_date"]
                    == prior_filing_date
                ]

                immediate_accession = (
                    immediate[-1]["accession"]
                )

            for row in batch_rows:

                curr_norm = row[
                    "core_normalized"
                ]

                curr_masked = row[
                    "core_number_masked"
                ]

                best_index = None
                best_score = 0.0
                match_method = (
                    "NO_PRIOR"
                )

                # Strict earlier-than-current source.
                exact_hits = [
                    i
                    for i in exact_index.get(
                        curr_norm, []
                    )
                    if history[i][
                        "filing_date"
                    ] < date
                ]

                if exact_hits:
                    best_index = max(
                        exact_hits,
                        key=lambda i: (
                            history[i][
                                "filing_date"
                            ],
                            history[i][
                                "accession"
                            ],
                        )
                    )
                    best_score = 1.0
                    match_method = (
                        "EXACT_HISTORY"
                    )

                else:
                    masked_hits = [
                        i
                        for i in masked_index.get(
                            curr_masked, []
                        )
                        if history[i][
                            "filing_date"
                        ] < date
                    ]

                    if masked_hits:
                        best_index = max(
                            masked_hits,
                            key=lambda i: (
                                history[i][
                                    "filing_date"
                                ],
                                history[i][
                                    "accession"
                                ],
                            )
                        )
                        best_score = 1.0
                        match_method = (
                            "NUMBER_MASK_HISTORY"
                        )

                    elif accessible_indices:
                        query = VECTORIZER.transform(
                            [curr_masked]
                        )

                        prior_vectors = (
                            history_vectors[
                                accessible_indices
                            ]
                        )

                        scores = (
                            prior_vectors @ query.T
                        ).toarray().ravel()

                        candidate = int(
                            np.argmax(scores)
                        )

                        best_index = (
                            accessible_indices[
                                candidate
                            ]
                        )

                        best_score = float(
                            scores[candidate]
                        )

                        match_method = (
                            "HASHING_COSINE"
                        )

                prior = (
                    history[best_index]
                    if best_index is not None
                    else None
                )

                is_immediate = (
                    prior is not None
                    and prior["accession"]
                    == immediate_accession
                )

                label = classification(
                    row,
                    prior,
                    best_score,
                    is_immediate,
                    bool(accessible_indices)
                )

                if prior is not None:
                    if not (
                        prior["filing_date"]
                        < date
                    ):
                        raise RuntimeError(
                            "Future evidence matched"
                        )

                results.append({
                    "evidence_id": row[
                        "evidence_id"
                    ],
                    "pair_id": stable_id(
                        "V31",
                        row["evidence_id"]
                    ),
                    "ticker": ticker,
                    "form": form,
                    "topic": topic,
                    "current_accession": accession,
                    "current_filing_date": date,
                    "available_from": row[
                        "available_from"
                    ],
                    "current_offset": row[
                        "character_offset"
                    ],
                    "current_excerpt": row[
                        "current_excerpt"
                    ],
                    "current_core": row[
                        "current_core"
                    ],
                    "current_document_sha256": row[
                        "document_sha256"
                    ],
                    "best_prior_evidence_id": (
                        prior["evidence_id"]
                        if prior else ""
                    ),
                    "best_prior_accession": (
                        prior["accession"]
                        if prior else ""
                    ),
                    "best_prior_filing_date": (
                        prior["filing_date"]
                        if prior else pd.NaT
                    ),
                    "best_prior_offset": (
                        prior["character_offset"]
                        if prior else -1
                    ),
                    "best_prior_excerpt": (
                        prior["current_excerpt"]
                        if prior else ""
                    ),
                    "best_prior_core": (
                        prior["current_core"]
                        if prior else ""
                    ),
                    "best_prior_document_sha256": (
                        prior["document_sha256"]
                        if prior else ""
                    ),
                    "prior_is_immediate": bool(
                        is_immediate
                    ),
                    "has_historical_topic_evidence": bool(
                        accessible_indices
                    ),
                    "similarity": round(
                        best_score, 8
                    ),
                    "match_method": match_method,
                    "v31_label": label,
                    "numeric_values_different": bool(
                        prior is not None
                        and row["core_numbers"]
                        != prior["core_numbers"]
                    ),
                    "historical_match": bool(
                        prior is not None
                    ),
                    "source_span_verified_v29": True,
                    "economic_event_verified": False,
                    "directed_relation_verified": False,
                })

            # Make this filing available only
            # after processing its entire batch.
            new_vectors = VECTORIZER.transform(
                [
                    r["core_number_masked"]
                    for r in batch_rows
                ]
            )

            history_vectors = (
                new_vectors
                if history_vectors is None
                else vstack(
                    [
                        history_vectors,
                        new_vectors
                    ],
                    format="csr"
                )
            )

            offset = len(history)

            for j, item in enumerate(
                batch_rows
            ):
                history.append(item)

                exact_index[
                    item["core_normalized"]
                ].append(offset + j)

                masked_index[
                    item["core_number_masked"]
                ].append(offset + j)

    result = pd.DataFrame(
        results
    )

    if result[
        "evidence_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate output evidence IDs"
        )

    return result


def add_comparisons(
    results,
    v29_labels,
    v30_labels
):
    results = results.copy()

    results["v29_label"] = (
        results["evidence_id"]
        .map(v29_labels)
    )

    results["v30_label"] = (
        results["evidence_id"]
        .map(v30_labels)
    )

    if results[
        [
            "v29_label",
            "v30_label"
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing previous model classification"
        )

    prior_conflict = results[
        "v30_label"
    ].eq(
        "NEW_DISCLOSURE_CANDIDATE"
    ) & results[
        "v29_label"
    ].isin(
        [
            "EXACT_REPEAT",
            "BOILERPLATE_SIMILAR",
            "NUMERIC_OR_FORMAT_CHANGE",
        ]
    )

    results[
        "v29_v30_conflict"
    ] = prior_conflict

    results[
        "v31_resolved_as_historical"
    ] = (
        prior_conflict
        & results[
            "v31_label"
        ].str.contains(
            "HISTORICAL_|PREVIOUS_",
            regex=True
        )
    )

    results[
        "is_new_text_candidate"
    ] = (
        results["v31_label"]
        == "NEW_TEXT_CANDIDATE"
    )

    return results


def make_review(results):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selections = []

    for _, group in results.groupby(
        [
            "ticker",
            "topic",
            "v31_label"
        ],
        sort=True
    ):
        count = min(
            MAX_REVIEW_PER_CELL,
            len(group)
        )

        indices = rng.choice(
            group.index.to_numpy(),
            size=count,
            replace=False
        )

        selections.append(
            results.loc[indices]
        )

    # Supplement with historical disagreements.
    conflict = results[
        results[
            "v29_v30_conflict"
        ]
    ]

    if len(conflict):
        n = min(
            100,
            len(conflict)
        )

        sampled = rng.choice(
            conflict.index.to_numpy(),
            n,
            replace=False
        )

        selections.append(
            results.loc[sampled]
        )

    review = pd.concat(
        selections,
        ignore_index=True
    ).drop_duplicates(
        "pair_id"
    )

    review = review.sort_values(
        [
            "ticker",
            "topic",
            "current_filing_date"
        ]
    ).reset_index(drop=True)

    review["reviewer_id"] = ""
    review["human_event_label"] = ""
    review["human_relation_label"] = ""
    review["subject_company"] = ""
    review["relation_type"] = ""
    review["object_entity"] = ""
    review["review_notes"] = ""

    return review


def build():
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        filings,
        evidence,
        v29_labels,
        v30_labels
    ) = load_data()

    prepared = prepare_evidence(
        evidence
    )

    results = align_historical(
        prepared
    )

    if len(results) != len(evidence):
        raise ValueError(
            "Not all evidence was aligned"
        )

    results = add_comparisons(
        results,
        v29_labels,
        v30_labels
    )

    results.to_parquet(
        OUT / "evidence_alignment.parquet",
        index=False
    )

    results.to_csv(
        OUT / "evidence_alignment.csv",
        index=False
    )

    conflicts = results[
        results[
            "v29_v30_conflict"
        ]
    ]

    conflicts.to_csv(
        OUT / "historical_conflicts.csv",
        index=False
    )

    candidates = results[
        results[
            "is_new_text_candidate"
        ]
    ]

    candidates.to_csv(
        OUT / "new_text_candidates.csv",
        index=False
    )

    review = make_review(
        results
    )

    # Never overwrite completed user annotation.
    review_path = (
        OUT / "human_review_template.csv"
    )

    if not review_path.exists():
        review.to_csv(
            review_path,
            index=False
        )
    else:
        print(
            "Existing human review template "
            "preserved."
        )

    summary = {
        "stocks": STOCKS,
        "filings": len(filings),
        "total_evidence": len(results),
        "v29_candidates": int(
            results["v29_label"]
            .eq(
                "SUBSTANTIVE_TEXT_CANDIDATE"
            ).sum()
        ),
        "v30_candidates": int(
            results["v30_label"]
            .eq(
                "NEW_DISCLOSURE_CANDIDATE"
            ).sum()
        ),
        "v31_new_text_candidates": int(
            results[
                "is_new_text_candidate"
            ].sum()
        ),
        "v29_v30_conflicts": int(
            results[
                "v29_v30_conflict"
            ].sum()
        ),
        "historical_conflicts_resolved": int(
            results[
                "v31_resolved_as_historical"
            ].sum()
        ),
        "labels": {
            str(k): int(v)
            for k, v in results[
                "v31_label"
            ].value_counts().items()
        },
        "human_review_sample": len(review),
        "strict_pit_approved": False,
        "economic_events_verified": 0,
        "directed_relations_verified": 0,
    }

    write_json(
        OUT / "report.json",
        summary
    )

    print("\n=== V31 BUILD ===")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ))


def audit():
    df = pd.read_parquet(
        OUT / "evidence_alignment.parquet"
    )

    if len(df) != 21016:
        raise ValueError(
            "Expected 21016 SEC evidence spans"
        )

    if set(
        df["ticker"]
    ) != set(STOCKS):
        raise ValueError(
            "Invalid company universe"
        )

    if df[
        "evidence_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicated source evidence"
        )

    current = pd.to_datetime(
        df["current_filing_date"]
    )

    available = pd.to_datetime(
        df["available_from"]
    )

    if not (
        current < available
    ).all():
        raise ValueError(
            "Bad availability chronology"
        )

    prior = pd.to_datetime(
        df["best_prior_filing_date"]
    )

    matched = df[
        "historical_match"
    ].astype(bool)

    if not (
        prior[matched]
        < current[matched]
    ).all():
        raise ValueError(
            "Future historical match"
        )

    if (
        df["economic_event_verified"]
        .astype(bool).any()
    ):
        raise ValueError(
            "Automatic event approval detected"
        )

    if (
        df["directed_relation_verified"]
        .astype(bool).any()
    ):
        raise ValueError(
            "Automatic directed relation "
            "approval detected"
        )

    if not df[
        "source_span_verified_v29"
    ].astype(bool).all():
        raise ValueError(
            "Unverified source span"
        )

    report = json.loads(
        (OUT / "report.json")
        .read_text(
            encoding="utf-8"
        )
    )

    if int(
        df["v29_v30_conflict"]
        .sum()
    ) != report[
        "v29_v30_conflicts"
    ]:
        raise ValueError(
            "Conflict report mismatch"
        )

    print("\n=== V31 AUDIT ===")
    print("Fixed five companies: PASS")
    print("Evidence lineage: PASS")
    print("Historical date order: PASS")
    print("No future matched filings: PASS")
    print("No automatic event approval: PASS")
    print("Conflict accounting: PASS")
    print(
        "Strict PIT approval: NOT COMPLETE"
    )


def report():
    path = (
        OUT / "evidence_alignment.parquet"
    )

    df = pd.read_parquet(
        path
    )

    print("\n=== V31 LABEL DISTRIBUTION ===")
    print(
        df["v31_label"]
        .value_counts()
        .to_string()
    )

    print("\n=== V29 / V30 / V31 ===")

    for label, name in [
        (
            "v29_label",
            "SUBSTANTIVE_TEXT_CANDIDATE"
        ),
        (
            "v30_label",
            "NEW_DISCLOSURE_CANDIDATE"
        ),
        (
            "v31_label",
            "NEW_TEXT_CANDIDATE"
        ),
    ]:
        print(
            label,
            int(
                df[label]
                .eq(name)
                .sum()
            )
        )

    print("\n=== HISTORICAL CONFLICTS ===")
    print(
        "V29/V30 conflicts:",
        int(
            df[
                "v29_v30_conflict"
            ].sum()
        )
    )

    print(
        "Resolved into repeat/numeric categories:",
        int(
            df[
                "v31_resolved_as_historical"
            ].sum()
        )
    )

    print("\n=== BY TICKER ===")
    print(
        pd.crosstab(
            df["ticker"],
            df["v31_label"]
        ).to_string()
    )

    print("\n=== V30 TO V31 ===")
    print(
        pd.crosstab(
            df["v30_label"],
            df["v31_label"]
        ).to_string()
    )

    print("\n=== ZERO SIMILARITY ===")
    candidates = df[
        df["v31_label"].eq(
            "NEW_TEXT_CANDIDATE"
        )
    ]

    print(
        "V31 candidates:",
        len(candidates)
    )

    print(
        "V31 zero similarity:",
        int(
            candidates[
                "similarity"
            ].eq(0).sum()
        )
    )

    print(
        "\nVerified economic events: 0"
    )

    print(
        "Strict PIT: NOT COMPLETE"
    )


def evaluate(path):
    labels = pd.read_csv(
        path,
        dtype=str
    ).fillna("")

    if not {
        "pair_id",
        "reviewer_id",
        "human_event_label"
    }.issubset(labels.columns):
        raise ValueError(
            "Missing review columns"
        )

    valid = {
        "TRUE_EVENT",
        "TEXT_CHANGE_ONLY",
        "BOILERPLATE",
        "UNCERTAIN",
    }

    labels[
        "human_event_label"
    ] = (
        labels[
            "human_event_label"
        ].str.upper()
        .str.strip()
    )

    labels = labels[
        labels[
            "human_event_label"
        ].isin(valid)
    ].copy()

    if labels.empty:
        print(
            "No human labels available."
        )
        return

    if labels.duplicated(
        [
            "pair_id",
            "reviewer_id"
        ]
    ).any():
        raise ValueError(
            "Duplicate review submission"
        )

    df = pd.read_parquet(
        OUT / "evidence_alignment.parquet"
    )

    if not set(
        labels["pair_id"]
    ).issubset(
        set(df["pair_id"])
    ):
        raise ValueError(
            "Unknown review pair ID"
        )

    joined = labels.merge(
        df[
            [
                "pair_id",
                "ticker",
                "topic",
                "v31_label"
            ]
        ],
        on="pair_id",
        how="inner",
        validate="many_to_one"
    )

    selected = joined[
        joined["v31_label"].eq(
            "NEW_TEXT_CANDIDATE"
        )
        & ~joined[
            "human_event_label"
        ].eq("UNCERTAIN")
    ]

    print(
        "\n=== V31 HUMAN REVIEW ==="
    )

    print(
        "Labeled entries:",
        len(joined)
    )

    if len(selected):
        score = (
            selected[
                "human_event_label"
            ].eq("TRUE_EVENT")
        ).mean()

        print(
            "Candidate event precision "
            "within labeled sample:",
            round(float(score), 4)
        )

        print(
            "Not population-wide precision."
        )

    else:
        print(
            "No non-uncertain labels "
            "on V31 candidates."
        )


def selftest():
    a = {
        "core_normalized": normalize(
            "Supplier cost was $100 million."
        ),
        "core_number_masked": mask_numbers(
            "Supplier cost was $100 million."
        ),
    }

    b = {
        "core_normalized": normalize(
            "Supplier cost was $130 million."
        ),
        "core_number_masked": mask_numbers(
            "Supplier cost was $130 million."
        ),
    }

    assert a[
        "core_normalized"
    ] != b[
        "core_normalized"
    ]

    assert a[
        "core_number_masked"
    ] == b[
        "core_number_masked"
    ]

    assert classification(
        a, a, 1.0,
        True, True
    ) == "PREVIOUS_EXACT_REPEAT"

    assert classification(
        a, a, 1.0,
        False, True
    ) == "HISTORICAL_EXACT_REPEAT"

    assert classification(
        b, a, 1.0,
        False, True
    ) == (
        "HISTORICAL_NUMERIC_OR_FORMAT_CHANGE"
    )

    assert classification(
        a, None, 0,
        False, False
    ) == "NO_PRIOR_TOPIC_EVIDENCE"

    print(
        "V31 CLASSIFICATION SELFTEST: PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "build",
            "audit",
            "report",
            "evaluate",
            "selftest",
            "all",
        ]
    )

    parser.add_argument(
        "--labels",
        default=str(
            OUT /
            "human_review_template.csv"
        )
    )

    args = parser.parse_args()

    if args.command in (
        "selftest",
        "all"
    ):
        selftest()

    if args.command in (
        "build",
        "all"
    ):
        build()

    if args.command in (
        "audit",
        "all"
    ):
        audit()

    if args.command in (
        "report",
        "all"
    ):
        report()

    if args.command == "evaluate":
        evaluate(
            args.labels
        )

    if args.command == "all":
        print(
            "\n=== V31 PIPELINE COMPLETE ==="
        )
        print(
            "Stocks:",
            STOCKS
        )
        print(
            "Strict PIT: NOT COMPLETE"
        )


if __name__ == "__main__":
    main()
