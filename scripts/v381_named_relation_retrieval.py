#!/usr/bin/env python3
"""
V38.1: Named-company SEC relation evidence retrieval.

Does not generate KG edges or claim LLM verification.

Inputs:
  outputs/v38/full_sec_relation_evidence.csv
  outputs/v34/entity_aliases.csv

Outputs:
  outputs/v381/named_company_candidates.csv
  outputs/v381/relation_diagnostics.csv
  outputs/v381/summary.json
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v381"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ROOT / "outputs/v38/full_sec_relation_evidence.csv"
ALIASES = ROOT / "outputs/v34/entity_aliases.csv"

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

SUPPLY = re.compile(
    r"\b(?:suppl(?:y|ies|ied|ier|iers)|"
    r"provid(?:e|es|ed|ing)|manufactur(?:e|es|ed|ing)|"
    r"purchas(?:e|es|ed|ing)|procure(?:d|ment)?|"
    r"source(?:s|d)?\s+from|buy\s+from|bought\s+from|"
    r"sells?\s+to|sold\s+to)\b",
    re.I,
)

COMP = re.compile(
    r"\b(?:competitor(?:s)?|compete(?:s|d)?|"
    r"competing|competition|rival(?:s)?)\b",
    re.I,
)

SECTOR = re.compile(
    r"\b(?:same\s+industry|same\s+sector|"
    r"industry\s+peers?|sector\s+peers?)\b",
    re.I,
)

BAD = re.compile(
    r"\b(?:equity awards?|stock[- ]based compensation|"
    r"shares purchased|weighted average|income taxes|"
    r"forward[- ]looking statements)\b",
    re.I,
)

def load_aliases():
    df = pd.read_csv(ALIASES).fillna("")
    mapping = {}

    if not {"canonical", "alias"}.issubset(df.columns):
        raise RuntimeError("Unexpected alias schema")

    for row in df.itertuples(index=False):
        canonical = str(row.canonical).strip()
        alias = str(row.alias).strip()

        if not canonical or not alias:
            continue

        # Skip aliases too ambiguous for standalone matching.
        if len(alias) < 4:
            continue

        mapping.setdefault(canonical, set()).add(alias)

    for canonical in ISSUERS.values():
        mapping.setdefault(canonical, set()).add(canonical)

    return {
        k: sorted(v, key=len, reverse=True)
        for k, v in mapping.items()
    }

def find_targets(text, issuer, alias_dict):
    targets = []

    for canonical, names in alias_dict.items():
        if canonical == issuer:
            continue

        hits = []

        for alias in names:
            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(alias)
                + r"(?![A-Za-z0-9])"
            )

            for match in re.finditer(pattern, text, re.I):
                hits.append({
                    "alias": alias,
                    "start": match.start(),
                    "end": match.end(),
                })

        if hits:
            targets.append({
                "target": canonical,
                "matches": sorted(
                    hits, key=lambda x: x["start"]
                ),
            })

    return targets

def text_quality(sentence, context):
    flags = []

    if len(sentence) < 30:
        flags.append("SHORT_SENTENCE")

    if len(sentence) > 2500:
        flags.append("LONG_PARSER_ARTIFACT")

    if BAD.search(sentence):
        flags.append("ACCOUNTING_OR_BOILERPLATE")

    if len(context) > 6500:
        flags.append("LONG_CONTEXT")

    return flags

def relation_hints(sentence, context):
    # Sentence-level cues receive stronger priority.
    found = []

    if SUPPLY.search(sentence):
        found.extend(["SUPPLIES_TO", "CUSTOMER_OF"])

    if COMP.search(sentence):
        found.append("COMPETES_WITH")

    if SECTOR.search(sentence):
        found.append("SAME_SECTOR")

    # Context may help recall, but must not independently
    # establish a relationship.
    return sorted(set(found))

def score_candidate(sentence, context, matches, relation):
    score = 0.0

    if relation == "COMPETES_WITH":
        if re.search(
            r"\b(?:our competitors include|"
            r"our current competitors include|"
            r"competitors to .{1,70}? include|"
            r"we compete with)\b",
            sentence, re.I
        ):
            score += 5

    if relation in {"SUPPLIES_TO", "CUSTOMER_OF"}:
        if re.search(
            r"\b(?:we|our company)\b",
            sentence, re.I
        ):
            score += 2

        if re.search(
            r"\b(?:from|to|by)\b",
            sentence, re.I
        ):
            score += 2

    if relation == "SAME_SECTOR":
        if SECTOR.search(sentence):
            score += 3

    if matches:
        first = matches[0]
        # Target mention near the middle of evidence is
        # a weak retrieval heuristic, NOT proof.
        score += 1.0

        if first["start"] < len(sentence):
            score += 0.5

    if 80 <= len(sentence) <= 1200:
        score += 1

    return score

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = pd.read_csv(
        SOURCE,
        dtype={"accession": str},
    ).fillna("")

    aliases = load_aliases()
    candidates = []
    diagnostics = []

    for r in source.to_dict("records"):
        ticker = str(r["ticker"])

        if ticker not in ISSUERS:
            continue

        issuer = ISSUERS[ticker]
        sentence = str(r["sentence"])
        context = str(r["context"])

        targets_sentence = find_targets(
            sentence, issuer, aliases
        )

        targets_context = find_targets(
            context, issuer, aliases
        )

        flags = text_quality(sentence, context)
        hints = relation_hints(sentence, context)

        diagnostics.append({
            "evidence_id": r["evidence_id"],
            "ticker": ticker,
            "relation_hint_count": len(hints),
            "named_targets_sentence": len(targets_sentence),
            "named_targets_context": len(targets_context),
            "quality_flags": "|".join(flags),
        })

        # Critical: a second NAMED business must occur
        # in the sentence. Context-only mentions are not
        # enough to create a directed candidate.
        if not targets_sentence:
            continue

        for target_info in targets_sentence:
            target = target_info["target"]
            matches = target_info["matches"]

            for relation in hints:
                # SAME_SECTOR must pass explicit statement
                # and must not be contaminated by accounting.
                if relation == "SAME_SECTOR":
                    if (
                        not SECTOR.search(sentence)
                        or BAD.search(sentence)
                    ):
                        continue

                score = score_candidate(
                    sentence,
                    context,
                    matches,
                    relation,
                )

                cid = hashlib.sha256(
                    (
                        str(r["evidence_id"])
                        + "|"
                        + issuer
                        + "|"
                        + relation
                        + "|"
                        + target
                    ).encode()
                ).hexdigest()[:24]

                candidates.append({
                    "candidate_id": cid,
                    "evidence_id": r["evidence_id"],
                    "ticker": ticker,
                    "issuer": issuer,
                    "target": target,
                    "relation_hint": relation,
                    "matched_alias": matches[0]["alias"],
                    "sentence": sentence,
                    "context": context,
                    "accession": r["accession"],
                    "filing_date": r["filing_date"],
                    "available_from": r["available_from"],
                    "source_sha256": r["source_sha256"],
                    "offset_start": r["offset_start"],
                    "offset_end": r["offset_end"],
                    "retrieval_score": score,
                    "quality_flags": "|".join(flags),
                    "is_verified_relation": False,
                })

    candidate_df = pd.DataFrame(candidates)

    if not candidate_df.empty:
        candidate_df = (
            candidate_df
            .drop_duplicates("candidate_id")
            .sort_values(
                ["retrieval_score", "ticker"],
                ascending=[False, True],
            )
        )

    diagnostic_df = pd.DataFrame(diagnostics)

    candidate_df.to_csv(
        OUT / "named_company_candidates.csv",
        index=False
    )

    diagnostic_df.to_csv(
        OUT / "relation_diagnostics.csv",
        index=False
    )

    counts = (
        candidate_df["relation_hint"]
        .value_counts()
        .reindex(RELATIONS, fill_value=0)
        .to_dict()
        if not candidate_df.empty
        else {r: 0 for r in RELATIONS}
    )

    report = {
        "v38_evidence": len(source),
        "named_company_candidates": len(candidate_df),
        "unique_evidence_with_named_targets": (
            candidate_df["evidence_id"].nunique()
            if not candidate_df.empty else 0
        ),
        "relation_candidate_counts": counts,
        "verified_relations": 0,
        "strict_pit_approved": False,
        "human_verified": False,
        "note": (
            "Named-company and keyword overlap is a "
            "candidate only; direction and semantic "
            "relationship still require validation."
        ),
    }

    (OUT / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n=== V38.1 NAMED COMPANY RETRIEVAL ===")
    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))

    if not candidate_df.empty:
        print("\n=== TOP NAMED CANDIDATES ===")
        print(
            candidate_df[[
                "ticker",
                "target",
                "relation_hint",
                "retrieval_score",
            ]]
            .head(25)
            .to_string(index=False)
        )

    print("\n=== QUALITY DIAGNOSTICS ===")
    print(
        diagnostic_df[[
            "named_targets_sentence",
            "named_targets_context",
        ]].describe().to_string()
    )

if __name__ == "__main__":
    main()
