#!/usr/bin/env python3
"""
V37 - Four-relation SEC-RAG Financial KG

Pipeline:
    build   : four-relation candidates from V34 evidence
    audit   : candidate provenance audit
    label   : Ollama 27B verification, resumable
    graph   : verified silver temporal graph + daily features
    wire    : connect V37 data to existing V36 FinRL runner
    report  : summarize extraction and graph

No fabricated relation edges.
No assumed human verification.
No claim of strict PIT certification.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v37"
SRC = ROOT / "outputs/v34/relation_evidence.csv"
ALIAS_FILE = ROOT / "outputs/v34/entity_aliases.csv"
V36_SCRIPT = ROOT / "scripts/v36_finrl_kg_ablation.py"

ISSUERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

TICKERS = list(ISSUERS)
CANONICAL_TICKERS = {v: k for k, v in ISSUERS.items()}

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

# Relation keywords are used ONLY for candidate retrieval.
# They are NOT sufficient proof of a relationship.
PATTERNS = {
    "SUPPLIES_TO": [
        r"\bsuppl(?:y|ies|ied|ier|iers)\b",
        r"\bprovide(?:s|d)?\b",
        r"\bmanufactur(?:e|es|ed|ing)\b",
        r"\bcomponents?\b",
        r"\bfoundr(?:y|ies)\b",
        r"\bcontract manufacturing\b",
        r"\bsource(?:s|d)? from\b",
    ],
    "CUSTOMER_OF": [
        r"\bcustomers?\b",
        r"\bclients?\b",
        r"\bpurchas(?:e|es|ed|ing)\b",
        r"\bprocure(?:d|ment)?\b",
        r"\bsales to\b",
        r"\brevenue from\b",
        r"\blicens(?:e|es|ed|ing)\b",
    ],
    "COMPETES_WITH": [
        r"\bcompetitors?\b",
        r"\bcompetes?\b",
        r"\bcompetitive\b",
        r"\brivals?\b",
    ],
    "SAME_SECTOR": [
        r"\bsame (?:industry|sector|market)\b",
        r"\b(?:industry|sector) peers?\b",
        r"\bpeer companies\b",
        r"\boperate in the .*? (?:industry|sector)\b",
        r"\b(?:companies|firms) in the .*? (?:industry|sector)\b",
    ],
}

COMPETITOR_LIST = re.compile(
    r"\b(?:our\s+(?:current\s+)?competitors\s+include"
    r"|competitors\s+to\s+.{1,70}?\s+include"
    r"|we\s+compete\s+with)\b",
    re.I,
)

BLOCKED_HEADINGS = re.compile(
    r"\b(?:Table of Contents|Item 1A|Patents and "
    r"Proprietary Rights|Forward-Looking Statements)\b",
    re.I,
)

KG11 = [
    "kg_known_competitors",
    "kg_competitor_mentions",
    "kg_recent_mentions_30d",
    "kg_recent_mentions_90d",
    "kg_new_edges_30d",
    "kg_new_edges_90d",
    "kg_days_since_last_disclosure",
    "kg_days_since_first_disclosure",
    "kg_portfolio_competitors",
    "kg_external_competitors",
    "kg_competitor_remention_ratio",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence_quote": {"type": "string"},
                    "reason": {"type": "string"},
                    "market_scope": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "supported",
                    "evidence_quote",
                    "reason",
                    "market_scope",
                ],
            },
        },
    },
    "required": ["decisions"],
}

SYSTEM = """
You are a conservative financial relation verifier
using historical SEC filings.

The issuer is the SEC reporting company.
"We", "our" and "us" normally refer to that issuer.

Verify each proposed directed relation separately.

Allowed relations:
SUPPLIES_TO:
  Subject supplies products or services TO object.
CUSTOMER_OF:
  Subject purchases or receives products or services FROM object.
COMPETES_WITH:
  Explicit issuer competitor lists support competition.
  Product-scoped competitor lists are also acceptable.
SAME_SECTOR:
  Explicit shared industry or sector classification.
  Merely competing or appearing in the same sentence
  does NOT prove SAME_SECTOR.

IMPORTANT:
- Named subject and named object must be supported.
- Generic suppliers and customers are NOT named entities.
- Do not infer suppliers from a list of competitors who
  happen to manufacture or supply products.
- Do not use external knowledge.
- Do not guess the customer-supplier direction.
- Risk factors about hypothetical future relationships
  do not prove an existing relationship.
- A competitor list is not evidence of a supply relationship.
- In quotes, use only exact contiguous substrings of
  CURRENT_SEC_SENTENCE.
- A supported quotation should contain the named target
  and a meaningful relation statement.
- If not supported, return false with an empty quote.
- Return one decision for every candidate ID.
- Never invent candidate IDs.
- Output JSON only.
"""


def stop(message):
    raise RuntimeError("V37 STOP: " + str(message))


def uid(*parts):
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode()
    ).hexdigest()[:24]


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_aliases():
    raw = pd.read_csv(ALIAS_FILE).fillna("")
    aliases = {}

    for r in raw.itertuples(index=False):
        canonical = str(r.canonical).strip()
        alias = str(r.alias).strip()

        if canonical and alias:
            aliases.setdefault(canonical, set()).add(alias)

    for ticker, canonical in ISSUERS.items():
        aliases.setdefault(canonical, set()).add(canonical)

    return {
        canonical: sorted(names, key=len, reverse=True)
        for canonical, names in aliases.items()
    }


def alias_spans(text, names):
    spans = []

    for name in names:
        pat = re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(name)
            + r"(?![A-Za-z0-9])",
            re.I,
        )

        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), name))

    return sorted(spans)


def relation_cues(sentence):
    matches = {}

    for relation, patterns in PATTERNS.items():
        hits = []

        for pattern in patterns:
            for match in re.finditer(
                pattern, sentence, re.I
            ):
                hits.append(
                    (match.start(), match.end(), match.group())
                )

        if hits:
            matches[relation] = sorted(hits)

    return matches


def load_source():
    df = pd.read_csv(
        SRC, dtype={"accession": str}
    ).fillna("")

    if df.empty:
        stop("V34 evidence is empty")

    if df["evidence_id"].duplicated().any():
        stop("Duplicate V34 evidence IDs")

    if not set(df["ticker"]).issubset(ISSUERS):
        stop("Unexpected SEC issuer")

    return df


def candidate_cols():
    return [
        "candidate_id",
        "evidence_id",
        "ticker",
        "issuer",
        "subject",
        "relation",
        "object",
        "target_alias",
        "sentence",
        "context",
        "cue",
        "accession",
        "filing_date",
        "available_from",
        "source_sha256",
        "offset_start",
        "offset_end",
        "retrieval_score",
        "candidate_method",
    ]


def build(args):
    OUT.mkdir(parents=True, exist_ok=True)

    src = load_source()
    aliases = load_aliases()

    records = []
    seen = set()

    for evidence in src.to_dict("records"):
        ticker = evidence["ticker"]
        issuer = ISSUERS[ticker]
        sentence = str(evidence["sentence"])
        cues = relation_cues(sentence)

        if not cues:
            continue

        # Restrict to actual named target companies.
        targets = []

        for canonical, names in aliases.items():
            if canonical == issuer:
                continue

            spans = alias_spans(sentence, names)

            if spans:
                start, end, name = spans[0]
                targets.append(
                    (canonical, name, start, end)
                )

        if not targets:
            continue

        explicit_competition = (
            COMPETITOR_LIST.search(sentence)
        )

        for relation, hits in cues.items():

            # Generic competition adjectives do not prove edges.
            # For competition, require a recognisable
            # issuer-oriented competitor-list construction.
            if (
                relation == "COMPETES_WITH"
                and explicit_competition is None
            ):
                continue

            # A competitor list containing the word "suppliers"
            # does not generate customer or supply candidates.
            if (
                explicit_competition is not None
                and relation in {
                    "SUPPLIES_TO",
                    "CUSTOMER_OF",
                    "SAME_SECTOR",
                }
            ):
                continue

            # SAME_SECTOR requires explicit sector phrasing.
            if relation == "SAME_SECTOR":
                if not re.search(
                    r"\b(?:same|peers?|industry|sector)\b",
                    sentence,
                    re.I,
                ):
                    continue

            for target, alias, target_start, target_end in targets:

                if explicit_competition is not None:
                    if target_start < explicit_competition.end():
                        continue

                    middle = sentence[
                        explicit_competition.end():target_start
                    ]

                    if BLOCKED_HEADINGS.search(middle):
                        continue

                # Customer/supplier direction is deliberately
                # NOT inferred here. Both possible directions
                # are independently checked by the LLM.
                proposed = [(issuer, relation, target)]

                if relation in {
                    "SUPPLIES_TO",
                    "CUSTOMER_OF",
                }:
                    proposed.append(
                        (target, relation, issuer)
                    )

                for subject, rel, obj in proposed:
                    cid = uid(
                        evidence["evidence_id"],
                        subject, rel, obj
                    )

                    if cid in seen:
                        continue

                    seen.add(cid)

                    records.append({
                        "candidate_id": cid,
                        "evidence_id": evidence["evidence_id"],
                        "ticker": ticker,
                        "issuer": issuer,
                        "subject": subject,
                        "relation": rel,
                        "object": obj,
                        "target_alias": alias,
                        "sentence": sentence,
                        "context": str(evidence["context"]),
                        "cue": "|".join(
                            str(h[2]) for h in hits[:5]
                        ),
                        "accession": str(evidence["accession"]),
                        "filing_date": str(
                            evidence["filing_date"]
                        )[:10],
                        "available_from": str(
                            evidence["available_from"]
                        )[:10],
                        "source_sha256": str(
                            evidence["source_sha256"]
                        ),
                        "offset_start": int(
                            evidence["offset_start"]
                        ),
                        "offset_end": int(
                            evidence["offset_end"]
                        ),
                        "retrieval_score": float(
                            evidence["retrieval_score"]
                        ),
                        "candidate_method": (
                            "LEXICAL_ENTITY_AND_RELATION_SCREEN"
                        ),
                    })

    df = pd.DataFrame(records, columns=candidate_cols())

    df = df.sort_values(
        ["retrieval_score", "ticker", "filing_date"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    df.to_csv(
        OUT / "candidates.csv", index=False
    )

    print("\n=== V37 CANDIDATE GENERATION ===")
    print("V34 evidence:", len(src))
    print("Total candidates:", len(df))
    print("Unique evidence:", df["evidence_id"].nunique())

    print("\nBY RELATION:")
    print(df["relation"].value_counts().to_string())

    print("\nBY ISSUER:")
    print(df["ticker"].value_counts().to_string())

    print("\nEXAMPLES:")
    print(
        df[[
            "subject", "relation", "object"
        ]].head(15).to_string(index=False)
    )

    print("\nWARNING:")
    print("Candidates are NOT verified relationships.")


def audit():
    df = pd.read_csv(
        OUT / "candidates.csv",
        dtype={"accession": str},
    )

    src = load_source().set_index("evidence_id")
    aliases = load_aliases()

    if df["candidate_id"].duplicated().any():
        stop("Duplicate candidate IDs")

    for r in df.itertuples(index=False):
        if r.relation not in RELATIONS:
            stop("Invalid relation")

        if r.ticker not in ISSUERS:
            stop("Unknown issuer")

        if r.issuer != ISSUERS[r.ticker]:
            stop("Issuer mismatch")

        if r.subject == r.object:
            stop("Self-edge candidate")

        if r.evidence_id not in src.index:
            stop("Evidence ID missing")

        e = src.loc[r.evidence_id]

        if r.sentence != str(e["sentence"]):
            stop("Sentence provenance mismatch")

        if str(r.source_sha256) != str(e["source_sha256"]):
            stop("SHA provenance mismatch")

        if str(r.accession) != str(e["accession"]):
            stop("Accession mismatch")

        if r.object not in aliases and r.subject not in aliases:
            stop("Unknown target entity")

        if not (
            pd.Timestamp(r.available_from)
            > pd.Timestamp(r.filing_date)
        ):
            stop("Invalid availability chronology")

    print("\n=== V37 AUDIT PASS ===")
    print("Candidates:", len(df))
    print("SEC text and accession consistency: PASS")
    print("Issuer identity: PASS")
    print("Date order: PASS")
    print("Semantic accuracy: NOT YET VERIFIED")
    print("Strict PIT: NOT CERTIFIED")


def parse_json(content):
    text = str(content or "").strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "", text, flags=re.I
    )
    text = re.sub(r"\s*```\s*$", "", text).strip()

    obj = json.loads(text)

    if isinstance(obj, list):
        obj = {"decisions": obj}

    if not isinstance(obj, dict):
        raise ValueError("JSON root not object/list")

    decisions = obj.get("decisions")

    if not isinstance(decisions, list):
        raise ValueError("Missing decisions array")

    fixed = []

    for item in decisions:
        if not isinstance(item, dict):
            continue

        quote = item.get("evidence_quote")

        if quote is None:
            quote = item.get("quote")

        if quote is None:
            quote = item.get("evidence")

        fixed.append({
            "candidate_id": str(
                item.get("candidate_id", "")
            ),
            "supported": item.get("supported"),
            "evidence_quote": (
                quote if isinstance(quote, str) else ""
            ),
            "reason": (
                item.get("reason")
                if isinstance(item.get("reason"), str)
                else ""
            ),
            "market_scope": (
                item.get("market_scope")
                if isinstance(item.get("market_scope"), str)
                else ""
            ),
        })

    return fixed


def relation_anchor(quote, relation):
    if relation == "COMPETES_WITH":
        return bool(
            COMPETITOR_LIST.search(quote)
        )

    patterns = PATTERNS.get(relation, [])

    return any(
        re.search(p, quote, re.I)
        for p in patterns
    )


def validate(row, answer, aliases):
    errors = []

    if type(answer.get("supported")) is not bool:
        errors.append("INVALID_BOOLEAN")

    quote = answer.get("evidence_quote", "")

    if not isinstance(quote, str):
        quote = ""
        errors.append("INVALID_QUOTE_TYPE")

    supported = answer.get("supported") is True

    if supported:
        sentence = row["sentence"]

        if len(quote.strip()) < 12:
            errors.append("SHORT_QUOTE")

        if quote not in sentence:
            errors.append("QUOTE_NOT_EXACT")

        issuer = row["issuer"]

        target = (
            row["object"]
            if row["subject"] == issuer
            else row["subject"]
        )

        target_hits = alias_spans(
            quote,
            aliases.get(target, []),
        )

        if not target_hits:
            errors.append("NAMED_TARGET_NOT_IN_QUOTE")

        if not relation_anchor(
            quote, row["relation"]
        ):
            errors.append("RELATION_CUE_NOT_IN_QUOTE")

        if row["relation"] == "COMPETES_WITH":
            anchor = COMPETITOR_LIST.search(quote)

            if anchor and target_hits:
                if target_hits[0][0] < anchor.end():
                    errors.append("TARGET_BEFORE_COMPETITOR_LIST")

        # Basic evidence quote checks cannot fully verify
        # semantic direction. That remains an AI silver label.

    return supported and not errors, "|".join(errors)


def load_log(path):
    records = {}

    if not path.exists():
        return records

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["evidence_id"]] = r

    return records


def export_labels(records, run_dir):
    rows = []

    for record in records.values():
        if record.get("status") == "ok":
            rows.extend(
                record.get("decisions", [])
            )

    cols = [
        "candidate_id", "evidence_id",
        "ticker", "issuer", "subject",
        "relation", "object", "target_alias",
        "filing_date", "available_from",
        "accession", "source_sha256",
        "offset_start", "offset_end",
        "supported", "evidence_quote",
        "market_scope", "reason",
        "validation_error", "accepted_silver",
        "model", "human_verified",
        "strict_pit_approved",
    ]

    df = pd.DataFrame(rows, columns=cols)

    df.to_csv(
        run_dir / "silver_decisions.csv",
        index=False,
    )

    accepted = df[
        df["accepted_silver"] == True
    ].copy()

    accepted.to_csv(
        run_dir / "silver_relations.csv",
        index=False,
    )


def label(args):
    from ollama import Client

    candidates = pd.read_csv(
        OUT / "candidates.csv",
        dtype={"accession": str},
    ).fillna("")

    if candidates.empty:
        stop("No candidates. Run build.")

    groups = [
        g for _, g in candidates.groupby(
            "evidence_id", sort=False
        )
    ]

    groups = sorted(
        groups,
        key=lambda g: (
            -g["retrieval_score"].max(),
            str(g["evidence_id"].iloc[0]),
        ),
    )

    if args.limit:
        groups = groups[:args.limit]

    run_dir = OUT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    log_path = run_dir / "responses.jsonl"

    config = {
        "model": args.model,
        "host": args.host,
        "think": False,
        "num_predict": args.num_predict,
        "max_candidates": args.max_candidates,
        "version": "v37_1",
    }

    if config_path.exists():
        old = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        if old != config:
            stop(
                "Run config differs. Use a new --run-name."
            )
    else:
        json_write(config_path, config)

    records = load_log(log_path)
    aliases = load_aliases()

    client = Client(
        host=args.host,
        timeout=args.timeout,
    )

    print("\n=== V37 OLLAMA ===")
    print("Model:", args.model)
    print("Thinking: OFF")
    print("Evidence groups:", len(groups))
    print("Already completed:", sum(
        records.get(str(g["evidence_id"].iloc[0]), {})
        .get("status") == "ok"
        for g in groups
    ))

    for i, group in enumerate(groups, 1):
        eid = str(group["evidence_id"].iloc[0])

        if records.get(eid, {}).get("status") == "ok":
            continue

        group = group.head(args.max_candidates)
        first = group.iloc[0]

        proposals = []

        for r in group.itertuples(index=False):
            proposals.append({
                "candidate_id": r.candidate_id,
                "subject": r.subject,
                "relation": r.relation,
                "object": r.object,
            })

        payload = {
            "issuer": first["issuer"],
            "ticker": first["ticker"],
            "filing_date": first["filing_date"],
            "CURRENT_SEC_SENTENCE": first["sentence"],
            "proposed_relations": proposals,
        }

        record = {
            "evidence_id": eid,
            "model": args.model,
            "ticker": first["ticker"],
            "candidate_count": len(proposals),
        }

        print(
            f"[{i}/{len(groups)}] "
            f"{first['ticker']} "
            f"{len(proposals)} candidates...",
            flush=True,
        )

        start = time.time()

        try:
            response = client.chat(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                    },
                ],
                format=SCHEMA,
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": args.num_predict,
                },
                stream=False,
            )

            content = response.message.content or ""

            record["raw_content"] = content
            record["done_reason"] = getattr(
                response, "done_reason", None
            )

            record["thinking_chars"] = len(
                getattr(
                    response.message,
                    "thinking",
                    "",
                ) or ""
            )

            if not content.strip():
                raise ValueError("EMPTY_MODEL_CONTENT")

            decisions = parse_json(content)

            ids = [
                d["candidate_id"] for d in decisions
            ]

            if len(ids) != len(set(ids)):
                raise ValueError(
                    "DUPLICATE_MODEL_CANDIDATE_IDS"
                )

            by_id = {
                d["candidate_id"]: d
                for d in decisions
            }

            expected = set(
                group["candidate_id"].astype(str)
            )

            if set(by_id) != expected:
                raise ValueError(
                    "MODEL_CANDIDATE_ID_MISMATCH "
                    f"missing={len(expected - set(by_id))} "
                    f"unknown={len(set(by_id) - expected)}"
                )

            verified = []

            for r in group.to_dict("records"):
                d = by_id[str(r["candidate_id"])]

                accepted, error = validate(
                    r, d, aliases
                )

                verified.append({
                    "candidate_id": r["candidate_id"],
                    "evidence_id": eid,
                    "ticker": r["ticker"],
                    "issuer": r["issuer"],
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "object": r["object"],
                    "target_alias": r["target_alias"],
                    "filing_date": r["filing_date"],
                    "available_from": r["available_from"],
                    "accession": r["accession"],
                    "source_sha256": r["source_sha256"],
                    "offset_start": r["offset_start"],
                    "offset_end": r["offset_end"],
                    "supported": d["supported"],
                    "evidence_quote": d["evidence_quote"],
                    "market_scope": d["market_scope"],
                    "reason": d["reason"],
                    "validation_error": error,
                    "accepted_silver": accepted,
                    "model": args.model,
                    "human_verified": False,
                    "strict_pit_approved": False,
                })

            record["status"] = "ok"
            record["decisions"] = verified

            print(
                "  OK accepted=",
                sum(
                    x["accepted_silver"]
                    for x in verified
                ),
                "/",
                len(verified),
                flush=True,
            )

        except Exception as exc:
            record["status"] = "error"
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )[:1000]

            print(
                "  ERROR:", record["error"],
                flush=True,
            )

        record["seconds"] = round(
            time.time() - start, 2
        )

        # Append raw response immediately for recovery.
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())

        records[eid] = record
        export_labels(records, run_dir)

    export_labels(records, run_dir)
    print("\n=== V37 LABELING COMPLETE ===")
    report(args)


def get_verified(args):
    path = (
        OUT / "runs" / args.run_name
        / "silver_relations.csv"
    )

    if not path.exists():
        stop("Run label first")

    df = pd.read_csv(
        path,
        dtype={"accession": str},
    ).fillna("")

    if df.empty:
        stop(
            "No accepted silver relations. "
            "Do not fabricate KG edges."
        )

    if not df["accepted_silver"].astype(
        str
    ).str.lower().isin(
        ["true", "1"]
    ).all():
        stop("Silver file contains unaccepted rows")

    return df


def graph(args):
    OUT.mkdir(parents=True, exist_ok=True)

    silver = get_verified(args)
    src = load_source().set_index("evidence_id")

    ledger = []

    for row in silver.to_dict("records"):
        eid = str(row["evidence_id"])

        if eid not in src.index:
            stop("Unknown evidence: " + eid)

        original = src.loc[eid]
        sentence = str(original["sentence"])
        quote = str(row["evidence_quote"])

        if (
            len(quote.strip()) < 12
            or quote not in sentence
        ):
            stop("Invalid quote: " + eid)

        if str(row["source_sha256"]) != str(
            original["source_sha256"]
        ):
            stop("Source hash mismatch: " + eid)

        if str(row["accession"]) != str(
            original["accession"]
        ):
            stop("Accession mismatch: " + eid)

        for col in ["offset_start", "offset_end"]:
            if int(row[col]) != int(original[col]):
                stop("Offset mismatch: " + eid)

        available = pd.Timestamp(
            row["available_from"]
        ).normalize()

        filing = pd.Timestamp(
            row["filing_date"]
        ).normalize()

        if available <= filing:
            stop("Invalid filing timeline")

        relation_id = uid(
            row["subject"],
            row["relation"],
            row["object"],
        )

        ledger.append({
            **row,
            "relation_id": relation_id,
            "sentence": sentence,
            "label_source": "AI_SILVER",
            "human_verified": False,
            "strict_pit_approved": False,
        })

    ledger = pd.DataFrame(ledger)

    ledger = ledger.drop_duplicates(
        "candidate_id"
    ).sort_values(
        ["available_from", "candidate_id"]
    )

    # Construct graph edges without deleting historical mentions.
    edge_rows = []

    for rel_id, g in ledger.groupby("relation_id"):
        g = g.sort_values("available_from")
        first = g.iloc[0]

        edge_rows.append({
            "relation_id": rel_id,
            "subject": first["subject"],
            "relation": first["relation"],
            "object": first["object"],
            "first_disclosed": g["available_from"].min(),
            "last_disclosed": g["available_from"].max(),
            "mentions": len(g),
            "filings": g["accession"].nunique(),
            "evidence_ids": json.dumps(
                g["evidence_id"].tolist()
            ),
            "human_verified": False,
        })

    edges = pd.DataFrame(edge_rows)

    # Five portfolio stocks ONLY.
    market = pd.read_csv(
        ROOT / "outputs/v20/fusion/dataset_conservative.csv"
    )

    date_col = (
        "date" if "date" in market.columns
        else "Date"
    )
    ticker_col = (
        "tic" if "tic" in market.columns
        else "ticker"
    )

    market["date_key"] = pd.to_datetime(
        market[date_col]
    ).dt.normalize()

    trading_dates = sorted(
        market["date_key"].unique()
    )

    evidence = ledger.copy()
    evidence["_available"] = pd.to_datetime(
        evidence["available_from"]
    ).dt.normalize()

    # Each graph feature is based only on what was
    # disclosed by a portfolio issuer before the trading date.
    # This remains a provisional disclosure-date safeguard.
    daily_rows = []

    for ticker in TICKERS:
        issuer = ISSUERS[ticker]

        own = evidence[
            (evidence["ticker"] == ticker)
            & (evidence["issuer"] == issuer)
        ].copy()

        for date in trading_dates:
            date = pd.Timestamp(date).normalize()

            # Strictly later calendar date.
            hist = own[
                own["_available"] < date
            ]

            stats = {
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
            }

            # Existing 11-slot V36 compatibility.
            comp = hist[
                (hist["relation"] == "COMPETES_WITH")
                & (hist["subject"] == issuer)
            ]

            if len(comp):
                first_edges = comp.groupby(
                    "relation_id"
                )["_available"].min()

                unique_targets = comp.drop_duplicates(
                    "relation_id"
                )

                portfolio_targets = unique_targets[
                    unique_targets["object"].isin(
                        CANONICAL_TICKERS
                    )
                ]

                stats.update({
                    "kg_known_competitors": len(first_edges),
                    "kg_competitor_mentions": len(comp),
                    "kg_recent_mentions_30d": int(
                        (comp["_available"] >=
                         date - pd.Timedelta(days=30)).sum()
                    ),
                    "kg_recent_mentions_90d": int(
                        (comp["_available"] >=
                         date - pd.Timedelta(days=90)).sum()
                    ),
                    "kg_new_edges_30d": int(
                        (first_edges >=
                         date - pd.Timedelta(days=30)).sum()
                    ),
                    "kg_new_edges_90d": int(
                        (first_edges >=
                         date - pd.Timedelta(days=90)).sum()
                    ),
                    "kg_days_since_last_disclosure": int(
                        (date - comp["_available"].max()).days
                    ),
                    "kg_days_since_first_disclosure": int(
                        (date - comp["_available"].min()).days
                    ),
                    "kg_portfolio_competitors": len(
                        portfolio_targets
                    ),
                    "kg_external_competitors": (
                        len(first_edges) - len(portfolio_targets)
                    ),
                    "kg_competitor_remention_ratio": (
                        (len(comp) - len(first_edges))
                        / max(len(comp), 1)
                    ),
                })
            else:
                stats.update({
                    key: 0.0 for key in KG11
                })

            # New four-relation typed features.
            for relation in RELATIONS:
                rel_hist = hist[
                    hist["relation"] == relation
                ]

                own_out = rel_hist[
                    rel_hist["subject"] == issuer
                ]

                own_in = rel_hist[
                    rel_hist["object"] == issuer
                ]

                prefix = relation.lower()

                stats[
                    "kg_" + prefix + "_out_degree"
                ] = own_out["object"].nunique()

                stats[
                    "kg_" + prefix + "_in_degree"
                ] = own_in["subject"].nunique()

                stats[
                    "kg_" + prefix + "_mentions_90d"
                ] = int(
                    (
                        rel_hist["_available"]
                        >= date - pd.Timedelta(days=90)
                    ).sum()
                )

            stats["kg_total_typed_edges"] = hist[
                "relation_id"
            ].nunique()

            stats["kg_total_mentions"] = len(hist)

            stats["kg_has_disclosed_relation"] = int(
                len(hist) > 0
            )

            stats["kg_relation_coverage"] = (
                "OBSERVED" if len(hist) else "UNOBSERVED"
            )

            stats["kg_strict_pit_approved"] = False

            daily_rows.append(stats)

    daily = pd.DataFrame(daily_rows)

    if len(daily) != len(market):
        stop("Daily panel mismatch")

    if daily.duplicated(["date", "tic"]).any():
        stop("Duplicate ticker-day in daily KG")

    # Recheck all 11 legacy feature columns.
    for col in KG11:
        if col not in daily:
            stop("Missing V36-compatible KG feature")

        if not np.isfinite(
            daily[col].to_numpy(dtype=float)
        ).all():
            stop("Invalid KG feature: " + col)

    ledger.to_csv(
        OUT / "evidence_ledger.csv", index=False
    )

    edges.to_csv(
        OUT / "temporal_edges.csv", index=False
    )

    daily.to_csv(
        OUT / "daily_kg_features.csv", index=False
    )

    with (
        OUT / "graph_events.jsonl"
    ).open("w", encoding="utf-8") as f:
        for r in ledger.to_dict("records"):
            event = {
                "relation_id": r["relation_id"],
                "evidence_id": r["evidence_id"],
                "subject": r["subject"],
                "relation": r["relation"],
                "object": r["object"],
                "available_from": r["available_from"],
                "accession": r["accession"],
                "evidence_quote": r["evidence_quote"],
                "human_verified": False,
                "strict_pit_approved": False,
            }
            f.write(
                json.dumps(
                    event, ensure_ascii=False,
                    default=str,
                ) + "\n"
            )

    report_data = {
        "run_name": args.run_name,
        "evidence_mentions": len(ledger),
        "unique_directed_edges": len(edges),
        "daily_feature_rows": len(daily),
        "relation_types": sorted(
            edges["relation"].unique().tolist()
        ),
        "requested_relation_types": RELATIONS,
        "relation_counts": edges[
            "relation"
        ].value_counts().to_dict(),
        "feature_columns": KG11,
        "typed_feature_columns": [
            c for c in daily.columns
            if (
                c.startswith("kg_")
                and c not in KG11
            )
        ],
        "temporal_policy": (
            "trading_date > provisional_available_from"
        ),
        "human_verified": False,
        "strict_pit_approved": False,
        "warning": (
            "AI silver graph. Unobserved is not absent. "
            "SEC acceptance timestamps and execution-time "
            "PIT are not independently certified."
        ),
    }

    json_write(
        OUT / "report.json",
        report_data,
    )

    print("\n=== V37 TEMPORAL KG BUILT ===")
    print(json.dumps(
        report_data,
        indent=2,
        ensure_ascii=False,
    ))

    print("\n=== ACTUAL RELATION EDGES ===")
    print(
        edges[[
            "subject", "relation", "object",
            "first_disclosed", "mentions"
        ]].to_string(index=False)
    )


def wire():
    if not V36_SCRIPT.exists():
        stop("Original V36 script not found")

    needed = [
        OUT / "daily_kg_features.csv",
        OUT / "evidence_ledger.csv",
        OUT / "report.json",
    ]

    for path in needed:
        if not path.exists():
            stop("Run V37 graph first: " + str(path))

    text = V36_SCRIPT.read_text(
        encoding="utf-8"
    )

    old_v35 = 'V35 = ROOT / "outputs/v35"'
    old_out = 'OUT = ROOT / "outputs/v36"'

    if old_v35 not in text or old_out not in text:
        stop(
            "Original V36 script structure changed. "
            "Refuse unsafe automatic patch."
        )

    text = text.replace(
        old_v35,
        'V35 = ROOT / "outputs/v37"',
        1,
    )

    text = text.replace(
        old_out,
        'OUT = ROOT / "outputs/v37/finrl"',
        1,
    )

    destination = (
        ROOT / "scripts/v37_finrl_ablation.py"
    )

    destination.write_text(
        text, encoding="utf-8"
    )

    import py_compile
    py_compile.compile(
        str(destination),
        doraise=True,
    )

    print("\n=== V37 FINRL BRIDGE CREATED ===")
    print(destination)
    print("V20 untouched")
    print("V21 untouched")
    print("V35 untouched")
    print("V36 untouched")
    print("V37 KG used for next FinRL experiment")
    print("\nNEXT:")
    print(
        ".venv-drl/bin/python -u "
        "scripts/v37_finrl_ablation.py inspect"
    )


def report(args):
    path = (
        OUT / "runs" / args.run_name
        / "responses.jsonl"
    )

    records = load_log(path)

    print("\n=== V37 EXTRACTION REPORT ===")
    print("Requests:", len(records))
    print("Successful:", sum(
        r.get("status") == "ok"
        for r in records.values()
    ))
    print("Failed:", sum(
        r.get("status") == "error"
        for r in records.values()
    ))

    silver_path = (
        OUT / "runs" / args.run_name
        / "silver_relations.csv"
    )

    if silver_path.exists():
        df = pd.read_csv(silver_path)

        print("Accepted silver mentions:", len(df))

        if len(df):
            print("\nRELATIONS:")
            print(
                df["relation"].value_counts().to_string()
            )

            print("\nUNIQUE EDGES:")
            print(
                df[[
                    "subject", "relation", "object"
                ]].drop_duplicates().to_string(index=False)
            )

    graph_report = OUT / "report.json"

    if graph_report.exists():
        print("\n=== TEMPORAL GRAPH REPORT ===")
        print(
            graph_report.read_text(
                encoding="utf-8"
            )
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "build", "audit", "label",
            "graph", "wire", "report",
        ],
    )

    p.add_argument(
        "--model",
        default="qwen3.5:27b",
    )

    p.add_argument(
        "--host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument(
        "--run-name",
        default="four_rel_27b",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of SEC evidence groups; 0 for all.",
    )

    p.add_argument(
        "--max-candidates",
        type=int,
        default=12,
    )

    p.add_argument(
        "--timeout",
        type=int,
        default=300,
    )

    p.add_argument(
        "--num-predict",
        type=int,
        default=2048,
    )

    args = p.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.command == "build":
        build(args)
    elif args.command == "audit":
        audit()
    elif args.command == "label":
        label(args)
    elif args.command == "graph":
        graph(args)
    elif args.command == "wire":
        wire()
    elif args.command == "report":
        report(args)


if __name__ == "__main__":
    main()
