#!/usr/bin/env python3
"""
V34: SEC Entity-aware Relation Retrieval + Ollama Silver Extraction

Usage:
  python scripts/v34_sec_relation_retrieval.py build
  python scripts/v34_sec_relation_retrieval.py audit
  python scripts/v34_sec_relation_retrieval.py label --limit 20
  python scripts/v34_sec_relation_retrieval.py report

Source:
  V20 SEC manifest and SHA256-verified cached documents
  V28 filing_features.csv for available_from

Outputs:
  outputs/v34/entity_aliases.csv
  outputs/v34/relation_evidence.csv
  outputs/v34/silver_labels.csv
  outputs/v34/silver_relations.csv
  outputs/v34/ollama_log.jsonl
  outputs/v34/report.json

All relation extractions are unverified SILVER candidates.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v34"
DOCS = ROOT / "outputs/v20/sec_cache/documents"
MANIFEST = ROOT / "outputs/v20/sec_filings_manifest.csv"
FILINGS = ROOT / "outputs/v28/filing_features.csv"

TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}

RELATIONS = {
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
}

# These are entity aliases, NOT asserted relationships.
# Extend via outputs/v34/entity_aliases.csv.
DEFAULT_ALIASES = [
    ("Apple", "Apple Inc.", "AAPL"),
    ("Microsoft", "Microsoft Corporation", "MSFT"),
    ("NVIDIA", "NVIDIA Corporation", "NVDA"),
    ("Amazon", "Amazon.com Inc.", "AMZN"),
    ("Alphabet", "Alphabet Inc.", "GOOGL"),
    ("Google", "Alphabet Inc.", "GOOGL"),
    ("TSMC", "Taiwan Semiconductor Manufacturing Company", ""),
    ("Taiwan Semiconductor Manufacturing", "Taiwan Semiconductor Manufacturing Company", ""),
    ("Samsung Electronics", "Samsung Electronics", ""),
    ("Intel", "Intel Corporation", ""),
    ("Advanced Micro Devices", "Advanced Micro Devices", ""),
    ("AMD", "Advanced Micro Devices", ""),
    ("Qualcomm", "Qualcomm Incorporated", ""),
    ("Broadcom", "Broadcom Inc.", ""),
    ("Foxconn", "Hon Hai Precision Industry", ""),
    ("Hon Hai", "Hon Hai Precision Industry", ""),
    ("SK Hynix", "SK Hynix", ""),
    ("Micron", "Micron Technology", ""),
    ("Oracle", "Oracle Corporation", ""),
    ("IBM", "International Business Machines", ""),
    ("Cisco", "Cisco Systems", ""),
    ("Salesforce", "Salesforce Inc.", ""),
    ("Meta Platforms", "Meta Platforms", ""),
    ("Adobe", "Adobe Inc.", ""),
    ("Sony", "Sony Group", ""),
    ("Walmart", "Walmart Inc.", ""),
]

CUES = {
    "SUPPLY": [
        r"\bsupplier(?:s)?\b",
        r"\bsupply(?:ing|ies|ied)?\b",
        r"\bprocure(?:ment|d)?\b",
        r"\bpurchas(?:e|es|ed|ing)\b",
        r"\bmanufactur(?:e|es|ed|ing|er|ers)\b",
        r"\bfoundr(?:y|ies)\b",
        r"\bcontract manufacturer(?:s)?\b",
        r"\bcomponent(?:s)?\b",
        r"\bvendor(?:s)?\b",
        r"\bsource(?:d|s|ing)? from\b",
    ],
    "CUSTOMER": [
        r"\bcustomer(?:s)?\b",
        r"\bclient(?:s)?\b",
        r"\bsell(?:s|ing)? to\b",
        r"\bsales to\b",
        r"\brevenue from\b",
        r"\bdistribut(?:or|ors|ion)\b",
        r"\blicens(?:e|es|ing)\b",
    ],
    "COMPETITION": [
        r"\bcompet(?:e|es|ed|ing|ition|itor|itors)\b",
        r"\brival(?:s)?\b",
        r"\bmarket share\b",
        r"\bcompetes with\b",
    ],
    "PARTNERSHIP_CONTEXT": [
        r"\bpartnership\b",
        r"\bcollaborat(?:e|ion|ing)\b",
        r"\bagreement\b",
        r"\bjoint venture\b",
    ],
}

RISK_PATTERNS = [
    r"\bmay adversely affect\b",
    r"\bcould adversely affect\b",
    r"\bsubject to uncertainty\b",
    r"\bthere can be no assurance\b",
    r"\bforward-looking statements\b",
]

MAX_CONTEXT_CHARS = 1300
MAX_SENTENCE_CHARS = 1100
MIN_SENTENCE_WORDS = 8
DEFAULT_PER_FILING = 35

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def uid(*values):
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode("utf-8")
    ).hexdigest()[:24]

def normalized(text):
    return " ".join(str(text).split())

def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if self.skip == 0:
            self.parts.append(data)

def extract_text(data):
    parser = HTMLText()
    parser.feed(data.decode("utf-8", errors="replace"))
    return normalized(" ".join(parser.parts))

def cache_path(row):
    safe = re.sub(
        r"[^A-Za-z0-9_.-]", "_", str(row["primary_document"])
    )
    return DOCS / (
        str(row["ticker"]) + "_" +
        str(row["accession"]) + "_" + safe
    )

def create_alias_file():
    p = OUT / "entity_aliases.csv"
    if not p.exists():
        pd.DataFrame(
            DEFAULT_ALIASES,
            columns=["alias", "canonical", "portfolio_ticker"]
        ).to_csv(p, index=False)
        print("Created editable entity dictionary:", p)
    return p

def load_aliases():
    p = create_alias_file()
    df = pd.read_csv(p).fillna("")
    needed = {"alias", "canonical", "portfolio_ticker"}
    if not needed.issubset(df.columns):
        raise ValueError("Invalid entity_aliases.csv columns")

    aliases = []
    lookup = {}
    for r in df.itertuples(index=False):
        alias = str(r.alias).strip()
        canonical = str(r.canonical).strip()
        ticker = str(r.portfolio_ticker).strip()
        if not alias or not canonical:
            continue
        if ticker and ticker not in TICKERS:
            raise ValueError("Entity dictionary changes portfolio universe")
        key = alias.casefold()
        if key in lookup and lookup[key] != canonical:
            raise ValueError("Ambiguous alias: " + alias)
        lookup[key] = canonical
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        aliases.append((alias, canonical, pattern))
    return aliases

def find_entities(text, aliases):
    names = []
    for alias, canonical, pattern in aliases:
        if pattern.search(text) and canonical not in names:
            names.append(canonical)
    return names

def find_cues(text):
    found = []
    for topic, patterns in CUES.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            found.append(topic)
    return found

def sentence_spans(text):
    """
    Offsets are character offsets of normalized parsed SEC text,
    NOT byte offsets in raw HTML.
    """
    boundaries = [0]
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(\"])",
                         text):
        boundaries.append(m.end())
    boundaries.append(len(text))

    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        if b <= a:
            continue
        section = text[a:b]
        leading = len(section) - len(section.lstrip())
        trailing = len(section.rstrip())
        a2 = a + leading
        b2 = a + trailing
        if b2 > a2:
            yield a2, b2, text[a2:b2]

def prepare_documents():
    manifest = pd.read_csv(MANIFEST, dtype={"accession": str})
    filings = pd.read_csv(FILINGS, dtype={"accession": str})

    for df in (manifest, filings):
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["accession"] = df["accession"].astype(str)

    manifest = manifest[
        manifest["ticker"].isin(TICKERS) &
        manifest["form"].isin(["10-K", "10-Q"])
    ].copy()

    if len(manifest) != 140:
        raise ValueError(f"Expected 140 source docs, got {len(manifest)}")

    if manifest.duplicated(["ticker", "accession"]).any():
        raise ValueError("Duplicate manifest accession")
    if filings.duplicated(["ticker", "accession"]).any():
        raise ValueError("Duplicate V28 accession")

    docs = manifest.merge(
        filings[["ticker", "accession", "available_from", "filing_date"]],
        on=["ticker", "accession"], how="left",
        validate="one_to_one", indicator=True,
        suffixes=("", "_v28")
    )

    if not docs["_merge"].eq("both").all():
        raise ValueError("SEC manifest and V28 metadata mismatch")

    docs["filing_date"] = pd.to_datetime(docs["filing_date"])
    docs["available_from"] = pd.to_datetime(docs["available_from"])
    if not (docs["available_from"] > docs["filing_date"]).all():
        raise ValueError("Filing disclosure chronology invalid")

    return docs.sort_values(
        ["available_from", "ticker", "accession"]
    ).to_dict("records")

def build(per_filing):
    OUT.mkdir(parents=True, exist_ok=True)
    aliases = load_aliases()
    documents = prepare_documents()
    all_rows = []
    audit_rows = []

    for i, doc in enumerate(documents, 1):
        p = cache_path(doc)
        if not p.is_file():
            raise FileNotFoundError(p)

        data = p.read_bytes()
        expected = str(doc["source_sha256"])
        if sha_bytes(data) != expected:
            raise ValueError("SHA-256 mismatch: " + str(p))
        if str(doc["text_verified"]).lower() not in (
            "true", "1", "1.0"
        ):
            raise ValueError("Text not verified: " + str(p))

        text = extract_text(data)
        spans = list(sentence_spans(text))
        selected = []

        for j, (start, end, sentence) in enumerate(spans):
            if len(sentence.split()) < MIN_SENTENCE_WORDS:
                continue

            # Include long complete sentences only when
            # practical for LLM context; do not truncate into
            # misleading claimed-complete evidence.
            if len(sentence) > MAX_SENTENCE_CHARS:
                continue

            neighbor_start = spans[max(0, j - 1)][0]
            neighbor_end = spans[min(len(spans) - 1, j + 1)][1]
            context = text[neighbor_start:neighbor_end]

            entities = find_entities(sentence, aliases)
            cues = find_cues(sentence)

            # Relation candidate needs a named entity AND
            # relation cue in the SAME sentence.
            if not entities or not cues:
                continue

            # Expand to surrounding sentence only if it fits;
            # otherwise preserve the complete focal sentence.
            if len(context) > MAX_CONTEXT_CHARS:
                context = sentence
                context_start = start
            else:
                context_start = neighbor_start

            risk = int(any(
                re.search(x, sentence, re.IGNORECASE)
                for x in RISK_PATTERNS
            ))

            selected.append({
                "evidence_id": uid(
                    doc["ticker"], doc["accession"], start, end
                ),
                "ticker": doc["ticker"],
                "form": doc["form"],
                "accession": doc["accession"],
                "filing_date": str(doc["filing_date"].date()),
                "available_from": str(doc["available_from"].date()),
                "source_sha256": expected,
                "offset_start": start,
                "offset_end": end,
                "context_start": context_start,
                "sentence": sentence,
                "context": context,
                "named_entities": json.dumps(entities),
                "relation_cues": json.dumps(cues),
                "risk_language": risk,
                "retrieval_score": (
                    3 * min(len(entities), 3)
                    + 2 * min(len(cues), 3)
                    - 2 * risk
                ),
                "extraction_status": "LEXICAL_CANDIDATE_ONLY",
                "human_verified": False,
            })

        # Deduplicate identical sentences within a filing.
        dedup = {}
        for row in selected:
            key = re.sub(r"\s+", " ", row["sentence"].casefold())
            if key not in dedup or (
                row["retrieval_score"] > dedup[key]["retrieval_score"]
            ):
                dedup[key] = row

        selected = sorted(
            dedup.values(),
            key=lambda x: (-x["retrieval_score"], x["offset_start"])
        )

        all_rows.extend(selected[:per_filing])
        audit_rows.append({
            "ticker": doc["ticker"],
            "accession": doc["accession"],
            "verified_sha256": True,
            "candidate_sentences": len(dedup),
            "retained_sentences": min(len(selected), per_filing),
        })

        print(
            f"[{i}/140] {doc['ticker']} "
            f"{doc['form']} "
            f"candidates={len(dedup)} "
            f"retained={min(len(selected), per_filing)}",
            flush=True
        )

    columns = [
        "evidence_id", "ticker", "form", "accession",
        "filing_date", "available_from", "source_sha256",
        "offset_start", "offset_end", "context_start",
        "sentence", "context", "named_entities",
        "relation_cues", "risk_language", "retrieval_score",
        "extraction_status", "human_verified"
    ]

    result = pd.DataFrame(all_rows, columns=columns)
    if result["evidence_id"].duplicated().any():
        raise ValueError("Duplicate evidence IDs")

    result.to_csv(OUT / "relation_evidence.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        OUT / "source_audit.csv", index=False
    )
    print("\n=== V34 RETRIEVAL BUILT ===")
    print("SEC filings:", len(documents))
    print("Relation evidence candidates:", len(result))
    print("File:", OUT / "relation_evidence.csv")

def audit():
    aliases = load_aliases()
    docs = prepare_documents()
    df = pd.read_csv(OUT / "relation_evidence.csv",
                     dtype={"accession": str}).fillna("")

    text_cache = {}
    checked = 0

    for doc in docs:
        key = (doc["ticker"], doc["accession"])
        p = cache_path(doc)
        data = p.read_bytes()
        if sha_bytes(data) != str(doc["source_sha256"]):
            raise ValueError("SHA mismatch: " + str(p))
        text_cache[key] = extract_text(data)
        checked += 1

    for r in df.itertuples(index=False):
        doc_text = text_cache[(r.ticker, str(r.accession))]
        a, b, c = (
            int(r.offset_start),
            int(r.offset_end),
            int(r.context_start)
        )
        if doc_text[a:b] != r.sentence:
            raise ValueError("Sentence offset mismatch: " + r.evidence_id)
        if doc_text[c:c + len(r.context)] != r.context:
            raise ValueError("Context offset mismatch: " + r.evidence_id)
        if not find_entities(r.sentence, aliases):
            raise ValueError("Missing named entity")
        if not find_cues(r.sentence):
            raise ValueError("Missing relation cue")
        if not pd.Timestamp(r.filing_date) < pd.Timestamp(r.available_from):
            raise ValueError("Date leakage")

    if not set(df["ticker"]).issubset(TICKERS):
        raise ValueError("Unexpected ticker")

    print("\n=== V34 AUDIT ===")
    print("Hashed documents verified:", checked)
    print("Evidence offsets checked:", len(df))
    print("Five-stock universe: PASS")
    print("Relation cue and entity presence: PASS")
    print("Source provenance: PASS")
    print("Strict PIT: NOT CERTIFIED")
    print("Relation semantics: SILVER / UNVERIFIED")

SCHEMA = {
    "type": "object",
    "properties": {
        "relation_found": {"type": "boolean"},
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": sorted(RELATIONS),
                    },
                    "object": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": [
                    "subject", "relation", "object", "evidence_quote"
                ],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["relation_found", "relations", "reason"],
    "additionalProperties": False,
}

SYSTEM = """
You extract financial entity relationships from SEC filings.
Return only JSON conforming to the schema.

The CURRENT sentence is the source of evidence.
Context is provided only for understanding.

Only extract SUPPLIES_TO, CUSTOMER_OF, COMPETES_WITH,
or SAME_SECTOR if explicitly supported by supplied text.

SUPPLIES_TO: subject supplies products/services to object.
CUSTOMER_OF: subject obtains products/services from object.
COMPETES_WITH: subject directly competes with object.
SAME_SECTOR: explicitly documented shared sector classification.

Do NOT guess entity names from outside knowledge.
Do NOT infer company identity from generic "suppliers",
"customers", "foundries", "vendors" or "partners".
Do NOT invent direction.
Do NOT mistake hypothetical risks for actual relationships.
If evidence is insufficient, return relations=[].
Return the exact quoted evidence as a contiguous substring
of the CURRENT sentence, not an invented explanation.
Relations may be pre-existing; do not call them new events.
"""

def retrieve_records(limit):
    p = OUT / "relation_evidence.csv"
    if not p.exists():
        raise FileNotFoundError("Run build first")
    df = pd.read_csv(p, dtype={"accession": str}).fillna("")
    df = df.sort_values(
        ["filing_date", "ticker", "evidence_id"]
    ).reset_index(drop=True)
    return df if limit == 0 else df.head(limit)

def load_log(path):
    last = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                last[r["evidence_id"]] = r
    return last

def export_silver(records):
    labels, relations = [], []
    for record in records.values():
        if record.get("status") != "ok":
            continue
        answer = record["answer"]
        labels.append({
            "evidence_id": record["evidence_id"],
            "ticker": record["ticker"],
            "accession": record["accession"],
            "available_from": record["available_from"],
            "relation_found": len(answer["relations"]) > 0,
            "relation_count": len(answer["relations"]),
            "reason": answer["reason"],
            "model": record["model"],
            "label_source": "AI_SILVER",
            "human_verified": False,
        })
        for edge in answer["relations"]:
            relations.append({
                "evidence_id": record["evidence_id"],
                "ticker": record["ticker"],
                "accession": record["accession"],
                "available_from": record["available_from"],
                **edge,
                "model": record["model"],
                "label_source": "AI_SILVER",
                "human_verified": False,
                "strict_pit_verified": False,
            })

    pd.DataFrame(labels, columns=[
        "evidence_id", "ticker", "accession", "available_from",
        "relation_found", "relation_count", "reason",
        "model", "label_source", "human_verified"
    ]).to_csv(OUT / "silver_labels.csv", index=False)

    pd.DataFrame(relations, columns=[
        "evidence_id", "ticker", "accession", "available_from",
        "subject", "relation", "object", "evidence_quote",
        "model", "label_source", "human_verified",
        "strict_pit_verified"
    ]).to_csv(OUT / "silver_relations.csv", index=False)

def label(model, host, limit, timeout, max_tokens):
    from ollama import Client
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    # Keep one request at a time on the shared server.
    client = Client(host=host, timeout=timeout)
    rows = retrieve_records(limit)
    path = OUT / "ollama_log.jsonl"
    config = OUT / "ollama_config.json"
    requested = {
        "model": model,
        "host": host,
        "schema": "v34_1",
        "num_predict": max_tokens,
    }
    if config.exists():
        if json.loads(config.read_text()) != requested:
            raise ValueError(
                "Previous run has different model/options. "
                "Move V34 Ollama logs to a separate folder first."
            )
    else:
        write_json(config, requested)

    results = load_log(path)
    done = sum(
        results.get(str(r.evidence_id), {}).get("status") == "ok"
        for r in rows.itertuples(index=False)
    )
    print(f"Selected={len(rows)} already_ok={done}", flush=True)

    for i, r in enumerate(rows.itertuples(index=False), 1):
        eid = str(r.evidence_id)
        if results.get(eid, {}).get("status") == "ok":
            continue

        payload = {
            "reporting_ticker": r.ticker,
            "form": r.form,
            "filing_date": r.filing_date,
            "named_entity_candidates": json.loads(r.named_entities),
            "relation_cues": json.loads(r.relation_cues),
            "CURRENT_SENTENCE": r.sentence,
            "CONTEXT": r.context,
        }

        entry = {
            "evidence_id": eid,
            "ticker": r.ticker,
            "accession": str(r.accession),
            "available_from": r.available_from,
            "model": model,
        }

        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                     json.dumps(payload, ensure_ascii=False)},
                ],
                format=SCHEMA,
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": max_tokens,
                },
                stream=False,
            )
            content = response.message.content or ""
            if not content.strip():
                raise ValueError(
                    "EMPTY_CONTENT "
                    f"done_reason={getattr(response, 'done_reason', None)} "
                    f"thinking_chars={len(getattr(response.message, 'thinking', '') or '')}"
                )
            obj = json.loads(content)
            if not isinstance(obj.get("relations"), list):
                raise ValueError("Missing relations array")

            accepted = []
            rejected = []
            for edge in obj["relations"]:
                subject = normalized(edge.get("subject", ""))
                target = normalized(edge.get("object", ""))
                relation = edge.get("relation", "")
                quote = normalized(edge.get("evidence_quote", ""))

                reason = None
                if relation not in RELATIONS:
                    reason = "BAD_RELATION_TYPE"
                elif not subject or not target or subject.casefold() == target.casefold():
                    reason = "BAD_ENTITIES"
                elif len(quote) < 12 or quote not in normalized(r.sentence):
                    reason = "QUOTE_NOT_IN_SENTENCE"
                # An evidence quote alone does not prove that
                # model subject/object were correctly identified.
                if reason:
                    rejected.append({"edge": edge, "reason": reason})
                else:
                    accepted.append({
                        "subject": subject, "relation": relation,
                        "object": target, "evidence_quote": quote
                    })

            entry["status"] = "ok"
            entry["answer"] = {
                "relations": accepted,
                "reason": str(obj.get("reason", ""))[:500],
            }
            entry["rejected_edges"] = rejected
            print(
                f"[{i}/{len(rows)}] OK {r.ticker} "
                f"relations={len(accepted)} "
                f"rejected={len(rejected)}",
                flush=True
            )
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"[:500]
            print(
                f"[{i}/{len(rows)}] ERROR {entry['error']}",
                flush=True
            )

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        results[eid] = entry
        export_silver(results)

    print("\n=== V34 LABELING FINISHED ===")
    report()

def report():
    evidence = pd.read_csv(OUT / "relation_evidence.csv")
    path = OUT / "ollama_log.jsonl"
    records = load_log(path)
    successful = [
        r for r in records.values() if r.get("status") == "ok"
    ]
    relations_path = OUT / "silver_relations.csv"
    edges = (
        pd.read_csv(relations_path)
        if relations_path.exists()
        else pd.DataFrame()
    )
    summary = {
        "filings": 140,
        "retrieval_candidates": len(evidence),
        "ollama_completed": len(successful),
        "ollama_errors_current": sum(
            r.get("status") == "error" for r in records.values()
        ),
        "silver_relation_mentions": len(edges),
        "relation_counts": (
            edges["relation"].value_counts().to_dict()
            if "relation" in edges else {}
        ),
        "human_verified": False,
        "strict_pit_certified": False,
    }
    write_json(OUT / "report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["build", "audit", "label", "report"]
    )
    parser.add_argument("--per-filing", type=int, default=DEFAULT_PER_FILING)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "OLLAMA_HOST", "http://140.113.87.164:11434"
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-predict", type=int, default=768)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        if args.per_filing < 1:
            raise ValueError("--per-filing must be >=1")
        build(args.per_filing)
    elif args.command == "audit":
        audit()
    elif args.command == "label":
        label(
            args.model, args.host, args.limit,
            args.timeout, args.num_predict
        )
    elif args.command == "report":
        report()

if __name__ == "__main__":
    main()
