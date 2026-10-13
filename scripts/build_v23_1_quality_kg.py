#!/usr/bin/env python3
"""
V23.1 Financial KG Quality Layer

Input:
  V22 relation_events.parquet
  V20 SEC filings manifest and cached documents

Output:
  normalized_events.parquet
  company_edges.parquet
  relation_quality_audit.csv
  daily_graph_features.csv
  report.json

Safety:
- Strict entity-type gate for company relations
- DEPENDS_ON is not CUSTOMER_OF
- No forced relation type for uncertain wording
- SEC lexical support is candidate-only
- PDF/accession identity is not assumed verified
- Event dates must precede trading dates
- No new FinRL training
"""

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

ALIASES = {
    "AAPL": [
        "apple", "apple inc", "apple computer"
    ],
    "MSFT": [
        "microsoft", "microsoft corporation"
    ],
    "NVDA": [
        "nvidia", "nvidia corporation"
    ],
    "AMZN": [
        "amazon", "amazon com",
        "amazon com inc", "amazon.com inc"
    ],
    "GOOGL": [
        "google", "google llc",
        "alphabet", "alphabet inc"
    ],
    "TSMC": [
        "tsmc",
        "taiwan semiconductor manufacturing",
        "taiwan semiconductor manufacturing company",
        "taiwan semiconductor manufacturing company limited",
        "taiwan semiconductor"
    ],
    "SAMSUNG": [
        "samsung electronics",
        "samsung electronics co ltd",
        "samsung electronics company limited"
    ],
    "MELLANOX": [
        "mellanox",
        "mellanox technologies",
        "mellanox technologies ltd"
    ],
    "AMD": [
        "amd",
        "advanced micro devices",
        "advanced micro devices inc"
    ],
    "INTC": [
        "intel", "intel corporation"
    ],
    "QCOM": [
        "qualcomm", "qualcomm incorporated"
    ],
    "AVGO": [
        "broadcom", "broadcom inc"
    ],
    "MU": [
        "micron", "micron technology"
    ],
    "ASML": [
        "asml", "asml holding"
    ],
    "TSLA": [
        "tesla", "tesla inc"
    ],
    "META": [
        "meta platforms",
        "facebook", "facebook inc"
    ],
    "IBM": [
        "ibm",
        "international business machines"
    ],
    "ORCL": [
        "oracle", "oracle corporation"
    ],
    "CSCO": [
        "cisco", "cisco systems"
    ],
    "DELL": [
        "dell", "dell technologies"
    ],
    "HPQ": [
        "hp inc",
        "hewlett packard"
    ],
    "ARM": [
        "arm limited",
        "arm holdings"
    ],
    "FOXCONN": [
        "foxconn",
        "hon hai precision industry",
        "hon hai"
    ],
}

COMPANY_RELATIONS = {
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
    "PARTNERS_WITH",
}

RELATION_RULES = [
    ("CUSTOMER_OF", [
        r"\bcustomer_of\b",
        r"\bpurchases?_from\b",
        r"\bprocures?_from\b",
        r"\bbuys?_from\b",
    ]),
    ("SUPPLIES_TO", [
        r"\bsupplies?_to\b",
        r"\bprovides?_to\b",
        r"\bsells?_to\b",
    ]),
    ("COMPETES_WITH", [
        r"\bcompetes?_with\b",
        r"\bcompetitor(?:s)?\b",
        r"\brival(?:s)?\b",
    ]),
    ("SAME_SECTOR", [
        r"\bsame_sector\b",
        r"\bsame_industry\b",
    ]),
    ("PARTNERS_WITH", [
        r"\bpartners?_with\b",
        r"\bcollaborates?_with\b",
        r"\bjoint_venture\b",
    ]),
    ("DEPENDS_ON", [
        r"\bdepends?_on\b",
        r"\breli(?:es|ed)_on\b",
        r"\bdependent_on\b",
    ]),
    ("MANUFACTURED_BY", [
        r"\bmanufactured?_by\b",
        r"\bfabricated?_by\b",
    ]),
    ("ACQUIRED", [
        r"\bacquired?\b",
        r"\bacquisition_of\b",
    ]),
    ("OPERATES_IN", [
        r"\boperates?_in\b",
    ]),
    ("DISCLOSES", [
        r"\bdiscloses?\b",
        r"\breports?\b",
    ]),
    ("EXPOSED_TO", [
        r"\bimpacted?_by\b",
        r"\bexposed?_to\b",
        r"\bfaces?\b",
    ]),
]

EVIDENCE_TERMS = {
    "SUPPLIES_TO": [
        "supplies", "supplier", "provides", "sells",
    ],
    "CUSTOMER_OF": [
        "customer", "purchase", "procure", "buys",
    ],
    "COMPETES_WITH": [
        "competitor", "competes", "competition", "rival",
    ],
    "SAME_SECTOR": [
        "industry", "sector",
    ],
    "PARTNERS_WITH": [
        "partner", "collaborat", "joint venture",
    ],
    "DEPENDS_ON": [
        "depend", "rely", "reliance",
    ],
    "MANUFACTURED_BY": [
        "manufactur", "fabricat", "foundry",
    ],
    "ACQUIRED": [
        "acquir", "acquisition",
    ],
}

def norm(value):
    if pd.isna(value):
        return ""
    x = str(value).casefold().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return " ".join(x.split())

def sha(*parts):
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode()
    ).hexdigest()[:24]

def make_alias_index():
    index = defaultdict(set)
    for company, names in ALIASES.items():
        for name in names + [company]:
            index[norm(name)].add(company)
    return dict(index)

ALIAS_INDEX = make_alias_index()

def resolve(value):
    key = norm(value)
    if not key:
        return "", "EMPTY", ""

    matches = ALIAS_INDEX.get(key, set())

    if len(matches) == 1:
        company = next(iter(matches))
        return "COMPANY:" + company, "COMPANY", company

    if len(matches) > 1:
        return "AMBIGUOUS:" + sha(key), "AMBIGUOUS", ""

    return "CONCEPT:" + sha(key), "CONCEPT", ""

def classify_relation(text):
    value = norm(text).replace(" ", "_")

    matches = [
        name for name, patterns in RELATION_RULES
        if any(re.search(p, value) for p in patterns)
    ]

    if len(matches) == 1:
        return matches[0], "RULE_EXACT"

    if len(matches) > 1:
        return "AMBIGUOUS", "MULTIPLE_RULES"

    return "OTHER", "UNMAPPED"

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

def as_text(raw):
    source = raw.decode("utf-8", errors="replace")
    parser = TextExtractor()
    try:
        parser.feed(source)
        return " ".join(" ".join(parser.parts).split())
    except Exception:
        return " ".join(
            html.unescape(
                re.sub(r"<[^>]*>", " ", source)
            ).split()
        )

def load_sec_manifest(path):
    manifest = pd.read_csv(path)
    required = [
        "ticker", "accession",
        "primary_document", "source_sha256",
        "text_verified",
    ]
    missing = [x for x in required if x not in manifest]
    if missing:
        raise ValueError(
            f"SEC manifest missing {missing}"
        )

    manifest["ticker"] = (
        manifest["ticker"].astype(str).str.upper()
    )

    return manifest

def get_document(row, cache, document_store):
    """
    Load a verified cached SEC document once per accession.
    Returns an empty list on missing/invalid document.
    """
    key = (
        str(row["ticker"]),
        str(row["accession"]),
    )

    if key in document_store:
        return document_store[key]

    if int(row["text_verified"]) != 1:
        document_store[key] = []
        return []

    name = (
        f"{row['ticker']}_{row['accession']}_"
        + re.sub(
            r"[^A-Za-z0-9_.-]", "_",
            str(row["primary_document"])
        )
    )

    path = cache / name

    if not path.is_file():
        document_store[key] = []
        return []

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(
        row["source_sha256"]
    ):
        document_store[key] = []
        return []

    text = as_text(raw)
    # Match within local sections rather than an entire filing.
    chunk_size, stride = 1800, 1400
    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), stride)
        if len(text[i:i + chunk_size]) >= 120
    ]
    document_store[key] = chunks
    return chunks

def mentions(text, alias):
    """
    Avoid loose substring matching for short company names.
    """
    phrase = norm(alias)
    if not phrase:
        return False
    clean = norm(text)
    return bool(
        re.search(
            r"(?<![a-z0-9])"
            + re.escape(phrase)
            + r"(?![a-z0-9])",
            clean
        )
    )

def entity_mentioned(text, value, company_id):
    if company_id:
        names = ALIASES.get(company_id, []) + [company_id]
        return any(
            mentions(text, name)
            for name in names
        )
    return mentions(text, value)


def evidence_status(row, manifest_lookup, cache, store):
    if not row["typed_company_relation"]:
        return "NOT_ELIGIBLE", ""

    accession = str(row.get("accession", ""))
    ticker = str(row["ticker"])

    record = manifest_lookup.get(
        (ticker, accession)
    )
    if record is None:
        return "NO_DOCUMENT", ""

    pieces = get_document(
        record, cache, store
    )
    if not pieces:
        return "NO_DOCUMENT", ""

    entity = str(row["entity"])
    target = str(row["target"])
    terms = EVIDENCE_TERMS.get(
        row["relation_normalized"], []
    )

    co_mention = ""

    for chunk in pieces:
        if not (
            entity_mentioned(
                chunk, entity, row["source_company"]
            )
            and entity_mentioned(
                chunk, target, row["target_company"]
            )
        ):
            continue

        if not co_mention:
            co_mention = chunk[:450]

        if any(
            term in chunk.casefold()
            for term in terms
        ):
            # Co-mention + lexical relation hint.
            # Direction and truth are NOT established.
            return (
                "TEXT_SUPPORT_CANDIDATE",
                chunk[:450]
            )

    if co_mention:
        return "CO_MENTION_ONLY", co_mention

    return "UNSUPPORTED", ""

def build(tier, args):
    path = ROOT / (
        f"outputs/v22/{tier}/relation_events.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    events = pd.read_parquet(path)

    required = [
        "ticker", "entity", "relationship", "target",
        "event_date", "event_id", "source_file",
        "link_status", "accession",
    ]
    missing = [c for c in required if c not in events]
    if missing:
        raise ValueError(
            f"V22 missing {missing}"
        )

    events["ticker"] = (
        events["ticker"].astype(str).str.upper()
    )
    events["event_date"] = pd.to_datetime(
        events["event_date"], errors="raise"
    ).dt.normalize()

    source_nodes = [
        resolve(x) for x in events["entity"]
    ]
    target_nodes = [
        resolve(x) for x in events["target"]
    ]

    events["source_node"] = [
        x[0] for x in source_nodes
    ]
    events["source_type"] = [
        x[1] for x in source_nodes
    ]
    events["source_company"] = [
        x[2] for x in source_nodes
    ]
    events["target_node"] = [
        x[0] for x in target_nodes
    ]
    events["target_type"] = [
        x[1] for x in target_nodes
    ]
    events["target_company"] = [
        x[2] for x in target_nodes
    ]

    relations = [
        classify_relation(x)
        for x in events["relationship"]
    ]

    events["relation_normalized"] = [
        x[0] for x in relations
    ]
    events["relation_quality"] = [
        x[1] for x in relations
    ]

    events["typed_company_relation"] = (
        events["source_type"].eq("COMPANY")
        & events["target_type"].eq("COMPANY")
        & events["relation_normalized"].isin(
            COMPANY_RELATIONS
        )
    )

    events["company_to_concept"] = (
        events["source_type"].eq("COMPANY")
        & events["target_type"].eq("CONCEPT")
    )

    # Recompute normalized semantic edge IDs.
    events["semantic_edge_id"] = [
        sha(a, b, c, d)
        for a, b, c, d in zip(
            events["ticker"],
            events["source_node"],
            events["relation_normalized"],
            events["target_node"],
        )
    ]

    events["candidate_link"] = (
        events["link_status"].eq("CANDIDATE_MATCH")
    )

    # No SEC text can upgrade an uncertain PDF identity
    # to verified without matching original PDF content.
    events["pdf_identity_verified"] = False

    manifest = load_sec_manifest(
        args.manifest
    )

    lookup = {
        (str(r["ticker"]), str(r["accession"])): r
        for _, r in manifest.iterrows()
    }

    cache = Path(args.cache)
    store = {}

    evidence = []
    snippets = []

    eligible = int(
        events["typed_company_relation"].sum()
    )

    print(
        f"{tier}: {len(events)} events, "
        f"{eligible} typed company-company candidates",
        flush=True,
    )

    for _, row in events.iterrows():
        if row["candidate_link"]:
            status, snippet = evidence_status(
                row, lookup, cache, store
            )
        else:
            status, snippet = (
                "UNVERIFIED_LINK", ""
            )
        evidence.append(status)
        snippets.append(snippet)

    events["evidence_status"] = evidence
    events["evidence_excerpt"] = snippets

    events = events.sort_values(
        [
            "semantic_edge_id",
            "event_date",
            "event_id",
        ]
    ).reset_index(drop=True)

    events["semantic_observation_number"] = (
        events.groupby(
            "semantic_edge_id"
        ).cumcount() + 1
    )

    events["semantic_event_type"] = np.where(
        events["semantic_observation_number"].eq(1),
        "FIRST_OBSERVED",
        "REOBSERVED",
    )

    # Conservative qualified edge:
    # both endpoints are companies,
    # relation belongs to permitted semantics,
    # and some document text support exists.
    events["quality_gate_pass"] = (
        events["typed_company_relation"]
        & events["candidate_link"]
        & events["evidence_status"].eq(
            "TEXT_SUPPORT_CANDIDATE"
        )
    )

    out = ROOT / "outputs/v23_1" / tier
    out.mkdir(parents=True, exist_ok=True)

    events.to_parquet(
        out / "normalized_events.parquet",
        index=False
    )

    company_edges = events[
        events["typed_company_relation"]
    ].copy()

    company_edges.to_parquet(
        out / "company_edges.parquet",
        index=False
    )

    audit = (
        events.groupby(
            [
                "relation_normalized",
                "source_type",
                "target_type",
                "evidence_status",
            ],
            dropna=False
        )
        .agg(
            event_count=("event_id", "size"),
            unique_edges=("semantic_edge_id", "nunique"),
            quality_pass=("quality_gate_pass", "sum"),
        )
        .reset_index()
    )

    audit.to_csv(
        out / "relation_quality_audit.csv",
        index=False
    )

    report = {
        "tier": tier,
        "raw_events": int(len(events)),
        "semantic_edges": int(
            events["semantic_edge_id"].nunique()
        ),
        "company_company_events": int(
            events["typed_company_relation"].sum()
        ),
        "company_concept_events": int(
            events["company_to_concept"].sum()
        ),
        "quality_gate_events": int(
            events["quality_gate_pass"].sum()
        ),
        "evidence_status": {
            str(k): int(v)
            for k, v in events[
                "evidence_status"
            ].value_counts().items()
        },
        "relation_types": {
            str(k): int(v)
            for k, v in events[
                "relation_normalized"
            ].value_counts().items()
        },
        "pdf_identity_verified": False,
        "strict_pit_approved": False,
        "note": (
            "Text support is lexical and may not "
            "verify direction or factual correctness."
        ),
    }

    (out / "report.json").write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V23.1 QUALITY RESULTS ===")
    print(json.dumps(report, indent=2))
    print("Saved:", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tiers", nargs="+",
        choices=["conservative", "inclusive"],
        default=["conservative", "inclusive"]
    )
    p.add_argument(
        "--manifest",
        default="outputs/v20/sec_filings_manifest.csv"
    )
    p.add_argument(
        "--cache",
        default="outputs/v20/sec_cache/documents"
    )
    args = p.parse_args()

    args.manifest = str(
        ROOT / args.manifest
    )
    args.cache = str(
        ROOT / args.cache
    )

    for tier in args.tiers:
        build(tier, args)

if __name__ == "__main__":
    main()
