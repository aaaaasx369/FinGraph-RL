#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from collections import Counter

import ollama


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "rag/data/processed/sec/sec_chunks.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "kg/output/portfolio_kg_ollama_smoke.jsonl"
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

MAX_CHUNKS = int(
    os.getenv(
        "KG_MAX_CHUNKS",
        "20",
    )
)

client = ollama.Client(
    host="http://140.113.87.164:11434"
)


TARGET_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
}

TARGET_FORMS = {
    "10-K",
    "10-Q",
    "8-K",
}


RELATIONS = [
    "COMPETES_WITH",
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "PARTNERS_WITH",
    "DEPENDS_ON",
    "ACQUIRES",
    "INVESTS_IN",
    "GUIDES_UP",
    "GUIDES_DOWN",
    "FACES_RISK_FROM",
    "BENEFITS_FROM",
]

ENTITY_TYPES = [
    "COMPANY",
    "PRODUCT",
    "SERVICE",
    "MARKET",
    "INDUSTRY",
    "COUNTRY",
    "REGULATOR",
    "TECHNOLOGY",
    "COMMODITY",
    "RISK_FACTOR",
]


KEYWORDS = [
    "compete",
    "competitor",
    "competition",

    "supplier",
    "supply",
    "vendor",

    "customer",
    "client",

    "partner",
    "partnership",
    "collaborat",

    "depend",
    "rely",
    "reliance",

    "acquire",
    "acquisition",
    "merge",

    "invest",
    "investment",

    "guidance",
    "outlook",
    "forecast",

    "risk",
    "exposure",
    "shortage",

    "benefit",
    "advantage",
]


def clean_text(x):
    if x is None:
        return ""

    x = str(x)

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    return x.strip()


def passes_prefilter(text):
    t = text.lower()

    return any(
        kw in t
        for kw in KEYWORDS
    )


def normalize_company(name):
    if name is None:
        return None

    raw = str(name).strip()

    x = raw.lower()

    aliases = {
        "apple": "AAPL",
        "apple inc.": "AAPL",
        "apple inc": "AAPL",

        "microsoft": "MSFT",
        "microsoft corporation": "MSFT",
        "microsoft corp.": "MSFT",

        "nvidia": "NVDA",
        "nvidia corporation": "NVDA",
        "nvidia corp.": "NVDA",

        "amazon": "AMZN",
        "amazon.com": "AMZN",
        "amazon.com, inc.": "AMZN",

        "alphabet": "GOOGL",
        "alphabet inc.": "GOOGL",
        "google": "GOOGL",
    }

    return aliases.get(
        x,
        raw,
    )


def parse_json_response(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.S,
    )

    if not match:
        raise ValueError(
            "No JSON object found:\n"
            + text[:1000]
        )

    return json.loads(
        match.group(0)
    )


def ollama_generate(prompt):

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0.0,
        },
    )

    return response["message"]["content"]


def extract_triples(row):

    prompt = f"""
You are extracting a portfolio-relevant financial knowledge graph
from SEC filings.

Use ONLY the source text below.

Allowed relations:
{", ".join(RELATIONS)}

Allowed entity types:
{", ".join(ENTITY_TYPES)}

Return JSON only.

Exact format:

{{
  "triples": [
    {{
      "subject": "...",
      "subject_type": "COMPANY",
      "relation": "COMPETES_WITH",
      "object": "...",
      "object_type": "COMPANY",
      "evidence": "short supporting phrase",
      "confidence": 0.95
    }}
  ]
}}

Rules:

1. Extract only facts explicitly supported by the source text.
2. Never invent an entity.
3. Never invent a relation.
4. Use only allowed relation labels.
5. Relation direction matters.

Examples:

"NVIDIA relies on TSMC"
-> NVDA DEPENDS_ON TSMC

"TSMC supplies NVIDIA"
-> TSMC SUPPLIES_TO NVDA

"NVIDIA buys manufacturing services from TSMC"
-> NVDA CUSTOMER_OF TSMC

6. Only extract portfolio-relevant relations.
7. If nothing useful exists, return:
{{"triples":[]}}
8. confidence must be 0 to 1.
9. Keep evidence concise.
10. Do not output explanations outside JSON.

Metadata:
ticker: {row.get("ticker")}
form: {row.get("form")}
published_at: {row.get("published_at")}
doc_id: {row.get("doc_id")}
chunk_id: {row.get("chunk_id")}

SOURCE TEXT:

{row["text"]}
"""

    raw = ollama_generate(
        prompt
    )

    parsed = parse_json_response(
        raw
    )

    triples = parsed.get(
        "triples",
        [],
    )

    if not isinstance(triples, list):
        raise ValueError(
            "triples must be a list"
        )

    return triples


def validate_triples(triples):

    valid = []

    seen = set()

    for t in triples:

        subject = t.get("subject")
        obj = t.get("object")

        relation = str(
            t.get(
                "relation",
                "",
            )
        ).upper()

        subject_type = str(
            t.get(
                "subject_type",
                "",
            )
        ).upper()

        object_type = str(
            t.get(
                "object_type",
                "",
            )
        ).upper()

        if not subject or not obj:
            continue

        if relation not in RELATIONS:
            continue

        if subject_type not in ENTITY_TYPES:
            continue

        if object_type not in ENTITY_TYPES:
            continue

        subject = normalize_company(
            subject
        )

        obj = normalize_company(
            obj
        )

        if str(subject).lower() == str(obj).lower():
            continue

        try:
            confidence = float(
                t.get(
                    "confidence",
                    0.0,
                )
            )
        except Exception:
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        key = (
            str(subject).lower(),
            relation,
            str(obj).lower(),
        )

        if key in seen:
            continue

        seen.add(key)

        valid.append(
            {
                "subject":
                    subject,

                "subject_type":
                    subject_type,

                "relation":
                    relation,

                "object":
                    obj,

                "object_type":
                    object_type,

                "evidence":
                    clean_text(
                        t.get(
                            "evidence",
                            "",
                        )
                    ),

                "confidence":
                    confidence,
            }
        )

    return valid


def reflect_triples(
    row,
    triples,
):

    if not triples:
        return []

    prompt = f"""
You are a strict financial knowledge graph reviewer.

Use ONLY the SEC source text below.

Review the candidate triples for:

- factual support
- relation direction
- entity correctness
- relation correctness
- unsupported inference
- duplicates

Allowed relations:
{", ".join(RELATIONS)}

Allowed entity types:
{", ".join(ENTITY_TYPES)}

You may remove incorrect triples.
You may correct a triple only when the source clearly supports it.

Return JSON only.

Format:

{{
  "triples": [
    {{
      "subject": "...",
      "subject_type": "...",
      "relation": "...",
      "object": "...",
      "object_type": "...",
      "evidence": "...",
      "confidence": 0.95
    }}
  ]
}}

SOURCE TEXT:

{row["text"]}

CANDIDATE TRIPLES:

{json.dumps(
    triples,
    ensure_ascii=False,
    indent=2,
)}
"""

    raw = ollama_generate(
        prompt
    )

    parsed = parse_json_response(
        raw
    )

    reviewed = parsed.get(
        "triples",
        [],
    )

    return validate_triples(
        reviewed
    )


# ============================================================
# Load chunks
# ============================================================

rows = []

scanned = 0
prefiltered = 0

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        obj = json.loads(line)

        scanned += 1

        ticker = str(
            obj.get(
                "ticker",
                "",
            )
        ).upper()

        form = str(
            obj.get(
                "form",
                "",
            )
        ).upper()

        text = clean_text(
            obj.get(
                "text",
                "",
            )
        )

        if ticker not in TARGET_TICKERS:
            continue

        if form not in TARGET_FORMS:
            continue

        if len(text) < 200:
            continue

        if not passes_prefilter(
            text
        ):
            continue

        prefiltered += 1

        obj["ticker"] = ticker
        obj["form"] = form
        obj["text"] = text

        rows.append(obj)

        if len(rows) >= MAX_CHUNKS:
            break


print("=" * 100)
print("PORTFOLIO KG OLLAMA SMOKE TEST")
print("=" * 100)

print(
    "Ollama host:",
    "http://140.113.87.164:11434",
)

print(
    "Model:",
    MODEL,
)

print(
    "Scanned chunks:",
    scanned,
)

print(
    "Prefiltered candidate chunks:",
    prefiltered,
)

print(
    "Selected smoke chunks:",
    len(rows),
)


# ============================================================
# Run
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

all_edges = []


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as out:

    for i, row in enumerate(
        rows,
        start=1,
    ):

        print("\n" + "-" * 100)

        print(
            f"[{i}/{len(rows)}] "
            f"{row.get('ticker')} "
            f"{row.get('form')} "
            f"{row.get('published_at')}"
        )

        print(
            "chunk_id:",
            row.get(
                "chunk_id"
            )
        )

        try:

            candidates = extract_triples(
                row
            )

            validated = validate_triples(
                candidates
            )

            reviewed = reflect_triples(
                row,
                validated,
            )

            print(
                "raw:",
                len(candidates),
                "| validated:",
                len(validated),
                "| reflected:",
                len(reviewed),
            )


            for t in reviewed:

                edge = {
                    "ticker":
                        row.get(
                            "ticker"
                        ),

                    "published_at":
                        row.get(
                            "published_at"
                        ),

                    "form":
                        row.get(
                            "form"
                        ),

                    "doc_id":
                        row.get(
                            "doc_id"
                        ),

                    "chunk_id":
                        row.get(
                            "chunk_id"
                        ),

                    "subject":
                        t["subject"],

                    "subject_type":
                        t["subject_type"],

                    "relation":
                        t["relation"],

                    "object":
                        t["object"],

                    "object_type":
                        t["object_type"],

                    "evidence":
                        t["evidence"],

                    "confidence":
                        t["confidence"],
                }

                out.write(
                    json.dumps(
                        edge,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                all_edges.append(
                    edge
                )

                print(
                    f"  {edge['subject']} "
                    f"--{edge['relation']}--> "
                    f"{edge['object']} "
                    f"| conf={edge['confidence']:.2f}"
                )


        except Exception as e:

            print(
                "ERROR:",
                repr(e),
            )


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Final edges:",
    len(all_edges),
)

print(
    "Output:",
    OUTPUT_PATH,
)


if all_edges:

    counts = Counter(
        x["relation"]
        for x in all_edges
    )

    print("\nRELATION COUNTS")

    for rel, count in (
        counts.most_common()
    ):

        print(
            f"{rel:20s}",
            count,
        )


    print("\nFIRST 30 EDGES")

    for x in all_edges[:30]:

        print(
            f"{x['published_at']} | "
            f"{x['ticker']} | "
            f"{x['form']:4s} | "
            f"{x['subject']} "
            f"--{x['relation']}--> "
            f"{x['object']} "
            f"| {x['confidence']:.2f}"
        )
