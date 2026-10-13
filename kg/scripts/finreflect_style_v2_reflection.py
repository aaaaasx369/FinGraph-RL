#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from collections import Counter

import ollama


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "rag/data/processed/sec/sec_chunks.jsonl"

OUTPUT_DIR = ROOT / "kg/output/finreflect_style_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "reflection_v2.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_edges.jsonl"

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://140.113.87.164:11434",
)

MAX_CHUNKS = int(os.getenv("KG_MAX_CHUNKS", "20"))

client = ollama.Client(host=OLLAMA_HOST)


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


# ============================================================
# V2 schema
# ============================================================

RELATIONS = [
    "COMPETES_WITH",
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "PARTNERS_WITH",
    "DEPENDS_ON",
    "ACQUIRES",
    "INVESTS_IN",
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


SIGNATURES = {
    "COMPETES_WITH": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "SUPPLIES_TO": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "CUSTOMER_OF": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "PARTNERS_WITH": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "ACQUIRES": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "INVESTS_IN": {
        "subject": {"COMPANY"},
        "object": {"COMPANY"},
    },

    "DEPENDS_ON": {
        "subject": {"COMPANY"},
        "object": {
            "COMPANY",
            "PRODUCT",
            "SERVICE",
            "TECHNOLOGY",
            "COMMODITY",
        },
    },

    "FACES_RISK_FROM": {
        "subject": {"COMPANY"},
        "object": {"RISK_FACTOR"},
    },

    "BENEFITS_FROM": {
        "subject": {"COMPANY"},
        "object": {
            "PRODUCT",
            "SERVICE",
            "TECHNOLOGY",
            "MARKET",
        },
    },
}


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
    "risk",
    "exposure",
    "shortage",
    "benefit",
    "advantage",
]


GENERIC_ENTITY_BLACKLIST = {
    "customers",
    "customer",
    "suppliers",
    "supplier",
    "vendors",
    "vendor",
    "products",
    "product",
    "services",
    "service",
    "partners",
    "partner",
    "competitors",
    "competitor",
    "market",
    "markets",
    "industry",
    "industries",
    "company",
    "companies",
    "business",
    "businesses",
}


ALIASES = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",

    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "microsoft corp": "MSFT",
    "microsoft corp.": "MSFT",

    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "nvidia corp": "NVDA",
    "nvidia corp.": "NVDA",

    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "amazon.com inc": "AMZN",
    "amazon.com, inc.": "AMZN",

    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "alphabet inc.": "GOOGL",
    "google": "GOOGL",
    "google llc": "GOOGL",
}


def clean_text(x):
    if x is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(x),
    ).strip()


def normalize_entity(x):
    x = clean_text(x)

    if not x:
        return x

    return ALIASES.get(
        x.lower(),
        x,
    )


def passes_prefilter(text):
    t = text.lower()

    return any(
        kw in t
        for kw in KEYWORDS
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


def call_qwen(prompt):
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


def signature_valid(t):
    relation = t["relation"]

    if relation not in SIGNATURES:
        return False

    sig = SIGNATURES[relation]

    if t["subject_type"] not in sig["subject"]:
        return False

    if t["object_type"] not in sig["object"]:
        return False

    return True


def generic_entity_invalid(t):
    subject = str(
        t["subject"]
    ).strip().lower()

    obj = str(
        t["object"]
    ).strip().lower()

    if subject in GENERIC_ENTITY_BLACKLIST:
        return True

    if obj in GENERIC_ENTITY_BLACKLIST:
        return True

    return False


def structural_validate(triples):
    valid = []
    rejected = []
    seen = set()

    if not isinstance(triples, list):
        return valid, rejected

    for t in triples:

        if not isinstance(t, dict):
            continue

        subject = normalize_entity(
            t.get("subject")
        )

        obj = normalize_entity(
            t.get("object")
        )

        relation = str(
            t.get(
                "relation",
                "",
            )
        ).upper().strip()

        subject_type = str(
            t.get(
                "subject_type",
                "",
            )
        ).upper().strip()

        object_type = str(
            t.get(
                "object_type",
                "",
            )
        ).upper().strip()

        evidence = clean_text(
            t.get(
                "evidence",
                "",
            )
        )

        try:
            confidence = float(
                t.get(
                    "confidence",
                    0,
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

        clean = {
            "subject": subject,
            "subject_type": subject_type,
            "relation": relation,
            "object": obj,
            "object_type": object_type,
            "evidence": evidence,
            "confidence": confidence,
        }

        reason = None

        if not subject or not obj:
            reason = "missing_entity"

        elif relation not in RELATIONS:
            reason = "invalid_relation"

        elif subject_type not in ENTITY_TYPES:
            reason = "invalid_subject_type"

        elif object_type not in ENTITY_TYPES:
            reason = "invalid_object_type"

        elif str(subject).lower() == str(obj).lower():
            reason = "self_loop"

        elif not signature_valid(clean):
            reason = "signature_violation"

        elif generic_entity_invalid(clean):
            reason = "generic_entity"

        key = (
            str(subject).lower(),
            relation,
            str(obj).lower(),
        )

        if reason is None and key in seen:
            reason = "duplicate"

        if reason is not None:
            rejected.append(
                {
                    **clean,
                    "reject_reason": reason,
                }
            )
            continue

        seen.add(key)
        valid.append(clean)

    return valid, rejected


# ============================================================
# Extractor
# ============================================================

def extract_candidates(row):

    prompt = f"""
You are extracting a portfolio-relevant financial
knowledge graph from an SEC filing.

Extract only relations EXPLICITLY supported by the source.

Allowed relations:

COMPETES_WITH
- COMPANY -> COMPANY
- explicit competition only

SUPPLIES_TO
- COMPANY -> COMPANY
- subject supplies products/services/components to object

CUSTOMER_OF
- COMPANY -> COMPANY
- subject purchases from / is customer of object

PARTNERS_WITH
- COMPANY -> COMPANY
- explicit partnership or collaboration

DEPENDS_ON
- COMPANY -> COMPANY/PRODUCT/SERVICE/TECHNOLOGY/COMMODITY
- explicit dependence, reliance, or critical dependency

ACQUIRES
- COMPANY -> COMPANY
- explicit completed or announced acquisition

INVESTS_IN
- COMPANY -> COMPANY
- explicit equity or strategic investment in another company

FACES_RISK_FROM
- COMPANY -> RISK_FACTOR
- explicit material risk

BENEFITS_FROM
- COMPANY -> PRODUCT/SERVICE/TECHNOLOGY/MARKET
- explicit benefit or favorable exposure


Important restrictions:

1. Do not infer relations merely because two entities appear together.
2. Do not use generic entities such as "customers", "suppliers",
   "products", or "services".
3. Do not create ACQUIRES or INVESTS_IN unless the text explicitly
   describes an acquisition or investment.
4. Do not invent companies.
5. Use only the allowed relation labels.
6. Relation direction must match the source.
7. If uncertain, omit the triple.
8. Prefer precision over recall.
9. Return JSON only.


Format:

{{
  "triples": [
    {{
      "subject": "...",
      "subject_type": "...",
      "relation": "...",
      "object": "...",
      "object_type": "...",
      "evidence": "short exact supporting phrase",
      "confidence": 0.95
    }}
  ]
}}


Metadata:
ticker: {row.get("ticker")}
form: {row.get("form")}
published_at: {row.get("published_at")}
chunk_id: {row.get("chunk_id")}


SOURCE:

{row["text"]}
"""

    raw = call_qwen(prompt)

    parsed = parse_json_response(raw)

    return parsed.get(
        "triples",
        [],
    )


# ============================================================
# Evidence verifier
# ============================================================

def verify_edge(row, edge):

    prompt = f"""
You are verifying ONE candidate financial KG edge.

Use ONLY the SEC source text.

Candidate claim:

subject:
{edge["subject"]}

relation:
{edge["relation"]}

object:
{edge["object"]}


Question:

Does the source text EXPLICITLY support this exact relation
with this direction?

Do not accept:
- co-occurrence
- vague association
- world knowledge
- likely relationships
- inferred business relationships

For ACQUIRES:
the text must explicitly state acquisition / acquired / acquisition.

For INVESTS_IN:
the text must explicitly state an investment or ownership stake.

For SUPPLIES_TO:
the source must clearly indicate supplier direction.

For CUSTOMER_OF:
the source must clearly indicate customer/purchasing direction.

For COMPETES_WITH:
explicit competitive relationship is required.

For PARTNERS_WITH:
explicit partnership/collaboration is required.

For DEPENDS_ON:
explicit reliance/dependence is required.

For FACES_RISK_FROM:
explicit risk exposure is required.

For BENEFITS_FROM:
explicit benefit/favorable exposure is required.


Return JSON only:

{{
  "supported": true,
  "evidence": "exact or near-exact supporting phrase",
  "reason": "brief reason",
  "confidence": 0.95
}}

or

{{
  "supported": false,
  "evidence": "",
  "reason": "why the claim is unsupported",
  "confidence": 0.0
}}


SOURCE TEXT:

{row["text"]}
"""

    raw = call_qwen(prompt)

    parsed = parse_json_response(raw)

    supported = bool(
        parsed.get(
            "supported",
            False,
        )
    )

    evidence = clean_text(
        parsed.get(
            "evidence",
            "",
        )
    )

    reason = clean_text(
        parsed.get(
            "reason",
            "",
        )
    )

    try:
        confidence = float(
            parsed.get(
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

    return {
        "supported": supported,
        "evidence": evidence,
        "reason": reason,
        "confidence": confidence,
    }


# ============================================================
# Load chunks
# ============================================================

rows = []

with open(
    INPUT_PATH,
    encoding="utf-8",
) as f:

    for line in f:

        x = json.loads(line)

        ticker = str(
            x.get(
                "ticker",
                "",
            )
        ).upper()

        form = str(
            x.get(
                "form",
                "",
            )
        ).upper()

        text = clean_text(
            x.get(
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

        if not passes_prefilter(text):
            continue

        x["ticker"] = ticker
        x["form"] = form
        x["text"] = text

        rows.append(x)

        if len(rows) >= MAX_CHUNKS:
            break


print("=" * 100)
print("FINREFLECT-STYLE V2")
print("HARD SCHEMA + EDGE-LEVEL EVIDENCE VERIFICATION")
print("=" * 100)

print("Model:", MODEL)
print("Host:", OLLAMA_HOST)
print("Chunks:", len(rows))


all_final = []
all_rejected = []


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as out, open(
    REJECTED_PATH,
    "w",
    encoding="utf-8",
) as rej:

    for i, row in enumerate(
        rows,
        start=1,
    ):

        print("\n" + "-" * 100)

        print(
            f"[{i}/{len(rows)}] "
            f"{row.get('ticker')} | "
            f"{row.get('form')} | "
            f"{row.get('published_at')}"
        )

        try:
            raw_candidates = extract_candidates(
                row
            )

        except Exception as e:
            print(
                "EXTRACTION ERROR:",
                repr(e),
            )
            continue

        candidates, structural_rejected = (
            structural_validate(
                raw_candidates
            )
        )

        print(
            "raw candidates:",
            len(raw_candidates),
            "| after hard schema:",
            len(candidates),
            "| structurally rejected:",
            len(structural_rejected),
        )

        for x in structural_rejected:

            record = {
                "ticker": row.get("ticker"),
                "published_at": row.get("published_at"),
                "form": row.get("form"),
                "doc_id": row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                **x,
            }

            rej.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            all_rejected.append(record)


        for edge in candidates:

            try:
                verification = verify_edge(
                    row,
                    edge,
                )

            except Exception as e:
                print(
                    "VERIFY ERROR:",
                    repr(e),
                )
                continue

            if not verification[
                "supported"
            ]:

                rejected_record = {
                    "ticker": row.get("ticker"),
                    "published_at": row.get("published_at"),
                    "form": row.get("form"),
                    "doc_id": row.get("doc_id"),
                    "chunk_id": row.get("chunk_id"),

                    **edge,

                    "reject_reason":
                        "evidence_verification_failed",

                    "verification_reason":
                        verification["reason"],
                }

                rej.write(
                    json.dumps(
                        rejected_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                all_rejected.append(
                    rejected_record
                )

                print(
                    "  REJECT",
                    edge["subject"],
                    edge["relation"],
                    edge["object"],
                )

                continue


            final_edge = {
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

                "mode":
                    "reflection_v2",

                "subject":
                    edge["subject"],

                "subject_type":
                    edge["subject_type"],

                "relation":
                    edge["relation"],

                "object":
                    edge["object"],

                "object_type":
                    edge["object_type"],

                "evidence":
                    verification["evidence"]
                    or edge["evidence"],

                "confidence":
                    verification["confidence"],
            }

            out.write(
                json.dumps(
                    final_edge,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            all_final.append(
                final_edge
            )

            print(
                "  KEEP  ",
                final_edge["subject"],
                f"--{final_edge['relation']}-->",
                final_edge["object"],
                f"| conf={final_edge['confidence']:.2f}",
            )


print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "Final edges:",
    len(all_final),
)

print(
    "Rejected edges:",
    len(all_rejected),
)

counts = Counter(
    x["relation"]
    for x in all_final
)

print("\nRELATION COUNTS")

for rel, n in counts.most_common():
    print(
        f"{rel:20s}",
        n,
    )


print("\nFINAL EDGES")

for x in all_final[:50]:

    print(
        f"{x['published_at']} | "
        f"{x['ticker']} | "
        f"{x['subject']} "
        f"--{x['relation']}--> "
        f"{x['object']} "
        f"| conf={x['confidence']:.2f}"
    )


print("\nSaved:")
print(OUTPUT_PATH)
print(REJECTED_PATH)
