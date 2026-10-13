#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import ollama


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "rag/data/processed/sec/sec_chunks.jsonl"
)

OUT_ROOT = (
    ROOT
    / "kg/output/finreflectkg_style_smoke"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

MAX_CHUNKS = int(
    os.getenv(
        "KG_MAX_CHUNKS",
        "10",
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


# ============================================================
# PORTFOLIO-SPECIFIC SCHEMA
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    if x is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(x),
    ).strip()


def passes_prefilter(text):
    t = text.lower()

    return any(
        k in t
        for k in KEYWORDS
    )


def normalize_company(name):
    if not name:
        return name

    raw = str(name).strip()
    x = raw.lower()

    aliases = {
        "apple": "AAPL",
        "apple inc": "AAPL",
        "apple inc.": "AAPL",

        "microsoft": "MSFT",
        "microsoft corporation": "MSFT",

        "nvidia": "NVDA",
        "nvidia corporation": "NVDA",

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

    m = re.search(
        r"\{.*\}",
        text,
        flags=re.S,
    )

    if not m:
        raise ValueError(
            "No JSON object found:\n"
            + text[:1000]
        )

    return json.loads(
        m.group(0)
    )


def llm(prompt):

    r = client.chat(
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

    return r["message"]["content"]


def validate(triples):

    output = []
    seen = set()

    for t in triples:

        subject = normalize_company(
            t.get("subject")
        )

        obj = normalize_company(
            t.get("object")
        )

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

        if str(subject).lower() == str(obj).lower():
            continue


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


        key = (
            str(subject).lower(),
            relation,
            str(obj).lower(),
        )

        if key in seen:
            continue

        seen.add(key)


        output.append(
            {
                "subject": subject,
                "subject_type": subject_type,
                "relation": relation,
                "object": obj,
                "object_type": object_type,
                "evidence": clean_text(
                    t.get(
                        "evidence",
                        "",
                    )
                ),
                "confidence": confidence,
            }
        )

    return output


# ============================================================
# BASE EXTRACTION PROMPT
# ============================================================

def extraction_prompt(row):

    return f"""
You are constructing a financial knowledge graph from SEC filings.

Use ONLY information explicitly supported by the source text.

ALLOWED RELATIONS:
{", ".join(RELATIONS)}

ALLOWED ENTITY TYPES:
{", ".join(ENTITY_TYPES)}

Return JSON only:

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

Rules:

1. Do not invent facts.
2. Use only allowed relation labels.
3. Use only allowed entity types.
4. Relation direction must match the text.
5. Each evidence field must directly support the triple.
6. Prefer financially and portfolio-relevant relations.
7. If no supported relation exists, return:
   {{"triples":[]}}

Direction examples:

"NVIDIA relies on TSMC"
=> NVDA DEPENDS_ON TSMC

"TSMC supplies NVIDIA"
=> TSMC SUPPLIES_TO NVDA

"NVIDIA purchases manufacturing services from TSMC"
=> NVDA CUSTOMER_OF TSMC


METADATA

ticker: {row.get("ticker")}
form: {row.get("form")}
published_at: {row.get("published_at")}
doc_id: {row.get("doc_id")}
chunk_id: {row.get("chunk_id")}


SOURCE TEXT

{row["text"]}
"""


def first_pass(row):

    raw = llm(
        extraction_prompt(row)
    )

    parsed = parse_json_response(
        raw
    )

    return validate(
        parsed.get(
            "triples",
            [],
        )
    )


# ============================================================
# MODE 1: SINGLE PASS
# ============================================================

def single_pass(row):

    return first_pass(
        row
    )


# ============================================================
# MODE 2: MULTI PASS
#
# First extraction
# -> normalization / correction pass
# ============================================================

def multi_pass(row):

    candidates = first_pass(
        row
    )

    if not candidates:
        return []


    prompt = f"""
You are performing the second pass of financial KG extraction.

The first pass already produced candidate triples.

Your job is to NORMALIZE and CORRECT them.

Check:

1. canonical entity naming
2. valid entity types
3. valid relation labels
4. relation direction
5. duplicates
6. direct support from source text

Do not introduce unsupported facts.

ALLOWED RELATIONS:
{", ".join(RELATIONS)}

ALLOWED ENTITY TYPES:
{", ".join(ENTITY_TYPES)}

Return JSON only:

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


SOURCE TEXT

{row["text"]}


FIRST-PASS TRIPLES

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}
"""

    raw = llm(prompt)

    parsed = parse_json_response(
        raw
    )

    return validate(
        parsed.get(
            "triples",
            [],
        )
    )


# ============================================================
# MODE 3: REFLECTION AGENT
#
# extraction
# -> critic / feedback
# -> correction
# ============================================================

def reflection_pass(row):

    candidates = first_pass(
        row
    )

    if not candidates:
        return []


    # --------------------------------------------------------
    # REFLECTION / CRITIC
    # --------------------------------------------------------

    critic_prompt = f"""
You are the critic in an agentic financial KG extraction system.

Do NOT rewrite the triples yet.

Evaluate every candidate triple against the source text.

Check:

- faithfulness to text
- entity correctness
- entity type correctness
- relation correctness
- direction
- duplicate facts
- unsupported inference
- missing obvious portfolio-relevant relations

Return JSON only:

{{
  "feedback": [
    {{
      "triple_index": 0,
      "status": "KEEP|CORRECT|REMOVE",
      "reason": "..."
    }}
  ],
  "missing_relations": [
    {{
      "subject": "...",
      "relation": "...",
      "object": "...",
      "reason": "..."
    }}
  ]
}}

ALLOWED RELATIONS:
{", ".join(RELATIONS)}

ALLOWED ENTITY TYPES:
{", ".join(ENTITY_TYPES)}


SOURCE TEXT

{row["text"]}


CANDIDATE TRIPLES

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}
"""

    critic_raw = llm(
        critic_prompt
    )

    feedback = parse_json_response(
        critic_raw
    )


    # --------------------------------------------------------
    # CORRECTION AGENT
    # --------------------------------------------------------

    correction_prompt = f"""
You are the correction agent in a financial KG extraction system.

Use:

1. original SEC source text
2. candidate triples
3. critic feedback

Produce the final corrected triples.

Rules:

- Keep only facts directly supported by source text.
- Remove hallucinations.
- Correct wrong direction.
- Correct entity normalization.
- Correct invalid relation labels.
- Add a missing relation only if clearly supported by source text.
- Do not use world knowledge not present in the source.

ALLOWED RELATIONS:
{", ".join(RELATIONS)}

ALLOWED ENTITY TYPES:
{", ".join(ENTITY_TYPES)}

Return JSON only:

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


SOURCE TEXT

{row["text"]}


CANDIDATE TRIPLES

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}


CRITIC FEEDBACK

{json.dumps(
    feedback,
    ensure_ascii=False,
    indent=2,
)}
"""

    corrected_raw = llm(
        correction_prompt
    )

    corrected = parse_json_response(
        corrected_raw
    )

    return validate(
        corrected.get(
            "triples",
            [],
        )
    )


# ============================================================
# LOAD SAME CHUNKS FOR ALL 3 MODES
# ============================================================

rows = []

scanned = 0

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

        obj["ticker"] = ticker
        obj["form"] = form
        obj["text"] = text

        rows.append(
            obj
        )

        if len(rows) >= MAX_CHUNKS:
            break


print("=" * 100)
print("FINREFLECTKG-STYLE EXTRACTION REPLICATION")
print("=" * 100)

print(
    "Model:",
    MODEL,
)

print(
    "Chunks:",
    len(rows),
)

print(
    "Scanned:",
    scanned,
)


# ============================================================
# RUN 3 MODES
# ============================================================

MODES = {
    "single_pass":
        single_pass,

    "multi_pass":
        multi_pass,

    "reflection":
        reflection_pass,
}


all_mode_results = {}


for mode_name, fn in MODES.items():

    print("\n")
    print("#" * 100)
    print(
        "MODE:",
        mode_name.upper()
    )
    print("#" * 100)

    edges = []

    output_path = (
        OUT_ROOT
        / f"{mode_name}.jsonl"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for i, row in enumerate(
            rows,
            start=1,
        ):

            print(
                f"\n[{i}/{len(rows)}] "
                f"{row['ticker']} "
                f"{row['form']} "
                f"{row.get('published_at')}"
            )

            try:

                triples = fn(
                    row
                )

                print(
                    "Final triples:",
                    len(triples),
                )

                for t in triples:

                    edge = {
                        "mode":
                            mode_name,

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

                        **t,
                    }

                    edges.append(
                        edge
                    )

                    out.write(
                        json.dumps(
                            edge,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    print(
                        f"  {t['subject']} "
                        f"--{t['relation']}--> "
                        f"{t['object']} "
                        f"| {t['confidence']:.2f}"
                    )


            except Exception as e:

                print(
                    "ERROR:",
                    repr(e),
                )


    all_mode_results[
        mode_name
    ] = edges


# ============================================================
# SIMPLE RULE-BASED EVALUATION
#
# Not a full reproduction of FinReflectKG CheckRules.
# This is our local smoke-test evaluator.
# ============================================================

summary = []


for mode_name, edges in (
    all_mode_results.items()
):

    relation_counts = Counter(
        e["relation"]
        for e in edges
    )

    unique_edges = {
        (
            str(e["subject"]).lower(),
            e["relation"],
            str(e["object"]).lower(),
        )
        for e in edges
    }

    avg_conf = (
        sum(
            e["confidence"]
            for e in edges
        )
        / len(edges)
        if edges
        else 0.0
    )

    chunks_with_edges = len(
        {
            e["chunk_id"]
            for e in edges
        }
    )

    summary.append(
        {
            "mode":
                mode_name,

            "n_edges":
                len(edges),

            "unique_edges":
                len(unique_edges),

            "chunks_with_edges":
                chunks_with_edges,

            "avg_confidence":
                avg_conf,

            "relation_types":
                len(
                    relation_counts
                ),
        }
    )


summary_path = (
    OUT_ROOT
    / "summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for x in summary:

    print(
        f"{x['mode']:15s} | "
        f"edges={x['n_edges']:3d} | "
        f"unique={x['unique_edges']:3d} | "
        f"chunks={x['chunks_with_edges']:3d} | "
        f"relations={x['relation_types']:2d} | "
        f"avg_conf={x['avg_confidence']:.3f}"
    )


# ============================================================
# CROSS-MODE AGREEMENT
# ============================================================

print("\n")
print("=" * 100)
print("CROSS-MODE EDGE AGREEMENT")
print("=" * 100)


sets = {}

for mode_name, edges in (
    all_mode_results.items()
):

    sets[mode_name] = {
        (
            str(e["subject"]).lower(),
            e["relation"],
            str(e["object"]).lower(),
        )
        for e in edges
    }


pairs = [
    (
        "single_pass",
        "multi_pass",
    ),
    (
        "single_pass",
        "reflection",
    ),
    (
        "multi_pass",
        "reflection",
    ),
]


for a, b in pairs:

    A = sets[a]
    B = sets[b]

    intersection = len(
        A & B
    )

    union = len(
        A | B
    )

    jaccard = (
        intersection / union
        if union
        else 0.0
    )

    print(
        f"{a:15s} vs {b:15s} | "
        f"intersection={intersection:3d} | "
        f"union={union:3d} | "
        f"Jaccard={jaccard:.3f}"
    )


print("\nSaved to:")
print(OUT_ROOT)
