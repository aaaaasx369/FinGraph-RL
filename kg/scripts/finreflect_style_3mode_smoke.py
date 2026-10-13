#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import ollama


# ============================================================
# Paths / config
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "rag/data/processed/sec/sec_chunks.jsonl"
)

OUT_DIR = (
    ROOT
    / "kg/output/finreflect_style_smoke"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://140.113.87.164:11434",
)

MAX_CHUNKS = int(
    os.getenv(
        "KG_MAX_CHUNKS",
        "20",
    )
)

client = ollama.Client(
    host=OLLAMA_HOST
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
# Portfolio-oriented schema
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
# Helpers
# ============================================================

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

    key = raw.lower()

    aliases = {
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

    return aliases.get(
        key,
        raw,
    )


def parse_json_response(text):

    text = clean_text(text)

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


    # Find first JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.S,
    )

    if not match:

        raise ValueError(
            "No JSON object found in model response:\n"
            + text[:1200]
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

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# Rule validation
# ============================================================

def validate_triples(triples):

    valid = []

    seen = set()


    if not isinstance(
        triples,
        list,
    ):
        return valid


    for t in triples:

        if not isinstance(
            t,
            dict,
        ):
            continue


        subject = t.get(
            "subject"
        )

        obj = t.get(
            "object"
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


        if (
            str(subject).lower()
            ==
            str(obj).lower()
        ):
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


        evidence = clean_text(
            t.get(
                "evidence",
                "",
            )
        )


        key = (
            str(subject).lower(),
            relation,
            str(obj).lower(),
        )


        if key in seen:
            continue


        seen.add(
            key
        )


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
                    evidence,

                "confidence":
                    confidence,
            }
        )


    return valid


# ============================================================
# Shared base extraction prompt
# ============================================================

def base_extraction_prompt(row):

    return f"""
You are extracting a financial knowledge graph from SEC filings
for downstream portfolio allocation research.

Use ONLY facts explicitly supported by the supplied SEC text.

ALLOWED RELATIONS:
{", ".join(RELATIONS)}

ALLOWED ENTITY TYPES:
{", ".join(ENTITY_TYPES)}

Required JSON format:

{{
  "triples": [
    {{
      "subject": "...",
      "subject_type": "COMPANY",
      "relation": "COMPETES_WITH",
      "object": "...",
      "object_type": "COMPANY",
      "evidence": "short supporting phrase from source",
      "confidence": 0.95
    }}
  ]
}}

Rules:

1. Extract only explicitly supported facts.

2. Do not invent entities, events, or relationships.

3. Use ONLY the allowed relation labels.

4. Use ONLY the allowed entity types.

5. Relation direction matters.

Examples:

"NVIDIA relies on TSMC"
-> NVDA DEPENDS_ON TSMC

"TSMC supplies NVIDIA"
-> TSMC SUPPLIES_TO NVDA

"NVIDIA purchases manufacturing services from TSMC"
-> NVDA CUSTOMER_OF TSMC

"Microsoft competes with Google"
-> MSFT COMPETES_WITH GOOGL

6. Focus on relations that may affect business fundamentals,
competitive position, supply chain, risks, or future performance.

7. Do not create a relation merely because two entities
appear in the same paragraph.

8. If there are no valid portfolio-relevant relations:
{{"triples":[]}}

9. confidence must be between 0 and 1.

10. Keep evidence short and grounded in the source.

11. Return JSON only. No explanation outside the JSON.


FILING METADATA

ticker: {row.get("ticker")}
form: {row.get("form")}
published_at: {row.get("published_at")}
doc_id: {row.get("doc_id")}
chunk_id: {row.get("chunk_id")}


SOURCE TEXT

{row["text"]}
"""


# ============================================================
# MODE 1: SINGLE PASS
# ============================================================

def run_single_pass(row):

    prompt = (
        base_extraction_prompt(
            row
        )
        +
        """

TASK:
Perform one extraction pass and return the final triples.
"""
    )

    raw = call_qwen(
        prompt
    )

    parsed = parse_json_response(
        raw
    )

    return validate_triples(
        parsed.get(
            "triples",
            [],
        )
    )


# ============================================================
# MODE 2: MULTI PASS
#
# pass 1 = extract
# pass 2 = normalize/correct
# ============================================================

def run_multi_pass(row):

    extraction_prompt = (
        base_extraction_prompt(
            row
        )
        +
        """

TASK:
This is extraction pass 1.

Extract candidate triples.
Do not perform a separate review yet.
"""
    )


    raw1 = call_qwen(
        extraction_prompt
    )

    parsed1 = parse_json_response(
        raw1
    )

    candidates = validate_triples(
        parsed1.get(
            "triples",
            [],
        )
    )


    if not candidates:
        return []


    normalization_prompt = f"""
You are performing the SECOND PASS of a financial
knowledge graph extraction pipeline.

Your job is NOT to freely re-extract the document.

Instead, normalize and correct the candidate triples.

Check:

1. Entity normalization
2. Relation normalization
3. Relation direction
4. Schema compliance
5. Duplicate triples
6. Obvious unsupported triples

Use ONLY the source text as evidence.

Allowed relations:
{", ".join(RELATIONS)}

Allowed entity types:
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


SOURCE TEXT:

{row["text"]}


FIRST-PASS CANDIDATE TRIPLES:

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}
"""


    raw2 = call_qwen(
        normalization_prompt
    )

    parsed2 = parse_json_response(
        raw2
    )


    return validate_triples(
        parsed2.get(
            "triples",
            [],
        )
    )


# ============================================================
# MODE 3: REFLECTION
#
# Extractor
# -> Critic feedback
# -> Corrector
# ============================================================

def run_reflection(row):

    extraction_prompt = (
        base_extraction_prompt(
            row
        )
        +
        """

TASK:
Act as the EXTRACTOR.

Produce candidate triples only.
A separate reviewer will inspect them afterward.
"""
    )


    raw1 = call_qwen(
        extraction_prompt
    )

    parsed1 = parse_json_response(
        raw1
    )

    candidates = validate_triples(
        parsed1.get(
            "triples",
            [],
        )
    )


    if not candidates:
        return [], {
            "issues": [],
            "summary": "No candidate triples."
        }


    # --------------------------------------------------------
    # Critic / reviewer
    # --------------------------------------------------------

    critic_prompt = f"""
You are the CRITIC in a reflection-based financial
knowledge graph extraction system.

Do NOT output corrected triples yet.

Review the candidate triples against the original SEC text.

Check for:

- unsupported facts
- incorrect entity
- wrong entity type
- wrong relation
- reversed direction
- duplicate triples
- missing obvious portfolio-relevant relation
- hallucination
- weak evidence
- schema violations

Allowed relations:
{", ".join(RELATIONS)}

Allowed entity types:
{", ".join(ENTITY_TYPES)}

Return JSON only:

{{
  "issues": [
    {{
      "triple_index": 0,
      "problem": "...",
      "suggestion": "..."
    }}
  ],
  "missing_relations": [
    {{
      "subject": "...",
      "relation": "...",
      "object": "...",
      "reason": "..."
    }}
  ],
  "summary": "..."
}}


SOURCE TEXT:

{row["text"]}


CANDIDATE TRIPLES:

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}
"""


    critic_raw = call_qwen(
        critic_prompt
    )

    critic = parse_json_response(
        critic_raw
    )


    # --------------------------------------------------------
    # Corrector
    # --------------------------------------------------------

    corrector_prompt = f"""
You are the CORRECTOR in a reflection-based financial
knowledge graph extraction system.

You receive:

1. Original SEC text
2. Candidate triples
3. Critic feedback

Produce the FINAL corrected triples.

Rules:

- Every final triple must be supported by the SEC text.
- Remove hallucinated or unsupported triples.
- Correct direction when justified.
- Normalize entities.
- Remove duplicates.
- Add a missing triple ONLY if clearly and explicitly
  supported by the source text.
- Use only the allowed schema.

Allowed relations:
{", ".join(RELATIONS)}

Allowed entity types:
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


SOURCE TEXT:

{row["text"]}


CANDIDATE TRIPLES:

{json.dumps(
    candidates,
    ensure_ascii=False,
    indent=2,
)}


CRITIC FEEDBACK:

{json.dumps(
    critic,
    ensure_ascii=False,
    indent=2,
)}
"""


    raw3 = call_qwen(
        corrector_prompt
    )

    parsed3 = parse_json_response(
        raw3
    )

    final = validate_triples(
        parsed3.get(
            "triples",
            [],
        )
    )


    return final, critic


# ============================================================
# Load smoke-test chunks
# ============================================================

rows = []

scanned = 0
candidate_count = 0


with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        obj = json.loads(
            line
        )

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


        candidate_count += 1


        obj[
            "ticker"
        ] = ticker

        obj[
            "form"
        ] = form

        obj[
            "text"
        ] = text


        rows.append(
            obj
        )


        if len(rows) >= MAX_CHUNKS:
            break


print("=" * 100)
print("FINREFLECT-STYLE 3-MODE KG SMOKE TEST")
print("=" * 100)

print(
    "Ollama host:",
    OLLAMA_HOST,
)

print(
    "Model:",
    MODEL,
)

print(
    "Scanned:",
    scanned,
)

print(
    "Prefilter candidates encountered:",
    candidate_count,
)

print(
    "Selected chunks:",
    len(rows),
)


# ============================================================
# Output files
# ============================================================

single_path = (
    OUT_DIR
    / "single_pass.jsonl"
)

multi_path = (
    OUT_DIR
    / "multi_pass.jsonl"
)

reflection_path = (
    OUT_DIR
    / "reflection.jsonl"
)

critic_path = (
    OUT_DIR
    / "reflection_feedback.jsonl"
)


files = {
    "single":
        open(
            single_path,
            "w",
            encoding="utf-8",
        ),

    "multi":
        open(
            multi_path,
            "w",
            encoding="utf-8",
        ),

    "reflection":
        open(
            reflection_path,
            "w",
            encoding="utf-8",
        ),

    "critic":
        open(
            critic_path,
            "w",
            encoding="utf-8",
        ),
}


all_edges = {
    "single": [],
    "multi": [],
    "reflection": [],
}


def make_edge(
    row,
    triple,
    mode,
):

    return {
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
            mode,

        "subject":
            triple[
                "subject"
            ],

        "subject_type":
            triple[
                "subject_type"
            ],

        "relation":
            triple[
                "relation"
            ],

        "object":
            triple[
                "object"
            ],

        "object_type":
            triple[
                "object_type"
            ],

        "evidence":
            triple[
                "evidence"
            ],

        "confidence":
            triple[
                "confidence"
            ],
    }


# ============================================================
# Run all three modes on exactly same chunks
# ============================================================

for i, row in enumerate(
    rows,
    start=1,
):

    print("\n" + "#" * 100)

    print(
        f"[{i}/{len(rows)}] "
        f"{row.get('ticker')} | "
        f"{row.get('form')} | "
        f"{row.get('published_at')}"
    )

    print(
        "chunk_id:",
        row.get(
            "chunk_id"
        )
    )

    print("#" * 100)


    # --------------------------------------------------------
    # Single
    # --------------------------------------------------------

    try:

        single = run_single_pass(
            row
        )

    except Exception as e:

        print(
            "[SINGLE ERROR]",
            repr(e),
        )

        single = []


    print(
        "Single:",
        len(single),
    )


    for t in single:

        edge = make_edge(
            row,
            t,
            "single",
        )

        files[
            "single"
        ].write(
            json.dumps(
                edge,
                ensure_ascii=False,
            )
            + "\n"
        )

        files[
            "single"
        ].flush()

        all_edges[
            "single"
        ].append(
            edge
        )


    # --------------------------------------------------------
    # Multi
    # --------------------------------------------------------

    try:

        multi = run_multi_pass(
            row
        )

    except Exception as e:

        print(
            "[MULTI ERROR]",
            repr(e),
        )

        multi = []


    print(
        "Multi:",
        len(multi),
    )


    for t in multi:

        edge = make_edge(
            row,
            t,
            "multi",
        )

        files[
            "multi"
        ].write(
            json.dumps(
                edge,
                ensure_ascii=False,
            )
            + "\n"
        )

        files[
            "multi"
        ].flush()

        all_edges[
            "multi"
        ].append(
            edge
        )


    # --------------------------------------------------------
    # Reflection
    # --------------------------------------------------------

    try:

        reflection, critic = (
            run_reflection(
                row
            )
        )

    except Exception as e:

        print(
            "[REFLECTION ERROR]",
            repr(e),
        )

        reflection = []

        critic = {
            "error":
                repr(e)
        }


    print(
        "Reflection:",
        len(reflection),
    )


    files[
        "critic"
    ].write(
        json.dumps(
            {
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

                "critic":
                    critic,
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    files[
        "critic"
    ].flush()


    for t in reflection:

        edge = make_edge(
            row,
            t,
            "reflection",
        )

        files[
            "reflection"
        ].write(
            json.dumps(
                edge,
                ensure_ascii=False,
            )
            + "\n"
        )

        files[
            "reflection"
        ].flush()

        all_edges[
            "reflection"
        ].append(
            edge
        )


    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    for mode, edges in [
        (
            "SINGLE",
            single,
        ),
        (
            "MULTI",
            multi,
        ),
        (
            "REFLECT",
            reflection,
        ),
    ]:

        for x in edges[:5]:

            print(
                f"  {mode:7s} | "
                f"{x['subject']} "
                f"--{x['relation']}--> "
                f"{x['object']} "
                f"| {x['confidence']:.2f}"
            )


for f in files.values():
    f.close()


# ============================================================
# Automated comparison summary
# ============================================================

summary_rows = []


for mode in [
    "single",
    "multi",
    "reflection",
]:

    edges = all_edges[
        mode
    ]

    relation_counts = Counter(
        x[
            "relation"
        ]
        for x in edges
    )


    unique_edges = {
        (
            str(
                x[
                    "subject"
                ]
            ).lower(),

            x[
                "relation"
            ],

            str(
                x[
                    "object"
                ]
            ).lower(),
        )
        for x in edges
    }


    confidences = [
        float(
            x[
                "confidence"
            ]
        )
        for x in edges
    ]


    summary_rows.append(
        {
            "mode":
                mode,

            "n_chunks":
                len(rows),

            "n_edges":
                len(edges),

            "unique_edges":
                len(
                    unique_edges
                ),

            "edges_per_chunk":
                (
                    len(edges)
                    /
                    max(
                        1,
                        len(rows),
                    )
                ),

            "mean_confidence":
                (
                    sum(
                        confidences
                    )
                    /
                    max(
                        1,
                        len(
                            confidences
                        ),
                    )
                ),

            "n_relation_types":
                len(
                    relation_counts
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUT_DIR
    / "mode_summary.csv",
    index=False,
)


# ============================================================
# Cross-mode overlap
# ============================================================

sets = {}


for mode in [
    "single",
    "multi",
    "reflection",
]:

    sets[
        mode
    ] = {
        (
            x[
                "ticker"
            ],

            x[
                "published_at"
            ],

            str(
                x[
                    "subject"
                ]
            ).lower(),

            x[
                "relation"
            ],

            str(
                x[
                    "object"
                ]
            ).lower(),
        )
        for x in all_edges[
            mode
        ]
    }


overlap_rows = []


pairs = [
    (
        "single",
        "multi",
    ),
    (
        "single",
        "reflection",
    ),
    (
        "multi",
        "reflection",
    ),
]


for a, b in pairs:

    A = sets[
        a
    ]

    B = sets[
        b
    ]

    union = A | B

    inter = A & B


    overlap_rows.append(
        {
            "mode_a":
                a,

            "mode_b":
                b,

            "intersection":
                len(
                    inter
                ),

            "union":
                len(
                    union
                ),

            "jaccard":
                (
                    len(
                        inter
                    )
                    /
                    len(
                        union
                    )
                )
                if union
                else 1.0,
        }
    )


pd.DataFrame(
    overlap_rows
).to_csv(
    OUT_DIR
    / "mode_overlap.csv",
    index=False,
)


# ============================================================
# Relation counts
# ============================================================

rel_rows = []


for mode in [
    "single",
    "multi",
    "reflection",
]:

    counts = Counter(
        x[
            "relation"
        ]
        for x in all_edges[
            mode
        ]
    )

    for rel, count in (
        counts
        .most_common()
    ):

        rel_rows.append(
            {
                "mode":
                    mode,

                "relation":
                    rel,

                "count":
                    count,
            }
        )


pd.DataFrame(
    rel_rows
).to_csv(
    OUT_DIR
    / "relation_counts.csv",
    index=False,
)


# ============================================================
# Print final summary
# ============================================================

print("\n")
print("=" * 100)
print("3-MODE SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


print("\nRELATION COUNTS")

for mode in [
    "single",
    "multi",
    "reflection",
]:

    print(
        f"\n[{mode.upper()}]"
    )

    counts = Counter(
        x[
            "relation"
        ]
        for x in all_edges[
            mode
        ]
    )

    for rel, count in (
        counts
        .most_common()
    ):

        print(
            f"{rel:20s}",
            count,
        )


print("\nMODE OVERLAP")

print(
    pd.DataFrame(
        overlap_rows
    ).to_string(
        index=False
    )
)


print("\nFIRST 20 REFLECTION EDGES")

for x in all_edges[
    "reflection"
][
    :20
]:

    print(
        f"{x['published_at']} | "
        f"{x['ticker']} | "
        f"{x['form']:4s} | "
        f"{x['subject']} "
        f"--{x['relation']}--> "
        f"{x['object']} "
        f"| conf={x['confidence']:.2f}"
    )


print("\nSaved to:")
print(
    OUT_DIR
)
