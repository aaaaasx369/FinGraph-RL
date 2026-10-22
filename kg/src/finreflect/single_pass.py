from __future__ import annotations

import json
import os

from kg.src.finreflect.common import (
    SCHEMA,
    format_canonical_signatures,
    call_llm,
    parse_json_response,
    normalize_triples_raw,
)


def build_prompt(
    row: dict,
) -> str:

    entity_types = "\n".join(
        f"- {x}"
        for x in SCHEMA[
            "entity_types"
        ]
    )

    relations = "\n".join(
        f"- {x}"
        for x in SCHEMA[
            "relations"
        ]
    )


    canonical_signatures = (
        format_canonical_signatures()
    )


    return f"""
You are a financial knowledge graph extraction agent.

Your task is to extract structured financial knowledge
from an SEC filing passage.

Extract factual subject-relation-object triples
that are supported by the source passage.

ENTITY SCHEMA

{entity_types}


RELATION SCHEMA

{relations}


CANONICAL SIGNATURE RULES

For the following relations, ONLY use these
subject-type / relation / object-type combinations.

Do not invent another type combination for these relations.

{canonical_signatures}


ENTITY NAME AND TYPE RULES

1. Entity names and entity types are different things.

2. Never use ontology labels such as:
   ORG, COMP, FIN_METRIC, ACCOUNTING_POLICY,
   RISK_FACTOR, PRODUCT, SERVICE, POLICY,
   REGULATORY_REQUIREMENT
   as the entity name unless that exact phrase is genuinely
   an entity mentioned in the source.

3. The subject and object should be concrete entities or
   concepts grounded in the source text.

4. Choose the most specific entity type supported by the
   source and canonical signature.

Examples:

BAD:
"object": "ACCOUNTING_POLICY",
"object_type": "POLICY"

BAD:
"object": "RISK_FACTOR",
"object_type": "POLICY"

GOOD:
"object": "revenue recognition policy",
"object_type": "ACCOUNTING_POLICY"

GOOD:
"object": "foreign exchange fluctuations",
"object_type": "RISK_FACTOR"

5. Do not replace missing information with a schema label.
If the source does not contain a concrete supported entity,
do not emit that triple.


INSTRUCTIONS

1. Every extracted fact must follow the provided schema.

2. Identify a subject entity, relation, and object entity.

3. Preserve relation direction.

4. Use the most specific entity name available in the source.

5. Do not create facts solely from outside knowledge.

The filing metadata supplied in this prompt is trusted context.
The ticker identifies the company that filed this SEC document
and may be used as an ORG subject when appropriate.

6. Multiple triples may be extracted from one passage.

7. Extract any factual relation that is supported by the
source passage and is expressible by the allowed schema.

Regulatory obligations, accounting policies, risk factors,
company activities, products, financial metrics, investments,
dependencies, partnerships, competition, and similar
schema-supported facts are valid knowledge.

Do not require a fact to contain a numerical financial metric
in order to be useful.

Return an empty list only when no supported relation can be
expressed using the allowed schema.

8. Provide a short supporting evidence span.

9. Return JSON only.

Required format:

{{
  "triples": [
    {{
      "subject": "...",
      "subject_type": "...",
      "relation": "...",
      "object": "...",
      "object_type": "...",
      "evidence": "...",
      "confidence": 0.90
    }}
  ]
}}


FILING METADATA

ticker:
{row.get("ticker")}

form:
{row.get("form")}

published_at:
{row.get("published_at")}

chunk_id:
{row.get("chunk_id")}


SOURCE PASSAGE

{row["text"]}
"""


def extract_single_pass(
    row: dict,
) -> list[dict]:

    debug = (
        os.getenv(
            "FINREFLECT_DEBUG",
            "0",
        )
        == "1"
    )

    prompt = build_prompt(
        row
    )

    response = call_llm(
        prompt
    )

    if debug:
        print(
            "\n" + "=" * 100
        )
        print(
            "CHUNK:",
            row.get("chunk_id"),
        )
        print(
            "\nRAW RESPONSE:"
        )
        print(
            repr(response)
        )

    parsed = parse_json_response(
        response
    )

    if debug:
        print(
            "\nPARSED RESPONSE:"
        )
        print(
            repr(parsed)
        )
        print(
            "PARSED TYPE:",
            type(parsed),
        )

    # Local LLMs may return either:
    # {"triples": [...]} or directly [...]
    if isinstance(parsed, list):
        triples = parsed

    elif isinstance(parsed, dict):
        triples = parsed.get(
            "triples",
            [],
        )

    else:
        triples = []

    if debug:
        print(
            "\nTRIPLES BEFORE NORMALIZE:"
        )
        print(
            repr(triples)
        )

    normalized = normalize_triples_raw(
        triples
    )

    if debug:
        print(
            "\nNORMALIZED TRIPLES:"
        )
        print(
            repr(normalized)
        )
        print(
            "\nN RAW TRIPLES:",
            len(triples),
        )
        print(
            "N NORMALIZED TRIPLES:",
            len(normalized),
        )

    return normalized
