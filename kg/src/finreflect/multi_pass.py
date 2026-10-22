from __future__ import annotations

import json
import os

from kg.src.finreflect.common import (
    SCHEMA,
    call_llm,
    parse_json_response,
    normalize_triples_raw,
)

from kg.src.finreflect.single_pass import (
    extract_single_pass,
)


MULTI_ROUNDS = int(
    os.getenv(
        "FINREFLECT_MULTI_ROUNDS",
        "2",
    )
)


def build_refinement_prompt(
    row: dict,
    current_triples: list[dict],
    round_id: int,
) -> str:

    entity_types = "\n".join(
        f"- {x}"
        for x in SCHEMA["entity_types"]
    )

    relations = "\n".join(
        f"- {x}"
        for x in SCHEMA["relations"]
    )

    return f"""
You are performing iterative financial knowledge graph extraction.

This is refinement round {round_id}.

You are given:

1. the original SEC passage
2. the financial entity schema
3. the financial relation schema
4. triples extracted in the previous pass

Your task is to RE-EXAMINE the original passage.

You may:

- preserve correct triples
- correct incorrect subject, object, type, or relation
- remove unsupported triples
- add important factual triples missed previously
- improve entity specificity
- correct relation direction

Do NOT invent facts from outside knowledge.

Do NOT merely paraphrase the previous result.

Every final triple should be supported by the source passage.


ENTITY SCHEMA

{entity_types}


RELATION SCHEMA

{relations}


PREVIOUS TRIPLES

{json.dumps(
    current_triples,
    ensure_ascii=False,
    indent=2,
)}


SOURCE PASSAGE

{row["text"]}


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
      "confidence": 0.90
    }}
  ]
}}
"""


def extract_multi_pass(
    row: dict,
) -> list[dict]:

    # Pass 1
    triples = extract_single_pass(
        row
    )

    # Iterative refinement
    for round_id in range(
        2,
        MULTI_ROUNDS + 1,
    ):

        prompt = build_refinement_prompt(
            row=row,
            current_triples=triples,
            round_id=round_id,
        )

        response = call_llm(
            prompt
        )

        parsed = parse_json_response(
            response
        )

        if isinstance(parsed, list):
            refined_triples = parsed

        elif isinstance(parsed, dict):
            refined_triples = parsed.get(
                "triples",
                [],
            )

        else:
            refined_triples = []

        triples = normalize_triples_raw(
            refined_triples
        )

    return triples
