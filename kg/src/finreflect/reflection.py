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


REFLECTION_ROUNDS = int(
    os.getenv(
        "FINREFLECT_REFLECTION_ROUNDS",
        "2",
    )
)


def build_reflection_prompt(
    row: dict,
    triples: list[dict],
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
You are the REFLECTION AGENT in a financial
knowledge graph construction pipeline.

Reflection round: {round_id}

Your role is NOT to produce the final KG.

Your role is to critically inspect the extracted triples
against the original SEC passage and provide feedback.

Evaluate:

1. Is the subject actually supported by the passage?
2. Is the object actually supported by the passage?
3. Is the relation semantically correct?
4. Is relation direction correct?
5. Are entity types correct?
6. Is the evidence actually supporting the triple?
7. Are entities overly long or malformed?
8. Are important financial facts missing?
9. Are any triples hallucinated?
10. Are there redundant or duplicate triples?

Pay special attention to agent/subject errors.

For example:

If the passage says:

"FASB issued a new accounting standard"

then:

Microsoft --INTRODUCES--> accounting standard

is incorrect because FASB, not Microsoft, is the agent.


ENTITY SCHEMA

{entity_types}


RELATION SCHEMA

{relations}


CURRENT TRIPLES

{json.dumps(
    triples,
    ensure_ascii=False,
    indent=2,
)}


SOURCE PASSAGE

{row["text"]}


Return JSON only:

{{
  "overall_assessment": "...",

  "issues": [
    {{
      "triple_index": 0,
      "issue_type": "wrong_subject",
      "explanation": "...",
      "suggestion": "..."
    }}
  ],

  "missing_facts": [
    {{
      "description": "...",
      "suggested_subject": "...",
      "suggested_relation": "...",
      "suggested_object": "..."
    }}
  ]
}}
"""


def build_refiner_prompt(
    row: dict,
    triples: list[dict],
    feedback: dict,
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
You are the REFINER in a reflection-driven financial
knowledge graph extraction system.

Refinement round: {round_id}

You receive:

1. the original SEC passage
2. the current extracted triples
3. feedback from a reflection agent

Produce a revised set of FINAL triples for this round.

You may:

- preserve correct triples
- remove unsupported triples
- correct wrong subjects or objects
- correct entity types
- correct relation labels
- correct direction
- split malformed long entities
- add important facts identified by the reflection agent

However:

- every triple must still be grounded in the source passage
- never introduce outside knowledge
- use only the provided schema


ENTITY SCHEMA

{entity_types}


RELATION SCHEMA

{relations}


CURRENT TRIPLES

{json.dumps(
    triples,
    ensure_ascii=False,
    indent=2,
)}


REFLECTION FEEDBACK

{json.dumps(
    feedback,
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


def extract_reflection(
    row: dict,
) -> tuple[list[dict], list[dict]]:

    # Initial extraction
    triples = extract_single_pass(
        row
    )

    feedback_history = []

    for round_id in range(
        1,
        REFLECTION_ROUNDS + 1,
    ):

        # ----------------------------------
        # Reflection Agent
        # ----------------------------------

        reflection_prompt = (
            build_reflection_prompt(
                row=row,
                triples=triples,
                round_id=round_id,
            )
        )

        reflection_raw = call_llm(
            reflection_prompt
        )

        feedback = parse_json_response(
            reflection_raw
        )

        # Some local LLMs return a bare JSON list instead of
        # the requested JSON object. Normalize it here.
        if isinstance(feedback, list):
            feedback = {
                "overall_assessment": "Model returned a bare issue list.",
                "issues": feedback,
                "missing_facts": [],
            }

        elif not isinstance(feedback, dict):
            feedback = {
                "overall_assessment": "Invalid reflection response.",
                "issues": [],
                "missing_facts": [],
            }

        feedback_history.append(
            {
                "round":
                    round_id,

                "feedback":
                    feedback,
            }
        )

        # ----------------------------------
        # Refiner
        # ----------------------------------

        refine_prompt = (
            build_refiner_prompt(
                row=row,
                triples=triples,
                feedback=feedback,
                round_id=round_id,
            )
        )

        refine_raw = call_llm(
            refine_prompt
        )

        parsed = parse_json_response(
            refine_raw
        )

        # Qwen may return either:
        # {"triples": [...]}  or directly [...]
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

    return (
        triples,
        feedback_history,
    )
