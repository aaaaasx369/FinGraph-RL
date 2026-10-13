from __future__ import annotations

import json
import os
import re
import time

import ollama

from kg.src.finreflect.common import (
    OLLAMA_HOST,
)


JUDGE_MODEL = os.getenv(
    "FINREFLECT_JUDGE_MODEL",
    "qwen3.5:9b",
)


client = ollama.Client(
    host=OLLAMA_HOST
)


def parse_json(text: str) -> dict:

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
        x = json.loads(text)

        if isinstance(x, dict):
            return x

    except Exception:
        pass

    m = re.search(
        r"\{.*\}",
        text,
        flags=re.S,
    )

    if not m:
        raise ValueError(
            "Judge returned no JSON object:\n"
            + text[:1000]
        )

    x = json.loads(
        m.group(0)
    )

    if not isinstance(x, dict):
        raise ValueError(
            "Judge response is not a JSON object"
        )

    return x


def clamp_score(x) -> float:

    try:
        x = float(x)
    except Exception:
        return 1.0

    return max(
        1.0,
        min(5.0, x),
    )


def judge_extraction(
    source_text: str,
    triples: list[dict],
) -> dict:

    triples_text = json.dumps(
        triples,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are an evaluator of financial knowledge graph extraction.

Evaluate the COMPLETE SET of extracted triples against the
original SEC filing passage.

Do not judge formatting style.
Judge whether the extracted financial knowledge is correct
and useful.

Score each dimension from 1 to 5.

DEFINITIONS

PRECISION
5 = nearly all extracted triples are semantically correct.
1 = many triples use wrong subjects, objects, relations,
    directions, or entity meanings.

FAITHFULNESS
5 = extracted facts are strictly grounded in the source.
1 = extraction contains hallucinations, unsupported inference,
    or claims not justified by the passage.

COMPREHENSIVENESS
Evaluate only important financial knowledge that can be
expressed using the provided financial KG entity/relation schema.

5 = most important schema-expressible financial relations
    in the passage are captured.
1 = most important schema-expressible financial relations
    are missing.

Do NOT penalize the extraction for omitting metadata such as:
- phone numbers
- ZIP codes
- IRS identifiers
- filing form labels
- document identifiers
unless such information is explicitly represented by the KG schema.

RELEVANCE
5 = extracted triples represent meaningful financial KG facts.
1 = extraction mostly contains noise, malformed table fragments,
    irrelevant phrases, or trivial statements.

Important:

- A relation can be schema-valid but still be semantically wrong.
- Example:
  COMPANY --PRODUCES--> Revenue
  should be penalized if "revenue" is a financial metric rather
  than something the company produces.

- Penalize incorrect agents.
  If the text says FASB issued a standard, do not credit a triple
  saying Microsoft introduced the standard.

- Penalize very long objects that are merely copied table rows
  or entire sentences rather than meaningful entities.

- For an empty triple list, precision/faithfulness may still be
  high if no false facts were produced, but comprehensiveness
  should be low when important relations were available.

Return JSON only:

{{
  "precision": 1,
  "faithfulness": 1,
  "comprehensiveness": 1,
  "relevance": 1,
  "precision_reason": "...",
  "faithfulness_reason": "...",
  "comprehensiveness_reason": "...",
  "relevance_reason": "..."
}}


SOURCE PASSAGE

{source_text}


EXTRACTED TRIPLES

{triples_text}
"""

    print(
        f"    Judge {JUDGE_MODEL} | "
        f"prompt chars={len(prompt):,}",
        flush=True,
    )

    t0 = time.time()

    last_error = None

    for attempt in range(1, 4):

        try:

            print(
                f"    judge attempt {attempt}/3",
                flush=True,
            )

            t0 = time.time()

            response = client.chat(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                think=False,
                format="json",
                options={
                    "temperature": 0.0,
                    "num_predict": 768,
                },
            )

            wall = time.time() - t0

            print(
                f"    judge wall={wall:.2f}s",
                flush=True,
            )

            content = response["message"]["content"]

            if not content or not content.strip():
                raise ValueError(
                    "Judge returned empty content"
                )

            result = parse_json(
                content
            )

            break

        except Exception as e:

            last_error = e

            print(
                "    judge attempt failed:",
                repr(e),
                flush=True,
            )

            if attempt == 3:
                raise RuntimeError(
                    f"Judge failed after 3 attempts: {last_error!r}"
                )

    return {
        "precision":
            clamp_score(
                result.get("precision", 1)
            ),

        "faithfulness":
            clamp_score(
                result.get("faithfulness", 1)
            ),

        "comprehensiveness":
            clamp_score(
                result.get(
                    "comprehensiveness",
                    1,
                )
            ),

        "relevance":
            clamp_score(
                result.get("relevance", 1)
            ),

        "precision_reason":
            str(
                result.get(
                    "precision_reason",
                    "",
                )
            ),

        "faithfulness_reason":
            str(
                result.get(
                    "faithfulness_reason",
                    "",
                )
            ),

        "comprehensiveness_reason":
            str(
                result.get(
                    "comprehensiveness_reason",
                    "",
                )
            ),

        "relevance_reason":
            str(
                result.get(
                    "relevance_reason",
                    "",
                )
            ),
    }
