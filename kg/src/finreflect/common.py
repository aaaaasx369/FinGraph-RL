from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import ollama


ROOT = Path(__file__).resolve().parents[3]

SCHEMA_PATH = ROOT / "kg/configs/finreflect_final_schema.json"

SIGNATURE_PATH = (
    ROOT
    / "kg/output/finreflectkg_schema/relation_signatures.json"
)

SEC_CHUNKS_PATH = (
    ROOT
    / "rag/data/processed/sec/sec_chunks.jsonl"
)

OUTPUT_ROOT = (
    ROOT
    / "kg/output/finreflect_reimplementation"
)


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://140.113.87.164:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b",
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


ALIASES = {
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
    "google llc": "GOOGL",
}


def load_schema() -> dict[str, Any]:
    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


SCHEMA = load_schema()


CANONICAL_SIGNATURE_PATH = (
    ROOT
    / "kg/configs/finreflect_canonical_signatures.json"
)


def load_canonical_signatures():
    with open(
        CANONICAL_SIGNATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


CANONICAL_SIGNATURES = load_canonical_signatures()


def format_canonical_signatures() -> str:
    lines = []

    for relation in sorted(
        CANONICAL_SIGNATURES
    ):
        for subject_type, object_type in (
            CANONICAL_SIGNATURES[relation]
        ):
            lines.append(
                f"- {subject_type} "
                f"--{relation}--> "
                f"{object_type}"
            )

    return "\n".join(lines)


def canonical_signature_valid(
    subject_type,
    relation,
    object_type,
):
    relation = str(relation).strip().upper()
    subject_type = str(subject_type).strip().upper()
    object_type = str(object_type).strip().upper()

    if relation not in CANONICAL_SIGNATURES:
        return True

    allowed = {
        (
            str(s).strip().upper(),
            str(o).strip().upper(),
        )
        for s, o in CANONICAL_SIGNATURES[
            relation
        ]
    }

    return (
        subject_type,
        object_type,
    ) in allowed


def load_relation_signatures():
    with open(
        SIGNATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


RELATION_SIGNATURES = load_relation_signatures()


def format_relation_signatures(
    min_count: int = 3,
    top_k_per_relation: int = 5,
) -> str:

    lines = []

    for relation in sorted(
        RELATION_SIGNATURES
    ):
        rows = RELATION_SIGNATURES[
            relation
        ]

        kept = [
            x
            for x in rows
            if int(
                x.get(
                    "count",
                    0,
                )
            ) >= min_count
        ]

        kept = kept[
            :top_k_per_relation
        ]

        for x in kept:
            lines.append(
                f"- {x['subject_type']} "
                f"--{relation}--> "
                f"{x['object_type']} "
                f"(observed {x['count']} times)"
            )

    return "\n".join(lines)


ENTITY_TYPES = set(
    SCHEMA["entity_types"]
)

RELATIONS = set(
    SCHEMA["relations"]
)


def get_client():
    return ollama.Client(
        host=OLLAMA_HOST
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_entity(
    entity: str,
) -> str:

    entity = clean_text(
        entity
    )

    return ALIASES.get(
        entity.lower(),
        entity,
    )


def parse_json_response(text: str):

    import json
    import re

    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    # Remove Markdown JSON fences such as:
    # ```json
    # {...}
    # ```
    text = re.sub(
        r"^```(?:json)?\\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\\s*```$",
        "",
        text,
    )

    text = text.strip()

    # First try direct JSON parsing.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract the outermost JSON object.
    first_obj = text.find("{")
    last_obj = text.rfind("}")

    if (
        first_obj != -1
        and
        last_obj != -1
        and
        last_obj > first_obj
    ):
        candidate = text[
            first_obj:last_obj + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: extract JSON list.
    first_arr = text.find("[")
    last_arr = text.rfind("]")

    if (
        first_arr != -1
        and
        last_arr != -1
        and
        last_arr > first_arr
    ):
        candidate = text[
            first_arr:last_arr + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

def call_llm(
    prompt: str,
) -> str:

    import time

    client = get_client()

    print(
        f"    prompt chars: {len(prompt):,}",
        flush=True,
    )

    t0 = time.time()

    response = client.chat(
        model=OLLAMA_MODEL,
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
            "num_predict": 1024,
        },
    )

    wall = time.time() - t0

    prompt_count = getattr(
        response,
        "prompt_eval_count",
        None,
    )

    prompt_duration = getattr(
        response,
        "prompt_eval_duration",
        None,
    )

    eval_count = getattr(
        response,
        "eval_count",
        None,
    )

    eval_duration = getattr(
        response,
        "eval_duration",
        None,
    )

    total_duration = getattr(
        response,
        "total_duration",
        None,
    )

    print(
        f"    wall time: {wall:.2f}s",
        flush=True,
    )

    print(
        f"    prompt tokens: {prompt_count}",
        flush=True,
    )

    print(
        f"    output tokens: {eval_count}",
        flush=True,
    )

    if (
        prompt_count
        and
        prompt_duration
    ):
        speed = (
            prompt_count
            /
            (prompt_duration / 1e9)
        )

        print(
            f"    prompt speed: {speed:.1f} tok/s",
            flush=True,
        )

    if (
        eval_count
        and
        eval_duration
    ):
        speed = (
            eval_count
            /
            (eval_duration / 1e9)
        )

        print(
            f"    generation speed: {speed:.1f} tok/s",
            flush=True,
        )

    if total_duration:
        print(
            f"    Ollama total: "
            f"{total_duration / 1e9:.2f}s",
            flush=True,
        )

    return response[
        "message"
    ][
        "content"
    ]



def normalize_triples_raw(
    triples: list,
) -> list[dict]:
    """
    Normalize model outputs WITHOUT enforcing the KG schema.

    Important:
    - invalid entity types are preserved
    - invalid relation labels are preserved
    - long/malformed entities are preserved

    Those errors must be measured later by CheckRules,
    not silently deleted during extraction.
    """

    if not isinstance(triples, list):
        return []

    results = []
    seen = set()

    for triple in triples:

        if not isinstance(triple, dict):
            continue

        subject = normalize_entity(
            triple.get("subject", "")
        )

        obj = normalize_entity(
            triple.get("object", "")
        )

        subject_type = clean_text(
            triple.get("subject_type", "")
        ).upper()

        object_type = clean_text(
            triple.get("object_type", "")
        ).upper()

        relation = clean_text(
            triple.get("relation", "")
        ).upper()

        evidence = clean_text(
            triple.get("evidence", "")
        )

        try:
            confidence = float(
                triple.get("confidence", 0.0)
            )
        except Exception:
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # Keep malformed/schema-invalid triples.
        # Only remove exact duplicate outputs.
        key = (
            subject.lower(),
            subject_type,
            relation,
            obj.lower(),
            object_type,
        )

        if key in seen:
            continue

        seen.add(key)

        results.append(
            {
                "subject": subject,
                "subject_type": subject_type,
                "relation": relation,
                "object": obj,
                "object_type": object_type,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    return results


def validate_schema_only(
    triples: list,
) -> list[dict]:

    """
    Important:
    This does NOT delete edges based on factual judgment.

    It only enforces basic output/schema validity.

    FinReflect-style evaluation such as CheckRules
    will be implemented separately.
    """

    results = []
    seen = set()

    for triple in triples:

        if not isinstance(
            triple,
            dict,
        ):
            continue

        subject = normalize_entity(
            triple.get(
                "subject",
                "",
            )
        )

        obj = normalize_entity(
            triple.get(
                "object",
                "",
            )
        )

        subject_type = clean_text(
            triple.get(
                "subject_type",
                "",
            )
        ).upper()

        object_type = clean_text(
            triple.get(
                "object_type",
                "",
            )
        ).upper()

        relation = clean_text(
            triple.get(
                "relation",
                "",
            )
        ).upper()

        evidence = clean_text(
            triple.get(
                "evidence",
                "",
            )
        )

        if not subject or not obj:
            continue

        if subject_type not in ENTITY_TYPES:
            continue

        if object_type not in ENTITY_TYPES:
            continue

        if relation not in RELATIONS:
            continue

        try:
            confidence = float(
                triple.get(
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
            subject.lower(),
            relation,
            obj.lower(),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        results.append(
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

    return results


def load_sec_chunks(
    max_chunks: int | None = None,
    offset: int = 0,
) -> list[dict]:

    rows = []
    eligible_seen = 0

    with open(
        SEC_CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            row = json.loads(
                line
            )

            ticker = clean_text(
                row.get(
                    "ticker",
                    "",
                )
            ).upper()

            form = clean_text(
                row.get(
                    "form",
                    "",
                )
            ).upper()

            text = clean_text(
                row.get(
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

            row["ticker"] = ticker
            row["form"] = form
            row["text"] = text

            if eligible_seen < offset:
                eligible_seen += 1
                continue

            rows.append(
                row
            )

            eligible_seen += 1

            if (
                max_chunks is not None
                and
                len(rows) >= max_chunks
            ):
                break

    return rows


def attach_metadata(
    row: dict,
    triple: dict,
    mode: str,
) -> dict:

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

        **triple,
    }
