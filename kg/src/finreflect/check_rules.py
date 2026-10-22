from __future__ import annotations

import re

from kg.src.finreflect.common import (
    ENTITY_TYPES,
    RELATIONS,
    ALIASES,
    clean_text,
    canonical_signature_valid,
)


MAX_ENTITY_WORDS = 5


def normalize_for_match(
    x: str,
) -> str:

    x = clean_text(
        x
    ).lower()

    x = re.sub(
        r"[^a-z0-9]+",
        " ",
        x,
    )

    return re.sub(
        r"\s+",
        " ",
        x,
    ).strip()


def word_count(
    x: str,
) -> int:

    return len(
        clean_text(x).split()
    )


def subject_reference_rule(
    edge: dict,
    source_text: str,
) -> bool:

    subject = clean_text(
        edge.get(
            "subject",
            "",
        )
    )

    ticker = clean_text(
        edge.get(
            "ticker",
            "",
        )
    ).upper()

    source_norm = normalize_for_match(
        source_text
    )

    subject_norm = normalize_for_match(
        subject
    )

    # Direct mention
    if (
        subject_norm
        and
        subject_norm in source_norm
    ):
        return True

    # Ticker represents filing company.
    if subject.upper() == ticker:
        return True

    # Check known aliases.
    for alias, canonical in (
        ALIASES.items()
    ):

        if (
            canonical == subject
            and
            normalize_for_match(alias)
            in source_norm
        ):
            return True

    return False


def entity_length_rule(
    edge: dict,
) -> bool:

    subject = clean_text(
        edge.get("subject", "")
    )

    obj = clean_text(
        edge.get("object", "")
    )

    if not subject or not obj:
        return False

    subject_ok = (
        word_count(subject)
        <= MAX_ENTITY_WORDS
    )

    object_ok = (
        word_count(obj)
        <= MAX_ENTITY_WORDS
    )

    return (
        subject_ok
        and
        object_ok
    )


def entity_schema_rule(
    edge: dict,
) -> bool:

    subject_type = clean_text(
        edge.get(
            "subject_type",
            "",
        )
    ).upper()

    object_type = clean_text(
        edge.get(
            "object_type",
            "",
        )
    ).upper()

    return (
        subject_type
        in ENTITY_TYPES
        and
        object_type
        in ENTITY_TYPES
    )


def relation_schema_rule(
    edge: dict,
) -> bool:

    relation = clean_text(
        edge.get(
            "relation",
            "",
        )
    ).upper()

    return (
        relation
        in RELATIONS
    )


def ontology_label_as_entity_rule(
    edge: dict,
) -> bool:

    subject = clean_text(
        edge.get(
            "subject",
            "",
        )
    ).upper()

    obj = clean_text(
        edge.get(
            "object",
            "",
        )
    ).upper()

    ontology_labels = {
        str(x).strip().upper()
        for x in ENTITY_TYPES
    }

    if subject in ontology_labels:
        return False

    if obj in ontology_labels:
        return False

    return True


def canonical_signature_rule(
    edge: dict,
) -> bool:

    return canonical_signature_valid(
        edge.get(
            "subject_type",
            "",
        ),
        edge.get(
            "relation",
            "",
        ),
        edge.get(
            "object_type",
            "",
        ),
    )


def evaluate_edge(
    edge: dict,
    source_text: str,
) -> dict:

    checks = {
        "subject_reference":
            subject_reference_rule(
                edge,
                source_text,
            ),

        "entity_length":
            entity_length_rule(
                edge
            ),

        "entity_schema":
            entity_schema_rule(
                edge
            ),

        "relation_schema":
            relation_schema_rule(
                edge
            ),

        "canonical_signature":
            canonical_signature_rule(
                edge
            ),

        "ontology_label_as_entity":
            ontology_label_as_entity_rule(
                edge
            ),
    }

    checks[
        "all_rules_pass"
    ] = all(
        checks.values()
    )

    return checks
