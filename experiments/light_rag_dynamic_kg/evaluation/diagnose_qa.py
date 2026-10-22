#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(
    "results/light_rag_dynamic_kg/kg_qa_evaluation_v1/"
    "AAPL_2018-08-06_d91c9f81398cd6ac_"
    "20261021T034705953967Z"
)

INPUT = ROOT / "qa_predictions.jsonl"
OUTPUT = ROOT / "context_diagnosis.csv"


def normalize(text):
    return re.sub(
        r"\s+", " ", str(text).casefold()
    )


def contains(context, value):
    return normalize(value) in normalize(context)


records = [
    json.loads(line)
    for line in INPUT.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = []

for record in records:

    context = record.get("retrieved_context", "")
    triple = record["gold_triple"]

    head_found = contains(
        context, triple["head"]
    )

    relation_found = contains(
        context, triple["relation"]
    )

    tail_found = contains(
        context, triple["tail"]
    )

    all_terms_found = (
        head_found
        and relation_found
        and tail_found
    )

    results.append({
        "id": record["id"],
        "mode": record["mode"],
        "task": record["task"],
        "context_chars": len(context),
        "head_found": head_found,
        "relation_found": relation_found,
        "tail_found": tail_found,
        "all_gold_terms_found": all_terms_found,
        "parse_error": record["parse_error"],
        "answer_correct": record["exact_set_correct"],
        "error_type": record["error_type"],
    })


with OUTPUT.open(
    "w", encoding="utf-8", newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0])
    )

    writer.writeheader()
    writer.writerows(results)


print("\n========== CONTEXT DIAGNOSIS ==========")

for mode in ("local", "global", "hybrid"):

    subset = [
        x for x in results
        if x["mode"] == mode
    ]

    n = len(subset)

    print(f"\n[{mode.upper()}]")

    print("Questions:", n)

    print(
        "Head text found:",
        sum(x["head_found"] for x in subset),
        "/", n
    )

    print(
        "Relation text found:",
        sum(x["relation_found"] for x in subset),
        "/", n
    )

    print(
        "Tail text found:",
        sum(x["tail_found"] for x in subset),
        "/", n
    )

    print(
        "All three terms found:",
        sum(x["all_gold_terms_found"] for x in subset),
        "/", n
    )

    print(
        "Parse errors:",
        sum(x["parse_error"] for x in subset),
        "/", n
    )


print("\n========== FIRST THREE CONTEXTS ==========")

for record in records[:3]:

    print("\n" + "=" * 80)

    print("ID:", record["id"])
    print("Mode:", record["mode"])
    print("Question:", record["question"])
    print("Gold:", record["gold_answers"])

    print("\nRETRIEVED CONTEXT:")

    print(
        record.get("retrieved_context", "")[:6000]
    )


print("\n[SAVED]", OUTPUT)
print("No model calls were made.")
