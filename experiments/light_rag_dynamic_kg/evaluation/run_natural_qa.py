#!/usr/bin/env python3
"""
Repeated Natural Language QA for FinGraph-RL.

Reuses existing Official LightRAG snapshots.
Does not rebuild or overwrite KG indexes.

Benchmark: single-answer KG-grounded natural-language QA.
Scores lexical answer mention as a diagnostic only.
Human/semantic evaluation is required for true correctness.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg.evaluation import (
    run_official_kg_qa_v2 as base
)

BASE = ROOT / "results/light_rag_dynamic_kg"
OUT_BASE = BASE / "natural_qa_v1"

MODES = ("local", "global", "hybrid")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            obj, ensure_ascii=False
        ) + "\n")


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f if x.strip()
        ]


def generate_candidates(rows):
    """
    Only generate unambiguous single-answer questions.

    A single triple can yield more than one question form,
    but ambiguity is checked against the entire snapshot.

    No KG IDs or hidden answers are included in the
    user-facing questions.
    """

    hr = defaultdict(set)
    ht = defaultdict(set)
    rt = defaultdict(set)

    unique = {}

    for r in rows:
        h = str(r["head"])
        rel = str(r["relation"])
        t = str(r["tail"])

        if not h.strip() or not rel.strip() or not t.strip():
            continue

        hr[(h, rel)].add(t)
        ht[(h, t)].add(rel)
        rt[(rel, t)].add(h)

        unique.setdefault(
            (h, rel, t), r
        )

    candidates = []

    for (h, rel, t), row in sorted(unique.items()):

        edge = {
            "head": h,
            "relation": rel,
            "tail": t,
            "event_id": str(row["event_id"]),
            "available_at": str(row["available_at"]),
        }

        if len(hr[(h, rel)]) == 1:
            candidates.append({
                "type": "FACT",
                "task": "HR_T",
                "question": (
                    f"According to the historical financial "
                    f"knowledge graph, which entity does "
                    f"{h} connect to through the "
                    f"'{rel}' relationship?"
                ),
                "gold_answer": t,
                "gold_triple": edge,
            })

        if len(ht[(h, t)]) == 1:
            candidates.append({
                "type": "RELATIONSHIP",
                "task": "HT_R",
                "question": (
                    f"How does the historical financial "
                    f"knowledge graph describe the direct "
                    f"relationship from {h} to {t}? "
                    f"Give the recorded relation label."
                ),
                "gold_answer": rel,
                "gold_triple": edge,
            })

        if len(rt[(rel, t)]) == 1:
            candidates.append({
                "type": "FACT",
                "task": "RT_H",
                "question": (
                    f"Which entity is recorded as having "
                    f"a '{rel}' relationship directed "
                    f"toward {t} in the historical graph?"
                ),
                "gold_answer": h,
                "gold_triple": edge,
            })

    return candidates


def sample_rounds(
    candidates, rounds, n, dev, seed
):
    if len(candidates) < n:
        raise RuntimeError(
            f"Only {len(candidates)} eligible questions. "
            f"Need {n}. Use a richer snapshot or lower --n."
        )

    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    # Prefer no repeated question until the pool is exhausted.
    cursor = 0
    output = []

    for round_id in range(1, rounds + 1):
        chosen = []

        while len(chosen) < n:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0

            item = pool[cursor]
            cursor += 1

            # No duplicate question within the same round.
            if any(
                x["question"] == item["question"]
                for x in chosen
            ):
                continue

            chosen.append(item)

        for i, item in enumerate(chosen):
            row = dict(item)
            row["round"] = round_id
            row["id"] = f"R{round_id:03d}_Q{i+1:03d}"
            row["split"] = "dev" if i < dev else "test"
            output.append(row)

    return output


def evaluate_answer(answer, gold):
    """
    Surface-form answer occurrence only.

    Important limitation:
    An answer can mention a gold entity while negating
    or contradicting the expected relationship.
    """

    found = base.contains_phrase(answer, gold)

    return {
        "lexical_gold_mentioned": bool(found),
        "semantic_correct": None,
        "faithfulness": None,
        "unsupported_claim": None,
    }


def write_review_csv(path, records):
    fields = [
        "round", "id", "split", "mode",
        "task", "question", "gold_answer",
        "official_answer",
        "semantic_correct",
        "faithfulness",
        "unsupported_claim",
        "reviewer_notes",
    ]

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in records:
            w.writerow({
                k: r.get(k, "")
                for k in fields
            })


def report_metrics(records):
    summary = {}

    for mode in MODES:
        test = [
            r for r in records
            if r["mode"] == mode
            and r["split"] == "test"
        ]

        if not test:
            continue

        by_round = {}

        for rid in sorted({r["round"] for r in test}):
            subset = [
                r for r in test
                if r["round"] == rid
            ]

            successful = [
                r for r in subset
                if not r["pipeline_error"]
            ]

            by_round[str(rid)] = {
                "attempted": len(subset),
                "successful": len(successful),
                "lexical_hit_rate": (
                    sum(
                        x["lexical_gold_mentioned"]
                        for x in successful
                    ) / len(successful)
                    if successful else None
                ),
            }

        successful_all = [
            r for r in test
            if not r["pipeline_error"]
        ]

        summary[mode] = {
            "test_attempted": len(test),
            "test_successful": len(successful_all),
            "lexical_hit_rate": (
                sum(
                    x["lexical_gold_mentioned"]
                    for x in successful_all
                ) / len(successful_all)
                if successful_all else None
            ),
            "rounds": by_round,
        }

    summary["_notes"] = {
        "automatic_score": "lexical diagnostic",
        "semantic_accuracy": None,
        "evidence_recall_at_k": None,
        "strict_pit_certified": False,
        "repeated_samples_not_independent": True,
        "description": (
            "KG-derived single-answer natural questions. "
            "Not the original LightRAG paper benchmark."
        ),
    }

    return summary


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--date", default="2018-08-06")
    ap.add_argument("--fingerprint", default=None)

    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--dev", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261021)

    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be positive")

    if not (0 <= args.dev < args.n):
        raise ValueError("Invalid dev/test split")

    workspace = base.find_snapshot(
        args.ticker,
        args.date,
        args.fingerprint
    )

    rows, marker, manifest = base.load_historical_snapshot(
        args.ticker,
        args.date,
        workspace
    )

    candidates = generate_candidates(rows)

    questions = sample_rounds(
        candidates,
        args.rounds,
        args.n,
        args.dev,
        args.seed
    )

    config = {
        "ticker": args.ticker,
        "date": args.date,
        "snapshot": str(workspace),
        "rounds": args.rounds,
        "questions_per_round": args.n,
        "dev_per_round": args.dev,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "question_count": len(questions),
        "modes": list(MODES),
        "version": "natural_qa_v1",
    }

    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]

    out = OUT_BASE / (
        f"{args.ticker}_{args.date}_{digest}"
    )

    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "config.json"

    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise RuntimeError("Run configuration mismatch")
    else:
        save_json(config_path, config)

    dataset_path = out / "questions.jsonl"

    if not dataset_path.exists():
        for q in questions:
            append_jsonl(dataset_path, q)
    elif load_jsonl(dataset_path) != questions:
        raise RuntimeError("Existing dataset differs")

    print("[SNAPSHOT]", workspace)
    print("[ELIGIBLE QA]", len(candidates))
    print("[ROUNDS]", args.rounds)
    print("[QUESTIONS]", len(questions))
    print("[TOTAL QA CALLS]", len(questions) * 3)
    print("[OUTPUT]", out)

    if args.dry_run:
        print("[DRY RUN COMPLETE]")
        return

    result_path = out / "qa_results.jsonl"
    existing = load_jsonl(result_path)

    finished = {
        (r["round"], r["id"], r["mode"])
        for r in existing
        if not r.get("pipeline_error")
    }

    rag = base.create_rag(workspace)
    await rag.initialize_storages()

    try:
        for mode in MODES:
            for q in questions:
                key = (q["round"], q["id"], mode)

                if key in finished:
                    continue

                print(
                    f"[ROUND {q['round']}] "
                    f"[{mode}] {q['id']} {q['task']}",
                    flush=True
                )

                record = {
                    **q,
                    "mode": mode,
                    "official_answer": "",
                    "retrieved_context": "",
                    "pipeline_error": None,
                }

                try:
                    context = await base.get_retrieved_context(
                        rag, q["question"], mode
                    )

                    answer = await base.run_official_qa(
                        rag, q["question"], mode
                    )

                    record["retrieved_context"] = context
                    record["official_answer"] = answer

                    record.update(
                        evaluate_answer(
                            answer, q["gold_answer"]
                        )
                    )

                except Exception as exc:
                    record["pipeline_error"] = repr(exc)
                    record.update({
                        "lexical_gold_mentioned": False,
                        "semantic_correct": None,
                        "faithfulness": None,
                        "unsupported_claim": None,
                    })

                append_jsonl(result_path, record)

                print(
                    "  lexical_hit=",
                    record["lexical_gold_mentioned"],
                    "error=",
                    record["pipeline_error"],
                    flush=True
                )

    finally:
        await rag.finalize_storages()

    # Keep latest successful attempt for each question/mode.
    latest = {}
    for r in load_jsonl(result_path):
        key = (r["round"], r["id"], r["mode"])
        if key not in latest or not r["pipeline_error"]:
            latest[key] = r

    final_records = list(latest.values())

    save_json(
        out / "metrics.json",
        report_metrics(final_records)
    )

    write_review_csv(
        out / "human_review.csv",
        final_records
    )

    print("\n[COMPLETE]", out)
    print("[METRICS]", out / "metrics.json")
    print("[REVIEW]", out / "human_review.csv")


if __name__ == "__main__":
    asyncio.run(main())
