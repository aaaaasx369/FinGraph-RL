#!/usr/bin/env python3
"""
FinGraph-RL: Historical LightRAG QA Evaluation

- Reuse an existing indexed official LightRAG snapshot
- Audit selected historical KG triples and dates
- Generate 25 QA questions (2 dev + 23 test)
- Evaluate local/global/hybrid retrieval
- Ask Qwen using retrieved context
- Save context, predictions, metrics and errors
- Never overwrite existing evaluation runs
- Never modify the original LightRAG snapshot

IMPORTANT:
QA gold answers come from the selected KG itself.
This tests knowledge recovery, not real-world KG truth.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg import (
    run_official_lightrag as official
)

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import (
    ollama_model_complete,
    ollama_embed,
)
from lightrag.utils import EmbeddingFunc
from functools import partial

BASE = ROOT / "results/light_rag_dynamic_kg"
OUTPUT_ROOT = BASE / "kg_qa_evaluation_v1"

TASKS = ["HR_T", "HT_R", "RT_H"]
MODES = ["local", "global", "hybrid"]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, ensure_ascii=False,
            indent=2, default=str
        ),
        encoding="utf-8"
    )


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, default=str)
                + "\n"
            )


def save_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=keys, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def norm(value):
    return re.sub(
        r"\s+", " ", str(value).strip().casefold()
    )


def triple_key(row):
    return (
        str(row["head_id"]),
        str(row["relation_id"]),
        str(row["tail_id"])
    )


def choose_workspace(ticker, day, fingerprint=None):
    folder = BASE / "snapshots"
    candidates = sorted(
        folder.glob(f"{ticker}_{day}_*")
    )

    candidates = [
        p for p in candidates
        if p.is_dir()
        and (p / "index_ready.json").is_file()
        and (p / "source_manifest.json").is_file()
    ]

    if fingerprint:
        candidates = [
            p for p in candidates
            if p.name.endswith("_" + fingerprint)
        ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one matching snapshot; "
            f"found {len(candidates)}:\n"
            + "\n".join(str(p) for p in candidates)
            + "\nSpecify --fingerprint if necessary."
        )

    return candidates[0]


def load_selected_kg(ticker, day, workspace):
    entities = official.load_entity_names()
    relations = official.load_relation_names()
    companies = official.load_company_mapping()

    marker = json.loads(
        (workspace / "index_ready.json").read_text()
    )
    kg_k = int(marker.get("kg_events", 80))

    records = official.load_historical_kg(
        ticker,
        pd.Timestamp(day),
        kg_k,
        entities,
        relations,
        companies
    )

    manifest = json.loads(
        (workspace / "source_manifest.json").read_text()
    )

    allowed_ids = {
        str(x["original_event_id"])
        for x in manifest
        if x.get("original_event_id") is not None
    }

    selected = [
        r for r in records
        if str(r["event_id"]) in allowed_ids
        and str(r["head"]).strip()
        != str(r["tail"]).strip()
    ]

    loaded_ids = {str(r["event_id"]) for r in selected}

    if loaded_ids != allowed_ids:
        missing = sorted(allowed_ids - loaded_ids)
        raise RuntimeError(
            "Snapshot KG source IDs do not match "
            f"reconstructed input. Missing: {missing[:10]}"
        )

    # Historical availability must precede decision date.
    cutoff = pd.Timestamp(day)

    for r in selected:
        at = pd.to_datetime(
            r["available_at"], errors="coerce",
            utc=True
        )
        if pd.isna(at):
            raise RuntimeError("Invalid available_at")

        if at.tz_localize(None).normalize() >= cutoff:
            raise RuntimeError(
                f"Temporal violation: {r['event_id']}"
            )

    if not selected:
        raise RuntimeError("Snapshot has no eligible KG triples")

    return selected, manifest, marker


def audit_snapshot(rows, manifest, marker, ticker, day):
    entities = set()
    relations = Counter()
    triples = Counter()
    sources = Counter()
    duplicate_ids = Counter()
    collisions = []

    for r in rows:
        entities.update([r["head_id"], r["tail_id"]])
        relations[r["relation_id"]] += 1
        triples[triple_key(r)] += 1
        sources[r["source"]] += 1
        duplicate_ids[r["event_id"]] += 1

        if r["head"] == r["tail"]:
            collisions.append(r["event_id"])

    return {
        "ticker": ticker,
        "decision_date": day,
        "selected_kg_events": len(rows),
        "unique_entity_ids": len(entities),
        "unique_relation_ids": len(relations),
        "unique_triples": len(triples),
        "repeated_triple_rows": sum(
            n - 1 for n in triples.values() if n > 1
        ),
        "duplicate_event_ids": sum(
            n - 1 for n in duplicate_ids.values() if n > 1
        ),
        "self_loop_after_import": len(collisions),
        "relation_distribution": dict(relations),
        "source_distribution": dict(sources),
        "source_manifest_entries": len(manifest),
        "index_marker": marker,
        "selected_date_filter_pass": True,
        "strict_pit_certified": False,
        "semantic_quality_certified": False,
        "notes": [
            "Audits imported snapshot KG, not entire Unified KG",
            "Repeated temporal observations are not always errors",
            "Original source publication timestamps not certified",
            "Semantic correctness requires external validation"
        ]
    }


def build_questions(rows, count=25, dev_count=2):
    rng = random.Random(20261021)

    unique = {}
    for row in rows:
        unique.setdefault(triple_key(row), row)

    items = list(unique.values())
    if len(items) < count:
        raise RuntimeError(
            f"Snapshot has only {len(items)} unique triples; "
            f"cannot generate {count} distinct questions."
        )

    # Use as many relation types as available.
    by_relation = defaultdict(list)
    for row in items:
        by_relation[row["relation_id"]].append(row)

    for group in by_relation.values():
        rng.shuffle(group)

    chosen = []
    while len(chosen) < count:
        progressed = False
        for relation in sorted(by_relation):
            if by_relation[relation] and len(chosen) < count:
                chosen.append(by_relation[relation].pop())
                progressed = True
        if not progressed:
            break

    rng.shuffle(chosen)

    questions = []

    for i, row in enumerate(chosen):
        task = TASKS[i % 3]

        h = row["head"]
        r = row["relation"]
        t = row["tail"]

        if task == "HR_T":
            question = (
                f"According to the historical graph, "
                f"which tail entity is connected from "
                f"'{h}' via relation '{r}'?"
            )
            gold = sorted({
                x["tail"]
                for x in rows
                if x["head_id"] == row["head_id"]
                and x["relation_id"] == row["relation_id"]
            })
        elif task == "HT_R":
            question = (
                f"According to the historical graph, "
                f"what relation connects head '{h}' "
                f"to tail '{t}'?"
            )
            gold = sorted({
                x["relation"]
                for x in rows
                if x["head_id"] == row["head_id"]
                and x["tail_id"] == row["tail_id"]
            })
        else:
            question = (
                f"According to the historical graph, "
                f"which head entity connects to tail "
                f"'{t}' via relation '{r}'?"
            )
            gold = sorted({
                x["head"]
                for x in rows
                if x["relation_id"] == row["relation_id"]
                and x["tail_id"] == row["tail_id"]
            })

        questions.append({
            "id": f"QA_{i+1:03d}",
            "split": "dev" if i < dev_count else "test",
            "task": task,
            "question": question,
            "gold_answers": gold,
            "gold_event_id": str(row["event_id"]),
            "gold_triple": {
                "head": h,
                "relation": r,
                "tail": t,
                "head_id": row["head_id"],
                "relation_id": row["relation_id"],
                "tail_id": row["tail_id"]
            },
            "multi_answer": len(gold) > 1,
        })

    return questions


def call_qwen(question, context, model, host, timeout):
    system = """
You are evaluating a historical financial knowledge graph.

Answer the question ONLY from the retrieved evidence.
Do not use outside knowledge or future information.

Important:
- Head, relation, and tail direction matters.
- Generic graph relations have no implied economic meaning.
- Multiple answers may be correct.
- Do not invent missing entities or relationships.
- If evidence is insufficient, return an empty answers list.
- Return a JSON object only.

Schema:
{
  "answers": ["exact name"],
  "reason": "brief evidence-based explanation"
}
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "QUESTION:\n" + question
                    + "\n\nRETRIEVED EVIDENCE:\n"
                    + context[:30000]
                )
            }
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192
        }
    }

    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        req, timeout=timeout
    ) as response:
        envelope = json.loads(
            response.read().decode("utf-8")
        )

    raw = envelope.get("message", {}).get("content", "")

    try:
        obj = json.loads(raw)
        answers = obj.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError("answers must be a list")
        return {
            "answers": [str(x) for x in answers],
            "reason": str(obj.get("reason", "")),
            "raw": raw,
            "parse_error": False
        }
    except (ValueError, TypeError):
        return {
            "answers": [],
            "reason": "",
            "raw": raw,
            "parse_error": True
        }


def judge(predicted, gold):
    a = {norm(x) for x in predicted}
    b = {norm(x) for x in gold}

    return {
        "exact_set_correct": a == b,
        "answer_hit": bool(a & b),
        "answer_recall": len(a & b) / len(b) if b else 0.0,
        "unsupported_answer_count": len(a - b)
    }


def create_rag(workspace):
    return LightRAG(
        working_dir=str(workspace),
        llm_model_name=official.LLM_MODEL,
        llm_model_func=ollama_model_complete,
        llm_model_kwargs={
            "host": official.HOST,
            "options": {"num_ctx": 8192},
            "timeout": 300
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=official.EMBED_MODEL,
                host=official.HOST
            )
        )
    )


async def retrieve_context(rag, question, mode):
    result = await rag.aquery(
        question,
        param=QueryParam(
            mode=mode,
            only_need_context=True,
            stream=False
        )
    )
    return str(result or "")


def metrics_for(records):
    results = {}

    for mode in MODES:
        for split in ("dev", "test"):
            group = [
                r for r in records
                if r["mode"] == mode and r["split"] == split
            ]

            if not group:
                continue

            n = len(group)
            key = f"{mode}/{split}"

            results[key] = {
                "questions": n,
                "exact_set_accuracy": sum(
                    x["exact_set_correct"] for x in group
                ) / n,
                "answer_hit_rate": sum(
                    x["answer_hit"] for x in group
                ) / n,
                "mean_answer_recall": sum(
                    x["answer_recall"] for x in group
                ) / n,
                "empty_context_rate": sum(
                    x["empty_context"] for x in group
                ) / n,
                "parse_error_rate": sum(
                    x["parse_error"] for x in group
                ) / n,
                "accuracy_by_task": {},
            }

            for task in TASKS:
                subset = [
                    x for x in group if x["task"] == task
                ]
                if subset:
                    results[key]["accuracy_by_task"][task] = {
                        "count": len(subset),
                        "accuracy": sum(
                            x["exact_set_correct"] for x in subset
                        ) / len(subset)
                    }

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--date", default="2018-08-06")
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--dev", type=int, default=2)
    parser.add_argument("--model", default=official.LLM_MODEL)
    parser.add_argument("--ollama-host", default=official.HOST)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build QA and audit only; no LightRAG or Qwen calls"
    )
    args = parser.parse_args()

    if args.n < 2 or not 0 <= args.dev < args.n:
        raise ValueError("Invalid n/dev split")

    workspace = choose_workspace(
        args.ticker, args.date, args.fingerprint
    )

    print("[SNAPSHOT]", workspace, flush=True)

    rows, manifest, marker = load_selected_kg(
        args.ticker, args.date, workspace
    )

    audit = audit_snapshot(
        rows, manifest, marker, args.ticker, args.date
    )

    questions = build_questions(
        rows, args.n, args.dev
    )

    # Unique output directory; never overwrite prior runs.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    out = OUTPUT_ROOT / (
        f"{args.ticker}_{args.date}_"
        f"{workspace.name.rsplit('_', 1)[-1]}_{timestamp}"
    )
    out.mkdir(parents=True, exist_ok=False)

    save_json(out / "snapshot_audit.json", audit)
    save_jsonl(out / "selected_kg_events.jsonl", rows)
    save_jsonl(out / "qa_dataset.jsonl", questions)
    save_jsonl(
        out / "qa_dev.jsonl",
        [q for q in questions if q["split"] == "dev"]
    )
    save_jsonl(
        out / "qa_test.jsonl",
        [q for q in questions if q["split"] == "test"]
    )

    save_csv(
        out / "semantic_review.csv",
        [{
            "event_id": r["event_id"],
            "head": r["head"],
            "relation": r["relation"],
            "tail": r["tail"],
            "available_at": r["available_at"],
            "source": r["source"],
            "source_verified": "",
            "triple_factually_correct": "",
            "notes": ""
        } for r in rows]
    )

    print("[AUDIT]", json.dumps(
        audit, ensure_ascii=False, indent=2
    ))
    print("[QA]", len(questions))
    print("[DEV]", args.dev)
    print("[TEST]", len(questions) - args.dev)
    print("[MULTI ANSWER]", sum(
        q["multi_answer"] for q in questions
    ))
    print("[OUTPUT]", out)

    if args.dry_run:
        print("[DRY RUN COMPLETE] No LLM calls.")
        return

    rag = create_rag(workspace)
    await rag.initialize_storages()

    outputs = []

    try:
        for mode in MODES:
            print(f"\n========== {mode.upper()} ==========")

            for i, q in enumerate(questions, 1):
                print(
                    f"[{mode}] {i}/{len(questions)} "
                    f"{q['id']} {q['task']}",
                    flush=True
                )

                try:
                    context = await retrieve_context(
                        rag, q["question"], mode
                    )

                    answer = call_qwen(
                        q["question"], context,
                        args.model,
                        args.ollama_host,
                        args.timeout
                    )

                    scores = judge(
                        answer["answers"],
                        q["gold_answers"]
                    )

                    if answer["parse_error"]:
                        error_type = "PARSE_ERROR"
                    elif scores["exact_set_correct"]:
                        error_type = "CORRECT"
                    elif not context.strip():
                        error_type = "NO_CONTEXT"
                    elif not answer["answers"]:
                        error_type = "ABSTENTION"
                    else:
                        error_type = "ANSWER_MISMATCH"

                    record = {
                        "id": q["id"],
                        "split": q["split"],
                        "mode": mode,
                        "task": q["task"],
                        "question": q["question"],
                        "gold_answers": q["gold_answers"],
                        "gold_event_id": q["gold_event_id"],
                        "gold_triple": q["gold_triple"],
                        "predicted_answers": answer["answers"],
                        "qwen_reason": answer["reason"],
                        "qwen_raw": answer["raw"],
                        "retrieved_context": context,
                        "empty_context": not bool(context.strip()),
                        "parse_error": answer["parse_error"],
                        "error_type": error_type,
                        **scores
                    }

                except Exception as exc:
                    record = {
                        "id": q["id"],
                        "split": q["split"],
                        "mode": mode,
                        "task": q["task"],
                        "question": q["question"],
                        "gold_answers": q["gold_answers"],
                        "gold_event_id": q["gold_event_id"],
                        "gold_triple": q["gold_triple"],
                        "predicted_answers": [],
                        "qwen_reason": "",
                        "qwen_raw": "",
                        "retrieved_context": "",
                        "empty_context": True,
                        "parse_error": False,
                        "error_type": "PIPELINE_ERROR",
                        "error_message": repr(exc),
                        "exact_set_correct": False,
                        "answer_hit": False,
                        "answer_recall": 0.0,
                        "unsupported_answer_count": 0
                    }

                outputs.append(record)

                # Preserve intermediate results after every question.
                save_jsonl(
                    out / "qa_predictions.jsonl", outputs
                )

                print(
                    f"  correct={record['exact_set_correct']} "
                    f"error={record['error_type']}",
                    flush=True
                )

    finally:
        await rag.finalize_storages()

    successful = [
        x for x in outputs
        if x["error_type"] != "PIPELINE_ERROR"
    ]
    metrics = metrics_for(successful)
    metrics["_run_status"] = {
        "attempted": len(outputs),
        "completed_without_pipeline_error": len(successful),
        "pipeline_errors": len(outputs) - len(successful),
        "strict_pit_certified": False,
        "kg_semantics_certified": False,
        "evidence_recall_at_k": None,
        "note": (
            "Evidence Recall@K requires structured, ranked "
            "retrieved triples; unranked text is insufficient."
        )
    }

    save_json(out / "metrics.json", metrics)

    errors = [
        {
            "id": x["id"],
            "mode": x["mode"],
            "split": x["split"],
            "task": x["task"],
            "error_type": x["error_type"],
            "gold_answers": json.dumps(
                x["gold_answers"], ensure_ascii=False
            ),
            "predicted_answers": json.dumps(
                x["predicted_answers"], ensure_ascii=False
            ),
            "question": x["question"]
        }
        for x in outputs
        if x["error_type"] != "CORRECT"
    ]

    save_csv(out / "error_analysis.csv", errors)

    print("\n========== FINAL METRICS ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n[SAVED]", out)


if __name__ == "__main__":
    asyncio.run(main())
