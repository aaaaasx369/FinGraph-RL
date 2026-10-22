#!/usr/bin/env python3
"""
Official LightRAG KG QA Evaluation V2

Uses the existing FinGraph-RL historical LightRAG snapshot.

Main features:
1. Reuse existing indexed historical snapshot
2. Reconstruct and audit historical KG evidence
3. Load the SAME 25 QA questions from previous benchmark
4. Official LightRAG local/global/hybrid QA
5. Save retrieved context separately
6. Evaluate exact-set accuracy, answer precision/recall/F1
7. Diagnose textual triple evidence coverage
8. Save raw answers and error analysis
9. Never recreate the KG index
10. Never overwrite earlier evaluation reports

IMPORTANT:
- Gold answers represent knowledge recorded in KG.
- Lexical answer scoring is a heuristic for natural-language
  output, not semantic entailment certification.
- Textual evidence coverage is not ranked Triple Recall@K.
- Strict point-in-time validity remains uncertified.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

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


BASE = ROOT / "results/light_rag_dynamic_kg"
OLD_QA = BASE / "kg_qa_evaluation_v1"
OUTPUT = BASE / "kg_qa_official_v2"

MODES = ("local", "global", "hybrid")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )


def save_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize(text):
    text = str(text).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_phrase(text, phrase):
    text = normalize(text)
    phrase = normalize(phrase)

    if not phrase:
        return False

    # Require phrase boundaries rather than arbitrary substrings.
    pattern = (
        r"(?<!\w)"
        + re.escape(phrase)
        + r"(?!\w)"
    )

    return re.search(pattern, text) is not None


def answer_scores(answer, gold_answers):
    """
    Lexical evaluation of the official natural-language answer.

    This is deliberately labelled a proxy metric:
    the mention of an answer does not prove the model
    asserted the correct directed relation.
    """

    gold = list(dict.fromkeys(
        str(x) for x in gold_answers
    ))

    found = [
        x for x in gold
        if contains_phrase(answer, x)
    ]

    missing = [
        x for x in gold
        if x not in found
    ]

    recall = (
        len(found) / len(gold)
        if gold else 0.0
    )

    return {
        "gold_answer_count": len(gold),
        "matched_gold_answers": found,
        "missing_gold_answers": missing,
        "lexical_answer_recall": recall,
        "lexical_all_gold_mentioned": (
            len(found) == len(gold)
        ),
        "lexical_any_gold_mentioned": bool(found),
        "single_answer_lexical_hit": (
            bool(found) if len(gold) == 1 else None
        ),
        "exact_set_accuracy": None,
        "answer_precision": None,
        "answer_f1": None,
        "metric_note": (
            "Natural-language mentions cannot establish "
            "a complete predicted answer set or entailment. "
            "Precision, F1 and exact-set accuracy require "
            "structured answer extraction."
        ),
    }


def context_scores(context, triple):
    """
    Surface-form evidence diagnosis only.

    All three terms being present does NOT prove
    the correct directed triple was retrieved.
    """

    head = str(triple["head"])
    relation = str(triple["relation"])
    tail = str(triple["tail"])

    h = contains_phrase(context, head)
    r = contains_phrase(context, relation)
    t = contains_phrase(context, tail)

    # The KG import writes event identifiers into
    # relationship descriptions. For now we only
    # diagnose exact source-event mentions separately.
    return {
        "head_text_found": h,
        "relation_text_found": r,
        "tail_text_found": t,
        "all_three_terms_found": h and r and t,
        "context_empty": not bool(context.strip()),
        "context_chars": len(context),
    }


def find_snapshot(ticker, date, fingerprint):
    folder = BASE / "snapshots"

    candidates = sorted(
        x for x in folder.glob(
            f"{ticker}_{date}_*"
        )
        if x.is_dir()
        and (x / "index_ready.json").is_file()
        and (x / "source_manifest.json").is_file()
    )

    if fingerprint:
        candidates = [
            x for x in candidates
            if x.name.endswith("_" + fingerprint)
        ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one indexed snapshot.\n"
            + "\n".join(map(str, candidates))
            + "\nUse --fingerprint to select one."
        )

    return candidates[0]


def find_previous_dataset(ticker, date, old_run=None):
    if old_run:
        candidates = [Path(old_run)]
    else:
        candidates = sorted(
            OLD_QA.glob(f"{ticker}_{date}_*"),
            reverse=True,
        )

    for folder in candidates:
        path = folder / "qa_dataset.jsonl"

        if path.is_file():
            return path

    raise FileNotFoundError(
        "Previous QA dataset not found. "
        "Pass --qa-dataset /path/to/qa_dataset.jsonl"
    )


def load_historical_snapshot(ticker, date, workspace):
    """
    Reuse the original snapshot selection logic.
    Never insert another Custom KG.
    """

    entity_names = official.load_entity_names()
    relation_names = official.load_relation_names()
    companies = official.load_company_mapping()

    marker = json.loads(
        (workspace / "index_ready.json").read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        (workspace / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    kg_events = int(marker.get("kg_events", 80))

    reconstructed = official.load_historical_kg(
        ticker,
        pd.Timestamp(date),
        kg_events,
        entity_names,
        relation_names,
        companies,
    )

    expected_ids = {
        str(x["original_event_id"])
        for x in manifest
        if x.get("original_event_id") is not None
    }

    rows = [
        r for r in reconstructed
        if str(r["event_id"]) in expected_ids
        and str(r["head"]).strip()
        != str(r["tail"]).strip()
    ]

    actual_ids = {
        str(x["event_id"]) for x in rows
    }

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Historical KG mismatch with snapshot manifest. "
            "Refusing to evaluate a different graph."
        )

    cutoff = pd.Timestamp(date)

    for r in rows:
        available = pd.to_datetime(
            r["available_at"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(available):
            raise RuntimeError(
                f"Invalid date: {r['event_id']}"
            )

        if available.tz_localize(
            None
        ).normalize() >= cutoff:
            raise RuntimeError(
                f"Future evidence: {r['event_id']}"
            )

    return rows, marker, manifest


def create_rag(workspace):
    """
    Match the existing Official LightRAG configuration.
    """

    rag = LightRAG(
        working_dir=str(workspace),

        llm_model_name=official.LLM_MODEL,

        llm_model_func=ollama_model_complete,

        llm_model_kwargs={
            "host": official.HOST,
            "options": {"num_ctx": 8192},
            "timeout": 300,
        },

        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=official.EMBED_MODEL,
                host=official.HOST,
            ),
        ),
    )

    return rag


async def run_official_qa(rag, question, mode):
    """
    Native Official LightRAG QA.

    Unlike V1, final answer generation is performed
    by LightRAG itself.
    """

    answer = await rag.aquery(
        question,
        param=QueryParam(
            mode=mode,
            stream=False,
        ),
    )

    return str(answer or "")


async def get_retrieved_context(rag, question, mode):
    """
    Retrieve context for diagnosis, separately
    from native answer generation.
    """

    context = await rag.aquery(
        question,
        param=QueryParam(
            mode=mode,
            only_need_context=True,
            stream=False,
        ),
    )

    return str(context or "")


def error_category(record):
    if record.get("pipeline_error"):
        return "PIPELINE_ERROR"

    if record["context_empty"]:
        return "NO_CONTEXT"

    if not record["all_three_terms_found"]:
        return "GOLD_TERMS_INCOMPLETE"

    if not record["official_answer"].strip():
        return "EMPTY_ANSWER"

    if not record["lexical_any_gold_mentioned"]:
        return "ANSWER_NO_GOLD_MENTION"

    if record["lexical_all_gold_mentioned"]:
        return "ALL_GOLD_TERMS_MENTIONED"

    return "PARTIAL_GOLD_MENTION"


def aggregate(records):
    output = {}

    for mode in MODES:
        for split in ("dev", "test"):
            group = [
                r for r in records
                if r["mode"] == mode
                and r["split"] == split
            ]

            if not group:
                continue

            completed = [
                r for r in group
                if not r.get("pipeline_error")
            ]

            key = f"{mode}/{split}"

            report = {
                "attempted": len(group),
                "successful": len(completed),
                "pipeline_errors": (
                    len(group) - len(completed)
                ),
                "metrics": None,
                "accuracy_by_task": {},
            }

            if completed:
                n = len(completed)

                report["metrics"] = {
                    "lexical_any_gold_rate": sum(
                        r["lexical_any_gold_mentioned"]
                        for r in completed
                    ) / n,

                    "lexical_all_gold_rate": sum(
                        r["lexical_all_gold_mentioned"]
                        for r in completed
                    ) / n,

                    "mean_lexical_answer_recall": sum(
                        r["lexical_answer_recall"]
                        for r in completed
                    ) / n,

                    "context_all_terms_rate": sum(
                        r["all_three_terms_found"]
                        for r in completed
                    ) / n,

                    "empty_context_rate": sum(
                        r["context_empty"]
                        for r in completed
                    ) / n,

                    "exact_set_accuracy": None,
                    "answer_precision": None,
                    "answer_f1": None,
                    "evidence_recall_at_k": None,
                }

                for task in ("HR_T", "HT_R", "RT_H"):
                    subset = [
                        r for r in completed
                        if r["task"] == task
                    ]

                    if subset:
                        report["accuracy_by_task"][task] = {
                            "count": len(subset),
                            "lexical_any_gold_rate": sum(
                                r["lexical_any_gold_mentioned"]
                                for r in subset
                            ) / len(subset),
                            "mean_lexical_recall": sum(
                                r["lexical_answer_recall"]
                                for r in subset
                            ) / len(subset),
                        }

            output[key] = report

    return output


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ticker",
        default="AAPL",
    )

    ap.add_argument(
        "--date",
        default="2018-08-06",
    )

    ap.add_argument(
        "--fingerprint",
        default=None,
    )

    ap.add_argument(
        "--old-run",
        default=None,
    )

    ap.add_argument(
        "--qa-dataset",
        default=None,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of QA questions. 0 means all.",
    )

    ap.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="all",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    workspace = find_snapshot(
        args.ticker,
        args.date,
        args.fingerprint,
    )

    if args.qa_dataset:
        dataset_path = Path(args.qa_dataset)
    else:
        dataset_path = find_previous_dataset(
            args.ticker,
            args.date,
            args.old_run,
        )

    questions = load_jsonl(dataset_path)

    if args.split != "all":
        questions = [
            q for q in questions
            if q["split"] == args.split
        ]

    if args.limit > 0:
        questions = questions[:args.limit]

    if not questions:
        raise RuntimeError("No QA questions selected")

    rows, marker, manifest = load_historical_snapshot(
        args.ticker,
        args.date,
        workspace,
    )

    valid_triples = {
        (
            str(r["head_id"]),
            str(r["relation_id"]),
            str(r["tail_id"]),
        )
        for r in rows
    }

    for q in questions:
        t = q["gold_triple"]

        key = (
            str(t["head_id"]),
            str(t["relation_id"]),
            str(t["tail_id"]),
        )

        if key not in valid_triples:
            raise RuntimeError(
                "QA gold triple absent from indexed snapshot: "
                + q["id"]
            )

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    output = OUTPUT / (
        f"{args.ticker}_{args.date}_"
        f"{run_id}"
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_report = {
        "ticker": args.ticker,
        "date": args.date,
        "snapshot": str(workspace),
        "qa_dataset": str(dataset_path),
        "questions": len(questions),
        "modes": list(MODES),
        "llm": official.LLM_MODEL,
        "embedding": official.EMBED_MODEL,
        "ollama_host": official.HOST,
        "snapshot_kg_events": len(rows),
        "index_marker": marker,
        "strict_pit_certified": False,
        "qa_gold_source": "existing_historical_KG",
        "native_official_lightrag_qa": True,
        "evaluation_version": "official_v2",
    }

    save_json(
        output / "run_manifest.json",
        manifest_report,
    )

    save_jsonl(
        output / "qa_questions.jsonl",
        questions,
    )

    save_jsonl(
        output / "selected_kg_events.jsonl",
        rows,
    )

    print("=" * 75)
    print("OFFICIAL LIGHTRAG KG QA V2")
    print("=" * 75)
    print("Snapshot:", workspace)
    print("QA dataset:", dataset_path)
    print("Questions:", len(questions))
    print("Expected evaluations:", len(questions) * 3)
    print("KG events:", len(rows))
    print("Qwen:", official.LLM_MODEL)
    print("Output:", output)

    if args.dry_run:
        print("[DRY RUN COMPLETE]")
        print("No LightRAG query executed.")
        return

    rag = create_rag(workspace)

    await rag.initialize_storages()

    results = []

    try:
        for mode in MODES:

            print(
                f"\n========== {mode.upper()} ==========",
                flush=True,
            )

            for i, q in enumerate(questions, 1):

                print(
                    f"[{mode}] {i}/{len(questions)} "
                    f"{q['id']} {q['task']}",
                    flush=True,
                )

                record = {
                    "id": q["id"],
                    "split": q["split"],
                    "mode": mode,
                    "task": q["task"],
                    "question": q["question"],
                    "gold_answers": q["gold_answers"],
                    "gold_event_id": q["gold_event_id"],
                    "gold_triple": q["gold_triple"],
                    "official_answer": "",
                    "retrieved_context": "",
                    "pipeline_error": None,
                }

                try:
                    # First: native retrieval context.
                    context = await get_retrieved_context(
                        rag,
                        q["question"],
                        mode,
                    )

                    # Second: official native QA generation.
                    answer = await run_official_qa(
                        rag,
                        q["question"],
                        mode,
                    )

                    record["official_answer"] = answer
                    record["retrieved_context"] = context

                    record.update(
                        context_scores(
                            context,
                            q["gold_triple"],
                        )
                    )

                    record.update(
                        answer_scores(
                            answer,
                            q["gold_answers"],
                        )
                    )

                except Exception as exc:

                    record["pipeline_error"] = repr(exc)

                    record.update({
                        "head_text_found": False,
                        "relation_text_found": False,
                        "tail_text_found": False,
                        "all_three_terms_found": False,
                        "context_empty": True,
                        "context_chars": 0,
                        "gold_answer_count": len(q["gold_answers"]),
                        "matched_gold_answers": [],
                        "missing_gold_answers": q["gold_answers"],
                        "lexical_answer_recall": 0.0,
                        "lexical_all_gold_mentioned": False,
                        "lexical_any_gold_mentioned": False,
                        "single_answer_lexical_hit": None,
                        "exact_set_accuracy": None,
                        "answer_precision": None,
                        "answer_f1": None,
                    })

                record["error_type"] = error_category(
                    record
                )

                results.append(record)

                # Checkpoint: preserve all completed questions.
                save_jsonl(
                    output / "qa_results.jsonl",
                    results,
                )

                print(
                    "  Error:",
                    record["error_type"],
                    "LexicalRecall:",
                    record["lexical_answer_recall"],
                    flush=True,
                )

    finally:
        await rag.finalize_storages()

    metrics = aggregate(results)

    metrics["_metadata"] = {
        "total_attempted": len(results),
        "pipeline_errors": sum(
            bool(x["pipeline_error"]) for x in results
        ),
        "native_official_lightrag": True,
        "strict_pit_certified": False,
        "semantic_kg_quality_certified": False,
        "exact_set_accuracy": None,
        "evidence_recall_at_k": None,
        "note": (
            "Native answers are free text. Lexical "
            "mention metrics are diagnostic proxies. "
            "Structured answer and ranked evidence "
            "evaluation remain separate tasks."
        ),
    }

    save_json(
        output / "metrics.json",
        metrics,
    )

    errors = []

    for r in results:
        if r["error_type"] == "ALL_GOLD_TERMS_MENTIONED":
            continue

        errors.append({
            "id": r["id"],
            "mode": r["mode"],
            "split": r["split"],
            "task": r["task"],
            "error_type": r["error_type"],
            "lexical_answer_recall": (
                r["lexical_answer_recall"]
            ),
            "gold_answers": json.dumps(
                r["gold_answers"],
                ensure_ascii=False,
            ),
            "matched_gold_answers": json.dumps(
                r["matched_gold_answers"],
                ensure_ascii=False,
            ),
            "question": r["question"],
            "pipeline_error": r["pipeline_error"],
        })

    save_csv(
        output / "error_analysis.csv",
        errors,
    )

    print("\n========== FINAL RESULTS ==========")

    print(json.dumps(
        metrics,
        ensure_ascii=False,
        indent=2,
    ))

    print("\n[SAVED]", output)


if __name__ == "__main__":
    asyncio.run(main())
