#!/usr/bin/env python3
"""
FinRL-X × Official LightRAG
Repeated Evidence-Grounded Triple-to-Natural-Language QA V3

Official components
-------------------
- Official LightRAG aquery()
- Official LightRAG only_need_context
- Optional official RAGAS Faithfulness

Custom components
-----------------
- Snapshot discovery / audit
- Triple -> natural language QA
- Single-answer and multi-answer evaluation
- Qwen answer extraction WITHOUT access to gold
- Exact Match / Precision / Recall / F1
- Conservative evidence verification
- Repeated sampling
- Checkpoint / resume
- Cross-snapshot metrics

The evaluated facts are KG-recorded facts.
They are NOT externally certified financial truths.
"""

import argparse
import asyncio
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg.evaluation import (
    run_official_kg_qa_v2 as base
)

SNAPSHOT_ROOT = (
    ROOT / "results/light_rag_dynamic_kg/snapshots"
)

OUTPUT_ROOT = (
    ROOT / "results/light_rag_dynamic_kg/repeated_qa_v3"
)

MODES = ("local", "global", "hybrid")


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(s):
    return re.sub(
        r"\s+", " ", str(s).casefold()
    ).strip()


def canonical(s):
    """
    Conservative normalization.
    Relation punctuation/underscores are retained.
    """
    return normalize(s).strip(" \t\n\r.,;:\"'`")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            data, ensure_ascii=False,
            indent=2, default=str
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row, ensure_ascii=False, default=str
            ) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not path.is_file():
        return []

    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_csv(path, rows):
    if not rows:
        return

    keys = list(dict.fromkeys(
        k for row in rows for k in row.keys()
    ))

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=keys
        )
        writer.writeheader()
        writer.writerows(rows)


def valid_snapshot(folder):
    marker = folder / "index_ready.json"
    manifest = folder / "source_manifest.json"

    if not marker.is_file() or not manifest.is_file():
        return False

    try:
        ready = json.loads(marker.read_text())
        sources = json.loads(manifest.read_text())

        if not isinstance(ready, dict):
            return False

        if not isinstance(sources, list):
            return False

        fp = folder.name.rsplit("_", 1)[-1]

        return ready.get("fingerprint") == fp

    except Exception:
        return False


def discover(args):
    selected = []

    for folder in sorted(SNAPSHOT_ROOT.glob("*")):
        if not folder.is_dir():
            continue

        match = re.fullmatch(
            r"([A-Za-z0-9]+)_"
            r"(\d{4}-\d{2}-\d{2})_"
            r"([0-9a-f]+)",
            folder.name,
        )

        if not match:
            continue

        ticker, day, fingerprint = match.groups()

        if args.ticker and ticker != args.ticker:
            continue

        if args.date and day != args.date:
            continue

        if args.fingerprint and (
            fingerprint != args.fingerprint
        ):
            continue

        if not valid_snapshot(folder):
            continue

        selected.append({
            "folder": folder,
            "ticker": ticker,
            "date": day,
            "fingerprint": fingerprint,
        })

    if args.max_snapshots > 0:
        selected = selected[:args.max_snapshots]

    return selected


def build_questions(rows, include_multi=False):
    """
    Build a complete answer set for each
    (head,relation), (head,tail), (relation,tail).

    Prevents treating one arbitrary gold triple
    as the only correct answer.
    """

    groups = {
        "HR_T": defaultdict(set),
        "HT_R": defaultdict(set),
        "RT_H": defaultdict(set),
    }

    evidence = {
        "HR_T": defaultdict(list),
        "HT_R": defaultdict(list),
        "RT_H": defaultdict(list),
    }

    seen = set()

    for r in rows:
        h = str(r["head"]).strip()
        rel = str(r["relation"]).strip()
        t = str(r["tail"]).strip()

        if not all((h, rel, t)) or h == t:
            continue

        triple_key = (h, rel, t)

        if triple_key in seen:
            continue

        seen.add(triple_key)

        item = {
            "head": h,
            "relation": rel,
            "tail": t,
            "event_id": str(r["event_id"]),
            "available_at": str(r["available_at"]),
        }

        configurations = [
            ("HR_T", (h, rel), t),
            ("HT_R", (h, t), rel),
            ("RT_H", (rel, t), h),
        ]

        for task, key, answer in configurations:
            groups[task][key].add(answer)
            evidence[task][key].append(item)

    questions = []

    for task in ("HR_T", "HT_R", "RT_H"):
        for key in sorted(groups[task]):

            answers = sorted(groups[task][key])

            if len(answers) > 1 and not include_multi:
                continue

            a, b = key
            plural = len(answers) > 1

            if task == "HR_T":
                question = (
                    f"According to the historical financial "
                    f"knowledge graph, "
                    f"{'list all entities' if plural else 'which entity'} "
                    f"has a directed '{b}' relationship "
                    f"from '{a}'? "
                    f"Only report graph-recorded relationships."
                )

            elif task == "HT_R":
                question = (
                    f"According to the historical financial "
                    f"knowledge graph, "
                    f"{'list all direct relationship labels' if plural else 'what is the direct relationship label'} "
                    f"from '{a}' to '{b}'? "
                    f"Only report graph-recorded relationships."
                )

            else:
                question = (
                    f"According to the historical financial "
                    f"knowledge graph, "
                    f"{'list all entities' if plural else 'which entity'} "
                    f"has a directed '{a}' relationship "
                    f"toward '{b}'? "
                    f"Only report graph-recorded relationships."
                )

            gold = evidence[task][key]

            question_id = hashlib.sha256(
                json.dumps(
                    [task, key, answers],
                    ensure_ascii=False
                ).encode()
            ).hexdigest()[:16]

            questions.append({
                "question_id": question_id,
                "task": task,
                "question": question,
                "gold_answers": answers,
                "gold_triples": gold,
                "gold_event_ids": sorted({
                    x["event_id"] for x in gold
                }),
                "is_multi_answer": plural,
                "gold_source": "historical_snapshot_kg",
                "external_fact_verified": False,
            })

    return questions


def sample_rounds(pool, rounds, n, seed):
    """
    Sample without repetition across rounds until
    the pool is exhausted. Each round is unique.
    """
    if len(pool) < n:
        raise ValueError(
            f"Only {len(pool)} eligible questions, need {n}"
        )

    rng = random.Random(seed)
    by_task = defaultdict(list)

    for q in pool:
        by_task[q["task"]].append(q)

    all_questions = list(pool)
    rng.shuffle(all_questions)

    remaining = list(all_questions)
    output = []

    for round_id in range(1, rounds + 1):
        if len(remaining) < n:
            # Restart shuffled pool only after exhaustion.
            remaining = list(all_questions)
            rng.shuffle(remaining)

        # Balanced selection where possible.
        chosen = []
        used = set()

        for task in ("HR_T", "HT_R", "RT_H"):
            target = n // 3

            eligible = [
                x for x in remaining
                if x["task"] == task
            ]

            for q in eligible[:target]:
                if len(chosen) >= n:
                    break

                if q["question_id"] not in used:
                    chosen.append(q)
                    used.add(q["question_id"])

        for q in remaining:
            if len(chosen) >= n:
                break

            if q["question_id"] not in used:
                chosen.append(q)
                used.add(q["question_id"])

        if len(chosen) != n:
            raise RuntimeError("Sampling failed")

        remaining = [
            q for q in remaining
            if q["question_id"] not in used
        ]

        for i, q in enumerate(chosen, 1):
            output.append({
                **q,
                "round": round_id,
                "sample_id": (
                    f"R{round_id:03d}_Q{i:03d}"
                ),
            })

    return output


def ollama_extract(question, answer):
    """
    Structured extraction WITHOUT GOLD ANSWERS.

    The extractor sees the original question and
    answer, not the KG reference answers.

    Its output is evaluated deterministically.
    """

    host = base.official.HOST.rstrip("/")
    model = base.official.LLM_MODEL

    prompt = f"""
You are a strict answer extraction system.

Extract ONLY the answers explicitly ASSERTED by the
candidate answer as responses to the question.

Rules:
- Do not answer the question yourself.
- Do not infer missing knowledge.
- Do not use external financial facts.
- If the candidate refuses or does not answer, return [].
- Do not count negated claims as positive answers.
- Do not count unrelated entity mentions.
- Preserve entity names and relation labels.
- Include every distinct asserted answer.
- Copy the exact answer span where possible.
- No reasoning text.

Question:
{question}

Candidate answer:
{answer[:12000]}

Return JSON only:
{{
  "answers": ["answer1", "answer2"]
}}
"""

    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
    }

    req = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        req, timeout=600
    ) as response:
        envelope = json.loads(
            response.read().decode()
        )

    raw = envelope.get(
        "message", {}
    ).get("content", "")

    obj = json.loads(raw)

    answers = obj.get("answers")

    if not isinstance(answers, list):
        raise ValueError("answers must be a JSON list")

    if any(not isinstance(x, str) for x in answers):
        raise ValueError("Every answer must be a string")

    return answers, raw


def score_answers(predicted, gold):
    """
    Conservative exact normalized set comparison.
    LLM extracts answers but never grades them.
    """

    p = {canonical(x) for x in predicted if canonical(x)}
    g = {canonical(x) for x in gold if canonical(x)}

    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "exact_match": (p == g),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extra_predictions": sorted(p - g),
        "missing_answers": sorted(g - p),
    }


def evidence_diagnosis(context, gold_triples):
    """
    Conservative positive evidence recognition.

    Count a hit only when a complete DIRECTED triple
    is recoverable from one structured record or one
    explicitly formatted directed-edge line.

    Unknown/unparseable LightRAG formats remain
    UNVERIFIED, not FALSE.

    This is NOT ranked Evidence Recall@K.
    """

    structured = set()

    for line in context.splitlines():
        s = line.strip().strip(",")

        if not s:
            continue

        if s.startswith("{") and s.endswith("}"):
            try:
                row = json.loads(s)
            except Exception:
                row = None

            if isinstance(row, dict):
                h = row.get("head")
                r = row.get("relation")
                t = row.get("tail")

                if all(isinstance(x, str) for x in (h, r, t)):
                    structured.add((
                        canonical(h),
                        canonical(r),
                        canonical(t),
                    ))

        match = re.search(
            r"(.+?)\s*--\[\s*(.+?)\s*\]-->\s*(.+)",
            s
        )

        if match:
            structured.add(tuple(
                canonical(x)
                for x in match.groups()
            ))

    matched = []

    for triple in gold_triples:
        key = (
            canonical(triple["head"]),
            canonical(triple["relation"]),
            canonical(triple["tail"]),
        )

        if key in structured:
            matched.append(triple["event_id"])

    return {
        "context_empty": not bool(context.strip()),
        "context_chars": len(context),
        "structured_gold_hit": bool(matched),
        "matched_gold_event_ids": sorted(set(matched)),
        "evidence_status": (
            "STRUCTURED_HIT"
            if matched
            else "UNVERIFIED"
        ),
        "ranked_evidence_recall_at_k": None,
    }


def init_ragas():
    """
    Official RAGAS Faithfulness.
    Does not implement a replacement scoring formula.
    """

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=base.official.HOST.rstrip("/") + "/v1",
        timeout=600,
        max_retries=0,
    )

    llm = llm_factory(
        base.official.LLM_MODEL,
        client=client,
    )

    return Faithfulness(llm=llm)


async def measure_faithfulness(metric, question, answer, context):
    if not answer.strip() or not context.strip():
        return None, "EMPTY_ANSWER_OR_CONTEXT"

    try:
        result = await metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=[context],
        )

        value = float(result.value)

        if not math.isfinite(value):
            raise ValueError("Non-finite faithfulness")

        if not 0 <= value <= 1:
            raise ValueError("Faithfulness out of range")

        return value, None

    except Exception as exc:
        return None, repr(exc)


def classify(row):
    if row.get("qa_error"):
        return "QA_PIPELINE_ERROR"

    if row.get("extract_error"):
        return "ANSWER_EXTRACTION_ERROR"

    if row.get("context_empty"):
        return "NO_CONTEXT"

    if row.get("exact_match") is True:
        if row.get("structured_gold_hit"):
            return "CORRECT_WITH_STRUCTURED_EVIDENCE"

        return "CORRECT_EVIDENCE_UNVERIFIED"

    if row.get("structured_gold_hit"):
        return "ANSWER_ERROR_WITH_RETRIEVED_EVIDENCE"

    return "ANSWER_ERROR_EVIDENCE_UNVERIFIED"


def summarize(records, modes, expected):
    latest = {}

    for r in records:
        latest[(r["sample_id"], r["mode"])] = r

    out = {
        "expected_evaluations": expected,
        "recorded_evaluations": len(latest),
        "by_mode": {},
        "methodology": {
            "qa": "Official LightRAG",
            "gold": "Known historical KG triples",
            "answer_scoring": "Extraction then set matching",
            "faithfulness": "Official RAGAS when enabled",
            "evidence": "Conservative structured hit",
            "ranked_recall_at_k": None,
            "strict_pit_certified": False,
            "external_factual_truth_certified": False,
        },
    }

    for mode in modes:
        group = [
            x for x in latest.values()
            if x["mode"] == mode
        ]

        valid = [
            x for x in group
            if x.get("qa_error") is None
            and x.get("extract_error") is None
        ]

        def mean(key, arr):
            values = [
                x[key] for x in arr
                if isinstance(x.get(key), (int, float))
                and not isinstance(x.get(key), bool)
            ]
            return sum(values) / len(values) if values else None

        by_task = {}

        for task in ("HR_T", "HT_R", "RT_H"):
            subset = [
                x for x in valid
                if x["task"] == task
            ]

            by_task[task] = {
                "count": len(subset),
                "exact_match_accuracy": (
                    sum(bool(x["exact_match"]) for x in subset)
                    / len(subset)
                    if subset else None
                ),
                "mean_f1": mean("f1", subset),
            }

        out["by_mode"][mode] = {
            "expected": expected // len(modes),
            "recorded": len(group),
            "validly_scored": len(valid),
            "coverage": (
                len(valid) / (expected // len(modes))
            ),
            "graded_exact_match_accuracy": (
                sum(bool(x["exact_match"]) for x in valid)
                / len(valid)
                if valid else None
            ),
            "correct_over_all_expected": (
                sum(bool(x["exact_match"]) for x in valid)
                / (expected // len(modes))
            ),
            "mean_precision": mean("precision", valid),
            "mean_recall": mean("recall", valid),
            "mean_f1": mean("f1", valid),
            "mean_ragas_faithfulness": mean(
                "ragas_faithfulness", valid
            ),
            "ragas_scored": sum(
                x.get("ragas_faithfulness") is not None
                for x in valid
            ),
            "structured_evidence_hits": sum(
                bool(x.get("structured_gold_hit"))
                for x in valid
            ),
            "evidence_unverified": sum(
                x.get("evidence_status") == "UNVERIFIED"
                for x in valid
            ),
            "by_task": by_task,
        }

    return out


async def process_snapshot(snapshot, args, modes):
    folder = snapshot["folder"]

    rows, marker, manifest = base.load_historical_snapshot(
        snapshot["ticker"],
        snapshot["date"],
        folder,
    )

    pool = build_questions(
        rows,
        include_multi=args.include_multi,
    )

    questions = sample_rounds(
        pool,
        args.rounds,
        args.n,
        args.seed,
    )

    config = {
        "snapshot": str(folder),
        "rounds": args.rounds,
        "n": args.n,
        "seed": args.seed,
        "modes": modes,
        "include_multi": args.include_multi,
        "use_ragas": not args.no_ragas,
        "model": base.official.LLM_MODEL,
        "version": "repeated_qa_v3",
    }

    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]

    out = OUTPUT_ROOT / f"{folder.name}_{digest}"
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "config.json"
    questions_path = out / "questions.json"
    results_path = out / "results.jsonl"

    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old != config:
            raise RuntimeError("Configuration mismatch")
    else:
        save_json(config_path, config)

    if questions_path.exists():
        old = json.loads(questions_path.read_text())
        if old != questions:
            raise RuntimeError("Question dataset mismatch")
    else:
        save_json(questions_path, questions)

    expected = len(questions) * len(modes)

    existing = read_jsonl(results_path)
    latest = {}

    for r in existing:
        latest[(r["sample_id"], r["mode"])] = r

    print("\n[SNAPSHOT]", folder.name, flush=True)
    print("[KG TRIPLES]", len(rows), flush=True)
    print("[QA POOL]", len(pool), flush=True)
    print("[QUESTIONS]", len(questions), flush=True)
    print("[EXPECTED]", expected, flush=True)
    print("[OUTPUT]", out, flush=True)

    if args.dry_run:
        return

    ragas_metric = None

    if not args.no_ragas:
        ragas_metric = init_ragas()

    rag = base.create_rag(folder)
    await rag.initialize_storages()

    try:
        for mode in modes:
            for q in questions:
                key = (q["sample_id"], mode)

                previous = latest.get(key)

                if (
                    previous is not None
                    and previous.get("qa_error") is None
                    and previous.get("extract_error") is None
                    and (
                        args.no_ragas
                        or previous.get("ragas_faithfulness") is not None
                    )
                ):
                    continue

                rec = dict(previous) if previous else {
                    **q,
                    "ticker": snapshot["ticker"],
                    "snapshot_date": snapshot["date"],
                    "snapshot": folder.name,
                    "mode": mode,
                    "answer": "",
                    "context": "",
                    "qa_error": None,
                    "extract_error": None,
                    "predicted_answers": None,
                    "extractor_raw": "",
                    "ragas_faithfulness": None,
                    "ragas_error": None,
                }

                print(
                    f"[QA] {folder.name} "
                    f"{mode} {q['sample_id']} {q['task']}",
                    flush=True,
                )

                if not rec.get("answer") or rec.get("qa_error"):
                    try:
                        context = await base.get_retrieved_context(
                            rag, q["question"], mode
                        )

                        answer = await base.run_official_qa(
                            rag, q["question"], mode
                        )

                        rec["context"] = context
                        rec["answer"] = answer
                        rec["qa_error"] = None

                        rec.update(evidence_diagnosis(
                            context, q["gold_triples"]
                        ))

                    except Exception as exc:
                        rec["qa_error"] = repr(exc)

                    # Save raw answer BEFORE extraction/metrics.
                    append_jsonl(results_path, rec)
                    latest[key] = dict(rec)

                if rec.get("qa_error"):
                    continue

                if rec.get("predicted_answers") is None:
                    try:
                        preds, raw = await asyncio.to_thread(
                            ollama_extract,
                            q["question"],
                            rec["answer"],
                        )

                        rec["predicted_answers"] = preds
                        rec["extractor_raw"] = raw
                        rec["extract_error"] = None

                        rec.update(score_answers(
                            preds, q["gold_answers"]
                        ))

                    except Exception as exc:
                        rec["extract_error"] = repr(exc)

                    append_jsonl(results_path, rec)
                    latest[key] = dict(rec)

                if (
                    ragas_metric is not None
                    and rec.get("ragas_faithfulness") is None
                ):
                    score, error = await measure_faithfulness(
                        ragas_metric,
                        q["question"],
                        rec["answer"],
                        rec["context"],
                    )

                    rec["ragas_faithfulness"] = score
                    rec["ragas_error"] = error

                rec["error_type"] = classify(rec)
                rec["updated_at"] = now()

                append_jsonl(results_path, rec)
                latest[key] = dict(rec)

                print(
                    "[RESULT]",
                    rec["error_type"],
                    "EM=", rec.get("exact_match"),
                    "F1=", rec.get("f1"),
                    "Faithfulness=", rec.get("ragas_faithfulness"),
                    flush=True,
                )

    finally:
        await rag.finalize_storages()

    final_rows = list(latest.values())

    report = summarize(
        final_rows,
        modes,
        expected,
    )

    report["snapshot"] = folder.name
    report["eligible_question_pool"] = len(pool)
    report["unique_questions"] = len({
        q["question_id"] for q in questions
    })
    report["completed_at"] = now()

    save_json(out / "metrics.json", report)

    # Save concise human review table.
    review = []

    for r in final_rows:
        review.append({
            "sample_id": r["sample_id"],
            "round": r["round"],
            "mode": r["mode"],
            "task": r["task"],
            "question": r["question"],
            "gold_answers": json.dumps(
                r["gold_answers"], ensure_ascii=False
            ),
            "predicted_answers": json.dumps(
                r.get("predicted_answers"),
                ensure_ascii=False
            ),
            "answer": r.get("answer", ""),
            "exact_match": r.get("exact_match"),
            "f1": r.get("f1"),
            "ragas_faithfulness": r.get("ragas_faithfulness"),
            "evidence_status": r.get("evidence_status"),
            "error_type": r.get("error_type"),
            "human_correct": "",
            "human_evidence_supported": "",
            "human_notes": "",
        })

    write_csv(out / "human_review.csv", review)

    errors = [
        r for r in review
        if r.get("exact_match") is not True
    ]

    write_csv(out / "error_analysis.csv", errors)

    print("[SAVED]", out, flush=True)
    print(json.dumps(
        report["by_mode"],
        indent=2,
        ensure_ascii=False,
    ), flush=True)


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ticker", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--fingerprint", default=None)

    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--modes",
        default="hybrid",
    )

    ap.add_argument(
        "--include-multi",
        action="store_true",
    )

    ap.add_argument(
        "--no-ragas",
        action="store_true",
    )

    ap.add_argument(
        "--max-snapshots",
        type=int,
        default=1,
        help="0 = all completed snapshots",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    modes = [
        m.strip()
        for m in args.modes.split(",")
        if m.strip()
    ]

    if (
        args.rounds < 1
        or args.n < 1
        or not modes
        or len(set(modes)) != len(modes)
        or any(m not in MODES for m in modes)
        or args.max_snapshots < 0
    ):
        ap.error("Invalid settings")

    snapshots = discover(args)

    print("[READY SNAPSHOTS]", len(snapshots), flush=True)

    if not snapshots:
        raise RuntimeError("No eligible snapshots found")

    print("[TARGET SNAPSHOTS]")
    for s in snapshots:
        print(" ", s["folder"].name)

    lock_path = OUTPUT_ROOT / "evaluation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another V3 QA evaluator is already running"
            )

        failures = []

        for i, snapshot in enumerate(snapshots, 1):
            print(
                f"\n[SNAPSHOT {i}/{len(snapshots)}]",
                flush=True,
            )

            try:
                await process_snapshot(
                    snapshot, args, modes
                )
            except Exception as exc:
                print(
                    "[SNAPSHOT ERROR]",
                    snapshot["folder"].name,
                    repr(exc),
                    flush=True,
                )
                failures.append({
                    "snapshot": snapshot["folder"].name,
                    "error": repr(exc),
                    "time": now(),
                })

        if failures:
            save_json(
                OUTPUT_ROOT / "last_failures.json",
                failures
            )
            raise SystemExit(
                f"{len(failures)} snapshot(s) failed."
            )

        print("\n[ALL SELECTED SNAPSHOTS COMPLETE]")


if __name__ == "__main__":
    asyncio.run(main())
