#!/usr/bin/env python3
"""
FinRL-X × Official LightRAG × FinanceBench-style × RAGAS

Official components:
    LightRAG.aquery()
    LightRAG QueryParam
    RAGAS Faithfulness
    RAGAS FactualCorrectness

Custom components:
    Historical KG -> FinanceBench-style QA dataset
    Repeated sampling
    Gold evidence provenance
    Checkpoint/resume
    ARES-compatible evaluation data export
    Aggregate metrics

Important:
    FinanceBench-style != official FinanceBench dataset.
    KG-derived references != externally verified truth.
    ARES export != ARES classifier/PPI evaluation.
    RAGAS FactualCorrectness != strict binary QA accuracy.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg.evaluation import (
    run_official_kg_qa_v2 as base
)

OUTPUT_ROOT = (
    ROOT / "results/light_rag_dynamic_kg/"
    "integrated_financial_qa"
)

MODES = ("local", "global", "hybrid")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    os.replace(tmp, path)


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_financebench_style_candidates(rows, ticker, date):
    """
    Build auditable QA from existing KG triples.

    Gold evidence is an exact source triple, with
    source event ID and availability timestamp.

    To avoid ambiguous questions, only use mapping keys
    whose unique answer count equals one.

    This benchmark measures reproduction of KG facts,
    not independently verified market facts.
    """

    hr = defaultdict(set)
    ht = defaultdict(set)
    rt = defaultdict(set)

    unique = {}

    for r in rows:
        h = str(r["head"]).strip()
        rel = str(r["relation"]).strip()
        t = str(r["tail"]).strip()

        if not h or not rel or not t or h == t:
            continue

        hr[(h, rel)].add(t)
        ht[(h, t)].add(rel)
        rt[(rel, t)].add(h)

        unique.setdefault((h, rel, t), r)

    candidates = []

    for (h, rel, t), source in sorted(unique.items()):

        evidence = {
            "evidence_text": (
                f"{h} --[{rel}]--> {t}"
            ),
            "evidence_doc_name": (
                str(source.get("source", "historical_kg"))
            ),
            "evidence_page_num": None,
            "evidence_text_full_page": None,
            "event_id": str(source["event_id"]),
            "available_at": str(source["available_at"]),
            "head_id": str(source["head_id"]),
            "relation_id": str(source["relation_id"]),
            "tail_id": str(source["tail_id"]),
        }

        templates = []

        if len(hr[(h, rel)]) == 1:
            templates.append((
                "HR_T",
                (
                    f"According to the historical financial "
                    f"knowledge graph, which entity is recorded "
                    f"as the destination of the '{rel}' "
                    f"relationship from '{h}'?"
                ),
                t,
                (
                    f"The historical graph records that "
                    f"{h} has a directed {rel} relationship "
                    f"to {t}."
                )
            ))

        if len(ht[(h, t)]) == 1:
            templates.append((
                "HT_R",
                (
                    f"According to the historical financial "
                    f"knowledge graph, what is the exact "
                    f"recorded relationship from '{h}' "
                    f"to '{t}'?"
                ),
                rel,
                (
                    f"The directed relationship from "
                    f"{h} to {t} is {rel}."
                )
            ))

        if len(rt[(rel, t)]) == 1:
            templates.append((
                "RT_H",
                (
                    f"According to the historical financial "
                    f"knowledge graph, which entity has "
                    f"a directed '{rel}' relationship "
                    f"toward '{t}'?"
                ),
                h,
                (
                    f"The historical graph records that "
                    f"{h} has a directed {rel} relationship "
                    f"toward {t}."
                )
            ))

        for task, question, gold, reference in templates:

            candidates.append({
                "question": question,
                "answer": reference,
                "gold_answer": gold,
                "evidence": [evidence],
                "justification": (
                    "Reference derived from the selected "
                    "historical KG triple, not independently "
                    "verified against an original filing."
                ),
                "question_type": "kg-derived-natural-language",
                "question_reasoning": "single-hop-graph-fact",
                "task": task,
                "company": ticker,
                "snapshot_date": date,
                "gold_source": "historical_kg",
                "independently_verified": False,
            })

    return candidates


def build_dataset(candidates, rounds, n, seed):
    """
    Repeated sampling across rounds.
    Each round has distinct questions.

    Repeats across rounds are possible if the candidate
    pool is exhausted; these are not independent samples.
    """
    if len(candidates) < n:
        raise RuntimeError(
            f"Only {len(candidates)} eligible single-answer "
            f"questions exist; requested {n}."
        )

    result = []

    for round_id in range(1, rounds + 1):
        rng = random.Random(seed + round_id)

        buckets = defaultdict(list)
        for item in candidates:
            buckets[item["task"]].append(item)

        for items in buckets.values():
            rng.shuffle(items)

        selected = []
        used = set()

        while len(selected) < n:
            progress = False

            for task in ("HR_T", "HT_R", "RT_H"):
                while buckets[task]:
                    item = buckets[task].pop()

                    if item["question"] in used:
                        continue

                    selected.append(item)
                    used.add(item["question"])
                    progress = True
                    break

                if len(selected) >= n:
                    break

            if not progress:
                raise RuntimeError(
                    "Insufficient distinct questions after "
                    "deduplication."
                )

        for i, item in enumerate(selected, 1):
            result.append({
                **item,
                "financebench_id": (
                    f"KG_{round_id:03d}_{i:03d}"
                ),
                "round": round_id,
            })

    return result


def setup_ragas(host, model):
    """
    Real RAGAS library integration.

    Uses Ollama OpenAI-compatible /v1 API.
    The evaluator must support structured outputs.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        Faithfulness,
        FactualCorrectness,
    )

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=host.rstrip("/") + "/v1",
        timeout=600,
        max_retries=0,
    )

    evaluator_llm = llm_factory(
        model,
        client=client,
    )

    return (
        Faithfulness(llm=evaluator_llm),
        FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
        ),
    )


async def ragas_score(scorer, **kwargs):
    try:
        result = await scorer.ascore(**kwargs)
        value = float(result.value)

        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Score outside [0,1]: {value}"
            )

        return value, None

    except Exception as exc:
        return None, repr(exc)


def export_ares(rows, output):
    """
    ARES dataset preparation only.

    ARES classifier training and PPI must be run later
    with human labels and official ares-ai.

    The data columns below expose Query/Document/Answer
    and empty human annotation fields.
    """

    path = output / "ares_unlabeled.tsv"
    review = output / "ares_human_review.tsv"

    columns = [
        "Query",
        "Document",
        "Answer",
        "Context_Relevance_Label",
        "Answer_Faithfulness_Label",
        "Answer_Relevance_Label",
    ]

    unique = {}
    for r in rows:
        key = (r["id"], r["mode"])

        if key not in unique or (
            not r.get("pipeline_error")
            and unique[key].get("pipeline_error")
        ):
            unique[key] = r

    valid = [
        r for r in unique.values()
        if not r.get("pipeline_error")
    ]

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter="\t"
        )
        writer.writeheader()

        for r in valid:
            writer.writerow({
                "Query": r["question"],
                "Document": r["retrieved_context"],
                "Answer": r["official_answer"],
                "Context_Relevance_Label": "",
                "Answer_Faithfulness_Label": "",
                "Answer_Relevance_Label": "",
            })

    # Human review file; never auto-fill labels with
    # the same model's RAGAS outputs.
    with review.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "mode", *columns],
            delimiter="\t"
        )
        writer.writeheader()

        for r in valid:
            writer.writerow({
                "id": r["id"],
                "mode": r["mode"],
                "Query": r["question"],
                "Document": r["retrieved_context"],
                "Answer": r["official_answer"],
                "Context_Relevance_Label": "",
                "Answer_Faithfulness_Label": "",
                "Answer_Relevance_Label": "",
            })

    return path, review


def summarize(rows, modes, expected):
    unique = {}

    for r in rows:
        key = (r["id"], r["mode"])
        prior = unique.get(key)

        if prior is None:
            unique[key] = r
        elif not r.get("pipeline_error"):
            unique[key] = r

    summary = {
        "expected_evaluations": expected,
        "recorded_unique_evaluations": len(unique),
        "by_mode": {},
        "notes": {
            "qa_engine": "Official LightRAG aquery",
            "dataset": "FinanceBench-style KG-derived",
            "financebench_official_test": False,
            "ragas": "Official RAGAS library",
            "ares": "Export only, not evaluated",
            "strict_pit_certified": False,
            "factual_correctness_is_not_binary_accuracy": True,
            "external_gold_verification": False,
        },
    }

    for mode in modes:
        subset = [
            r for r in unique.values()
            if r["mode"] == mode
        ]

        def mean(key):
            values = [
                r[key] for r in subset
                if isinstance(r.get(key), (int, float))
                and not isinstance(r.get(key), bool)
            ]

            return (
                sum(values) / len(values)
                if values else None
            )

        summary["by_mode"][mode] = {
            "attempted": len(subset),
            "pipeline_success": sum(
                not r.get("pipeline_error")
                for r in subset
            ),
            "ragas_faithfulness_scored": sum(
                r.get("ragas_faithfulness") is not None
                for r in subset
            ),
            "ragas_factual_scored": sum(
                r.get("ragas_factual_correctness") is not None
                for r in subset
            ),
            "mean_ragas_faithfulness": mean(
                "ragas_faithfulness"
            ),
            "mean_ragas_factual_correctness": mean(
                "ragas_factual_correctness"
            ),
            "pipeline_errors": sum(
                bool(r.get("pipeline_error"))
                for r in subset
            ),
        }

    return summary


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--date", default="2018-08-06")
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", default="local,global,hybrid"
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run native QA and save datasets, no RAGAS LLM calls"
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    modes = [
        m.strip()
        for m in args.modes.split(",")
        if m.strip()
    ]

    if (
        args.rounds < 1
        or args.n < 1
        or not modes
        or len(modes) != len(set(modes))
        or any(m not in MODES for m in modes)
    ):
        parser.error("Invalid rounds, n or modes")

    workspace = base.find_snapshot(
        args.ticker,
        args.date,
        args.fingerprint,
    )

    kg_rows, marker, manifest = (
        base.load_historical_snapshot(
            args.ticker, args.date, workspace
        )
    )

    candidates = make_financebench_style_candidates(
        kg_rows, args.ticker, args.date
    )

    config = {
        "snapshot": str(workspace),
        "rounds": args.rounds,
        "n": args.n,
        "seed": args.seed,
        "modes": modes,
        "ragas": not args.skip_ragas,
        "model": base.official.LLM_MODEL,
        "version": "integrated_v1",
    }

    digest = hashlib.sha256(
        json.dumps(
            config, sort_keys=True
        ).encode()
    ).hexdigest()[:16]

    output = OUTPUT_ROOT / (
        workspace.name + "_" + digest
    )
    output.mkdir(parents=True, exist_ok=True)

    dataset_path = output / "financebench_style.jsonl"
    result_path = output / "qa_results.jsonl"

    if dataset_path.exists():
        dataset = load_jsonl(dataset_path)
    else:
        dataset = build_dataset(
            candidates,
            args.rounds,
            args.n,
            args.seed,
        )

        for item in dataset:
            append_jsonl(dataset_path, item)

    if (output / "config.json").exists():
        previous_cfg = json.loads(
            (output / "config.json").read_text()
        )
        if previous_cfg != config:
            raise RuntimeError("Configuration mismatch")
    else:
        save_json(output / "config.json", config)

    print("[SNAPSHOT]", workspace, flush=True)
    print("[ELIGIBLE]", len(candidates), flush=True)
    print("[QUESTIONS]", len(dataset), flush=True)
    print("[MODES]", modes, flush=True)
    print("[EXPECTED]", len(dataset) * len(modes), flush=True)
    print("[OUTPUT]", output, flush=True)

    if args.dry_run:
        print("[DRY RUN COMPLETE]", flush=True)
        return

    faithfulness = correctness = None

    if not args.skip_ragas:
        faithfulness, correctness = setup_ragas(
            base.official.HOST,
            base.official.LLM_MODEL,
        )

    previous = load_jsonl(result_path)

    # Incomplete RAGAS records are not discarded.
    # They can be retried without calling aquery again.
    latest = {}
    for row in previous:
        latest[(row["id"], row["mode"])] = row

    rag = base.create_rag(workspace)
    await rag.initialize_storages()

    try:
        for mode in modes:
            for q in dataset:

                key = (
                    q["financebench_id"],
                    mode,
                )

                previous_row = latest.get(key)

                if (
                    previous_row
                    and not previous_row.get("pipeline_error")
                ):
                    record = dict(previous_row)
                else:
                    record = {
                        "id": q["financebench_id"],
                        "round": q["round"],
                        "mode": mode,
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "gold_evidence": q["evidence"],
                        "official_answer": "",
                        "retrieved_context": "",
                        "pipeline_error": None,
                        "ragas_faithfulness": None,
                        "ragas_factual_correctness": None,
                        "ragas_faithfulness_error": None,
                        "ragas_factual_error": None,
                    }

                    print(
                        "[LIGHTRAG]",
                        mode, q["financebench_id"],
                        flush=True
                    )

                    try:
                        # Official LightRAG native retrieval
                        record["retrieved_context"] = (
                            await base.get_retrieved_context(
                                rag,
                                q["question"],
                                mode
                            )
                        )

                        # Official LightRAG native QA
                        record["official_answer"] = (
                            await base.run_official_qa(
                                rag,
                                q["question"],
                                mode
                            )
                        )

                    except Exception as exc:
                        record["pipeline_error"] = repr(exc)

                    append_jsonl(result_path, record)

                if record["pipeline_error"]:
                    latest[key] = record
                    continue

                if not args.skip_ragas:
                    if record["ragas_faithfulness"] is None:
                        print(
                            "[RAGAS FAITHFULNESS]",
                            mode, q["financebench_id"],
                            flush=True
                        )

                        context = record["retrieved_context"]

                        if context.strip() and (
                            record["official_answer"].strip()
                        ):
                            score, error = await ragas_score(
                                faithfulness,
                                user_input=q["question"],
                                response=record["official_answer"],
                                retrieved_contexts=[context],
                            )
                        else:
                            score = None
                            error = "Empty context or answer"

                        record["ragas_faithfulness"] = score
                        record["ragas_faithfulness_error"] = error

                    if record["ragas_factual_correctness"] is None:
                        print(
                            "[RAGAS FACTUAL]",
                            mode, q["financebench_id"],
                            flush=True
                        )

                        score, error = await ragas_score(
                            correctness,
                            response=record["official_answer"],
                            reference=q["answer"],
                        )

                        record["ragas_factual_correctness"] = score
                        record["ragas_factual_error"] = error

                    # Append updated evaluation checkpoint.
                    append_jsonl(result_path, record)

                latest[key] = record

                print(
                    "[RESULT]",
                    mode,
                    q["financebench_id"],
                    "faithfulness=",
                    record["ragas_faithfulness"],
                    "factual=",
                    record["ragas_factual_correctness"],
                    flush=True
                )

    finally:
        await rag.finalize_storages()

    final_records = list(latest.values())

    report = summarize(
        final_records,
        modes,
        len(dataset) * len(modes),
    )

    save_json(output / "metrics.json", report)

    ares_path, review_path = export_ares(
        final_records, output
    )

    print("\n========== FINAL METRICS ==========")
    print(json.dumps(
        report, ensure_ascii=False, indent=2
    ))

    print("\n[SAVED]", output)
    print("[FINANCEBENCH-STYLE]", dataset_path)
    print("[QA RESULTS]", result_path)
    print("[RAGAS METRICS]", output / "metrics.json")
    print("[ARES UNLABELED]", ares_path)
    print("[ARES REVIEW]", review_path)


if __name__ == "__main__":
    asyncio.run(main())
