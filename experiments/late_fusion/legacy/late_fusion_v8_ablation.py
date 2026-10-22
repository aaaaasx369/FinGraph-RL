#!/usr/bin/env python3
"""V8 Evidence Reliance and Source Ablation.

Uses an existing V6 fused evidence JSON.
Does NOT run indexing, V4, V5, V6, or V7.
Six sequential Ollama calls; same prompt and generation settings.
Python standard library only.
"""
import argparse
import csv
import json
import math
import re
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

CONDITIONS = {
    "SEC-only": ("SEC",),
    "FinDKG-only": ("FinDKG",),
    "ReflectKG-only": ("ReflectKG",),
    "SEC+FinDKG": ("SEC", "FinDKG"),
    "SEC+ReflectKG": ("SEC", "ReflectKG"),
    "Full": ("SEC", "FinDKG", "ReflectKG"),
}

SCORES = (
    "impact", "risk", "confidence",
    "graph_change", "event_intensity"
)

SYSTEM = """
You are a historical financial evidence analyst.

Analyze the provided evidence for the requested company and date.
Only use information in EVIDENCE. Never use future information.
Do not invent claims, relations, sources, or citations.

Important:
- Generic SEC risk disclosures do NOT establish actual fraud,
  accounting deficiencies or confirmed adverse events.
- Competition between companies is not proof of losses.
- FIRST_OBSERVED and REASSERTED describe dataset observations,
  not real-world formation or changes in a relationship.
- Do not confuse AMD (Advanced Micro Devices) with
  AVGO (Broadcom).
- Do not claim that missing evidence means no risk exists.
- Missing evidence should reduce certainty, not create facts.
- Do not artificially cite every source. Cite only evidence
  that directly supports the specific reasoning.
- Do not mention or guess which experiment condition you are in.

Return exactly one JSON object:
{
  "impact": number,
  "risk": number,
  "confidence": number,
  "graph_change": number,
  "event_intensity": number,
  "citations": ["evidence_id"],
  "reasoning": "short explanation"
}

Each number must be in [-1,1].
impact: negative is adverse, positive is favorable.
risk: higher indicates more uncertainty/risk.
confidence: self-rated evidence adequacy, NOT calibrated accuracy.
graph_change: importance of observed relationship updates,
not actual economic relationship changes.
event_intensity: importance of recent evidence.
If evidence is insufficient, state that explicitly and
use conservative, low-confidence values.
""".strip()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path, rows, columns):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def before(value, as_of):
    try:
        return date.fromisoformat(str(value)[:10]) < date.fromisoformat(as_of)
    except (ValueError, TypeError):
        return False


def score(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric score")
    x = float(value)
    if not math.isfinite(x) or not -1 <= x <= 1:
        raise ValueError("Score outside [-1,1]")
    return x


def collect_evidence(data, max_sec, max_fin, max_ref, max_events):
    as_of = data["as_of"]
    pools = {"SEC": [], "FinDKG": [], "ReflectKG": []}
    seen = set()

    def add(source, item, body):
        eid = item.get("evidence_id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("Evidence item has no ID")
        if eid in seen:
            raise ValueError("Duplicate evidence ID: " + eid)
        seen.add(eid)
        pools[source].append({
            "id": eid,
            "source": source,
            "content": body
        })

    for e in data.get("sec_evidence", [])[:max_sec]:
        if not before(e.get("published_at"), as_of):
            raise ValueError("Unsafe SEC date: " + str(e.get("evidence_id")))
        add("SEC", e, {
            "doc_id": e.get("doc_id"),
            "published_at": e["published_at"],
            "text": str(e.get("text", ""))[:1600],
        })

    for e in data.get("findkg_evidence", [])[:max_fin]:
        if not before(e.get("observed_at"), as_of):
            raise ValueError("Unsafe FinDKG observation date")
        add("FinDKG", e, {
            "observed_at": e["observed_at"],
            "record": e.get("record", {}),
            "note": "Dataset observation, not relation creation",
        })

    for e in data.get("reflectkg_snapshot", [])[:max_ref]:
        if not before(e.get("last_observed_at"), as_of):
            raise ValueError("Unsafe ReflectKG snapshot date")
        add("ReflectKG", e, {
            "subject": e.get("subject"),
            "relation": e.get("relation"),
            "object": e.get("object"),
            "first_observed_at": e.get("first_observed_at"),
            "last_observed_at": e.get("last_observed_at"),
            "latest_evidence": str(
                e.get("latest_evidence", "")
            )[:700],
        })

    for e in data.get("reflectkg_changes", [])[:max_events]:
        if not before(e.get("published_at"), as_of):
            raise ValueError("Unsafe ReflectKG event date")
        add("ReflectKG", e, {
            "subject": e.get("subject"),
            "relation": e.get("relation"),
            "object": e.get("object"),
            "event_type": e.get("event_type"),
            "published_at": e.get("published_at"),
            "evidence": str(e.get("evidence", ""))[:400],
        })

    return pools


def make_prompt(data, selected):
    # No condition name is passed to Qwen.
    # Order is fixed SEC -> FinDKG -> ReflectKG.
    return json.dumps({
        "ticker": data["ticker"],
        "as_of": data["as_of"],
        "task": "Assess evidence-based financial outlook and risks.",
        "EVIDENCE": selected,
    }, ensure_ascii=False, sort_keys=True)


def ollama_generate(url, model, prompt, seed, ctx, timeout):
    payload = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": ctx,
        },
    }

    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(
        request, timeout=timeout
    ) as response:
        result = json.load(response)

    answer = json.loads(result["response"])
    return answer, result


def validate_answer(answer, allowed_ids):
    values = {k: score(answer[k]) for k in SCORES}

    citations = answer.get("citations")
    if not isinstance(citations, list):
        raise ValueError("citations must be a list")
    if not all(isinstance(c, str) for c in citations):
        raise ValueError("citation IDs must be strings")

    unknown = set(citations) - set(allowed_ids)
    if unknown:
        raise ValueError("Unknown citations: " + str(unknown))

    reasoning = answer.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string")

    return {
        **values,
        "citations": list(dict.fromkeys(citations)),
        "reasoning": reasoning,
    }


def source_audit(answer, selected):
    lookup = {e["id"]: e for e in selected}
    citations = answer["citations"]
    counts = Counter(lookup[c]["source"] for c in citations)
    total = len(citations)

    result = {}
    for source in ("SEC", "FinDKG", "ReflectKG"):
        available = sum(e["source"] == source for e in selected)
        cited = counts[source]
        result[source] = {
            "available": available,
            "cited": cited,
            "citation_share": round(cited / total, 4) if total else 0,
            "coverage_ratio": round(cited / available, 4)
            if available else None,
        }

    # Citation presence is a coarse usage proxy. It is not
    # a causal explanation of the model's internal reliance.
    return result


def grounding_flags(answer, selected):
    reasoning = answer.get("reasoning", "")
    flags = []

    mismatches = [
        r"Advanced Micro Devices\s*\(AVGO\)",
        r"Broadcom\s*\(AMD\)",
    ]
    if any(re.search(p, reasoning, re.I) for p in mismatches):
        flags.append("ENTITY_MISMATCH")

    if re.search(
        r"(?:underlying (?:financial|operational|financial or operational) issues)"
        r"|(?:confirmed|actual) fraud"
        r"|(?:fraud (?:occurred|was detected))",
        reasoning, re.I,
    ):
        flags.append("ACTUAL_PROBLEM_CLAIM_REVIEW")

    if (
        abs(answer["impact"]) >= 0.9
        and answer["confidence"] >= 0.9
    ):
        flags.append("EXTREME_SCORE_REVIEW")

    if not answer["citations"]:
        flags.append("NO_CITATIONS")

    # Heuristic: whether cited items contain ANY meaningful
    # non-stopword text overlap with the reasoning.
    # This is NOT entailment or factual-grounding verification.
    lookup = {x["id"]: x for x in selected}
    cited_text = " ".join(
        json.dumps(lookup[c]["content"], ensure_ascii=False)
        for c in answer["citations"]
    ).lower()

    stop = {
        "the", "and", "with", "from", "that", "this",
        "risk", "risks", "company", "companies",
        "financial", "evidence", "market", "could",
        "their", "there", "about", "which", "such",
    }
    terms = {
        w.lower() for w in re.findall(r"[A-Za-z]{5,}", reasoning)
        if w.lower() not in stop
    }
    overlap = sorted(w for w in terms if w in cited_text)
    if answer["citations"] and not overlap:
        flags.append("CITED_TEXT_LOW_LEXICAL_OVERLAP")

    return {
        "flags": flags,
        "cited_text_overlap_terms": overlap[:20],
        "grounding_status": "HEURISTIC_ONLY_NEEDS_HUMAN_REVIEW",
    }


def run_experiment(args):
    original = load(args.input)
    if original.get("schema_version") != "late_fusion_v6":
        raise ValueError("Expected late_fusion_v6 JSON")

    pools = collect_evidence(
        original, args.max_sec, args.max_fin,
        args.max_ref, args.max_events,
    )

    summary = {
        "schema": "source_ablation_v8",
        "ticker": original["ticker"],
        "as_of": original["as_of"],
        "model": args.model,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "temperature": 0,
        "input_v6": args.input,
        "source_counts": {
            k: len(v) for k, v in pools.items()
        },
        "conditions": {},
        "limitations": [
            "Citation frequency is not causal model reliance.",
            "Lexical overlap is not evidence entailment.",
            "Scores are uncalibrated model outputs.",
            "Changing evidence sources also changes prompt length.",
            "LightRAG Pilot SEC coverage is intentionally limited.",
            "Evidence availability is fixed to the input V6 file.",
        ],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "experiment_meta.json", summary)

    for name, enabled in CONDITIONS.items():
        # Keeping source order stable makes comparisons cleaner.
        selected = [
            item for source in ("SEC", "FinDKG", "ReflectKG")
            if source in enabled
            for item in pools[source]
        ]

        condition_path = out / (
            name.replace("+", "_").replace("-", "_") + ".json"
        )

        if not selected:
            record = {
                "condition": name,
                "status": "SKIPPED_EMPTY_EVIDENCE",
                "enabled_sources": list(enabled),
                "evidence_count": 0,
            }
            summary["conditions"][name] = record
            save(condition_path, record)
            save(out / "experiment_meta.json", summary)
            continue

        prompt = make_prompt(original, selected)
        start = time.monotonic()

        try:
            response, raw = ollama_generate(
                args.ollama_url, args.model, prompt,
                args.seed, args.num_ctx, args.timeout,
            )
            answer = validate_answer(
                response, [x["id"] for x in selected]
            )
            record = {
                "condition": name,
                "status": "OK",
                "enabled_sources": list(enabled),
                "evidence_count": len(selected),
                "prompt_chars": len(prompt),
                "financial_features": answer,
                "citation_audit": source_audit(
                    answer, selected
                ),
                "grounding_screen": grounding_flags(
                    answer, selected
                ),
                "duration_seconds": round(
                    time.monotonic() - start, 2
                ),
                "ollama_metrics": {
                    "prompt_eval_count": raw.get(
                        "prompt_eval_count"
                    ),
                    "eval_count": raw.get("eval_count"),
                },
            }
        except Exception as exc:
            record = {
                "condition": name,
                "status": "ERROR",
                "enabled_sources": list(enabled),
                "evidence_count": len(selected),
                "error": f"{type(exc).__name__}: {exc}",
            }

        save(condition_path, record)
        summary["conditions"][name] = record
        save(out / "experiment_meta.json", summary)

        print(
            name, record["status"],
            "evidence:", len(selected),
            flush=True,
        )

    full = summary["conditions"].get("Full", {})
    full_features = (
        full.get("financial_features")
        if full.get("status") == "OK"
        else None
    )

    rows = []
    citation_rows = []

    for name, record in summary["conditions"].items():
        features = record.get("financial_features") or {}
        row = {
            "condition": name,
            "status": record["status"],
            "evidence_count": record.get("evidence_count", 0),
            **{k: features.get(k, "") for k in SCORES},
            "citation_count": len(features.get("citations", [])),
            "flags": "|".join(
                record.get("grounding_screen", {}).get(
                    "flags", []
                )
            ),
        }

        for k in SCORES:
            row[f"delta_full_minus_{k}"] = (
                round(full_features[k] - features[k], 6)
                if full_features and k in features else ""
            )
        rows.append(row)

        for source, counts in record.get(
            "citation_audit", {}
        ).items():
            citation_rows.append({
                "condition": name,
                "source": source,
                **counts,
            })

    write_csv(
        out / "ablation_scores.csv", rows,
        ["condition", "status", "evidence_count", *SCORES,
         "citation_count", "flags",
         *[f"delta_full_minus_{k}" for k in SCORES]],
    )
    write_csv(
        out / "citation_reliance.csv", citation_rows,
        ["condition", "source", "available", "cited",
         "citation_share", "coverage_ratio"],
    )

    print("\n=== ABLATION SCORES ===")
    for row in rows:
        print(
            f"{row['condition']:17s}",
            row["status"],
            "impact=", row["impact"],
            "risk=", row["risk"],
            "confidence=", row["confidence"],
        )

    print("\nSaved to:", out)


def self_test():
    base = {
        "ticker": "NVDA",
        "as_of": "2021-06-01",
        "sec_evidence": [{
            "evidence_id": "SEC_001",
            "published_at": "2021-05-26",
            "text": "Supply chain disruption is a possible risk."
        }],
        "findkg_evidence": [{
            "evidence_id": "FIN_001",
            "observed_at": "2021-05-30",
            "record": {"relation": "example"}
        }],
        "reflectkg_snapshot": [],
        "reflectkg_changes": [],
    }
    pools = collect_evidence(base, 5, 5, 5, 5)
    assert len(pools["SEC"]) == 1
    assert len(pools["FinDKG"]) == 1

    answer = validate_answer({
        "impact": -0.2,
        "risk": 0.4,
        "confidence": 0.3,
        "graph_change": 0.0,
        "event_intensity": 0.2,
        "citations": ["SEC_001"],
        "reasoning": "Potential supply chain disruption."
    }, ["SEC_001", "FIN_001"])
    audit = source_audit(
        answer, pools["SEC"] + pools["FinDKG"]
    )
    assert audit["SEC"]["cited"] == 1
    assert audit["FinDKG"]["cited"] == 0

    try:
        validate_answer(
            {**answer, "citations": ["FAKE_001"]},
            ["SEC_001"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Fake citation was not rejected")

    future = json.loads(json.dumps(base))
    future["sec_evidence"][0]["published_at"] = "2022-01-01"
    try:
        collect_evidence(future, 5, 5, 5, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("Future SEC evidence was not rejected")

    print("SELF-TEST PASSED: source selection, citations, date")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="outputs/v6/nvda_20210601_qwen.json"
    )
    p.add_argument("--out-dir", default="outputs/v8/nvda_20210601")
    p.add_argument("--ollama-url",
                   default="http://140.113.87.164:11434")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-sec", type=int, default=5)
    p.add_argument("--max-fin", type=int, default=12)
    p.add_argument("--max-ref", type=int, default=12)
    p.add_argument("--max-events", type=int, default=12)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
