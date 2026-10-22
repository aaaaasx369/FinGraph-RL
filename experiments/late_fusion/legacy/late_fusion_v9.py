#!/usr/bin/env python3
"""V9: Controlled source ablation, claim-level citations, repeatability.

Input: existing V6 JSON.
No LightRAG indexing or KG retrieval.
Python standard library only.
"""

import argparse
import csv
import json
import math
import re
import statistics
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
ORDER = ("SEC", "FinDKG", "ReflectKG")
SCORES = ("impact", "risk", "confidence")

SYSTEM = """
You are a historical corporate financial-evidence analyst.

Analyze only the EVIDENCE supplied in the user message.
Do not use future knowledge.
The evidence may be incomplete. Do not infer that an event
occurred simply because a risk factor describes its possibility.

Interpretation:
- SEC risk-factor language describes possible risks unless it
  explicitly documents a realized event.
- Competition is not proof of negative realized returns.
- FinDKG edges are observed graph data, not proof that a
  commercial relationship began or ended on that date.
- ReflectKG FIRST_OBSERVED and REASSERTED describe when a
  relationship appeared in the available SEC evidence.
- AMD is Advanced Micro Devices.
- AVGO is Broadcom.
- Do not claim real-world relation changes without explicit proof.
- Do not invent evidence IDs.
- Cite only evidence directly relevant to a particular claim.
- Prefer a small number of useful citations, not every evidence item.
- No forced citations from any source.

Return one JSON object containing:
{
  "impact": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "claims": [
    {
      "claim": "specific, evidence-based factual statement",
      "evidence_ids": ["SEC_001"],
      "certainty": "explicit",
      "reason": "why the cited evidence supports the claim"
    }
  ],
  "reasoning": "short synthesis"
}

The three scores must be numeric in [-1,1].
impact: negative adverse, positive favorable.
risk: higher means greater assessed risk.
confidence: self-rated strength of the available evidence, not
empirically calibrated probability of correctness.

certainty must be exactly one of:
explicit, inferred, uncertain.

Return 1 to 5 claims if meaningful evidence is available.
A claim can have multiple evidence IDs but prefer 1 or 2.
For uncertain claims, explain the uncertainty.
Claims must not assert facts absent from the evidence.
If the evidence is insufficient, return an empty claims list
and low confidence.
""".strip()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def csv_write(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def before(value, as_of):
    try:
        return date.fromisoformat(str(value)[:10]) < date.fromisoformat(as_of)
    except (ValueError, TypeError):
        return False


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Score must be a numeric JSON value")
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Score must be finite and in [-1,1]")
    return float(value)


def make_pools(d, args):
    cutoff = d["as_of"]
    pools = {source: [] for source in ORDER}
    all_ids = set()

    specs = [
        ("SEC", "sec_evidence", args.max_sec, "published_at"),
        ("FinDKG", "findkg_evidence", args.max_fin, "observed_at"),
        ("ReflectKG", "reflectkg_snapshot",
         args.max_ref, "last_observed_at"),
        ("ReflectKG", "reflectkg_changes",
         args.max_events, "published_at"),
    ]

    for source, field, maximum, time_key in specs:
        items = d.get(field, [])
        if not isinstance(items, list):
            raise ValueError(field + " must be a list")

        for e in items[:maximum]:
            if not isinstance(e, dict):
                raise ValueError(field + " has nonobject item")

            eid = e.get("evidence_id")
            if not isinstance(eid, str) or not eid:
                raise ValueError(field + " has missing evidence_id")
            if eid in all_ids:
                raise ValueError("Duplicate evidence ID: " + eid)
            all_ids.add(eid)

            timestamp = e.get(time_key)
            if not before(timestamp, cutoff):
                raise ValueError(
                    f"Invalid point-in-time evidence {eid}: {timestamp}"
                )

            if field == "sec_evidence":
                body = {
                    "doc_id": e.get("doc_id"),
                    "published_at": timestamp,
                    "text": str(e.get("text", ""))[:args.sec_chars]
                }
            elif field == "findkg_evidence":
                body = {
                    "observed_at": timestamp,
                    "relation_record": e.get("record", {}),
                    "time_semantics": "dataset_observation"
                }
            elif field == "reflectkg_snapshot":
                body = {
                    k: e.get(k) for k in (
                        "subject", "relation", "object",
                        "first_observed_at", "last_observed_at",
                        "observation_count"
                    )
                }
                body["latest_evidence"] = str(
                    e.get("latest_evidence", "")
                )[:args.ref_chars]
                body["status"] = "previously_observed_not_confirmed_active"
            else:
                body = {
                    k: e.get(k) for k in (
                        "subject", "relation", "object",
                        "event_type", "published_at"
                    )
                }
                body["evidence"] = str(
                    e.get("evidence", "")
                )[:args.event_chars]

            pools[source].append({
                "id": eid,
                "source": source,
                "kind": field,
                "content": body,
            })

    return pools


def selected_evidence(pools, enabled):
    # Fixed source order across all experiments.
    return [
        item for source in ORDER if source in enabled
        for item in pools[source]
    ]


def prepare_prompt(d, evidence):
    # Condition labels deliberately excluded.
    return json.dumps({
        "ticker": d["ticker"],
        "as_of": d["as_of"],
        "task": "Assess historical financial outlook and risks",
        "EVIDENCE": evidence,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def qwen_call(args, prompt, seed):
    payload = {
        "model": args.model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": args.num_ctx,
        },
    }
    request = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(
        request, timeout=args.timeout
    ) as response:
        meta = json.load(response)
    answer = json.loads(meta["response"])
    return answer, meta


def normalize_answer(raw, allowed):
    if not isinstance(raw, dict):
        raise ValueError("Qwen output must be an object")

    scores = {k: number(raw[k]) for k in SCORES}
    claims = raw.get("claims")
    if not isinstance(claims, list) or len(claims) > 5:
        raise ValueError("claims must be a list with <=5 items")

    normalized_claims = []
    for index, c in enumerate(claims, 1):
        if not isinstance(c, dict):
            raise ValueError("Claim must be an object")
        text = c.get("claim")
        reason = c.get("reason")
        certainty = c.get("certainty")
        ids = c.get("evidence_ids")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty claim")
        if not isinstance(reason, str):
            raise ValueError("Invalid claim reason")
        if certainty not in ("explicit", "inferred", "uncertain"):
            raise ValueError("Invalid claim certainty")
        if not isinstance(ids, list) or not all(
            isinstance(x, str) for x in ids
        ):
            raise ValueError("Invalid claim citations")
        if set(ids) - allowed:
            raise ValueError(
                "Unknown claim citations: " + str(set(ids) - allowed)
            )
        normalized_claims.append({
            "claim_id": f"CLAIM_{index:02d}",
            "claim": text.strip(),
            "evidence_ids": list(dict.fromkeys(ids)),
            "certainty": certainty,
            "reason": reason,
        })

    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError("Missing reasoning string")

    return {
        **scores,
        "claims": normalized_claims,
        "reasoning": reasoning,
    }


def word_set(text):
    stopwords = {
        "about", "their", "there", "which", "where",
        "could", "would", "should", "these", "those",
        "company", "companies", "financial", "risks",
        "evidence", "given", "provides", "indicates",
        "potential", "related", "including", "during",
        "because", "between", "other", "might",
    }
    return {
        w for w in re.findall(r"[a-z]{4,}", text.lower())
        if w not in stopwords
    }


def claim_audit(answer, selected):
    lookup = {e["id"]: e for e in selected}
    claim_rows = []
    flags = []

    wrong_entity = [
        r"\bAdvanced Micro Devices\s*\(AVGO\)",
        r"\bBroadcom\s*\(AMD\)",
    ]
    overstatement = [
        r"\bunderlying financial or operational issues\b",
        r"\bactual fraud\b",
        r"\bfraud (?:occurred|was detected)\b",
        r"\bconfirmed accounting (?:fraud|irregularities)\b",
    ]

    for c in answer["claims"]:
        claim = c["claim"]
        ids = c["evidence_ids"]
        quoted = " ".join(
            json.dumps(lookup[eid]["content"], ensure_ascii=False)
            for eid in ids
        )
        tokens = word_set(claim)
        supporting = word_set(quoted)
        shared = tokens.intersection(supporting)

        cflags = []
        if not ids:
            cflags.append("UNCITED_CLAIM")
        elif not shared:
            cflags.append("LOW_LEXICAL_OVERLAP_REVIEW")

        if any(re.search(p, claim, re.I) for p in wrong_entity):
            cflags.append("ENTITY_MISMATCH")

        if any(re.search(p, claim, re.I) for p in overstatement):
            cflags.append("UNSUPPORTED_EVENT_CLAIM_REVIEW")

        if c["certainty"] == "explicit" and not ids:
            cflags.append("EXPLICIT_WITHOUT_EVIDENCE")

        flags.extend(cflags)
        claim_rows.append({
            **c,
            "source_types": sorted({
                lookup[eid]["source"] for eid in ids
            }),
            "lexical_overlap_count": len(shared),
            "lexical_overlap_terms": sorted(shared)[:15],
            "flags": cflags,
            "grounding_status": (
                "REVIEW" if cflags else "NOT_AUTOMATICALLY_VERIFIED"
            )
        })

    all_text = answer["reasoning"] + " " + " ".join(
        c["claim"] for c in answer["claims"]
    )
    if any(re.search(p, all_text, re.I) for p in wrong_entity):
        flags.append("ENTITY_MISMATCH")
    if abs(answer["impact"]) >= 0.9 and answer["confidence"] >= 0.9:
        flags.append("EXTREME_SCORE_REVIEW")

    ids_cited = {
        eid for c in answer["claims"]
        for eid in c["evidence_ids"]
    }

    per_source = {}
    for source in ORDER:
        available = sum(e["source"] == source for e in selected)
        cited = sum(
            lookup[eid]["source"] == source
            for eid in ids_cited
        )
        per_source[source] = {
            "available": available,
            "unique_cited": cited,
            "coverage": cited / available if available else None,
            "share": cited / len(ids_cited) if ids_cited else 0.0,
        }

    return {
        "claims": claim_rows,
        "unique_citations": sorted(ids_cited),
        "source_citations": per_source,
        "flags": sorted(set(flags)),
        "note": (
            "Citation validity and lexical overlap are proxies only. "
            "No semantic entailment has been established."
        ),
    }


def deterministic_graph_metrics(selected):
    # Metrics based solely on the OBSERVED input graph records.
    # These are not real-world network change estimates.
    graph = [e for e in selected if e["source"] != "SEC"]
    events = [
        e for e in graph
        if e["kind"] == "reflectkg_changes"
    ]
    first = sum(
        e["content"].get("event_type") == "FIRST_OBSERVED"
        for e in events
    )
    reassert = sum(
        e["content"].get("event_type") == "REASSERTED"
        for e in events
    )
    return {
        "has_graph_evidence": bool(graph),
        "graph_record_count": len(graph),
        "reflect_event_count": len(events),
        "first_observed_count": first,
        "reasserted_count": reassert,
        # Not defined from SEC-only evidence.
        "graph_change": None if not graph else len(events),
        "graph_change_definition": (
            "count of supplied ReflectKG observation events"
            if graph else "not_applicable_no_KG"
        ),
    }


def summary_stats(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def run(args):
    d = load(args.input)
    if d.get("schema_version") != "late_fusion_v6":
        raise ValueError("Expected late_fusion_v6 file")

    pools = make_pools(d, args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    configuration = {
        "input": args.input,
        "ticker": d["ticker"],
        "as_of": d["as_of"],
        "model": args.model,
        "seed_base": args.seed,
        "repeats": args.repeats,
        "num_ctx": args.num_ctx,
        "temperature": 0,
        "source_counts": {k: len(v) for k, v in pools.items()},
        "evidence_caps": {
            "sec": args.max_sec, "fin": args.max_fin,
            "ref": args.max_ref, "events": args.max_events,
        },
        "limitations": [
            "Self-rated confidence is not calibrated.",
            "Claim-to-citation matching is not entailment verification.",
            "Different source combinations imply different token counts.",
            "Graph change counts are observation counts only.",
            "Only the supplied V6 evidence universe is evaluated.",
            "Source ablation is not a portfolio performance experiment.",
        ],
    }
    save(out / "configuration.json", configuration)

    records = []
    for name, enabled in CONDITIONS.items():
        evidence = selected_evidence(pools, enabled)
        prompt = prepare_prompt(d, evidence)

        if not evidence:
            print(name, "SKIP no evidence", flush=True)
            continue

        for repeat in range(args.repeats):
            seed = args.seed + repeat
            safe = name.replace("+", "_").replace("-", "_")
            path = out / f"{safe}_repeat{repeat+1:02d}.json"

            if path.exists() and not args.overwrite:
                record = load(path)
                if record.get("status") == "OK":
                    print(name, repeat+1, "REUSED", flush=True)
                    records.append(record)
                    continue

            start = time.monotonic()
            print(name, repeat+1, "RUNNING", flush=True)

            try:
                raw, meta = qwen_call(args, prompt, seed)
                answer = normalize_answer(
                    raw, {e["id"] for e in evidence}
                )
                audit = claim_audit(answer, evidence)

                # If prompt was truncated, do not consider it
                # a controlled comparison.
                prompt_tokens = meta.get("prompt_eval_count")
                if (
                    isinstance(prompt_tokens, int)
                    and prompt_tokens >= args.num_ctx - 256
                ):
                    raise RuntimeError(
                        "Possible context truncation: "
                        f"prompt_eval_count={prompt_tokens}"
                    )

                record = {
                    "status": "OK",
                    "condition": name,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "enabled_sources": list(enabled),
                    "evidence_count": len(evidence),
                    "prompt_chars": len(prompt),
                    "scores": {k: answer[k] for k in SCORES},
                    "answer": answer,
                    "audit": audit,
                    "graph_metrics": deterministic_graph_metrics(
                        evidence
                    ),
                    "generation": {
                        "model": meta.get("model", args.model),
                        "prompt_eval_count": prompt_tokens,
                        "eval_count": meta.get("eval_count"),
                        "duration_seconds": round(
                            time.monotonic() - start, 2
                        ),
                    },
                }
            except Exception as exc:
                record = {
                    "status": "ERROR",
                    "condition": name,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "evidence_count": len(evidence),
                    "error": f"{type(exc).__name__}: {exc}",
                }

            save(path, record)
            records.append(record)
            print(name, repeat+1, record["status"], flush=True)

    rows = []
    claim_rows = []
    citation_rows = []
    summary_rows = []

    for r in records:
        if r["status"] != "OK":
            rows.append({
                "condition": r["condition"],
                "repeat": r["repeat"],
                "status": r["status"],
                "error": r.get("error", ""),
            })
            continue

        counts = r["audit"]["source_citations"]
        score_row = {
            "condition": r["condition"],
            "repeat": r["repeat"],
            "status": "OK",
            **r["scores"],
            "claim_count": len(r["audit"]["claims"]),
            "unique_citations": len(
                r["audit"]["unique_citations"]
            ),
            "audit_flags": "|".join(r["audit"]["flags"]),
            "graph_change_observed": r["graph_metrics"][
                "graph_change"
            ],
            "prompt_tokens": r["generation"]["prompt_eval_count"],
        }
        rows.append(score_row)

        for c in r["audit"]["claims"]:
            claim_rows.append({
                "condition": r["condition"],
                "repeat": r["repeat"],
                "claim_id": c["claim_id"],
                "claim": c["claim"],
                "certainty": c["certainty"],
                "evidence_ids": "|".join(c["evidence_ids"]),
                "source_types": "|".join(c["source_types"]),
                "lexical_overlap_count": c[
                    "lexical_overlap_count"
                ],
                "flags": "|".join(c["flags"]),
                "review_required": (
                    "YES" if c["flags"] else "NOT_VERIFIED"
                ),
            })

        for source in ORDER:
            stats = counts[source]
            citation_rows.append({
                "condition": r["condition"],
                "repeat": r["repeat"],
                "source": source,
                **stats,
            })

    for condition in CONDITIONS:
        valid = [
            r for r in records
            if r["condition"] == condition
            and r["status"] == "OK"
        ]
        row = {
            "condition": condition,
            "successful_repeats": len(valid),
        }
        for key in SCORES:
            result = summary_stats([v["scores"][key] for v in valid])
            row[key + "_mean"] = result["mean"]
            row[key + "_std"] = result["std"]
            row[key + "_min"] = result["min"]
            row[key + "_max"] = result["max"]
        summary_rows.append(row)

    csv_write(out / "scores_per_run.csv", rows, [
        "condition", "repeat", "status", *SCORES,
        "claim_count", "unique_citations", "audit_flags",
        "graph_change_observed", "prompt_tokens", "error",
    ])
    csv_write(out / "claims_audit.csv", claim_rows, [
        "condition", "repeat", "claim_id", "claim",
        "certainty", "evidence_ids", "source_types",
        "lexical_overlap_count", "flags", "review_required",
    ])
    csv_write(out / "citation_reliance.csv", citation_rows, [
        "condition", "repeat", "source", "available",
        "unique_cited", "coverage", "share",
    ])
    summary_columns = ["condition", "successful_repeats"]
    for key in SCORES:
        summary_columns.extend([
            key + "_mean", key + "_std",
            key + "_min", key + "_max"
        ])
    csv_write(
        out / "ablation_summary.csv",
        summary_rows, summary_columns
    )

    save(out / "results_summary.json", {
        "configuration": configuration,
        "ablation": summary_rows,
        "successful_runs": sum(
            r["status"] == "OK" for r in records
        ),
        "failed_runs": sum(
            r["status"] == "ERROR" for r in records
        ),
    })

    print("\n=== V9 SUMMARY ===")
    for row in summary_rows:
        print(
            f"{row['condition']:17s}",
            "runs=", row["successful_repeats"],
            "impact=", row["impact_mean"],
            "risk=", row["risk_mean"],
            "confidence=", row["confidence_mean"]
        )
    print("\nSaved:", out)


def self_test():
    import argparse as _argparse
    args = _argparse.Namespace(
        max_sec=5, max_fin=5, max_ref=5, max_events=5,
        sec_chars=1600, ref_chars=700, event_chars=400,
    )
    d = {
        "ticker": "NVDA",
        "as_of": "2021-06-01",
        "sec_evidence": [{
            "evidence_id": "SEC_001",
            "published_at": "2021-05-26",
            "text": "NVIDIA disclosed supply chain uncertainty."
        }],
        "findkg_evidence": [{
            "evidence_id": "FIN_001",
            "observed_at": "2021-05-30",
            "record": {"relation": "sample"}
        }],
        "reflectkg_snapshot": [],
        "reflectkg_changes": [],
    }

    pools = make_pools(d, args)
    assert len(selected_evidence(pools, ("SEC",))) == 1
    assert len(selected_evidence(pools, ("FinDKG",))) == 1

    answer = normalize_answer({
        "impact": -0.2, "risk": 0.4, "confidence": 0.3,
        "claims": [{
            "claim": "NVIDIA disclosed supply chain uncertainty.",
            "evidence_ids": ["SEC_001"],
            "certainty": "explicit",
            "reason": "The SEC text says so."
        }],
        "reasoning": "Supply chain uncertainty."
    }, {"SEC_001"})

    audit = claim_audit(answer, pools["SEC"])
    assert audit["source_citations"]["SEC"]["unique_cited"] == 1
    assert deterministic_graph_metrics(pools["SEC"])[
        "graph_change"
    ] is None

    try:
        normalize_answer({
            **answer,
            "claims": [{
                **answer["claims"][0],
                "evidence_ids": ["FAKE_123"],
            }],
        }, {"SEC_001"})
        raise AssertionError("Fake evidence ID was accepted")
    except ValueError:
        pass

    future = json.loads(json.dumps(d))
    future["sec_evidence"][0]["published_at"] = "2022-01-01"
    try:
        make_pools(future, args)
        raise AssertionError("Future document was accepted")
    except ValueError:
        pass

    assert math.isclose(summary_stats([0.2, 0.4])["mean"], 0.3)
    print("SELF-TEST PASSED: dates, citations, no-KG, stats")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", default="outputs/v6/nvda_20210601_qwen.json"
    )
    p.add_argument(
        "--out-dir", default="outputs/v9/nvda_20210601"
    )
    p.add_argument(
        "--ollama-url", default="http://140.113.87.164:11434"
    )
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-sec", type=int, default=5)
    p.add_argument("--max-fin", type=int, default=12)
    p.add_argument("--max-ref", type=int, default=12)
    p.add_argument("--max-events", type=int, default=12)
    p.add_argument("--sec-chars", type=int, default=1600)
    p.add_argument("--ref-chars", type=int, default=700)
    p.add_argument("--event-chars", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.repeats < 1:
        p.error("--repeats must be at least 1")

    run(args)


if __name__ == "__main__":
    main()
