#!/usr/bin/env python3
"""
V11 historical KG-RAG repair and guarded inference.

Input:
  V6 focal FinDKG JSON
  V11 independently retrieved focal FinDKG JSON (optional)

Outputs:
  clean_evidence.json
  findkg_review.csv
  inference.json (optional)
  validation_report.json (optional)
  finrl_candidate.json (optional)

This is a conservative rule screen, not independent fact verification.
No original files are modified.
"""
import argparse
import csv
import json
import math
import re
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

ALIASES = {
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "NVDA": ("NVIDIA", "Nvidia Corp."),
    "AMZN": ("Amazon",),
    "GOOGL": ("Alphabet", "Google"),
}
PORTFOLIO_IDS = {
    "AAPL": 9, "MSFT": 30, "NVDA": 221,
    "AMZN": 20, "GOOGL": 11,
}
# Narrowly interpretable, descriptive observations.
# ACCEPT here means eligible as DATASET OBSERVATION,
# NOT that the underlying claim is independently true.
DESCRIPTIVE = {"Produce", "Operate_In"}
REVIEW_RELATIONS = {
    "Control", "Has", "Relate_To", "Invests_In",
    "Impact", "Negative_Impact_On", "Decrease",
    "Raise", "Announce", "Acquires", "Acquired_By",
}
FEATURE_KEYS = ("impact", "risk", "confidence")

SYSTEM = """
You are an evidence-bound historical financial analyst.
Assess only TARGET_TICKER, not a competitor.

The evidence is incomplete. Do not rely on outside knowledge.
No supplied graph relation proves a real-world transaction
or corporate relationship without independent provenance.
A graph edge is a dataset observation, not confirmed fact.
Do not reverse directed subject/relation/object edges.
Do not describe historically observed relationships as
currently active.
Generic SEC risk warnings are possibilities, not proof
that fraud, losses, or material weaknesses occurred.

AMD = Advanced Micro Devices.
AVGO = Broadcom.
GOOG and GOOGL = Alphabet.

Return JSON with exactly these required top-level fields:
{
  "ticker": "NVDA",
  "impact": null,
  "risk": null,
  "confidence": 0.0,
  "evidence_sufficient": false,
  "claims": [],
  "reasoning": "Insufficient evidence"
}

Replace ticker with TARGET_TICKER.
impact: number in [-1,1] OR null if insufficient evidence.
risk: number in [0,1] OR null if insufficient evidence.
confidence: number in [0,1], self-reported not calibrated.
If insufficient for a financial assessment,
set impact=null, risk=null, evidence_sufficient=false.
Never use risk=0 as a placeholder for missing information.

Every claim must have:
{
 "claim": "a concise evidence-grounded statement",
 "evidence_ids": ["SEC_001"],
 "certainty": "explicit",
 "reason": "why this evidence supports the claim"
}
certainty must be explicit, inferred, or uncertain.
Maximum 4 claims and maximum 2 evidence IDs per claim.
Use no claim rather than inventing evidence.
A citation is NOT proof of factual or causal support.
""".strip()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def historical(value, cutoff):
    try:
        return date.fromisoformat(str(value)[:10]) < date.fromisoformat(cutoff)
    except (ValueError, TypeError):
        return False


def finite(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and low <= value <= high


def numeric_id(value):
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None


def resolve_findkg(items, ticker, cutoff):
    focal = PORTFOLIO_IDS[ticker]
    accepted = []
    reviewed = []

    for item in items:
        record = item.get("record", {})
        if not isinstance(record, dict):
            reviewed.append((item, "INVALID_RECORD"))
            continue

        s = numeric_id(record.get("subject_id"))
        o = numeric_id(record.get("object_id"))
        rel = record.get("relation")
        observed = item.get("observed_at")

        reasons = []

        if not historical(observed, cutoff):
            reasons.append("INVALID_OR_FUTURE_DATE")
        if not record.get("mapping_resolved"):
            reasons.append("ENTITY_MAPPING_UNRESOLVED")
        if s is None or o is None:
            reasons.append("ENTITY_ID_MISSING")
        if s != focal and o != focal:
            reasons.append("NOT_FOCAL_1HOP")
        if rel not in DESCRIPTIVE:
            reasons.append("RELATION_REQUIRES_REVIEW")
        if rel in REVIEW_RELATIONS:
            reasons.append("HIGH_RISK_SEMANTIC_RELATION")

        if reasons:
            reviewed.append((item, "|".join(reasons)))
            continue

        accepted.append({
            "evidence_id": item["evidence_id"],
            "source": "FinDKG",
            "observed_at": observed,
            "record": {
                "subject": record.get("subject"),
                "relation": rel,
                "object": record.get("object"),
                "subject_id": s,
                "object_id": o,
                "direction": "subject -> relation -> object",
                "focal_ticker": ticker,
                "independently_verified": False,
                "semantics": "historical_dataset_observation",
            },
        })

    return accepted, reviewed


def build_clean(args):
    data = load(args.input)
    ticker = data.get("ticker")
    cutoff = data.get("as_of")

    if ticker not in PORTFOLIO_IDS:
        raise ValueError("Unsupported ticker")
    date.fromisoformat(cutoff)

    # Optional independently retrieved focal source.
    if args.focal:
        focal_data = load(args.focal)
        if (
            focal_data.get("ticker") != ticker
            or focal_data.get("as_of") != cutoff
        ):
            raise ValueError("Focal graph ticker/date mismatch")
        fin_input = focal_data.get("evidence", [])
    else:
        fin_input = data.get("findkg_evidence", [])

    fin, review = resolve_findkg(fin_input, ticker, cutoff)

    sec = []
    for item in data.get("sec_evidence", []):
        if not historical(item.get("published_at"), cutoff):
            continue
        if not item.get("doc_id") or not item.get("lightrag_chunk_id"):
            continue
        sec.append({
            "evidence_id": item["evidence_id"],
            "source": "SEC",
            "doc_id": item["doc_id"],
            "published_at": item["published_at"],
            "text": str(item.get("text", ""))[:1600],
            "provenance_status": item.get("provenance_status"),
            "exact_chunk_source_verified": bool(
                item.get("chunk_exact_match_verified", False)
            ),
        })

    reflect = []
    for item in data.get("reflectkg_snapshot", []):
        if not historical(item.get("last_observed_at"), cutoff):
            continue
        reflect.append({
            "evidence_id": item["evidence_id"],
            "source": "ReflectKG",
            "subject": item.get("subject"),
            "relation": item.get("relation"),
            "object": item.get("object"),
            "last_observed_at": item.get("last_observed_at"),
            "evidence": str(item.get("latest_evidence", ""))[:700],
            "current_status": "NOT_VERIFIED",
            "rule_screen_only": True,
        })

    events = []
    for item in data.get("reflectkg_changes", []):
        if not historical(item.get("published_at"), cutoff):
            continue
        events.append({
            "evidence_id": item["evidence_id"],
            "source": "ReflectKG_Event",
            "subject": item.get("subject"),
            "relation": item.get("relation"),
            "object": item.get("object"),
            "event_type": item.get("event_type"),
            "published_at": item.get("published_at"),
            "evidence": str(item.get("evidence", ""))[:350],
            "event_semantics": "observation_not_formation",
        })

    # Prevent alias / evidence ID collisions.
    all_items = sec + fin + reflect + events
    identifiers = [x["evidence_id"] for x in all_items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate evidence IDs")

    reviewed_rows = []
    for item, reason in review:
        record = item.get("record") or {}
        reviewed_rows.append({
            "evidence_id": item.get("evidence_id"),
            "subject": record.get("subject"),
            "relation": record.get("relation"),
            "object": record.get("object"),
            "observed_at": item.get("observed_at"),
            "reason": reason,
        })

    out = Path(args.out_dir)
    write_csv(out / "findkg_review.csv", reviewed_rows, [
        "evidence_id", "subject", "relation",
        "object", "observed_at", "reason"
    ])

    clean = {
        "schema_version": "guarded_fusion_v11",
        "ticker": ticker,
        "as_of": cutoff,
        "sec_evidence": sec,
        "findkg_evidence": fin,
        "reflectkg_snapshot": reflect,
        "reflectkg_changes": events,
        "counts": {
            "sec": len(sec),
            "findkg_eligible": len(fin),
            "findkg_review": len(review),
            "reflectkg_snapshot": len(reflect),
            "reflectkg_events": len(events),
        },
        "restrictions": [
            "No unreviewed directional/investment/control FinDKG edges.",
            "FinDKG dates are not verified publication dates.",
            "ReflectKG observations are not current-status evidence.",
            "SEC chunk provenance may require further verification.",
            "Rule eligible does not establish real-world truth.",
        ],
    }
    save(out / "clean_evidence.json", clean)
    return clean


def call_model(args, clean):
    prompt_items = (
        clean["sec_evidence"][:4]
        + clean["findkg_evidence"][:10]
        + clean["reflectkg_snapshot"][:10]
        + clean["reflectkg_changes"][:8]
    )
    prompt = json.dumps({
        "TARGET_TICKER": clean["ticker"],
        "as_of": clean["as_of"],
        "EVIDENCE": prompt_items,
    }, ensure_ascii=False)

    request = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps({
            "model": args.model,
            "system": SYSTEM,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": args.seed,
                "num_ctx": args.num_ctx,
            },
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(
        request, timeout=600
    ) as response:
        reply = json.load(response)

    answer = json.loads(reply["response"])
    return answer, prompt_items, {
        "prompt_eval_count": reply.get("prompt_eval_count"),
        "eval_count": reply.get("eval_count"),
        "model": reply.get("model", args.model),
    }


def validate_answer(answer, clean, presented, meta, args):
    flags = []
    ticker = clean["ticker"]
    ids = {x["evidence_id"]: x for x in presented}

    if not isinstance(answer, dict):
        raise ValueError("LLM did not return a JSON object")

    if answer.get("ticker") != ticker:
        flags.append("TARGET_TICKER_MISMATCH")

    sufficient = answer.get("evidence_sufficient")
    if not isinstance(sufficient, bool):
        flags.append("INVALID_SUFFICIENCY_FLAG")
        sufficient = False

    impact = answer.get("impact")
    risk = answer.get("risk")
    confidence = answer.get("confidence")

    if impact is not None and not finite(impact, -1, 1):
        flags.append("INVALID_IMPACT")
    if risk is not None and not finite(risk, 0, 1):
        flags.append("INVALID_RISK")
    if not finite(confidence, 0, 1):
        flags.append("INVALID_CONFIDENCE")

    if sufficient and (impact is None or risk is None):
        flags.append("SUFFICIENT_BUT_MISSING_SCORES")
    if not sufficient and (impact is not None or risk is not None):
        flags.append("INSUFFICIENT_BUT_NUMERIC_SCORES")

    claims = answer.get("claims")
    if not isinstance(claims, list) or len(claims) > 4:
        flags.append("INVALID_CLAIMS")
        claims = []

    # When no evidence was supplied, claims/scores must not
    # masquerade as knowledge.
    if not presented and (claims or sufficient):
        flags.append("OUTPUT_WITHOUT_EVIDENCE")

    for c in claims:
        if not isinstance(c, dict):
            flags.append("INVALID_CLAIM_OBJECT")
            continue
        text = c.get("claim", "")
        cited = c.get("evidence_ids", [])
        if not isinstance(text, str) or not text.strip():
            flags.append("EMPTY_CLAIM")
            continue
        if not isinstance(cited, list):
            flags.append("INVALID_CLAIM_CITATIONS")
            continue
        if len(cited) == 0:
            flags.append("UNCITED_CLAIM")
        if len(cited) > 2:
            flags.append("CITATION_SATURATION")
        if any(not isinstance(i, str) or i not in ids
               for i in cited):
            flags.append("UNKNOWN_CITATION")

        if c.get("certainty") not in (
            "explicit", "inferred", "uncertain"
        ):
            flags.append("INVALID_CERTAINTY")

        # Known model mistakes from earlier V6-V10 runs.
        if re.search(
            r"Advanced Micro Devices\s*\(AVGO\)"
            r"|Broadcom\s*\(AMD\)", text, re.I
        ):
            flags.append("ENTITY_MISMATCH")

        if ticker == "NVDA" and re.search(
            r"^\s*(?:Advanced Micro Devices|AMD)"
            r"(?:\s*\(AMD\))?\s+"
            r"(?:faces|suffers|has|is|reports)\b",
            text, re.I,
        ):
            flags.append("TARGET_COMPANY_MISMATCH")

        if re.search(
            r"\bcurrent competitors?\b"
            r"|\bcurrently competes?\b",
            text, re.I,
        ) and any(
            i in ids and ids[i]["source"].startswith("ReflectKG")
            for i in cited
        ):
            flags.append("CURRENT_RELATION_STATUS_UNVERIFIED")

        if re.search(
            r"\bactual fraud\b|\bconfirmed fraud\b"
            r"|\bunderlying financial or operational issues\b",
            text, re.I
        ):
            flags.append("ACTUAL_EVENT_REQUIRES_PROOF")

        # Investment direction rule for the known failure mode.
        for i in cited:
            e = ids.get(i)
            if not e or e["source"] != "FinDKG":
                continue
            r = e["record"]
            if (
                r["relation"] == "Invests_In"
                and "technology companies" in str(r["subject"]).lower()
                and r["object_id"] == PORTFOLIO_IDS[ticker]
                and re.search(
                    r"(?:nvidia|nvda).*invest(?:s|ed|ing)?"
                    r"\s+in\s+technology companies",
                    text, re.I
                )
            ):
                flags.append("REVERSED_RELATION_DIRECTION")

    count_tokens = meta.get("prompt_eval_count")
    if (
        isinstance(count_tokens, int)
        and count_tokens >= args.num_ctx - 256
    ):
        flags.append("POSSIBLE_CONTEXT_TRUNCATION")

    # Not possible to verify claim entailment automatically
    # from citation IDs. Mark all schema-clean outputs pending
    # human semantic review.
    flags = sorted(set(flags))
    invalid = bool(flags)

    return {
        "ticker": ticker,
        "as_of": clean["as_of"],
        "model_output": answer,
        "flags": flags,
        "validation_status": (
            "BLOCKED" if invalid else "AWAITING_HUMAN_GROUNDING"
        ),
        "usable_for_formal_finrl": False,
        "reason": (
            "Known errors detected"
            if invalid else
            "Claims and source semantics not independently verified"
        ),
        "model_metadata": meta,
    }


def self_test():
    data = {
        "evidence_id": "FIN_001",
        "observed_at": "2021-01-01",
        "record": {
            "subject": "Nvidia Corp.",
            "relation": "Produce",
            "object": "Computer Chips",
            "subject_id": 221, "object_id": 500,
            "mapping_resolved": True,
        }
    }
    selected, rejected = resolve_findkg(
        [data], "NVDA", "2021-06-01"
    )
    assert len(selected) == 1 and not rejected

    bad = json.loads(json.dumps(data))
    bad["record"]["relation"] = "Control"
    chosen, rejected = resolve_findkg(
        [bad], "NVDA", "2021-06-01"
    )
    assert not chosen and len(rejected) == 1

    future = json.loads(json.dumps(data))
    future["observed_at"] = "2022-01-01"
    chosen, rejected = resolve_findkg(
        [future], "NVDA", "2021-06-01"
    )
    assert not chosen and len(rejected) == 1

    unrelated = json.loads(json.dumps(data))
    unrelated["record"]["subject_id"] = 1
    chosen, rejected = resolve_findkg(
        [unrelated], "NVDA", "2021-06-01"
    )
    assert not chosen and len(rejected) == 1

    assert not finite(-0.75, 0, 1)
    assert finite(-0.75, -1, 1)
    print("SELF-TEST PASSED: focal, semantic, time, score")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="outputs/v6/nvda_20210601_focal_findkg.json"
    )
    ap.add_argument(
        "--focal",
        default="outputs/findkg_focal_v2/nvda_20210601.json"
    )
    ap.add_argument("--out-dir", default="outputs/v11/nvda_20210601")
    ap.add_argument("--invoke-qwen", action="store_true")
    ap.add_argument("--ollama-url",
                    default="http://140.113.87.164:11434")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    clean = build_clean(args)
    print("=== CLEAN EVIDENCE ===")
    print(json.dumps(
        clean["counts"], indent=2, ensure_ascii=False
    ))
    print("Saved:", Path(args.out_dir) / "clean_evidence.json")

    if not args.invoke_qwen:
        print("Qwen was NOT called. Add --invoke-qwen to run.")
        return

    out = Path(args.out_dir)
    try:
        answer, presented, meta = call_model(args, clean)
        save(out / "inference.json", {
            "model_output": answer,
            "presented_evidence_ids": [
                e["evidence_id"] for e in presented
            ],
            "metadata": meta,
        })
        report = validate_answer(
            answer, clean, presented, meta, args
        )
        save(out / "validation_report.json", report)

        # Candidate feature values are kept even if blocked,
        # but explicit approval remains false.
        candidate = {
            "ticker": clean["ticker"],
            "as_of": clean["as_of"],
            "impact": answer.get("impact"),
            "risk": answer.get("risk"),
            "confidence": answer.get("confidence"),
            "evidence_sufficient": answer.get(
                "evidence_sufficient"
            ),
            "validation_status": report["validation_status"],
            "usable_for_formal_finrl": False,
            "flags": report["flags"],
        }
        save(out / "finrl_candidate.json", candidate)
        print("\n=== VALIDATION ===")
        print(json.dumps({
            "status": report["validation_status"],
            "flags": report["flags"],
            "features": candidate,
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        save(out / "inference_error.json", {
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
