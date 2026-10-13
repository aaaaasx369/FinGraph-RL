#!/usr/bin/env python3
"""
V12 evidence-aware feature extraction.

Input: V11 clean_evidence.json
Output:
  evidence_catalog.json
  ablation_runs.json
  features.csv
  claim_audit.csv
  report.json

Python standard library only.
Never overwrites V6/V9/V10/V11.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

CONDITIONS = {
    "SEC-only": ("sec_evidence",),
    "FinDKG-only": ("findkg_evidence",),
    "ReflectKG-only": ("reflectkg_snapshot", "reflectkg_changes"),
    "SEC+FinDKG": ("sec_evidence", "findkg_evidence"),
    "SEC+ReflectKG": (
        "sec_evidence", "reflectkg_snapshot", "reflectkg_changes"
    ),
    "Full": (
        "sec_evidence", "findkg_evidence",
        "reflectkg_snapshot", "reflectkg_changes"
    ),
}

# These are categories of disclosure, NOT calibrated probabilities.
RISK_PATTERNS = {
    "PANDEMIC": r"\bCOVID[-\s]?19\b|\bpandemic\b",
    "SUPPLY_CHAIN": (
        r"\bsupply[\s-]?chain\b|\bcomponent shortage"
        r"|\bmanufactur(?:ing|er)\s+capacity\b"
    ),
    "COMPETITION": (
        r"\bcompetition\b|\bcompetitors?\b"
        r"|\bcompetitive pressures?\b"
    ),
    "REGULATION": (
        r"\bregulat(?:ion|ory|ions)\b"
        r"|\bexport controls?\b|\brestrictions?\b"
    ),
    "CYBERSECURITY": (
        r"\bcyber(?:security|attack)\b"
        r"|\bdata breach\b|\bsecurity incident\b"
    ),
    "LITIGATION": (
        r"\blitigation\b|\blawsuits?\b"
        r"|\blegal proceedings?\b"
    ),
    "DEMAND": (
        r"\bcustomer spending\b|\bdemand\b"
        r"|\brecession\b"
    ),
}

SYSTEM = """You are a cautious historical evidence assessor.
Only evaluate TARGET_TICKER using the supplied evidence.
Do not use outside information.

Important:
- SEC risk disclosure is not proof that financial losses occurred.
- FinDKG triples are dataset observations, not verified business facts.
- ReflectKG edges and events are historical observations,
  not proof that relationships began, ended, or are currently active.
- Preserve directed graph relation direction.
- Do not assert ownership, investment, financial loss, or stock movement
  without evidence directly supporting that precise assertion.
- A competitor mention does not change TARGET_TICKER.
- Never manufacture evidence IDs.
- Do not output free-form factual claims.
- If outlook or realized impact is unsupported, use null, not 0.
- Disclosed risks can exist even if outlook cannot be assessed.

Return JSON with the following fields:
{
  "ticker": "TARGET_TICKER",
  "selected_disclosure_ids": [],
  "realized_impact": null,
  "outlook_impact": null,
  "outlook_evidence_ids": [],
  "reasoning": "Short explanation"
}

selected_disclosure_ids must come ONLY from provided SEC disclosure IDs.
A disclosed-risk category describes source text, not future returns.
realized_impact and outlook_impact are either null or numbers in [-1,1].
outlook_impact is not a verified forecast.
Keep reasoning short and cautious.
"""

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def save_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def valid_score(v):
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and -1 <= v <= 1
    )

def make_catalog(clean):
    """Extract local, directly inspectable SEC risk mention windows."""
    disclosures = []
    for item in clean.get("sec_evidence", []):
        eid = str(item.get("evidence_id", ""))
        content = str(item.get("text", ""))
        if not eid or not content:
            continue

        for category, pattern in RISK_PATTERNS.items():
            hit = re.search(pattern, content, re.I)
            if not hit:
                continue
            window = content[
                max(0, hit.start()-140):
                min(len(content), hit.end()+240)
            ].strip()
            disclosures.append({
                "disclosure_id": "DISC_%03d" % (len(disclosures)+1),
                "ticker": clean["ticker"],
                "source_evidence_id": eid,
                "category": category,
                "excerpt": window,
                "evidence_type": "SEC_TEXT_MENTION",
                "interpretation": "disclosure_or_context_not_realized_loss",
                "published_at": item.get("published_at"),
                "exact_chunk_source_verified": item.get(
                    "exact_chunk_source_verified", False
                )
            })
    return disclosures

def build_prompt_evidence(clean, condition, catalog):
    sections = CONDITIONS[condition]
    selected = []
    for section in sections:
        for e in clean.get(section, []):
            if section == "sec_evidence":
                selected.append({
                    "evidence_id": e["evidence_id"],
                    "source": "SEC",
                    "text": e.get("text", "")[:1700],
                    "published_at": e.get("published_at"),
                })
            elif section == "findkg_evidence":
                selected.append({
                    "evidence_id": e["evidence_id"],
                    "source": "FinDKG",
                    "record": e.get("record"),
                    "observed_at": e.get("observed_at"),
                    "status": "unverified_dataset_observation",
                })
            else:
                selected.append({
                    "evidence_id": e["evidence_id"],
                    "source": section,
                    "subject": e.get("subject"),
                    "relation": e.get("relation"),
                    "object": e.get("object"),
                    "observed_at": (
                        e.get("last_observed_at")
                        or e.get("published_at")
                    ),
                    "evidence": e.get("evidence", "")[:450],
                    "status": "historical_observation",
                })

    visible_disclosures = (
        catalog if "sec_evidence" in sections else []
    )
    return selected, visible_disclosures

def call_qwen(args, ticker, evidence, disclosures):
    prompt = json.dumps({
        "TARGET_TICKER": ticker,
        "evidence": evidence,
        "allowed_sec_disclosures": disclosures,
    }, ensure_ascii=False)
    payload = {
        "model": args.model,
        "system": SYSTEM.replace("TARGET_TICKER", ticker),
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": args.seed,
            "num_ctx": args.num_ctx,
        },
    }
    request = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = json.load(response)
    parsed = json.loads(raw["response"])
    if not isinstance(parsed, dict):
        raise ValueError("Qwen did not return a JSON object")
    return parsed, {
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "eval_count": raw.get("eval_count"),
    }

def baseline_answer(ticker, disclosures):
    # Offline baseline: no predictions, no hallucinated claims.
    return {
        "ticker": ticker,
        "selected_disclosure_ids": [
            e["disclosure_id"] for e in disclosures
        ],
        "realized_impact": None,
        "outlook_impact": None,
        "outlook_evidence_ids": [],
        "reasoning": "Deterministic SEC mention extraction; "
                     "no financial outcome prediction.",
    }

def audit(clean, condition, answer, evidence, disclosures, meta, args):
    flags = []
    all_ids = {str(e["evidence_id"]) for e in evidence}
    disclosure_ids = {
        str(d["disclosure_id"]): d for d in disclosures
    }

    if answer.get("ticker") != clean["ticker"]:
        flags.append("TARGET_COMPANY_MISMATCH")

    raw_selected = answer.get("selected_disclosure_ids")
    if not isinstance(raw_selected, list):
        flags.append("INVALID_DISCLOSURE_LIST")
        raw_selected = []
    selected = []
    for eid in raw_selected:
        if not isinstance(eid, str) or eid not in disclosure_ids:
            flags.append("UNKNOWN_DISCLOSURE_CITATION")
        elif eid not in selected:
            selected.append(eid)

    realized = answer.get("realized_impact")
    outlook = answer.get("outlook_impact")
    for key, val in (
        ("REALIZED", realized),
        ("OUTLOOK", outlook)
    ):
        if val is not None and not valid_score(val):
            flags.append("INVALID_" + key + "_SCORE")

    raw_outlook_ids = answer.get("outlook_evidence_ids")
    if not isinstance(raw_outlook_ids, list):
        flags.append("INVALID_OUTLOOK_CITATION_LIST")
        raw_outlook_ids = []
    outlook_ids = []
    for eid in raw_outlook_ids:
        if not isinstance(eid, str) or eid not in all_ids:
            flags.append("UNKNOWN_OUTLOOK_CITATION")
        else:
            outlook_ids.append(eid)

    # We have not independently verified any realized financial outcomes.
    # Any non-null realized impact stays blocked.
    if realized is not None:
        flags.append("REALIZED_IMPACT_UNVERIFIED")

    if outlook is not None:
        if not outlook_ids:
            flags.append("OUTLOOK_WITHOUT_CITATIONS")
        flags.append("OUTLOOK_REQUIRES_HUMAN_REVIEW")

    reasoning = str(answer.get("reasoning", ""))
    if re.search(
        r"\b(?:current competitors|currently competing)\b",
        reasoning, re.I
    ):
        flags.append("CURRENT_STATUS_LANGUAGE_REVIEW")

    tok = meta.get("prompt_eval_count")
    if isinstance(tok, int) and tok >= args.num_ctx - 256:
        flags.append("POSSIBLE_CONTEXT_TRUNCATION")

    # Deterministic SEC mention extraction is independent of
    # Qwen's optional selection. Preserve all detected categories.
    # The Qwen citations remain available separately for auditing.
    selected = sorted(disclosure_ids)
    categories = sorted({
        disclosure_ids[i]["category"]
        for i in selected
    })

    scores_ok = all(
        v is None or valid_score(v)
        for v in (realized, outlook)
    )

    return {
        "condition": condition,
        "ticker": clean["ticker"],
        "as_of": clean["as_of"],
        "disclosed_risk_present": bool(selected),
        "disclosed_risk_categories": categories,
        "disclosed_risk_category_count": len(categories),
        "realized_impact": (
            realized if scores_ok else None
        ),
        "outlook_impact": (
            outlook if scores_ok else None
        ),
        "confidence": None,
        "outlook_evidence_ids": outlook_ids,
        "selected_disclosure_ids": selected,
        "evidence_sufficiency": {
            "risk_disclosure": bool(selected),
            "realized_impact": False,
            "outlook_impact": False,
        },
        "flags": sorted(set(flags)),
        "status": (
            "BLOCKED" if flags
            else "CANDIDATE_AWAITING_SOURCE_REVIEW"
        ),
        "usable_for_formal_finrl": False,
        "reasoning": reasoning[:1000],
        "model_metadata": meta,
    }

def run_pipeline(args):
    clean = read_json(args.input)
    ticker = clean["ticker"]
    catalog = make_catalog(clean)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "evidence_catalog.json", catalog)

    runs = []
    feature_rows = []
    audit_rows = []
    for condition in CONDITIONS:
        evidence, disclosures = build_prompt_evidence(
            clean, condition, catalog
        )
        meta = {}
        try:
            if args.invoke_qwen:
                answer, meta = call_qwen(
                    args, ticker, evidence, disclosures
                )
            else:
                answer = baseline_answer(ticker, disclosures)

            result = audit(
                clean, condition, answer, evidence,
                disclosures, meta, args
            )
            runs.append({
                "condition": condition,
                "answer": answer,
                "result": result,
                "presented_evidence_ids": [
                    e["evidence_id"] for e in evidence
                ],
            })
        except Exception as exc:
            result = {
                "condition": condition,
                "ticker": ticker,
                "as_of": clean["as_of"],
                "disclosed_risk_present": False,
                "disclosed_risk_categories": [],
                "disclosed_risk_category_count": 0,
                "realized_impact": None,
                "outlook_impact": None,
                "confidence": None,
                "outlook_evidence_ids": [],
                "selected_disclosure_ids": [],
                "evidence_sufficiency": {
                    "risk_disclosure": False,
                    "realized_impact": False,
                    "outlook_impact": False,
                },
                "flags": ["INFERENCE_ERROR"],
                "status": "BLOCKED",
                "usable_for_formal_finrl": False,
                "reasoning": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            runs.append({
                "condition": condition,
                "answer": None,
                "result": result,
                "presented_evidence_ids": [
                    e["evidence_id"] for e in evidence
                ],
            })

        feature_rows.append({
            "ticker": ticker,
            "as_of": clean["as_of"],
            "condition": condition,
            "disclosed_risk_present": (
                result["disclosed_risk_present"]
            ),
            "risk_category_count": (
                result["disclosed_risk_category_count"]
            ),
            "risk_categories": "|".join(
                result["disclosed_risk_categories"]
            ),
            "realized_impact": result["realized_impact"],
            "outlook_impact": result["outlook_impact"],
            "confidence": result["confidence"],
            "status": result["status"],
            "usable_for_formal_finrl": False,
        })

        audit_rows.append({
            "condition": condition,
            "status": result["status"],
            "flags": "|".join(result["flags"]),
            "cited_disclosures": "|".join(
                result["selected_disclosure_ids"]
            ),
            "outlook_citations": "|".join(
                result["outlook_evidence_ids"]
            ),
            "reasoning": result.get("reasoning", ""),
        })
        print(
            f"{condition:20s} "
            f"categories={result['disclosed_risk_category_count']:2d} "
            f"outlook={str(result['outlook_impact']):>5s} "
            f"status={result['status']}"
        )

    save_json(out / "ablation_runs.json", runs)
    save_csv(out / "features.csv", feature_rows, [
        "ticker", "as_of", "condition",
        "disclosed_risk_present", "risk_category_count",
        "risk_categories", "realized_impact",
        "outlook_impact", "confidence", "status",
        "usable_for_formal_finrl",
    ])
    save_csv(out / "claim_audit.csv", audit_rows, [
        "condition", "status", "flags", "cited_disclosures",
        "outlook_citations", "reasoning",
    ])

    report = {
        "ticker": ticker,
        "as_of": clean["as_of"],
        "mode": "QWEN" if args.invoke_qwen else "OFFLINE_BASELINE",
        "disclosures_detected": len(catalog),
        "conditions_completed": len(runs),
        "statuses": dict(Counter(
            r["result"]["status"] for r in runs
        )),
        "input_hash": hashlib.sha256(
            Path(args.input).read_bytes()
        ).hexdigest(),
        "limitations": [
            "Risk category count is not a risk severity score.",
            "Keyword matching may have false positives and negatives.",
            "KG observations are not independently verified facts.",
            "No historical public-availability audit completed.",
            "Model outlook is not calibrated or fact-verified.",
            "No formal FinRL feature approved automatically.",
        ],
    }
    save_json(out / "report.json", report)
    print("\nSaved:", out)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def self_test():
    sample = {
        "ticker": "NVDA",
        "sec_evidence": [{
            "evidence_id": "SEC_001",
            "text": (
                "The COVID-19 pandemic may adversely affect "
                "our supply chain and customer demand."
            )
        }],
        "findkg_evidence": [],
        "reflectkg_snapshot": [],
        "reflectkg_changes": [],
    }
    cat = make_catalog(sample)
    categories = {x["category"] for x in cat}
    assert {"PANDEMIC", "SUPPLY_CHAIN", "DEMAND"} <= categories
    assert not valid_score(float("nan"))
    assert not valid_score(True)
    assert valid_score(-0.2)
    print("SELF-TEST PASSED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/v11/nvda_20210601/clean_evidence.json"
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/v12/nvda_20210601"
    )
    parser.add_argument("--invoke-qwen", action="store_true")
    parser.add_argument(
        "--ollama-url",
        default="http://140.113.87.164:11434"
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_pipeline(args)
