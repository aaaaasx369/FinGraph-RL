#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ollama

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/light_rag_dynamic_kg/ablation_v1"
OUT = ROOT / "results/light_rag_dynamic_kg/verified_v1"

TICKERS = {"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"}
GROUPS = {
    "A_KG_RAG": {"KG"},
    "B_SEC_RAG": {"SEC"},
    "C_KG_SEC_RAG": {"KG", "SEC"},
}
HOST = "http://140.113.87.164:11434"
JUDGE_MODEL = "qwen3.6:27b"
VERIFIER_VERSION = "financial_claim_score_v1"

SYSTEM = """
You are a conservative independent financial evidence auditor.

You receive:
- a company and historical decision date
- original evidence statements and source IDs
- model-generated claims with verbatim quotes
- impact, risk, confidence scores

You must review the claim and score support using ONLY the
provided evidence.

Treat evidence as possibly noisy.

A KG triple with a generic relation such as Relate_To,
Has, or Control does not by itself prove financial impact.

A company's product, customer, or sector membership does
not prove future stock appreciation or business performance.

Do not use external financial facts, future knowledge,
market outcomes, stock prices, or company reputation.

For each claim:
- supported=true ONLY when the cited evidence directly
  supports the ENTIRE substantive claim
- false for overstatement, speculation or invented causality

For impact and risk:
- Evaluate whether the claimed sign AND numerical magnitude
  are reasonably supported by specific evidence
- Strong magnitudes need strong, specific evidence
- Mere existence of risks does not justify risk=0.8
- Ordinary business activity does not justify impact=+0.8
- Reject unsupported precision or magnitude
- Null generated scores must remain null

Confidence is subjective evidence sufficiency, not
a calibrated probability. Reject unusually high
confidence if evidence is sparse, generic or weak.

Return ONLY JSON with:
{
 "claims": [
   {"index":0,"supported":false,"reason":"..."}
 ],
 "scores": {
   "impact":{"supported":false,"reason":"..."},
   "risk":{"supported":false,"reason":"..."},
   "confidence":{"supported":false,"reason":"..."}
 },
 "overall_reason":"..."
}

Always include one claim result per input claim,
using the original zero-based index.
Always include all three score assessments.
"""


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def sha(data):
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, ensure_ascii=False,
            default=str
        ).encode("utf-8")
    ).hexdigest()


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def valid_number(value, lo, hi):
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not lo <= x <= hi:
        return None
    return x


def deterministic_audit(record, path):
    ticker = record.get("ticker")
    date = record.get("date")
    group = record.get("group")
    generated = record.get("generated")
    evidence = record.get("evidence") or []

    errors = []

    if ticker not in TICKERS or group not in GROUPS:
        errors.append("invalid_identity")

    try:
        cutoff = pd.Timestamp(date).normalize()
    except Exception:
        cutoff = None
        errors.append("invalid_date")

    if not isinstance(evidence, list):
        evidence = []
        errors.append("invalid_evidence")

    source_map = {}

    for item in evidence:
        if not isinstance(item, dict):
            errors.append("evidence_not_dict")
            continue

        sid = str(item.get("id", ""))
        typ = item.get("type")

        if not sid or sid in source_map:
            errors.append("duplicate_or_empty_source_id")
            continue

        if typ not in GROUPS.get(group, set()):
            errors.append("source_ablation_violation")

        if typ == "KG" and not sid.startswith("KG_"):
            errors.append("kg_id_mismatch")
        if typ == "SEC" and not sid.startswith("SEC_"):
            errors.append("sec_id_mismatch")

        available = pd.to_datetime(
            item.get("available_at"),
            utc=True,
            errors="coerce",
        )

        if (
            cutoff is None or pd.isna(available)
            or available.tz_localize(None) >= cutoff
        ):
            errors.append("pit_violation")

        if not str(item.get("statement", "")).strip():
            errors.append("empty_source_statement")

        source_map[sid] = item

    claims = (
        generated.get("claims", [])
        if isinstance(generated, dict) else []
    )

    if not isinstance(claims, list):
        claims = []
        errors.append("invalid_claims")

    claim_checks = []

    for i, claim in enumerate(claims):
        reasons = []

        if not isinstance(claim, dict):
            claim = {}
            reasons.append("claim_not_object")

        ids = claim.get("evidence_ids", [])
        quote = claim.get("support_quote", "")
        assertion = claim.get("claim", "")

        if not isinstance(ids, list) or len(ids) != 1:
            reasons.append("requires_exactly_one_source")
        elif str(ids[0]) not in source_map:
            reasons.append("unknown_source")
        else:
            statement = source_map[str(ids[0])]["statement"]

            if (
                not isinstance(quote, str)
                or len(quote.strip()) < 15
                or norm(quote) not in norm(statement)
            ):
                reasons.append("quote_not_verbatim")

        if not isinstance(assertion, str) or not assertion.strip():
            reasons.append("empty_claim")

        claim_checks.append({
            "index": i,
            "citation_pass": not reasons,
            "reasons": reasons,
        })

    values = {}

    for key, lo, hi in [
        ("impact", -1, 1),
        ("risk", 0, 1),
        ("confidence", 0, 1),
    ]:
        raw = (
            generated.get(key)
            if isinstance(generated, dict) else None
        )
        value = valid_number(raw, lo, hi)

        if raw is not None and value is None:
            errors.append(f"invalid_{key}")

        values[key] = value

    return {
        "errors": sorted(set(errors)),
        "claims": claim_checks,
        "values": values,
        "source_map": source_map,
        "generated": generated,
        "cutoff": str(date),
    }


def ask_judge(client, record, audit):
    generated = audit["generated"]

    claims = [
        {
            "index": i,
            "claim": c.get("claim"),
            "evidence_ids": c.get("evidence_ids"),
            "support_quote": c.get("support_quote"),
        }
        for i, c in enumerate(generated.get("claims", []))
    ]

    payload = {
        "ticker": record["ticker"],
        "date": record["date"],
        "group": record["group"],
        "evidence": record["evidence"],
        "claims": claims,
        "scores": audit["values"],
    }

    response = client.chat(
        model=JUDGE_MODEL,
        think=False,
        format="json",
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False
                ),
            },
        ],
        options={
            "temperature": 0,
            "num_ctx": 16384,
            "num_predict": 2000,
        },
    )

    raw = response.message.content or ""
    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise ValueError("Judge returned non-object")

    return parsed


def verify_one(client, path, force=False):
    record = json.loads(path.read_text(encoding="utf-8"))
    input_hash = sha(record)

    outfile = OUT / path.name

    if outfile.exists() and not force:
        previous = json.loads(
            outfile.read_text(encoding="utf-8")
        )
        if (
            previous.get("input_sha256") == input_hash
            and previous.get("verifier_version")
            == VERIFIER_VERSION
            and previous.get("judge_model") == JUDGE_MODEL
        ):
            return previous

    audit = deterministic_audit(record, path)

    evidence = record.get("evidence") or []
    generated = audit["generated"]
    citation_checks = audit["claims"]

    result = {
        "ticker": record.get("ticker"),
        "date": record.get("date"),
        "group": record.get("group"),
        "input_path": str(path),
        "input_sha256": input_hash,
        "verifier_version": VERIFIER_VERSION,
        "judge_model": JUDGE_MODEL,
        "deterministic_errors": audit["errors"],
        "citation_checks": citation_checks,
        "judge": None,
        "automated_semantic_pass": False,
        "human_verified": False,
        "accepted_scores": {
            "impact": None,
            "risk": None,
            "confidence": None,
        },
        "accepted_masks": {
            "impact": 0,
            "risk": 0,
            "confidence": 0,
        },
        "status": "REJECTED",
    }

    if (
        audit["errors"]
        or not evidence
        or not isinstance(generated, dict)
        or not citation_checks
        or not all(x["citation_pass"] for x in citation_checks)
    ):
        result["status"] = "DETERMINISTIC_REJECT"
        write_atomic(outfile, result)
        return result

    try:
        judged = ask_judge(client, record, audit)
        result["judge"] = judged

        decisions = judged.get("claims", [])
        score_decisions = judged.get("scores", {})

        if not isinstance(decisions, list):
            raise ValueError("Malformed claim decisions")

        if not isinstance(score_decisions, dict):
            raise ValueError("Malformed score decisions")

        by_index = {}

        for item in decisions:
            if (
                not isinstance(item, dict)
                or type(item.get("index")) is not int
            ):
                raise ValueError("Malformed claim index")

            idx = item["index"]

            if idx in by_index:
                raise ValueError("Duplicate claim index")

            by_index[idx] = item

        expected = set(range(len(citation_checks)))

        if set(by_index) != expected:
            raise ValueError("Missing or extra claim reviews")

        claims_supported = all(
            by_index[i].get("supported") is True
            for i in expected
        )

        result["claim_support_pass"] = claims_supported

        if claims_supported:
            for key, value in audit["values"].items():
                decision = score_decisions.get(key, {})

                if (
                    value is not None
                    and isinstance(decision, dict)
                    and decision.get("supported") is True
                ):
                    result["accepted_scores"][key] = value
                    result["accepted_masks"][key] = 1

        # A score is admitted only when all cited claims
        # are supported AND that score itself is supported.
        any_score = any(result["accepted_masks"].values())

        result["automated_semantic_pass"] = bool(
            claims_supported and any_score
        )

        result["status"] = (
            "AUTO_JUDGE_APPROVED"
            if result["automated_semantic_pass"]
            else "AUTO_JUDGE_REJECTED"
        )

    except Exception as exc:
        result["status"] = "JUDGE_ERROR"
        result["judge_error"] = repr(exc)

    write_atomic(outfile, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", nargs="+")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(SRC.glob("*.json"))
    work = []

    for p in paths:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            t = r.get("ticker")
            d = r.get("date")
            g = r.get("group")

            if t not in TICKERS or g not in GROUPS:
                continue

            if args.ticker and t not in args.ticker:
                continue

            if not (args.start <= d <= args.end):
                continue

            work.append(p)
        except Exception:
            continue

    if args.max_files > 0:
        work = work[:args.max_files]

    print("[VERIFY FILES]", len(work), flush=True)

    if args.dry_run:
        for p in work[:20]:
            print(p)
        return

    if not work:
        return

    client = ollama.Client(host=HOST, timeout=1200)

    rows = []

    for i, path in enumerate(work, 1):
        print(
            f"[{i}/{len(work)}] {path.name}",
            flush=True,
        )

        result = verify_one(
            client, path, force=args.force
        )

        row = {
            "ticker": result["ticker"],
            "date": result["date"],
            "group": result["group"],
            "status": result["status"],
            "automated_semantic_pass":
                result["automated_semantic_pass"],
            "human_verified": False,
        }

        for key in ("impact", "risk", "confidence"):
            row[key] = result["accepted_scores"][key]
            row[key + "_valid"] = (
                result["accepted_masks"][key]
            )

        rows.append(row)

        pd.DataFrame(rows).to_csv(
            OUT / "verification_summary.csv",
            index=False,
        )

        print(
            "[STATUS]", result["status"],
            "[MASKS]", result["accepted_masks"],
            flush=True,
        )

    print("[SAVED]", OUT / "verification_summary.csv")


if __name__ == "__main__":
    main()
