#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from alignscore import AlignScore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--checkpoint",
        default="models/alignscore/AlignScore-base.ckpt"
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Invalid threshold")

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(s)
        for s in src.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]

    scorer = AlignScore(
        model="roberta-base",
        batch_size=1,
        device=args.device,
        ckpt_path=args.checkpoint,
        evaluation_mode="nli_sp",
    )

    outputs = []

    for row in rows:
        result = row.get("result") or {}
        sec_ids = set(result.get("sec_evidence_ids") or [])
        kg_ids = set(result.get("kg_evidence_ids") or [])

        sec = [
            x for x in row.get("sec_evidence", [])
            if x.get("id") in sec_ids
        ]

        kg = [
            x for x in row.get("kg_evidence", [])
            if x.get("id") in kg_ids
        ]

        reasoning = str(result.get("reasoning") or "")
        issues = []

        if row.get("status") != "GENERATED":
            issues.append("GENERATION_FAILED")

        if not sec:
            issues.append("NO_CITED_SEC_EVIDENCE")

        if sec_ids != {x["id"] for x in sec}:
            issues.append("MISSING_SEC_CITATION_SOURCE")

        if kg_ids != {x["id"] for x in kg}:
            issues.append("MISSING_KG_CITATION_SOURCE")

        opaque = any(
            str(x.get("relation", "")).startswith("FINDKG_REL:")
            or not x.get("semantic_label_verified", False)
            for x in kg
        )

        # KG economic semantics are not established.
        # Do not treat a citation ID as evidence that
        # a supply, customer, competitive, or acquisition
        # relation has a known meaning.
        kg_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", reasoning)
            if re.search(
                r"\b(KG|knowledge graph|graph evidence|"
                r"graph relations?|graph suggests)\b",
                s,
                flags=re.I,
            )
        ]

        if opaque and kg_sentences:
            issues.append("OPAQUE_KG_SEMANTIC_INFERENCE")

        sec_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", reasoning)
            if re.search(
                r"\b(SEC|filing|filings|disclosure|"
                r"report|reports)\b",
                s,
                flags=re.I,
            )
        ]

        evidence_text = " ".join(
            str(x.get("text", ""))
            for x in sec
        )

        score_records = []

        if not sec_sentences:
            issues.append("NO_EXPLICIT_SEC_CLAIM_TO_CHECK")

        if sec and sec_sentences:
            for sentence in sec_sentences:
                try:
                    scores = scorer.score(
                        contexts=[evidence_text],
                        claims=[sentence],
                    )
                    value = float(scores[0])
                except Exception as exc:
                    issues.append(
                        "ALIGNSCORE_ERROR:" + str(exc)[:120]
                    )
                    continue

                score_records.append({
                    "claim": sentence,
                    "score": value,
                    "passes_threshold": value >= args.threshold,
                })

                if value < args.threshold:
                    issues.append("LOW_SEC_ALIGNMENT")

        status = (
            "AUTO_GATE_PASS"
            if not issues
            else "REVIEW_REQUIRED"
        )

        outputs.append({
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "status": status,
            "issues": sorted(set(issues)),
            "sec_citation_count": len(sec),
            "kg_citation_count": len(kg),
            "opaque_kg_cited": opaque,
            "sec_sentence_scores": score_records,
            "impact": result.get("impact"),
            "risk": result.get("risk"),
            "confidence": result.get("confidence"),
            "reasoning": reasoning,
        })

        print(
            row.get("date"),
            row.get("ticker"),
            status,
            sorted(set(issues)),
            flush=True,
        )

    dst.write_text(
        "".join(
            json.dumps(x, ensure_ascii=False) + "\n"
            for x in outputs
        ),
        encoding="utf-8",
    )

    print("\n========== JOINT GROUNDING AUDIT ==========")
    print("Cases:", len(outputs))
    print(
        "Statuses:",
        dict(Counter(x["status"] for x in outputs))
    )

    issues = Counter(
        issue
        for x in outputs
        for issue in x["issues"]
    )

    print("Issues:", dict(issues))
    print("Output:", dst)


if __name__ == "__main__":
    main()
