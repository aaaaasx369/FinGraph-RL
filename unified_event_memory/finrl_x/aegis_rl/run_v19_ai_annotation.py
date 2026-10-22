#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import ollama

ROOT = Path("results/aegis_rl_v1")
AUDIT = ROOT / "v18_2_audit"
OUT = ROOT / "v19_final/ai_annotation"
OUT.mkdir(parents=True, exist_ok=True)

def ask(client, model, task, evidence, claim):
    prompt = {
        "task": task,
        "evidence": evidence,
        "claim": claim,
        "instructions": [
            "Use only provided evidence.",
            "Do not infer unstated business relationships.",
            "Distinguish realized outcomes from hypothetical risks.",
            "Unknown is not neutral.",
            "Return JSON with supported, occurrence, outcome, potential, realized, confidence, reason.",
            "If evidence is insufficient, abstain.",
        ],
    }

    response = client.chat(
        model=model,
        format="json",
        options={"temperature": 0},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an independent financial "
                    "evidence annotation judge. "
                    "Do not assume the claim is true."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    prompt,
                    ensure_ascii=False
                ),
            },
        ],
    )

    return json.loads(response['message']['content'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    client = ollama.Client(
        host=os.getenv(
            "OLLAMA_HOST",
            "http://140.113.87.164:11434"
        ),
        timeout=180
    )

    grounding = pd.read_csv(
        AUDIT / "grounding_review.csv"
    ).fillna("")

    events = pd.read_csv(
        AUDIT / "event_review.csv"
    ).fillna("")

    original = ROOT / (
        "v17_joint_rag_ppo_smoke/"
        "joint_rag_cache.jsonl"
    )

    cache = {}

    for line in original.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[(row["date"], row["ticker"])] = row

    grounding_rows = []
    event_rows = []

    for i, (_, row) in enumerate(grounding.iterrows()):
        if args.limit and i >= args.limit:
            break

        key = (row["date"], row["ticker"])
        source = cache.get(key, {})

        matches = [
            h for h in source.get("sec_evidence", [])
            if str(h.get("id")) == str(row["sec_evidence_id"])
        ]

        evidence = "\n\n".join(
            h.get("text", "")
            for h in matches
        )

        if not evidence:
            result = {
                "supported": "abstain",
                "confidence": 0,
                "reason": "Original SEC evidence unavailable",
            }
        else:
            result = ask(
                client,
                args.model,
                "SEC_CLAIM_SUPPORT",
                evidence,
                row["final_claim"],
            )

        grounding_rows.append({
            **row.to_dict(),
            "ai_supported": result.get("supported"),
            "ai_confidence": result.get("confidence"),
            "ai_reason": result.get("reason"),
        })

        print(
            "SEC", i + 1, "/", len(grounding),
            row["ticker"],
            result.get("supported"),
            flush=True,
        )

    for i, (_, row) in enumerate(events.iterrows()):
        if args.limit and i >= args.limit:
            break

        key = (row["date"], row["ticker"])
        source = cache.get(key, {})

        evidence = "\n\n".join(
            h.get("text", "")
            for h in source.get("sec_evidence", [])
        )

        result = ask(
            client,
            args.model,
            "EVENT_OUTCOME_CLASSIFICATION",
            evidence,
            row["final_claim"],
        ) if evidence else {
            "occurrence": "unknown",
            "outcome": "unknown",
            "potential": "unknown",
            "realized": "unknown",
            "confidence": 0,
            "reason": "SEC evidence unavailable",
        }

        event_rows.append({
            **row.to_dict(),
            "ai_occurrence": result.get("occurrence"),
            "ai_outcome": result.get("outcome"),
            "ai_potential": result.get("potential"),
            "ai_realized": result.get("realized"),
            "ai_confidence": result.get("confidence"),
            "ai_reason": result.get("reason"),
        })

        print(
            "EVENT", i + 1, "/", len(events),
            row["ticker"],
            flush=True,
        )

    pd.DataFrame(grounding_rows).to_csv(
        OUT / "grounding_ai_labeled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(event_rows).to_csv(
        OUT / "event_ai_labeled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAI ANNOTATION COMPLETE")
    print("Output:", OUT)

if __name__ == "__main__":
    main()
