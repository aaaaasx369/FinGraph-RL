#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-B4: Deterministic SEC quote grounding.

Fixes:
1. LLM selects quote IDs, not reconstructed quotes.
2. Original SEC text is restored by Python.
3. SEC numeric scores survive fusion.
4. Opaque KG relations remain structural only.
5. Zero-claim results are quarantined.
6. New cache version avoids previous zero outputs.

Does not validate semantic entailment automatically.
Does not train PPO.
"""

from collections import defaultdict

import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17b2_grounded_rag as b2,
)

b2.VERSION = "v17b4_quote_grounded_1"


SEC_PROMPT = """
You are analyzing historical SEC financial disclosures.

You receive candidate passages extracted directly
from SEC filing excerpts. Each passage has a quote_id.

Your task:
1. Identify relevant financial facts.
2. Select 1 to 3 supporting quote_ids.
3. Explain their financial implications cautiously.
4. Give impact, risk, confidence scores.

Return JSON only:

{
  "impact": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "claims": [
    {
      "quote_id": "Q0001",
      "interpretation": "A concise financial interpretation"
    }
  ]
}

IMPORTANT:
- Never generate or rewrite a quotation.
- Cite a quote_id exactly as supplied.
- Do not invent events, figures, or relationships.
- A possible risk is NOT an event that occurred.
- Do not use KG information.
- Do not use future information.
- Impact is not a stock-return forecast.
- A neutral impact is allowed.
- A disclosed risk may justify nonzero risk
  without implying negative future returns.
- Prefer concrete supported facts.
- If no relevant fact exists, use empty claims
  and zero scores.

All financial interpretations must be supported
by the corresponding selected passage.
"""


def build_quote_bank(sec_hits, width=220, overlap=50):
    """
    Extract exact, immutable text windows.

    The model chooses IDs. It cannot alter the
    original wording of the cited quotations.
    """
    bank = {}
    candidates = []

    counter = 0

    for hit in sec_hits:
        text = hit["text"]

        if not text.strip():
            continue

        start = 0
        step = width - overlap

        while start < len(text):
            end = min(
                start + width,
                len(text),
            )

            # Avoid very short trailing fragments.
            if (
                len(text) - end < 45
                and end < len(text)
            ):
                end = len(text)

            quote = text[start:end]

            if len(quote.strip()) >= 35:
                counter += 1
                qid = f"Q{counter:04d}"

                record = {
                    "quote_id": qid,
                    "evidence_id": hit["id"],
                    "quote": quote,
                    "available_from": hit[
                        "available_from"
                    ],
                    "form": hit["form"],
                    "topic": hit["topic"],
                }

                bank[qid] = record
                candidates.append(record)

            if end >= len(text):
                break

            start += step

    return bank, candidates


def validate_selected_claims(raw, bank, sec_hits):
    score = b2.numbers(raw)

    claims = raw.get("claims")

    if not isinstance(claims, list):
        raise ValueError(
            "claims must be a list"
        )

    if len(claims) == 0:
        raise ValueError(
            "No SEC claims extracted from "
            "nonempty SEC evidence"
        )

    if len(claims) > 5:
        raise ValueError(
            "Too many SEC claims"
        )

    verified = []
    seen = set()

    for item in claims:
        if not isinstance(item, dict):
            raise ValueError(
                "Claim is not a JSON object"
            )

        qid = item.get("quote_id")

        if qid not in bank:
            raise ValueError(
                f"Unknown quote_id: {qid}"
            )

        if qid in seen:
            raise ValueError(
                f"Duplicate quote_id: {qid}"
            )

        seen.add(qid)

        interpretation = str(
            item.get("interpretation", "")
        ).strip()

        if not interpretation:
            raise ValueError(
                "Empty interpretation"
            )

        if len(interpretation) > 600:
            raise ValueError(
                "Interpretation too long"
            )

        selected = bank[qid]

        verified.append({
            "evidence_id": selected[
                "evidence_id"
            ],
            "quote": selected["quote"],
            "interpretation": interpretation,
            "quote_id": qid,
        })

    # Reuse the existing exact-match validator.
    # The quotation now comes from Python,
    # not from text reconstructed by Qwen.
    grounded = b2.validate_sec(
        {
            **score,
            "claims": verified,
        },
        sec_hits,
    )

    return grounded


def run_one(
    client,
    args,
    ticker,
    day,
    sec_hits,
    kg_hits,
):
    """
    SEC = Qwen-grounded numeric assessment.
    KG  = deterministic structural representation.

    Fusion is conservative: KG structure is included
    separately and cannot overwrite SEC scores.
    """

    kg_summary = b2.kg_structural_summary(
        kg_hits,
        day,
    )

    if sec_hits:
        bank, candidates = build_quote_bank(
            sec_hits
        )

        if not candidates:
            raise ValueError(
                "SEC retrieval returned no quote candidates"
            )

        raw = b2.ollama_json(
            client,
            args.llm,
            SEC_PROMPT,
            {
                "ticker": ticker,
                "decision_date": str(
                    day.date()
                ),
                "candidate_quotes": [
                    {
                        "quote_id": x["quote_id"],
                        "evidence_id": x[
                            "evidence_id"
                        ],
                        "text": x["quote"],
                        "form": x["form"],
                        "topic": x["topic"],
                    }
                    for x in candidates
                ],
            },
        )

        sec = validate_selected_claims(
            raw,
            bank,
            sec_hits,
        )

    else:
        sec = {
            "impact": 0.0,
            "risk": 0.0,
            "confidence": 0.0,
            "claims": [],
            "unsupported": [],
        }

    sec_ids = list(dict.fromkeys(
        claim["evidence_id"]
        for claim in sec["claims"]
    ))

    # Conservative deterministic fusion.
    #
    # Do not permit a second LLM call to erase
    # already verified SEC-grounded scores.
    #
    # KG graph evidence is retained independently.
    # No KG edge is falsely cited as proof of an
    # economic event.
    fusion = {
        "impact": sec["impact"],
        "risk": sec["risk"],
        "confidence": sec["confidence"],
        "sec_evidence_ids": sec_ids,
        "kg_evidence_ids": [],
        "kg_role": "structural_only",
    }

    fusion = b2.validate_fusion(
        fusion,
        sec,
        sec_hits,
        kg_hits,
    )

    return {
        "sec_analysis": sec,
        "kg_structure": kg_summary,
        "fusion": fusion,
        "status": "SUPPORTED",
    }


# Patch the imported V17-B2 orchestration.
# Its retrieval, temporal audits, cache management,
# unsupported quarantine and CSV writers remain active.
b2.run_one = run_one


if __name__ == "__main__":
    b2.main()
