#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-B7 Paper-grounded Financial Event RAG.

Based on methodological ideas from:
- FActScore: atomic factual claims
- AlignScore: independent factual alignment
- RARR: evidence-guided revision
- Ollama: schema-constrained generation

SEC:
  Historical PIT evidence, strict date cutoff.

KG:
  Historical 1-hop structural evidence only.

IMPORTANT:
  This is an adapted research pipeline, not an
  official reproduction of any single paper.

  --dev-no-alignscore is DIAGNOSTIC ONLY.
  Rows generated without AlignScore are not
  independently verified and must not enter PPO.

  AlignScore thresholds must be calibrated
  against manually labeled financial claims.

  No test data is used.
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import ollama
from pydantic import BaseModel, ConfigDict, Field

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17_unified_llm_rag as v17,
)
from unified_event_memory.finrl_x.aegis_rl import (
    run_v17b2_grounded_rag as b2,
)
from unified_event_memory.finrl_x.aegis_rl import (
    run_v17b4_quote_grounded as b4,
)


VERSION = "v17b7_paper_grounded_1"
TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

RISK_TYPES = [
    "supply_chain", "regulatory", "competition",
    "financial", "operational", "cybersecurity",
    "macroeconomic", "other", "none",
]

IMPACTS = ["positive", "negative", "neutral", "unknown"]
STATUSES = [
    "realized", "planned", "hypothetical",
    "historical", "unknown",
]
SEVERITIES = ["low", "medium", "high", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedClaim(StrictModel):
    quote_id: str
    claim: str


class Extraction(StrictModel):
    claims: list[ExtractedClaim]


class Revision(StrictModel):
    revised_claim: str


class Classification(StrictModel):
    risk_category: Literal[
        "supply_chain", "regulatory",
        "competition", "financial",
        "operational", "cybersecurity",
        "macroeconomic", "other", "none"
    ]
    risk_severity: Literal[
        "low", "medium", "high", "unknown"
    ]
    potential_impact: Literal[
        "positive", "negative", "neutral", "unknown"
    ]
    realized_impact: Literal[
        "positive", "negative", "neutral", "unknown"
    ]
    event_status: Literal[
        "realized", "planned", "hypothetical",
        "historical", "unknown"
    ]
    rationale: str = Field(min_length=5)


EXTRACT_INSTRUCTION = """
Extract up to three DISTINCT atomic financial
facts from the provided SEC passages.

For each fact select an actual quote_id and write
one short claim directly supported by that quote.

A claim must describe something concrete in the
source, not a generic statement or an instruction.

Do not repeat phrases from this instruction.

Do not infer facts from external knowledge.
Do not predict future stock prices.
A possible future risk is not a realized event.
Return no claims if evidence is insufficient.
"""

REVISION_INSTRUCTION = """
An extracted financial claim may be unsupported
by its SEC source passage.

Rewrite it to ONE smaller, directly supported fact
using ONLY the supplied quote.

Keep its original subject when possible.
Remove unsupported places, amounts, outcomes,
and causal conclusions.

If nothing can be supported, return an empty
revised_claim.

Do not invent new facts.
This revision will be independently checked again.
"""

CLASSIFICATION_INSTRUCTION = """
Classify ONE externally verified SEC atomic fact.

Categories refer to financial disclosure topics,
not to predicted stock returns.

potential_impact concerns hypothetical or planned
business/financial consequences.

realized_impact concerns explicitly documented
realized outcomes.

For hypothetical risks with no realized evidence:
  event_status = hypothetical
  realized_impact = unknown

For vague historical descriptions, use unknown
instead of inventing a current event.

Do not infer a realized outcome from "may",
"could", "might", "if", or "would" conditional risks.

risk_severity = unknown when magnitude or
seriousness cannot be established from the quote.

rationale must explain the classification using
only the quoted SEC evidence.

Do not mention knowledge graph semantics.
"""


def emit(*args):
    print(*args, flush=True)


def jdump(obj):
    return json.dumps(
        obj, sort_keys=True,
        ensure_ascii=False, default=str,
    )


def stable_key(obj):
    return hashlib.sha256(
        jdump({"version": VERSION, "data": obj})
        .encode("utf-8")
    ).hexdigest()[:24]


def atomic_write(path, contents):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(contents, encoding="utf-8")
    temp.replace(path)


class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS items(
                k TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
        """)
        self.db.commit()

    def get(self, key):
        row = self.db.execute(
            "SELECT payload FROM items WHERE k=?",
            (key,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO items(k,payload) "
            "VALUES (?,?)",
            (key, jdump(value)),
        )
        self.db.commit()

    def close(self):
        self.db.close()


def ask(client, model, schema, instruction, data):
    reply = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": jdump(data),
            },
        ],
        format=schema.model_json_schema(),
        options={
            "temperature": 0,
            "num_predict": 700,
        },
    )

    return schema.model_validate_json(
        reply.message.content
    )


def is_placeholder(text):
    x = " ".join(text.lower().split())
    bad = [
        "one directly supported atomic fact",
        "a supported atomic fact",
        "one short claim",
        "the company has financial information",
    ]
    return any(z in x for z in bad)


def check_entities(quote, claim):
    """
    Conservative deterministic guard for
    hallucinated numbers and selected locations.

    Not a full semantic factuality validator.
    """
    number_pattern = (
        r"(?<![A-Za-z])"
        r"[$€£]?\d[\d,.]*(?:\s?%|\s?billion|\s?million)?"
    )

    qnums = {
        re.sub(r"\s+", "", x.lower())
        for x in re.findall(
            number_pattern, quote,
            flags=re.IGNORECASE,
        )
    }

    cnums = {
        re.sub(r"\s+", "", x.lower())
        for x in re.findall(
            number_pattern, claim,
            flags=re.IGNORECASE,
        )
    }

    if not cnums.issubset(qnums):
        return False, "UNSUPPORTED_NUMBER"

    locations = [
        "australia", "singapore", "ireland",
        "united kingdom", "uk", "india",
        "china", "japan", "germany",
        "france", "canada", "brazil",
    ]

    low_quote = quote.lower()
    low_claim = claim.lower()

    for place in locations:
        pattern = r"\b" + re.escape(place) + r"\b"
        if (
            re.search(pattern, low_claim)
            and not re.search(pattern, low_quote)
        ):
            return False, "UNSUPPORTED_LOCATION:" + place

    if is_placeholder(claim):
        return False, "PROMPT_PLACEHOLDER"

    if not 12 <= len(claim.strip()) <= 450:
        return False, "INVALID_CLAIM_LENGTH"

    return True, "PASS"


class EvidenceAligner:
    def __init__(self, args):
        self.available = False
        self.mode = "UNVERIFIED_DEV"
        self.scorer = None
        self.threshold = args.align_threshold

        if args.dev_no_alignscore:
            return

        checkpoint = Path(args.align_ckpt)

        if not checkpoint.is_file():
            raise RuntimeError(
                "AlignScore checkpoint missing: "
                + str(checkpoint)
            )

        try:
            from alignscore import AlignScore
        except ImportError as exc:
            raise RuntimeError(
                "AlignScore is not installed in the "
                "current Python environment"
            ) from exc

        self.scorer = AlignScore(
            model="roberta-base",
            batch_size=8,
            device=args.align_device,
            ckpt_path=str(checkpoint),
            evaluation_mode="nli_sp",
        )

        self.available = True
        self.mode = "ALIGNSCORE_BASE"

    def evaluate(self, quote, claim):
        if not self.available:
            return {
                "alignment_score": None,
                "independent_verified": False,
                "verification_status": "UNVERIFIED_DEV",
            }

        scores = self.scorer.score(
            contexts=[quote],
            claims=[claim],
        )

        score = float(scores[0])

        if not math.isfinite(score):
            raise ValueError(
                "Nonfinite AlignScore"
            )

        return {
            "alignment_score": score,
            "independent_verified": (
                score >= self.threshold
            ),
            "verification_status": (
                "ALIGNSCORE_PASS"
                if score >= self.threshold
                else "ALIGNSCORE_LOW"
            ),
        }


def extract(client, args, hits):
    bank, candidates = b4.build_quote_bank(
        hits, width=220, overlap=50,
    )

    if not candidates:
        return [], bank

    reply = ask(
        client, args.llm,
        Extraction,
        EXTRACT_INSTRUCTION,
        {
            "candidate_quotes": [
                {
                    "quote_id": x["quote_id"],
                    "text": x["quote"],
                    "topic": x["topic"],
                }
                for x in candidates
            ]
        },
    )

    if len(reply.claims) > 3:
        raise ValueError(
            "More than three extracted claims"
        )

    seen = set()
    claims = []

    for i, item in enumerate(reply.claims, 1):
        if item.quote_id not in bank:
            raise ValueError(
                "Unknown quote_id: " + item.quote_id
            )

        if item.quote_id in seen:
            raise ValueError(
                "Duplicate quote ID"
            )
        seen.add(item.quote_id)

        source = bank[item.quote_id]

        claims.append({
            "claim_id": f"C{i:02d}",
            "quote_id": item.quote_id,
            "sec_evidence_id": (
                source["evidence_id"]
            ),
            "quote": source["quote"],
            "original_claim": item.claim,
        })

    return claims, bank


def verify_and_revise(client, args, aligner, claim):
    quote = claim["quote"]
    original = claim["original_claim"]

    def judge(text):
        passed, reason = check_entities(
            quote, text
        )

        if not passed:
            return {
                "claim": text,
                "deterministic_ok": False,
                "deterministic_reason": reason,
                "alignment_score": None,
                "independent_verified": False,
                "verification_status": reason,
            }

        result = aligner.evaluate(
            quote, text
        )
        return {
            "claim": text,
            "deterministic_ok": True,
            "deterministic_reason": "PASS",
            **result,
        }

    first = judge(original)

    if (
        first["independent_verified"]
        or (
            args.dev_no_alignscore
            and first["deterministic_ok"]
        )
    ):
        return {
            "original_check": first,
            "revision_check": None,
            "final_check": first,
            "was_revised": False,
        }

    # RARR-inspired single evidence-guided revision.
    revised = ask(
        client, args.llm,
        Revision,
        REVISION_INSTRUCTION,
        {
            "quote": quote,
            "original_claim": original,
            "failure_reason": (
                first["verification_status"]
            ),
        },
    ).revised_claim.strip()

    second = (
        judge(revised)
        if revised
        else {
            "claim": "",
            "deterministic_ok": False,
            "deterministic_reason": "EMPTY_REVISION",
            "alignment_score": None,
            "independent_verified": False,
            "verification_status": "EMPTY_REVISION",
        }
    )

    return {
        "original_check": first,
        "revision_check": second,
        "final_check": second,
        "was_revised": True,
    }


def validate_temporal_classification(cls, claim):
    quote = claim["quote"].lower()
    text = claim["final_claim"].lower()

    # A deterministic safety check, not a
    # complete linguistic event-time parser.
    conditionals = re.search(
        r"\b(if|may|might|could|would|"
        r"potentially|in the event of)\b",
        quote,
    )

    # Only guard obvious conditional-only cases.
    # Do not override an explicit realized event.
    realized_cues = re.search(
        r"\b(was completed|were completed|"
        r"has occurred|have occurred|"
        r"occurred|we settled|was settled|"
        r"were settled|we paid|has paid|"
        r"completed the acquisition)\b",
        quote,
    )

    if (
        conditionals
        and not realized_cues
        and cls.event_status == "realized"
    ):
        raise ValueError(
            "HYPOTHETICAL_AS_REALIZED"
        )

    if (
        cls.event_status == "hypothetical"
        and cls.realized_impact != "unknown"
    ):
        raise ValueError(
            "HYPOTHETICAL_REALIZED_IMPACT"
        )

    if (
        cls.event_status in [
            "planned", "unknown"
        ]
        and cls.realized_impact not in [
            "unknown", "neutral"
        ]
    ):
        raise ValueError(
            "UNJUSTIFIED_REALIZED_IMPACT"
        )


def classify_one(client, args, claim):
    classification = ask(
        client, args.llm,
        Classification,
        CLASSIFICATION_INSTRUCTION,
        {
            "quote": claim["quote"],
            "verified_claim": claim["final_claim"],
            "decision_date": claim["date"],
        },
    )

    validate_temporal_classification(
        classification, claim
    )

    return classification.model_dump()


def process_case(
    client, args, aligner,
    day, ticker, sec_hits, kg_hits,
):
    kg = b2.kg_structural_summary(
        kg_hits, day
    )

    if not sec_hits:
        return {
            "status": "NO_SEC_EVIDENCE",
            "claims": [],
            "kg": kg,
        }

    claims, _ = extract(
        client, args, sec_hits
    )

    if not claims:
        return {
            "status": "ABSTAINED",
            "claims": [],
            "kg": kg,
        }

    verified = []

    for claim in claims:
        decision = verify_and_revise(
            client, args, aligner, claim
        )

        final = decision["final_check"]

        record = {
            **claim,
            **decision,
            "date": str(day.date()),
            "final_claim": final["claim"],
            "classification": None,
        }

        # Independent evidence check is mandatory
        # for a research-ready output.
        if final["independent_verified"]:
            try:
                record["classification"] = classify_one(
                    client, args, record
                )
            except Exception as exc:
                record["classification_error"] = (
                    str(exc)[:300]
                )

        elif (
            args.dev_no_alignscore
            and final["deterministic_ok"]
        ):
            # Useful only for testing JSON/temporal
            # processing. Never a verified signal.
            try:
                record["classification"] = classify_one(
                    client, args, record
                )
            except Exception as exc:
                record["classification_error"] = (
                    str(exc)[:300]
                )

        verified.append(record)

    good = [
        x for x in verified
        if (
            x["final_check"]["independent_verified"]
            and x["classification"] is not None
        )
    ]

    dev = [
        x for x in verified
        if x["classification"] is not None
    ]

    if args.dev_no_alignscore:
        status = (
            "UNVERIFIED_DEV"
            if dev else "NO_USABLE_CLAIMS"
        )

    else:
        status = (
            "VERIFIED"
            if good else "NO_VERIFIED_CLAIMS"
        )

    return {
        "status": status,
        "claims": verified,
        "kg": kg,
    }


def row_features(day, ticker, case, key):
    claims = case["claims"]
    accepted = [
        c for c in claims
        if (
            c.get("classification") is not None
            and (
                c["final_check"][
                    "independent_verified"
                ]
                or case["status"] == "UNVERIFIED_DEV"
            )
        )
    ]

    kg = case["kg"]

    row = {
        "date": str(day.date()),
        "ticker": ticker,
        "status": case["status"],
        "cache_key": key,
        "claim_count": len(claims),
        "verified_claim_count": sum(
            x["final_check"][
                "independent_verified"
            ]
            for x in claims
        ),
        "classified_claim_count": len(accepted),
        "revised_claim_count": sum(
            x["was_revised"] for x in claims
        ),
        "has_verified_sec_signal": int(
            case["status"] == "VERIFIED"
        ),
        "kg_edge_count": kg["count"],
        "kg_relation_types": len(
            kg["relation_counts"]
        ),
        "kg_in": kg[
            "direction_counts"
        ]["in"],
        "kg_out": kg[
            "direction_counts"
        ]["out"],
    }

    for risk in RISK_TYPES:
        row["risk_" + risk] = sum(
            c["classification"]["risk_category"] == risk
            for c in accepted
        )

    for impact in IMPACTS:
        row["potential_" + impact] = sum(
            c["classification"]["potential_impact"] == impact
            for c in accepted
        )
        row["realized_" + impact] = sum(
            c["classification"]["realized_impact"] == impact
            for c in accepted
        )

    for status in STATUSES:
        row["event_" + status] = sum(
            c["classification"]["event_status"] == status
            for c in accepted
        )

    for severity in SEVERITIES:
        row["severity_" + severity] = sum(
            c["classification"]["risk_severity"] == severity
            for c in accepted
        )

    return row


def get_dates(args):
    if args.dates:
        dates = pd.DatetimeIndex(
            pd.to_datetime(args.dates)
        )
    else:
        raw = pd.read_csv(args.dates_csv)
        if "date" not in raw:
            raise ValueError(
                "dates-csv needs date column"
            )
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                raw["date"],
                errors="coerce",
            ).dropna()
        )

    dates = (
        dates.normalize().unique().sort_values()
    )
    dates = dates[
        (dates >= pd.Timestamp(args.start_date))
        & (dates <= pd.Timestamp(args.end_date))
    ]

    if not len(dates):
        raise ValueError("No matching dates")

    if (
        dates.min() < pd.Timestamp("2018-01-01")
        or dates.max() >= pd.Timestamp("2022-01-01")
    ):
        raise ValueError(
            "2022-2023 sealed test is forbidden"
        )

    return dates


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        )
    )
    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        )
    )
    p.add_argument(
        "--dates-csv",
        default=(
            "results/aegis_rl_v1/"
            "v16_dual_rag_ppo/"
            "kg_retrieval_audit.csv"
        )
    )
    p.add_argument("--dates", nargs="+")
    p.add_argument(
        "--start-date", default="2020-01-01"
    )
    p.add_argument(
        "--end-date", default="2020-12-31"
    )
    p.add_argument(
        "--tickers", nargs="+",
        default=TICKERS
    )
    p.add_argument(
        "--sample-count", type=int, default=20
    )
    p.add_argument(
        "--sample-seed", type=int, default=42
    )
    p.add_argument(
        "--sec-top-k", type=int, default=5
    )
    p.add_argument(
        "--kg-top-k", type=int, default=12
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434"
    )
    p.add_argument(
        "--llm", default="qwen2.5:3b"
    )
    p.add_argument(
        "--align-ckpt",
        default=(
            "models/alignscore/"
            "AlignScore-base.ckpt"
        )
    )
    p.add_argument(
        "--align-device", default="cpu"
    )
    p.add_argument(
        "--align-threshold",
        type=float, default=0.5
    )
    p.add_argument(
        "--dev-no-alignscore",
        action="store_true"
    )
    p.add_argument(
        "--retries", type=int, default=2
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v17b7_paper_20"
        )
    )

    args = p.parse_args()

    if not set(args.tickers).issubset(
        TICKERS
    ):
        raise ValueError("Unexpected ticker")

    if (
        args.sample_count < 1
        or args.retries < 1
        or args.sec_top_k < 1
        or args.kg_top_k < 1
    ):
        raise ValueError("Invalid numeric settings")

    if not 0 <= args.align_threshold <= 1:
        raise ValueError(
            "AlignScore threshold must be [0,1]"
        )

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    sec = v17.load_sec(args.pit_sec)
    kg = v17.load_kg(args.temporal_kg)
    dates = get_dates(args)

    pairs = [
        (pd.Timestamp(d), t)
        for d in dates
        for t in args.tickers
    ]

    rng = np.random.default_rng(
        args.sample_seed
    )
    n = min(args.sample_count, len(pairs))
    indices = rng.choice(
        len(pairs), n, replace=False
    )
    selected = sorted(
        [pairs[int(i)] for i in indices],
        key=lambda p: (p[0], p[1])
    )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    available = {
        m.model
        for m in client.list().models
    }
    if args.llm not in available:
        raise RuntimeError(
            f"Ollama model missing: {args.llm}"
        )

    aligner = EvidenceAligner(args)

    cache = Cache(
        args.out / "cache.sqlite"
    )

    rows = []
    audit = []
    cache_hits = 0

    # Include verification settings in cache identity.
    config = {
        "model": args.llm,
        "host": args.ollama_host,
        "aligner_mode": aligner.mode,
        "threshold": args.align_threshold,
        "checkpoint": args.align_ckpt,
        "device": args.align_device,
    }

    try:
        for day, ticker in selected:
            sec_hits = v17.sec_retrieve(
                sec, ticker, day, args.sec_top_k
            )
            kg_hits = v17.kg_retrieve(
                kg, ticker, day, args.kg_top_k
            )

            for h in sec_hits:
                assert pd.Timestamp(
                    h["available_from"]
                ) < day, "SEC PIT violation"

            for h in kg_hits:
                assert pd.Timestamp(
                    h["first_observed"]
                ) < day, "KG temporal violation"

            key = stable_key({
                "date": str(day.date()),
                "ticker": ticker,
                "config": config,
                "sec": sec_hits,
                "kg": kg_hits,
            })

            case = cache.get(key)

            if case is not None:
                cache_hits += 1
            else:
                error = None

                for attempt in range(
                    args.retries
                ):
                    try:
                        case = process_case(
                            client, args, aligner,
                            day, ticker,
                            sec_hits, kg_hits
                        )
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        emit(
                            "[RETRY]",
                            day.date(), ticker,
                            attempt + 1,
                            str(exc)[:250],
                        )
                        time.sleep(min(2 ** attempt, 8))

                if error is not None:
                    case = {
                        "status": "ERROR",
                        "claims": [],
                        "kg": b2.kg_structural_summary(
                            kg_hits, day
                        ),
                        "error": str(error),
                    }
                else:
                    cache.put(key, case)

            row = row_features(
                day, ticker, case, key
            )

            rows.append(row)
            audit.append({
                **row,
                "case": case,
                "sec_hits": sec_hits,
                "kg_hits": kg_hits,
            })

            # Save progress after every case.
            pd.DataFrame(rows).to_csv(
                args.out / "signals.csv",
                index=False,
            )

            atomic_write(
                args.out / "audit.jsonl",
                "".join(
                    jdump(x) + "\n"
                    for x in audit
                ),
            )

            emit(
                "[B7]", row["date"], ticker,
                row["status"],
                "verified=",
                row["verified_claim_count"],
                "classified=",
                row["classified_claim_count"],
            )

    finally:
        cache.close()

    statuses = Counter(
        row["status"] for row in rows
    )

    manifest = {
        "version": VERSION,
        "model": args.llm,
        "aligner": aligner.mode,
        "threshold_uncalibrated": (
            args.align_threshold
        ),
        "cases": len(rows),
        "statuses": dict(statuses),
        "cache_hits": cache_hits,
        "test_2022_2023_used": False,
        "paper_inspirations": [
            "FActScore",
            "AlignScore",
            "RARR",
            "Ollama JSON schema",
        ],
        "limitations": [
            "AlignScore threshold must be validated "
            "against manually labeled SEC claims.",
            "Automated factual consistency is not "
            "financial correctness ground truth.",
            "KG first-observed dates are not "
            "certified public disclosure times.",
            "Opaque KG relations are structural only.",
            "No PPO training or live trading.",
        ],
    }

    atomic_write(
        args.out / "manifest.json",
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
    )

    emit()
    emit("========== V17-B7 SUMMARY ==========")
    emit("[CASES]", len(rows))
    emit("[STATUSES]", dict(statuses))
    emit("[ALIGNER]", aligner.mode)
    emit("[CACHE HITS]", cache_hits)
    emit("[OUTPUT]", args.out)
    emit("V17-B7: AUDIT COMPLETE")
    if args.dev_no_alignscore:
        emit(
            "WARNING: UNVERIFIED DEVELOPMENT RUN"
        )


if __name__ == "__main__":
    main()
