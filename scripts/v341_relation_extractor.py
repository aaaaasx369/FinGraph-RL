#!/usr/bin/env python3
"""
V34.1: SEC competitor candidate generation + Ollama verification.

Commands:
  build  - build rule-based relation candidates
  audit  - verify evidence, offsets and candidate entities
  label  - verify candidate groups with Ollama; resumable
  report - show saved results

This is SILVER extraction, NOT human-verified ground truth.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v341"
SOURCE = ROOT / "outputs/v34/relation_evidence.csv"
ALIASES = ROOT / "outputs/v34/entity_aliases.csv"

ISSUERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

RELATION = "COMPETES_WITH"

# Explicit issuer-oriented competition list triggers.
# Do not interpret general competitive pressure as an edge.
TRIGGERS = [
    re.compile(
        r"\bour\s+(?:current\s+)?competitors\s+include\b",
        re.I
    ),
    re.compile(
        r"\bcompetitors\s+to\s+[A-Za-z0-9][A-Za-z0-9 .&/+_-]{0,50}?"
        r"\s+include\b",
        re.I
    ),
    re.compile(
        r"\bour\s+competitors\s+(?:are|include)\b",
        re.I
    ),
    re.compile(
        r"\bwe\s+compete\s+with\b",
        re.I
    ),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence_quote": {"type": "string"},
                    "market_scope": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "supported",
                    "evidence_quote",
                    "market_scope",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

SYSTEM = """
You verify proposed company competitor relationships from SEC filings.

CRITICAL INTERPRETATION RULES:
- The reporting company is the issuer of this SEC filing.
- "Our", "we" and "us" refer to the reporting company.
- A sentence saying "Our current competitors include X" explicitly
  supports issuer COMPETES_WITH X.
- "Competitors to Office include X" in a Microsoft SEC filing
  supports product-scoped Microsoft COMPETES_WITH X.
- The issuer does not need to repeat its name in every sentence.
- "Suppliers of GPUs" listed under "Our competitors include"
  describes competing suppliers, NOT necessarily suppliers to issuer.
- A company appearing merely near a word "competition" is insufficient.
- Never infer SUPPLIES_TO or CUSTOMER_OF from competitor lists.
- Do not use external knowledge.
- Only assess the proposed candidate IDs.
- For supported=true, quote an EXACT contiguous substring from the
  CURRENT SEC SENTENCE. The quote must contain the competitor-list
  statement AND the target company's actual printed name or alias.
- If no suitable exact quote exists, supported=false.
- Do not invent company names, quotations, dates or relationships.
- Output one decision for each candidate ID. JSON only.
"""

def norm(s):
    return " ".join(str(s).split())

def digest(*values):
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()
    ).hexdigest()[:24]

def save_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

def load_aliases():
    a = pd.read_csv(ALIASES).fillna("")
    required = {"alias", "canonical"}
    if not required.issubset(a.columns):
        raise ValueError("Invalid alias dictionary")

    result = {}
    for r in a.itertuples(index=False):
        alias = str(r.alias).strip()
        canonical = str(r.canonical).strip()
        if not alias or not canonical:
            continue

        result.setdefault(canonical, set()).add(alias)

    for ticker, issuer in ISSUERS.items():
        result.setdefault(issuer, set()).add(issuer)

    return {
        k: sorted(v, key=len, reverse=True)
        for k, v in result.items()
    }

def alias_matches(text, aliases):
    matches = []
    for alias in aliases:
        pat = (
            r"(?<![A-Za-z0-9])"
            + re.escape(alias)
            + r"(?![A-Za-z0-9])"
        )
        for m in re.finditer(pat, text, re.I):
            matches.append((m.start(), m.end(), alias))
    return sorted(matches)

def find_trigger(sentence):
    found = []
    for pat in TRIGGERS:
        found.extend(pat.finditer(sentence))
    if not found:
        return None
    return min(found, key=lambda x: x.start())

def candidate_columns():
    return [
        "candidate_id", "evidence_id", "ticker",
        "issuer", "subject", "relation", "object",
        "matched_alias", "form", "accession",
        "filing_date", "available_from",
        "source_sha256", "offset_start", "offset_end",
        "sentence", "context", "trigger_text",
        "trigger_start", "target_start", "retrieval_score",
    ]

def build():
    if not SOURCE.exists():
        raise FileNotFoundError("Run V34 build first")

    src = pd.read_csv(
        SOURCE, dtype={"accession": str}
    ).fillna("")

    names = load_aliases()
    rows = []
    used = set()

    for r in src.itertuples(index=False):
        ticker = str(r.ticker)
        if ticker not in ISSUERS:
            raise ValueError("Unexpected portfolio ticker " + ticker)

        sentence = str(r.sentence)
        trigger = find_trigger(sentence)
        if trigger is None:
            continue

        issuer = ISSUERS[ticker]
        trigger_end = trigger.end()

        # A target must occur AFTER the competition-list trigger.
        # For "we compete with X", the same rule applies.
        # Reject targets excessively far from the trigger.
        search_end = min(len(sentence), trigger_end + 1000)
        trailing = sentence[trigger_end:search_end]

        for canonical, aliases in names.items():
            if canonical == issuer:
                continue

            found = alias_matches(trailing, aliases)
            if not found:
                continue

            # Match the first explicit alias in the relevant list.
            pos, end, alias = found[0]
            target_pos = trigger_end + pos

            # A heading or new section indicates the list may have ended.
            between = sentence[trigger_end:target_pos]
            if re.search(
                r"\b(?:Table of Contents|Patents and Proprietary "
                r"Rights|Risk Factors|Item 1A|Item 7)\b",
                between,
                re.I
            ):
                continue

            cid = digest(
                r.evidence_id, issuer, RELATION, canonical
            )
            if cid in used:
                continue
            used.add(cid)

            rows.append({
                "candidate_id": cid,
                "evidence_id": str(r.evidence_id),
                "ticker": ticker,
                "issuer": issuer,
                "subject": issuer,
                "relation": RELATION,
                "object": canonical,
                "matched_alias": alias,
                "form": str(r.form),
                "accession": str(r.accession),
                "filing_date": str(r.filing_date),
                "available_from": str(r.available_from),
                "source_sha256": str(r.source_sha256),
                "offset_start": int(r.offset_start),
                "offset_end": int(r.offset_end),
                "sentence": sentence,
                "context": str(r.context),
                "trigger_text": trigger.group(),
                "trigger_start": int(trigger.start()),
                "target_start": int(target_pos),
                "retrieval_score": float(r.retrieval_score),
            })

    df = pd.DataFrame(rows, columns=candidate_columns())

    if not df.empty:
        df = df.sort_values(
            ["retrieval_score", "ticker", "filing_date"],
            ascending=[False, True, True]
        )

    df.to_csv(OUT / "candidates.csv", index=False)

    print("\n=== V34.1 CANDIDATE BUILD ===")
    print("V34 evidence:", len(src))
    print("Rule-based relation candidates:", len(df))
    print("Unique SEC evidence:", df["evidence_id"].nunique())
    print("Unique issuer-target pairs:",
          df[["subject", "object"]].drop_duplicates().shape[0])

    if len(df):
        print("\nBY ISSUER")
        print(df["ticker"].value_counts().to_string())
        print("\nEXAMPLES")
        print(
            df[[
                "ticker", "subject", "relation",
                "object", "matched_alias"
            ]].head(12).to_string(index=False)
        )

    print("\nOutput:", OUT / "candidates.csv")

def audit():
    candidates = pd.read_csv(
        OUT / "candidates.csv", dtype={"accession": str}
    )
    evidence = pd.read_csv(
        SOURCE, dtype={"accession": str}
    )

    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs")

    lookup = evidence.set_index("evidence_id")
    aliases = load_aliases()

    for r in candidates.itertuples(index=False):
        if r.ticker not in ISSUERS:
            raise ValueError("Unexpected ticker")
        if r.issuer != ISSUERS[r.ticker]:
            raise ValueError("Issuer mismatch")
        if r.relation != RELATION:
            raise ValueError("Unexpected relation")
        if r.subject == r.object:
            raise ValueError("Self relation")

        if r.evidence_id not in lookup.index:
            raise ValueError("Missing evidence")

        source = lookup.loc[r.evidence_id]
        if str(source["sentence"]) != r.sentence:
            raise ValueError("Sentence provenance mismatch")
        if str(source["source_sha256"]) != r.source_sha256:
            raise ValueError("Hash provenance mismatch")
        if str(source["accession"]) != r.accession:
            raise ValueError("Accession mismatch")

        t = find_trigger(r.sentence)
        if t is None:
            raise ValueError("Missing competitor-list trigger")

        if r.object not in aliases:
            raise ValueError("Unknown target canonical entity")
        matches = alias_matches(
            r.sentence[t.end():], aliases[r.object]
        )
        if not matches:
            raise ValueError("Target absent after trigger")

        if pd.Timestamp(r.available_from) <= pd.Timestamp(r.filing_date):
            raise ValueError("Invalid availability date")

    print("\n=== V34.1 AUDIT PASS ===")
    print("Candidate rows:", len(candidates))
    print("Evidence and issuer provenance: PASS")
    print("Competitor trigger / target ordering: PASS")
    print("Five tradeable assets: PASS")
    print("Semantic accuracy: NOT HUMAN VERIFIED")
    print("Strict PIT approval: NO")

def read_log(path):
    result = {}
    if not path.exists():
        return result

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                result[item["evidence_id"]] = item
    return result

def export_results(records):
    decisions = []
    accepted = []

    for record in records.values():
        if record.get("status") != "ok":
            continue

        for d in record.get("verified_decisions", []):
            decisions.append(d)
            if d["accepted_silver"]:
                accepted.append(d)

    decision_cols = [
        "candidate_id", "evidence_id", "ticker",
        "subject", "relation", "object",
        "filing_date", "available_from", "accession",
        "source_sha256", "offset_start", "offset_end",
        "matched_alias", "supported",
        "evidence_quote", "market_scope", "reason",
        "validation_error", "accepted_silver",
        "model", "label_source", "human_verified",
        "strict_pit_approved",
    ]

    pd.DataFrame(
        decisions, columns=decision_cols
    ).to_csv(OUT / "silver_decisions.csv", index=False)

    pd.DataFrame(
        accepted, columns=decision_cols
    ).to_csv(OUT / "silver_relations.csv", index=False)

def make_batches(limit, max_candidates):
    candidates = pd.read_csv(
        OUT / "candidates.csv", dtype={"accession": str}
    ).fillna("")

    if candidates.empty:
        return [], candidates

    # A single LLM request receives several candidates belonging
    # to the SAME SEC sentence. No cross-document mixing.
    groups = []
    for eid, group in candidates.groupby(
        "evidence_id", sort=False
    ):
        group = group.sort_values(
            "target_start"
        ).head(max_candidates)
        groups.append(group)

    # Highest-scoring evidence first, rather than earliest dates.
    groups.sort(
        key=lambda g: (
            -float(g["retrieval_score"].max()),
            str(g["evidence_id"].iloc[0])
        )
    )

    if limit:
        groups = groups[:limit]

    return groups, candidates

def validate_decision(candidate, decision, aliases):
    errors = []

    if type(decision.get("supported")) is not bool:
        errors.append("INVALID_SUPPORTED_BOOL")

    quote = decision.get("evidence_quote", "")
    if not isinstance(quote, str):
        quote = ""
        errors.append("INVALID_QUOTE_TYPE")

    if not isinstance(decision.get("market_scope"), str):
        errors.append("INVALID_MARKET_SCOPE")
    if not isinstance(decision.get("reason"), str):
        errors.append("INVALID_REASON")

    supported = decision.get("supported") is True
    sentence = candidate["sentence"]
    target = candidate["object"]

    if supported:
        if len(quote.strip()) < 12:
            errors.append("SHORT_QUOTE")
        elif quote not in sentence:
            errors.append("QUOTE_NOT_EXACT")

        # Supported quote must include a target alias.
        alias_hits = alias_matches(
            quote, aliases.get(target, [])
        )
        if not alias_hits:
            errors.append("TARGET_NOT_IN_QUOTE")

        # It must also contain a competition-list anchor.
        trigger = find_trigger(quote)
        if trigger is None:
            errors.append("COMPETITION_ANCHOR_NOT_IN_QUOTE")
        elif alias_hits and alias_hits[0][0] < trigger.end():
            errors.append("TARGET_BEFORE_ANCHOR")

        if candidate["subject"] == target:
            errors.append("SELF_EDGE")

    return (
        supported and not errors,
        "|".join(errors)
    )

def run_label(args):
    from ollama import Client

    groups, candidates = make_batches(
        args.limit, args.max_candidates
    )

    if not groups:
        print("No candidates. Run build first.")
        return

    aliases = load_aliases()
    run_dir = OUT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "responses.jsonl"
    config_path = run_dir / "config.json"

    config = {
        "model": args.model,
        "host": args.host,
        "think": False,
        "schema": "v341_batch1",
        "num_predict": args.num_predict,
        "max_candidates": args.max_candidates,
    }

    if config_path.exists():
        existing = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        if existing != config:
            raise ValueError(
                "Run-name already uses different settings. "
                "Choose a different --run-name."
            )
    else:
        save_json(config_path, config)

    records = read_log(log_path)

    selected_ids = {
        str(g["evidence_id"].iloc[0])
        for g in groups
    }

    print("\n=== V34.1 OLLAMA ===")
    print("Model:", args.model)
    print("Thinking: OFF")
    print("Selected evidence groups:", len(groups))
    print("Already successful:", sum(
        eid in selected_ids and rec.get("status") == "ok"
        for eid, rec in records.items()
    ))
    print("Run directory:", run_dir, flush=True)

    client = Client(host=args.host, timeout=args.timeout)

    for index, group in enumerate(groups, 1):
        first = group.iloc[0]
        eid = str(first["evidence_id"])

        if records.get(eid, {}).get("status") == "ok":
            continue

        proposed = [
            {
                "candidate_id": str(r.candidate_id),
                "subject": str(r.subject),
                "relation": RELATION,
                "object": str(r.object),
                "matched_alias": str(r.matched_alias),
            }
            for r in group.itertuples(index=False)
        ]

        payload = {
            "issuer": first["issuer"],
            "ticker": first["ticker"],
            "filing_date": first["filing_date"],
            "current_sec_sentence": first["sentence"],
            "proposed_relations": proposed,
        }

        record = {
            "evidence_id": eid,
            "ticker": first["ticker"],
            "model": args.model,
            "candidate_count": len(proposed),
        }

        print(
            f"[{index}/{len(groups)}] "
            f"{first['ticker']} {len(proposed)} candidates "
            f"→ Ollama...",
            flush=True
        )

        start = time.time()

        try:
            response = client.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload, ensure_ascii=False
                        ),
                    },
                ],
                format=SCHEMA,
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": args.num_predict,
                },
                stream=False,
            )

            content = response.message.content or ""

            record["raw_content"] = content
            record["done_reason"] = getattr(
                response, "done_reason", None
            )
            record["thinking_chars"] = len(
                getattr(response.message, "thinking", "") or ""
            )

            if not content.strip():
                raise ValueError(
                    "EMPTY_CONTENT done_reason="
                    + str(record["done_reason"])
                )

            from v341_repair import parse_llm_json
            parsed = parse_llm_json(content)
            decisions = parsed.get("decisions")

            if not isinstance(decisions, list):
                raise ValueError("JSON_MISSING_DECISIONS_ARRAY")

            decisions_by_id = {}
            duplicates = set()

            for d in decisions:
                if not isinstance(d, dict):
                    continue
                cid = str(d.get("candidate_id", ""))
                if cid in decisions_by_id:
                    duplicates.add(cid)
                decisions_by_id[cid] = d

            verified = []

            for r in group.to_dict("records"):
                cid = str(r["candidate_id"])
                d = decisions_by_id.get(cid)

                if d is None:
                    d = {
                        "supported": False,
                        "evidence_quote": "",
                        "market_scope": "",
                        "reason": "MISSING_MODEL_DECISION",
                    }
                    accepted = False
                    validation_error = "MISSING_MODEL_DECISION"
                elif cid in duplicates:
                    accepted = False
                    validation_error = "DUPLICATE_MODEL_DECISION"
                else:
                    accepted, validation_error = validate_decision(
                        r, d, aliases
                    )

                verified.append({
                    "candidate_id": cid,
                    "evidence_id": eid,
                    "ticker": r["ticker"],
                    "subject": r["subject"],
                    "relation": RELATION,
                    "object": r["object"],
                    "filing_date": r["filing_date"],
                    "available_from": r["available_from"],
                    "accession": r["accession"],
                    "source_sha256": r["source_sha256"],
                    "offset_start": r["offset_start"],
                    "offset_end": r["offset_end"],
                    "matched_alias": r["matched_alias"],
                    "supported": d.get("supported"),
                    "evidence_quote": d.get("evidence_quote", ""),
                    "market_scope": d.get("market_scope", ""),
                    "reason": d.get("reason", ""),
                    "validation_error": validation_error,
                    "accepted_silver": accepted,
                    "model": args.model,
                    "label_source": "AI_SILVER",
                    "human_verified": False,
                    "strict_pit_approved": False,
                })

            record["status"] = "ok"
            record["verified_decisions"] = verified
            record["unknown_returned_ids"] = [
                cid for cid in decisions_by_id
                if cid not in {
                    str(x["candidate_id"])
                    for x in group.to_dict("records")
                }
            ]

            count = sum(
                d["accepted_silver"] for d in verified
            )
            print(
                f"  OK silver={count}/{len(verified)} "
                f"elapsed={time.time()-start:.1f}s",
                flush=True
            )

        except Exception as exc:
            record["status"] = "error"
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )[:1000]
            print(
                "  ERROR:", record["error"],
                flush=True
            )

        record["elapsed_seconds"] = round(
            time.time() - start, 2
        )

        # Save EVERY response immediately; failed records retry
        # on the next run.
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    record, ensure_ascii=False, default=str
                ) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())

        records[eid] = record
        export_results(records, run_dir)

    print("\n=== V34.1 RUN COMPLETE ===")
    show_report(args.run_name)

def export_results(records, run_dir):
    decisions = []
    accepted = []

    for record in records.values():
        if record.get("status") != "ok":
            continue
        for d in record.get("verified_decisions", []):
            decisions.append(d)
            if d.get("accepted_silver") is True:
                accepted.append(d)

    columns = [
        "candidate_id", "evidence_id", "ticker",
        "subject", "relation", "object",
        "filing_date", "available_from", "accession",
        "source_sha256", "offset_start", "offset_end",
        "matched_alias", "supported",
        "evidence_quote", "market_scope", "reason",
        "validation_error", "accepted_silver",
        "model", "label_source", "human_verified",
        "strict_pit_approved",
    ]

    pd.DataFrame(
        decisions, columns=columns
    ).to_csv(run_dir / "silver_decisions.csv", index=False)

    pd.DataFrame(
        accepted, columns=columns
    ).to_csv(run_dir / "silver_relations.csv", index=False)

def show_report(run_name):
    run_dir = OUT / "runs" / run_name
    logs = read_log(run_dir / "responses.jsonl")

    decisions_path = run_dir / "silver_decisions.csv"
    edges_path = run_dir / "silver_relations.csv"

    decisions = (
        pd.read_csv(decisions_path)
        if decisions_path.exists()
        else pd.DataFrame()
    )
    edges = (
        pd.read_csv(edges_path)
        if edges_path.exists()
        else pd.DataFrame()
    )

    rejected = {}
    if "validation_error" in decisions.columns:
        rejected = (
            decisions["validation_error"]
            .fillna("")
            .replace("", "NONE")
            .value_counts()
            .to_dict()
        )

    summary = {
        "evidence_requests": len(logs),
        "successful_requests": sum(
            x.get("status") == "ok" for x in logs.values()
        ),
        "failed_requests": sum(
            x.get("status") == "error" for x in logs.values()
        ),
        "decisions": len(decisions),
        "silver_relation_mentions": len(edges),
        "unique_issuer_target_edges": (
            edges[["subject", "relation", "object"]]
            .drop_duplicates().shape[0]
            if not edges.empty else 0
        ),
        "validation_error_counts": rejected,
        "human_verified": False,
        "strict_pit_approved": False,
    }

    save_json(run_dir / "report.json", summary)
    print(json.dumps(
        summary, indent=2, ensure_ascii=False
    ))

    if len(edges):
        print("\n=== SILVER RELATIONS ===")
        print(
            edges[[
                "subject", "relation", "object",
                "market_scope"
            ]].head(15).to_string(index=False)
        )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["build", "audit", "label", "report"],
    )
    parser.add_argument(
        "--model", default="qwen3.5:27b"
    )
    parser.add_argument(
        "--host",
        default="http://140.113.87.164:11434",
    )
    parser.add_argument(
        "--run-name", default="pilot_27b"
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Number of evidence groups; 0 means all"
    )
    parser.add_argument(
        "--max-candidates", type=int, default=8
    )
    parser.add_argument(
        "--timeout", type=int, default=300
    )
    parser.add_argument(
        "--num-predict", type=int, default=1600
    )

    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.limit < 0 or args.max_candidates < 1:
        raise ValueError("Invalid limit")

    if args.command == "build":
        build()
    elif args.command == "audit":
        audit()
    elif args.command == "label":
        run_label(args)
    elif args.command == "report":
        show_report(args.run_name)

if __name__ == "__main__":
    main()
