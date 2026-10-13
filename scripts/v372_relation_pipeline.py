#!/usr/bin/env python3
"""
V37.2: Conservative, direction-aware four-relation SEC candidates.

Commands:
  build  - generate candidates with stricter relation grammar
  audit  - check provenance, coverage, directions and batch sizes
  label  - run balanced, resumable Ollama verification
  report - show saved extraction results

Compatible with existing V37 validator, but does not overwrite V37.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v37_full_financial_kg as old

SOURCE = ROOT / "outputs/v34/relation_evidence.csv"
OUT = ROOT / "outputs/v372"

ISSUERS = old.ISSUERS
RELATIONS = old.RELATIONS

# Restrict trading universe, not knowledge graph entities.
ALLOWED_TICKERS = set(ISSUERS)

# The word "supplier" in a competitor list does not prove a
# supplier/customer business relationship.
COMPETITOR = re.compile(
    r"\b(?:our\s+(?:current\s+)?competitors\s+include"
    r"|competitors\s+to\s+.{1,70}?\s+include"
    r"|we\s+compete\s+with"
    r"|we\s+face\s+competition\s+from)\b",
    re.I,
)

# Only explicit issuer-directed competitor lists.
COMPETITOR_STRICT = re.compile(
    r"\b(?:our\s+(?:current\s+)?competitors\s+include"
    r"|competitors\s+to\s+.{1,70}?\s+include"
    r"|we\s+compete\s+with)\b",
    re.I,
)

# Strong named transaction patterns.
# The company alias is inserted into TARGET.
RULES = {
    "ISSUER_SUPPLIES": [
        r"\b(?:we|our company)\s+"
        r"(?:supply|supplied|provide|provided|sell|sold|license|licensed)"
        r"\s+(?:.{0,120}?\s+)?to\s+TARGET\b",

        r"\bTARGET\s+"
        r"(?:purchases|purchased|buys|bought|sources|sourced)"
        r"\s+.{0,120}?\s+from\s+(?:us|our company)\b",
    ],

    "ISSUER_CUSTOMER": [
        r"\b(?:we|our company)\s+"
        r"(?:purchase|purchased|buy|bought|procure|procured|source|sourced)"
        r"\s+.{0,120}?\s+from\s+TARGET\b",

        r"\bTARGET\s+"
        r"(?:supplies|supplied|provides|provided|sells|sold|licenses|licensed)"
        r"\s+.{0,120}?\s+to\s+(?:us|our company)\b",
    ],

    "SAME_SECTOR": [
        r"\b(?:we|our company)\s+and\s+TARGET\s+"
        r"(?:operate|compete|are)\s+.{0,90}?"
        r"\b(?:same industry|same sector)\b",

        r"\bTARGET\s+and\s+(?:we|our company)\s+"
        r"(?:operate|are)\s+.{0,90}?"
        r"\b(?:same industry|same sector)\b",
    ],
}

SECTOR_DIRECT = re.compile(
    r"\b(?:same industry|same sector)\b",
    re.I,
)

def alias_regex(alias):
    return (
        r"(?<![A-Za-z0-9])"
        + re.escape(alias)
        + r"(?![A-Za-z0-9])"
    )

def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_source():
    df = pd.read_csv(
        SOURCE, dtype={"accession": str}
    ).fillna("")

    if df["evidence_id"].duplicated().any():
        raise RuntimeError("Duplicate V34 evidence IDs")
    if not set(df["ticker"]).issubset(ALLOWED_TICKERS):
        raise RuntimeError("Unexpected issuer ticker")
    return df

def find_explicit_patterns(sentence, alias, kind):
    target_pattern = alias_regex(alias)
    results = []

    for template in RULES[kind]:
        pattern = template.replace(
            "TARGET", target_pattern
        )
        m = re.search(pattern, sentence, re.I | re.S)
        if m:
            results.append(m.group(0))

    return results

def make_row(src, subject, relation, obj, alias, cue, method):
    return {
        "candidate_id": old.uid(
            src["evidence_id"], subject, relation, obj
        ),
        "evidence_id": src["evidence_id"],
        "ticker": src["ticker"],
        "issuer": ISSUERS[src["ticker"]],
        "subject": subject,
        "relation": relation,
        "object": obj,
        "target_alias": alias,
        "sentence": src["sentence"],
        "context": src["context"],
        "cue": cue,
        "accession": src["accession"],
        "filing_date": str(src["filing_date"])[:10],
        "available_from": str(src["available_from"])[:10],
        "source_sha256": src["source_sha256"],
        "offset_start": int(src["offset_start"]),
        "offset_end": int(src["offset_end"]),
        "retrieval_score": float(src["retrieval_score"]),
        "candidate_method": method,
    }

def build():
    source = load_source()
    aliases = old.load_aliases()
    rows = []

    for src in source.to_dict("records"):
        sentence = str(src["sentence"])
        issuer = ISSUERS[src["ticker"]]

        competitor_trigger = COMPETITOR.search(sentence)
        strict_trigger = COMPETITOR_STRICT.search(sentence)

        for entity, name_variants in aliases.items():
            if entity == issuer:
                continue

            matches = []

            for alias in name_variants:
                pattern = alias_regex(alias)
                matches += [
                    (m.start(), m.end(), alias)
                    for m in re.finditer(
                        pattern, sentence, re.I
                    )
                ]

            if not matches:
                continue

            matches.sort()
            first_pos, _, first_alias = matches[0]

            # Competition can be generated only when company
            # appears AFTER an explicit competitor-list anchor.
            if strict_trigger:
                after = [
                    m for m in matches
                    if m[0] > strict_trigger.end()
                ]

                if after:
                    pos, _, alias = after[0]
                    middle = sentence[
                        strict_trigger.end():pos
                    ]
                    if not old.BLOCKED_HEADINGS.search(middle):
                        rows.append(make_row(
                            src, issuer, "COMPETES_WITH",
                            entity, alias,
                            strict_trigger.group(),
                            "EXPLICIT_COMPETITOR_LIST"
                        ))

            # Exclude generic competitor passages from
            # supply/customer/sector extraction.
            if competitor_trigger:
                continue

            for alias in name_variants:
                found = find_explicit_patterns(
                    sentence, alias, "ISSUER_SUPPLIES"
                )
                if found:
                    rows.append(make_row(
                        src, issuer, "SUPPLIES_TO",
                        entity, alias, found[0],
                        "DIRECTED_ISSUER_SUPPLY"
                    ))
                    # Derived inverse, not an independent
                    # SEC assertion. Track it explicitly.
                    rows.append(make_row(
                        src, entity, "CUSTOMER_OF",
                        issuer, alias, found[0],
                        "DERIVED_INVERSE_CUSTOMER"
                    ))

                found = find_explicit_patterns(
                    sentence, alias, "ISSUER_CUSTOMER"
                )
                if found:
                    rows.append(make_row(
                        src, issuer, "CUSTOMER_OF",
                        entity, alias, found[0],
                        "DIRECTED_ISSUER_CUSTOMER"
                    ))
                    rows.append(make_row(
                        src, entity, "SUPPLIES_TO",
                        issuer, alias, found[0],
                        "DERIVED_INVERSE_SUPPLIER"
                    ))

                if SECTOR_DIRECT.search(sentence):
                    found = find_explicit_patterns(
                        sentence, alias, "SAME_SECTOR"
                    )
                    if found:
                        rows.append(make_row(
                            src, issuer, "SAME_SECTOR",
                            entity, alias, found[0],
                            "EXPLICIT_SAME_SECTOR"
                        ))

    columns = old.candidate_cols()
    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        df = df.drop_duplicates("candidate_id")
        df = df.sort_values(
            ["relation", "retrieval_score", "evidence_id"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

    save(df, OUT / "candidates.csv")

    print("\n=== V37.2 BUILD ===")
    print("Source evidence:", len(source))
    print("Candidates:", len(df))
    print("Unique evidence:", df["evidence_id"].nunique())

    print("\nRELATION COUNTS")
    print(
        df["relation"].value_counts()
        .reindex(RELATIONS, fill_value=0)
        .to_string()
    )

    print("\nMETHOD COUNTS")
    print(df["candidate_method"].value_counts().to_string())

    print("\nIMPORTANT")
    print("Missing relation candidates are NOT fabricated.")
    print("SEC lexical coverage is incomplete.")

def audit():
    candidates = pd.read_csv(
        OUT / "candidates.csv",
        dtype={"accession": str}
    ).fillna("")

    source = load_source().set_index("evidence_id")

    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate candidate ID")

    for r in candidates.itertuples(index=False):
        if r.evidence_id not in source.index:
            raise RuntimeError("Unknown evidence")

        e = source.loc[r.evidence_id]

        for field in ["sentence", "source_sha256", "accession"]:
            if str(getattr(r, field)) != str(e[field]):
                raise RuntimeError(
                    f"Provenance mismatch: {field}"
                )

        for field in ["offset_start", "offset_end"]:
            if int(getattr(r, field)) != int(e[field]):
                raise RuntimeError(
                    f"Offset mismatch: {field}"
                )

        if r.subject == r.object:
            raise RuntimeError("Self edge")

        if pd.Timestamp(r.available_from) <= pd.Timestamp(r.filing_date):
            raise RuntimeError("Invalid disclosure chronology")

    groups = candidates.groupby("evidence_id").size()

    print("\n=== V37.2 AUDIT PASS ===")
    print("Candidates:", len(candidates))
    print("Evidence groups:", len(groups))
    print("Maximum group size:", int(groups.max()) if len(groups) else 0)
    print("Groups over 12:", int((groups > 12).sum()))
    print("All groups will be split into batches.")
    print("Four relation classes: SUPPORTED")
    print("All four relation classes observed: NOT ASSUMED")
    print("Strict PIT: NOT CERTIFIED")

def balanced_groups(candidates, batch_size, limit):
    # Split ALL candidates of each evidence, never discard
    # candidates beyond max_candidates.
    batches = []

    for eid, group in candidates.groupby(
        "evidence_id", sort=False
    ):
        group = group.sort_values(
            ["relation", "candidate_id"]
        )

        for start in range(0, len(group), batch_size):
            part = group.iloc[start:start + batch_size]
            present = set(part["relation"])

            # Prioritize novel supply/customer/sector candidates.
            priority = (
                0 if "SAME_SECTOR" in present else
                1 if "SUPPLIES_TO" in present else
                2 if "CUSTOMER_OF" in present else
                3
            )

            batches.append((
                priority,
                -float(part["retrieval_score"].max()),
                str(eid),
                start,
                part
            ))

    batches.sort(key=lambda t: t[:4])

    return [
        t[-1] for t in (
            batches if limit == 0 else batches[:limit]
        )
    ]

def label(args):
    from ollama import Client

    candidates = pd.read_csv(
        OUT / "candidates.csv",
        dtype={"accession": str}
    ).fillna("")

    batches = balanced_groups(
        candidates, args.max_candidates, args.limit
    )

    aliases = old.load_aliases()
    run_dir = OUT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config = {
        "model": args.model,
        "host": args.host,
        "thinking": False,
        "batch_size": args.max_candidates,
        "version": "v372",
    }

    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise RuntimeError(
                "Existing run has different settings. "
                "Use a new --run-name."
            )
    else:
        old.json_write(config_path, config)

    log_path = run_dir / "responses.jsonl"

    # Batch IDs include the complete candidate set.
    def batch_id(group):
        return old.uid(*sorted(
            group["candidate_id"].astype(str).tolist()
        ))

    records = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                records[entry["batch_id"]] = entry

    print("\n=== V37.2 OLLAMA ===")
    print("Total batches selected:", len(batches))
    print("Thinking: OFF", flush=True)

    client = Client(host=args.host, timeout=args.timeout)

    for index, group in enumerate(batches, 1):
        bid = batch_id(group)

        if records.get(bid, {}).get("status") == "ok":
            continue

        first = group.iloc[0]
        proposals = [
            {
                "candidate_id": str(r.candidate_id),
                "subject": str(r.subject),
                "relation": str(r.relation),
                "object": str(r.object),
                "method": str(r.candidate_method),
            }
            for r in group.itertuples(index=False)
        ]

        payload = {
            "issuer": str(first["issuer"]),
            "current_sec_sentence": str(first["sentence"]),
            "proposals": proposals,
        }

        entry = {
            "batch_id": bid,
            "evidence_id": str(first["evidence_id"]),
            "model": args.model,
        }

        print(
            f"[{index}/{len(batches)}] "
            f"{first['ticker']} candidates={len(group)}",
            flush=True
        )

        try:
            response = client.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": old.SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload, ensure_ascii=False
                        ),
                    },
                ],
                format=old.SCHEMA,
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": args.num_predict,
                },
                stream=False,
            )

            content = response.message.content or ""
            entry["raw_content"] = content
            entry["done_reason"] = getattr(
                response, "done_reason", None
            )

            decisions = old.parse_json(content)
            ids = [d["candidate_id"] for d in decisions]
            expected = set(group["candidate_id"].astype(str))

            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError(
                    "Missing, extra or duplicate candidate IDs"
                )

            by_id = {
                d["candidate_id"]: d for d in decisions
            }

            verified = []

            for r in group.to_dict("records"):
                d = by_id[str(r["candidate_id"])]

                accepted, error = old.validate(
                    r, d, aliases
                )

                # Existing validator has limited semantic
                # direction checking. Derived inverse edges
                # are kept separate from direct assertions.
                verified.append({
                    **r,
                    "supported": d["supported"],
                    "evidence_quote": d["evidence_quote"],
                    "reason": d["reason"],
                    "market_scope": d["market_scope"],
                    "validation_error": error,
                    "accepted_silver": accepted,
                    "model": args.model,
                    "human_verified": False,
                    "strict_pit_approved": False,
                })

            entry["status"] = "ok"
            entry["decisions"] = verified

            print(
                "  PASS accepted:",
                sum(d["accepted_silver"] for d in verified),
                flush=True
            )

        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = (
                f"{type(exc).__name__}: {exc}"
            )[:1000]
            print("  ERROR:", entry["error"], flush=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(entry, ensure_ascii=False, default=str)
                + "\n"
            )
            f.flush()

        records[bid] = entry

        # Export after EVERY request.
        valid = [
            d for record in records.values()
            if record.get("status") == "ok"
            for d in record.get("decisions", [])
        ]

        frame = pd.DataFrame(valid)
        frame.to_csv(
            run_dir / "silver_decisions.csv", index=False
        )

        selected = (
            frame[frame["accepted_silver"] == True]
            if not frame.empty
            else frame
        )

        selected.to_csv(
            run_dir / "silver_relations.csv", index=False
        )

    print("\n=== V37.2 LABEL RUN FINISHED ===")

def report(args):
    p = OUT / "runs" / args.run_name / "silver_relations.csv"
    if not p.exists():
        print("No silver relations file yet.")
        return

    df = pd.read_csv(p)

    print("\n=== V37.2 RELATIONS ===")
    print("Accepted mentions:", len(df))

    print(
        df["relation"].value_counts()
        .reindex(RELATIONS, fill_value=0)
        .to_string()
    )

    if len(df):
        print("\nUNIQUE EDGES")
        print(
            df[["subject", "relation", "object"]]
            .drop_duplicates()
            .to_string(index=False)
        )

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command", choices=["build", "audit", "label", "report"]
    )
    p.add_argument("--model", default="qwen3.5:27b")
    p.add_argument(
        "--host", default="http://140.113.87.164:11434"
    )
    p.add_argument("--run-name", default="strict_27b")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--num-predict", type=int, default=2048)

    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.max_candidates < 1 or args.limit < 0:
        raise ValueError("Invalid batch size or limit")

    if args.command == "build":
        build()
    elif args.command == "audit":
        audit()
    elif args.command == "label":
        label(args)
    else:
        report(args)

if __name__ == "__main__":
    main()
