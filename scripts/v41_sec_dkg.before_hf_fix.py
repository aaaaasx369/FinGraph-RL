#!/usr/bin/env python3
"""
V41 SEC -> GLiNER -> GLiREL -> Ollama -> Temporal KG -> FinRL.

Usage:
  extract --limit 30
  verify --limit 30
  build
  wire
  report

Important:
  - GLiREL predictions are candidate scores, not ground truth.
  - LLM accepted relations are SILVER, not human-gold.
  - SEC dates are still provisional PIT.
  - No missing sector/customer/supplier relationships fabricated.
  - V35 existing competition silver is preserved.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v41"
OUT.mkdir(parents=True, exist_ok=True)

V38 = ROOT / "outputs/v38/full_sec_relation_evidence.csv"
V35 = ROOT / "outputs/v35/evidence_ledger.csv"
V40 = ROOT / "scripts/v40_dynamic_graph_finrl.py"
ALIASES = ROOT / "outputs/v34/entity_aliases.csv"

ISSUERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

GLIREL_LABELS = [
    "supplies products or services to",
    "is a customer of",
    "competes with",
    "operates in the same industry as",
]

LABEL_MAP = {
    GLIREL_LABELS[0]: "SUPPLIES_TO",
    GLIREL_LABELS[1]: "CUSTOMER_OF",
    GLIREL_LABELS[2]: "COMPETES_WITH",
    GLIREL_LABELS[3]: "SAME_SECTOR",
}

CUES = re.compile(
    r"\b(supplies?|supplied|supplier|suppliers|"
    r"customer|customers|purchas(?:e|ed|es|ing)|"
    r"buy|buys|bought|sells?|sold|provides?|"
    r"competitor|competitors|competes?|"
    r"same industry|same sector|industry peers?)\b",
    re.I,
)

COMPETITION_CONTEXT = re.compile(
    r"\b(we face competition from|"
    r"our (?:current )?competitors include|"
    r"competitors to .{0,80}? include|"
    r"we compete with)\b",
    re.I,
)


def fail(message):
    raise RuntimeError("V41 STOP: " + str(message))


def stable_id(*values):
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()
    ).hexdigest()[:24]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def alias_lookup():
    raw = pd.read_csv(ALIASES).fillna("")
    mapping = {}

    for r in raw.itertuples(index=False):
        entity = str(r.canonical).strip()
        name = str(r.alias).strip()

        if entity and len(name) >= 4:
            mapping.setdefault(entity, set()).add(name)

    for canonical in ISSUERS.values():
        mapping.setdefault(canonical, set()).add(canonical)

    return {
        key: sorted(values, key=len, reverse=True)
        for key, values in mapping.items()
    }


def canonical_entity(name, mapping):
    name = str(name).strip()

    candidates = []

    for entity, aliases in mapping.items():
        for alias in aliases:
            if name.casefold() == alias.casefold():
                candidates.append((entity, len(alias)))

    candidates = list({x[0] for x in candidates})

    # Ambiguous aliases must not silently resolve.
    return candidates[0] if len(candidates) == 1 else None


def exact_alias_in_text(text, entity, mapping):
    for alias in mapping.get(entity, []):
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(alias)
            + r"(?![A-Za-z0-9])"
        )

        if re.search(pattern, text, re.I):
            return True

    return False


def source():
    if not V38.exists():
        fail("Missing V38 evidence")

    raw = pd.read_csv(
        V38, dtype={"accession": str},
        low_memory=False
    ).fillna("")

    required = [
        "evidence_id", "ticker", "sentence",
        "context", "filing_date", "available_from",
        "accession", "source_sha256",
        "offset_start", "offset_end",
    ]

    if not set(required).issubset(raw.columns):
        fail("V38 source schema mismatch")

    if raw["evidence_id"].duplicated().any():
        fail("Duplicate evidence IDs")

    if not set(raw["ticker"]).issubset(ISSUERS):
        fail("Unknown issuer ticker")

    return raw


def load_models():
    from gliner import GLiNER
    from glirel import GLiREL
    import spacy

    print("Loading GLiNER...", flush=True)
    ner = GLiNER.from_pretrained(
        "urchade/gliner_mediumv2.1"
    )

    print("Loading GLiREL...", flush=True)
    rel = GLiREL.from_pretrained(
        "jackboyla/glirel-large-v0"
    )

    nlp = spacy.blank("en")

    return ner, rel, nlp


def token_span(doc, start, end):
    span = doc.char_span(
        int(start), int(end),
        alignment_mode="expand"
    )

    if span is None or len(span) == 0:
        return None

    return [
        span.start,
        span.end - 1,
    ]


def extract(args):
    rows = source()
    mapping = alias_lookup()
    ner_model, rel_model, nlp = load_models()

    output = OUT / "extracted_candidates.jsonl"
    records = {}

    if output.exists():
        for line in output.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                item = json.loads(line)
                records[item["evidence_id"]] = item

    rows = rows.sort_values(
        ["filing_date", "evidence_id"]
    )

    if args.limit:
        rows = rows.head(args.limit)

    for i, record in enumerate(
        rows.to_dict("records"), 1
    ):
        eid = str(record["evidence_id"])

        if eid in records and records[eid].get("status") == "ok":
            continue

        ticker = record["ticker"]
        issuer = ISSUERS[ticker]

        # SEC parser sometimes emits huge accounting blocks.
        sentence = str(record["sentence"])
        text = str(record["context"])

        if len(text) > 2200:
            text = sentence[:2200]

        result = {
            "evidence_id": eid,
            "ticker": ticker,
            "issuer": issuer,
            "accession": str(record["accession"]),
            "filing_date": str(record["filing_date"])[:10],
            "available_from": str(
                record["available_from"]
            )[:10],
            "source_sha256": str(
                record["source_sha256"]
            ),
            "offset_start": int(
                record["offset_start"]
            ),
            "offset_end": int(
                record["offset_end"]
            ),
            "sentence": sentence,
            "context": text,
            "candidates": [],
            "unresolved_entities": [],
        }

        try:
            entities = ner_model.predict_entities(
                text,
                ["company", "organization"],
                threshold=args.ner_threshold,
            )

            doc = nlp.make_doc(text)
            spans = []
            linked = []

            for entity in entities:
                name = str(entity.get("text", "")).strip()

                canonical = canonical_entity(
                    name, mapping
                )

                position = token_span(
                    doc,
                    entity.get("start", -1),
                    entity.get("end", -1),
                )

                if position is None:
                    continue

                if canonical is None:
                    result["unresolved_entities"].append(
                        name
                    )
                    continue

                linked.append({
                    "canonical": canonical,
                    "mention": name,
                    "position": position,
                })

                spans.append([
                    position[0],
                    position[1],
                    "ORG",
                    name,
                ])

            spans = [
                list(x)
                for x in {
                    tuple(s) for s in spans
                }
            ]

            # GLiREL requires entity spans.
            relations = []

            if len(spans) >= 2:
                relations = rel_model.predict_relations(
                    [t.text for t in doc],
                    GLIREL_LABELS,
                    threshold=args.rel_threshold,
                    ner=spans,
                    top_k=-1,
                )

            candidate_dict = {}

            for relation in relations:
                label = str(
                    relation.get("label", "")
                )

                kind = LABEL_MAP.get(label)

                if kind is None:
                    continue

                head_text = relation.get(
                    "head_text", ""
                )
                tail_text = relation.get(
                    "tail_text", ""
                )

                head = canonical_entity(
                    head_text, mapping
                )
                tail = canonical_entity(
                    tail_text, mapping
                )

                if (
                    not head or not tail
                    or head == tail
                ):
                    continue

                score = float(
                    relation.get("score", 0)
                )

                cid = stable_id(
                    eid, head, kind, tail
                )

                candidate_dict[cid] = {
                    "candidate_id": cid,
                    "subject": head,
                    "relation": kind,
                    "object": tail,
                    "glirel_score": score,
                    "method": "GLIREL",
                }

            # Explicit issuer disclosure fallback:
            # "we" in SEC refers to reporting issuer.
            # These are candidates, NOT confirmed edges.
            #
            # This matters when only one external company
            # is named and GLiREL lacks a second NER span.
            if CUES.search(sentence):
                targets = {
                    x["canonical"] for x in linked
                    if x["canonical"] != issuer
                    and exact_alias_in_text(
                        sentence,
                        x["canonical"],
                        mapping
                    )
                }

                for target in sorted(targets):
                    for kind in [
                        "SUPPLIES_TO",
                        "CUSTOMER_OF",
                        "COMPETES_WITH",
                    ]:
                        if (
                            COMPETITION_CONTEXT.search(sentence)
                            and kind != "COMPETES_WITH"
                        ):
                            continue

                        for subject, obj in [
                            (issuer, target),
                            (target, issuer),
                        ]:
                            cid = stable_id(
                                eid, subject, kind, obj
                            )

                            candidate_dict.setdefault(
                                cid,
                                {
                                    "candidate_id": cid,
                                    "subject": subject,
                                    "relation": kind,
                                    "object": obj,
                                    "glirel_score": None,
                                    "method": (
                                        "ISSUER_CONTEXT_CANDIDATE"
                                    ),
                                }
                            )

            result["candidates"] = list(
                candidate_dict.values()
            )

            result["status"] = "ok"

        except Exception as exc:
            result["status"] = "error"
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        with output.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    result, ensure_ascii=False
                ) + "\n"
            )

        records[eid] = result

        print(
            f"[{i}/{len(rows)}] "
            f"{ticker} "
            f"{result['status']} "
            f"candidates={len(result['candidates'])}",
            flush=True,
        )

    print("\n=== V41 EXTRACTION COMPLETE ===")


def latest_records(path, key):
    values = {}

    if not path.exists():
        return values

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            obj = json.loads(line)
            values[str(obj[key])] = obj

    return values


VERIFY_SYSTEM = """
You are an extremely conservative SEC filing
financial relation verifier.

The issuer is the reporting company.
We, our, and us refer to the issuer.

Allowed relations:
SUPPLIES_TO: Subject supplies goods/services to object.
CUSTOMER_OF: Subject buys from or is a customer of object.
COMPETES_WITH: Subject competes against object.
SAME_SECTOR: Explicitly documented common sector.

Rules:
- Use ONLY the provided historical SEC evidence.
- Evaluate the EXACT proposed direction.
- Generic customers or suppliers do not prove a
  specific named-company relationship.
- Competition does NOT imply supply or customer.
- A competing technology vendor is not automatically
  an existing supplier.
- Hypothetical future transactions do not count.
- Do not use external world knowledge.
- Quote must be an EXACT substring of the SEC sentence.
- Do not infer SAME_SECTOR from competition.
- Return JSON with:
  supported: boolean
  evidence_quote: string
  explanation: string
- When uncertain, supported MUST be false.
"""


def ollama_request(host, model, payload, timeout):
    import requests

    url = host.rstrip("/") + "/api/chat"

    response = requests.post(
        url,
        json={
            "model": model,
            "stream": False,
            "think": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": VERIFY_SYSTEM,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False
                    ),
                },
            ],
            "options": {
                "temperature": 0,
                "num_predict": 768,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()
    content = response.json().get(
        "message", {}
    ).get("content", "")

    content = re.sub(
        r"^\s*```(?:json)?\s*",
        "", content.strip(), flags=re.I
    )
    content = re.sub(
        r"\s*```\s*$", "", content
    ).strip()

    return json.loads(content)


def verify(args):
    extracted = latest_records(
        OUT / "extracted_candidates.jsonl",
        "evidence_id",
    )

    if not extracted:
        fail("Run extract first")

    decisions_path = OUT / "verified_decisions.jsonl"

    decisions = latest_records(
        decisions_path, "candidate_id"
    )

    mapping = alias_lookup()

    queue = []

    for evidence in extracted.values():
        if evidence.get("status") != "ok":
            continue

        for candidate in evidence.get("candidates", []):
            if candidate["candidate_id"] not in decisions:
                queue.append((evidence, candidate))

    queue.sort(
        key=lambda x: (
            str(x[0]["filing_date"]),
            str(x[1]["candidate_id"]),
        )
    )

    if args.limit:
        queue = queue[:args.limit]

    print(
        "Verification requests:",
        len(queue),
        flush=True,
    )

    for i, (e, candidate) in enumerate(queue, 1):
        cid = candidate["candidate_id"]
        result = {
            **candidate,
            "evidence_id": e["evidence_id"],
            "ticker": e["ticker"],
            "issuer": e["issuer"],
            "accession": e["accession"],
            "filing_date": e["filing_date"],
            "available_from": e["available_from"],
            "source_sha256": e["source_sha256"],
            "offset_start": e["offset_start"],
            "offset_end": e["offset_end"],
            "model": args.model,
            "human_verified": False,
            "strict_pit_approved": False,
        }

        payload = {
            "issuer": e["issuer"],
            "SEC_sentence": e["sentence"],
            "SEC_context": e["context"],
            "proposed_relation": {
                "subject": candidate["subject"],
                "relation": candidate["relation"],
                "object": candidate["object"],
            },
        }

        try:
            response = ollama_request(
                args.host,
                args.model,
                payload,
                args.timeout,
            )

            supported = response.get("supported")

            if type(supported) is not bool:
                raise ValueError(
                    "supported must be boolean"
                )

            quote = response.get(
                "evidence_quote", ""
            )

            if not isinstance(quote, str):
                raise ValueError("quote not string")

            accepted = bool(
                supported
                and len(quote.strip()) >= 15
                and quote in e["sentence"]
            )

            if accepted:
                target = (
                    candidate["object"]
                    if candidate["subject"] == e["issuer"]
                    else candidate["subject"]
                )

                accepted = exact_alias_in_text(
                    quote, target, mapping
                )

            # Forbid easy false supply/customer relations
            # inside an explicitly competitive passage.
            if (
                COMPETITION_CONTEXT.search(
                    e["sentence"]
                )
                and candidate["relation"] in {
                    "SUPPLIES_TO", "CUSTOMER_OF"
                }
            ):
                accepted = False

            # Sector relationships require historical
            # structured classification, not an LLM guess.
            if candidate["relation"] == "SAME_SECTOR":
                accepted = False

            result.update({
                "status": "ok",
                "supported": supported,
                "accepted_silver": accepted,
                "evidence_quote": quote,
                "reason": str(
                    response.get("explanation", "")
                ),
            })

        except Exception as exc:
            result.update({
                "status": "error",
                "accepted_silver": False,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

        with decisions_path.open(
            "a", encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    result, ensure_ascii=False
                ) + "\n"
            )

        print(
            f"[{i}/{len(queue)}] "
            f"{candidate['relation']} "
            f"{result['status']} "
            f"accepted={result['accepted_silver']}",
            flush=True,
        )

    print("\n=== V41 VERIFICATION COMPLETE ===")


def build(args):
    if not V35.exists():
        fail("Missing original V35 ledger")

    original = pd.read_csv(
        V35, low_memory=False
    )

    if original.empty:
        fail("V35 historical silver is empty")

    decisions = latest_records(
        OUT / "verified_decisions.jsonl",
        "candidate_id",
    )

    accepted = [
        x for x in decisions.values()
        if x.get("accepted_silver") is True
        and x.get("status") == "ok"
    ]

    source_index = source().set_index(
        "evidence_id"
    )

    new_rows = []

    for d in accepted:
        if d["evidence_id"] not in source_index.index:
            fail("Missing V38 provenance")

        e = source_index.loc[d["evidence_id"]]

        if str(d["accession"]) != str(e["accession"]):
            fail("Accession mismatch")

        if str(d["source_sha256"]) != str(
            e["source_sha256"]
        ):
            fail("Source SHA mismatch")

        if d["evidence_quote"] not in str(e["sentence"]):
            fail("Evidence quote mismatch")

        if pd.Timestamp(
            d["available_from"]
        ) <= pd.Timestamp(
            d["filing_date"]
        ):
            fail("Incorrect time chronology")

        new_rows.append({
            "ticker": d["ticker"],
            "issuer": d["issuer"],
            "subject": d["subject"],
            "relation": d["relation"],
            "object": d["object"],
            "available_from": d["available_from"],
            "filing_date": d["filing_date"],
            "evidence_id": d["evidence_id"],
            "candidate_id": d["candidate_id"],
            "accession": d["accession"],
            "source_sha256": d["source_sha256"],
            "offset_start": d["offset_start"],
            "offset_end": d["offset_end"],
            "evidence_quote": d["evidence_quote"],
            "model": d["model"],
            "label_source": "V41_AI_SILVER",
            "human_verified": False,
            "strict_pit_approved": False,
        })

    new = pd.DataFrame(new_rows)

    if len(new):
        merged = pd.concat(
            [original, new],
            ignore_index=True,
        )
    else:
        merged = original.copy()

    required = [
        "ticker", "issuer", "subject", "relation",
        "object", "available_from", "evidence_id"
    ]

    if not set(required).issubset(merged.columns):
        fail("KG ledger schema mismatch")

    merged = merged.drop_duplicates(
        [
            "evidence_id",
            "subject",
            "relation",
            "object",
        ]
    )

    if not set(
        merged["relation"].dropna()
    ).issubset(RELATIONS):
        fail("Invalid KG relation")

    # Unknown relationships are not turned into edges.
    # Missing sectors remain missing until a valid
    # historical SIC source is supplied.
    destination = OUT / "kg_source"
    destination.mkdir(
        parents=True, exist_ok=True
    )

    merged.to_csv(
        destination / "evidence_ledger.csv",
        index=False,
    )

    unique_edges = merged[
        ["subject", "relation", "object"]
    ].drop_duplicates()

    summary = {
        "original_v35_mentions": len(original),
        "new_ai_silver_mentions": len(new),
        "merged_mentions": len(merged),
        "unique_edges": len(unique_edges),
        "relation_counts": (
            unique_edges["relation"]
            .value_counts()
            .reindex(RELATIONS, fill_value=0)
            .to_dict()
        ),
        "human_verified": False,
        "strict_pit_approved": False,
        "sector_edges_added": 0,
        "original_findkg_model_trained": False,
    }

    save_json(
        OUT / "kg_manifest.json",
        summary
    )

    print("\n=== V41 KG BUILT ===")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ))


def wire():
    if not V40.exists():
        fail("Missing existing V40 FinRL bridge")

    ledger = OUT / "kg_source/evidence_ledger.csv"

    if not ledger.exists():
        fail("Run V41 build first")

    code = V40.read_text(encoding="utf-8")

    replacements = {
        'OUT = ROOT / "outputs/v40"':
        'OUT = ROOT / "outputs/v41/finrl"',

        'V35 = ROOT / "outputs/v35"':
        'V35 = ROOT / "outputs/v41/kg_source"',
    }

    for old, new in replacements.items():
        if code.count(old) != 1:
            fail(
                "V40 source differs; unsafe patch refused: "
                + old
            )
        code = code.replace(old, new, 1)

    path = ROOT / "scripts/v41_finrl_bridge.py"
    path.write_text(
        code, encoding="utf-8"
    )

    import py_compile

    py_compile.compile(
        str(path), doraise=True
    )

    print("\n=== V41 FINRL CONNECTED ===")
    print("New script:", path)
    print("Source:", ledger)
    print("Original V20-V40 files unchanged.")


def report():
    manifest = OUT / "kg_manifest.json"

    if manifest.exists():
        print(manifest.read_text(
            encoding="utf-8"
        ))

    finrl = (
        OUT / "finrl/runs/DYNAMIC_GRAPH/summary.csv"
    )

    if finrl.exists():
        data = pd.read_csv(finrl)

        selection = data[
            (data["tier"] == "conservative")
            & (data["split"] == "test")
            & (data["condition"].isin(
                ["CLEAN", "FULL_DYNAMIC"]
            ))
        ]

        print("\n=== FINRL TEST ===")
        print(
            selection[[
                "condition", "agent",
                "total_return_mean",
                "sharpe_mean",
                "max_drawdown_mean",
            ]].to_string(index=False)
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "extract", "verify", "build",
            "wire", "report"
        ]
    )

    p.add_argument(
        "--limit", type=int, default=30,
        help="0 = process all remaining records"
    )

    p.add_argument(
        "--ner-threshold",
        type=float, default=0.35
    )

    p.add_argument(
        "--rel-threshold",
        type=float, default=0.3
    )

    p.add_argument(
        "--model", default="qwen3.5:27b"
    )

    p.add_argument(
        "--host",
        default="http://140.113.87.164:11434"
    )

    p.add_argument(
        "--timeout", type=int, default=300
    )

    args = p.parse_args()

    if args.command == "extract":
        extract(args)
    elif args.command == "verify":
        verify(args)
    elif args.command == "build":
        build(args)
    elif args.command == "wire":
        wire()
    else:
        report()


if __name__ == "__main__":
    main()
