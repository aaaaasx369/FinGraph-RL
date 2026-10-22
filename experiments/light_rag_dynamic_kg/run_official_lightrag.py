#!/usr/bin/env python3

import asyncio
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import (
    ollama_model_complete,
    ollama_embed,
)
from lightrag.utils import EmbeddingFunc


ROOT = Path(__file__).resolve().parents[2]

KG_FILE = (
    ROOT /
    "outputs/unified_temporal_kg_v3/unified_temporal_edges.csv"
)

SEC_FILE = (
    ROOT /
    "results/aegis_rl_v1/v11f_sec_pit/sec_pit_retrieval.parquet"
)

MAP_DIR = (
    ROOT /
    "outputs/finrl_dkg_original_full/original_run/"
    "FinDKG_dataset/FinDKG-full"
)

ENTITY_FILE = MAP_DIR / "entity2id.txt"
RELATION_FILE = MAP_DIR / "relation2id.txt"

COMPANY_MAP_FILE = (
    ROOT /
    "outputs/unified_temporal_kg_v3/entity_mapping_report.csv"
)

OUT = ROOT / "results/light_rag_dynamic_kg"
OUT.mkdir(parents=True, exist_ok=True)

HOST = "http://140.113.87.164:11434"

LLM_MODEL = "qwen3.5:27b"
EMBED_MODEL = "bge-m3:latest"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


def load_entity_names():
    """
    Original FinDKG entity2id.txt:
      Entity Name <tab> old_id <tab> ...
    """
    mapping = {}

    with ENTITY_FILE.open(
        encoding="utf-8"
    ) as f:

        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 2:
                continue

            try:
                entity_id = int(parts[1])
            except ValueError:
                continue

            mapping[
                f"FINDKG:{entity_id}"
            ] = parts[0].strip()

    return mapping


def load_relation_names():
    mapping = {}

    with RELATION_FILE.open(
        encoding="utf-8"
    ) as f:

        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            try:
                relation_id = int(parts[-1])
            except ValueError:
                continue

            name = "_".join(parts[:-1])

            mapping[
                f"FINDKG_REL:{relation_id}"
            ] = name

    if len(mapping) != 15:
        raise RuntimeError(
            f"Expected 15 FinDKG relations, got {len(mapping)}"
        )

    return mapping


def load_company_mapping():
    df = pd.read_csv(COMPANY_MAP_FILE)

    mapping = {}

    for row in df.itertuples(index=False):
        if not bool(row.verified):
            continue

        mapping[str(row.ticker)] = {
            "old": str(row.old_id),
            "canonical": str(row.canonical_id),
            "name": str(row.entity_name),
        }

    for ticker in TICKERS:
        if ticker not in mapping:
            raise RuntimeError(
                f"Company mapping not verified: {ticker}"
            )

    return mapping


def decode_entity(
    value,
    entity_names,
    companies,
):
    value = str(value)

    for entry in companies.values():
        if value in (
            entry["old"],
            entry["canonical"],
        ):
            return entry["name"]

    return entity_names.get(value, value)


def load_historical_kg(
    ticker,
    date,
    top_k,
    entity_names,
    relation_names,
    companies,
):
    df = pd.read_csv(
        KG_FILE,
        low_memory=False
    )

    available = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce"
    ).dt.tz_localize(None)

    # Conservative rule: strictly earlier calendar date.
    eligible = (
        available.notna()
        & (available.dt.normalize() < date)
    )

    target = companies[ticker]

    identifiers = {
        target["old"],
        target["canonical"],
    }

    focal = (
        df["head"].astype(str).isin(identifiers)
        | df["tail"].astype(str).isin(identifiers)
    )

    selected = df.loc[
        eligible & focal
    ].copy()

    selected["_available"] = (
        available.loc[selected.index]
    )

    print("[KG HISTORICAL MATCH]", len(selected))

    selected = selected.sort_values(
        ["_available", "event_id"],
        ascending=[False, True],
        kind="stable"
    )

    # Deduplicate exact triples, keeping most recent
    # eligible observation. Preserve originals on disk.
    selected = selected.drop_duplicates(
        ["head", "relation", "tail"],
        keep="first"
    ).head(top_k)

    rows = []

    for row in selected.itertuples(index=False):

        head = str(row.head)
        tail = str(row.tail)
        relation = str(row.relation)

        readable_head = decode_entity(
            head, entity_names, companies
        )

        readable_tail = decode_entity(
            tail, entity_names, companies
        )

        readable_relation = (
            relation_names.get(
                relation, relation
            )
        )

        # Coded / non-semantic relations must remain
        # uninterpreted. Generic relations such as
        # Relate_To and Has must not imply causality.
        rows.append({
            "event_id": str(row.event_id),
            "head_id": head,
            "tail_id": tail,
            "head": readable_head,
            "relation_id": relation,
            "relation": readable_relation,
            "tail": readable_tail,
            "available_at": str(row._available)
            if hasattr(row, "_available")
            else None,
            "source": str(row.source),
        })

    # Fix timestamps by using the selected DataFrame,
    # since itertuples may rename underscore columns.
    for item, ts in zip(
        rows, selected["_available"]
    ):
        item["available_at"] = (
            pd.Timestamp(ts).isoformat()
        )

    return rows


def load_historical_sec(
    ticker,
    date,
    top_k,
):
    df = pd.read_parquet(SEC_FILE)

    available = pd.to_datetime(
        df["pit_usable_from"],
        utc=True,
        errors="coerce"
    ).dt.tz_localize(None)

    eligible = (
        df["ticker"].astype(str).eq(ticker)
        & available.notna()
        & (available.dt.normalize() < date)
    )

    rows = df.loc[eligible].copy()

    rows["_available"] = available.loc[
        rows.index
    ]

    rows["current_excerpt"] = (
        rows["current_excerpt"]
        .fillna("")
        .astype(str)
    )

    rows = rows.loc[
        rows["current_excerpt"].str.strip().ne("")
    ]

    rows = rows.sort_values(
        ["_available", "evidence_id"],
        ascending=[False, True],
        kind="stable"
    ).drop_duplicates(
        "evidence_id"
    ).head(top_k)

    result = []

    for row in rows.itertuples(index=False):
        result.append({
            "evidence_id": str(row.evidence_id),
            "ticker": ticker,
            "form": str(row.form),
            "available_at": None,
            "text": str(row.current_excerpt)[:1500],
        })

    for item, ts in zip(
        result, rows["_available"]
    ):
        item["available_at"] = (
            pd.Timestamp(ts).isoformat()
        )

    return result



def make_custom_kg(kg_rows, sec_rows):
    """
    Convert an existing historical Temporal KG snapshot
    into official LightRAG custom KG format.

    Preserve original source events.
    Exclude self-loops from the LightRAG import only.
    Audit every excluded relationship.
    """

    entities = {}
    relationships = []
    chunks = []
    manifest = []
    excluded = []

    for edge in kg_rows:
        source_id = "KG_" + edge["event_id"]

        head = str(edge["head"]).strip()
        tail = str(edge["tail"]).strip()
        relation = str(edge["relation"])

        head_id = str(edge["head_id"])
        tail_id = str(edge["tail_id"])

        if head == tail:
            excluded.append({
                "event_id": edge["event_id"],
                "head_id": head_id,
                "tail_id": tail_id,
                "decoded_head": head,
                "decoded_tail": tail,
                "relation": relation,
                "reason": (
                    "original_self_loop"
                    if head_id == tail_id
                    else "decoded_name_collision"
                )
            })
            continue

        description = (
            f"Historical KG relation: "
            f"{head} --[{relation}]--> {tail}. "
            f"Original head ID: {head_id}. "
            f"Original tail ID: {tail_id}. "
            f"Source: {edge['source']}. "
            f"Available at: {edge['available_at']}. "
            f"Original relation ID: {edge['relation_id']}. "
            f"Event ID: {edge['event_id']}. "
            f"Do not infer causality from generic "
            f"relations such as Relate_To or Has."
        )

        for name in (head, tail):
            if name not in entities:
                entities[name] = {
                    "entity_name": name,
                    "entity_type": "UNKNOWN",
                    "description": (
                        f"Financial knowledge graph entity: {name}. "
                        "Entity type and economic implications "
                        "must not be guessed."
                    ),
                    "source_id": source_id,
                }

        chunks.append({
            "content": description,
            "source_id": source_id,
            "source_chunk_index": 0,
        })

        relationships.append({
            "src_id": head,
            "tgt_id": tail,
            "description": description,
            "keywords": relation.replace("_", " "),
            "weight": 1.0,
            "source_id": source_id,
        })

        manifest.append({
            "source_id": source_id,
            "original_event_id": edge["event_id"],
            "available_at": edge["available_at"],
        })

    for item in sec_rows:
        source_id = "SEC_" + item["evidence_id"]

        chunks.append({
            "content": (
                f"SEC filing: {item['ticker']} "
                f"{item['form']}. "
                f"Available at: {item['available_at']}. "
                f"Evidence ID: {item['evidence_id']}.\n"
                f"{item['text']}"
            ),
            "source_id": source_id,
            "source_chunk_index": 0,
        })

        manifest.append({
            "source_id": source_id,
            "original_evidence_id": item["evidence_id"],
            "available_at": item["available_at"],
        })

    audit_path = OUT / "excluded_self_loops.json"
    audit_path.write_text(
        json.dumps(
            excluded,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[LIGHTRAG] KG input events:", len(kg_rows))
    print("[LIGHTRAG] Imported relationships:", len(relationships))
    print("[LIGHTRAG] Excluded self-loops:", len(excluded))
    print("[LIGHTRAG] Self-loop audit:", audit_path)

    return {
        "entities": list(entities.values()),
        "relationships": relationships,
        "chunks": chunks,
    }, manifest


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker", default="AAPL",
        choices=TICKERS
    )
    parser.add_argument(
        "--date", default="2021-12-30"
    )
    parser.add_argument(
        "--kg-k", type=int, default=80
    )
    parser.add_argument(
        "--sec-k", type=int, default=5
    )

    args = parser.parse_args()

    date = pd.Timestamp(
        args.date
    ).normalize()

    if not (
        pd.Timestamp("2018-01-01")
        <= date
        < pd.Timestamp("2022-01-01")
    ):
        raise RuntimeError(
            "Only 2018–2021 is allowed for this run."
        )

    from ollama import Client

    client = Client(host=HOST)

    models = {
        x.model
        for x in client.list().models
    }

    for needed in (LLM_MODEL, EMBED_MODEL):
        if needed not in models:
            raise RuntimeError(
                f"Required Ollama model unavailable: "
                f"{needed}"
            )

    print("[OLLAMA]", HOST)
    print("[LLM]", LLM_MODEL)
    print("[EMBED]", EMBED_MODEL)

    entity_names = load_entity_names()
    relation_names = load_relation_names()
    companies = load_company_mapping()

    kg_rows = load_historical_kg(
        args.ticker,
        date,
        args.kg_k,
        entity_names,
        relation_names,
        companies,
    )

    sec_rows = load_historical_sec(
        args.ticker,
        date,
        args.sec_k,
    )

    print("[KG SELECTED]", len(kg_rows))
    print("[SEC SELECTED]", len(sec_rows))

    if not kg_rows and not sec_rows:
        raise RuntimeError(
            "NO_EVIDENCE: neither historical KG "
            "nor SEC evidence is available."
        )

    if not kg_rows:
        print(
            "[SOURCE MISSING] KG unavailable; "
            "SEC-only snapshot candidate.",
            flush=True,
        )

    if not sec_rows:
        print(
            "[SOURCE MISSING] SEC unavailable; "
            "KG-only snapshot candidate.",
            flush=True,
        )

    print("[KG SAMPLE]")
    print(json.dumps(
        kg_rows[:3],
        indent=2,
        ensure_ascii=False,
    ))

    custom, source_manifest = make_custom_kg(
        kg_rows, sec_rows
    )

    # Fingerprint includes input evidence and model
    # settings to prevent mixing different snapshots.
    payload = json.dumps(
        {
            "kg": kg_rows,
            "sec": sec_rows,
            "llm": LLM_MODEL,
            "embedding": EMBED_MODEL,
            "import_transform_version": "selfloop_filter_v2",
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    fingerprint = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    workspace = (
        OUT /
        "snapshots" /
        f"{args.ticker}_{date.date()}_{fingerprint}"
    )

    workspace.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        workspace / "source_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            source_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # BGE-M3 embedding dimensionality is 1024.
    # LightRAG official Ollama integration.
    rag = LightRAG(
        working_dir=str(workspace),
        embedding_func_max_async=2,
        embedding_batch_num=5,
        default_embedding_timeout=600,
        default_llm_timeout=600,
        llm_model_name=LLM_MODEL,
        llm_model_func=ollama_model_complete,
        llm_model_kwargs={
            "host": HOST,
            "options": {"num_ctx": 8192},
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=EMBED_MODEL,
                host=HOST,
            ),
        ),
    )

    await rag.initialize_storages()

    try:
        ready_marker = workspace / "index_ready.json"

        if not ready_marker.exists():

            print(
                "[LIGHTRAG] Importing existing "
                "Temporal KG snapshot..."
            )

            # Official API, not custom retrieval.
            await rag.ainsert_custom_kg(custom)

            ready_marker.write_text(
                json.dumps({
                    "fingerprint": fingerprint,
                    "kg_events": len(kg_rows),
                    "sec_chunks": len(sec_rows),
                }, indent=2),
                encoding="utf-8",
            )

        print("[LIGHTRAG] Running hybrid retrieval")

        query = (
            f"Using only indexed historical evidence, "
            f"summarize the business and financial "
            f"risks relevant to {companies[args.ticker]['name']} "
            f"as of {date.date()}. "
            f"Use both KG relationships and SEC filings "
            f"when supported. "
            f"Relate_To and Has are generic relationships; "
            f"do not interpret them as suppliers, "
            f"customers, competitors, or causality. "
            f"Distinguish directly supported evidence "
            f"from uncertain inference. "
            f"Cite source IDs when available. "
            f"If evidence is insufficient, say so."
        )

        answer = await rag.aquery(
            query,
            param=QueryParam(
                mode="hybrid",
                stream=False,
            ),
        )

        result = {
            "ticker": args.ticker,
            "decision_date": str(date.date()),
            "method": "Official LightRAG Hybrid",
            "llm": LLM_MODEL,
            "embedding": EMBED_MODEL,
            "ollama_host": HOST,
            "kg_events": len(kg_rows),
            "sec_chunks": len(sec_rows),
            "answer": str(answer),
            "snapshot_fingerprint": fingerprint,
            "strict_pit_certified": False,
            "claim_grounding_verified": False,
            "ppo_integrated": False,
        }

        output = (
            OUT /
            f"lightrag_{args.ticker}_{date.date()}.json"
        )

        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n========== OFFICIAL LIGHTRAG RESULT ==========")
        print(str(answer)[:10000])

        print("\n[SAVED]", output)
        print("[KG EVENTS]", len(kg_rows))
        print("[SEC CHUNKS]", len(sec_rows))
        print("[SNAPSHOT]", workspace)

    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
