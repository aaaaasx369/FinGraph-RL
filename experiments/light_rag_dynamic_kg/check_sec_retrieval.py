import asyncio
import json
from pathlib import Path
from functools import partial

import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import (
    ollama_model_complete,
    ollama_embed,
)
from lightrag.utils import EmbeddingFunc


ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "results/light_rag_dynamic_kg"

OUT = BASE / "sec_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TICKER = "AAPL"
DATE = "2021-12-30"

HOST = "http://140.113.87.164:11434"

result = json.loads(
    (BASE / f"lightrag_{TICKER}_{DATE}.json").read_text()
)

fingerprint = result["snapshot_fingerprint"]

snapshot = (
    BASE / "snapshots" /
    f"{TICKER}_{DATE}_{fingerprint}"
)

assert (snapshot / "index_ready.json").exists()

sec = pd.read_parquet(
    ROOT /
    "results/aegis_rl_v1/v11f_sec_pit/sec_pit_retrieval.parquet"
)

dates = pd.to_datetime(
    sec["pit_usable_from"],
    utc=True,
    errors="coerce",
).dt.tz_localize(None)

mask = (
    sec["ticker"].astype(str).eq(TICKER)
    & dates.notna()
    & (
        dates.dt.normalize()
        < pd.Timestamp(DATE)
    )
)

selected = sec.loc[mask].copy()
selected["_date"] = dates.loc[selected.index]

selected = selected[
    selected["current_excerpt"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
]

selected = (
    selected.sort_values(
        ["_date", "evidence_id"],
        ascending=[False, True],
    )
    .drop_duplicates("evidence_id")
    .head(5)
)

evidence = [
    {
        "id": str(r.evidence_id),
        "text": str(r.current_excerpt),
    }
    for r in selected.itertuples()
]

print("[SNAPSHOT]", snapshot)
print("[EXPECTED SEC CHUNKS]", len(evidence))


async def main():
    rag = LightRAG(
        working_dir=str(snapshot),

        llm_model_name="qwen3.5:27b",

        llm_model_func=ollama_model_complete,

        llm_model_kwargs={
            "host": HOST,
            "timeout": 300,
            "options": {"num_ctx": 8192},
        },

        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model="bge-m3:latest",
                host=HOST,
            ),
        ),
    )

    await rag.initialize_storages()

    try:
        queries = {
            "hybrid": (
                "Apple Inc. SEC filings, financial "
                "disclosures and business risk factors"
            ),
            "naive": (
                "Apple Inc. 2021 Form 10-K services "
                "business risks financial disclosures"
            ),
            "mix": (
                "Apple Inc. SEC 10-K and Knowledge "
                "Graph financial business risks"
            ),
        }

        for mode, query in queries.items():
            print("\n==========", mode.upper(), "==========")

            try:
                context = await rag.aquery(
                    query,
                    param=QueryParam(
                        mode=mode,
                        only_need_context=True,
                        enable_rerank=False,
                    ),
                )

                text = str(context)

                file = OUT / f"{mode}_context.txt"
                file.write_text(
                    text,
                    encoding="utf-8",
                )

                matched = []

                for item in evidence:
                    id_match = item["id"] in text

                    fragment = (
                        " ".join(item["text"].split())[:90]
                    )

                    context_normalized = " ".join(text.split())

                    text_match = (
                        len(fragment) >= 50
                        and fragment in context_normalized
                    )

                    if id_match or text_match:
                        matched.append(item["id"])

                print("[CONTEXT LENGTH]", len(text))
                print("[SEC MATCHED]", len(matched))
                print("[SEC IDs]", matched)
                print("[SAVED]", file)

            except Exception as exc:
                print("[MODE ERROR]", repr(exc))

    finally:
        await rag.finalize_storages()


asyncio.run(main())
