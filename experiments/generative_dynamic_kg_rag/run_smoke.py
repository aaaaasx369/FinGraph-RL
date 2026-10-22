import argparse
import json
from pathlib import Path

import ollama
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/generative_dynamic_kg_rag"
OUT.mkdir(parents=True, exist_ok=True)

KG_FILE = ROOT / "outputs/unified_temporal_kg_v3/unified_temporal_edges.csv"
SEC_FILE = ROOT / "results/aegis_rl_v1/v11f_sec_pit/sec_pit_retrieval.parquet"

ALIASES = {
    "AAPL": ["AAPL", "Apple Inc", "Apple"],
    "AMZN": ["AMZN", "Amazon.com", "Amazon"],
    "GOOGL": ["GOOGL", "GOOG", "Alphabet Inc", "Alphabet", "Google"],
    "MSFT": ["MSFT", "Microsoft Corporation", "Microsoft"],
    "NVDA": ["NVDA", "NVIDIA Corporation", "NVIDIA"],
}

client = ollama.Client(
    host="http://140.113.87.164:11434",
    timeout=300,
)



def retrieve_kg(ticker, date, k):
    df = pd.read_csv(KG_FILE, low_memory=False)

    # Verified against portfolio_entity_map.csv
    # and unified_temporal_kg_v3/entity_mapping_report.csv
    entity_map = {
        "AAPL": ["FINDKG:9", "COMPANY:AAPL"],
        "AMZN": ["FINDKG:20", "COMPANY:AMZN"],
        "GOOGL": ["FINDKG:11", "COMPANY:GOOGL"],
        "MSFT": ["FINDKG:30", "COMPANY:MSFT"],
        "NVDA": ["FINDKG:221", "COMPANY:NVDA"],
    }

    entities = set(entity_map[ticker])

    available = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce"
    ).dt.tz_localize(None).dt.normalize()

    # Strictly earlier than decision date
    valid_mask = available.notna() & (available < date)

    company_mask = (
        df["head"].astype(str).isin(entities)
        | df["tail"].astype(str).isin(entities)
    )

    matched = df.loc[valid_mask & company_mask].copy()
    matched["_available"] = available.loc[matched.index]

    print("[KG TARGET ENTITIES]", sorted(entities))
    print("[KG MATCHED HISTORICAL]", len(matched))

    if matched.empty:
        print("[KG] No historical matching edges")
        return []

    # First smoke test: most recent relations.
    # Not yet official LightRAG hybrid retrieval.
    matched = matched.sort_values(
        ["_available", "event_id"],
        ascending=[False, True],
        kind="stable"
    ).head(k)

    results = []

    for n, (_, row) in enumerate(matched.iterrows(), 1):
        results.append({
            "id": f"KG{n}",
            "event_id": str(row["event_id"]),
            "head": str(row["head"]),
            "relation": str(row["relation"]),
            "tail": str(row["tail"]),
            "available_at": str(row["_available"].date()),
            "source": str(row["source"]),
            "confidence": (
                float(row["confidence"])
                if pd.notna(row["confidence"])
                else None
            )
        })

    return results


def retrieve_sec(ticker, date, k):
    df = pd.read_parquet(SEC_FILE)

    print("[SEC ROWS]", len(df))

    available = pd.to_datetime(
        df["pit_usable_from"],
        utc=True,
        errors="coerce"
    ).dt.tz_localize(None).dt.normalize()

    mask = (
        df["ticker"].astype(str).str.upper().eq(ticker)
        & available.notna()
        & (available < date)
    )

    selected = df.loc[mask].copy()
    selected["_available"] = available.loc[selected.index]

    selected["current_excerpt"] = (
        selected["current_excerpt"].fillna("").astype(str)
    )

    selected = selected.loc[
        selected["current_excerpt"].str.strip().ne("")
    ]

    selected = selected.sort_values(
        "_available", ascending=False, kind="stable"
    ).head(k)

    result = []

    for n, (_, row) in enumerate(selected.iterrows(), 1):
        result.append({
            "id": f"SEC{n}",
            "evidence_id": str(row["evidence_id"]),
            "available_at": str(row["_available"].date()),
            "form": str(row["form"]),
            "text": row["current_excerpt"][:1600],
        })

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL", choices=list(ALIASES))
    p.add_argument("--date", default="2021-12-30")
    p.add_argument("--model", default="qwen3.5:27b")
    p.add_argument("--kg-k", type=int, default=10)
    p.add_argument("--sec-k", type=int, default=5)
    args = p.parse_args()

    date = pd.Timestamp(args.date).normalize()

    if not (
        pd.Timestamp("2018-01-01")
        <= date
        < pd.Timestamp("2022-01-01")
    ):
        raise RuntimeError("Only 2018–2021 dates are allowed")

    print("[OLLAMA]", "http://140.113.87.164:11434")
    print("[MODEL]", args.model)
    print("[DATE]", str(date.date()))
    print("[TICKER]", args.ticker)

    kg = retrieve_kg(args.ticker, date, args.kg_k)
    sec = retrieve_sec(args.ticker, date, args.sec_k)

    print("[KG RETRIEVED]", len(kg))
    print("[SEC RETRIEVED]", len(sec))

    print("[KG PREVIEW]")
    print(json.dumps(kg[:2], ensure_ascii=False, indent=2))

    print("[SEC PREVIEW]")
    print(json.dumps(sec[:1], ensure_ascii=False, indent=2)[:2500])

    if not kg:
        raise RuntimeError(
            "KG entity mapping unresolved. "
            "Do not generate a KG-grounded answer without KG evidence."
        )

    if not sec:
        raise RuntimeError(
            "No eligible nonempty SEC excerpts."
        )

    evidence = {
        "ticker": args.ticker,
        "date": str(date.date()),
        "kg": kg,
        "sec": sec,
    }

    system = """
You are a financial evidence-grounded research assistant.

Use ONLY the provided SEC and temporal KG evidence.
Do not invent financial events, relationships, or dates.
Do not interpret coded FINDKG_REL relations without a mapping.
Do not predict future stock prices.
Separate directly supported facts from uncertain inference.
Cite supplied evidence IDs for each claim.
A citation ID alone does not prove semantic support.
If evidence is insufficient, return null financial signals.

Return JSON:
{
  "summary": "...",
  "claims": [
    {"claim": "...", "evidence_ids": ["SEC1", "KG1"]}
  ],
  "impact": null,
  "risk": null,
  "confidence": null,
  "grounding_status": "insufficient"
}

impact: number in [-1,1] or null
risk: number in [0,1] or null
confidence: number in [0,1] or null
grounding_status: supported, partial, insufficient
"""

    response = client.chat(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(evidence, ensure_ascii=False),
            },
        ],
        format="json",
        options={"temperature": 0},
    )

    generated = json.loads(response.message.content)

    valid_ids = {x["id"] for x in kg + sec}

    for item in generated.get("claims", []):
        refs = item.get("evidence_ids", [])
        if not isinstance(refs, list) or not refs:
            raise RuntimeError("Claim missing evidence references")
        if not set(refs).issubset(valid_ids):
            raise RuntimeError("Claim contains unknown evidence ID")

    for name, low, high in [
        ("impact", -1, 1),
        ("risk", 0, 1),
        ("confidence", 0, 1),
    ]:
        value = generated.get(name)
        if value is not None:
            number = float(value)
            if not low <= number <= high:
                raise RuntimeError(f"Invalid {name}: {value}")

    output = {
        "ticker": args.ticker,
        "date": str(date.date()),
        "model": args.model,
        "kg_evidence": kg,
        "sec_evidence": sec,
        "generation": generated,
        "strict_pit_certified": False,
        "grounding_semantically_verified": False,
    }

    destination = OUT / f"smoke_{args.ticker}_{date.date()}.json"

    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== QWEN RESULT ==========")
    print(json.dumps(generated, ensure_ascii=False, indent=2))
    print("[SAVED]", destination)


if __name__ == "__main__":
    main()
