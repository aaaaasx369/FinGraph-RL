#!/usr/bin/env python3
"""Late Fusion V6: LightRAG + FinDKG + validated Temporal ReflectKG.

- Reuses existing V4 LightRAG index; never runs index.
- Reuses V5 historical FinDKG retrieval.
- Uses only the separate validated ReflectKG V4 event store.
- Enforces a strict date-only, before-as_of cutoff on supported SEC data.
- Creates clean evidence for an optional Ollama/Qwen call.
- Keeps baseline (potentially invalid) ReflectKG out of the prompt.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path


DEFAULT_ROOT = "rag/data/lightrag_pilot"
DEFAULT_STORE = "outputs/reflectkg_temporal_v4/events.json"
DEFAULT_OLLAMA = "http://140.113.87.164:11434"
COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet Google",
}

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        obj, indent=2, ensure_ascii=False
    ), encoding="utf-8")

def cutoff_before(d, as_of):
    try:
        return date.fromisoformat(str(d)[:10]) < date.fromisoformat(as_of)
    except (ValueError, TypeError):
        return False

def default_query(ticker):
    name = COMPANY[ticker]
    return (
        f"{name} {ticker}: identify material financial and operational "
        "risks, disclosed changes, demand, revenue, supply dependencies, "
        "competition and industry developments. "
        "Ground every conclusion in SEC filing evidence."
    )

def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)

def find_manifest(args):
    folder = Path(args.index_root) / args.ticker / args.as_of
    candidates = []
    for p in folder.glob("*/manifest.json"):
        data = load(p)
        if (
            data.get("ticker") == args.ticker
            and data.get("as_of") == args.as_of
            and data.get("llm") == args.llm
            and data.get("embed") == args.embed
            and data.get("embed_dim") == args.embed_dim
            and data.get("max_docs") == args.max_docs
            and data.get("max_chars_per_doc") == args.max_chars
        ):
            candidates.append((p, data))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one matching index manifest, found "
            f"{len(candidates)} in {folder}. "
            "Do not rebuild automatically; verify Pilot parameters."
        )
    p, data = candidates[0]
    docs = data.get("documents", [])
    if not docs:
        raise RuntimeError("Manifest has no SEC documents")
    for doc in docs:
        if not cutoff_before(doc.get("published_at"), args.as_of):
            raise RuntimeError(
                f"Future/same-day SEC document in index: {doc}"
            )
    return str(p), data

def run_v4(args, cache_path):
    cmd = [
        sys.executable, "late_fusion_v4.py", "query",
        "--ticker", args.ticker,
        "--as-of", args.as_of,
        "--sec", args.sec,
        "--query", args.query,
        "--llm", args.llm,
        "--embed", args.embed,
        "--embed-dim", str(args.embed_dim),
        "--index-root", args.index_root,
        "--max-docs", str(args.max_docs),
        "--max-chars-per-doc", str(args.max_chars),
        "--num-ctx", str(args.num_ctx),
        "--out", cache_path,
    ]
    run(cmd)
    return load(cache_path)

def run_v5(args, cache_path):
    cmd = [
        sys.executable, "late_fusion_v5.py", "query",
        "--ticker", args.ticker,
        "--as-of", args.as_of,
        "--days", str(args.days),
        "--out", cache_path,
    ]
    run(cmd)
    return load(cache_path)

def run_reflect(args, cache_path):
    cmd = [
        sys.executable,
        "kg/scripts/reflectkg_temporal_v3.py", "query",
        "--store", args.reflect_store,
        "--ticker", args.ticker,
        "--as-of", args.as_of,
        "--days", str(args.days),
        "--hops", str(args.hops),
        "--out", cache_path,
    ]
    run(cmd)
    return load(cache_path)

def get_findkg(v5):
    """Inspect several known/reasonable V5 layouts; never use ReflectKG."""
    candidates = [
        ("findkg.snapshot", (v5.get("findkg") or {}).get("snapshot")
            if isinstance(v5.get("findkg"), dict) else None),
        ("findkg.edges", (v5.get("findkg") or {}).get("edges")
            if isinstance(v5.get("findkg"), dict) else None),
        ("fin_snapshot", v5.get("fin_snapshot")),
        ("findkg_snapshot", v5.get("findkg_snapshot")),
        ("fin_evidence", v5.get("fin_evidence")),
    ]
    for location, value in candidates:
        if isinstance(value, list):
            return location, value
    return None, None

def prepare_sec(v4, manifest, as_of):
    raw = v4.get("text_evidence", [])
    if not isinstance(raw, list):
        raise RuntimeError("V4 text_evidence is not a list")

    documents = manifest["documents"]
    document_ids = {d.get("doc_id"): d for d in documents}
    result = []
    skipped = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        # SEC V4 often embeds Document: DOC_ID in text.
        match = re.search(r"Document:\s*(SEC_[A-Za-z0-9_-]+)", text)
        doc_id = match.group(1) if match else None

        if doc_id not in document_ids:
            # Only infer for a one-document pilot if source is
            # explicitly a snippet of that point-in-time index.
            if len(documents) == 1 and item.get("lightrag_chunk_id"):
                doc_id = documents[0]["doc_id"]
                confidence = "manifest_single_document_inference"
            else:
                skipped.append("chunk_without_verified_document")
                continue
        else:
            confidence = "embedded_document_id_matches_manifest"

        published = document_ids[doc_id]["published_at"]
        if not cutoff_before(published, as_of):
            skipped.append("not_strictly_before_as_of")
            continue

        result.append({
            "evidence_id": f"SEC_{len(result)+1:03d}",
            "source": "LightRAG_SEC",
            "doc_id": doc_id,
            "published_at": published,
            "lightrag_chunk_id": item.get("lightrag_chunk_id"),
            "text": text[:5000],
            "provenance_status": confidence,
            "chunk_exact_match_verified": False,
        })

    return result, skipped

def prepare_findkg(edges, as_of):
    out = []
    rejected = 0
    for edge in edges:
        if not isinstance(edge, dict):
            rejected += 1
            continue

        # An observation date is required for a lookahead-safe
        # financial backtest. A missing date is not trusted.
        observed = (
            edge.get("date")
            or edge.get("observed_at")
            or edge.get("published_at")
            or edge.get("snapshot_date")
        )
        if not observed or not cutoff_before(observed, as_of):
            rejected += 1
            continue

        out.append({
            "evidence_id": f"FIN_{len(out)+1:03d}",
            "source": "FinDKG",
            "observed_at": str(observed)[:10],
            "record": edge,
            "date_semantics": "dataset_observation_not_event_creation",
        })
    return out, rejected

def prepare_reflect(r, as_of):
    snapshots = []
    changes = []
    for item in r.get("snapshot", []):
        if not cutoff_before(item.get("last_observed_at"), as_of):
            continue
        snapshots.append({
            "evidence_id": f"REF_{len(snapshots)+1:03d}",
            "source": "ReflectKG_V4",
            **item,
        })
    for item in r.get("changes", []):
        if not cutoff_before(item.get("published_at"), as_of):
            continue
        changes.append({
            "evidence_id": f"EVT_{len(changes)+1:03d}",
            "source": "ReflectKG_V4",
            **item,
        })
    return snapshots, changes

def finite_score(value):
    import math
    if isinstance(value, bool):
        raise ValueError("Boolean is not a score")
    value = float(value)
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Score must be finite and between -1 and 1")
    return value

def call_qwen(args, fused):
    allowed = set()
    context = []

    for item in fused["sec_evidence"][:5]:
        allowed.add(item["evidence_id"])
        context.append({
            "id": item["evidence_id"],
            "source": "SEC",
            "published_at": item["published_at"],
            "text": item["text"][:1600],
        })
    for item in fused["findkg_evidence"][:12]:
        allowed.add(item["evidence_id"])
        context.append({
            "id": item["evidence_id"],
            "source": "FinDKG",
            "observed_at": item["observed_at"],
            "record": item["record"],
        })
    for item in fused["reflectkg_snapshot"][:12]:
        allowed.add(item["evidence_id"])
        context.append({
            "id": item["evidence_id"],
            "source": "ReflectKG",
            "relation": {
                k: item.get(k) for k in (
                    "subject", "relation", "object",
                    "first_observed_at", "last_observed_at",
                    "latest_evidence",
                )
            }
        })
    for item in fused["reflectkg_changes"][:12]:
        allowed.add(item["evidence_id"])
        context.append({
            "id": item["evidence_id"],
            "source": "ReflectKG_Event",
            "event_type": item.get("event_type"),
            "date": item.get("published_at"),
            "relation": [
                item.get("subject"), item.get("relation"),
                item.get("object")
            ],
        })

    instructions = (
        "You analyze historical financial risk evidence. "
        "Use ONLY the provided evidence and date. "
        "Treat company graph edges as observed evidence, not "
        "proof of new or ended economic relationships. "
        "Do not invent citations. "
        "Return one JSON object with impact, risk, confidence, "
        "graph_change, event_intensity (all numbers in [-1,1]), "
        "citations (array of evidence IDs), and reasoning (string). "
        "impact: negative means adverse outlook, positive favorable. "
        "risk: larger means higher risk. "
        "graph_change: estimated importance of OBSERVED relationship "
        "updates, not number of actual corporate changes. "
        "event_intensity: salience of recent evidence. "
        "If evidence is insufficient, keep confidence low."
    )
    prompt = json.dumps({
        "ticker": args.ticker,
        "as_of": args.as_of,
        "evidence": context,
    }, ensure_ascii=False)

    url = args.ollama_url.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model": args.llm,
        "system": instructions,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": args.num_ctx, "temperature": 0},
    }).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        generated = json.load(response)

    parsed = json.loads(generated["response"])
    fields = (
        "impact", "risk", "confidence",
        "graph_change", "event_intensity"
    )
    features = {k: finite_score(parsed[k]) for k in fields}
    citations = parsed.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("Qwen citations must be a list")
    if any(not isinstance(c, str) or c not in allowed
           for c in citations):
        raise ValueError(
            "Qwen returned citations not present in prompt evidence"
        )
    return {
        **features,
        "citations": list(dict.fromkeys(citations)),
        "reasoning": str(parsed.get("reasoning", "")),
        "model": args.llm,
        "evidence_ids_available": len(allowed),
        "validation": "schema_checked_not_factually_verified",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True,
                        choices=sorted(COMPANY))
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--sec", required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--hops", type=int, choices=[1, 2], default=1)
    parser.add_argument("--index-root", default=DEFAULT_ROOT)
    parser.add_argument("--reflect-store", default=DEFAULT_STORE)
    parser.add_argument("--llm", default="qwen2.5:3b")
    parser.add_argument("--embed", default="nomic-embed-text:latest")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--max-docs", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--query", default=None)
    parser.add_argument("--invoke-qwen", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    date.fromisoformat(args.as_of)
    args.query = args.query or default_query(args.ticker)

    out = args.out or (
        f"outputs/v6/{args.ticker}_{args.as_of}_fused.json"
    )
    prefix = Path(out).with_suffix("")
    manifest_path, manifest = find_manifest(args)

    # V4 itself might use Ollama for keyword extraction.
    # However this never re-indexes SEC documents.
    v4 = run_v4(args, str(prefix) + ".v4.json")
    v5 = run_v5(args, str(prefix) + ".v5.json")
    reflect = run_reflect(args, str(prefix) + ".reflect.json")

    if v4.get("ticker") != args.ticker:
        raise RuntimeError("V4 ticker mismatch")
    if v4.get("query") != args.query:
        raise RuntimeError("V4 query mismatch")
    if v4.get("as_of") != args.as_of:
        raise RuntimeError("V4 as_of mismatch")

    sec, sec_rejected = prepare_sec(
        v4, manifest, args.as_of
    )
    fin_path, raw_edges = get_findkg(v5)
    if raw_edges is None:
        raise RuntimeError(
            "Cannot identify FinDKG snapshot in V5 output. "
            "Inspect the saved .v5.json and update get_findkg(). "
            "No empty-graph fallback will be fabricated."
        )
    fin, fin_rejected = prepare_findkg(
        raw_edges, args.as_of
    )
    snapshot, changes = prepare_reflect(reflect, args.as_of)

    # Fail closed for a non-empty raw graph whose record structure
    # does not expose dates.
    if raw_edges and not fin:
        raise RuntimeError(
            f"V5 returned {len(raw_edges)} FinDKG records but "
            "none passed observation-date validation. "
            "Inspect .v5.json rather than silently dropping edges."
        )

    fused = {
        "schema_version": "late_fusion_v6",
        "ticker": args.ticker,
        "as_of": args.as_of,
        "query": args.query,
        "retrieval": {
            "sec": "existing_LightRAG_V4_pilot",
            "findkg": "V5_temporal_graph",
            "reflectkg": "V4_validated_temporal_store",
        },
        "index_manifest": manifest_path,
        "indexed_documents": manifest["documents"],
        "sec_evidence": sec,
        "findkg_evidence": fin,
        "reflectkg_snapshot": snapshot,
        "reflectkg_changes": changes,
        "counts": {
            "sec": len(sec),
            "findkg": len(fin),
            "reflectkg_snapshot": len(snapshot),
            "reflectkg_changes": len(changes),
        },
        "financial_features": None,
        "warnings": [
            "Pilot LightRAG covers only selected SEC documents.",
            "SEC chunk-to-document assignment derives from index "
            "manifest, not verified byte-exact chunk provenance.",
            "FinDKG dates are observations, not relation start dates.",
            "ReflectKG ACCEPT is rule-based, not human ground truth.",
            "Date-only lookahead policy excludes same-day data.",
            f"V4 SEC rejected: {len(sec_rejected)}",
            f"V5 FinDKG records rejected: {fin_rejected}",
        ],
        "diagnostics": {
            "findkg_v5_path": fin_path,
            "v4_cache": str(prefix) + ".v4.json",
            "v5_cache": str(prefix) + ".v5.json",
            "reflect_cache": str(prefix) + ".reflect.json",
        },
    }

    # Keep failed Qwen calls from destroying validated evidence.
    save(out, fused)
    print("Evidence fusion written:", out, flush=True)

    if args.invoke_qwen:
        fused["financial_features"] = call_qwen(args, fused)
        save(out, fused)

    print(json.dumps({
        "out": out,
        "counts": fused["counts"],
        "findkg_v5_path": fin_path,
        "qwen_called": args.invoke_qwen,
        "features_validated": fused["financial_features"] is not None,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
