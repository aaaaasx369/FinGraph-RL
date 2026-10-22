#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "outputs/unified_event_memory_v2"

OUT = ROOT / "outputs/unified_temporal_kg_v3"
OUT.mkdir(parents=True, exist_ok=True)

EDGES = SRC / "unified_edges.csv"
SIGNALS = SRC / "sec_numeric_signals.csv"

ENTITY_DICT = ROOT / (
    "references/FinDKG/FinDKG_dataset/"
    "FinDKG-full/entity2id.txt"
)

PORTFOLIO_MAP = ROOT / (
    "kg/data/findkg/portfolio_entity_map.csv"
)

def read_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

def normalize(s):
    return " ".join(
        s.lower().replace(",", "")
        .replace(".", "").split()
    )

def safe_number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# ============================================
# 1. Load existing data
# ============================================

edges = read_csv(EDGES)
signals = read_csv(SIGNALS)

print("Loaded graph edges:", len(edges))
print("Loaded SEC signals:", len(signals))

# ============================================
# 2. Load original FinDKG entity dictionary
# ============================================

findkg_names = {}

with ENTITY_DICT.open(
    encoding="utf-8", errors="replace"
) as f:
    for line in f:
        parts = line.strip().split("\t")

        if len(parts) < 2:
            continue

        try:
            eid = int(parts[1])
        except ValueError:
            continue

        findkg_names[eid] = parts[0].strip()

# ============================================
# 3. Canonical company mapping
# ============================================

aliases = {
    "AAPL": {"apple inc"},
    "MSFT": {"microsoft corporation"},
    "NVDA": {"nvidia corp", "nvidia corporation"},
    "AMZN": {"amazoncom inc", "amazon inc"},
    "GOOGL": {"alphabet inc"},
}

portfolio = read_csv(PORTFOLIO_MAP)

entity_alias_map = {}
mapping_report = []

for r in portfolio:
    ticker = r["ticker"].upper()
    eid = int(r["entity_id"])

    name = findkg_names.get(eid, "")
    normalized = normalize(name)

    accepted = {
        normalize(a)
        for a in aliases.get(ticker, set())
    }

    verified = bool(name) and normalized in accepted

    old_id = f"FINDKG:{eid}"
    new_id = f"COMPANY:{ticker}"

    if verified:
        entity_alias_map[old_id] = new_id

    mapping_report.append({
        "ticker": ticker,
        "old_id": old_id,
        "canonical_id": new_id,
        "entity_name": name,
        "verified": verified,
    })

def canonical(node):
    return entity_alias_map.get(node, node)

print("\nCompany mappings:")
for row in mapping_report:
    print(row)

if not all(r["verified"] for r in mapping_report):
    raise RuntimeError(
        "Some portfolio mappings are unverified. "
        "Stop before building Unified KG."
    )

# ============================================
# 4. Unified graph edges
# ============================================

graph_rows = []
event_nodes = []
seen_event_ids = set()

for r in edges:
    event_id = r["event_id"]

    if event_id in seen_event_ids:
        raise RuntimeError(
            f"Duplicate event_id: {event_id}"
        )

    seen_event_ids.add(event_id)

    head = canonical(r["head"])
    tail = canonical(r["tail"])

    graph_rows.append({
        "event_id": event_id,
        "head": head,
        "relation": r["relation"],
        "tail": tail,
        "available_at": r["available_at"],
        "event_date": r["event_date"],
        "source": r["source"],
        "time_precision": r["time_precision"],
        "persistence": r["persistence"],
        "occurrence_count": int(
            r["occurrence_count"]
        ),
        "confidence": r.get("confidence", ""),
        "source_id": r.get("source_id", ""),
    })

# ============================================
# 5. Convert SEC-RAG signals to event nodes
# ============================================

for r in signals:
    event_id = r["event_id"]

    if event_id in seen_event_ids:
        raise RuntimeError(
            f"Duplicate event_id: {event_id}"
        )

    seen_event_ids.add(event_id)

    company = canonical(r["head"])
    signal_node = "SEC_SIGNAL:" + event_id

    graph_rows.append({
        "event_id": event_id,
        "head": company,
        "relation": "HAS_SEC_SIGNAL",
        "tail": signal_node,
        "available_at": r["available_at"],
        "event_date": r["event_date"],
        "source": "SEC_RAG",
        "time_precision": r["time_precision"],
        "persistence": "EVENT",
        "occurrence_count": 1,
        "confidence": r.get("confidence", ""),
        "source_id": r.get("source_id", ""),
    })

    event_nodes.append({
        "node": signal_node,
        "company": company,
        "available_at": r["available_at"],
        "impact": r.get("impact", ""),
        "risk": r.get("risk", ""),
        "confidence": r.get("confidence", ""),
        "reason": r.get("evidence", ""),
    })

# ============================================
# 6. Global entity / relation vocabulary
# ============================================

entity_names = sorted({
    x
    for r in graph_rows
    for x in (r["head"], r["tail"])
})

relation_names = sorted({
    r["relation"] for r in graph_rows
})

entity2id = {
    name: i
    for i, name in enumerate(entity_names)
}

relation2id = {
    name: i
    for i, name in enumerate(relation_names)
}

for r in graph_rows:
    r["head_id"] = entity2id[r["head"]]
    r["relation_id"] = relation2id[r["relation"]]
    r["tail_id"] = entity2id[r["tail"]]

# ============================================
# 7. Sort chronological observations
# ============================================

graph_rows.sort(key=lambda r: (
    r["available_at"],
    r["event_id"]
))

# ============================================
# 8. Export all artifacts
# ============================================

fields = [
    "event_id",
    "head",
    "relation",
    "tail",
    "head_id",
    "relation_id",
    "tail_id",
    "available_at",
    "event_date",
    "source",
    "time_precision",
    "persistence",
    "occurrence_count",
    "confidence",
    "source_id",
]

write_csv(
    OUT / "unified_temporal_edges.csv",
    graph_rows,
    fields,
)

write_csv(
    OUT / "sec_signal_nodes.csv",
    event_nodes,
    [
        "node", "company", "available_at",
        "impact", "risk", "confidence", "reason"
    ],
)

write_csv(
    OUT / "entity_mapping_report.csv",
    mapping_report,
    [
        "ticker", "old_id", "canonical_id",
        "entity_name", "verified"
    ],
)

(OUT / "entity2id.json").write_text(
    json.dumps(
        entity2id, indent=2, ensure_ascii=False
    ),
    encoding="utf-8",
)

(OUT / "relation2id.json").write_text(
    json.dumps(
        relation2id, indent=2, ensure_ascii=False
    ),
    encoding="utf-8",
)

metadata = {
    "version": "unified_temporal_kg_v3",
    "entities": len(entity2id),
    "relations": len(relation2id),
    "edges": len(graph_rows),
    "source_counts": dict(Counter(
        r["source"] for r in graph_rows
    )),
    "earliest_date": graph_rows[0]["available_at"],
    "latest_date": graph_rows[-1]["available_at"],
    "time_policy": (
        "Preserve source-reported availability; "
        "not yet strict PIT certified"
    ),
    "relation_validity": (
        "Persistent candidates are observations, "
        "not verified open-ended validity"
    ),
}

(OUT / "metadata.json").write_text(
    json.dumps(
        metadata, indent=2, ensure_ascii=False
    ),
    encoding="utf-8",
)

# ============================================
# 9. Basic validations
# ============================================

assert len(graph_rows) == len(edges) + len(signals)

assert len(set(
    r["event_id"] for r in graph_rows
)) == len(graph_rows)

assert all(
    0 <= r["head_id"] < len(entity2id)
    and 0 <= r["tail_id"] < len(entity2id)
    and 0 <= r["relation_id"] < len(relation2id)
    for r in graph_rows
)

assert "COMPANY:NVDA" in entity2id
assert "FINDKG:221" not in entity2id

for ticker in [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
]:
    assert f"COMPANY:{ticker}" in entity2id

assert len(event_nodes) == len(signals)

print("\n" + "=" * 65)
print("UNIFIED TEMPORAL KG V3")
print("=" * 65)

print("Entities:", len(entity2id))
print("Relations:", len(relation2id))
print("Temporal edges:", len(graph_rows))
print("SEC signal nodes:", len(event_nodes))

print("\nSource counts:")
for source, count in Counter(
    r["source"] for r in graph_rows
).most_common():
    print(f"  {source}: {count:,}")

print("\nDate range:")
print(
    graph_rows[0]["available_at"],
    "→",
    graph_rows[-1]["available_at"]
)

print("\nPortfolio nodes:")
for ticker in [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
]:
    key = f"COMPANY:{ticker}"
    print(ticker, entity2id[key])

print("\nOutput:", OUT)
print("ALL BASIC VALIDATIONS PASSED")
print("NOTE: Not yet PIT-certified.")
print("NOTE: No KGTransformer training performed.")
