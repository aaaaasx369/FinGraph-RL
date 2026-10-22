import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/unified_event_memory_v2"
OUT.mkdir(parents=True, exist_ok=True)

V1 = ROOT / "outputs/unified_event_memory_v1/unified_events.csv"
NEWS = ROOT / "kg/data/findkg/findkg_full_edges_with_dates.csv"
PORTFOLIO = ROOT / "kg/data/findkg/portfolio_entity_map.csv"
ENTITY = ROOT / "references/FinDKG/FinDKG_dataset/FinDKG-full/entity2id.txt"

def load_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

events = load_csv(V1)
news = load_csv(NEWS)
portfolio = load_csv(PORTFOLIO)

# 1. Retain original weekly triple multiplicity
counts = Counter(
    (
        r["subject_id"],
        r["relation_id"],
        r["object_id"],
        r["date"],
    )
    for r in news
)

# 2. Read original FinDKG entity vocabulary
names = {}
with ENTITY.open(encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            try:
                names[int(parts[1])] = parts[0].strip()
            except ValueError:
                pass

# Conservative ticker-to-name verification
aliases = {
    "AAPL": {"apple", "apple inc", "apple inc."},
    "MSFT": {"microsoft", "microsoft corporation", "microsoft corp"},
    "NVDA": {"nvidia", "nvidia corporation", "nvidia corp", "nvidia corp."},
    "AMZN": {"amazon", "amazon.com", "amazon.com inc", "amazon.com inc."},
    "GOOGL": {"alphabet", "alphabet inc", "alphabet inc.", "google"},
}

def normalize(s):
    return " ".join(s.casefold().replace(",", "").split())

verified = {}
mapping_report = []

for r in portfolio:
    ticker = r["ticker"].strip().upper()
    eid = int(r["entity_id"])
    name = names.get(eid, "")

    valid = (
        ticker in aliases
        and normalize(name) in {
            normalize(a) for a in aliases[ticker]
        }
    )

    if valid:
        verified[f"FINDKG:{eid}"] = f"COMPANY:{ticker}"

    mapping_report.append({
        "ticker": ticker,
        "findkg_id": eid,
        "entity_name": name,
        "verified": valid,
    })

# 3. Build unified edges / numeric signal records
edges = []
signals = []

for r in events:
    item = dict(r)

    item["head"] = verified.get(item["head"], item["head"])
    item["tail"] = verified.get(item["tail"], item["tail"])
    item["occurrence_count"] = 1

    if item["source"] == "FINDKG":
        key = (
            r["head"].split(":")[-1],
            r["relation"].split(":")[-1],
            r["tail"].split(":")[-1],
            r["event_date"],
        )
        item["occurrence_count"] = counts[key]

    # Keep SEC-RAG signals as node attributes,
    # not invented company-to-company edges.
    if item["source"] == "SEC_RAG":
        signals.append(item)
    else:
        edges.append(item)

def write_csv(path, data, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

fields = list(events[0].keys()) + ["occurrence_count"]

write_csv(OUT / "unified_edges.csv", edges, fields)
write_csv(OUT / "sec_numeric_signals.csv", signals, fields)
write_csv(
    OUT / "entity_mapping_report.csv",
    mapping_report,
    ["ticker", "findkg_id", "entity_name", "verified"],
)

print("=" * 60)
print("UNIFIED KG V2 - DATA INTEGRATION")
print("=" * 60)
print("Graph edges:", len(edges))
print("SEC numeric signals:", len(signals))
print("Total records:", len(edges) + len(signals))
print("FinDKG raw occurrences:", sum(
    r["occurrence_count"] for r in edges
    if r["source"] == "FINDKG"
))
print("\nVerified entity mappings:")
for r in mapping_report:
    print(r)
print("\nOutput:", OUT)
print("NOTE: This is a staging dataset, not yet PIT-certified.")
