import csv
from collections import Counter, defaultdict
from datetime import date

EDGE = "kg/data/findkg/findkg_full_edges_with_dates.csv"
BASE = "external/FinDKG-FinRL-Train2020/data/FinDKG/"

def mapping(filename):
    result = {}
    with open(BASE + filename, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                result[p[1]] = p[0]
    return result

entities = mapping("entity2id.txt")
relations = mapping("relation2id.txt")

edges = defaultdict(list)

with open(EDGE, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        d = date.fromisoformat(r["date"])
        edges[d].append(r)

dates = sorted(edges)

def related(e, eid):
    return eid in (e["subject_id"], e["object_id"])

def signature(e):
    return (
        e["subject_id"],
        e["relation_id"],
        e["object_id"]
    )

def window_rows(eid, ds):
    return [
        e for d in ds for e in edges[d]
        if related(e, eid)
    ]

def tv_distance(a, b):
    if not a or not b:
        return None
    ca = Counter(e["relation_id"] for e in a)
    cb = Counter(e["relation_id"] for e in b)
    keys = set(ca) | set(cb)
    return 0.5 * sum(
        abs(ca[k] / len(a) - cb[k] / len(b))
        for k in keys
    )

def jaccard(a, b):
    sa = {signature(e) for e in a}
    sb = {signature(e) for e in b}
    if not sa or not sb:
        return None
    return 1 - len(sa & sb) / len(sa | sb)

print("===== NVIDIA ENTITY INSPECTION =====")

for eid in ["221", "664"]:
    print("\nENTITY:", eid, entities.get(eid))
    matches = [
        e for d in dates
        if d <= date(2021, 6, 1)
        for e in edges[d] if related(e, eid)
    ]
    matches.sort(key=lambda e: e["date"], reverse=True)

    for e in matches[:12]:
        print(
            e["date"],
            entities.get(e["subject_id"], e["subject_id"]),
            "--",
            relations.get(e["relation_id"], e["relation_id"]),
            "-->",
            entities.get(e["object_id"], e["object_id"])
        )

print("\n===== GRAPH METRIC COMPARISON =====")

portfolio = {
    "AAPL": "9",
    "MSFT": "30",
    "NVDA": "221",
    "AMZN": "20",
    "GOOGL": "11"
}

for query in [
    date(2021, 5, 11),
    date(2021, 5, 18),
    date(2021, 5, 25),
    date(2021, 6, 1)
]:
    eligible = [d for d in dates if d <= query]
    print("\nDATE:", query)

    for ticker, eid in portfolio.items():
        current_dates = eligible[-4:]
        previous_dates = eligible[-8:-4]

        current = window_rows(eid, current_dates)
        previous = window_rows(eid, previous_dates)

        j = jaccard(previous, current)
        tv = tv_distance(previous, current)

        print(
            ticker,
            "previous:", len(previous),
            "current:", len(current),
            "edge_turnover:",
            round(j, 4) if j is not None else None,
            "relation_change:",
            round(tv, 4) if tv is not None else None
        )

print("\nResearch warning: publication availability not verified.")
