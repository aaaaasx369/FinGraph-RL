#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kg/data/findkg/findkg_full_edges_with_dates.csv"

rows = []
with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("=" * 70)
print("1. FINDKG DUPLICATE ANALYSIS")
print("=" * 70)

def triple_key(r):
    return (
        r["subject_id"],
        r["relation_id"],
        r["object_id"],
        r["date"],
    )

groups = defaultdict(list)

for i, r in enumerate(rows):
    groups[triple_key(r)].append((i, r))

duplicates = {
    k: v for k, v in groups.items()
    if len(v) > 1
}

extra = sum(len(v) - 1 for v in duplicates.values())

print("Original rows:", len(rows))
print("Unique temporal triples:", len(groups))
print("Extra duplicate occurrences:", extra)
print("Duplicate groups:", len(duplicates))

print("\nDuplicate multiplicity:")
for n, count in sorted(
    Counter(len(v) for v in duplicates.values()).items()
):
    print(f"  multiplicity={n}: {count} groups")

print("\nSample duplicate groups:")
for key, values in list(duplicates.items())[:8]:
    print("TRIPLE:", key)
    print("COUNT:", len(values))
    print(
        "SPLITS:",
        Counter(
            (
                v["original_split"],
                v["finrl_split"],
                v["time_id"],
            )
            for _, v in values
        )
    )

print("\n" + "=" * 70)
print("2. DUPLICATES ACROSS SPLITS")
print("=" * 70)

cross_split = 0

for values in duplicates.values():
    splits = {
        (v["original_split"], v["finrl_split"])
        for _, v in values
    }
    if len(splits) > 1:
        cross_split += 1

print("Duplicate groups crossing splits:", cross_split)

print("\n" + "=" * 70)
print("3. DATE COVERAGE")
print("=" * 70)

dates = sorted({r["date"] for r in rows})

print("First date:", dates[0])
print("Last date:", dates[-1])
print("Number of weeks:", len(dates))

by_date = Counter(r["date"] for r in rows)

print("\nLast 12 weeks:")
for d in dates[-12:]:
    print(d, by_date[d])

print("\n" + "=" * 70)
print("4. ENTITY DICTIONARY DISCOVERY")
print("=" * 70)

roots = [
    ROOT / "kg/data",
    ROOT / "references/FinDKG",
    ROOT / "references/Financial-TKG",
    ROOT / "outputs/finrl_dkg_original",
    ROOT / "outputs/finrl_dkg_original_full",
]

keywords = (
    "entity2id",
    "entity2idx",
    "entity_id",
    "entity.dict",
    "entities.dict",
    "entity_map",
    "id2entity",
    "relation2id",
    "relation.dict",
    "entities.txt",
)

candidates = []

for root in roots:
    if not root.exists():
        continue

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        name = p.name.lower()

        if any(word in name for word in keywords):
            candidates.append(p)

for p in sorted(set(candidates))[:70]:
    print(p.relative_to(ROOT))

    if p.stat().st_size > 2_000_000:
        print("  (large file; sample omitted)")
        continue

    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                print("  ", line.strip()[:180])
    except Exception as e:
        print("  Read error:", e)

print("\nTOTAL MAPPING CANDIDATES:", len(set(candidates)))

print("\n" + "=" * 70)
print("5. FINDKG SPLIT COVERAGE")
print("=" * 70)

for field in ["original_split", "finrl_split"]:
    print("\n", field)
    for split, count in Counter(
        r[field] for r in rows
    ).most_common():
        split_dates = [
            r["date"] for r in rows
            if r[field] == split
        ]
        print(
            f"{split}: {count:,} rows, "
            f"{min(split_dates)} to {max(split_dates)}"
        )

print("\nAUDIT COMPLETE")
