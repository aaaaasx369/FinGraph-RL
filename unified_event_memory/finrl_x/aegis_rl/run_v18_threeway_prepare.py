#!/usr/bin/env python3

import json
import csv
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path("results/aegis_rl_v1")
OUT = ROOT / "v18_threeway"
OUT.mkdir(parents=True, exist_ok=True)

V17 = ROOT / "v17_joint_rag_ppo_smoke"
B7 = ROOT / "v17b7_alignscore_20"
B8 = ROOT / "v17b8_event_outcome_20"

TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}
SEEDS = [42, 123, 2026]


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return rows


def save_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_key(row):
    return (row["date"], row["ticker"])


def impact_value(label):
    return {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0
    }.get(str(label).lower())


print("=" * 72)
print("V18 THREE-WAY RAG FUSION PREPARATION")
print("=" * 72)

original_path = V17 / "joint_rag_cache.jsonl"
b7_path = B7 / "audit.jsonl"
b8_path = B8 / "audit.jsonl"

original = load_jsonl(original_path)
b7 = load_jsonl(b7_path)
b8 = load_jsonl(b8_path)

print("\nOriginal Joint Cache:", len(original))
print("B7 Grounding Cases:", len(b7))
print("B8 Outcome Cases:", len(b8))

# ---------------------------------------------------------
# 1. Original Joint RAG
# ---------------------------------------------------------

original_by_key = {}

for row in original:
    if row.get("ticker") not in TICKERS:
        continue

    key = get_key(row)

    if key in original_by_key:
        raise ValueError(f"Duplicate original key: {key}")

    original_by_key[key] = row


# ---------------------------------------------------------
# 2. Independently Verified SEC Claims
# ---------------------------------------------------------

verified_claims = {}
verification_stats = Counter()

for row in b7:
    key = get_key(row)

    if key in verified_claims:
        raise ValueError(f"Duplicate B7 key: {key}")

    accepted = {}

    for claim in row.get("case", {}).get("claims", []):
        check = claim.get("final_check") or {}
        verification_stats[
            check.get("verification_status", "MISSING")
        ] += 1

        if (
            check.get("independent_verified") is True
            and check.get("verification_status") == "ALIGNSCORE_PASS"
            and check.get("deterministic_ok") is True
        ):
            claim_id = claim.get("claim_id")

            if claim_id in accepted:
                raise ValueError(
                    f"Duplicate claim ID: {key}, {claim_id}"
                )

            accepted[claim_id] = claim

    verified_claims[key] = accepted


# ---------------------------------------------------------
# 3. B8 Event and Outcome Classification
# ---------------------------------------------------------

b8_by_key = {}

for row in b8:
    key = get_key(row)

    if key in b8_by_key:
        raise ValueError(f"Duplicate B8 key: {key}")

    b8_by_key[key] = row


# ---------------------------------------------------------
# 4. Grounded Joint Features
# ---------------------------------------------------------

grounded = []
audit = []

for key in sorted(set(verified_claims) | set(b8_by_key)):
    date, ticker = key

    verified = verified_claims.get(key, {})
    outcome = b8_by_key.get(key, {})
    decisions = outcome.get("claim_decisions", [])

    verified_ids = set(verified)

    classified = []
    review_count = 0

    for decision in decisions:
        cid = decision.get("claim_id")
        state = decision.get("b8", {})
        status = state.get("status")

        if cid not in verified_ids:
            continue

        if status == "CLASSIFIED":
            classification = state.get("classification")
            if isinstance(classification, dict):
                classified.append(classification)
        else:
            review_count += 1

    potential_values = []
    realized_values = []

    for item in classified:
        potential = impact_value(
            item.get("potential_business_impact")
        )
        realized = impact_value(
            item.get("realized_business_impact")
        )

        if potential is not None:
            potential_values.append(potential)

        if realized is not None:
            realized_values.append(realized)

    # Unknown is NOT neutral.
    # Zero is only a placeholder accompanied by a mask.
    potential_known = int(bool(potential_values))
    realized_known = int(bool(realized_values))

    has_verified = int(bool(verified))
    has_classified = int(bool(classified))

    features = {
        "date": date,
        "ticker": ticker,

        "verified_claim_count": len(verified),
        "classified_claim_count": len(classified),
        "review_claim_count": review_count,

        "sec_verified_mask": has_verified,
        "event_classified_mask": has_classified,

        "potential_impact": (
            sum(potential_values) / len(potential_values)
            if potential_values else 0.0
        ),
        "potential_impact_known_mask": potential_known,

        "realized_impact": (
            sum(realized_values) / len(realized_values)
            if realized_values else 0.0
        ),
        "realized_impact_known_mask": realized_known,

        # Verified text does not imply a known investment direction.
        "grounded_signal_mask": int(
            potential_known or realized_known
        ),
    }

    grounded.append(features)

    audit.append({
        "date": date,
        "ticker": ticker,
        "verified_claim_ids": sorted(verified_ids),
        "b8_case_status": outcome.get("status", "MISSING"),
        "features": features,
    })


# ---------------------------------------------------------
# 5. Paired Comparison
# ---------------------------------------------------------

grounded_by_key = {
    get_key(row): row
    for row in grounded
}

all_keys = sorted(
    set(original_by_key) | set(grounded_by_key)
)

comparison = []

for key in all_keys:
    date, ticker = key

    orig = original_by_key.get(key)
    ground = grounded_by_key.get(key)

    comparison.append({
        "date": date,
        "ticker": ticker,
        "original_available": int(orig is not None),
        "grounded_available": int(ground is not None),
        "paired_available": int(
            orig is not None and ground is not None
        ),
        "verified_sec_mask": (
            ground["sec_verified_mask"] if ground else 0
        ),
        "grounded_signal_mask": (
            ground["grounded_signal_mask"] if ground else 0
        ),
    })


columns = [
    "date",
    "ticker",
    "verified_claim_count",
    "classified_claim_count",
    "review_claim_count",
    "sec_verified_mask",
    "event_classified_mask",
    "potential_impact",
    "potential_impact_known_mask",
    "realized_impact",
    "realized_impact_known_mask",
    "grounded_signal_mask",
]

save_csv(
    OUT / "grounded_features.csv",
    grounded,
    columns
)

save_csv(
    OUT / "pairing_audit.csv",
    comparison,
    [
        "date", "ticker",
        "original_available",
        "grounded_available",
        "paired_available",
        "verified_sec_mask",
        "grounded_signal_mask"
    ]
)

with (OUT / "grounded_audit.jsonl").open("w") as f:
    for row in audit:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# 6. Coverage and Provenance
# ---------------------------------------------------------

n = len(grounded)

stats = {
    "grounded_cases": n,
    "verified_sec_cases": sum(
        r["sec_verified_mask"] for r in grounded
    ),
    "classified_cases": sum(
        r["event_classified_mask"] for r in grounded
    ),
    "known_potential_cases": sum(
        r["potential_impact_known_mask"] for r in grounded
    ),
    "known_realized_cases": sum(
        r["realized_impact_known_mask"] for r in grounded
    ),
    "usable_numeric_cases": sum(
        r["grounded_signal_mask"] for r in grounded
    ),
    "paired_cases": sum(
        r["paired_available"] for r in comparison
    ),
    "original_cache_records": len(original_by_key),
    "verification_status_counts": dict(verification_stats),
}

manifest = {
    "experiment": "V18 Three-Way RAG Fusion Preparation",
    "arms": [
        "INDEPENDENT_FUSION",
        "UNVERIFIED_JOINT_FUSION",
        "GROUNDED_JOINT_FUSION"
    ],
    "seeds": SEEDS,
    "train": "2018-2020",
    "validation": "2021",
    "test_used": False,
    "ppo_timesteps_target": 30000,
    "sources": {
        str(original_path): sha256(original_path),
        str(b7_path): sha256(b7_path),
        str(b8_path): sha256(b8_path),
    },
    "statistics": stats,
    "important": [
        "No PPO training performed by this script.",
        "Unknown impact is masked, never treated as neutral.",
        "B7 AlignScore threshold is not calibrated.",
        "KG semantic correctness is not independently certified.",
        "Feature rows are sampled cases, not daily PPO tensors.",
        "Grounded impact averages are provisional features.",
        "Full PPO requires matched dates and feature dimensions.",
    ]
}

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False)
)

print("\n========== VERIFICATION COUNTS ==========")
for k, v in sorted(verification_stats.items()):
    print(f"{k:30s}: {v}")

print("\n========== FEATURE COVERAGE ==========")
for k, v in stats.items():
    print(f"{k:30s}: {v}")

print("\n========== OUTPUT ==========")
for p in sorted(OUT.iterdir()):
    print(p)

print("\nV18 FEATURE PREPARATION: COMPLETE")
print("PPO TRAINING: NOT STARTED")
