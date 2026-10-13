from pathlib import Path
import pandas as pd
import json

ROOT = Path.cwd()

OUT = (
    ROOT /
    "outputs/finrl_dkg_original_full/four_relation_kg_v2"
)

base = pd.read_csv(
    OUT / "four_relation_candidates.csv"
)

sector = pd.read_csv(
    OUT / "same_sector_historical_candidates.csv"
)

base["evidence_group"] = [
    f"BASE-{i}" for i in range(len(base))
]

base["derivation"] = "EXTRACTED_CANDIDATE"

inverse = []

for _, row in base.iterrows():
    if row["relation"] == "SUPPLIES_TO":
        inverse_relation = "CUSTOMER_OF"
    elif row["relation"] == "CUSTOMER_OF":
        inverse_relation = "SUPPLIES_TO"
    else:
        continue

    new = row.copy()
    new["head"] = row["tail"]
    new["tail"] = row["head"]
    new["relation"] = inverse_relation
    new["derivation"] = "LOGICAL_INVERSE"

    # Keep identical evidence group and timestamps.
    inverse.append(new)

if inverse:
    base = pd.concat(
        [base, pd.DataFrame(inverse)],
        ignore_index=True,
    )

sector["filing_date"] = pd.NA
sector["available_at"] = pd.NA
sector["human_verified"] = False
sector["pit_verified"] = False
sector["evidence_group"] = [
    f"RSR-{i}" for i in range(len(sector))
]
sector["derivation"] = "UNDATED_RSR_CLASSIFICATION"

for col in base.columns:
    if col not in sector.columns:
        sector[col] = pd.NA

merged = pd.concat(
    [base, sector[base.columns]],
    ignore_index=True,
)

merged = merged.drop_duplicates(
    subset=[
        "head", "relation", "tail",
        "evidence_group",
    ]
)

merged["pit_eligible"] = False

# A filing date alone is not a verified
# publication/acceptance timestamp.
merged["pit_block_reason"] = merged["derivation"].map({
    "EXTRACTED_CANDIDATE":
        "UNVERIFIED_SOURCE_AND_TIMESTAMP",
    "LOGICAL_INVERSE":
        "INHERITS_UNVERIFIED_SOURCE_AND_TIMESTAMP",
    "UNDATED_RSR_CLASSIFICATION":
        "MISSING_HISTORICAL_VALID_FROM",
})

merged.to_csv(
    OUT / "four_relation_all_candidates.csv",
    index=False,
)

print("\n=== FOUR RELATIONS MERGED ===")

print(
    merged.groupby("relation")
    .agg(
        candidate_edges=("head", "size"),
        independent_evidence_groups=(
            "evidence_group", "nunique"
        ),
    )
    .to_string()
)

print("\n=== HISTORICALLY CERTIFIED ===")
print(
    merged.groupby("relation")["pit_eligible"]
    .sum()
    .to_string()
)

print("\nSaved:", OUT / "four_relation_all_candidates.csv")
print("Model training: NOT STARTED")
print("Strict PIT: NOT CERTIFIED")
