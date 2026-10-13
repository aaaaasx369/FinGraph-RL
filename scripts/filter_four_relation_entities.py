from pathlib import Path
import pandas as pd

BASE = Path(
    "outputs/finrl_dkg_original_full/four_relation_pipeline"
)

SOURCE = BASE / "relation_candidates.csv"

df = pd.read_csv(SOURCE, dtype=str).fillna("")

# Known securities / share-class aliases.
# Only GOOG is unambiguously another Alphabet share class here.
ALIASES = {
    "GOOG": "GOOGL",
}

# These ticker matches need entity review before use.
REVIEW_TICKERS = {
    "GOOGM",
    "GOOGN",
    "HXSCL",
    "SKHY",
    "BRK-A",
    "BRK-B",
}

accepted = []
rejected = []

for _, row in df.iterrows():
    r = row.to_dict()

    head = ALIASES.get(r["head"], r["head"])
    tail = ALIASES.get(r["tail"], r["tail"])

    reasons = []

    if head == tail:
        reasons.append("SAME_COMPANY_OR_SHARE_CLASS")

    if (
        r["head"] in REVIEW_TICKERS
        or r["tail"] in REVIEW_TICKERS
    ):
        reasons.append("ENTITY_IDENTITY_UNVERIFIED")

    if r["issuer"] == r["counterparty"]:
        reasons.append("SELF_ENTITY")

    if reasons:
        r["reject_reason"] = ";".join(reasons)
        rejected.append(r)
    else:
        r["head"] = head
        r["tail"] = tail
        r["review_status"] = "NEEDS_SEMANTIC_REVIEW"
        accepted.append(r)

columns = list(df.columns)

pd.DataFrame(
    accepted, columns=columns
).to_csv(
    BASE / "entity_filtered_candidates.csv",
    index=False
)

pd.DataFrame(
    rejected,
    columns=columns + ["reject_reason"],
).to_csv(
    BASE / "entity_rejected_candidates.csv",
    index=False
)

print("=== ENTITY FILTER ===")
print("Original candidates:", len(df))
print("Remaining candidates:", len(accepted))
print("Rejected/quarantined:", len(rejected))
print("Entity verification: NOT COMPLETE")
