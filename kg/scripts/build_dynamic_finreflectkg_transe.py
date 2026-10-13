#!/usr/bin/env python3

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

KG_PATH = (
    ROOT
    / "kg/data/finreflectkg_5stocks_2018_2023.parquet"
)

SEC_PATH = (
    ROOT
    / "rag/data/processed/sec/sec_chunks.jsonl"
)

OUTPUT = (
    ROOT
    / "kg/output/finreflectkg_dynamic_transe_32d.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "kg/output/finreflectkg_dynamic_filing_map.csv"
)


TARGET_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

COMPANY_TYPES = {
    "ORG",
    "COMP",
}

DIM = 32
EPOCHS = 250
BATCH_SIZE = 512
LR = 0.01
MARGIN = 1.0
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# Helpers
# ============================================================

def clean_text(x):

    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    if (
        not x
        or x in {
            "nan",
            "none",
            "null",
        }
    ):
        return None

    return x


def parse_source(source_file):

    m = re.match(
        r"^([A-Za-z]+)_10k_(\d{4})\.pdf$",
        str(source_file).strip(),
        flags=re.I,
    )

    if not m:
        return None, None

    return (
        m.group(1).upper(),
        int(m.group(2)),
    )


# ============================================================
# TransE
# ============================================================

class TransE(nn.Module):

    def __init__(
        self,
        n_entities,
        n_relations,
        dim,
    ):

        super().__init__()

        self.entity = nn.Embedding(
            n_entities,
            dim,
        )

        self.relation = nn.Embedding(
            n_relations,
            dim,
        )

        bound = 6.0 / np.sqrt(dim)

        nn.init.uniform_(
            self.entity.weight,
            -bound,
            bound,
        )

        nn.init.uniform_(
            self.relation.weight,
            -bound,
            bound,
        )

        with torch.no_grad():

            self.entity.weight[:] = (
                F.normalize(
                    self.entity.weight,
                    p=2,
                    dim=1,
                )
            )


    def score(
        self,
        h,
        r,
        t,
    ):

        return torch.norm(
            self.entity(h)
            + self.relation(r)
            - self.entity(t),
            p=1,
            dim=1,
        )


# ============================================================
# SEC 10-K publication map
# ============================================================

docs = {}

with open(
    SEC_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        obj = json.loads(line)

        if obj.get("form") != "10-K":
            continue

        ticker = str(
            obj.get("ticker", "")
        ).upper()

        published_at = pd.to_datetime(
            obj.get("published_at"),
            errors="coerce",
        )

        if pd.isna(published_at):
            continue

        year = int(
            published_at.year
        )

        key = (
            ticker,
            year,
        )

        docs[key] = published_at.normalize()


print("=" * 100)
print("SEC 10-K PUBLICATION MAP")
print("=" * 100)

for key in sorted(docs):
    print(
        key,
        "->",
        docs[key].strftime("%Y-%m-%d"),
    )


# ============================================================
# KG
# ============================================================

kg = pd.read_parquet(
    KG_PATH
)

kg["entity_type"] = (
    kg["entity_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

kg["target_type"] = (
    kg["target_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

kg["relationship"] = (
    kg["relationship"]
    .astype(str)
    .str.lower()
    .str.strip()
)


parsed = kg["source_file"].map(
    parse_source
)

kg["source_ticker"] = [
    x[0]
    for x in parsed
]

kg["source_year"] = [
    x[1]
    for x in parsed
]


kg["available_at"] = [
    docs.get(
        (
            ticker,
            year,
        ),
        pd.NaT,
    )
    for ticker, year
    in zip(
        kg["source_ticker"],
        kg["source_year"],
    )
]


if kg["available_at"].isna().any():

    missing = (
        kg.loc[
            kg["available_at"].isna(),
            [
                "source_file",
                "source_ticker",
                "source_year",
            ],
        ]
        .drop_duplicates()
    )

    raise RuntimeError(
        "Missing publication dates:\n"
        + missing.to_string(
            index=False
        )
    )


# ============================================================
# Company-company only
# ============================================================

company = kg[
    kg["entity_type"].isin(
        COMPANY_TYPES
    )
    &
    kg["target_type"].isin(
        COMPANY_TYPES
    )
].copy()


company["head"] = (
    company["entity"]
    .map(clean_text)
)

company["tail"] = (
    company["target"]
    .map(clean_text)
)

company["rel"] = (
    company["relationship"]
    .map(clean_text)
)


company = company.dropna(
    subset=[
        "head",
        "tail",
        "rel",
        "available_at",
    ]
)


company = company[
    company["head"]
    != company["tail"]
].copy()


print("\n")
print("=" * 100)
print("COMPANY-COMPANY KG")
print("=" * 100)

print(
    "Triples:",
    len(company),
)

print(
    "Publication events:",
    company["available_at"].nunique(),
)


# ============================================================
# Filing audit
# ============================================================

audit = (
    kg[
        [
            "source_ticker",
            "source_year",
            "source_file",
            "available_at",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "available_at",
            "source_ticker",
        ]
    )
)

audit.to_csv(
    AUDIT_OUTPUT,
    index=False,
)

print("\nFILING UPDATE DATES")

print(
    audit.to_string(
        index=False
    )
)


# ============================================================
# Dynamic snapshots
# ============================================================

snapshot_dates = sorted(
    company["available_at"]
    .drop_duplicates()
    .tolist()
)

previous_model = None
previous_entity_to_id = {}
previous_relation_to_id = {}

rows = []


for snapshot_idx, snapshot_date in enumerate(
    snapshot_dates,
    start=1,
):

    print("\n")
    print("=" * 100)

    print(
        f"SNAPSHOT {snapshot_idx}/{len(snapshot_dates)} "
        f"THROUGH {pd.Timestamp(snapshot_date).strftime('%Y-%m-%d')}"
    )

    print("=" * 100)


    current = company[
        company["available_at"]
        <= snapshot_date
    ].copy()


    triples_df = (
        current[
            [
                "head",
                "rel",
                "tail",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


    entities = sorted(
        set(
            triples_df["head"]
        )
        |
        set(
            triples_df["tail"]
        )
    )

    relations = sorted(
        triples_df["rel"]
        .unique()
        .tolist()
    )


    entity_to_id = {
        x: i
        for i, x
        in enumerate(entities)
    }

    relation_to_id = {
        x: i
        for i, x
        in enumerate(relations)
    }


    triples = np.asarray(
        [
            [
                entity_to_id[h],
                relation_to_id[r],
                entity_to_id[t],
            ]
            for h, r, t
            in triples_df.itertuples(
                index=False,
                name=None,
            )
        ],
        dtype=np.int64,
    )


    print(
        "Entities:",
        len(entities),
    )

    print(
        "Relations:",
        len(relations),
    )

    print(
        "Unique triples:",
        len(triples),
    )


    model = TransE(
        len(entities),
        len(relations),
        DIM,
    ).to(DEVICE)


    # ========================================================
    # Warm start
    # ========================================================

    copied_entities = 0
    copied_relations = 0


    if previous_model is not None:

        with torch.no_grad():

            for entity, new_id in entity_to_id.items():

                if entity in previous_entity_to_id:

                    old_id = (
                        previous_entity_to_id[
                            entity
                        ]
                    )

                    model.entity.weight[
                        new_id
                    ].copy_(
                        previous_model.entity.weight[
                            old_id
                        ]
                    )

                    copied_entities += 1


            for relation, new_id in relation_to_id.items():

                if relation in previous_relation_to_id:

                    old_id = (
                        previous_relation_to_id[
                            relation
                        ]
                    )

                    model.relation.weight[
                        new_id
                    ].copy_(
                        previous_model.relation.weight[
                            old_id
                        ]
                    )

                    copied_relations += 1


    print(
        "Warm-start entities:",
        copied_entities,
    )

    print(
        "Warm-start relations:",
        copied_relations,
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )


    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
        + snapshot_idx
    )


    n = len(triples)


    # ========================================================
    # Training
    # ========================================================

    model.train()


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        perm = torch.randperm(
            n,
            generator=generator,
        )

        epoch_loss = 0.0


        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            idx = (
                perm[
                    start:
                    start + BATCH_SIZE
                ]
                .numpy()
            )

            batch = triples[idx]


            h = torch.tensor(
                batch[:, 0],
                dtype=torch.long,
                device=DEVICE,
            )

            r = torch.tensor(
                batch[:, 1],
                dtype=torch.long,
                device=DEVICE,
            )

            t = torch.tensor(
                batch[:, 2],
                dtype=torch.long,
                device=DEVICE,
            )


            neg_h = h.clone()
            neg_t = t.clone()


            corrupt_head = (
                torch.rand(
                    len(batch),
                    generator=generator,
                )
                < 0.5
            )


            random_entities = torch.randint(
                0,
                len(entities),
                (
                    len(batch),
                ),
                generator=generator,
            ).to(DEVICE)


            neg_h[
                corrupt_head
            ] = (
                random_entities[
                    corrupt_head
                ]
            )

            neg_t[
                ~corrupt_head
            ] = (
                random_entities[
                    ~corrupt_head
                ]
            )


            positive = model.score(
                h,
                r,
                t,
            )

            negative = model.score(
                neg_h,
                r,
                neg_t,
            )


            loss = torch.relu(
                MARGIN
                + positive
                - negative
            ).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            with torch.no_grad():

                model.entity.weight[:] = (
                    F.normalize(
                        model.entity.weight,
                        p=2,
                        dim=1,
                    )
                )


            epoch_loss += (
                loss.item()
                * len(batch)
            )


        epoch_loss /= n


        if (
            epoch == 1
            or epoch % 50 == 0
            or epoch == EPOCHS
        ):

            print(
                f"epoch={epoch:3d} "
                f"loss={epoch_loss:.6f}"
            )


    # ========================================================
    # Snapshot embeddings
    # ========================================================

    model.eval()


    with torch.no_grad():

        for ticker in TARGET_TICKERS:

            node = ticker.lower()


            if node in entity_to_id:

                emb = (
                    model.entity.weight[
                        entity_to_id[node]
                    ]
                    .cpu()
                    .numpy()
                    .copy()
                )

                available = 1

            else:

                emb = np.zeros(
                    DIM,
                    dtype=float,
                )

                available = 0


            row = {
                "snapshot_date":
                    pd.Timestamp(
                        snapshot_date
                    ).strftime(
                        "%Y-%m-%d"
                    ),

                "ticker":
                    ticker,

                "transe_available":
                    available,

                "kg_entities":
                    len(entities),

                "kg_relations":
                    len(relations),

                "kg_unique_triples":
                    len(triples),

                "warmstart_entities":
                    copied_entities,

                "warmstart_relations":
                    copied_relations,
            }


            for i, value in enumerate(
                emb
            ):

                row[
                    f"transe_{i:02d}"
                ] = float(
                    value
                )


            rows.append(row)


    previous_model = model
    previous_entity_to_id = entity_to_id
    previous_relation_to_id = relation_to_id


# ============================================================
# Save
# ============================================================

result = pd.DataFrame(
    rows
)

result = result.sort_values(
    [
        "snapshot_date",
        "ticker",
    ]
).reset_index(
    drop=True
)


result.to_csv(
    OUTPUT,
    index=False,
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Output:",
    OUTPUT
)

print(
    "Shape:",
    result.shape
)

print(
    "Snapshots:",
    result["snapshot_date"].nunique(),
)

print("\nAvailability by ticker:")

print(
    result.groupby(
        "ticker"
    )[
        "transe_available"
    ].sum()
)

print("\nSnapshot growth:")

print(
    result[
        [
            "snapshot_date",
            "kg_entities",
            "kg_relations",
            "kg_unique_triples",
        ]
    ]
    .drop_duplicates()
    .to_string(
        index=False
    )
)
