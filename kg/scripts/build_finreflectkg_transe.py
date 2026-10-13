#!/usr/bin/env python3

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg"
    / "data"
    / "finreflectkg_5stocks_2018_2023.parquet"
)

OUTPUT = (
    ROOT
    / "kg"
    / "output"
    / "finreflectkg_company_transe_32d.csv"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

YEARS = [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
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
# Cleaning
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
            self.entity.weight[:] = F.normalize(
                self.entity.weight,
                p=2,
                dim=1,
            )


    def score(
        self,
        h,
        r,
        t,
    ):

        h = self.entity(h)
        r = self.relation(r)
        t = self.entity(t)

        return torch.norm(
            h + r - t,
            p=1,
            dim=1,
        )


# ============================================================
# Load
# ============================================================

print("=" * 100)
print("LOAD FINREFLECTKG")
print("=" * 100)

df = pd.read_parquet(
    INPUT
)


df["entity_type"] = (
    df["entity_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["target_type"] = (
    df["target_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["relationship"] = (
    df["relationship"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce",
)


df = df.dropna(
    subset=[
        "entity",
        "target",
        "relationship",
        "year",
    ]
).copy()


df["year"] = (
    df["year"]
    .astype(int)
)


# ============================================================
# Company-company triples only
# ============================================================

company_df = df[
    df["entity_type"].isin(
        COMPANY_TYPES
    )
    &
    df["target_type"].isin(
        COMPANY_TYPES
    )
].copy()


company_df["head"] = (
    company_df["entity"]
    .map(clean_text)
)

company_df["tail"] = (
    company_df["target"]
    .map(clean_text)
)

company_df["rel"] = (
    company_df["relationship"]
    .map(clean_text)
)


company_df = company_df.dropna(
    subset=[
        "head",
        "tail",
        "rel",
    ]
)


company_df = company_df[
    company_df["head"]
    != company_df["tail"]
].copy()


print(
    "All triples:",
    len(df),
)

print(
    "Company-company triples:",
    len(company_df),
)

print(
    "\nTop company-company relations:"
)

print(
    company_df["rel"]
    .value_counts()
    .head(30)
)


# ============================================================
# Train year-by-year using cumulative triples
#
# Warm-start previous embeddings:
#
# 2018 model
#   ↓
# initialize common entities / relations in 2019
#   ↓
# continue training
#
# Therefore dimensions remain in a consistent coordinate system.
# ============================================================

previous = None
previous_entity_to_id = None
previous_relation_to_id = None

rows = []


for year in YEARS:

    print("\n")
    print("=" * 100)
    print(
        f"TRANSE CUMULATIVE THROUGH {year}"
    )
    print("=" * 100)


    ydf = company_df[
        company_df["year"]
        <= year
    ].copy()


    # Remove exact duplicate triples.
    ydf = (
        ydf[
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
            ydf["head"]
        )
        |
        set(
            ydf["tail"]
        )
    )

    relations = sorted(
        ydf["rel"]
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


    print(
        "Unique triples:",
        len(ydf),
    )

    print(
        "Entities:",
        len(entities),
    )

    print(
        "Relations:",
        len(relations),
    )


    triples = np.asarray(
        [
            [
                entity_to_id[h],
                relation_to_id[r],
                entity_to_id[t],
            ]
            for h, r, t
            in ydf[
                [
                    "head",
                    "rel",
                    "tail",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        ],
        dtype=np.int64,
    )


    model = TransE(
        n_entities=len(entities),
        n_relations=len(relations),
        dim=DIM,
    ).to(DEVICE)


    # ========================================================
    # Warm-start from previous year
    # ========================================================

    copied_entities = 0
    copied_relations = 0


    if previous is not None:

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
                        previous.entity.weight[
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
                        previous.relation.weight[
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
        SEED + year
    )


    n = len(triples)


    # ========================================================
    # Train
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

            idx = perm[
                start:
                start + BATCH_SIZE
            ].numpy()


            batch = triples[
                idx
            ]


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


            # -----------------------------------------------
            # Negative sampling
            # randomly corrupt head OR tail
            # -----------------------------------------------

            neg_h = h.clone()
            neg_t = t.clone()


            corrupt_head = torch.rand(
                len(batch),
                generator=generator,
            ) < 0.5


            random_entities = torch.randint(
                low=0,
                high=len(entities),
                size=(
                    len(batch),
                ),
                generator=generator,
            ).to(DEVICE)


            neg_h[
                corrupt_head
            ] = random_entities[
                corrupt_head
            ]

            neg_t[
                ~corrupt_head
            ] = random_entities[
                ~corrupt_head
            ]


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


            # Standard TransE entity norm constraint.
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
    # Extract exact portfolio company embeddings
    # ========================================================

    model.eval()


    with torch.no_grad():

        for ticker in TARGET_TICKERS:

            node = ticker.lower()


            if node in entity_to_id:

                vector = (
                    model.entity.weight[
                        entity_to_id[
                            node
                        ]
                    ]
                    .cpu()
                    .numpy()
                    .copy()
                )

                available = 1

            else:

                vector = np.zeros(
                    DIM,
                    dtype=float,
                )

                available = 0


            row = {
                "ticker":
                    ticker,

                "year":
                    year,

                "transe_available":
                    available,

                "kg_entities":
                    len(entities),

                "kg_relations":
                    len(relations),

                "kg_unique_triples":
                    len(ydf),

                "warmstart_entities":
                    copied_entities,

                "warmstart_relations":
                    copied_relations,
            }


            for i, value in enumerate(
                vector
            ):

                row[
                    f"transe_{i:02d}"
                ] = float(
                    value
                )


            rows.append(
                row
            )


            print(
                f"{ticker:5s}",
                "FOUND"
                if available
                else "MISSING",
            )


    # Keep trained model for next year's warm start.
    previous = model

    previous_entity_to_id = (
        entity_to_id
    )

    previous_relation_to_id = (
        relation_to_id
    )


# ============================================================
# Save
# ============================================================

result = pd.DataFrame(
    rows
)

result = result.sort_values(
    [
        "ticker",
        "year",
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


print("\nAvailability:")

print(
    result.groupby(
        "ticker"
    )[
        "transe_available"
    ].sum()
)


print(
    "\nKG growth:"
)

print(
    result[
        [
            "year",
            "kg_entities",
            "kg_relations",
            "kg_unique_triples",
            "warmstart_entities",
            "warmstart_relations",
        ]
    ]
    .drop_duplicates()
    .to_string(
        index=False
    )
)


print(
    "\nFirst rows:"
)

print(
    result.head()
    .to_string()
)
