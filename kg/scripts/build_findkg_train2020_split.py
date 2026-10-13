from pathlib import Path

import pandas as pd


ROOT = Path(
    "external/FinDKG-FinRL"
)

SRC = (
    ROOT
    / "data"
    / "FinDKG"
)

DST = Path(
    "external/FinDKG-FinRL-Train2020/"
    "data/FinDKG"
)

TIME_MAP = Path(
    "kg/data/findkg/time_map.csv"
)


# ============================================================
# Load all currently available edges
# ============================================================

frames = []

for name in [
    "train.txt",
    "valid.txt",
    "test.txt",
]:

    p = SRC / name

    df = pd.read_csv(
        p,
        sep="\t",
        header=None,
    )

    # FinDKG uses:
    # head relation tail time [ignored]
    if df.shape[1] < 4:
        raise RuntimeError(
            f"{p}: expected >=4 columns, "
            f"got {df.shape[1]}"
        )

    while df.shape[1] < 5:
        df[df.shape[1]] = 0

    df = df.iloc[:, :5]

    df.columns = [
        "head",
        "relation",
        "tail",
        "time_id",
        "ignored",
    ]

    frames.append(df)


edges = pd.concat(
    frames,
    ignore_index=True,
)

# Existing splits should be disjoint.
# Drop exact duplicates defensively.
edges = edges.drop_duplicates(
    subset=[
        "head",
        "relation",
        "tail",
        "time_id",
    ]
).reset_index(drop=True)


# ============================================================
# Time map
# ============================================================

tm = pd.read_csv(
    TIME_MAP
)

print("time-map columns:", tm.columns.tolist())


time_col = None
date_col = None

for c in tm.columns:

    lc = str(c).lower()

    if lc in {
        "time_id",
        "timeid",
        "id",
    }:
        time_col = c

    if lc in {
        "date",
        "date_wk",
        "week",
    }:
        date_col = c


if time_col is None:
    raise RuntimeError(
        "Could not detect time-id column"
    )

if date_col is None:
    raise RuntimeError(
        "Could not detect date column"
    )


tm[time_col] = pd.to_numeric(
    tm[time_col],
    errors="coerce",
)

tm[date_col] = pd.to_datetime(
    tm[date_col],
    errors="coerce",
)


lookup = dict(
    zip(
        tm[time_col].astype(int),
        tm[date_col],
    )
)


edges["date"] = (
    edges["time_id"]
    .astype(int)
    .map(lookup)
)


if edges["date"].isna().any():

    raise RuntimeError(
        "Some FinDKG time IDs have no date."
    )


# ============================================================
# Leakage-free split
#
# Train weights:
# <= 2020-12-20
#
# One final 2020 week is reserved because DKG loader
# requires a non-empty valid split.
# It will NOT be used for model selection.
#
# Test:
# 2021-01-01 ... 2022-12-31
# ============================================================

TRAIN_END = pd.Timestamp(
    "2020-12-20"
)

BRIDGE_DATE = pd.Timestamp(
    "2020-12-27"
)

TEST_START = pd.Timestamp(
    "2021-01-01"
)

TEST_END = pd.Timestamp(
    "2022-12-31"
)


train = edges[
    edges["date"] <= TRAIN_END
].copy()

valid = edges[
    edges["date"] == BRIDGE_DATE
].copy()

test = edges[
    (edges["date"] >= TEST_START)
    &
    (edges["date"] <= TEST_END)
].copy()


for name, df in [
    ("train", train),
    ("valid", valid),
    ("test", test),
]:

    if len(df) == 0:
        raise RuntimeError(
            f"{name} split is empty"
        )


# ============================================================
# QA
# ============================================================

print()
print("=" * 72)
print("FINDKG TRAIN2020 SPLIT")
print("=" * 72)

for name, df in [
    ("TRAIN", train),
    ("BRIDGE", valid),
    ("TEST", test),
]:

    print()
    print(name)
    print("rows:", len(df))
    print(
        "date:",
        df["date"].min(),
        "->",
        df["date"].max(),
    )
    print(
        "weeks:",
        df["time_id"].nunique(),
    )


# No overlap
assert (
    set(train["time_id"])
    &
    set(test["time_id"])
) == set()

assert train["date"].max() < test["date"].min()

print()
print(
    "TRAIN/TEST temporal overlap: 0"
)


# ============================================================
# Write dataset
# ============================================================

DST.mkdir(
    parents=True,
    exist_ok=True,
)


def write_split(
    df,
    filename,
):

    (
        df[
            [
                "head",
                "relation",
                "tail",
                "time_id",
                "ignored",
            ]
        ]
        .to_csv(
            DST / filename,
            sep="\t",
            header=False,
            index=False,
        )
    )


write_split(
    train,
    "train.txt",
)

write_split(
    valid,
    "valid.txt",
)

write_split(
    test,
    "test.txt",
)


# Copy metadata
for name in [
    "entity2id.txt",
    "relation2id.txt",
    "time2id.txt",
    "stat.txt",
]:

    src = SRC / name

    if not src.exists():
        raise FileNotFoundError(
            src
        )

    (
        DST
        /
        name
    ).write_bytes(
        src.read_bytes()
    )


print()
print("OUTPUT:", DST)
print("DONE")
