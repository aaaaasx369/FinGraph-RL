from pathlib import Path
import shutil
import pandas as pd

SRC = Path("external/FinDKG/FinDKG_dataset/FinDKG-full")
DST = Path("external/FinDKG/FinDKG_dataset/FinDKG-FinRL")

DST.mkdir(parents=True, exist_ok=True)


def load_time_map(path):
    df = pd.read_csv(path)

    df.columns = [
        c.strip()
        for c in df.columns
    ]

    df["TimeID"] = pd.to_numeric(
        df["TimeID"],
        errors="coerce"
    )

    df["DATE_WK"] = pd.to_datetime(
        df["DATE_WK"],
        errors="coerce"
    )

    return df.dropna().copy()


def load_quadruples(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            parts = line.strip().split()

            if len(parts) < 4:
                continue

            try:
                s = int(parts[0])
                r = int(parts[1])
                o = int(parts[2])
                t = int(parts[3])

            except ValueError:
                continue

            rows.append(
                [s, r, o, t]
            )

    return pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "relation_id",
            "object_id",
            "time_id",
        ],
    )


def save_quadruples(df, path):
    with open(path, "w", encoding="utf-8") as f:

        for row in df.itertuples(index=False):

            f.write(
                f"{row.subject_id}\t"
                f"{row.relation_id}\t"
                f"{row.object_id}\t"
                f"{row.time_id}\n"
            )


def main():

    print("=" * 72)
    print("LOAD ORIGINAL FINDKG-FULL")
    print("=" * 72)

    time_map = load_time_map(
        SRC / "time2id.txt"
    )

    time_lookup = dict(
        zip(
            time_map["TimeID"].astype(int),
            time_map["DATE_WK"],
        )
    )

    pieces = []

    for split in [
        "train",
        "valid",
        "test",
    ]:

        df = load_quadruples(
            SRC / f"{split}.txt"
        )

        print(
            split,
            len(df)
        )

        pieces.append(df)

    all_edges = pd.concat(
        pieces,
        ignore_index=True
    )

    all_edges["date"] = (
        all_edges["time_id"]
        .map(time_lookup)
    )

    if all_edges["date"].isna().any():
        raise RuntimeError(
            "Missing time mapping."
        )

    print()
    print(
        "TOTAL:",
        len(all_edges)
    )

    print(
        "DATE RANGE:",
        all_edges["date"].min(),
        "->",
        all_edges["date"].max()
    )

    # ========================================================
    # New chronological split
    # ========================================================

    train = all_edges[
        (
            all_edges["date"]
            >=
            "2018-01-01"
        )
        &
        (
            all_edges["date"]
            <=
            "2021-12-31"
        )
    ].copy()

    valid = all_edges[
        (
            all_edges["date"]
            >=
            "2022-01-01"
        )
        &
        (
            all_edges["date"]
            <=
            "2022-06-30"
        )
    ].copy()

    test = all_edges[
        (
            all_edges["date"]
            >=
            "2022-07-01"
        )
        &
        (
            all_edges["date"]
            <=
            "2023-01-01"
        )
    ].copy()

    print()
    print("=" * 72)
    print("NEW SPLIT")
    print("=" * 72)

    for name, df in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:

        print(
            name,
            "rows:",
            len(df),
            "range:",
            df["date"].min(),
            "->",
            df["date"].max(),
            "timesteps:",
            df["time_id"].nunique(),
        )

    # ========================================================
    # Save in author's expected raw format
    # ========================================================

    save_quadruples(
        train[
            [
                "subject_id",
                "relation_id",
                "object_id",
                "time_id",
            ]
        ],
        DST / "train.txt",
    )

    save_quadruples(
        valid[
            [
                "subject_id",
                "relation_id",
                "object_id",
                "time_id",
            ]
        ],
        DST / "valid.txt",
    )

    save_quadruples(
        test[
            [
                "subject_id",
                "relation_id",
                "object_id",
                "time_id",
            ]
        ],
        DST / "test.txt",
    )

    # ========================================================
    # Copy metadata mappings unchanged
    # ========================================================

    for filename in [
        "entity2id.txt",
        "relation2id.txt",
        "time2id.txt",
        "stat.txt",
    ]:

        src = SRC / filename

        if src.exists():

            shutil.copy2(
                src,
                DST / filename,
            )

            print(
                "copied:",
                filename
            )

    # ========================================================
    # QA: portfolio edge counts
    # ========================================================

    portfolio = {
        "AAPL": 9,
        "GOOGL": 11,
        "AMZN": 20,
        "MSFT": 30,
        "NVDA": 221,
    }

    print()
    print("=" * 72)
    print("PORTFOLIO COVERAGE IN NEW SPLIT")
    print("=" * 72)

    for ticker, eid in portfolio.items():

        print()
        print(ticker, "entity", eid)

        for name, df in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:

            mask = (
                (df["subject_id"] == eid)
                |
                (df["object_id"] == eid)
            )

            print(
                f"  {name}:",
                int(mask.sum())
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        "Dataset written to:",
        DST
    )


if __name__ == "__main__":
    main()
