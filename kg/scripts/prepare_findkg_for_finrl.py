from pathlib import Path
import re
import pandas as pd


ROOT = Path("external/FinDKG/FinDKG_dataset/FinDKG-full")
OUT = Path("kg/data/findkg")
OUT.mkdir(parents=True, exist_ok=True)

PORTFOLIO = {
    "AAPL": 9,
    "GOOGL": 11,
    "AMZN": 20,
    "MSFT": 30,
    "NVDA": 221,
}


def load_time_map(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["TimeID"] = pd.to_numeric(
        df["TimeID"],
        errors="coerce"
    )

    df["DATE_WK"] = pd.to_datetime(
        df["DATE_WK"],
        errors="coerce"
    )

    return df.dropna()


def load_quadruples(path, split_name):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()

            if len(parts) < 4:
                continue

            try:
                subject = int(parts[0])
                relation = int(parts[1])
                obj = int(parts[2])
                time_id = int(parts[3])
            except ValueError:
                continue

            rows.append({
                "subject_id": subject,
                "relation_id": relation,
                "object_id": obj,
                "time_id": time_id,
                "original_split": split_name,
            })

    return pd.DataFrame(rows)


def main():

    print("=" * 72)
    print("LOAD FINDKG-FULL")
    print("=" * 72)

    time_map = load_time_map(
        ROOT / "time2id.txt"
    )

    print(
        "time range:",
        time_map["DATE_WK"].min(),
        "->",
        time_map["DATE_WK"].max()
    )

    print(
        "time steps:",
        len(time_map)
    )

    pieces = []

    for split in ["train", "valid", "test"]:

        p = ROOT / f"{split}.txt"

        df = load_quadruples(
            p,
            split
        )

        print(
            split,
            "rows:",
            len(df)
        )

        pieces.append(df)

    edges = pd.concat(
        pieces,
        ignore_index=True
    )

    edges = edges.merge(
        time_map.rename(
            columns={
                "TimeID": "time_id",
                "DATE_WK": "date"
            }
        ),
        on="time_id",
        how="left",
        validate="many_to_one"
    )

    missing_date = edges["date"].isna().sum()

    print(
        "missing dates:",
        missing_date
    )

    if missing_date:
        raise ValueError(
            f"{missing_date} edges have missing date mapping"
        )

    edges = edges.sort_values(
        ["date", "subject_id", "relation_id", "object_id"]
    ).reset_index(drop=True)

    print()
    print("=" * 72)
    print("FULL TEMPORAL GRAPH")
    print("=" * 72)

    print(
        "rows:",
        len(edges)
    )

    print(
        "range:",
        edges["date"].min(),
        "->",
        edges["date"].max()
    )

    print(
        "entities referenced:",
        pd.unique(
            pd.concat(
                [
                    edges["subject_id"],
                    edges["object_id"]
                ]
            )
        ).size
    )

    print(
        "relations:",
        edges["relation_id"].nunique()
    )

    print(
        "time steps:",
        edges["time_id"].nunique()
    )

    # --------------------------------------------------------
    # Our FinRL experiment split
    #
    # Train : 2018-01-01 -> 2021-12-31
    # Val   : 2022-01-01 -> 2022-06-30
    # Test  : 2022-07-01 -> 2023-01-01
    # --------------------------------------------------------

    edges["finrl_split"] = "other"

    edges.loc[
        (
            edges["date"] >= "2018-01-01"
        )
        &
        (
            edges["date"] <= "2021-12-31"
        ),
        "finrl_split"
    ] = "train"

    edges.loc[
        (
            edges["date"] >= "2022-01-01"
        )
        &
        (
            edges["date"] <= "2022-06-30"
        ),
        "finrl_split"
    ] = "val"

    edges.loc[
        (
            edges["date"] >= "2022-07-01"
        )
        &
        (
            edges["date"] <= "2023-01-01"
        ),
        "finrl_split"
    ] = "test"

    usable = edges[
        edges["finrl_split"] != "other"
    ].copy()

    print()
    print("=" * 72)
    print("FINRL TEMPORAL SPLITS")
    print("=" * 72)

    print(
        usable.groupby(
            "finrl_split"
        ).size()
    )

    print()

    print(
        usable.groupby(
            "finrl_split"
        )["date"].agg(
            ["min", "max", "nunique"]
        )
    )

    # --------------------------------------------------------
    # Portfolio entity QA
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PORTFOLIO ENTITY COVERAGE")
    print("=" * 72)

    portfolio_rows = []

    for ticker, entity_id in PORTFOLIO.items():

        mask = (
            (usable["subject_id"] == entity_id)
            |
            (usable["object_id"] == entity_id)
        )

        sub = usable[
            mask
        ]

        row = {
            "ticker": ticker,
            "entity_id": entity_id,
            "edge_count": len(sub),
            "first_date":
                sub["date"].min()
                if len(sub)
                else pd.NaT,
            "last_date":
                sub["date"].max()
                if len(sub)
                else pd.NaT,
            "train_edges":
                int(
                    (
                        sub["finrl_split"]
                        ==
                        "train"
                    ).sum()
                ),
            "val_edges":
                int(
                    (
                        sub["finrl_split"]
                        ==
                        "val"
                    ).sum()
                ),
            "test_edges":
                int(
                    (
                        sub["finrl_split"]
                        ==
                        "test"
                    ).sum()
                ),
        }

        portfolio_rows.append(row)

    portfolio_df = pd.DataFrame(
        portfolio_rows
    )

    print(
        portfolio_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    edges.to_csv(
        OUT / "findkg_full_edges_with_dates.csv",
        index=False
    )

    usable.to_csv(
        OUT / "findkg_finrl_edges_2018_2022.csv",
        index=False
    )

    portfolio_df.to_csv(
        OUT / "portfolio_entity_coverage.csv",
        index=False
    )

    pd.DataFrame(
        [
            {
                "ticker": ticker,
                "entity_id": entity_id
            }
            for ticker, entity_id
            in PORTFOLIO.items()
        ]
    ).to_csv(
        OUT / "portfolio_entity_map.csv",
        index=False
    )

    time_map.to_csv(
        OUT / "time_map.csv",
        index=False
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        OUT / "findkg_full_edges_with_dates.csv"
    )

    print(
        OUT / "findkg_finrl_edges_2018_2022.csv"
    )

    print(
        OUT / "portfolio_entity_map.csv"
    )

    print(
        OUT / "portfolio_entity_coverage.csv"
    )


if __name__ == "__main__":
    main()
