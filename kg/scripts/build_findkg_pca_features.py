from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT = Path(
    "kg/data/findkg/"
    "findkg_finrl_daily_2018_2022.csv"
)

OUTPUT = Path(
    "kg/data/findkg/"
    "findkg_finrl_daily_pca_2018_2022.csv"
)

MODEL_DIR = Path(
    "kg/data/findkg/pca_models"
)

TRAIN_END = pd.Timestamp("2021-12-31")


def main():

    print("=" * 76)
    print("LOAD FINDKG DAILY DATA")
    print("=" * 76)

    df = pd.read_csv(INPUT)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    emb_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("findkg_emb_")
        ]
    )

    if len(emb_cols) != 200:
        raise RuntimeError(
            f"Expected 200 raw FinDKG dims, "
            f"found {len(emb_cols)}"
        )

    train_mask = (
        df["date"] <= TRAIN_END
    )

    print("rows:", len(df))
    print("raw dims:", len(emb_cols))
    print("train rows:", int(train_mask.sum()))
    print(
        "date:",
        df["date"].min(),
        "->",
        df["date"].max(),
    )

    X = (
        df[emb_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if X.isna().any().any():
        raise RuntimeError(
            "NaN/Inf found in raw FinDKG embeddings."
        )

    X = X.to_numpy(
        dtype=np.float32
    )

    X_train = X[
        train_mask.to_numpy()
    ]

    # ========================================================
    # Train-only standardization
    # ========================================================

    print()
    print("=" * 76)
    print("FIT TRAIN-ONLY STANDARD SCALER")
    print("=" * 76)

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    X_train_scaled = scaler.transform(
        X_train
    )

    X_all_scaled = scaler.transform(
        X
    )

    # ========================================================
    # PCA32
    #
    # PCA16 is simply first 16 PCs from SAME PCA basis.
    # This gives nested feature sets:
    # PCA16 ⊂ PCA32
    # ========================================================

    print()
    print("=" * 76)
    print("FIT TRAIN-ONLY PCA32")
    print("=" * 76)

    pca32 = PCA(
        n_components=32,
        random_state=0,
    )

    pca32.fit(
        X_train_scaled
    )

    Z32 = pca32.transform(
        X_all_scaled
    ).astype(
        np.float32
    )

    Z16 = Z32[:, :16]

    pca16_var = float(
        pca32.explained_variance_ratio_[
            :16
        ].sum()
    )

    pca32_var = float(
        pca32.explained_variance_ratio_.sum()
    )

    print(
        "PCA16 explained variance:",
        f"{pca16_var:.6f}",
    )

    print(
        "PCA32 explained variance:",
        f"{pca32_var:.6f}",
    )

    # ========================================================
    # Add PCA columns
    # ========================================================

    for i in range(16):

        df[
            f"findkg_pca16_{i:02d}"
        ] = Z16[:, i]

    for i in range(32):

        df[
            f"findkg_pca32_{i:02d}"
        ] = Z32[:, i]

    # ========================================================
    # QA
    # ========================================================

    pca16_cols = [
        f"findkg_pca16_{i:02d}"
        for i in range(16)
    ]

    pca32_cols = [
        f"findkg_pca32_{i:02d}"
        for i in range(32)
    ]

    print()
    print("=" * 76)
    print("PCA QA")
    print("=" * 76)

    print(
        "PCA16 dims:",
        len(pca16_cols)
    )

    print(
        "PCA32 dims:",
        len(pca32_cols)
    )

    print(
        "missing PCA16:",
        int(
            df[pca16_cols]
            .isna()
            .sum()
            .sum()
        )
    )

    print(
        "missing PCA32:",
        int(
            df[pca32_cols]
            .isna()
            .sum()
            .sum()
        )
    )

    print(
        "duplicate date/ticker:",
        int(
            df.duplicated(
                ["date", "ticker"]
            ).sum()
        )
    )

    print()
    print("split:")
    print(
        df.groupby("split").size()
    )

    print()
    print("ticker:")
    print(
        df.groupby("ticker").size()
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    joblib.dump(
        scaler,
        MODEL_DIR
        /
        "findkg_train_scaler.joblib",
    )

    joblib.dump(
        pca32,
        MODEL_DIR
        /
        "findkg_train_pca32.joblib",
    )

    pd.DataFrame(
        {
            "n_components": [
                16,
                32,
            ],
            "explained_variance": [
                pca16_var,
                pca32_var,
            ],
        }
    ).to_csv(
        MODEL_DIR
        /
        "explained_variance.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("PCA BUILD COMPLETE")
    print("=" * 76)

    print("OUTPUT:", OUTPUT)


if __name__ == "__main__":
    main()
