#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

NPZ = ROOT / (
    "results/light_rag_dynamic_kg/kg_dual_channel_v1/"
    "kg_dual_channel_finrl_aligned.npz"
)

JOBS = ROOT / (
    "results/light_rag_dynamic_kg/event_driven_v3/"
    "event_jobs.csv"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

MAX_AGE = 35


def simulate(dates, tickers, snapshot_days):
    covered = np.zeros(
        (len(dates), len(tickers)), dtype=bool
    )

    for j, ticker in enumerate(tickers):
        events = sorted(set(snapshot_days.get(ticker, [])))

        if not events:
            continue

        last = None
        p = 0

        for i, day in enumerate(dates):
            while p < len(events) and events[p] <= day:
                last = events[p]
                p += 1

            if last is None:
                continue

            age = (day - last).days

            if 0 <= age <= MAX_AGE:
                covered[i, j] = True

    return covered


def display(name, mask, n_train):
    tr = mask[:n_train]
    va = mask[n_train:]

    print(
        f"{name:32s}",
        f"TRAIN={tr.mean():7.2%}",
        f"VAL={va.mean():7.2%}",
        f"TRAIN_ROWS={tr.sum():5d}",
        f"VAL_ROWS={va.sum():5d}",
    )


def main():
    with np.load(NPZ, allow_pickle=True) as z:
        dates = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )

        tickers = [str(x) for x in z["tickers"]]
        n_train = int(z["n_train"])

        struct = z["structured"]
        names = [str(x) for x in z["feature_names"]]

    assert tickers == TICKERS

    jobs = pd.read_csv(JOBS)
    jobs["date"] = pd.to_datetime(jobs["date"])

    selected = jobs[jobs["status"] == "CANDIDATE"]

    candidate_days = {
        ticker: selected.loc[
            selected["ticker"] == ticker, "date"
        ].tolist()
        for ticker in TICKERS
    }

    all_trigger_days = {
        ticker: jobs.loc[
            jobs["ticker"] == ticker, "date"
        ].tolist()
        for ticker in TICKERS
    }

    existing_mask = (
        struct[:, :, names.index("retrieval_valid")]
        > 0.5
    )

    simulated_selected = simulate(
        dates, tickers, candidate_days
    )

    simulated_all = simulate(
        dates, tickers, all_trigger_days
    )

    print("=" * 95)
    print("EVENT-DRIVEN RAG COVERAGE SIMULATION")
    print("=" * 95)
    print("ASSUMPTION: EVERY NEW SNAPSHOT SUCCEEDS")
    print("ASSUMPTION: SNAPSHOT REMAINS USABLE FOR 35 DAYS")
    print("NOT AN ACTUAL RAG COVERAGE RESULT")
    print()

    display("CURRENT RETRIEVAL", existing_mask, n_train)
    display("EVENT-DRIVEN SELECTED", simulated_selected, n_train)
    display("ALL THRESHOLD-PASSING TRIGGERS", simulated_all, n_train)
    display(
        "CURRENT OR SELECTED",
        existing_mask | simulated_selected,
        n_train,
    )

    print("\nBY TICKER: SELECTED JOBS")

    for j, ticker in enumerate(TICKERS):
        print(
            ticker,
            "jobs=", len(candidate_days[ticker]),
            "train_coverage=",
            f"{simulated_selected[:n_train,j].mean():.2%}",
            "val_coverage=",
            f"{simulated_selected[n_train:,j].mean():.2%}",
        )

    print("\nSIMULATION COMPLETE")


if __name__ == "__main__":
    main()
