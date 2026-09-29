import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_events.csv"
)

DAILY_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_features_business_days.csv"
)


# ============================================================
# Load
# ============================================================

events = pd.read_csv(
    EVENT_PATH
)

daily = pd.read_csv(
    DAILY_PATH
)


events["event_date"] = pd.to_datetime(
    events["event_date"]
)

daily["date"] = pd.to_datetime(
    daily["date"]
)

daily["source_filing_date"] = pd.to_datetime(
    daily["source_filing_date"]
)


print("=" * 70)
print("SEC RAG QA")
print("=" * 70)


# ============================================================
# 1. Basic event counts
# ============================================================

print("\n1. EVENT COUNTS")
print("-" * 70)

print(
    "Total events:",
    len(events)
)

print(
    "\nEvents by ticker:"
)

print(
    events[
        "ticker"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nStatus:"
)

print(
    events[
        "status"
    ]
    .value_counts()
)


# ============================================================
# 2. Direction distribution
# ============================================================

print("\n2. DIRECTION DISTRIBUTION")
print("-" * 70)

direction_table = pd.crosstab(
    events["ticker"],
    events["direction"],
    margins=True,
)

print(
    direction_table
)


# ============================================================
# 3. Numeric distributions
# ============================================================

print("\n3. NUMERIC FEATURE SUMMARY")
print("-" * 70)

cols = [
    "impact_score",
    "risk_score",
    "confidence",
]

print(
    events[
        cols
    ]
    .describe()
    .round(4)
)


print(
    "\nMean by ticker:"
)

print(
    events.groupby(
        "ticker"
    )[cols]
    .mean()
    .round(4)
)


# ============================================================
# 4. Extreme-score checks
# ============================================================

print("\n4. EXTREME SCORE CHECK")
print("-" * 70)

n = len(events)

impact_neg1 = (
    events[
        "impact_score"
    ]
    <= -0.99
).sum()

impact_pos1 = (
    events[
        "impact_score"
    ]
    >= 0.99
).sum()

risk_1 = (
    events[
        "risk_score"
    ]
    >= 0.99
).sum()

confidence_1 = (
    events[
        "confidence"
    ]
    >= 0.99
).sum()


print(
    f"Impact ≈ -1 : "
    f"{impact_neg1}/{n} "
    f"({impact_neg1/n:.1%})"
)

print(
    f"Impact ≈ +1 : "
    f"{impact_pos1}/{n} "
    f"({impact_pos1/n:.1%})"
)

print(
    f"Risk ≈ 1   : "
    f"{risk_1}/{n} "
    f"({risk_1/n:.1%})"
)

print(
    f"Conf ≈ 1   : "
    f"{confidence_1}/{n} "
    f"({confidence_1/n:.1%})"
)


# ============================================================
# 5. Direction / sign consistency
# ============================================================

print("\n5. DIRECTION / IMPACT CONSISTENCY")
print("-" * 70)

bad_negative = events[
    (events["direction"] == "negative")
    &
    (events["impact_score"] >= 0)
]

bad_positive = events[
    (events["direction"] == "positive")
    &
    (events["impact_score"] <= 0)
]

bad_neutral = events[
    (events["direction"] == "neutral")
    &
    (
        events[
            "impact_score"
        ].abs()
        > 0.25
    )
]


bad_consistency = (
    len(bad_negative)
    + len(bad_positive)
    + len(bad_neutral)
)

print(
    "Inconsistent events:",
    bad_consistency
)


# ============================================================
# 6. Duplicate events
# ============================================================

print("\n6. DUPLICATE EVENT CHECK")
print("-" * 70)

duplicates = events.duplicated(
    subset=[
        "ticker",
        "event_date",
    ],
    keep=False,
)

print(
    "Duplicate ticker/date rows:",
    duplicates.sum()
)


# ============================================================
# 7. Forward-fill leakage check
# ============================================================

print("\n7. FORWARD-FILL LEAKAGE CHECK")
print("-" * 70)

valid_daily = daily.dropna(
    subset=[
        "source_filing_date"
    ]
).copy()


future_leakage = valid_daily[
    valid_daily[
        "source_filing_date"
    ]
    >
    valid_daily[
        "date"
    ]
]


print(
    "Rows using future filings:",
    len(future_leakage)
)


# ============================================================
# 8. Days-since-filing sanity check
# ============================================================

negative_age = valid_daily[
    valid_daily[
        "days_since_filing"
    ]
    < 0
]


print(
    "Negative days_since_filing:",
    len(negative_age)
)


# ============================================================
# 9. Missing values
# ============================================================

print("\n8. DAILY FEATURE MISSING VALUES")
print("-" * 70)

print(
    daily[
        [
            "impact_score",
            "risk_score",
            "confidence",
        ]
    ]
    .isna()
    .sum()
)


print(
    "\nNOTE:"
)

print(
    "Missing values before each ticker's first SEC filing "
    "are expected because we intentionally do not "
    "backward-fill future information."
)


# ============================================================
# 10. Date ranges by ticker
# ============================================================

print("\n9. EVENT DATE RANGE BY TICKER")
print("-" * 70)

print(
    events.groupby(
        "ticker"
    )[
        "event_date"
    ]
    .agg(
        [
            "min",
            "max",
            "count",
        ]
    )
)


# ============================================================
# Overall pass/fail
# ============================================================

print("\n" + "=" * 70)
print("QA SUMMARY")
print("=" * 70)


problems = []


if (
    events["status"]
    != "ok"
).any():

    problems.append(
        "Some event generations failed."
    )


if bad_consistency > 0:

    problems.append(
        "Direction / impact inconsistencies found."
    )


if duplicates.sum() > 0:

    problems.append(
        "Duplicate ticker/date events found."
    )


if len(
    future_leakage
) > 0:

    problems.append(
        "Future-information leakage detected."
    )


if len(
    negative_age
) > 0:

    problems.append(
        "Negative days_since_filing detected."
    )


if problems:

    print(
        "QA RESULT: CHECK REQUIRED"
    )

    for problem in problems:

        print(
            "-",
            problem,
        )

else:

    print(
        "QA RESULT: STRUCTURAL CHECKS PASSED"
    )


print(
    "\nImportant:"
)

print(
    "Extreme-score percentages still need "
    "human review before using the features "
    "for the final DRL experiment."
)