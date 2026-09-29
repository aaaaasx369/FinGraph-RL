import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = (
    ROOT
    / "rag"
    / "data"
    / "index"
)

EMBEDDING_PATH = (
    INDEX_DIR
    / "sec_embeddings.npy"
)

METADATA_PATH = (
    INDEX_DIR
    / "sec_metadata.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "rag"
    / "data"
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "sec_rag_events_checkpoint.jsonl"
)

EVENT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "sec_rag_events.csv"
)

INTERVAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "sec_rag_intervals.csv"
)

DAILY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "sec_rag_features_business_days.csv"
)


# ============================================================
# Configuration
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

DEFAULT_LLM_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

PIPELINE_VERSION = (
    "sec_event_signal_v1"
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--start",
    default="2018-01-01",
)

parser.add_argument(
    "--end",
    default="2023-12-31",
)

parser.add_argument(
    "--top-k",
    type=int,
    default=5,
)

parser.add_argument(
    "--llm-model",
    default=DEFAULT_LLM_MODEL,
)

parser.add_argument(
    "--max-events",
    type=int,
    default=None,
    help=(
        "Optional limit for testing. "
        "Example: --max-events 5"
    ),
)

parser.add_argument(
    "--reset",
    action="store_true",
    help=(
        "Delete existing checkpoint and start over."
    ),
)

args = parser.parse_args()


START_DATE = args.start
END_DATE = args.end
TOP_K = args.top_k


# ============================================================
# Optional reset
# ============================================================

if args.reset:

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        "Old checkpoint removed."
    )


# ============================================================
# General event query
#
# IMPORTANT:
# We ask about BOTH positive and negative business effects.
# ============================================================

def build_query(ticker):

    return (
        f"Identify the most material investment-relevant "
        f"information in the newly filed SEC disclosure for "
        f"{ticker}. Consider both positive and negative effects "
        f"on revenue, demand, growth, market access, costs, "
        f"operations, competitive position, financial condition, "
        f"material risks, and material opportunities."
    )


# ============================================================
# Parse JSON from LLM
# ============================================================

def parse_json_object(text):

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# Normalize signal
# ============================================================

def normalize_signal(data):

    required = [
        "direction",
        "impact_score",
        "risk_score",
        "confidence",
        "reason",
    ]

    for key in required:

        if key not in data:
            raise ValueError(
                f"Missing key: {key}"
            )


    direction = str(
        data["direction"]
    ).strip().lower()


    if direction not in {
        "negative",
        "neutral",
        "positive",
    }:

        raise ValueError(
            f"Invalid direction: {direction}"
        )


    impact = float(
        data["impact_score"]
    )

    risk = float(
        data["risk_score"]
    )

    confidence = float(
        data["confidence"]
    )

    reason = str(
        data["reason"]
    ).strip()


    impact = max(
        -1.0,
        min(
            1.0,
            impact,
        ),
    )

    risk = max(
        0.0,
        min(
            1.0,
            risk,
        ),
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


    return {
        "direction": direction,
        "impact_score": impact,
        "risk_score": risk,
        "confidence": confidence,
        "reason": reason,
    }


# ============================================================
# Direction / score consistency
# ============================================================

def is_consistent(signal):

    direction = signal[
        "direction"
    ]

    impact = signal[
        "impact_score"
    ]


    if (
        direction == "negative"
        and impact >= 0
    ):
        return False


    if (
        direction == "positive"
        and impact <= 0
    ):
        return False


    if (
        direction == "neutral"
        and abs(impact) > 0.25
    ):
        return False


    return True


# ============================================================
# Load vectors + metadata
# ============================================================

print("=" * 70)
print("LOAD SEC INDEX")
print("=" * 70)


embeddings = np.load(
    EMBEDDING_PATH
).astype(
    np.float32
)


metadata = []

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        metadata.append(
            json.loads(line)
        )


if len(metadata) != len(embeddings):

    raise RuntimeError(
        "Embedding / metadata length mismatch.\n"
        f"Embeddings: {len(embeddings)}\n"
        f"Metadata: {len(metadata)}"
    )


print(
    "SEC chunks:",
    len(metadata),
)

print(
    "Embedding shape:",
    embeddings.shape,
)


# ============================================================
# Build filing-event map
#
# One event =
# one ticker + one filing date.
#
# If multiple filings occur on the same date,
# their chunks are pooled together.
# ============================================================

event_indices = {}


for i, doc in enumerate(
    metadata
):

    ticker = doc[
        "ticker"
    ].upper()

    date = doc[
        "published_at"
    ]


    if ticker not in TICKERS:
        continue


    if date < START_DATE:
        continue


    if date > END_DATE:
        continue


    key = (
        ticker,
        date,
    )


    if key not in event_indices:

        event_indices[
            key
        ] = []


    event_indices[
        key
    ].append(
        i
    )


events = sorted(
    event_indices.keys(),
    key=lambda x: (
        x[1],
        x[0],
    ),
)


print(
    "Unique filing events:",
    len(events),
)


for ticker in TICKERS:

    count = sum(
        1
        for x in events
        if x[0] == ticker
    )

    print(
        ticker,
        "events:",
        count,
    )


# ============================================================
# Load previous checkpoint
# ============================================================

latest_checkpoint = {}


if CHECKPOINT_PATH.exists():

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except Exception:

                continue


            if (
                row.get(
                    "pipeline_version"
                )
                != PIPELINE_VERSION
            ):
                continue


            key = (
                row.get(
                    "ticker"
                ),
                row.get(
                    "event_date"
                ),
            )


            latest_checkpoint[
                key
            ] = row


completed_keys = {
    key
    for key, row
    in latest_checkpoint.items()
    if row.get("status") == "ok"
}


print(
    "\nPreviously completed:",
    len(completed_keys),
)


# ============================================================
# Pre-compute 5 query embeddings
#
# Load embedding model only once.
# ============================================================

print("\n" + "=" * 70)
print("BUILD QUERY EMBEDDINGS")
print("=" * 70)


embedding_device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=embedding_device,
)


query_vectors = {}


for ticker in TICKERS:

    query = build_query(
        ticker
    )

    query_text = (
        QUERY_PREFIX
        + query
    )

    vector = embedding_model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(
        np.float32
    )

    query_vectors[
        ticker
    ] = vector


# Free BGE GPU memory before loading Qwen
del embedding_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# Load local LLM ONCE
# ============================================================

print("\n" + "=" * 70)
print("LOAD LOCAL LLM")
print("=" * 70)

print(
    "Model:",
    args.llm_model,
)


tokenizer = AutoTokenizer.from_pretrained(
    args.llm_model
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


if torch.cuda.is_available():

    if torch.cuda.is_bf16_supported():

        model_dtype = (
            torch.bfloat16
        )

    else:

        model_dtype = (
            torch.float16
        )


    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=model_dtype,
        device_map="auto",
    )

else:

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.float32,
    )

    llm = llm.to(
        "cpu"
    )


llm.eval()


# ============================================================
# LLM generation helper
# ============================================================

def generate_text(
    system_prompt,
    user_prompt,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    model_inputs = tokenizer(
        chat_text,
        return_tensors="pt",
    )


    model_device = next(
        llm.parameters()
    ).device


    model_inputs = {
        key: value.to(
            model_device
        )
        for key, value
        in model_inputs.items()
    }


    prompt_length = (
        model_inputs[
            "input_ids"
        ].shape[1]
    )


    with torch.no_grad():

        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    new_tokens = generated_ids[
        0,
        prompt_length:
    ]


    output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


    return output


# ============================================================
# System calibration prompt
# ============================================================

SYSTEM_PROMPT = """
You are a financial SEC-filing evidence scoring system.

Use ONLY the supplied SEC evidence.

The evidence comes from SEC filings newly published on the
stated event date.

Do not use outside knowledge.
Do not use future information.
Do not predict future events.
Do not provide investment advice.

Your task is to quantify the investment implication of the
NEW filing evidence for the company.

impact_score measures whether the disclosed information is
favorable or unfavorable for the company.

It does NOT measure importance.
It does NOT measure confidence.

Negative examples:
- revenue decline
- demand decline
- loss of market access
- higher costs
- operational disruption
- supply-chain disruption
- weaker competitive position

Positive examples:
- stronger demand
- revenue growth
- improved margins
- expanded market access
- lower costs
- stronger competitive position

If evidence is mixed or does not clearly imply a positive or
negative business impact, choose neutral and keep impact_score
near zero.

Return exactly one JSON object:

{
  "direction": "negative" | "neutral" | "positive",
  "impact_score": number,
  "risk_score": number,
  "confidence": number,
  "reason": "one short evidence-based sentence"
}

Ranges:

impact_score:
-1 = extremely unfavorable
 0 = neutral / mixed
+1 = extremely favorable

risk_score:
0 = little material business risk
1 = extremely high material business risk

confidence:
0 = weak / insufficient evidence
1 = direct and strong evidence

Consistency rules:

negative -> impact_score < 0
positive -> impact_score > 0
neutral -> impact_score between -0.25 and +0.25

Return JSON only.
No markdown.
""".strip()


# ============================================================
# Process filing events
# ============================================================

attempted = 0


for event_number, (
    ticker,
    event_date,
) in enumerate(
    events,
    start=1,
):

    key = (
        ticker,
        event_date,
    )


    if key in completed_keys:

        continue


    if (
        args.max_events is not None
        and attempted >= args.max_events
    ):

        break


    attempted += 1


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EVENT {event_number}/{len(events)}"
    )

    print(
        ticker,
        event_date,
    )

    print(
        "=" * 70
    )


    # ========================================================
    # IMPORTANT:
    #
    # Only use chunks from NEW filings published exactly on
    # this event date.
    #
    # This makes the signal event-driven.
    # ========================================================

    indices = event_indices[
        key
    ]


    event_embeddings = embeddings[
        indices
    ]


    query_vector = query_vectors[
        ticker
    ]


    scores = (
        event_embeddings
        @ query_vector
    )


    ranking = np.argsort(
        scores
    )[::-1]


    top_n = min(
        TOP_K,
        len(ranking),
    )


    retrieved = []


    for rank in range(
        top_n
    ):

        local_idx = ranking[
            rank
        ]

        global_idx = indices[
            local_idx
        ]

        doc = metadata[
            global_idx
        ]

        retrieved.append(
            {
                "rank":
                    rank + 1,

                "score":
                    float(
                        scores[
                            local_idx
                        ]
                    ),

                "chunk_id":
                    doc[
                        "chunk_id"
                    ],

                "doc_id":
                    doc[
                        "doc_id"
                    ],

                "form":
                    doc[
                        "form"
                    ],

                "published_at":
                    doc[
                        "published_at"
                    ],

                "text":
                    doc[
                        "text"
                    ],

                "url":
                    doc[
                        "url"
                    ],
            }
        )


    # ========================================================
    # Build evidence
    # ========================================================

    evidence_blocks = []


    for item in retrieved:

        evidence_blocks.append(
            (
                f"[Evidence {item['rank']}]\n"
                f"Published: {item['published_at']}\n"
                f"Form: {item['form']}\n"
                f"Similarity: {item['score']:.4f}\n"
                f"Text:\n"
                f"{item['text']}\n"
            )
        )


    evidence_context = "\n\n".join(
        evidence_blocks
    )


    query = build_query(
        ticker
    )


    user_prompt = f"""
Ticker:
{ticker}

SEC filing event date:
{event_date}

Research question:
{query}

The following evidence comes ONLY from SEC filing(s)
published on the event date:

{evidence_context}

Determine whether the new filing evidence is favorable,
unfavorable, or neutral for the company.

Return JSON only.
""".strip()


    # ========================================================
    # Generate
    # ========================================================

    raw_output = None
    repair_output = None
    repair_used = False


    try:

        raw_output = generate_text(
            SYSTEM_PROMPT,
            user_prompt,
        )


        print(
            "Raw output:",
            raw_output,
        )


        data = parse_json_object(
            raw_output
        )


        signal = normalize_signal(
            data
        )


        # ====================================================
        # Repair sign inconsistency if necessary
        # ====================================================

        if not is_consistent(
            signal
        ):

            repair_used = True


            repair_prompt = f"""
The previous answer was internally inconsistent.

Previous output:

{json.dumps(signal, ensure_ascii=False)}

Correct it using the original evidence.

Rules:

negative -> impact_score < 0
positive -> impact_score > 0
neutral -> impact_score between -0.25 and +0.25

Original evidence:

{evidence_context}

Return corrected JSON only.
""".strip()


            repair_output = generate_text(
                SYSTEM_PROMPT,
                repair_prompt,
            )


            signal = normalize_signal(
                parse_json_object(
                    repair_output
                )
            )


        if not is_consistent(
            signal
        ):

            raise RuntimeError(
                "Direction / impact_score "
                "still inconsistent."
            )


        row = {
            "pipeline_version":
                PIPELINE_VERSION,

            "status":
                "ok",

            "ticker":
                ticker,

            "event_date":
                event_date,

            "direction":
                signal[
                    "direction"
                ],

            "impact_score":
                signal[
                    "impact_score"
                ],

            "risk_score":
                signal[
                    "risk_score"
                ],

            "confidence":
                signal[
                    "confidence"
                ],

            "reason":
                signal[
                    "reason"
                ],

            "repair_used":
                repair_used,

            "candidate_chunks":
                len(indices),

            "top_k":
                top_n,

            "retrieved_chunk_ids":
                [
                    x["chunk_id"]
                    for x in retrieved
                ],

            "retrieval_scores":
                [
                    x["score"]
                    for x in retrieved
                ],

            "retrieved_forms":
                [
                    x["form"]
                    for x in retrieved
                ],

            "raw_model_output":
                raw_output,

            "repair_model_output":
                repair_output,
        }


        print(
            "Direction :",
            signal[
                "direction"
            ]
        )

        print(
            "Impact    :",
            signal[
                "impact_score"
            ]
        )

        print(
            "Risk      :",
            signal[
                "risk_score"
            ]
        )

        print(
            "Confidence:",
            signal[
                "confidence"
            ]
        )

        print(
            "Reason    :",
            signal[
                "reason"
            ]
        )


    except Exception as e:

        print(
            "ERROR:",
            repr(e),
        )


        row = {
            "pipeline_version":
                PIPELINE_VERSION,

            "status":
                "error",

            "ticker":
                ticker,

            "event_date":
                event_date,

            "error":
                repr(e),

            "raw_model_output":
                raw_output,

            "repair_model_output":
                repair_output,
        }


    # ========================================================
    # Append immediately to checkpoint
    #
    # If SSH disconnects, completed work remains.
    # ========================================================

    with open(
        CHECKPOINT_PATH,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


# ============================================================
# Reload checkpoint and keep latest result per event
# ============================================================

latest = {}


if CHECKPOINT_PATH.exists():

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue


            try:

                row = json.loads(
                    line
                )

            except Exception:
                continue


            if (
                row.get(
                    "pipeline_version"
                )
                != PIPELINE_VERSION
            ):
                continue


            ticker = row.get(
                "ticker"
            )

            event_date = row.get(
                "event_date"
            )


            if ticker not in TICKERS:
                continue


            if (
                event_date < START_DATE
                or event_date > END_DATE
            ):
                continue


            latest[
                (
                    ticker,
                    event_date,
                )
            ] = row


all_rows = list(
    latest.values()
)


# ============================================================
# Event-level table
# ============================================================

event_df = pd.DataFrame(
    all_rows
)


if not event_df.empty:

    event_df = event_df.sort_values(
        [
            "ticker",
            "event_date",
        ]
    )


event_df.to_csv(
    EVENT_OUTPUT_PATH,
    index=False,
)


print(
    "\nSaved event table:"
)

print(
    EVENT_OUTPUT_PATH
)


# ============================================================
# Successful signals only
# ============================================================

if event_df.empty:

    print(
        "No event signals available yet."
    )

    raise SystemExit(0)


success = event_df[
    event_df["status"] == "ok"
].copy()


if success.empty:

    print(
        "No successful event signals available yet."
    )

    raise SystemExit(0)


success[
    "event_date"
] = pd.to_datetime(
    success[
        "event_date"
    ]
)


# ============================================================
# Build effective intervals
#
# Signal becomes active on filing date and remains active
# until the next SUCCESSFUL filing signal.
# ============================================================

interval_rows = []


for ticker in TICKERS:

    temp = success[
        success["ticker"] == ticker
    ].copy()


    temp = temp.sort_values(
        "event_date"
    )


    if temp.empty:
        continue


    dates = temp[
        "event_date"
    ].tolist()


    for i, (
        _,
        row,
    ) in enumerate(
        temp.iterrows()
    ):

        if i + 1 < len(dates):

            next_date = dates[
                i + 1
            ]

            effective_until = (
                next_date
            )

        else:

            effective_until = (
                pd.Timestamp(
                    END_DATE
                )
                + pd.Timedelta(
                    days=1
                )
            )


        interval_rows.append(
            {
                "ticker":
                    ticker,

                "effective_from":
                    row[
                        "event_date"
                    ].date(),

                "effective_until_exclusive":
                    effective_until.date(),

                "direction":
                    row[
                        "direction"
                    ],

                "impact_score":
                    row[
                        "impact_score"
                    ],

                "risk_score":
                    row[
                        "risk_score"
                    ],

                "confidence":
                    row[
                        "confidence"
                    ],

                "reason":
                    row[
                        "reason"
                    ],
            }
        )


interval_df = pd.DataFrame(
    interval_rows
)


interval_df.to_csv(
    INTERVAL_OUTPUT_PATH,
    index=False,
)


print(
    "\nSaved interval table:"
)

print(
    INTERVAL_OUTPUT_PATH
)


# ============================================================
# Business-day forward-fill table
#
# NOTE:
# This uses weekdays.
# Later we will merge onto the ACTUAL FinRL market trading
# dates, so holidays will naturally disappear.
#
# IMPORTANT:
# We do NOT backward-fill before the first filing.
# That would leak future information.
# ============================================================

daily_outputs = []


for ticker in TICKERS:

    ticker_events = success[
        success["ticker"] == ticker
    ][
        [
            "event_date",
            "direction",
            "impact_score",
            "risk_score",
            "confidence",
            "reason",
        ]
    ].copy()


    ticker_events = ticker_events.sort_values(
        "event_date"
    )


    if ticker_events.empty:
        continue


    daily = pd.DataFrame(
        {
            "date":
                pd.date_range(
                    START_DATE,
                    END_DATE,
                    freq="B",
                )
        }
    )


    ticker_events = ticker_events.rename(
        columns={
            "event_date":
                "source_filing_date"
        }
    )


    merged = pd.merge_asof(
        daily.sort_values(
            "date"
        ),
        ticker_events.sort_values(
            "source_filing_date"
        ),
        left_on="date",
        right_on="source_filing_date",
        direction="backward",
        allow_exact_matches=True,
    )


    merged.insert(
        1,
        "ticker",
        ticker,
    )


    merged[
        "days_since_filing"
    ] = (
        merged["date"]
        - merged[
            "source_filing_date"
        ]
    ).dt.days


    daily_outputs.append(
        merged
    )


if daily_outputs:

    daily_df = pd.concat(
        daily_outputs,
        ignore_index=True,
    )


    daily_df = daily_df.sort_values(
        [
            "date",
            "ticker",
        ]
    )


    daily_df.to_csv(
        DAILY_OUTPUT_PATH,
        index=False,
    )


    print(
        "\nSaved forward-filled table:"
    )

    print(
        DAILY_OUTPUT_PATH
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("BATCH SUMMARY")
print("=" * 70)

print(
    "Events discovered :",
    len(events),
)

print(
    "Checkpoint rows   :",
    len(event_df),
)

print(
    "Successful events :",
    len(success),
)

print(
    "Failed events     :",
    len(
        event_df[
            event_df["status"]
            == "error"
        ]
    ),
)

print(
    "\nDONE"
)