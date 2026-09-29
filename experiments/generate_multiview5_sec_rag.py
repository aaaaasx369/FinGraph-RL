import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--views",
    type=int,
    default=5,
)

parser.add_argument(
    "--top-k",
    type=int,
    default=5,
)

args = parser.parse_args()

N_VIEWS = args.views
TOP_K = args.top_k


if N_VIEWS != 5:
    raise ValueError(
        "This experiment currently expects exactly 5 views."
    )


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# CONFIG
# ============================================================

MASTER_SEED = 20260914

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# PATHS
# ============================================================

EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_events.csv"
)

EMBED_PATH = (
    ROOT
    / "rag"
    / "data"
    / "index"
    / "sec_embeddings.npy"
)

META_PATH = (
    ROOT
    / "rag"
    / "data"
    / "index"
    / "sec_metadata.jsonl"
)

RESULT_DIR = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "multiview5"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    RESULT_DIR
    / "multiview5_checkpoint.jsonl"
)

VIEWS_PATH = (
    RESULT_DIR
    / "sec_rag_multiview5_views.csv"
)

EVENTS_PATH = (
    RESULT_DIR
    / "sec_rag_multiview5_events.csv"
)

RETRIEVAL_PATH = (
    RESULT_DIR
    / "sec_rag_multiview5_retrieval.csv"
)


for path in [
    EVENT_PATH,
    EMBED_PATH,
    META_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded SEC filing analyst.

Use ONLY the supplied SEC filing excerpts.
Do not use external knowledge.
Do not use future information or hindsight.

Return exactly one JSON object:

{
  "direction": "negative" | "neutral" | "positive",
  "impact_score": number,
  "risk_score": number,
  "confidence": number,
  "reason": string
}

impact_score:
- Range [-1, 1].
- -1 = strongly unfavorable for the company.
- +1 = strongly favorable for the company.
- 0 = approximately balanced or immaterial.

risk_score:
- Range [0, 1].
- 0 = little material risk.
- 1 = very high material risk.

confidence:
- Range [0, 1].
- Confidence that the supplied evidence supports the assessment.

Consistency:
- negative => impact_score < 0
- positive => impact_score > 0
- neutral => abs(impact_score) <= 0.25

Consider both positive and negative implications.
Do not exaggerate.
Return JSON only.
""".strip()


TASK_VARIANTS = [
    (
        "Assess the material investment implications of this newly "
        "filed SEC disclosure using only the evidence below."
    ),
    (
        "Using only these SEC excerpts, evaluate the material "
        "investment-relevant information in the new filing."
    ),
    (
        "Review the following excerpts from the newly filed SEC "
        "document and provide the evidence-grounded investment signal."
    ),
    (
        "Determine the overall investor-relevant implication of the "
        "new disclosure based strictly on the supplied filing excerpts."
    ),
    (
        "Analyze the supplied new SEC disclosure excerpts for their "
        "overall material effect on the company from an investor perspective."
    ),
]


def make_query(ticker):

    return (
        "Identify the most material investment-relevant information "
        f"in the newly filed SEC disclosure for {ticker}. "
        "Consider both positive and negative effects on revenue, demand, "
        "growth, market access, costs, operations, competitive position, "
        "financial condition, material risks, and material opportunities."
    )


# ============================================================
# EVENTS
# ============================================================

events = pd.read_csv(
    EVENT_PATH
)

events = events[
    events["status"] == "ok"
].copy()

events["ticker"] = (
    events["ticker"]
    .astype(str)
    .str.upper()
)

events["event_date"] = pd.to_datetime(
    events["event_date"]
).dt.normalize()


events = (
    events
    .sort_values(
        [
            "ticker",
            "event_date",
        ]
    )
    .drop_duplicates(
        [
            "ticker",
            "event_date",
        ]
    )
    .reset_index(
        drop=True
    )
)


print("=" * 75)
print("FULL MULTI-VIEW SEC RAG GENERATION")
print("=" * 75)

print(
    "Unique SEC events:",
    len(events),
)

print(
    "Views per event:",
    N_VIEWS,
)

print(
    "Target generations:",
    len(events) * N_VIEWS,
)


# ============================================================
# INDEX
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBED_PATH
).astype(np.float32)


metadata = []

with open(
    META_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if line:
            metadata.append(
                json.loads(line)
            )


if len(metadata) != len(embeddings):
    raise RuntimeError(
        "Embedding / metadata length mismatch."
    )


def first(meta, names):

    for name in names:

        if name in meta:
            value = meta[name]

            if value is not None:
                return value

    return None


def meta_ticker(meta):

    value = first(
        meta,
        [
            "ticker",
            "tic",
            "symbol",
        ],
    )

    return (
        str(value).upper()
        if value is not None
        else ""
    )


def meta_date(meta):

    value = first(
        meta,
        [
            "published_at",
            "filing_date",
            "event_date",
            "date",
        ],
    )

    if value is None:
        return ""

    try:
        return (
            pd.Timestamp(value)
            .normalize()
            .strftime("%Y-%m-%d")
        )
    except Exception:
        return ""


def meta_text(meta):

    value = first(
        meta,
        [
            "text",
            "chunk_text",
            "content",
            "chunk",
        ],
    )

    return (
        str(value)
        if value is not None
        else ""
    )


META_TICKERS = np.array(
    [
        meta_ticker(m)
        for m in metadata
    ],
    dtype=object,
)

META_DATES = np.array(
    [
        meta_date(m)
        for m in metadata
    ],
    dtype=object,
)


# ============================================================
# EMBEDDER
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Embedding device:",
    device,
)

embedder = SentenceTransformer(
    EMBED_MODEL,
    device=device,
)


# ============================================================
# CLEAN RETRIEVAL
# ============================================================

retrieval_cache = {}

retrieval_audit = []


def retrieve_event(
    ticker,
    event_date,
):

    date_string = (
        pd.Timestamp(event_date)
        .normalize()
        .strftime("%Y-%m-%d")
    )

    key = (
        ticker,
        date_string,
    )


    if key in retrieval_cache:
        return retrieval_cache[key]


    indices = np.where(
        (
            META_TICKERS
            ==
            ticker
        )
        &
        (
            META_DATES
            ==
            date_string
        )
    )[0]


    if len(indices) == 0:

        raise RuntimeError(
            f"No chunks found for "
            f"{ticker} {date_string}"
        )


    query_embedding = embedder.encode(
        [
            QUERY_PREFIX
            +
            make_query(ticker)
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(
        np.float32
    )


    scores = (
        embeddings[
            indices
        ]
        @
        query_embedding
    )


    order = np.argsort(
        scores
    )[::-1]


    selected = indices[
        order[
            :TOP_K
        ]
    ]

    selected_scores = scores[
        order[
            :TOP_K
        ]
    ]


    chunks = []


    for rank, (
        index,
        score,
    ) in enumerate(
        zip(
            selected,
            selected_scores,
        ),
        start=1,
    ):

        text = meta_text(
            metadata[
                int(index)
            ]
        )


        if not text.strip():

            raise RuntimeError(
                f"Empty chunk {index}"
            )


        chunks.append(
            {
                "index":
                    int(index),

                "rank":
                    rank,

                "score":
                    float(score),

                "text":
                    text,
            }
        )


        retrieval_audit.append(
            {
                "ticker":
                    ticker,

                "event_date":
                    date_string,

                "rank":
                    rank,

                "metadata_index":
                    int(index),

                "similarity":
                    float(score),
            }
        )


    retrieval_cache[key] = chunks

    return chunks


# ============================================================
# LLM
# ============================================================

print("\nLoading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()


# ============================================================
# PARSER
# ============================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")


    if start < 0 or end < start:

        raise ValueError(
            "JSON not found."
        )


    return json.loads(
        text[
            start:
            end + 1
        ]
    )


def validate(obj):

    required = [
        "direction",
        "impact_score",
        "risk_score",
        "confidence",
        "reason",
    ]


    for key in required:

        if key not in obj:
            raise ValueError(
                f"Missing {key}"
            )


    direction = str(
        obj["direction"]
    ).strip().lower()


    if direction not in {
        "negative",
        "neutral",
        "positive",
    }:

        raise ValueError(
            "Bad direction."
        )


    impact = float(
        obj["impact_score"]
    )

    risk = float(
        obj["risk_score"]
    )

    confidence = float(
        obj["confidence"]
    )


    if not -1 <= impact <= 1:
        raise ValueError(
            "Bad impact."
        )

    if not 0 <= risk <= 1:
        raise ValueError(
            "Bad risk."
        )

    if not 0 <= confidence <= 1:
        raise ValueError(
            "Bad confidence."
        )


    if (
        direction == "negative"
        and
        impact >= 0
    ):
        raise ValueError(
            "Direction mismatch."
        )


    if (
        direction == "positive"
        and
        impact <= 0
    ):
        raise ValueError(
            "Direction mismatch."
        )


    if (
        direction == "neutral"
        and
        abs(impact) > 0.25
    ):
        raise ValueError(
            "Neutral mismatch."
        )


    return {
        "direction":
            direction,

        "impact_score":
            impact,

        "risk_score":
            risk,

        "confidence":
            confidence,

        "reason":
            str(
                obj["reason"]
            ).strip(),
    }


def call_qwen(
    user_prompt,
):

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = tokenizer(
        [
            prompt
        ],
        return_tensors="pt",
    )


    inputs = {
        key:
            value.to(
                model.device
            )
        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


    n_input = inputs[
        "input_ids"
    ].shape[1]


    return tokenizer.decode(
        output[
            0,
            n_input:
        ],
        skip_special_tokens=True,
    )


# ============================================================
# ONE VIEW
# ============================================================

def generate_view(
    ticker,
    event_date,
    chunks,
    event_index,
    view_id,
):

    rng = np.random.default_rng(
        MASTER_SEED
        +
        event_index
        * 1009
        +
        view_id
        * 100_003
    )


    template_id = (
        view_id
        %
        len(
            TASK_VARIANTS
        )
    )


    evidence_order = (
        rng.permutation(
            len(chunks)
        )
        .tolist()
    )


    evidence = []


    for display_rank, pos in enumerate(
        evidence_order,
        start=1,
    ):

        evidence.append(
            (
                f"[Evidence {display_rank}]\n"
                f"{chunks[pos]['text']}"
            )
        )


    user_prompt = (
        TASK_VARIANTS[
            template_id
        ]
        +
        "\n\n"
        +
        "Evaluate material effects on revenue, demand, growth, "
        "market access, costs, operations, competitive position, "
        "financial condition, risks, and opportunities.\n\n"
        +
        f"Ticker: {ticker}\n"
        +
        f"Filing date: "
        f"{pd.Timestamp(event_date).date()}\n\n"
        +
        "\n\n".join(
            evidence
        )
        +
        "\n\nReturn JSON only."
    )


    response = call_qwen(
        user_prompt
    )


    try:

        signal = validate(
            extract_json(
                response
            )
        )

        repaired = False


    except Exception:

        repair_prompt = (
            user_prompt
            +
            "\n\nThe previous output was invalid:\n"
            +
            response
            +
            "\n\nReturn ONLY a corrected JSON object "
            "using the same evidence."
        )


        repaired_response = call_qwen(
            repair_prompt
        )


        signal = validate(
            extract_json(
                repaired_response
            )
        )

        repaired = True


    return (
        signal,
        template_id,
        evidence_order,
        repaired,
    )


# ============================================================
# CHECKPOINT
# ============================================================

completed = set()


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


            row = json.loads(
                line
            )


            completed.add(
                (
                    row["ticker"],
                    row["event_date"],
                    int(
                        row["view_id"]
                    ),
                )
            )


print(
    "\nExisting generations:",
    len(completed),
)


# ============================================================
# GENERATE
# ============================================================

for event_index, event in events.iterrows():

    ticker = event[
        "ticker"
    ]

    event_date = event[
        "event_date"
    ]

    date_string = (
        event_date.strftime(
            "%Y-%m-%d"
        )
    )


    chunks = retrieve_event(
        ticker,
        event_date,
    )


    print(
        f"\n[{event_index + 1}/"
        f"{len(events)}] "
        f"{ticker} {date_string}"
    )


    for view_id in range(
        N_VIEWS
    ):

        key = (
            ticker,
            date_string,
            view_id,
        )


        if key in completed:

            print(
                f"  View {view_id + 1}: SKIP"
            )

            continue


        (
            signal,
            template_id,
            evidence_order,
            repaired,
        ) = generate_view(
            ticker=ticker,
            event_date=event_date,
            chunks=chunks,
            event_index=event_index,
            view_id=view_id,
        )


        record = {
            "ticker":
                ticker,

            "event_date":
                date_string,

            "view_id":
                view_id,

            "template_id":
                template_id,

            "evidence_order":
                evidence_order,

            "retrieval_indices":
                [
                    x[
                        "index"
                    ]
                    for x in chunks
                ],

            "repaired":
                repaired,

            **signal,
        }


        with open(
            CHECKPOINT_PATH,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


        completed.add(
            key
        )


        print(
            f"  View {view_id + 1}: "
            f"{signal['direction']} "
            f"impact={signal['impact_score']:+.2f}"
        )


# ============================================================
# LOAD FINAL CHECKPOINT
# ============================================================

records = []


with open(
    CHECKPOINT_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if line:
            records.append(
                json.loads(
                    line
                )
            )


views = pd.DataFrame(
    records
)


views["event_date"] = pd.to_datetime(
    views["event_date"]
)


views = (
    views
    .sort_values(
        [
            "ticker",
            "event_date",
            "view_id",
        ]
    )
    .drop_duplicates(
        [
            "ticker",
            "event_date",
            "view_id",
        ],
        keep="last",
    )
)


expected = (
    len(events)
    *
    N_VIEWS
)


if len(views) != expected:

    raise RuntimeError(
        f"Expected {expected} views, "
        f"found {len(views)}."
    )


views.to_csv(
    VIEWS_PATH,
    index=False,
)


pd.DataFrame(
    retrieval_audit
).drop_duplicates().to_csv(
    RETRIEVAL_PATH,
    index=False,
)


# ============================================================
# RELIABILITY GATE
#
# 1. Direction consensus
#
# With 5 views and 3 classes, the minimum possible largest
# vote share is 2/5 = 0.4.
#
# Normalize:
#
# consensus = (agreement - 0.4) / 0.6
#
# So:
# 2-2-1 -> 0
# 3-x-x -> 0.333...
# 4-1   -> 0.667
# 5-0   -> 1
#
# 2. Numerical dispersion
#
# Robust MAD for impact/risk/confidence.
#
# impact range length = 2
# risk/conf range length = 1
#
# 3. gate = consensus * (1 - dispersion)
# ============================================================

def aggregate_event(
    group,
):

    if len(group) != 5:

        raise RuntimeError(
            "Expected exactly 5 views."
        )


    impact = group[
        "impact_score"
    ].astype(float).to_numpy()

    risk = group[
        "risk_score"
    ].astype(float).to_numpy()

    conf = group[
        "confidence"
    ].astype(float).to_numpy()


    median_impact = float(
        np.median(
            impact
        )
    )

    median_risk = float(
        np.median(
            risk
        )
    )

    median_conf = float(
        np.median(
            conf
        )
    )


    counts = Counter(
        group[
            "direction"
        ]
        .astype(str)
        .tolist()
    )


    majority_direction, max_count = (
        counts.most_common(
            1
        )[0]
    )


    direction_agreement = (
        max_count
        /
        5.0
    )


    consensus_strength = (
        direction_agreement
        -
        0.4
    ) / 0.6


    consensus_strength = float(
        np.clip(
            consensus_strength,
            0.0,
            1.0,
        )
    )


    mad_impact = float(
        np.median(
            np.abs(
                impact
                -
                median_impact
            )
        )
    )


    mad_risk = float(
        np.median(
            np.abs(
                risk
                -
                median_risk
            )
        )
    )


    mad_conf = float(
        np.median(
            np.abs(
                conf
                -
                median_conf
            )
        )
    )


    normalized_dispersion = float(
        np.mean(
            [
                np.clip(
                    mad_impact / 2.0,
                    0,
                    1,
                ),

                np.clip(
                    mad_risk,
                    0,
                    1,
                ),

                np.clip(
                    mad_conf,
                    0,
                    1,
                ),
            ]
        )
    )


    reliability_gate = (
        consensus_strength
        *
        (
            1.0
            -
            normalized_dispersion
        )
    )


    reliability_gate = float(
        np.clip(
            reliability_gate,
            0.0,
            1.0,
        )
    )


    return pd.Series(
        {
            "median_impact_score":
                median_impact,

            "median_risk_score":
                median_risk,

            "median_confidence":
                median_conf,

            "majority_direction":
                majority_direction,

            "direction_agreement":
                direction_agreement,

            "consensus_strength":
                consensus_strength,

            "mad_impact":
                mad_impact,

            "mad_risk":
                mad_risk,

            "mad_confidence":
                mad_conf,

            "normalized_dispersion":
                normalized_dispersion,

            "reliability_gate":
                reliability_gate,
        }
    )


aggregated = (
    views
    .groupby(
        [
            "ticker",
            "event_date",
        ],
        as_index=False,
    )
    .apply(
        aggregate_event,
        include_groups=False,
    )
    .reset_index()
)


# Compatibility across pandas versions
if "level_0" in aggregated.columns:
    aggregated = aggregated.drop(
        columns=[
            "level_0",
        ]
    )

if "level_1" in aggregated.columns:
    aggregated = aggregated.drop(
        columns=[
            "level_1",
        ]
    )


aggregated.to_csv(
    EVENTS_PATH,
    index=False,
)


# ============================================================
# QA
# ============================================================

print("\n" + "=" * 75)
print("MULTI-VIEW DATASET COMPLETE")
print("=" * 75)

print(
    "Views:",
    len(views),
)

print(
    "Events:",
    len(aggregated),
)

print(
    "Mean reliability:",
    f"{aggregated['reliability_gate'].mean():.4f}",
)

print(
    "Median reliability:",
    f"{aggregated['reliability_gate'].median():.4f}",
)

print(
    "Gate <= 0.25:",
    f"{(aggregated['reliability_gate'] <= 0.25).mean():.2%}",
)

print(
    "Gate >= 0.75:",
    f"{(aggregated['reliability_gate'] >= 0.75).mean():.2%}",
)

print(
    "Mean direction agreement:",
    f"{aggregated['direction_agreement'].mean():.4f}",
)

print(
    "\nSaved:"
)

print(
    EVENTS_PATH
)

print("\nDONE")