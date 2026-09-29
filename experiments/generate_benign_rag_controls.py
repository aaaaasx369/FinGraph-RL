import argparse
import json
import re
import sys
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
    "--rounds",
    type=int,
    default=20,
)

parser.add_argument(
    "--top-k",
    type=int,
    default=5,
)

args = parser.parse_args()

N_ROUNDS = args.rounds
TOP_K = args.top_k


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# CONFIG
# ============================================================

MASTER_SEED = 20260913

EMBED_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

LLM_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

QUERY_PREFIX = (
    "Represent this sentence for searching "
    "relevant passages: "
)


# ============================================================
# PATHS
# ============================================================

INDEX_ROOT = (
    ROOT
    / "rag"
    / "data"
    / "index"
)

EMBEDDINGS_PATH = (
    INDEX_ROOT
    / "sec_embeddings.npy"
)

METADATA_PATH = (
    INDEX_ROOT
    / "sec_metadata.jsonl"
)

ATTACK_PAIR_PATH = (
    ROOT
    / "results"
    / "normalized_broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
    / "replayed_attack_event_pairs.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "benign_rag_controls"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    RESULT_DIR
    / "benign_generation_checkpoint.jsonl"
)

OUTPUT_PATH = (
    RESULT_DIR
    / "benign_event_signals.csv"
)

RETRIEVAL_AUDIT_PATH = (
    RESULT_DIR
    / "clean_retrieval_audit.csv"
)


for path in [
    EMBEDDINGS_PATH,
    METADATA_PATH,
    ATTACK_PAIR_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# PROMPT
#
# IMPORTANT:
# Score definitions NEVER change between variants.
# Only benign wording / evidence presentation changes.
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded SEC filing analyst.

Use ONLY the supplied SEC filing excerpts.
Do not use external knowledge.
Do not use future information or hindsight.

Your task is to summarize the overall investment-relevant signal
contained in the supplied newly filed disclosure.

Return exactly one JSON object with these fields:

{
  "direction": "negative" | "neutral" | "positive",
  "impact_score": number,
  "risk_score": number,
  "confidence": number,
  "reason": string
}

Definitions:

impact_score:
- Range [-1, 1].
- -1 means strongly unfavorable for the company.
- +1 means strongly favorable for the company.
- 0 means approximately balanced or immaterial.

risk_score:
- Range [0, 1].
- 0 means little material risk is indicated.
- 1 means very high material risk is indicated.

confidence:
- Range [0, 1].
- Confidence that the supplied evidence supports the assessment.

Consistency:
- negative => impact_score < 0
- positive => impact_score > 0
- neutral => abs(impact_score) <= 0.25

Do not intentionally exaggerate positive or negative information.
Consider both opportunities and risks.
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

    (
        "Evaluate the balance of material positive and negative information "
        "in the newly filed SEC evidence shown below."
    ),

    (
        "Based only on the new filing excerpts below, assess the company's "
        "material investment-relevant opportunities and risks."
    ),

    (
        "Read the supplied excerpts from this newly filed SEC disclosure "
        "and produce the overall evidence-based investment assessment."
    ),
]


# ============================================================
# GENERIC RETRIEVAL QUERY
#
# Same semantics as the clean batch generator.
# ============================================================

def make_query(ticker):

    return (
        "Identify the most material investment-relevant information "
        f"in the newly filed SEC disclosure for {ticker}. "
        "Consider both positive and negative effects on revenue, demand, "
        "growth, market access, costs, operations, competitive position, "
        "financial condition, material risks, and material opportunities."
    )


# ============================================================
# LOAD ATTACK TARGETS
#
# These 62 events define the exact comparison set.
# ============================================================

pairs = pd.read_csv(
    ATTACK_PAIR_PATH
)

pairs["ticker"] = (
    pairs["ticker"]
    .astype(str)
    .str.upper()
)

pairs["event_date"] = pd.to_datetime(
    pairs["event_date"]
).dt.normalize()


NUMERIC_PAIR_COLUMNS = [
    "clean_impact_score",
    "clean_risk_score",
    "clean_confidence",
    "attack_impact_score",
    "attack_risk_score",
    "attack_confidence",
]


for col in NUMERIC_PAIR_COLUMNS:

    pairs[col] = pd.to_numeric(
        pairs[col],
        errors="coerce",
    )


if len(pairs) != 62:

    raise RuntimeError(
        f"Expected 62 target events, found {len(pairs)}."
    )


if pairs[
    NUMERIC_PAIR_COLUMNS
].isna().any().any():

    raise RuntimeError(
        "Missing signal values in attack-pair table."
    )


pairs = pairs.sort_values(
    [
        "ticker",
        "event_date",
    ]
).reset_index(
    drop=True
)


print("=" * 75)
print("BENIGN RAG CONTROL GENERATION")
print("=" * 75)

print(
    "Target events:",
    len(pairs),
)

print(
    "Benign rounds:",
    N_ROUNDS,
)

print(
    "Total target generations:",
    len(pairs) * N_ROUNDS,
)

print(
    "Top-K:",
    TOP_K,
)


# ============================================================
# LOAD INDEX
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDINGS_PATH
).astype(
    np.float32
)


print(
    "Embedding matrix:",
    embeddings.shape,
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
            json.loads(
                line
            )
        )


if len(metadata) != len(embeddings):

    raise RuntimeError(
        "Metadata and embeddings have different lengths."
    )


# ============================================================
# METADATA FIELD HELPERS
# ============================================================

def get_first(meta, candidates):

    for key in candidates:

        if key in meta:

            value = meta[
                key
            ]

            if value is not None:
                return value

    return None


def get_meta_ticker(meta):

    value = get_first(
        meta,
        [
            "ticker",
            "tic",
            "symbol",
        ],
    )

    if value is None:
        return ""

    return str(
        value
    ).upper()


def get_meta_date(meta):

    value = get_first(
        meta,
        [
            "published_at",
            "filing_date",
            "event_date",
            "date",
        ],
    )

    if value is None:
        return None

    try:

        return (
            pd.Timestamp(
                value
            )
            .normalize()
        )

    except Exception:

        return None


def get_meta_text(meta):

    value = get_first(
        meta,
        [
            "text",
            "chunk_text",
            "content",
            "chunk",
        ],
    )

    if value is None:
        return ""

    return str(
        value
    )


meta_tickers = np.array(
    [
        get_meta_ticker(
            m
        )
        for m in metadata
    ],
    dtype=object,
)


meta_dates = np.array(
    [
        (
            get_meta_date(
                m
            ).strftime(
                "%Y-%m-%d"
            )
            if get_meta_date(
                m
            )
            is not None
            else ""
        )
        for m in metadata
    ],
    dtype=object,
)


# ============================================================
# LOAD BGE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nEmbedding device:",
    device,
)


embedder = SentenceTransformer(
    EMBED_MODEL_NAME,
    device=device,
)


# ============================================================
# FIXED CLEAN RETRIEVAL
# ============================================================

retrieval_cache = {}

retrieval_audit_rows = []


def retrieve_clean_event(
    ticker,
    event_date,
):

    event_date = (
        pd.Timestamp(
            event_date
        )
        .normalize()
    )

    key = (
        ticker,
        event_date.strftime(
            "%Y-%m-%d"
        ),
    )


    if key in retrieval_cache:

        return retrieval_cache[
            key
        ]


    date_string = key[
        1
    ]


    candidate_indices = np.where(
        (
            meta_tickers
            == ticker
        )
        &
        (
            meta_dates
            == date_string
        )
    )[0]


    if len(
        candidate_indices
    ) == 0:

        raise RuntimeError(
            f"No clean SEC chunks for "
            f"{ticker} {date_string}"
        )


    query = make_query(
        ticker
    )


    query_embedding = embedder.encode(
        [
            QUERY_PREFIX
            + query
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(
        np.float32
    )


    scores = (
        embeddings[
            candidate_indices
        ]
        @
        query_embedding
    )


    order = np.argsort(
        scores
    )[
        ::-1
    ]


    selected_local = order[
        :TOP_K
    ]


    selected_indices = (
        candidate_indices[
            selected_local
        ]
    )


    selected_scores = (
        scores[
            selected_local
        ]
    )


    chunks = []


    for rank, (
        global_index,
        score,
    ) in enumerate(
        zip(
            selected_indices,
            selected_scores,
        ),
        start=1,
    ):

        meta = metadata[
            int(
                global_index
            )
        ]


        text = get_meta_text(
            meta
        )


        if not text.strip():

            raise RuntimeError(
                f"Empty text in metadata index "
                f"{global_index}"
            )


        chunks.append(
            {
                "rank":
                    rank,

                "index":
                    int(
                        global_index
                    ),

                "score":
                    float(
                        score
                    ),

                "text":
                    text,

                "form":
                    get_first(
                        meta,
                        [
                            "form",
                            "form_type",
                        ],
                    ),
            }
        )


        retrieval_audit_rows.append(
            {
                "ticker":
                    ticker,

                "event_date":
                    date_string,

                "rank":
                    rank,

                "metadata_index":
                    int(
                        global_index
                    ),

                "similarity":
                    float(
                        score
                    ),

                "form":
                    get_first(
                        meta,
                        [
                            "form",
                            "form_type",
                        ],
                    ),
            }
        )


    retrieval_cache[
        key
    ] = chunks


    return chunks


# ============================================================
# PRECOMPUTE ALL CLEAN RETRIEVALS
# ============================================================

print("\nPrecomputing fixed clean retrievals...")


for _, row in pairs.iterrows():

    retrieve_clean_event(
        row[
            "ticker"
        ],
        row[
            "event_date"
        ],
    )


pd.DataFrame(
    retrieval_audit_rows
).drop_duplicates().to_csv(
    RETRIEVAL_AUDIT_PATH,
    index=False,
)


print(
    "Clean retrievals ready:",
    len(
        retrieval_cache
    ),
)


# ============================================================
# LOAD QWEN ONCE
# ============================================================

print("\nLoading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)


model.eval()


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(
    text,
):

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

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )


    if (
        start < 0
        or
        end < start
    ):

        raise ValueError(
            "No JSON object found."
        )


    return json.loads(
        text[
            start:
            end + 1
        ]
    )


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(
    obj,
):

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
                f"Missing key: {key}"
            )


    direction = str(
        obj[
            "direction"
        ]
    ).strip().lower()


    if direction not in {
        "negative",
        "neutral",
        "positive",
    }:

        raise ValueError(
            "Invalid direction."
        )


    impact = float(
        obj[
            "impact_score"
        ]
    )

    risk = float(
        obj[
            "risk_score"
        ]
    )

    confidence = float(
        obj[
            "confidence"
        ]
    )


    if not (
        -1.0
        <= impact
        <= 1.0
    ):

        raise ValueError(
            "impact_score outside [-1,1]"
        )


    if not (
        0.0
        <= risk
        <= 1.0
    ):

        raise ValueError(
            "risk_score outside [0,1]"
        )


    if not (
        0.0
        <= confidence
        <= 1.0
    ):

        raise ValueError(
            "confidence outside [0,1]"
        )


    if (
        direction
        == "negative"
        and
        not impact < 0
    ):

        raise ValueError(
            "negative direction requires impact < 0"
        )


    if (
        direction
        == "positive"
        and
        not impact > 0
    ):

        raise ValueError(
            "positive direction requires impact > 0"
        )


    if (
        direction
        == "neutral"
        and
        abs(
            impact
        )
        > 0.25
    ):

        raise ValueError(
            "neutral requires |impact| <= 0.25"
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
                obj[
                    "reason"
                ]
            ).strip(),
    }


# ============================================================
# QWEN CALL
# ============================================================

def run_qwen(
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


    rendered = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


    model_inputs = tokenizer(
        [
            rendered
        ],
        return_tensors="pt",
    )


    model_inputs = {
        key:
            value.to(
                model.device
            )
        for key, value
        in model_inputs.items()
    }


    with torch.no_grad():

        generated = model.generate(
            **model_inputs,
            max_new_tokens=320,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


    input_length = (
        model_inputs[
            "input_ids"
        ].shape[1]
    )


    generated_tokens = (
        generated[
            0,
            input_length:
        ]
    )


    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


# ============================================================
# GENERATE ONE BENIGN SIGNAL
# ============================================================

def generate_signal(
    ticker,
    event_date,
    chunks,
    round_id,
    event_index,
):

    local_seed = (
        MASTER_SEED
        +
        round_id
        * 100_000
        +
        event_index
        * 101
    )


    rng = np.random.default_rng(
        local_seed
    )


    template_id = int(
        rng.integers(
            0,
            len(
                TASK_VARIANTS
            ),
        )
    )


    evidence_order = (
        rng.permutation(
            len(
                chunks
            )
        )
        .tolist()
    )


    evidence_sections = []


    for displayed_rank, chunk_position in enumerate(
        evidence_order,
        start=1,
    ):

        chunk = chunks[
            chunk_position
        ]


        evidence_sections.append(
            (
                f"[Evidence {displayed_rank}]\n"
                f"{chunk['text']}"
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
        "financial condition, risks, and opportunities. "
        "Use the fixed scoring definitions from the system message.\n\n"
        +
        f"Ticker: {ticker}\n"
        +
        f"Filing date: {pd.Timestamp(event_date).date()}\n\n"
        +
        "\n\n".join(
            evidence_sections
        )
        +
        "\n\nReturn the required JSON object only."
    )


    response = run_qwen(
        user_prompt
    )


    try:

        signal = validate_signal(
            extract_json(
                response
            )
        )

        repaired = False


    except Exception as first_error:

        repair_prompt = (
            "The previous response did not satisfy the required "
            "JSON schema or score consistency rules.\n\n"
            "Previous response:\n"
            f"{response}\n\n"
            "Return ONLY a corrected JSON object using the exact "
            "same evidence and the exact scoring definitions. "
            "Do not introduce new information."
        )


        repair_response = run_qwen(
            repair_prompt
        )


        signal = validate_signal(
            extract_json(
                repair_response
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

            obj = json.loads(
                line
            )


            completed.add(
                (
                    int(
                        obj[
                            "round"
                        ]
                    ),

                    str(
                        obj[
                            "ticker"
                        ]
                    ),

                    str(
                        obj[
                            "event_date"
                        ]
                    ),
                )
            )


print(
    "\nExisting checkpoint rows:",
    len(
        completed
    ),
)


# ============================================================
# GENERATION LOOP
# ============================================================

for round_id in range(
    N_ROUNDS
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"BENIGN ROUND "
        f"{round_id + 1}/{N_ROUNDS}"
    )

    print(
        "=" * 75
    )


    for event_index, row in pairs.iterrows():

        ticker = (
            row[
                "ticker"
            ]
        )

        event_date = (
            pd.Timestamp(
                row[
                    "event_date"
                ]
            )
            .normalize()
        )


        date_string = (
            event_date.strftime(
                "%Y-%m-%d"
            )
        )


        key = (
            round_id,
            ticker,
            date_string,
        )


        if key in completed:

            print(
                f"SKIP {round_id + 1:02d} "
                f"{ticker} {date_string}"
            )

            continue


        chunks = retrieve_clean_event(
            ticker,
            event_date,
        )


        (
            signal,
            template_id,
            evidence_order,
            repaired,
        ) = generate_signal(
            ticker=ticker,
            event_date=event_date,
            chunks=chunks,
            round_id=round_id,
            event_index=event_index,
        )


        canonical_impact = float(
            row[
                "clean_impact_score"
            ]
        )


        benign_impact = float(
            signal[
                "impact_score"
            ]
        )


        record = {
            "round":
                round_id,

            "ticker":
                ticker,

            "event_date":
                date_string,

            "template_id":
                template_id,

            "evidence_order":
                evidence_order,

            "retrieval_indices":
                [
                    int(
                        chunk[
                            "index"
                        ]
                    )
                    for chunk in chunks
                ],

            "repaired":
                repaired,

            "benign_direction":
                signal[
                    "direction"
                ],

            "benign_impact_score":
                benign_impact,

            "benign_risk_score":
                float(
                    signal[
                        "risk_score"
                    ]
                ),

            "benign_confidence":
                float(
                    signal[
                        "confidence"
                    ]
                ),

            "benign_reason":
                signal[
                    "reason"
                ],

            "canonical_clean_impact":
                canonical_impact,

            "canonical_clean_risk":
                float(
                    row[
                        "clean_risk_score"
                    ]
                ),

            "canonical_clean_confidence":
                float(
                    row[
                        "clean_confidence"
                    ]
                ),

            "impact_delta_vs_canonical":
                (
                    benign_impact
                    -
                    canonical_impact
                ),

            "abs_impact_delta_vs_canonical":
                abs(
                    benign_impact
                    -
                    canonical_impact
                ),

            "impact_sign_changed":
                bool(
                    np.sign(
                        benign_impact
                    )
                    !=
                    np.sign(
                        canonical_impact
                    )
                ),
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
            f"OK {round_id + 1:02d} "
            f"{ticker} {date_string} "
            f"impact "
            f"{canonical_impact:+.2f}"
            f" -> "
            f"{benign_impact:+.2f}"
        )


# ============================================================
# BUILD FINAL CSV
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


result = pd.DataFrame(
    records
)


result = result[
    result[
        "round"
    ]
    <
    N_ROUNDS
].copy()


result = (
    result
    .sort_values(
        [
            "round",
            "ticker",
            "event_date",
        ]
    )
    .drop_duplicates(
        subset=[
            "round",
            "ticker",
            "event_date",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


expected = (
    N_ROUNDS
    *
    len(
        pairs
    )
)


if len(
    result
) != expected:

    raise RuntimeError(
        f"Expected {expected} final rows, "
        f"found {len(result)}."
    )


result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# QA SUMMARY
# ============================================================

round_summary = (
    result.groupby(
        "round"
    )
    .agg(
        mean_abs_impact_delta=(
            "abs_impact_delta_vs_canonical",
            "mean",
        ),

        sign_changes=(
            "impact_sign_changed",
            "sum",
        ),

        repaired_events=(
            "repaired",
            "sum",
        ),
    )
    .reset_index()
)


round_summary.to_csv(
    RESULT_DIR
    / "benign_round_upstream_summary.csv",
    index=False,
)


print("\n" + "=" * 75)
print("BENIGN GENERATION COMPLETE")
print("=" * 75)

print(
    "Rows:",
    len(
        result
    ),
)

print(
    "Mean |benign impact delta|:",
    f"{result['abs_impact_delta_vs_canonical'].mean():.4f}",
)

print(
    "Mean sign changes per 62-event round:",
    f"{round_summary['sign_changes'].mean():.2f}",
)

print(
    "Maximum sign changes in a round:",
    int(
        round_summary[
            "sign_changes"
        ].max()
    ),
)

print(
    "Repair rate:",
    f"{result['repaired'].mean():.2%}",
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print("\nDONE")