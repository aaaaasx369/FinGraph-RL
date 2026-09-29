import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = ROOT / "rag" / "data" / "index"

EMBEDDING_PATH = (
    INDEX_DIR / "sec_embeddings.npy"
)

METADATA_PATH = (
    INDEX_DIR / "sec_metadata.jsonl"
)

OUTPUT_DIR = (
    ROOT / "rag" / "data" / "signals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Models
# ============================================================

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

DEFAULT_LLM_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ticker",
    required=True,
)

parser.add_argument(
    "--as-of",
    required=True,
    help="YYYY-MM-DD",
)

parser.add_argument(
    "--query",
    default=None,
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

args = parser.parse_args()


ticker = args.ticker.upper()
as_of = args.as_of
top_k = args.top_k


if args.query:
    query = args.query
else:
    query = (
        f"Assess the most material investment-relevant "
        f"risks and opportunities for {ticker}."
    )


# ============================================================
# Load SEC vector index
# ============================================================

print("=" * 70)
print("LOAD SEC RETRIEVAL INDEX")
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
    "Total SEC chunks:",
    len(metadata),
)

print(
    "Embedding shape:",
    embeddings.shape,
)


# ============================================================
# Point-in-time filtering
# ============================================================

candidate_indices = []

for i, doc in enumerate(metadata):

    if (
        doc["ticker"].upper()
        != ticker
    ):
        continue

    # Never allow future information
    if (
        doc["published_at"]
        > as_of
    ):
        continue

    candidate_indices.append(i)


if not candidate_indices:

    raise RuntimeError(
        f"No SEC evidence available for "
        f"{ticker} as of {as_of}"
    )


print(
    "Candidate chunks:",
    len(candidate_indices),
)


# ============================================================
# Embedding model
# ============================================================

embedding_device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Embedding device:",
    embedding_device,
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=embedding_device,
)


# ============================================================
# Query embedding
# ============================================================

query_text = (
    QUERY_PREFIX
    + query
)

query_embedding = embedding_model.encode(
    [query_text],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0].astype(
    np.float32
)


# ============================================================
# Retrieval
# ============================================================

candidate_embeddings = embeddings[
    candidate_indices
]

scores = (
    candidate_embeddings
    @ query_embedding
)

ranking = np.argsort(
    scores
)[::-1]

top_n = min(
    top_k,
    len(ranking),
)


retrieved = []

for rank in range(top_n):

    local_index = ranking[
        rank
    ]

    global_index = candidate_indices[
        local_index
    ]

    doc = metadata[
        global_index
    ]

    score = float(
        scores[local_index]
    )

    retrieved.append(
        {
            "rank": rank + 1,
            "score": score,
            "chunk_id": doc["chunk_id"],
            "doc_id": doc["doc_id"],
            "ticker": doc["ticker"],
            "published_at": doc["published_at"],
            "form": doc["form"],
            "title": doc["title"],
            "url": doc["url"],
            "text": doc["text"],
        }
    )


# ============================================================
# Show evidence
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVED EVIDENCE")
print("=" * 70)

for item in retrieved:

    print(
        f"\nRANK {item['rank']}"
    )

    print(
        f"Score : {item['score']:.4f}"
    )

    print(
        "Date  :",
        item["published_at"],
    )

    print(
        "Form  :",
        item["form"],
    )

    print(
        "Chunk :",
        item["chunk_id"],
    )

    print(
        "Text  :",
        item["text"][:500],
    )

    print(
        "-" * 70
    )


# ============================================================
# Build evidence context
# ============================================================

evidence_blocks = []

for item in retrieved:

    evidence_blocks.append(
        (
            f"[Evidence {item['rank']}]\n"
            f"Ticker: {item['ticker']}\n"
            f"Published: {item['published_at']}\n"
            f"Form: {item['form']}\n"
            f"Similarity: {item['score']:.4f}\n"
            f"Text:\n{item['text']}\n"
        )
    )


evidence_context = "\n\n".join(
    evidence_blocks
)


# ============================================================
# Calibration prompt
# ============================================================

system_prompt = """
You are a financial evidence scoring system.

You must use ONLY the supplied SEC filing evidence.

Do not use outside knowledge.
Do not use information after the stated as-of date.
Do not predict future events.
Do not give investment advice.

Your task is to measure the direction and magnitude of the
investment implications contained in the supplied evidence.

IMPORTANT:

"impact_score" measures whether the evidence is favorable or
unfavorable for the COMPANY.

It does NOT measure importance.
It does NOT measure confidence.
It does NOT measure severity alone.

Examples:

Evidence saying:
- revenue may decline
- demand may decline
- company may lose access to a market
- costs may rise
- operations may be disrupted
- supply chain may be disrupted

must generally have NEGATIVE impact_score.

Example:
{
  "direction": "negative",
  "impact_score": -0.8,
  "risk_score": 0.9,
  "confidence": 0.9,
  "reason": "The evidence describes reduced market access and possible revenue loss."
}

Evidence saying:
- demand is increasing
- revenue is growing
- market access is expanding
- costs are falling
- competitive position is improving

must generally have POSITIVE impact_score.

Example:
{
  "direction": "positive",
  "impact_score": 0.8,
  "risk_score": 0.2,
  "confidence": 0.9,
  "reason": "The evidence describes stronger demand and improved business prospects."
}

If evidence is mixed or has no clear directional implication,
use direction "neutral" and keep impact_score close to zero.

Output exactly one JSON object.

Required schema:

{
  "direction": "negative" | "neutral" | "positive",
  "impact_score": number,
  "risk_score": number,
  "confidence": number,
  "reason": "short evidence-based explanation"
}

Ranges:

impact_score:
-1 = extremely unfavorable
 0 = neutral or mixed
+1 = extremely favorable

risk_score:
0 = very little material business/investment risk
1 = extremely high material business/investment risk

confidence:
0 = evidence is weak or insufficient
1 = evidence is direct and strongly supports the assessment

Consistency requirements:

If direction is "negative",
impact_score MUST be below 0.

If direction is "positive",
impact_score MUST be above 0.

If direction is "neutral",
impact_score MUST be between -0.25 and +0.25.

Do not use markdown.
Return JSON only.
""".strip()


user_prompt = f"""
Ticker:
{ticker}

As-of date:
{as_of}

Research question:
{query}

SEC evidence:

{evidence_context}

First determine whether the evidence is favorable,
unfavorable, or neutral for the company.

Then score the magnitude.

Return only the required JSON object.
""".strip()


# ============================================================
# Load local LLM
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
    tokenizer.pad_token = tokenizer.eos_token


if torch.cuda.is_available():

    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16

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
# Generation helper
# ============================================================

def generate_json(
    system_text,
    user_text,
):

    messages = [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": user_text,
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
        key: value.to(model_device)
        for key, value
        in model_inputs.items()
    }

    prompt_length = (
        model_inputs["input_ids"]
        .shape[1]
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
# JSON parsing
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
            "No JSON object found in model output."
        )

    return json.loads(
        text[
            start:end + 1
        ]
    )


# ============================================================
# Signal validation
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
# Direction-score consistency check
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
# First generation
# ============================================================

print(
    "\nGenerating calibrated RAG signal..."
)

raw_output = generate_json(
    system_prompt,
    user_prompt,
)


print(
    "\nRaw LLM output:"
)

print(
    raw_output
)


data = parse_json_object(
    raw_output
)

signal = normalize_signal(
    data
)


# ============================================================
# Automatic repair if sign is inconsistent
# ============================================================

repair_used = False
repair_output = None


if not is_consistent(
    signal
):

    repair_used = True

    print(
        "\nWARNING:"
        " direction and impact_score are inconsistent."
    )

    print(
        "Running one calibration repair pass..."
    )


    repair_prompt = f"""
The previous output was internally inconsistent.

Previous output:

{json.dumps(signal, ensure_ascii=False)}

You must correct the JSON.

Remember:

negative direction -> impact_score < 0

positive direction -> impact_score > 0

neutral direction -> impact_score between -0.25 and +0.25

Use ONLY the original SEC evidence below.

Ticker:
{ticker}

As-of:
{as_of}

Question:
{query}

Evidence:

{evidence_context}

Return corrected JSON only.
""".strip()


    repair_output = generate_json(
        system_prompt,
        repair_prompt,
    )


    print(
        "\nRepair output:"
    )

    print(
        repair_output
    )


    repair_data = parse_json_object(
        repair_output
    )

    signal = normalize_signal(
        repair_data
    )


    if not is_consistent(
        signal
    ):

        raise RuntimeError(
            "LLM still produced an inconsistent "
            "direction / impact_score after repair."
        )


# ============================================================
# Save final result
# ============================================================

result = {
    "ticker": ticker,

    "as_of": as_of,

    "query": query,

    "retrieval_model":
        EMBEDDING_MODEL_NAME,

    "generation_model":
        args.llm_model,

    "top_k":
        top_n,

    "direction":
        signal["direction"],

    "impact_score":
        signal["impact_score"],

    "risk_score":
        signal["risk_score"],

    "confidence":
        signal["confidence"],

    "reason":
        signal["reason"],

    "calibration_repair_used":
        repair_used,

    "evidence": [
        {
            "rank": x["rank"],
            "score": x["score"],
            "chunk_id": x["chunk_id"],
            "published_at":
                x["published_at"],
            "form": x["form"],
            "url": x["url"],
        }
        for x in retrieved
    ],

    "raw_model_output":
        raw_output,

    "repair_model_output":
        repair_output,
}


safe_date = (
    as_of.replace(
        "-",
        "",
    )
)


output_path = (
    OUTPUT_DIR
    / (
        f"{ticker}_"
        f"{safe_date}_"
        f"sec_signal_calibrated.json"
    )
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Final display
# ============================================================

print("\n" + "=" * 70)
print("FINAL CALIBRATED RAG SIGNAL")
print("=" * 70)

print(
    "Ticker      :",
    ticker,
)

print(
    "As-of       :",
    as_of,
)

print(
    "Direction   :",
    signal["direction"],
)

print(
    "Impact      :",
    signal["impact_score"],
)

print(
    "Risk        :",
    signal["risk_score"],
)

print(
    "Confidence  :",
    signal["confidence"],
)

print(
    "Reason      :",
    signal["reason"],
)

print(
    "Repair used :",
    repair_used,
)

print(
    "\nSaved:"
)

print(
    output_path
)

print(
    "\nDONE"
)