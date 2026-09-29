from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def chunk_words(text: str, chunk_words: int = 900, overlap_words: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        c = words[start:start + chunk_words]
        if len(c) < 80 and chunks:
            break
        chunks.append(" ".join(c))
    return chunks


def _extract_json(text: str) -> dict:
    # Remove markdown fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "")
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    obj = json.loads(m.group(0))
    return obj


def _clip_signal(obj: dict) -> dict:
    return {
        "impact": float(np.clip(float(obj["impact"]), -1.0, 1.0)),
        "risk": float(np.clip(float(obj["risk"]), 0.0, 1.0)),
        "confidence": float(np.clip(float(obj["confidence"]), 0.0, 1.0)),
        "rationale": str(obj.get("rationale", ""))[:1500],
    }


class SECRAGSignalGenerator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg["embedding_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"])
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg["llm_model"],
            torch_dtype="auto",
            device_map="auto",
        )
        self.llm.eval()

        self.queries = list(cfg["queries"])
        self.query_embeddings = self.embedder.encode(
            self.queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def retrieve(self, text: str) -> Tuple[List[str], List[float]]:
        chunks = chunk_words(
            text,
            int(self.cfg["chunk_words"]),
            int(self.cfg["chunk_overlap_words"]),
        )
        if not chunks:
            return [], []

        emb = self.embedder.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        # Max similarity across several investment-oriented queries.
        sims = emb @ self.query_embeddings.T
        score = sims.max(axis=1)
        k = min(int(self.cfg["top_k"]), len(chunks))
        idx = np.argsort(score)[-k:][::-1]
        return [chunks[i] for i in idx], [float(score[i]) for i in idx]

    def infer_signal(self, ticker: str, filing_date: str, filing_type: str, evidence: List[str]) -> dict:
        ev = "\n\n".join(
            f"[Evidence {i+1}]\n{x}" for i, x in enumerate(evidence)
        )
        ev = ev[: int(self.cfg.get("max_evidence_chars", 24000))]

        system = (
            "You are a conservative financial filing analyst. "
            "Use ONLY the supplied SEC evidence. Do not invent facts. "
            "Return one strict JSON object and no markdown."
        )
        user = f"""
Ticker: {ticker}
Filing date: {filing_date}
Filing type: {filing_type}

Assess the filing information that could matter for future portfolio allocation.

Return exactly:
{{
  "impact": <number from -1.0 strongly negative to +1.0 strongly positive>,
  "risk": <number from 0.0 low material risk to 1.0 very high material risk>,
  "confidence": <number from 0.0 to 1.0 reflecting evidence strength>,
  "rationale": "<brief evidence-grounded explanation>"
}}

Rules:
- impact measures expected fundamental direction, not same-day price direction.
- risk measures material downside/uncertainty in the evidence.
- confidence must be LOW when evidence is ambiguous, incomplete, boilerplate, or conflicting.
- no outside knowledge.
- do not infer information dated after the filing date.

SEC evidence:
{ev}
""".strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.llm.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            out = self.llm.generate(
                **model_inputs,
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        prompt_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return _clip_signal(_extract_json(text))

    def process_dataframe(self, sec_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
        rows = []
        for r in tqdm(sec_df.itertuples(index=False), total=len(sec_df), desc="SEC-RAG"):
            ticker = str(r.ticker)
            filing_date = str(pd.Timestamp(r.filing_date).date())
            filing_type = str(getattr(r, "filing_type", "UNKNOWN"))

            raw_text = getattr(r, "text", None)
            if raw_text is None or (isinstance(raw_text, float) and np.isnan(raw_text)):
                path = Path(str(r.text_path))
                if not path.is_absolute():
                    path = base_dir / path
                raw_text = path.read_text(encoding="utf-8", errors="ignore")

            evidence, scores = self.retrieve(str(raw_text))
            if not evidence:
                sig = {
                    "impact": 0.0, "risk": 0.0, "confidence": 0.0,
                    "rationale": "No usable evidence retrieved."
                }
            else:
                try:
                    sig = self.infer_signal(ticker, filing_date, filing_type, evidence)
                except Exception as e:
                    # Fail closed: do not allow malformed LLM output to become a strong signal.
                    sig = {
                        "impact": 0.0, "risk": 0.0, "confidence": 0.0,
                        "rationale": f"LLM parse/inference failure: {type(e).__name__}"
                    }

            rows.append({
                "ticker": ticker,
                # The signal becomes legally/causally available on the SEC filing date.
                "available_date": filing_date,
                "filing_type": filing_type,
                **sig,
                "retrieval_score_mean": float(np.mean(scores)) if scores else np.nan,
                "retrieval_score_max": float(np.max(scores)) if scores else np.nan,
            })

        return pd.DataFrame(rows)
