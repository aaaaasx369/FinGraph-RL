#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V8 Phase 1

REPLUG-inspired:
  - Point-in-time SEC retrieval
  - Frozen downstream teacher
  - KL-distilled document reranker

FinAgent-inspired:
  - Historical market information
  - Retrieved financial document representation
  - Return + risk auxiliary prediction heads

Train:
  2018-2019: downstream teacher
  2020:      retriever distillation
  2018-2020: auxiliary predictor
Validation:
  2021 only

No 2022-2023 evaluation.
No new LLM requests.
No A2C retraining.

This is a research adaptation, not exact paper replication.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b
)

TICKERS = v7b.TICKERS
DOC_DIM = 64
HORIZON = 5


def log(*x):
    print(*x, flush=True)


def load_documents(path):
    raw = pd.read_parquet(path)

    required = [
        "ticker",
        "available_from",
        "current_excerpt",
        "evidence_id",
    ]

    for col in required:
        if col not in raw:
            raise RuntimeError(
                f"Missing document column: {col}"
            )

    raw = raw[
        raw["ticker"].isin(TICKERS)
    ].copy()

    raw["available_from"] = pd.to_datetime(
        raw["available_from"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert(None).dt.normalize()

    raw = raw.dropna(
        subset=["available_from"]
    )

    raw["text"] = (
        raw["current_excerpt"]
        .fillna("")
        .astype(str)
    )

    raw = raw[
        raw["text"].str.len() >= 25
    ].copy()

    # Keep unique evidence IDs.
    raw = raw.drop_duplicates(
        "evidence_id",
        keep="first",
    ).reset_index(drop=True)

    # Source dates are date-resolution only.
    # Apply an additional one-calendar-day lag.
    raw["usable_from"] = (
        raw["available_from"]
        + pd.Timedelta(days=1)
    )

    return raw


def market_features(prices):
    """
    Uses prices at or before t only.
    Returns [days, stocks, 5].
    """

    p = np.asarray(
        prices,
        dtype=np.float64
    )

    n = len(p)
    out = np.zeros((n, 5, 5), dtype=np.float32)

    for t in range(n):
        for j in range(5):
            r1 = (
                np.log(p[t,j] / p[t-1,j])
                if t >= 1 else 0.0
            )

            r5 = (
                np.log(p[t,j] / p[t-5,j])
                if t >= 5 else 0.0
            )

            r20 = (
                np.log(p[t,j] / p[t-20,j])
                if t >= 20 else 0.0
            )

            if t >= 20:
                daily = np.diff(
                    np.log(
                        p[t-20:t+1,j]
                    )
                )

                vol = daily.std(ddof=1)
                mean = daily.mean()
            else:
                vol = 0.0
                mean = 0.0

            out[t,j] = [
                r1,
                r5,
                r20,
                mean,
                vol,
            ]

    return out


def make_document_vectors(raw):
    vectorizer = HashingVectorizer(
        n_features=DOC_DIM,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )

    documents = vectorizer.transform(
        raw["text"].tolist()
    ).toarray().astype(np.float32)

    return vectorizer, documents


def build_retrieval_dataset(
    split_data,
    raw,
    doc_vectors,
    vectorizer,
    top_k,
    split,
):
    """
    Each sample:
      target company
      decision date
      historical market features
      top-k PIT SEC documents
      5-day future return (label only)

    Never puts future return into retriever input.
    """

    dates = pd.DatetimeIndex(
        split_data["dates"]
    )

    prices = np.asarray(
        split_data["prices"],
        dtype=np.float64,
    )

    mkt = market_features(prices)

    index_by_ticker = {}

    for ticker in TICKERS:
        ids = np.flatnonzero(
            raw["ticker"].to_numpy() == ticker
        )

        index_by_ticker[ticker] = ids

    usable = raw[
        "usable_from"
    ].to_numpy(dtype="datetime64[ns]")

    records = []
    contexts = []
    docs = []
    semantic = []
    targets = []
    risks = []
    doc_ids = []

    for t in range(20, len(dates)-HORIZON):
        date = pd.Timestamp(dates[t])

        if split == "train":
            if date.year > 2020:
                continue
        elif split == "val":
            if date.year != 2021:
                continue
        else:
            raise RuntimeError(
                "Only train/val supported"
            )

        for j, ticker in enumerate(TICKERS):
            ids = index_by_ticker[ticker]

            # All candidate documents were available
            # strictly before decision day.
            valid = ids[
                usable[ids] <= np.datetime64(date)
            ]

            if len(valid) == 0:
                continue

            recent_return = mkt[t,j,2]

            regime = (
                "decline loss risk"
                if recent_return < 0
                else "growth earnings revenue"
            )

            query = (
                f"{ticker} financial filing "
                f"business performance revenue "
                f"risk cash flow {regime}"
            )

            q = vectorizer.transform(
                [query]
            ).toarray()[0]

            scores = (
                doc_vectors[valid] @ q
            )

            # Deterministic highest-score retrieval.
            order = np.lexsort(
                (valid, -scores)
            )[:top_k]

            chosen = valid[order]
            selected = doc_vectors[chosen]
            selected_scores = scores[order]

            # Pad to fixed candidate count.
            k = len(chosen)

            padded = np.zeros(
                (top_k, DOC_DIM),
                dtype=np.float32
            )

            padded_scores = np.full(
                top_k,
                -1e4,
                dtype=np.float32
            )

            padded[:k] = selected
            padded_scores[:k] = selected_scores

            ids_out = [""] * top_k

            for a, di in enumerate(chosen):
                ids_out[a] = str(
                    raw.iloc[int(di)]["evidence_id"]
                )

            context = np.concatenate([
                mkt[t,j],
                np.eye(5, dtype=np.float32)[j],
            ])

            future = (
                prices[t+HORIZON,j]
                / prices[t,j]
                - 1.0
            )

            contexts.append(context)
            docs.append(padded)
            semantic.append(padded_scores)
            targets.append(float(future))
            risks.append(abs(float(future)))

            doc_ids.append(ids_out)

            records.append({
                "date": str(date.date()),
                "target_date": str(
                    dates[t+HORIZON].date()
                ),
                "ticker": ticker,
                "year": date.year,
                "n_documents": k,
            })

    if not records:
        raise RuntimeError(
            f"No retrieval samples in {split}"
        )

    return {
        "meta": pd.DataFrame(records),
        "market": np.asarray(
            contexts,
            dtype=np.float32
        ),
        "docs": np.asarray(
            docs,
            dtype=np.float32
        ),
        "semantic": np.asarray(
            semantic,
            dtype=np.float32
        ),
        "returns": np.asarray(
            targets,
            dtype=np.float32
        ),
        "risks": np.asarray(
            risks,
            dtype=np.float32
        ),
        "doc_ids": doc_ids,
    }


def softmax_np(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x / temperature
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(
        axis=-1,
        keepdims=True
    )


def semantic_weights(data):
    return softmax_np(
        data["semantic"],
        temperature=0.2
    ).astype(np.float32)


def masked_softmax(logits, valid):
    logits = logits.masked_fill(
        ~valid,
        -1e5,
    )
    return torch.softmax(logits, dim=-1)


class Reranker(nn.Module):
    """
    Candidate document scorer.

    Input:
      market context 10D
      candidate document vector 64D
      semantic similarity 1D
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(75, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        market,
        docs,
        semantic,
    ):
        k = docs.shape[1]

        context = market[:,None,:].expand(
            -1,k,-1
        )

        features = torch.cat([
            context,
            docs,
            semantic.unsqueeze(-1),
        ],dim=-1)

        return self.net(
            features
        ).squeeze(-1)


class AuxiliaryPredictor(nn.Module):
    """
    FinAgent-inspired financial evidence representation.

    Shared trunk:
      market context + retrieved evidence

    Auxiliary heads:
      forward return
      absolute forward return (risk proxy)

    This is our adaptation, not original FinAgent.
    """

    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(10 + DOC_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.return_head = nn.Linear(32,1)
        self.risk_head = nn.Sequential(
            nn.Linear(32,1),
            nn.Softplus(),
        )

    def forward(self, market, doc):
        z = self.trunk(
            torch.cat([market,doc],dim=1)
        )

        return (
            self.return_head(z).squeeze(-1),
            self.risk_head(z).squeeze(-1),
        )


def get_tensors(data):
    return {
        "m": torch.from_numpy(
            data["market"]
        ),
        "d": torch.from_numpy(
            data["docs"]
        ),
        "s": torch.from_numpy(
            data["semantic"]
        ),
        "y": torch.from_numpy(
            data["returns"]
        ),
        "risk": torch.from_numpy(
            data["risks"]
        ),
    }


def train_reranker(
    train,
    epochs,
    batch_size,
    seed,
):
    """
    Teacher:
      frozen Ridge trained on 2018-2019,
      then scores candidate document predictions
      for 2020 by squared forecast error.

    REPLUG-inspired:
      teacher distribution from downstream quality
      KL(teacher || retriever).

    This is a financial adaptation of REPLUG LSR.
    """

    years = train["meta"]["year"].to_numpy()

    target_dates = pd.to_datetime(
        train["meta"]["target_date"]
    ).to_numpy()

    pretrain = (
        (years <= 2019)
        & (
            target_dates
            <= np.datetime64("2019-12-31")
        )
    )

    distill = years == 2020

    if pretrain.sum() < 100 or distill.sum() < 100:
        raise RuntimeError(
            "Insufficient chronological samples"
        )

    sem = semantic_weights(train)

    pooled = (
        train["docs"] * sem[:,:,None]
    ).sum(axis=1)

    teacher_x = np.concatenate([
        train["market"],
        pooled,
    ],axis=1)

    teacher = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0),
    )

    teacher.fit(
        teacher_x[pretrain],
        train["returns"][pretrain],
    )

    # Each candidate is evaluated separately
    # by the frozen teacher.
    distill_ids = np.flatnonzero(distill)

    quality = np.zeros(
        (len(distill_ids),train["docs"].shape[1]),
        dtype=np.float32,
    )

    for a, i in enumerate(distill_ids):
        candidates = np.concatenate([
            np.repeat(
                train["market"][i:i+1],
                train["docs"].shape[1],
                axis=0,
            ),
            train["docs"][i],
        ],axis=1)

        predictions = teacher.predict(candidates)

        errors = (
            predictions - train["returns"][i]
        ) ** 2

        quality[a] = -errors

    # Normalize teacher scores to avoid a numerical
    # collapse due to extremely small return losses.
    valid = (
        train["semantic"][distill_ids] > -1e3
    )

    quality_scale = max(
        float(
            np.std(
                quality[valid]
            )
        ),
        1e-6,
    )

    teacher_scores = (
        quality / quality_scale
    )

    teacher_scores[~valid] = -1e4

    teacher_probs = softmax_np(
        teacher_scores,
        temperature=1.0,
    ).astype(np.float32)

    tensors = get_tensors(train)

    model = Reranker()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(
            len(distill_ids)
        )

        losses = []

        for start in range(
            0,len(order),batch_size
        ):
            idx = order[start:start+batch_size]
            selected = distill_ids[idx]

            m = tensors["m"][selected]
            d = tensors["d"][selected]
            s = tensors["s"][selected]

            mask = s > -1e3

            logits = model(m,d,s)

            log_probs = torch.log(
                masked_softmax(
                    logits,
                    mask,
                ).clamp_min(1e-9)
            )

            target = torch.from_numpy(
                teacher_probs[idx]
            )

            # Cross-entropy differs from KL
            # only by constant target entropy.
            loss = -(
                target * log_probs
            ).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        log(
            f"[RETRIEVER] epoch={epoch+1} "
            f"loss={np.mean(losses):.6f}"
        )

    return teacher,model,quality_scale


@torch.no_grad()
def reranker_weights(model,data):
    t = get_tensors(data)

    logits = model(
        t["m"],
        t["d"],
        t["s"],
    )

    valid = t["s"] > -1e3

    return masked_softmax(
        logits,
        valid,
    ).numpy().astype(np.float32)


def pooled_evidence(data,weights):
    return (
        data["docs"] * weights[:,:,None]
    ).sum(axis=1).astype(np.float32)


def fit_auxiliary(
    train,
    doc_train,
    epochs,
    seed,
):
    torch.manual_seed(seed)

    model = AuxiliaryPredictor()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    m = torch.from_numpy(
        train["market"]
    )

    d = torch.from_numpy(
        doc_train
    )

    y = torch.from_numpy(
        train["returns"]
    )

    r = torch.from_numpy(
        train["risks"]
    )

    # Scale both targets for stable training.
    scale = max(
        float(y.std().item()),
        1e-3,
    )

    yy = y / scale
    rr = r / scale

    for epoch in range(epochs):
        perm = torch.randperm(len(y))
        losses = []

        for ids in perm.split(256):
            pred_y,pred_r = model(
                m[ids],
                d[ids],
            )

            loss_return = (
                (pred_y-yy[ids])**2
            ).mean()

            loss_risk = (
                (pred_r-rr[ids])**2
            ).mean()

            loss = (
                loss_return
                + 0.3 * loss_risk
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(
                float(loss.item())
            )

        if (
            epoch == 0
            or (epoch+1)%10 == 0
            or epoch+1 == epochs
        ):
            log(
                f"[AUX] epoch={epoch+1} "
                f"loss={np.mean(losses):.6f}"
            )

    return model,scale


@torch.no_grad()
def evaluate_aux(
    model,
    scale,
    data,
    doc_features,
):
    y_pred,r_pred = model(
        torch.from_numpy(
            data["market"]
        ),
        torch.from_numpy(
            doc_features
        ),
    )

    return (
        y_pred.numpy()*scale,
        r_pred.numpy()*scale,
    )


def metrics(data,pred,risk_pred):
    y = data["returns"]
    r = data["risks"]

    baseline = np.full_like(
        y,
        y.mean(),
    )

    mse = float(
        np.mean((y-pred)**2)
    )

    mae = float(
        np.mean(np.abs(y-pred))
    )

    directional = float(
        np.mean(
            (pred > 0) == (y > 0)
        )
    )

    risk_mae = float(
        np.mean(
            np.abs(r-risk_pred)
        )
    )

    values = pd.DataFrame({
        "date": data["meta"]["date"],
        "true": y,
        "pred": pred,
    })

    daily_ic = []

    for _,group in values.groupby("date"):
        if (
            len(group) < 3
            or group["pred"].std() < 1e-10
            or group["true"].std() < 1e-10
        ):
            continue

        ic = group[
            ["true","pred"]
        ].corr(method="spearman").iloc[0,1]

        if np.isfinite(ic):
            daily_ic.append(float(ic))

    return {
        "n": len(y),
        "mse": mse,
        "mae": mae,
        "directional_accuracy": directional,
        "risk_mae": risk_mae,
        "rank_ic": (
            float(np.mean(daily_ic))
            if daily_ic else float("nan")
        ),
        "rank_ic_days": len(daily_ic),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--documents",
        default="outputs/v31/evidence_alignment.parquet",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/v8_replug_finagent"
        ),
    )

    p.add_argument("--top-k",type=int,default=5)
    p.add_argument("--retriever-epochs",type=int,default=20)
    p.add_argument("--aux-epochs",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cpu")

    # Compatibility with the existing project loader.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
        ),
    )

    p.add_argument(
        "--sec-signals",
        default="outputs/unified_event_memory_v2/sec_numeric_signals.csv"
    )

    p.add_argument(
        "--kgt",
        default="outputs/aegis_kgt_daily_2018_2023/daily_dynamic_kgt.csv"
    )

    p.add_argument(
        "--kg",
        default="outputs/unified_temporal_kg_v3/unified_temporal_edges.csv"
    )

    p.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl"
    )

    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434"
    )

    p.add_argument("--llm",default="qwen2.5:3b")
    p.add_argument("--embed",default="bge-m3:latest")
    p.add_argument("--transaction-cost",type=float,default=0.001)
    p.add_argument("--cost-scale",type=float,default=0)
    p.add_argument("--reward-scaling",type=float,default=1)
    p.add_argument("--tier",default="conservative")
    p.add_argument("--learning-rate",type=float,default=0.0003)
    p.add_argument("--max-days",type=int,default=0)
    p.add_argument("--split",default="train")
    p.add_argument("--algo",default="A2C")

    args = p.parse_args()

    if args.top_k < 2:
        raise RuntimeError("top-k must be at least 2")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log("===== V8 LOAD PROJECT DATA =====")

    train_data = v7b.load_split(args,"train")
    val_data = v7b.load_split(args,"val")

    raw = load_documents(args.documents)

    log("[DOCUMENTS]",len(raw))

    vectorizer, doc_vectors = make_document_vectors(raw)

    log("[DOCUMENT EMBEDDINGS]",doc_vectors.shape)

    train = build_retrieval_dataset(
        train_data,
        raw,
        doc_vectors,
        vectorizer,
        args.top_k,
        "train",
    )

    val = build_retrieval_dataset(
        val_data,
        raw,
        doc_vectors,
        vectorizer,
        args.top_k,
        "val",
    )

    log("[TRAIN SAMPLES]",len(train["returns"]))
    log("[VAL SAMPLES]",len(val["returns"]))

    log()
    log("===== REPLUG-INSPIRED RETRIEVER =====")

    teacher,ranker,score_scale = train_reranker(
        train,
        epochs=args.retriever_epochs,
        batch_size=128,
        seed=args.seed,
    )

    torch.save(
        ranker.state_dict(),
        args.out/"retriever.pt",
    )

    train_sem = semantic_weights(train)
    val_sem = semantic_weights(val)

    train_ranked = reranker_weights(
        ranker,train
    )

    val_ranked = reranker_weights(
        ranker,val
    )

    options = {
        "MARKET_ONLY": (
            np.zeros(
                (len(train["returns"]),DOC_DIM),
                dtype=np.float32
            ),
            np.zeros(
                (len(val["returns"]),DOC_DIM),
                dtype=np.float32
            ),
        ),
        "SEMANTIC_RAG": (
            pooled_evidence(train,train_sem),
            pooled_evidence(val,val_sem),
        ),
        "LEARNED_RAG": (
            pooled_evidence(train,train_ranked),
            pooled_evidence(val,val_ranked),
        ),
    }

    rows = []

    for name,(doc_train,doc_val) in options.items():
        log()
        log("="*65)
        log("[ARM]",name)
        log("="*65)

        predictor,scale = fit_auxiliary(
            train,
            doc_train,
            epochs=args.aux_epochs,
            seed=args.seed,
        )

        pred,risk_pred = evaluate_aux(
            predictor,
            scale,
            val,
            doc_val,
        )

        result = metrics(
            val,pred,risk_pred
        )

        result["arm"] = name
        result["seed"] = args.seed

        rows.append(result)

        torch.save({
            "state_dict": predictor.state_dict(),
            "target_scale": scale,
            "arm": name,
        },args.out/f"{name}_predictor.pt")

        output = val["meta"].copy()
        output["actual_return"] = val["returns"]
        output["predicted_return"] = pred
        output["actual_risk"] = val["risks"]
        output["predicted_risk"] = risk_pred

        output.to_csv(
            args.out/f"{name}_predictions.csv",
            index=False,
        )

        log("[VALIDATION]",result)

    results = pd.DataFrame(rows)

    results.to_csv(
        args.out/"validation_metrics.csv",
        index=False,
    )

    # Retrieval transparency: document IDs and scores.
    audit = val["meta"].copy()

    for j in range(args.top_k):
        audit[f"doc_{j}_id"] = [
            items[j]
            for items in val["doc_ids"]
        ]
        audit[f"doc_{j}_semantic_weight"] = (
            val_sem[:,j]
        )
        audit[f"doc_{j}_learned_weight"] = (
            val_ranked[:,j]
        )

    audit.to_csv(
        args.out/"validation_retrieval_audit.csv",
        index=False,
    )

    delta = float(
        np.mean(
            np.abs(
                val_ranked-val_sem
            )
        )
    )

    report = {
        "documents": int(len(raw)),
        "train_samples": int(len(train["returns"])),
        "validation_samples": int(len(val["returns"])),
        "mean_retrieval_weight_difference": delta,
        "teacher_score_scale": score_scale,
        "test_used": False,
        "note": (
            "REPLUG-inspired return prediction teacher; "
            "FinAgent-inspired auxiliary return/risk heads; "
            "not original paper replication."
        ),
    }

    (args.out/"experiment_info.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    log()
    log("===== V8 VALIDATION RESULTS =====")
    log(results.to_string(index=False))

    log()
    log("===== EXPERIMENT INFO =====")
    log(json.dumps(report,indent=2))

    log()
    log("V8 PHASE 1: PASS")
    log("TEST 2022-2023 NOT USED")


if __name__ == "__main__":
    main()
