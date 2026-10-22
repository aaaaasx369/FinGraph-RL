#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17 End-to-End PPO smoke / preliminary ablation.

Four arms:
  MARKET
  MARKET_SEC
  MARKET_KG
  MARKET_SEC_KG

Train: 2018-2020
Val:   2021
Test:  2022-2023 SEALED

Market:
  Actual local daily adjusted-close/close prices.

SEC:
  Point-in-time SEC text-derived proxy features.
  NOT B8.1 verified semantic classifications.

KG:
  Strictly past observed temporal 1-hop
  structural features only.

RL:
  Standalone SB3 PPO, long-only softmax weights,
  proportional turnover cost, log-portfolio reward.

IMPORTANT: This is NOT a byte-for-byte
reproduction of the V16 trading environment.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

ARMS = {
    "MARKET": (False, False),
    "MARKET_SEC": (True, False),
    "MARKET_KG": (False, True),
    "MARKET_SEC_KG": (True, True),
}

SEC_WORDS = {
    "regulatory": r"\b(regulat|investigat|litigat|lawsuit|legal)\w*",
    "supply": r"\b(supplier|supply|shortage|manufactur)\w*",
    "competition": r"\b(competit|marketplace|antitrust)\w*",
    "financial": r"\b(revenue|cash|lease|debt|financ)\w*",
    "cyber": r"\b(cyber|security|privacy|breach)\w*",
    "negative_words": r"\b(adverse|loss|decline|decrease|risk)\w*",
}

START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2020-12-31")
VAL_START = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")


def log(*x):
    print(*x, flush=True)


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".gz"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table: {path}")


def pick(df, candidates):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def discover_market(root):
    patterns = [
        "**/*price*.parquet",
        "**/*stock*.parquet",
        "**/*market*.parquet",
        "**/*price*.csv",
        "**/*stock*.csv",
        "**/*market*.csv",
        "**/*ohlcv*.csv",
        "**/*ohlcv*.parquet",
    ]

    found = []
    for pattern in patterns:
        found.extend(root.glob(pattern))

    found = sorted(
        set(found),
        key=lambda p: (
            "market" not in p.name.lower(),
            "price" not in p.name.lower(),
            len(str(p)),
        )
    )

    found = [
        p for p in found
        if (
            p.is_file()
            and p.stat().st_size < 350_000_000
            and "audit" not in p.name.lower()
            and "signals" not in p.name.lower()
        )
    ]

    return found[:100]


def standardize_market(raw):
    raw = raw.copy()

    date_col = pick(
        raw,
        ["date", "datetime", "timestamp", "time"]
    )
    tic_col = pick(
        raw,
        ["tic", "ticker", "symbol", "asset"]
    )
    close_col = pick(
        raw,
        [
            "adj_close", "adjusted_close",
            "adj close", "close", "Close",
        ]
    )

    if (
        date_col is None
        or tic_col is None
        or close_col is None
    ):
        raise ValueError(
            "Need long-format columns: "
            "date, ticker/tic, close/adj_close"
        )

    df = pd.DataFrame({
        "date": pd.to_datetime(
            raw[date_col], errors="coerce"
        ).dt.tz_localize(None).dt.normalize(),
        "ticker": raw[tic_col].astype(str).str.upper(),
        "close": pd.to_numeric(
            raw[close_col], errors="coerce"
        ),
    })

    df = df[
        df["ticker"].isin(TICKERS)
        & df["date"].notna()
        & df["close"].notna()
        & (df["close"] > 0)
    ].copy()

    df = df.sort_values(
        ["date", "ticker"]
    ).drop_duplicates(
        ["date", "ticker"], keep="last"
    )

    return df


def get_market(path, root):
    if path:
        log("[MARKET FILE]", path)
        return standardize_market(
            read_table(path)
        ), str(path)

    candidates = discover_market(root)
    failures = []

    for candidate in candidates:
        try:
            df = standardize_market(
                read_table(candidate)
            )

            if (
                df["ticker"].nunique() == 5
                and df["date"].min() <= pd.Timestamp("2018-03-01")
                and df["date"].max() >= VAL_END
            ):
                log("[AUTO MARKET]", candidate)
                return df, str(candidate)

        except Exception as exc:
            failures.append(
                (str(candidate), str(exc)[:100])
            )

    log("\nCANDIDATE MARKET FILES:")
    for p in candidates[:30]:
        log(" ", p)

    raise RuntimeError(
        "Cannot locate 5-ticker market data. "
        "Run again with --market /path/to/your/"
        "market_prices.csv (or .parquet). "
        "Expected columns: date, tic/ticker, close."
    )


def build_market(market):
    prices = market.pivot(
        index="date",
        columns="ticker",
        values="close"
    ).sort_index()

    prices = prices.reindex(
        columns=TICKERS
    )

    prices = prices.loc[
        (prices.index >= START)
        & (prices.index <= VAL_END)
    ]

    missing = prices.isna().sum()
    if missing.any():
        log(
            "[MARKET MISSING]",
            missing.to_dict()
        )

    # Only common trading days; do not invent prices.
    prices = prices.dropna(how="any")

    if len(prices) < 500:
        raise ValueError(
            f"Insufficient common trading days: {len(prices)}"
        )

    if prices.index.min() > pd.Timestamp("2018-03-01"):
        raise ValueError("Market history starts too late")

    if prices.index.max() < pd.Timestamp("2021-12-20"):
        raise ValueError("Validation market history incomplete")

    returns = prices.pct_change()

    # Past-only technical features, one trading day
    # lag. No use of future realized returns.
    parts = []
    for t in TICKERS:
        p = prices[t]
        r = returns[t]

        columns = pd.DataFrame({
            f"{t}_r1": r.shift(1),
            f"{t}_r5": (
                p.pct_change(5).shift(1)
            ),
            f"{t}_r20": (
                p.pct_change(20).shift(1)
            ),
            f"{t}_vol20": (
                r.rolling(20).std().shift(1)
            ),
        }, index=prices.index)

        parts.append(columns)

    features = pd.concat(
        parts, axis=1
    ).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    # t observation -> t to t+1 actual return
    next_ret = (
        prices.shift(-1) / prices - 1.0
    )

    return prices, features, next_ret



def prepare_sec(sec_path):
    """
    Reuse the existing V17 SEC PIT loader
    instead of assuming raw Parquet columns.
    """
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import run_v17_unified_llm_rag as v17

    raw = v17.load_sec(sec_path)

    print(
        "[V17 SEC NORMALIZED COLUMNS]",
        raw.columns.tolist(),
        flush=True,
    )

    required = {
        "ticker", "available_from", "excerpt"
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Original V17 loader missing columns: "
            + str(sorted(missing))
        )

    sec = pd.DataFrame({
        "ticker": (
            raw["ticker"]
            .astype(str)
            .str.upper()
        ),
        "available": pd.to_datetime(
            raw["available_from"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None).dt.normalize(),
        "excerpt": (
            raw["excerpt"]
            .fillna("")
            .astype(str)
        ),
    })

    sec = sec[
        sec["ticker"].isin(TICKERS)
        & sec["available"].notna()
        & (sec["available"] < TEST_START)
    ].sort_values(
        ["ticker", "available"]
    )

    print(
        "[SEC ROWS]", len(sec),
        flush=True,
    )

    return sec


def build_sec_features(sec, days):
    """
    As-of latest SEC text proxy.

    Text is historical original excerpt,
    not verified LLM claim or trading alpha.
    Strict available_from < decision date.
    """

    output = pd.DataFrame(
        0.0, index=days,
        columns=[
            f"{t}_sec_{name}"
            for t in TICKERS
            for name in [
                "present", "age", *SEC_WORDS
            ]
        ]
    )

    for t in TICKERS:
        sub = sec[
            sec["ticker"] == t
        ].copy()

        if sub.empty:
            continue

        # Multiple text excerpts from a same day:
        # deterministic concatenation, capped size.
        grouped = (
            sub.groupby("available")["excerpt"]
            .apply(
                lambda x: " ".join(
                    x.astype(str).head(10)
                )[:12000]
            )
            .reset_index()
            .sort_values("available")
        )

        source_dates = (
            grouped["available"].to_numpy(
                dtype="datetime64[ns]"
            )
        )
        source_texts = grouped[
            "excerpt"
        ].to_list()

        for day in days:
            # Strictly before the decision date.
            pos = np.searchsorted(
                source_dates,
                np.datetime64(day),
                side="left",
            ) - 1

            if pos < 0:
                continue

            text = source_texts[pos]
            age = (
                pd.Timestamp(day)
                - pd.Timestamp(source_dates[pos])
            ).days

            output.loc[
                day, f"{t}_sec_present"
            ] = 1.0

            output.loc[
                day, f"{t}_sec_age"
            ] = min(age, 365) / 365.0

            for name, pattern in SEC_WORDS.items():
                output.loc[
                    day, f"{t}_sec_{name}"
                ] = float(
                    min(
                        len(re.findall(
                            pattern,
                            text,
                            flags=re.I
                        )),
                        20,
                    )
                ) / 20.0

    return output


def build_kg_features(kg_path, days):
    """
    KG features via existing V17 company-linked
    loader/retriever; no opaque relation semantics.
    Strict first_observed < decision day.
    """

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import run_v17_unified_llm_rag as v17

    kg = v17.load_kg(kg_path)

    columns = [
        f"{t}_kg_{name}"
        for t in TICKERS
        for name in [
            "present", "edges", "relations",
            "in", "out", "age",
        ]
    ]

    out = pd.DataFrame(
        0.0,
        index=days,
        columns=columns,
    )

    total = len(days) * len(TICKERS)
    n = 0

    for day in days:
        for t in TICKERS:
            hits = v17.kg_retrieve(
                kg, t,
                pd.Timestamp(day),
                top_k=12,
            )

            n += 1

            if not hits:
                continue

            # Check source timestamps explicitly.
            for h in hits:
                assert (
                    pd.Timestamp(
                        h["first_observed"]
                    )
                    < pd.Timestamp(day)
                ), "KG temporal leakage"

            relations = {
                str(h["relation"])
                for h in hits
            }

            directions = [
                str(
                    h.get(
                        "direction_from_company",
                        ""
                    )
                ).lower()
                for h in hits
            ]

            ages = [
                (
                    pd.Timestamp(day)
                    - pd.Timestamp(
                        h["first_observed"]
                    )
                ).days
                for h in hits
            ]

            prefix = f"{t}_kg_"

            out.loc[
                day, prefix + "present"
            ] = 1.0

            out.loc[
                day, prefix + "edges"
            ] = min(len(hits), 12) / 12.0

            out.loc[
                day, prefix + "relations"
            ] = min(
                len(relations), 19
            ) / 19.0

            out.loc[
                day, prefix + "in"
            ] = sum(
                d in ("in", "incoming")
                for d in directions
            ) / 12.0

            out.loc[
                day, prefix + "out"
            ] = sum(
                d in ("out", "outgoing")
                for d in directions
            ) / 12.0

            out.loc[
                day, prefix + "age"
            ] = min(
                min(ages), 365
            ) / 365.0

        if (n // len(TICKERS)) % 100 == 0:
            log(
                "[KG PROGRESS]",
                n, "/", total
            )

    return out


class PortfolioEnv(gym.Env):
    """
    A transparent, independent PPO environment.

    At day t:
      state contains information available before
      day t's decision.
      agent allocates weights.
      portfolio earns close(t)->close(t+1).
      transaction cost charged on allocation change.
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        obs,
        next_returns,
        cost=0.001,
        reward_scale=100.0,
    ):
        super().__init__()

        self.obs_data = np.asarray(
            obs, dtype=np.float32
        )
        self.next_returns = np.asarray(
            next_returns, dtype=np.float64
        )

        if len(self.obs_data) != len(
            self.next_returns
        ):
            raise ValueError(
                "Observation/return length mismatch"
            )

        if len(self.obs_data) < 5:
            raise ValueError(
                "Not enough environment steps"
            )

        if not np.isfinite(
            self.obs_data
        ).all():
            raise ValueError(
                "Nonfinite observation"
            )

        if not np.isfinite(
            self.next_returns
        ).all():
            raise ValueError(
                "Nonfinite market returns"
            )

        self.cost = float(cost)
        self.reward_scale = float(
            reward_scale
        )

        self.action_space = spaces.Box(
            low=-5,
            high=5,
            shape=(len(TICKERS),),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.obs_data.shape[1],
            ),
            dtype=np.float32,
        )

        self.i = 0
        self.wealth = 1.0
        self.prev_weights = np.full(
            len(TICKERS),
            1 / len(TICKERS),
            dtype=np.float64,
        )
        self.history = []

    def _get_obs(self):
        return self.obs_data[self.i]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.i = 0
        self.wealth = 1.0

        self.prev_weights[:] = (
            1 / len(TICKERS)
        )
        self.history = []

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(
            action, dtype=np.float64
        )

        action = np.clip(
            action, -5, 5
        )

        z = np.exp(
            action - np.max(action)
        )
        weights = z / z.sum()

        turnover = float(
            np.abs(
                weights - self.prev_weights
            ).sum()
        )

        gross = 1.0 + float(
            np.dot(
                weights,
                self.next_returns[self.i],
            )
        )

        net = (
            gross - self.cost * turnover
        )
        net = max(net, 1e-8)

        reward = (
            np.log(net) * self.reward_scale
        )

        self.wealth *= net
        self.prev_weights = weights

        self.history.append({
            "wealth": float(self.wealth),
            "daily_return": net - 1.0,
            "turnover": turnover,
        })

        self.i += 1

        done = (
            self.i >= len(self.obs_data)
        )

        if done:
            return (
                np.zeros(
                    self.observation_space.shape,
                    dtype=np.float32,
                ),
                float(reward),
                True,
                False,
                {},
            )

        return (
            self._get_obs(),
            float(reward),
            False,
            False,
            {},
        )


def evaluate(model, env, dates):
    obs, _ = env.reset()

    done = False

    while not done:
        action, _ = model.predict(
            obs, deterministic=True
        )

        obs, _, done, _, _ = env.step(
            action
        )

    hist = pd.DataFrame(
        env.history,
        index=dates,
    )

    r = hist[
        "daily_return"
    ].to_numpy(dtype=float)

    wealth = hist[
        "wealth"
    ].to_numpy(dtype=float)

    sharpe = (
        float(
            np.mean(r)
            / (np.std(r, ddof=1) + 1e-12)
            * np.sqrt(252)
        )
        if len(r) > 1 else 0.0
    )

    peak = np.maximum.accumulate(
        np.r_[1.0, wealth]
    )[1:]

    mdd = float(
        np.min(
            wealth / peak - 1
        )
    )

    return {
        "final_multiple": float(
            wealth[-1]
        ),
        "sharpe": sharpe,
        "mdd": mdd,
        "mean_turnover": float(
            hist["turnover"].mean()
        ),
    }, hist


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--market")
    p.add_argument(
        "--root", default="."
    )
    p.add_argument(
        "--sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        )
    )
    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        )
    )
    p.add_argument(
        "--timesteps",
        type=int, default=2048
    )
    p.add_argument(
        "--seed", type=int,
        default=42
    )
    p.add_argument(
        "--cost", type=float,
        default=0.001
    )
    p.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "v17_e2e_smoke"
        )
    )

    args = p.parse_args()

    if args.timesteps < 128:
        raise ValueError(
            "Use at least 128 PPO timesteps"
        )

    if not (
        0.0 <= args.cost < 0.1
    ):
        raise ValueError(
            "Invalid transaction cost"
        )

    root = Path(
        args.root
    ).resolve()

    out = Path(
        args.out
    )

    out.mkdir(
        parents=True, exist_ok=True
    )

    set_random_seed(
        args.seed
    )

    market, market_source = get_market(
        args.market, root
    )

    prices, market_features, next_ret = (
        build_market(market)
    )

    days = prices.index

    # No test period access.
    assert days.max() < TEST_START

    log(
        "[MARKET DAYS]", len(days),
        days.min().date(),
        days.max().date()
    )

    sec_raw = prepare_sec(
        args.sec
    )

    sec_features = build_sec_features(
        sec_raw, days
    )

    log(
        "[SEC FEATURE SHAPE]",
        sec_features.shape
    )

    kg_features = build_kg_features(
        args.kg, days
    )

    log(
        "[KG FEATURE SHAPE]",
        kg_features.shape
    )

    # Fit normalization on TRAIN ONLY.
    train_mask = (
        (days >= START)
        & (days <= TRAIN_END)
        & (days < next_ret.index.max())
    )

    val_mask = (
        (days >= VAL_START)
        & (days < VAL_END)
    )

    valid_return = (
        next_ret.notna().all(axis=1)
    )

    train_mask &= valid_return.to_numpy()
    val_mask &= valid_return.to_numpy()

    # The return target for day t belongs to
    # the following trading day. Never cross
    # train/validation boundaries.
    following_dates = pd.Series(
        days, index=days
    ).shift(-1)

    train_mask &= (
        following_dates <= TRAIN_END
    ).fillna(False).to_numpy()

    val_mask &= (
        following_dates <= VAL_END
    ).fillna(False).to_numpy()

    # Preserve information availability.
    # Standardization cannot use validation stats.
    feature_groups = {
        "market": market_features,
        "sec": sec_features,
        "kg": kg_features,
    }

    normalized = {}

    for key, data in feature_groups.items():
        values = data.to_numpy(
            dtype=np.float64
        )

        mean = values[
            train_mask
        ].mean(axis=0)

        std = values[
            train_mask
        ].std(axis=0)

        std = np.where(
            std < 1e-6, 1.0, std
        )

        values = np.clip(
            (values - mean) / std,
            -10.0, 10.0,
        )

        normalized[key] = pd.DataFrame(
            values.astype(
                np.float32
            ),
            index=days,
            columns=data.columns,
        )

    dates_train = days[
        train_mask
    ]

    dates_val = days[
        val_mask
    ]

    if (
        len(dates_train) < 400
        or len(dates_val) < 150
    ):
        raise ValueError(
            "Insufficient train or validation days"
        )

    log(
        "[TRAIN]", len(dates_train),
        dates_train.min().date(),
        dates_train.max().date()
    )

    log(
        "[VAL]", len(dates_val),
        dates_val.min().date(),
        dates_val.max().date()
    )

    result_rows = []

    for arm, (use_sec, use_kg) in ARMS.items():
        frames = [
            normalized["market"]
        ]

        if use_sec:
            frames.append(
                normalized["sec"]
            )

        if use_kg:
            frames.append(
                normalized["kg"]
            )

        fused = pd.concat(
            frames, axis=1
        )

        train_obs = fused.loc[
            dates_train
        ].to_numpy(
            dtype=np.float32
        )

        val_obs = fused.loc[
            dates_val
        ].to_numpy(
            dtype=np.float32
        )

        train_returns = next_ret.loc[
            dates_train, TICKERS
        ].to_numpy(
            dtype=np.float64
        )

        val_returns = next_ret.loc[
            dates_val, TICKERS
        ].to_numpy(
            dtype=np.float64
        )

        train_env = PortfolioEnv(
            train_obs,
            train_returns,
            cost=args.cost,
        )

        val_env = PortfolioEnv(
            val_obs,
            val_returns,
            cost=args.cost,
        )

        log()
        log("=" * 70)
        log("[TRAIN]", arm)
        log("[OBS DIM]", fused.shape[1])
        log("=" * 70)

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            seed=args.seed,
            verbose=0,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            ent_coef=0.0,
            device="cpu",
        )

        model.learn(
            total_timesteps=args.timesteps
        )

        metrics, history = evaluate(
            model,
            val_env,
            dates_val,
        )

        metrics.update({
            "arm": arm,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "observation_dim": fused.shape[1],
            "train_days": len(dates_train),
            "val_days": len(dates_val),
        })

        result_rows.append(
            metrics
        )

        model.save(
            str(out / f"{arm}_ppo")
        )

        history.to_csv(
            out / f"{arm}_equity.csv",
            index_label="date",
        )

        pd.DataFrame(
            result_rows
        ).to_csv(
            out / "summary.csv",
            index=False,
        )

        log(
            "[RESULT]",
            arm,
            "multiple=",
            round(
                metrics["final_multiple"], 4
            ),
            "sharpe=",
            round(
                metrics["sharpe"], 4
            ),
            "mdd=",
            round(
                metrics["mdd"], 4
            )
        )

    manifest = {
        "experiment": "V17_E2E_SMOKE",
        "status": "COMPLETED",
        "market_source": market_source,
        "sec_source": args.sec,
        "kg_source": args.kg,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "cost": args.cost,
        "tickers": TICKERS,
        "train": "2018-2020",
        "validation": "2021",
        "test_accessed": False,
        "sec_method": (
            "as-of SEC excerpt keyword proxy; "
            "NOT B8.1 verified event semantics"
        ),
        "kg_method": (
            "as-of 1-hop structural features; "
            "opaque relations not interpreted"
        ),
        "rl_method": (
            "independent standalone PPO; "
            "not identical to V16 environment"
        ),
        "limitations": [
            "Development smoke test only.",
            "SEC features are not verified B8.1 outputs.",
            "KG first-observed not certified public time.",
            "Market field adjustment must be checked.",
            "Price, execution, costs, and timing "
            "require validation before alpha claims.",
            "No sealed test data used.",
        ],
    }

    (
        out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("========== V17-E2E SUMMARY ==========")

    result = pd.DataFrame(
        result_rows
    )

    log(
        result[
            [
                "arm",
                "final_multiple",
                "sharpe",
                "mdd",
                "mean_turnover",
                "observation_dim",
            ]
        ].to_string(
            index=False
        )
    )

    log(
        "\nV17-E2E SMOKE: PASS"
    )
    log("[OUTPUT]", out)


if __name__ == "__main__":
    main()
