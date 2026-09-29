from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .env import PortfolioEnv


def performance_metrics(history: pd.DataFrame, periods_per_year: int = 252) -> dict:
    if history.empty:
        return {}

    r = history["net_return"].astype(float).to_numpy()
    equity = history["portfolio_value"].astype(float).to_numpy()

    total_return = equity[-1] / equity[0] - 1.0 if len(equity) > 1 else 0.0
    n = len(r)
    ann_return = (np.prod(1.0 + r) ** (periods_per_year / max(n, 1))) - 1.0

    vol = np.std(r, ddof=1) * np.sqrt(periods_per_year) if n > 1 else 0.0
    mean_ann = np.mean(r) * periods_per_year
    sharpe = mean_ann / vol if vol > 1e-12 else np.nan

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(np.min(dd))

    downside = r[r < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = mean_ann / down_vol if down_vol and down_vol > 1e-12 else np.nan

    # Historical 5% daily CVaR.
    if len(r) >= 20:
        q = np.quantile(r, 0.05)
        cvar_5 = float(r[r <= q].mean())
    else:
        cvar_5 = np.nan

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(vol),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": max_dd,
        "daily_cvar_5pct": cvar_5,
        "avg_turnover": float(history["turnover"].mean()),
        "total_transaction_cost_return": float(history["transaction_cost_return"].sum()),
        "final_value": float(equity[-1]),
    }


def evaluate_model(model, data, cfg: dict, mode: str):
    env = PortfolioEnv(data, cfg, mode=mode)
    obs, _ = env.reset()

    done = False
    weights = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        weights.append(np.asarray(info["target_weights"], dtype=float))
        done = terminated or truncated

    hist = pd.DataFrame(env.history)
    W = np.vstack(weights) if weights else np.empty((0, env.n_assets))
    metrics = performance_metrics(hist)
    return hist, W, metrics


def save_evaluation(hist: pd.DataFrame, weights: np.ndarray, metrics: dict, tickers, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_to_save = hist.drop(columns=["raw_target_weights", "target_weights", "veto_score"], errors="ignore")
    hist_to_save.to_csv(out_dir / "backtest.csv", index=False)

    names = list(tickers) + ["CASH"]
    pd.DataFrame(weights, columns=names).to_csv(out_dir / "weights.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    if not hist.empty:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(hist["date"]), hist["portfolio_value"])
        plt.xlabel("Date")
        plt.ylabel("Portfolio value")
        plt.title("Out-of-sample equity curve")
        plt.tight_layout()
        fig.savefig(out_dir / "equity_curve.png", dpi=160)
        plt.close(fig)
