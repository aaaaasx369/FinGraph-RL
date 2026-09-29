from __future__ import annotations
import numpy as np


def soft_confidence_gate(
    stock_weights: np.ndarray,
    cash_weight: float,
    rag_raw: np.ndarray,
    cfg: dict,
):
    """
    rag_raw columns: impact [-1,1], risk [0,1], confidence [0,1].

    g_i = 1 + alpha * C_i * I_i - beta * C_i * R_i
    when confidence >= threshold; otherwise g_i = 1.
    """
    impact = rag_raw[:, 0].astype(float)
    risk = rag_raw[:, 1].astype(float)
    conf = rag_raw[:, 2].astype(float)

    active = conf >= float(cfg["confidence_threshold"])
    g = np.ones_like(stock_weights, dtype=float)
    g_active = (
        1.0
        + float(cfg["alpha_impact"]) * conf * impact
        - float(cfg["beta_risk"]) * conf * risk
    )
    g_active = np.clip(
        g_active,
        float(cfg["multiplier_min"]),
        float(cfg["multiplier_max"]),
    )
    g[active] = g_active[active]

    stocks = np.maximum(stock_weights, 0.0) * g
    cash = max(float(cash_weight), 0.0)

    total = stocks.sum() + cash
    if total <= 0:
        return stock_weights.copy(), cash_weight
    return stocks / total, cash / total


def apply_negative_veto(
    stock_weights: np.ndarray,
    cash_weight: float,
    rag_raw: np.ndarray,
    cfg: dict,
):
    """
    Negative veto:
      V_i = C_i * R_i * max(-I_i, 0)

    If V_i >= veto_threshold, cap that stock at veto_max_weight.
    Removed weight is moved to CASH so the cap cannot be undone by renormalization.
    """
    impact = rag_raw[:, 0].astype(float)
    risk = rag_raw[:, 1].astype(float)
    conf = rag_raw[:, 2].astype(float)

    veto_score = conf * risk * np.maximum(-impact, 0.0)
    mask = veto_score >= float(cfg["veto_threshold"])
    cap = float(cfg["veto_max_weight"])

    stocks = stock_weights.astype(float).copy()
    cash = float(cash_weight)

    removed = 0.0
    for i in np.where(mask)[0]:
        if stocks[i] > cap:
            removed += stocks[i] - cap
            stocks[i] = cap

    cash += removed

    # Optional bullish floor. Disabled by default because strong fundamentals
    # do not imply positive short-horizon returns.
    if bool(cfg.get("positive_floor_enabled", False)):
        bullish_score = conf * np.maximum(impact, 0.0)
        floor_mask = bullish_score >= float(cfg["positive_floor_threshold"])
        floor = float(cfg["positive_min_weight"])
        for i in np.where(floor_mask)[0]:
            if stocks[i] < floor and cash > 0:
                add = min(floor - stocks[i], cash)
                stocks[i] += add
                cash -= add

    total = stocks.sum() + cash
    if total <= 0:
        return stock_weights.copy(), cash_weight, veto_score
    return stocks / total, cash / total, veto_score
