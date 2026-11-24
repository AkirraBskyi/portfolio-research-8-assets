"""Сценарные стресс-тесты: мгновенный шок цены каждого актива.

Шоки заданы заново, по мотивам исторических эпизодов, и НЕ подгонялись под ранее
опубликованные результаты. Это допущения, а не прогноз: stress test не говорит о вероятности сценария.
"""
import pandas as pd

from . import config

_ORDER = config.TICKERS  # SPY QQQ NVDA MSFT TLT SHY GLD FXF

SCENARIOS = {
    "equity_crash": pd.Series([-0.34, -0.28, -0.38, -0.28, +0.08, +0.01, -0.03, +0.01], index=_ORDER),
    "rate_shock":   pd.Series([-0.19, -0.33, -0.50, -0.28, -0.31, -0.04, -0.01, 0.00], index=_ORDER),
    "ai_bubble":    pd.Series([-0.15, -0.25, -0.60, -0.30, +0.05, +0.01, +0.05, +0.02], index=_ORDER),
    "stagflation":  pd.Series([-0.25, -0.30, -0.45, -0.30, -0.20, -0.03, +0.10, 0.00], index=_ORDER),
}

LABELS = {
    "equity_crash": "Equity crash (COVID-style)",
    "rate_shock": "Rate shock (2022-style)",
    "ai_bubble": "AI-bubble burst",
    "stagflation": "Combined (stagflation-style)",
}


def run_stress(weights, shocks, budget):
    """Потеря портфеля = сумма(вес * шок). Возвращает долю и сумму в евро (отрицательные при убытке)."""
    pct = float((weights.reindex(shocks.index) * shocks).sum())
    return {"pct": pct, "eur": pct * budget}


def run_all(weights, budget):
    rows = {LABELS[k]: run_stress(weights, s, budget) for k, s in SCENARIOS.items()}
    return pd.DataFrame(rows).T
