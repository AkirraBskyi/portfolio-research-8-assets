"""VaR и CVaR тремя методами: Historical, Parametric (нормальный) и Monte Carlo (bootstrap).

Потери положительные, в евро, считаются от бюджета портфеля.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config


def portfolio_returns(returns, weights):
    """Дневная доходность портфеля с постоянными весами (без ребалансировки внутри дня)."""
    return (returns * weights.reindex(returns.columns)).sum(axis=1)


def _tail(losses, level):
    """VaR = квантиль потерь, CVaR = среднее потерь не меньше VaR."""
    v = np.quantile(losses, level)
    tail = losses[losses >= v]
    return float(v), float(tail.mean())


def _pack(loss_pct, budget):
    out = {}
    for lvl in config.CONF_LEVELS:
        v, c = _tail(np.asarray(loss_pct), lvl)
        out[lvl] = {"var": v * budget, "cvar": c * budget}
    return out


def historical(r, budget, horizon=1):
    """По реальным данным. Для horizon>1 берутся перекрывающиеся суммы доходностей за horizon дней."""
    r = np.asarray(r)
    if horizon > 1:
        r = np.convolve(r, np.ones(horizon), mode="valid")
    return _pack(-r, budget)


def parametric(r, budget, horizon=1):
    """Нормальное приближение: потери ~ N(-mu*h, sigma*sqrt(h))."""
    r = np.asarray(r)
    mu = r.mean() * horizon
    sd = r.std(ddof=1) * np.sqrt(horizon)
    out = {}
    for lvl in config.CONF_LEVELS:
        z = norm.ppf(lvl)
        var = -mu + z * sd
        cvar = -mu + sd * norm.pdf(z) / (1 - lvl)
        out[lvl] = {"var": float(var * budget), "cvar": float(cvar * budget)}
    return out


def bootstrap_mc(r, budget, horizon=1, n=config.BOOTSTRAP_N, seed=config.BOOTSTRAP_SEED):
    """Monte Carlo как bootstrap реальных дней: пересэмплирование с возвратом.

    Сохраняет настоящую (не гауссову) форму распределения. Для horizon>1 суммируются
    horizon случайных дней.
    """
    r = np.asarray(r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n, horizon))
    sims = r[idx].sum(axis=1)
    return _pack(-sims, budget)


def summary_table(r, budget, horizon=1):
    """Сводная таблица метода x (VaR/CVaR) в евро для печати и графика."""
    rows = {}
    for name, res in (("Historical", historical(r, budget, horizon)),
                      ("Monte Carlo (bootstrap)", bootstrap_mc(r, budget, horizon)),
                      ("Parametric", parametric(r, budget, horizon))):
        rows[name] = {"95% VaR": res[0.95]["var"], "95% CVaR": res[0.95]["cvar"],
                      "99% VaR": res[0.99]["var"], "99% CVaR": res[0.99]["cvar"]}
    return pd.DataFrame(rows).T
