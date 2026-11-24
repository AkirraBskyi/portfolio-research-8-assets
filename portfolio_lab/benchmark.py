"""Проверка вне выборки и сравнение со SPY.

Веса портфеля заданы допущением, поэтому здесь проверяется не "альфа стратегии", а
(1) насколько оценки риска, сделанные на первых двух годах, держатся на последнем годе,
(2) как портфель выглядит рядом со SPY, (3) стабильна ли оценка альфы NVDA по половинам окна.
"""
import numpy as np
import pandas as pd

from . import config, factors, var


def max_drawdown(r):
    """Максимальная просадка от пика по кривой капитала (число <= 0)."""
    equity = (1 + pd.Series(r)).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def split(r, start):
    """(обучение, тест): тест начинается с даты `start` включительно."""
    r = pd.Series(r)
    cut = pd.Timestamp(start)
    return r[r.index < cut], r[r.index >= cut]


def _stats(r, rf):
    r = pd.Series(r)
    cagr = float((1 + r).prod() ** (config.TRADING_DAYS / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(config.TRADING_DAYS))
    sharpe = (float(r.mean() * config.TRADING_DAYS) - rf) / vol
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_drawdown(r)}


def compare_to_spy(returns, port_ret, rf, start=None):
    """Модельный портфель (постоянные веса) рядом со SPY. `start` ограничивает окно (например, тест)."""
    spy = returns[config.MARKET]
    port = pd.Series(port_ret)
    if start is not None:
        spy = spy[spy.index >= pd.Timestamp(start)]
        port = port[port.index >= pd.Timestamp(start)]
    return pd.DataFrame({"Model portfolio": _stats(port, rf), "SPY": _stats(spy, rf)}).T


def var_holdout(port_ret, split_date, budget):
    """VaR оценивается на обучающей части, а на тесте считаются дни с потерей больше VaR.

    Для корректной модели доля пробоев близка к 1 - уровень (5% и 1%).
    """
    train, test = split(port_ret, split_date)
    methods = {
        "Historical": var.historical(train, budget),
        "Parametric": var.parametric(train, budget),
        "Monte Carlo (bootstrap)": var.bootstrap_mc(train, budget),
    }
    losses = -test.values * budget
    rows = []
    for name, res in methods.items():
        for lvl in config.CONF_LEVELS:
            v = res[lvl]["var"]
            breaches = int((losses > v).sum())
            rows.append({"method": name, "level": lvl, "var_eur": v, "breaches": breaches,
                         "n_test": len(test), "rate": breaches / len(test), "expected": 1 - lvl})
    return pd.DataFrame(rows)


def alpha_halves(returns, ff5, end, split_date):
    """Альфа NVDA по FF5 отдельно на первой и второй части окна: насколько она устойчива."""
    cut = pd.Timestamp(split_date)
    first = factors.ff5_regression(returns, ff5, end=cut - pd.Timedelta(days=1))
    second = factors.ff5_regression(returns, ff5, end=end, start=cut)
    cols = ["alpha", "r2", "n_obs"]
    return pd.DataFrame({"train": first.loc["NVDA", cols], "test": second.loc["NVDA", cols]}).T
