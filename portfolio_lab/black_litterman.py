"""Black-Litterman: обратная оптимизация (reverse optimization) без личных views."""
import numpy as np
import pandas as pd

from . import config


def market_weights():
    """Двухуровневые веса: макро-аллокация по классам x доли внутри акционного блока.

    Наивные веса по капитализации внутри узкой вселенной дают NVDA ~50% и ~90% годовых:
    единицы несопоставимы, а SPY/QQQ уже содержат NVDA и MSFT. Поэтому веса задаются явно.
    """
    eq = config.ALLOC_CLASS["equities"]
    nvda = eq * config.SP500_SHARE["NVDA"]
    msft = eq * config.SP500_SHARE["MSFT"]
    rest = eq - nvda - msft
    w = {
        "SPY": rest * config.SPY_SHARE_OF_REST,
        "QQQ": rest * (1 - config.SPY_SHARE_OF_REST),
        "NVDA": nvda,
        "MSFT": msft,
        "TLT": config.ALLOC_CLASS["TLT"],
        "SHY": config.ALLOC_CLASS["SHY"],
        "GLD": config.ALLOC_CLASS["GLD"],
        "FXF": config.ALLOC_CLASS["FXF"],
    }
    return pd.Series(w)[config.TICKERS]


def calibrate_delta(w, betas, cov, erp):
    """delta = (w . beta * ERP) / (w' cov w): премия и риск считаются на одном и том же портфеле."""
    w = w.reindex(cov.index)
    port_var = float(w.values @ cov.values @ w.values)
    return float((w * betas.reindex(w.index)).sum() * erp / port_var)


def implied_excess_returns(cov, w, delta):
    """pi = delta * cov @ w (годовые, избыточные над rf)."""
    w = w.reindex(cov.index)
    return pd.Series(delta * cov.values @ w.values, index=cov.index)


def implied_returns(cov, w, delta, rf):
    return rf + implied_excess_returns(cov, w, delta)
