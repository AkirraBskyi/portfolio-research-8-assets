"""Базовые метрики риска: волатильность, корреляции, беты, разложение риска."""
import numpy as np
import pandas as pd

from . import config


def corr_matrix(returns):
    return returns.corr()


def risk_table(returns, market=config.MARKET):
    """По каждому активу: corr к рынку, годовая vol, beta, R2, системная и идиосинкратическая vol."""
    vol = returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS)
    mkt = returns[market]
    corr = returns.corrwith(mkt)
    beta = returns.apply(lambda s: s.cov(mkt)) / mkt.var()
    r2 = corr ** 2
    return pd.DataFrame({
        "corr": corr,
        "vol": vol,
        "beta": beta,
        "r2": r2,
        "sys_vol": corr.abs() * vol,           # sqrt(R2) * vol
        "idio_vol": vol * np.sqrt(1 - r2),
    })
