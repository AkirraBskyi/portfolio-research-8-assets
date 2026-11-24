"""Регрессия Fama-French 5 факторов (множественная, все факторы сразу)."""
import pandas as pd
import statsmodels.api as sm

from . import config

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def ff5_regression(returns, ff5, end=None, start=None):
    """R_i - RF = alpha + b1*Mkt-RF + b2*SMB + b3*HML + b4*RMW + b5*CMA + e.

    alpha возвращается годовой (дневная константа * 252).
    `end` (необязательно) ограничивает окно регрессии последней датой включительно.
    """
    data = returns.join(ff5, how="inner")
    if start is not None:
        data = data.loc[start:]
    if end is not None:
        data = data.loc[:end]
    x = sm.add_constant(data[FACTORS])
    rows = {}
    for t in returns.columns:
        res = sm.OLS(data[t] - data["RF"], x).fit()
        rows[t] = {
            "alpha": res.params["const"] * config.TRADING_DAYS,
            "mkt": res.params["Mkt-RF"],
            "smb": res.params["SMB"],
            "hml": res.params["HML"],
            "rmw": res.params["RMW"],
            "cma": res.params["CMA"],
            "r2": res.rsquared,
            "n_obs": int(res.nobs),
        }
    return pd.DataFrame(rows).T
