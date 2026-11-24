"""Look-through: реальная (эффективная) доля NVDA и MSFT с учётом того, что их держат SPY и QQQ."""
import pandas as pd

from . import config


def effective_exposure(weights, spy_share=None, qqq_share=None):
    """Для каждой из двух акций: прямая доля + доля через SPY + доля через QQQ.

    via_spy = w_SPY * (доля акции в S&P 500), via_qqq = w_QQQ * (доля акции в QQQ).
    """
    spy_share = config.SP500_SHARE if spy_share is None else spy_share
    qqq_share = config.QQQ_SHARE if qqq_share is None else qqq_share
    rows = {}
    for stock in ("NVDA", "MSFT"):
        direct = float(weights[stock])
        via_spy = float(weights["SPY"]) * spy_share[stock]
        via_qqq = float(weights["QQQ"]) * qqq_share[stock]
        rows[stock] = {"direct": direct, "via_spy": via_spy, "via_qqq": via_qqq,
                       "total": direct + via_spy + via_qqq}
    return pd.DataFrame(rows).T
