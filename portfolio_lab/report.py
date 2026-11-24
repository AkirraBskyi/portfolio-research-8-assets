"""Полный прогон пайплайна и сверка с опубликованными числами."""
import pandas as pd

from . import (benchmark, black_litterman as bl, capm, config, factors, frontier, lookthrough,
               risk, stress, var)


def run_pipeline(returns, ff5, ff_end=None):
    """Считает все блоки исследования и возвращает их результаты одним словом dict."""
    cov = returns.cov() * config.TRADING_DAYS
    rt = risk.risk_table(returns)
    w = bl.market_weights()
    delta = bl.calibrate_delta(w, rt["beta"], cov, config.ERP)
    pi = bl.implied_excess_returns(cov, w, delta)
    pr = var.portfolio_returns(returns, w)
    return {
        "cov": cov,
        "risk": rt,
        "ff5": factors.ff5_regression(returns, ff5, end=ff_end),
        "capm": capm.capm_returns(rt["beta"], config.RF, config.ERP),
        "weights": w,
        "delta": delta,
        "pi": pi,
        "tangency": frontier.tangency(pi, cov, config.RF, long_only=True),
        "min_var_short": frontier.min_variance(cov, long_only=False),
        "min_var_long": frontier.min_variance(cov, long_only=True),
        "var_1d": var.summary_table(pr, config.BUDGET_EUR, 1),
        "var_21d": var.summary_table(pr, config.BUDGET_EUR, config.MONTH_DAYS),
        "stress": stress.run_all(w, config.BUDGET_EUR),
        "lookthrough": lookthrough.effective_exposure(w),
        "spy_full": benchmark.compare_to_spy(returns, pr, config.RF),
        "spy_test": benchmark.compare_to_spy(returns, pr, config.RF, start=config.HOLDOUT_START),
        "holdout": benchmark.var_holdout(pr, config.HOLDOUT_START, config.BUDGET_EUR),
        "alpha_halves": benchmark.alpha_halves(returns, ff5, ff_end or returns.index.max(),
                                               config.HOLDOUT_START),
    }


def compare_with_published(res):
    """Таблица: метрика, опубликовано, воспроизведено, допуск, совпало ли."""
    rt, ff, tg = res["risk"], res["ff5"], res["tangency"]
    v1 = res["var_1d"]
    rows = [
        ("NVDA beta", 2.044, rt.loc["NVDA", "beta"], 0.002),
        ("NVDA volatility, %", 46.59, rt.loc["NVDA", "vol"] * 100, 0.05),
        ("NVDA idiosyncratic vol, %", 34.6, rt.loc["NVDA", "idio_vol"] * 100, 0.1),
        ("NVDA FF5 alpha, % p.a.", 31.97, ff.loc["NVDA", "alpha"] * 100, 0.3),
        ("NVDA FF5 R2, %", 56.5, ff.loc["NVDA", "r2"] * 100, 0.3),
        ("delta (risk aversion)", 2.345, res["delta"], 0.01),
        ("CAPM NVDA, %", 12.98, res["capm"]["NVDA"] * 100, 0.05),
        ("BL implied return NVDA, %", 13.34, (res["pi"]["NVDA"] + config.RF) * 100, 0.05),
        ("BL implied return SPY, %", 7.95, (res["pi"]["SPY"] + config.RF) * 100, 0.05),
        ("Tangency E[R], %", 7.16, tg["ret"] * 100, 0.05),
        ("Tangency volatility, %", 11.25, tg["vol"] * 100, 0.05),
        ("Tangency Sharpe", 0.264, tg["sharpe"], 0.002),
        ("Historical 95% VaR, EUR", 21.33, v1.loc["Historical", "95% VaR"], 0.1),
        ("Historical 99% CVaR, EUR", 46.19, v1.loc["Historical", "99% CVaR"], 0.1),
        ("Parametric 99% CVaR, EUR", 36.40, v1.loc["Parametric", "99% CVaR"], 0.15),
        ("Monte Carlo 99% CVaR, EUR", 46.17, v1.loc["Monte Carlo (bootstrap)", "99% CVaR"], 0.7),
    ]
    df = pd.DataFrame(rows, columns=["metric", "published", "reproduced", "tolerance"])
    df["ok"] = (df["published"] - df["reproduced"]).abs() <= df["tolerance"]
    return df


PUBLISHED_STRESS = {  # справочно: в исследовании были другие, не сохранившиеся шоки
    "Equity crash (COVID-style)": -14.64,
    "Rate shock (2022-style)": -11.18,
    "AI-bubble burst": -8.98,
    "Combined (stagflation-style)": -15.62,
}
