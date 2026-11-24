"""Граница эффективности Markowitz: tangency, минимальная дисперсия, кривая границы."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve(fun, n, long_only, constraints, x0=None):
    bounds = [(0.0, 1.0)] * n if long_only else None
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"оптимизация не сошлась: {res.message}")
    return res.x


def min_variance(cov, long_only=False):
    """Портфель глобальной минимальной дисперсии (сумма весов = 1)."""
    n = len(cov)
    c = cov.values
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w = _solve(lambda w: w @ c @ w, n, long_only, cons)
    return {"weights": pd.Series(w, index=cov.index), "vol": float(np.sqrt(w @ c @ w))}


def tangency(mu_excess, cov, rf, long_only=False):
    """Портфель с максимальным Sharpe. `mu_excess` - избыточные над rf годовые доходности.

    Возвращает веса (сумма 1), полную ожидаемую доходность (rf + w.mu_excess), vol и Sharpe.
    """
    n = len(cov)
    c = cov.values
    mu = mu_excess.reindex(cov.index).values
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ c @ w)

    w = _solve(neg_sharpe, n, long_only, cons)
    vol = float(np.sqrt(w @ c @ w))
    ex = float(w @ mu)
    return {"weights": pd.Series(w, index=cov.index), "ret": rf + ex,
            "vol": vol, "sharpe": ex / vol}


def efficient_frontier(mu, cov, long_only=False, n=60):
    """Эффективная граница: минимум дисперсии при заданной доходности (сумма весов = 1).

    `mu` - полные годовые ожидаемые доходности. Риск-фри актив не используется,
    поэтому состав портфеля меняется вдоль кривой.
    """
    c = cov.values
    m = mu.reindex(cov.index).values
    k = len(cov)
    w_min = min_variance(cov, long_only)["weights"].values
    lo = float(w_min @ m)
    hi = float(m.max()) if long_only else float(m.max()) * 1.4
    targets = np.linspace(lo, hi, n)

    rows, x0 = [], w_min
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=t: w @ m - t}]
        w = _solve(lambda w: w @ c @ w, k, long_only, cons, x0=x0)
        x0 = w
        rows.append({"vol": float(np.sqrt(w @ c @ w)), "ret": float(w @ m),
                     **{a: float(x) for a, x in zip(cov.index, w)}})
    return pd.DataFrame(rows)
