import numpy as np
import pandas as pd
from portfolio_lab import benchmark, config, factors


def test_max_drawdown_hand():
    r = pd.Series([0.1, -0.2, 0.1])
    assert abs(benchmark.max_drawdown(r) - (-0.2)) < 1e-12
    assert benchmark.max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_compare_to_spy_structure(returns, port_ret):
    t = benchmark.compare_to_spy(returns, port_ret, config.RF)
    assert list(t.index) == ["Model portfolio", "SPY"]
    assert {"cagr", "vol", "sharpe", "max_drawdown"} <= set(t.columns)
    assert (t["max_drawdown"] <= 0).all()
    assert abs(t.loc["SPY", "vol"] - 0.1524) < 0.0005       # волатильность SPY из исследования
    # портфель с 30% облигаций спокойнее SPY
    assert t.loc["Model portfolio", "vol"] < t.loc["SPY", "vol"]


def test_split_sizes(port_ret):
    train, test = benchmark.split(port_ret, config.HOLDOUT_START)
    assert len(train) + len(test) == len(port_ret)
    assert train.index.max() < pd.Timestamp(config.HOLDOUT_START) <= test.index.min()
    assert 240 <= len(test) <= 260                              # около года


def test_var_holdout_logic():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=600)
    r = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    t = benchmark.var_holdout(r, str(idx[400].date()), budget=1000.0)
    assert {"method", "level", "var_eur", "breaches", "n_test", "rate", "expected"} <= set(t.columns)
    assert set(t["method"]) == {"Historical", "Parametric", "Monte Carlo (bootstrap)"}
    assert (t["n_test"] == 200).all()
    assert ((t["rate"] - t["breaches"] / t["n_test"]).abs() < 1e-12).all()
    # при стационарном нормальном ряде частота пробоев 95% близка к 5%
    h95 = t[(t["method"] == "Historical") & (t["level"] == 0.95)].iloc[0]
    assert 0.0 <= h95["rate"] <= 0.12


def test_var_holdout_detects_regime_break():
    idx = pd.bdate_range("2024-01-01", periods=400)
    calm = np.full(300, 0.0)
    calm[::2] = 0.001
    calm[1::2] = -0.001
    r = pd.Series(np.concatenate([calm, np.full(100, -0.02)]), index=idx)
    t = benchmark.var_holdout(r, str(idx[300].date()), budget=1000.0)
    # после смены режима все 100 дней хуже любого VaR, оценённого на спокойной истории
    assert (t["breaches"] == 100).all()


def test_alpha_halves(returns, ff5):
    t = benchmark.alpha_halves(returns, ff5, config.FF_FIXED_END, config.HOLDOUT_START)
    assert list(t.index) == ["train", "test"]
    assert {"alpha", "r2", "n_obs"} <= set(t.columns)
    full = factors.ff5_regression(returns, ff5, end=config.FF_FIXED_END).loc["NVDA"]
    assert t["n_obs"].sum() == full["n_obs"]
