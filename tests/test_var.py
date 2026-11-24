from portfolio_lab import var, config

B = config.BUDGET_EUR


def _chk(res, exp, tol):
    got = [res[0.95]["var"], res[0.95]["cvar"], res[0.99]["var"], res[0.99]["cvar"]]
    for g, e in zip(got, exp):
        assert abs(g - e) <= tol, (got, exp)


def test_portfolio_returns(returns, weights, port_ret):
    assert len(port_ret) == 768
    assert abs(port_ret.iloc[0] - (returns.iloc[0] * weights).sum()) < 1e-12


def test_historical_1d(port_ret):
    _chk(var.historical(port_ret, B), [21.33, 30.50, 34.69, 46.19], 0.1)


def test_parametric_1d(port_ret):
    _chk(var.parametric(port_ret, B), [21.95, 27.87, 31.60, 36.40], 0.15)


def test_bootstrap_1d(port_ret):
    res = var.bootstrap_mc(port_ret, B, n=config.BOOTSTRAP_N, seed=config.BOOTSTRAP_SEED)
    _chk(res, [21.40, 30.46, 34.99, 46.17], 0.7)


def test_cvar_not_below_var(port_ret):
    for fn in (var.historical, var.parametric):
        r = fn(port_ret, B)
        for lvl in (0.95, 0.99):
            assert r[lvl]["cvar"] >= r[lvl]["var"]
        assert r[0.99]["var"] > r[0.95]["var"]


def test_parametric_understates_tail(port_ret):
    h = var.historical(port_ret, B)[0.99]["cvar"]
    p = var.parametric(port_ret, B)[0.99]["cvar"]
    assert 0.15 < (h - p) / h < 0.35        # около 24% в опубликованном исследовании


def test_bootstrap_reproducible(port_ret):
    a = var.bootstrap_mc(port_ret, B, n=5000, seed=1)
    b = var.bootstrap_mc(port_ret, B, n=5000, seed=1)
    assert a == b


def test_monthly_horizon(port_ret):
    r = var.bootstrap_mc(port_ret, B, horizon=config.MONTH_DAYS,
                         n=config.BOOTSTRAP_N, seed=config.BOOTSTRAP_SEED)
    assert abs(r[0.95]["cvar"] - 105) < 10
    assert abs(r[0.99]["cvar"] - 146) < 10
    d1 = var.bootstrap_mc(port_ret, B, horizon=1, n=20000, seed=1)
    assert r[0.99]["var"] > d1[0.99]["var"]
