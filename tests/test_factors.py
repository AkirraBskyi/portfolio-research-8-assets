from portfolio_lab import factors


def test_ff5_nvda(returns, ff5):
    t = factors.ff5_regression(returns, ff5)
    n = t.loc["NVDA"]
    assert 745 <= n["n_obs"] <= 755
    assert abs(n["alpha"] - 0.3197) < 0.003
    assert abs(n["r2"] - 0.565) < 0.003
    assert abs(n["mkt"] - 1.913) < 0.01
    assert abs(t.loc["SPY", "r2"] - 0.992) < 0.003
    assert abs(t.loc["MSFT", "r2"] - 0.448) < 0.005
