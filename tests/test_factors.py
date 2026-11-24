from portfolio_lab import factors, config


def test_ff5_nvda(returns, ff5):
    # окно факторов как в исследовании: до 2026-08-31 (FF публикуется с лагом)
    t = factors.ff5_regression(returns, ff5, end=config.FF_FIXED_END)
    n = t.loc["NVDA"]
    assert 745 <= n["n_obs"] <= 755
    assert abs(n["alpha"] - 0.3197) < 0.003
    assert abs(n["r2"] - 0.565) < 0.003
    assert abs(n["mkt"] - 1.913) < 0.01
    assert abs(t.loc["SPY", "r2"] - 0.992) < 0.003
    assert abs(t.loc["MSFT", "r2"] - 0.448) < 0.005
