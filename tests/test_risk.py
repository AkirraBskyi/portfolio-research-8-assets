import numpy as np
from portfolio_lab import risk


def test_nvda_numbers(returns):
    t = risk.risk_table(returns)
    n = t.loc["NVDA"]
    assert abs(n["beta"] - 2.044) < 0.002
    assert abs(n["vol"] - 0.4659) < 0.0005
    assert abs(n["sys_vol"] - 0.312) < 0.001
    assert abs(n["idio_vol"] - 0.346) < 0.001
    assert abs(t.loc["SHY", "beta"] - 0.008) < 0.002
    assert abs(t.loc["SPY", "beta"] - 1.0) < 1e-9


def test_variance_additivity(returns):
    t = risk.risk_table(returns)
    np.testing.assert_allclose(t["sys_vol"] ** 2 + t["idio_vol"] ** 2, t["vol"] ** 2, atol=1e-9)


def test_corr_matrix(returns):
    c = risk.corr_matrix(returns)
    assert c.shape == (8, 8)
    np.testing.assert_allclose(np.diag(c.values), 1.0)
    np.testing.assert_allclose(c.values, c.values.T)
    assert abs(c.loc["SPY", "QQQ"] - 0.95) < 0.01
