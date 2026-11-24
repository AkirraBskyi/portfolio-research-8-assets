import numpy as np
from portfolio_lab import capm, black_litterman as bl, config


def test_weights():
    w = bl.market_weights()
    assert abs(w.sum() - 1) < 1e-9
    pub = {"SPY": .2959, "QQQ": .1832, "NVDA": .0457, "MSFT": .0253,
           "TLT": .15, "SHY": .15, "GLD": .10, "FXF": .05}
    for k, v in pub.items():
        assert abs(w[k] - v) < 0.0005, k


def test_capm(betas):
    c = capm.capm_returns(betas, config.RF, config.ERP)
    assert abs(c["NVDA"] - 0.1298) < 0.0005
    assert abs(c["SPY"] - 0.0849) < 0.0005


def test_delta_and_pi(betas, cov, weights):
    d = bl.calibrate_delta(weights, betas, cov, config.ERP)
    assert abs(d - 2.345) < 0.01
    ex = bl.implied_excess_returns(cov, weights, d)
    assert abs(ex["NVDA"] - 0.0915) < 0.0005
    assert abs(ex["SPY"] - 0.0376) < 0.0005
    full = bl.implied_returns(cov, weights, d, config.RF)
    assert abs(full["NVDA"] - 0.1334) < 0.0005
    assert abs(full["SPY"] - 0.0795) < 0.0005
    np.testing.assert_allclose(full - config.RF, ex)
