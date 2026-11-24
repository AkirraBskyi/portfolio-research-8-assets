import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from portfolio_lab import config


@pytest.fixture(scope="session")
def returns():
    from portfolio_lab import data
    return data.load_returns(config.FIXED_START, config.FIXED_END)


@pytest.fixture(scope="session")
def ff5():
    from portfolio_lab import data
    return data.load_ff5_daily()


@pytest.fixture(scope="session")
def cov(returns):
    return returns.cov() * config.TRADING_DAYS


@pytest.fixture(scope="session")
def betas(returns):
    from portfolio_lab import risk
    return risk.risk_table(returns)["beta"]


@pytest.fixture(scope="session")
def weights():
    from portfolio_lab import black_litterman as bl
    return bl.market_weights()


@pytest.fixture(scope="session")
def port_ret(returns, weights):
    from portfolio_lab import var
    return var.portfolio_returns(returns, weights)
