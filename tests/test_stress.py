import pandas as pd
from portfolio_lab import stress, config


def test_run_stress_hand_example():
    w = pd.Series({"A": 0.5, "B": 0.5})
    s = pd.Series({"A": -0.2, "B": 0.1})
    r = stress.run_stress(w, s, 2000.0)
    assert abs(r["pct"] - (-0.05)) < 1e-12
    assert abs(r["eur"] - (-100.0)) < 1e-9


def test_scenarios_defined():
    assert set(stress.SCENARIOS) == {"equity_crash", "rate_shock", "ai_bubble", "stagflation"}
    for name, sh in stress.SCENARIOS.items():
        assert set(sh.index) == set(config.TICKERS), name
        assert (sh.abs() <= 1).all()
    assert stress.SCENARIOS["equity_crash"]["SPY"] == -0.34
    assert stress.SCENARIOS["ai_bubble"]["NVDA"] == -0.60
    assert stress.SCENARIOS["stagflation"]["GLD"] == 0.10


def test_all_scenarios_are_losses(weights):
    for name, sh in stress.SCENARIOS.items():
        r = stress.run_stress(weights, sh, config.BUDGET_EUR)
        assert r["pct"] < 0 and r["eur"] < 0, name
