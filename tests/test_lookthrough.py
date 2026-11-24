import pandas as pd
from portfolio_lab import lookthrough, config


def test_hand_example():
    w = pd.Series({"SPY": 0.5, "QQQ": 0.2, "NVDA": 0.1, "MSFT": 0.0, "TLT": 0.2,
                   "SHY": 0.0, "GLD": 0.0, "FXF": 0.0})
    t = lookthrough.effective_exposure(w, spy_share={"NVDA": 0.1, "MSFT": 0.05},
                                       qqq_share={"NVDA": 0.2, "MSFT": 0.1})
    assert abs(t.loc["NVDA", "direct"] - 0.1) < 1e-12
    assert abs(t.loc["NVDA", "via_spy"] - 0.05) < 1e-12
    assert abs(t.loc["NVDA", "via_qqq"] - 0.04) < 1e-12
    assert abs(t.loc["NVDA", "total"] - 0.19) < 1e-12
    assert abs(t.loc["MSFT", "total"] - (0.5 * 0.05 + 0.2 * 0.1)) < 1e-12


def test_model_portfolio(weights):
    t = lookthrough.effective_exposure(weights)
    assert abs(t.loc["NVDA", "total"] - 0.0852) < 0.0004
    assert abs(t.loc["MSFT", "total"] - 0.0486) < 0.0004
    # прямая доля сильно занижает реальную: NVDA почти вдвое
    assert 1.7 < t.loc["NVDA", "total"] / t.loc["NVDA", "direct"] < 2.1


def test_sensitivity_monotonic(weights):
    lo = lookthrough.effective_exposure(weights, qqq_share={"NVDA": 0.08, "MSFT": 0.05})
    hi = lookthrough.effective_exposure(weights, qqq_share={"NVDA": 0.10, "MSFT": 0.09})
    assert hi.loc["NVDA", "total"] > lo.loc["NVDA", "total"]
    assert hi.loc["MSFT", "total"] > lo.loc["MSFT", "total"]


def test_config_has_sources():
    assert set(config.QQQ_SHARE) == {"NVDA", "MSFT"}
