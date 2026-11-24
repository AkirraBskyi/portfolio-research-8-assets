import pandas as pd
from portfolio_lab import config


def test_returns_window_and_shape(returns):
    assert list(returns.columns) == config.TICKERS
    assert len(returns) == 768
    assert returns.index.min() == pd.Timestamp("2023-09-05")
    assert returns.index.max() == pd.Timestamp("2026-09-25")
    assert not returns.isna().any().any()


def test_ff5_format(ff5):
    assert list(ff5.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert ff5.abs().max().max() < 0.2          # доли, не проценты
    assert ff5.index.is_monotonic_increasing
