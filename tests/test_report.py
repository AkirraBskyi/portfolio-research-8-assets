from portfolio_lab import report, config


def test_reproduces_published_numbers(returns, ff5):
    res = report.run_pipeline(returns, ff5, config.FF_FIXED_END)
    cmp = report.compare_with_published(res)
    bad = cmp[~cmp["ok"]]
    assert bad.empty, bad.to_string()


def test_pipeline_keys(returns, ff5):
    res = report.run_pipeline(returns, ff5, config.FF_FIXED_END)
    for k in ("risk", "ff5", "capm", "weights", "delta", "pi", "tangency", "var_1d", "var_21d", "stress"):
        assert k in res
