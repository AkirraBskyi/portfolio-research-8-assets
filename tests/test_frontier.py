from portfolio_lab import black_litterman as bl, frontier, config


def _pi(betas, cov, weights):
    d = bl.calibrate_delta(weights, betas, cov, config.ERP)
    return bl.implied_excess_returns(cov, weights, d)


def test_tangency_long_only_equals_market(betas, cov, weights):
    pi = _pi(betas, cov, weights)
    t = frontier.tangency(pi, cov, config.RF, long_only=True)
    for k in weights.index:
        assert abs(t["weights"][k] - weights[k]) < 0.002, k
    assert abs(t["ret"] - 0.0716) < 0.0005
    assert abs(t["vol"] - 0.1125) < 0.0005
    assert abs(t["sharpe"] - 0.264) < 0.002
    assert abs(t["weights"].sum() - 1) < 1e-8


def test_tangency_long_only_has_no_shorts(betas, cov, weights):
    t = frontier.tangency(_pi(betas, cov, weights), cov, config.RF, long_only=True)
    assert (t["weights"] >= -1e-9).all()


def test_min_variance(cov):
    short = frontier.min_variance(cov, long_only=False)
    long = frontier.min_variance(cov, long_only=True)
    assert abs(short["weights"].sum() - 1) < 1e-8
    assert abs(short["vol"] - 0.013) < 0.003
    assert (long["weights"] >= -1e-9).all()
    assert long["vol"] >= short["vol"] - 1e-9


def test_frontier_curve(betas, cov, weights):
    pi = _pi(betas, cov, weights)
    f = frontier.efficient_frontier(pi + config.RF, cov, long_only=True, n=30)
    assert len(f) == 30
    assert {"vol", "ret"} <= set(f.columns)
    assert f["vol"].is_monotonic_increasing
    # граница с шортом не ниже границы без шорта
    fs = frontier.efficient_frontier(pi + config.RF, cov, long_only=False, n=30)
    assert fs["ret"].max() >= f["ret"].max() - 1e-9
