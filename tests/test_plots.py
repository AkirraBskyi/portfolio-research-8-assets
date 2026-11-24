from portfolio_lab import plots

FILES = ["correlation_matrix.png", "beta_bars.png", "risk_decomposition.png",
         "efficient_frontier.png", "var_cvar_comparison.png", "stress_scenarios.png"]


def test_make_all(tmp_path, returns):
    plots.make_all(tmp_path, returns)
    for f in FILES:
        p = tmp_path / f
        assert p.exists() and p.stat().st_size > 5000, f
