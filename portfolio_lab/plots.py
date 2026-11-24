"""Шесть графиков исследования (подписи на английском)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import black_litterman as bl, config, frontier, lookthrough, risk, stress, var

BLUE, ORANGE, GREEN, RED, AMBER, GRAY = "#2a78d6", "#ec6a35", "#1aaf7a", "#e24b4b", "#eca200", "#8a8f98"


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=140, facecolor="white")
    plt.close(fig)


def correlation_matrix(returns, out_dir):
    c = risk.corr_matrix(returns)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Correlation matrix ({len(returns)} daily returns)", fontweight="bold")
    _save(fig, out_dir, "correlation_matrix.png")


def beta_bars(returns, out_dir):
    b = risk.risk_table(returns)["beta"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(b.index, b.values, color=[BLUE if v >= 0 else RED for v in b.values], width=0.6)
    ax.axhline(1, color=GRAY, ls="--", lw=1)
    for i, v in enumerate(b.values):
        ax.text(i, v + (0.04 if v >= 0 else -0.12), f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("Beta to SPY")
    ax.set_title("Beta of each asset to SPY", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "beta_bars.png")


def risk_decomposition(returns, out_dir):
    t = risk.risk_table(returns)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(t.index, t["sys_vol"] * 100, color=BLUE, label="Systematic (to SPY)", width=0.6)
    ax.bar(t.index, t["idio_vol"] * 100, bottom=t["sys_vol"] * 100, color=ORANGE,
           label="Idiosyncratic", width=0.6)
    ax.set_ylabel("Annual volatility (%), stacked for display\n(variances add, not volatilities)")
    ax.set_title("Risk decomposition: systematic vs idiosyncratic", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "risk_decomposition.png")


def efficient_frontier(returns, out_dir):
    cov = returns.cov() * config.TRADING_DAYS
    w = bl.market_weights()
    betas = risk.risk_table(returns)["beta"]
    delta = bl.calibrate_delta(w, betas, cov, config.ERP)
    mu = bl.implied_returns(cov, w, delta, config.RF)
    pi = mu - config.RF
    fl = frontier.efficient_frontier(mu, cov, long_only=True, n=60)
    fs = frontier.efficient_frontier(mu, cov, long_only=False, n=60)
    tg = frontier.tangency(pi, cov, config.RF, long_only=True)
    vol = np.sqrt(np.diag(cov.values))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fs["vol"] * 100, fs["ret"] * 100, color=AMBER, lw=2.5, label="With shorting")
    ax.plot(fl["vol"] * 100, fl["ret"] * 100, color=GREEN, lw=2.5, label="Long-only")
    xs = np.array([0, fs["vol"].max() * 100])
    ax.plot(xs, config.RF * 100 + tg["sharpe"] * xs, color=GRAY, ls="--", lw=1.2, label="Capital Market Line")
    ax.scatter(vol * 100, mu.values * 100, color="black", zorder=3)
    for name, x, y in zip(cov.index, vol * 100, mu.values * 100):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")
    ax.scatter([tg["vol"] * 100], [tg["ret"] * 100], s=110, color=BLUE, zorder=4,
               label="Tangency (max Sharpe)")
    ax.scatter([0], [config.RF * 100], color=GRAY, zorder=3)
    ax.annotate("Rf", (0, config.RF * 100), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("Risk, annual volatility (%)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Efficient frontier: 8 assets (Black-Litterman equilibrium returns)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "efficient_frontier.png")


def var_cvar_comparison(returns, out_dir):
    pr = var.portfolio_returns(returns, bl.market_weights())
    t = var.summary_table(pr, config.BUDGET_EUR)
    cols = ["95% VaR", "95% CVaR", "99% VaR", "99% CVaR"]
    colors = [GREEN, BLUE, AMBER, RED]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (i - 1.5) * 0.2, t[c].values, width=0.2, color=col, label=c)
    ax.set_xticks(x, t.index)
    ax.set_ylabel(f"EUR (budget {config.BUDGET_EUR:,.0f}, 1-day horizon)")
    ax.set_title("VaR / CVaR by method (model portfolio)", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "var_cvar_comparison.png")


def stress_scenarios(out_dir):
    t = stress.run_all(bl.market_weights(), config.BUDGET_EUR)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [RED if "Combined" in n else ORANGE for n in t.index]
    ax.barh(t.index, t["eur"].values, color=colors)
    for i, (e, p) in enumerate(zip(t["eur"], t["pct"])):
        ax.text(e / 2, i, f"{e:,.0f} EUR ({p * 100:.1f}%)", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.set_xlabel(f"EUR on a {config.BUDGET_EUR:,.0f} budget")
    ax.set_title("Scenario stress tests (model portfolio)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir, "stress_scenarios.png")


def portfolio_vs_spy(returns, out_dir):
    pr = var.portfolio_returns(returns, bl.market_weights())
    cut = pd.Timestamp(config.HOLDOUT_START)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot((1 + pr).cumprod() * 100, color=BLUE, lw=2, label="Model portfolio (assumed weights)")
    ax.plot((1 + returns[config.MARKET]).cumprod() * 100, color=ORANGE, lw=2, label="SPY")
    ax.axvline(cut, color=GRAY, ls="--", lw=1.2)
    ax.axvspan(cut, returns.index.max(), color=GRAY, alpha=0.12)
    ax.text(cut, ax.get_ylim()[0] + 2, "  hold-out year", fontsize=9, color=GRAY, va="bottom")
    ax.set_ylabel("Growth of 100 (constant weights)")
    ax.set_title("Model portfolio vs SPY", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "portfolio_vs_spy.png")


def lookthrough_exposure(out_dir):
    t = lookthrough.effective_exposure(bl.market_weights()) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottoms = np.zeros(len(t))
    for col, color, label in (("direct", BLUE, "Held directly"), ("via_spy", ORANGE, "Inside SPY"),
                              ("via_qqq", GREEN, "Inside QQQ")):
        ax.bar(t.index, t[col], bottom=bottoms, color=color, label=label, width=0.5)
        bottoms += t[col].values
    for i, tot in enumerate(t["total"]):
        ax.text(i, tot + 0.2, f"{tot:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% of portfolio")
    ax.set_title("Look-through exposure: direct vs effective", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "lookthrough_exposure.png")


def make_all(out_dir, returns):
    """Создаёт 8 PNG в out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    correlation_matrix(returns, out_dir)
    beta_bars(returns, out_dir)
    risk_decomposition(returns, out_dir)
    efficient_frontier(returns, out_dir)
    var_cvar_comparison(returns, out_dir)
    stress_scenarios(out_dir)
    portfolio_vs_spy(returns, out_dir)
    lookthrough_exposure(out_dir)
