"""python -m portfolio_lab [--fixed] [--no-charts] [--out charts]"""
import argparse

import pandas as pd

from . import config, data, plots, report


def main():
    ap = argparse.ArgumentParser(description="8-asset portfolio research pipeline")
    ap.add_argument("--fixed", action="store_true",
                    help="окно проверки 2023-09 .. 2026-09 (воспроизводит опубликованное исследование)")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--out", default="charts")
    args = ap.parse_args()

    if args.fixed:
        start, end, ff_end = config.FIXED_START, config.FIXED_END, config.FF_FIXED_END
    else:
        today = pd.Timestamp.today().normalize()
        start, end, ff_end = (today - pd.DateOffset(years=3)).strftime("%Y-%m-%d"), \
            today.strftime("%Y-%m-%d"), None

    returns = data.load_returns(start, end)
    ff5 = data.load_ff5_daily()
    res = report.run_pipeline(returns, ff5, ff_end)
    pd.set_option("display.width", 160, "display.float_format", lambda v: f"{v:,.4f}")

    print(f"\nWindow: {returns.index.min().date()} .. {returns.index.max().date()} ({len(returns)} daily returns)")
    print("\n== Risk table (annual, fractions) ==\n", res["risk"])
    print("\n== Fama-French 5 factors ==\n", res["ff5"])
    print("\n== CAPM expected returns ==\n", res["capm"])
    print(f"\n== Black-Litterman ==\ndelta = {res['delta']:.3f}")
    print(pd.DataFrame({"weight": res["weights"], "pi_excess": res["pi"]}))
    tg = res["tangency"]
    print(f"\n== Tangency (long-only) ==\nE[R] {tg['ret']:.2%}  vol {tg['vol']:.2%}  Sharpe {tg['sharpe']:.3f}")
    print(f"Min-variance vol: with shorting {res['min_var_short']['vol']:.2%}, long-only {res['min_var_long']['vol']:.2%}")
    print(f"\n== VaR / CVaR, 1 day, EUR {config.BUDGET_EUR:,.0f} ==\n", res["var_1d"])
    print(f"\n== VaR / CVaR, {config.MONTH_DAYS} days ==\n", res["var_21d"])
    st = res["stress"].copy()
    st["pct"] *= 100
    print("\n== Stress scenarios (pct, EUR) ==\n", st)

    if args.fixed:
        cmp = report.compare_with_published(res)
        print("\n== Published vs reproduced ==\n", cmp.to_string(index=False))
        print(f"\nMatched: {int(cmp['ok'].sum())}/{len(cmp)}")
        print("\nStress shocks were re-specified, so these are NOT expected to match the published values:")
        for k, v in report.PUBLISHED_STRESS.items():
            print(f"  {k}: published {v:.2f}%  |  this run {st.loc[k, 'pct']:.2f}%")

    if not args.no_charts:
        plots.make_all(args.out, returns)
        print(f"\nCharts saved to {args.out}/")


if __name__ == "__main__":
    main()
