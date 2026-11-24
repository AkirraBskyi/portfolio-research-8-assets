# SPEC: PortfolioLab (восстановление исследования 8-активного портфеля)

Цель: рабочий воспроизводимый пайплайн, который заново получает числа из опубликованного исследования
(сентябрь 2026) и пригоден для публикации на GitHub. Оригинальный код утерян: это восстановление по описанию.

## Данные
- 8 тикеров из `config.TICKERS`, цены `yfinance` (auto_adjust=True), кэш в `data/cache/*.csv` (в git не попадает).
- Фиксированное окно проверки: `FIXED_START..FIXED_END` даёт ровно 768 дневных доходностей (2023-09-05 .. 2026-09-25).
- Факторы Fama-French 5 daily (Ken French Data Library, zip), значения в долях (не в процентах), колонки
  `Mkt-RF, SMB, HML, RMW, CMA, RF`.
- Основной запуск (`python -m portfolio_lab`) идёт на скользящем окне до сегодня; тесты всегда на фиксированном.

## Публичный API (имена фиксированы)

| Модуль | Функции |
|---|---|
| `data.py` | `load_prices(start, end, cache_dir="data/cache")` -> DataFrame (Close, колонки=TICKERS); `load_returns(start, end, ...)` -> простые дневные доходности, без NaN; `load_ff5_daily(cache_dir=...)` -> DataFrame |
| `risk.py` | `risk_table(returns, market="SPY")` -> DataFrame по тикерам, колонки `corr, vol, beta, r2, sys_vol, idio_vol` (годовые, в долях); `corr_matrix(returns)` |
| `factors.py` | `ff5_regression(returns, ff5)` -> DataFrame по тикерам, колонки `alpha, mkt, smb, hml, rmw, cma, r2, n_obs`; регрессия `R_i - RF` на 5 факторов (OLS, одновременно); `alpha` годовая (дневная * 252) |
| `capm.py` | `capm_returns(betas, rf, erp)` -> Series полных годовых ожидаемых доходностей `rf + beta*erp` |
| `black_litterman.py` | `market_weights()` -> Series весов (сумма 1) из `config`; `calibrate_delta(w, betas, cov, erp)` = `(w.beta * erp) / (w' cov w)`; `implied_excess_returns(cov, w, delta)` = `delta * cov @ w` (годовые избыточные); `implied_returns(...)` = `rf + excess` |
| `frontier.py` | `tangency(mu_excess, cov, rf, long_only)` -> dict `weights, ret, vol, sharpe`; `min_variance(cov, long_only)` -> dict `weights, vol`; `efficient_frontier(mu, cov, long_only, n=60)` -> DataFrame `vol, ret` + веса |
| `var.py` | `portfolio_returns(returns, weights)`; `historical(r, budget, horizon=1)`, `parametric(r, budget, horizon=1)`, `bootstrap_mc(r, budget, horizon=1, n, seed)` -> dict `{0.95: {"var": x, "cvar": y}, 0.99: {...}}` (потери положительные, в евро) |
| `stress.py` | `SCENARIOS` (dict имя -> Series шоков по тикерам, 4 сценария); `run_stress(weights, shocks, budget)` -> dict `pct, eur` |
| `plots.py` | `make_all(out_dir, returns, ...)` -> создаёт 6 PNG: `correlation_matrix.png, beta_bars.png, risk_decomposition.png, efficient_frontier.png, var_cvar_comparison.png, stress_scenarios.png` (подписи на английском) |
| `__main__.py` | `python -m portfolio_lab [--fixed]` печатает все таблицы, пишет графики в `charts/` |

## v2: проверка вне выборки, сравнение со SPY, look-through

| Модуль | Функции |
|---|---|
| `lookthrough.py` | `effective_exposure(weights, spy_share=None, qqq_share=None)` -> DataFrame (NVDA, MSFT) с колонками `direct, via_spy, via_qqq, total`; `via_x = w_x * доля акции в фонде x` |
| `benchmark.py` | `max_drawdown(r)`; `split(r, start)` -> (train, test); `compare_to_spy(returns, port_ret, rf, start=None)`; `var_holdout(port_ret, split_date, budget)` -> пробои VaR, оценённого на train, на test; `alpha_halves(returns, ff5, end, split_date)` -> альфа NVDA по FF5 на train и test |
| `plots.py` | + `portfolio_vs_spy.png`, `lookthrough_exposure.png` (всего 8 графиков) |

Конфиг: `QQQ_SHARE` (NVDA 8.15%, MSFT 5.31%, Trackinsight на 2026-05-31), `HOLDOUT_START = 2025-09-26`.
Результаты v2 эталонов не имеют (новые блоки), проверяются логикой и ручными примерами.

## Допущения (в README описать честно)
- Rf = 4.2%, ERP = 4.29% (из CAPM опубликованных чисел: (12.98 - 4.2) / 2.044), бюджет €2000.
- Веса: двухуровневая конструкция из `config` (55/15/15/10/5; NVDA 8.3%, MSFT 4.6% от акционного блока; SPY:QQQ из опубликованных весов).
- Ковариация: выборочная дневная * 252.
- Стресс-шоки: заданы заново (см. ниже), НЕ подгоняются под опубликованные -14.64/-11.18/-8.98/-15.62%.

## Стресс-шоки (изменение цены актива, доля)
| Сценарий | SPY | QQQ | NVDA | MSFT | TLT | SHY | GLD | FXF |
|---|---|---|---|---|---|---|---|---|
| Equity crash (COVID-style) | -0.34 | -0.28 | -0.38 | -0.28 | +0.08 | +0.01 | -0.03 | +0.01 |
| Rate shock (2022-style) | -0.19 | -0.33 | -0.50 | -0.28 | -0.31 | -0.04 | -0.01 | 0.00 |
| AI-bubble burst | -0.15 | -0.25 | -0.60 | -0.30 | +0.05 | +0.01 | +0.05 | +0.02 |
| Combined (stagflation-style) | -0.25 | -0.30 | -0.45 | -0.30 | -0.20 | -0.03 | +0.10 | 0.00 |
Названия ключей в `SCENARIOS`: `equity_crash`, `rate_shock`, `ai_bubble`, `stagflation`.

## Эталоны и допуски (тесты на фиксированном окне)
| Что | Эталон | Допуск |
|---|---|---|
| NVDA beta | 2.044 | ±0.002 |
| NVDA vol / sys / idio | 46.59% / 31.2% / 34.6% | ±0.05 / 0.1 / 0.1 п.п. |
| FF5 NVDA alpha, R2 | 31.97%, 56.5% | ±0.3 п.п. |
| delta | 2.345 | ±0.01 |
| BL pi NVDA / SPY (полные) | 13.34% / 7.95% | ±0.05 п.п. |
| CAPM NVDA / SPY | 12.98% / 8.49% | ±0.05 п.п. |
| Тангенс long-only: веса | = рыночные веса | ±0.2 п.п. |
| Тангенс: E[R], vol, Sharpe | 7.16%, 11.25%, 0.264 | ±0.05 п.п., ±0.05 п.п., ±0.002 |
| 1-day Historical (€): 95 VaR/CVaR, 99 VaR/CVaR | 21.33 / 30.50 / 34.69 / 46.19 | ±0.1 |
| 1-day Parametric | 21.95 / 27.87 / 31.60 / 36.40 | ±0.15 |
| 1-day MC bootstrap (seed 42) | 21.40 / 30.46 / 34.99 / 46.17 | ±0.7 (шум семплирования) |
| Месяц (21 д), CVaR 95 / 99, MC | ≈105 / ≈146 | ±10 |

Если эталон не воспроизводится, воркер НЕ подгоняет код под число: сообщает конвенцию, которую нашёл, или расхождение.
Расхождения фиксируются в README.

## Критерий приёмки проекта
`pytest -q` зелёный; `python -m portfolio_lab --fixed` печатает таблицу «опубликовано vs воспроизведено» и создаёт 6 графиков;
README содержит оговорку о восстановлении, допущения и таблицу сверки.
