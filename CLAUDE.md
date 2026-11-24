# PortfolioLab

Восстановленное исследование портфеля из 8 активов: риск, FF5, CAPM, граница эффективности, Black-Litterman,
VaR/CVaR, стресс-тесты. Источник истины: `SPEC.md`. Публикуется на GitHub.

## Стек
Python 3.10, numpy, pandas, scipy, statsmodels, yfinance, matplotlib, pytest. Без API-ключей.

## Структура
```
portfolio_lab/   пакет (config, data, risk, factors, capm, black_litterman, frontier, var, stress, plots, __main__)
tests/           pytest, окно фиксировано (config.FIXED_START..FIXED_END), эталоны из SPEC.md
data/cache/      кэш цен и факторов (в git не попадает)
charts/          графики
```

## Команды
- Тесты: `python -m pytest -q`
- Запуск: `python -m portfolio_lab` (скользящее окно) или `python -m portfolio_lab --fixed` (окно проверки)

## Соглашения
- Доходности простые дневные; годовые = дневные * 252; все величины в долях (не в процентах).
- Тесты не ослаблять и не подгонять код под эталон: расхождение фиксировать в README.
- Личных данных (имя, почта, пути пользователя) в репозитории нет.
- Коммит после каждой подзадачи, сообщения на русском, кратко.
