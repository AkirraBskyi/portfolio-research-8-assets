"""Загрузка данных: цены (yfinance) и факторы Fama-French 5 (Ken French Data Library)."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from . import config

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def load_prices(start, end, cache_dir="data/cache"):
    """Скорректированные цены закрытия, колонки = config.TICKERS. `end` включительно."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"prices_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)[config.TICKERS]

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(config.TICKERS, start=start, end=end_excl,
                      auto_adjust=True, progress=False)["Close"]
    prices = raw[config.TICKERS].dropna(how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    prices.to_csv(path)
    return prices


def load_returns(start, end, cache_dir="data/cache"):
    """Простые дневные доходности, без NaN."""
    return load_prices(start, end, cache_dir).pct_change().dropna()


def load_ff5_daily(cache_dir="data/cache"):
    """Дневные факторы FF5 + RF в долях (не в процентах)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "ff5_daily.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)[FF5_COLUMNS]

    resp = requests.get(FF5_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("latin-1")

    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7 or not parts[0].isdigit():
            break
        rows.append(parts[:7])
    df = pd.DataFrame(rows, columns=["Date"] + FF5_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float) / 100.0
    df.to_csv(path)
    return df[FF5_COLUMNS]
