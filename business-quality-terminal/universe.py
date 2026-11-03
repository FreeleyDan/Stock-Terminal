"""
universe.py

Ticker universes for the market scanner:
  - S&P 500 (Wikipedia constituent table)
  - Every common-stock listing on NYSE + Nasdaq (SEC's exchange-tagged
    ticker file, ~6,000 symbols). OTC/pink-sheet names are excluded; funds,
    shells and loss-makers drop out quickly during the scan itself.
"""

from __future__ import annotations

from io import StringIO
from typing import List

import pandas as pd
import requests

from data_fetcher import (
    SEC_TICKER_HEADERS,
    SEC_USER_AGENT,
    _get_ticker_to_cik,
    cached,
    normalize_ticker,
    sec_get,
)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_EXCHANGE_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
US_EXCHANGES = {"NYSE", "NASDAQ"}

# Used ONLY if the live fetch fails, so the scanner degrades visibly
# (a ~50 ticker universe) instead of crashing.
_FALLBACK_TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA", "UNH",
    "JNJ", "V", "XOM", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "COST", "AVGO", "MCD", "PFE", "TMO", "CSCO", "ACN", "ABT",
    "DHR", "LIN", "VZ", "ADBE", "NKE", "CRM", "NEE", "TXN", "PM", "WFC",
    "RTX", "UNP", "BMY", "QCOM", "UPS", "HON", "ORCL", "SBUX", "LOW", "IBM",
]


@cached(ttl_seconds=24 * 60 * 60)
def get_sp500_tickers() -> List[str]:
    try:
        resp = requests.get(
            WIKIPEDIA_SP500_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=15
        )
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        tickers = [normalize_ticker(t) for t in table["Symbol"].astype(str)]
        tickers = [t for t in tickers if t and t != "NAN"]
        if len(tickers) < 400:
            return list(_FALLBACK_TICKERS)
        return tickers
    except Exception:
        return list(_FALLBACK_TICKERS)


@cached(ttl_seconds=24 * 60 * 60)
def get_all_us_listed_tickers() -> List[str]:
    """NYSE + Nasdaq tickers from SEC's exchange file. If that file can't be
    read, falls back to every ticker in SEC's plain ticker map (a superset
    that includes OTC names -- slower to scan, but still complete)."""
    try:
        resp = sec_get(SEC_EXCHANGE_TICKERS_URL, SEC_TICKER_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        fields = payload["fields"]
        t_idx, e_idx = fields.index("ticker"), fields.index("exchange")
        tickers = [
            normalize_ticker(row[t_idx])
            for row in payload["data"]
            if str(row[e_idx]).upper() in US_EXCHANGES and row[t_idx]
        ]
        tickers = sorted(set(t for t in tickers if t))
        if len(tickers) >= 1000:
            return tickers
    except Exception:
        pass
    try:
        mapping = _get_ticker_to_cik()
        if mapping:
            return sorted(mapping.keys())
    except Exception:
        pass
    return list(_FALLBACK_TICKERS)
