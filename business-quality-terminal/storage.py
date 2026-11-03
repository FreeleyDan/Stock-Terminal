"""
storage.py

Persists the portfolio and watchlist to user_data.json next to the app so
they survive restarts. (On Streamlit Community Cloud the filesystem is
ephemeral -- run locally if you want them to persist.)
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from data_fetcher import normalize_ticker

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")


def _empty() -> Dict[str, Any]:
    return {"portfolio": [], "watchlist": []}


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def load_user_data(path: str = DATA_FILE) -> Dict[str, Any]:
    """Never raises. A corrupt file is moved aside (.corrupt) and replaced
    by an empty one rather than crashing the app."""
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("bad shape")
    except Exception:
        try:
            shutil.move(path, path + ".corrupt")
        except Exception:
            pass
        return _empty()

    portfolio = []
    for row in data.get("portfolio", []) or []:
        if isinstance(row, dict) and normalize_ticker(row.get("ticker", "")):
            portfolio.append(
                {
                    "ticker": normalize_ticker(row["ticker"]),
                    "shares": _to_float(row.get("shares")),
                    "avg_cost": _to_float(row.get("avg_cost")),
                }
            )
    watchlist = [normalize_ticker(t) for t in (data.get("watchlist", []) or []) if normalize_ticker(t)]
    return {"portfolio": portfolio, "watchlist": list(dict.fromkeys(watchlist))}


def save_user_data(data: Dict[str, Any], path: str = DATA_FILE) -> bool:
    """Atomic write. Returns False (never raises) if it couldn't save."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def save_portfolio(holdings: List[Dict[str, Any]], path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    data["portfolio"] = holdings
    return save_user_data(data, path)


def save_watchlist(tickers: List[str], path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    data["watchlist"] = tickers
    return save_user_data(data, path)
