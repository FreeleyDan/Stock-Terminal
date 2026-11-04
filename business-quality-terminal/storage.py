"""
storage.py

Persists the portfolio, watchlist, and Buffett notes to user_data.json
next to the app so they survive restarts. (On Streamlit Community Cloud
the filesystem is ephemeral — a sleep/restart can wipe these files, which
is why the app also offers Download/Import for the portfolio and saves
the last scan to a file. Export anything you want to keep.)

Last-scan results are saved as a pickle (not JSON) because snapshots
carry integer-keyed gatekeeper dicts that JSON would turn into strings.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from data_fetcher import normalize_ticker

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")
SCAN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_scan.pkl")

NOTE_FIELDS = ("meaning", "management", "moat_type", "thesis", "sell_trigger")


def _empty() -> Dict[str, Any]:
    return {"portfolio": [], "watchlist": [], "notes": {}}


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
    notes: Dict[str, Any] = {}
    raw_notes = data.get("notes", {}) or {}
    if isinstance(raw_notes, dict):
        for ticker, note in raw_notes.items():
            clean_ticker = normalize_ticker(ticker or "")
            if not clean_ticker or not isinstance(note, dict):
                continue
            notes[clean_ticker] = {k: note.get(k, "") for k in NOTE_FIELDS}
            # Meaning / Management are 1-5 sliders; coerce anything else to "".
            for k in ("meaning", "management"):
                try:
                    notes[clean_ticker][k] = int(notes[clean_ticker][k])
                except (TypeError, ValueError):
                    notes[clean_ticker][k] = ""
    return {"portfolio": portfolio, "watchlist": list(dict.fromkeys(watchlist)), "notes": notes}


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


def save_notes(notes: Dict[str, Any], path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    data["notes"] = notes
    return save_user_data(data, path)


def save_note(ticker: str, note: Dict[str, Any], path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    notes = data.get("notes", {}) or {}
    notes[normalize_ticker(ticker)] = {k: note.get(k, "") for k in NOTE_FIELDS}
    return save_notes(notes, path)


def save_last_scan(scan: Dict[str, Any], path: str = SCAN_FILE) -> bool:
    """Save a scan result + timestamp. Never raises; returns False on failure."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"saved_at": time.time(), "scan": scan}, f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def load_last_scan(path: str = SCAN_FILE) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Returns (scan, saved_at_epoch) or (None, None). Never raises."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload.get("scan"), payload.get("saved_at")
    except Exception:
        return None, None


def parse_imported_data(data: Any) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Validate an uploaded portfolio JSON. Raises ValueError on bad shape.

    Accepted shape: {"portfolio": [{"ticker", "shares", "avg_cost"}], 
    "watchlist": ["AAPL", ...], "notes": {"AAPL": {...}}} — portfolio is
    required, watchlist/notes are optional and default to empty.
    """
    if not isinstance(data, dict) or not isinstance(data.get("portfolio"), list):
        raise ValueError("Expected a JSON object with a 'portfolio' list")
    portfolio = []
    for row in data["portfolio"]:
        if not isinstance(row, dict) or not normalize_ticker(row.get("ticker", "")):
            continue
        portfolio.append(
            {
                "ticker": normalize_ticker(row["ticker"]),
                "shares": _to_float(row.get("shares")),
                "avg_cost": _to_float(row.get("avg_cost")),
            }
        )
    watchlist = []
    for t in data.get("watchlist", []) or []:
        if normalize_ticker(t):
            watchlist.append(normalize_ticker(t))
    notes: Dict[str, Any] = {}
    raw_notes = data.get("notes", {}) or {}
    if isinstance(raw_notes, dict):
        for ticker, note in raw_notes.items():
            if normalize_ticker(ticker or "") and isinstance(note, dict):
                notes[normalize_ticker(ticker)] = {k: note.get(k, "") for k in NOTE_FIELDS}
    return portfolio, list(dict.fromkeys(watchlist)), notes
