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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from data_fetcher import normalize_ticker

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")
SCAN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_scan.pkl")

NOTE_FIELDS = ("meaning", "management", "moat_type", "thesis", "sell_trigger")

# Manual dividend data the user fills in when SEC filings don't carry a
# field. Stored per ticker, and ALWAYS flagged as manual in the UI (✏️) —
# manual values never pretend to be pulled data. Only these fields can
# be set manually, and only in the dividend tab.
MANUAL_FIELDS = (
    "dividend_per_share",  # $ per share, latest annual
    "payout_ratio",  # decimal, e.g. 0.54 for 54%
    "dividend_growth_5yr",  # decimal CAGR, e.g. 0.10 for 10%
    "dividend_streak",  # int, consecutive years of dividend increases
    "fcf_dividend_coverage",  # x, free cash flow per share / DPS
    "debt_ratio",  # x, long-term debt / owner earnings
)

# Manual sector tag (string, not numeric): used only to pick the
# sector-aware payout tiers in the dividend safety score. The app does
# NOT fetch sector from Yahoo — this is user-entered and flagged ✏️.
MANUAL_SECTOR_OPTIONS = ("Utility", "REIT", "Other / standard")


def _empty() -> Dict[str, Any]:
    return {
        "portfolio": [], "watchlist": [], "notes": {}, "manual": {},
        "portfolios": {"Main": []}, "watchlists": {"Main": []},
        "portfolio_cash": {"Main": 0.0}, "scan_presets": {"undervalued": {}, "dividend": {}},
        "signal_log": [], "portfolio_account_types": {"Main": "Not set"},
        "tax_rates": {"lt_pct": 0.0, "st_pct": 0.0, "ca_pct": 0.0}, "sales_log": [],
        "dividend_income_goal": 0.0,
        "dividend_watch": {}, "dividend_goal_history": [],
        "holdings_baseline": {},
    }


def _parse_date(value: Any) -> Optional[str]:
    """Normalize a date to YYYY-MM-DD, or None if it is not a real date.
    Accepts 'YYYY-MM-DD' strings and date/datetime objects."""
    if value is None or value == "":
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return None
    text = str(value).strip()[:10]
    try:
        from datetime import datetime as _dt
        return _dt.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean_holding(row: Any) -> Optional[Dict[str, Any]]:
    """One holding: ticker, shares, avg_cost, plus optional target_pct
    (0-100, % of account), drip (bool), and purchase_date (YYYY-MM-DD).
    Missing optionals stay None/False. Purchase date is a single-lot
    assumption: if you bought in multiple lots, one date cannot split
    long-term vs short-term per lot — the UI says so."""
    if not isinstance(row, dict) or not normalize_ticker(row.get("ticker", "")):
        return None
    holding: Dict[str, Any] = {
        "ticker": normalize_ticker(row["ticker"]),
        "shares": _to_float(row.get("shares")),
        "avg_cost": _to_float(row.get("avg_cost")),
    }
    target = _to_float(row.get("target_pct"))
    if target is not None:
        holding["target_pct"] = max(0.0, min(100.0, target))
    holding["drip"] = bool(row.get("drip")) if row.get("drip") is not None else False
    purchase_date = _parse_date(row.get("purchase_date"))
    if purchase_date:
        holding["purchase_date"] = purchase_date
    return holding


ACCOUNT_TYPES = ("Taxable", "Roth IRA", "Traditional IRA", "Not set")


def _clean_portfolios(raw: Any, legacy: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Named portfolios. Legacy single-list files migrate to {"Main": [...]}.
    Empty names are dropped; a file with no portfolios at all gets Main."""
    portfolios: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for name, rows in raw.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(rows, list):
                continue
            cleaned = [h for h in (_clean_holding(r) for r in rows) if h is not None]
            portfolios[name.strip()] = cleaned
    if not portfolios and isinstance(legacy, list):
        cleaned = [h for h in (_clean_holding(r) for r in legacy) if h is not None]
        portfolios["Main"] = cleaned
    if not portfolios:
        portfolios["Main"] = []
    return portfolios


def _clean_watchlists(raw: Any, legacy: Any) -> Dict[str, List[str]]:
    watchlists: Dict[str, List[str]] = {}
    if isinstance(raw, dict):
        for name, tickers in raw.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(tickers, list):
                continue
            cleaned = list(dict.fromkeys(t for t in (normalize_ticker(t) for t in tickers) if t))
            watchlists[name.strip()] = cleaned
    if not watchlists and isinstance(legacy, list):
        cleaned = list(dict.fromkeys(t for t in (normalize_ticker(t) for t in legacy) if t))
        watchlists["Main"] = cleaned
    if not watchlists:
        watchlists["Main"] = []
    return watchlists


def _clean_manual(raw_manual: Any) -> Dict[str, Any]:
    """Validate manual entries: known fields only, numeric, streak as int."""
    manual: Dict[str, Any] = {}
    if not isinstance(raw_manual, dict):
        return manual
    for ticker, entry in raw_manual.items():
        clean_ticker = normalize_ticker(ticker or "")
        if not clean_ticker or not isinstance(entry, dict):
            continue
        clean: Dict[str, Any] = {}
        for k in MANUAL_FIELDS:
            if k not in entry or entry.get(k) is None or entry.get(k) == "":
                continue
            if k == "dividend_streak":
                try:
                    clean[k] = int(entry[k])
                except (TypeError, ValueError):
                    continue
            else:
                v = _to_float(entry.get(k))
                if v is not None:
                    clean[k] = v
        sector = entry.get("sector")
        if isinstance(sector, str) and sector in MANUAL_SECTOR_OPTIONS:
            clean["sector"] = sector
        if clean:
            manual[clean_ticker] = clean
    return manual


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def _clean_dividend_watch(raw_watch: Any) -> Dict[str, Any]:
    """Validate the payout baselines for the dividend-cut watch: one entry
    per ticker with the last seen payout ratio and the date it was set.
    (Dividend cuts themselves are judged from the actual payment history
    in the snapshot, not from this baseline.)"""
    watch: Dict[str, Any] = {}
    if not isinstance(raw_watch, dict):
        return watch
    for ticker, entry in raw_watch.items():
        clean_ticker = normalize_ticker(ticker or "")
        if not clean_ticker or not isinstance(entry, dict):
            continue
        watch[clean_ticker] = {
            "payout": _to_float(entry.get("payout")),
            "date": _parse_date(entry.get("date")),
        }
    return watch


def _clean_dividend_goal_history(raw_history: Any) -> List[Dict[str, Any]]:
    """Validate the monthly income-vs-goal log: one entry per YYYY-MM month."""
    history: List[Dict[str, Any]] = []
    if not isinstance(raw_history, list):
        return history
    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        month = str(entry.get("month", ""))[:7]
        if len(month) != 7 or month[4] != "-":
            continue
        history.append({
            "month": month,
            "income": _to_float(entry.get("income")) or 0.0,
            "goal": _to_float(entry.get("goal")) or 0.0,
            "buys": _to_float(entry.get("buys")) or 0.0,
            "raises": _to_float(entry.get("raises")) or 0.0,
        })
    history.sort(key=lambda e: e.get("month", ""))
    return history[-24:]


def _clean_holdings_baseline(raw_baseline: Any) -> Dict[str, Any]:
    """Validate the last-seen holdings state used to attribute income
    changes: one entry per ticker with shares, DPS, income and the date
    it was seen."""
    baseline: Dict[str, Any] = {}
    if not isinstance(raw_baseline, dict):
        return baseline
    for ticker, entry in raw_baseline.items():
        clean_ticker = normalize_ticker(ticker or "")
        if not clean_ticker or not isinstance(entry, dict):
            continue
        baseline[clean_ticker] = {
            "shares": _to_float(entry.get("shares")),
            "dps": _to_float(entry.get("dps")),
            "income": _to_float(entry.get("income")),
            "date": _parse_date(entry.get("date")),
        }
    return baseline


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

    portfolios = _clean_portfolios(data.get("portfolios"), data.get("portfolio", []))
    watchlists = _clean_watchlists(data.get("watchlists"), data.get("watchlist", []))
    # Legacy keys stay populated (first portfolio / watchlist) so older
    # callers keep working; new code should use portfolios/watchlists.
    portfolio = portfolios.get("Main") or next(iter(portfolios.values()), [])
    watchlist = watchlists.get("Main") or next(iter(watchlists.values()), [])
    cash: Dict[str, float] = {}
    raw_cash = data.get("portfolio_cash", {}) or {}
    if isinstance(raw_cash, dict):
        for name, v in raw_cash.items():
            f = _to_float(v)
            if isinstance(name, str) and name.strip() and f is not None:
                cash[name.strip()] = max(0.0, f)
    presets: Dict[str, Dict[str, Any]] = {"undervalued": {}, "dividend": {}}
    raw_presets = data.get("scan_presets", {}) or {}
    if isinstance(raw_presets, dict):
        for tab in ("undervalued", "dividend"):
            tab_presets = raw_presets.get(tab, {}) or {}
            if isinstance(tab_presets, dict):
                presets[tab] = {k: v for k, v in tab_presets.items() if isinstance(k, str) and isinstance(v, dict)}
    log: List[Dict[str, Any]] = []
    for entry in data.get("signal_log", []) or []:
        if isinstance(entry, dict) and entry.get("date") and normalize_ticker(entry.get("ticker", "")):
            log.append({
                "date": str(entry["date"])[:10], "ticker": normalize_ticker(entry["ticker"]),
                "price": _to_float(entry.get("price")), "sticker": _to_float(entry.get("sticker")),
                "signal": str(entry.get("signal", "")), "lean": str(entry.get("lean", "")),
            })
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
    account_types: Dict[str, str] = {}
    raw_types = data.get("portfolio_account_types", {}) or {}
    if isinstance(raw_types, dict):
        for name, v in raw_types.items():
            if isinstance(name, str) and name.strip() and v in ACCOUNT_TYPES:
                account_types[name.strip()] = v
    for name in portfolios:
        account_types.setdefault(name, "Not set")
    raw_rates = data.get("tax_rates", {}) or {}
    tax_rates = {
        "lt_pct": _to_float(raw_rates.get("lt_pct")) or 0.0,
        "st_pct": _to_float(raw_rates.get("st_pct")) or 0.0,
        "ca_pct": _to_float(raw_rates.get("ca_pct")) or 0.0,
    }
    sales: List[Dict[str, Any]] = []
    for entry in data.get("sales_log", []) or []:
        if isinstance(entry, dict) and normalize_ticker(entry.get("ticker", "")) and _parse_date(entry.get("date")):
            sales.append({
                "date": _parse_date(entry["date"]), "ticker": normalize_ticker(entry["ticker"]),
                "portfolio": str(entry.get("portfolio", "")) or "",
                "shares": _to_float(entry.get("shares")), "sale_price": _to_float(entry.get("sale_price")),
                "avg_cost": _to_float(entry.get("avg_cost")), "gain": _to_float(entry.get("gain")),
                "term": str(entry.get("term", "")) or "",
                "purchase_date": _parse_date(entry.get("purchase_date")),
            })
    return {
        "portfolio": portfolio, "watchlist": list(dict.fromkeys(watchlist)),
        "portfolios": portfolios, "watchlists": watchlists, "portfolio_cash": cash,
        "scan_presets": presets, "signal_log": log,
        "portfolio_account_types": account_types, "tax_rates": tax_rates, "sales_log": sales,
        "dividend_income_goal": _to_float(data.get("dividend_income_goal")) or 0.0,
        "dividend_watch": _clean_dividend_watch(data.get("dividend_watch")),
        "dividend_goal_history": _clean_dividend_goal_history(data.get("dividend_goal_history")),
        "holdings_baseline": _clean_holdings_baseline(data.get("holdings_baseline")),
        "notes": notes, "manual": _clean_manual(data.get("manual", {})),
    }


def last_saved_display(path: str = DATA_FILE) -> Optional[str]:
    """When user_data.json was last written on this server, or None if
    there is no saved file here. Used for the persistence warning in
    the UI: Streamlit Cloud's filesystem is ephemeral, so a restart or
    sleep can wipe the file, and the export download is the permanent
    copy. The file's mtime is the save time — no extra field needed."""
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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


def save_portfolio(holdings: List[Dict[str, Any]], name: str = "Main", path: str = DATA_FILE) -> bool:
    """Save one named portfolio, preserving the others. Legacy single-arg
    calls save to Main, as before."""
    data = load_user_data(path)
    portfolios = dict(data.get("portfolios") or {})
    portfolios[name] = [h for h in (_clean_holding(r) for r in holdings) if h is not None]
    data["portfolios"] = portfolios
    if name == "Main":
        data["portfolio"] = portfolios["Main"]
    return save_user_data(data, path)


def delete_portfolio(name: str, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    portfolios = dict(data.get("portfolios") or {})
    if name not in portfolios or len(portfolios) <= 1:
        return False
    portfolios.pop(name, None)
    cash = dict(data.get("portfolio_cash") or {})
    cash.pop(name, None)
    types = dict(data.get("portfolio_account_types") or {})
    types.pop(name, None)
    data["portfolios"] = portfolios
    data["portfolio_cash"] = cash
    data["portfolio_account_types"] = types
    data["portfolio"] = portfolios.get("Main") or next(iter(portfolios.values()), [])
    return save_user_data(data, path)


def save_watchlist(tickers: List[str], name: str = "Main", path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    watchlists = dict(data.get("watchlists") or {})
    watchlists[name] = list(dict.fromkeys(t for t in (normalize_ticker(t) for t in tickers) if t))
    data["watchlists"] = watchlists
    if name == "Main":
        data["watchlist"] = watchlists["Main"]
    return save_user_data(data, path)


def delete_watchlist(name: str, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    watchlists = dict(data.get("watchlists") or {})
    if name not in watchlists or len(watchlists) <= 1:
        return False
    watchlists.pop(name, None)
    data["watchlists"] = watchlists
    data["watchlist"] = watchlists.get("Main") or next(iter(watchlists.values()), [])
    return save_user_data(data, path)


def save_cash(portfolio_name: str, cash: float, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    cash_map = dict(data.get("portfolio_cash") or {})
    cash_map[portfolio_name] = max(0.0, float(cash) if cash else 0.0)
    data["portfolio_cash"] = cash_map
    return save_user_data(data, path)


def save_account_type(portfolio_name: str, account_type: str, path: str = DATA_FILE) -> bool:
    if account_type not in ACCOUNT_TYPES:
        return False
    data = load_user_data(path)
    types = dict(data.get("portfolio_account_types") or {})
    types[portfolio_name] = account_type
    data["portfolio_account_types"] = types
    return save_user_data(data, path)


def delete_account_type(portfolio_name: str, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    types = dict(data.get("portfolio_account_types") or {})
    types.pop(portfolio_name, None)
    data["portfolio_account_types"] = types
    return save_user_data(data, path)


def save_tax_rates(lt_pct: float, st_pct: float, ca_pct: float, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    data["tax_rates"] = {
        "lt_pct": max(0.0, min(100.0, float(lt_pct or 0.0))),
        "st_pct": max(0.0, min(100.0, float(st_pct or 0.0))),
        "ca_pct": max(0.0, min(100.0, float(ca_pct or 0.0))),
    }
    return save_user_data(data, path)


def save_dividend_goal(goal: float, path: str = DATA_FILE) -> bool:
    """Save the annual dividend income goal ($/yr) tracked on the Dashboard."""
    data = load_user_data(path)
    data["dividend_income_goal"] = max(0.0, float(goal or 0.0))
    return save_user_data(data, path)


def save_dividend_watch(watch: Dict[str, Any], path: str = DATA_FILE) -> bool:
    """Save the dividend-watch payout baselines (last seen payout ratio per ticker)."""
    data = load_user_data(path)
    data["dividend_watch"] = watch or {}
    return save_user_data(data, path)


def record_dividend_goal_month(month: str, income: float, goal: float, buys: float = 0.0, raises: float = 0.0, path: str = DATA_FILE) -> bool:
    """Upsert one monthly income-vs-goal point (one entry per month, latest wins); keeps the last 24 months.
    buys/raises split the month's income change into new shares vs dividend raises ($/yr)."""
    data = load_user_data(path)
    history = [e for e in (data.get("dividend_goal_history") or []) if e.get("month") != month]
    history.append({"month": month, "income": max(0.0, float(income or 0.0)), "goal": max(0.0, float(goal or 0.0)),
                    "buys": float(buys or 0.0), "raises": float(raises or 0.0)})
    history.sort(key=lambda e: e.get("month", ""))
    data["dividend_goal_history"] = history[-24:]
    return save_user_data(data, path)


def save_holdings_baseline(baseline: Dict[str, Any], path: str = DATA_FILE) -> bool:
    """Save the last-seen holdings state used to attribute income changes."""
    data = load_user_data(path)
    data["holdings_baseline"] = baseline or {}
    return save_user_data(data, path)


def append_sale(entry: Dict[str, Any], path: str = DATA_FILE) -> bool:
    """Append one recorded sale. Gain/term should already be computed
    by the caller; this only validates and stores the entry."""
    date = _parse_date(entry.get("date"))
    ticker = normalize_ticker(entry.get("ticker", ""))
    if not date or not ticker:
        return False
    data = load_user_data(path)
    sales = list(data.get("sales_log") or [])
    sales.append({
        "date": date, "ticker": ticker, "portfolio": str(entry.get("portfolio", "")) or "",
        "shares": _to_float(entry.get("shares")), "sale_price": _to_float(entry.get("sale_price")),
        "avg_cost": _to_float(entry.get("avg_cost")), "gain": _to_float(entry.get("gain")),
        "term": str(entry.get("term", "")) or "", "purchase_date": _parse_date(entry.get("purchase_date")),
    })
    data["sales_log"] = sales
    return save_user_data(data, path)


def save_scan_preset(tab: str, name: str, values: Dict[str, Any], path: str = DATA_FILE) -> bool:
    if tab not in ("undervalued", "dividend") or not name.strip():
        return False
    data = load_user_data(path)
    presets = data.get("scan_presets") or {"undervalued": {}, "dividend": {}}
    presets.setdefault(tab, {})[name.strip()] = values
    data["scan_presets"] = presets
    return save_user_data(data, path)


def delete_scan_preset(tab: str, name: str, path: str = DATA_FILE) -> bool:
    data = load_user_data(path)
    presets = data.get("scan_presets") or {}
    if name in (presets.get(tab) or {}):
        presets[tab].pop(name, None)
        data["scan_presets"] = presets
        return save_user_data(data, path)
    return False


def append_signal_log(entries: List[Dict[str, Any]], path: str = DATA_FILE) -> bool:
    """Append signal-log entries, one per (date, ticker). Today's entry
    for a ticker is updated (not duplicated) if it is logged again, so
    re-checking during the day refreshes today's price/signal instead
    of piling up rows. Past days are never rewritten. Capped at 2000
    entries (oldest dropped)."""
    if not entries:
        return True
    data = load_user_data(path)
    log = list(data.get("signal_log") or [])
    by_key = {(e.get("date"), e.get("ticker")): i for i, e in enumerate(log)}
    for entry in entries:
        date = str(entry.get("date", ""))[:10]
        ticker = normalize_ticker(entry.get("ticker", ""))
        if not date or not ticker:
            continue
        clean = {
            "date": date, "ticker": ticker,
            "price": _to_float(entry.get("price")), "sticker": _to_float(entry.get("sticker")),
            "signal": str(entry.get("signal", "")), "lean": str(entry.get("lean", "")),
        }
        if (date, ticker) in by_key:
            log[by_key[(date, ticker)]] = clean
        else:
            by_key[(date, ticker)] = len(log)
            log.append(clean)
    data["signal_log"] = log[-2000:]
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


def save_manual(ticker: str, fields: Dict[str, Any], path: str = DATA_FILE) -> bool:
    """Merge manual fields for one ticker. Empty fields dict clears it."""
    data = load_user_data(path)
    manual = data.get("manual", {}) or {}
    clean_ticker = normalize_ticker(ticker)
    if not clean_ticker:
        return False
    if not fields:
        manual.pop(clean_ticker, None)
    else:
        merged = dict(manual.get(clean_ticker, {}))
        merged.update(_clean_manual({clean_ticker: fields}).get(clean_ticker, {}))
        if merged:
            manual[clean_ticker] = merged
        else:
            manual.pop(clean_ticker, None)
    data["manual"] = manual
    return save_user_data(data, path)


def clear_manual(ticker: str, path: str = DATA_FILE) -> bool:
    return save_manual(ticker, {}, path)


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


def parse_imported_data(data: Any) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any], Dict[str, Any]]:
    """Validate an uploaded portfolio JSON. Raises ValueError on bad shape.

    Accepted shape: {"portfolio": [{"ticker", "shares", "avg_cost"}],
    "watchlist": ["AAPL", ...], "notes": {"AAPL": {...}},
    "manual": {"AAPL": {...}}} — portfolio is required, watchlist/notes/
    manual are optional and default to empty.
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
    manual = _clean_manual(data.get("manual", {}))
    return portfolio, list(dict.fromkeys(watchlist)), notes, manual


def parse_imported_full(data: Any) -> Dict[str, Any]:
    """Full import: named portfolios/watchlists if present, else the
    legacy single portfolio/watchlist as Main. Raises ValueError if
    neither a portfolios dict nor a portfolio list is present."""
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with portfolios or a portfolio list")
    portfolios: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(data.get("portfolios"), dict):
        portfolios = _clean_portfolios(data.get("portfolios"), [])
    if not portfolios and isinstance(data.get("portfolio"), list):
        portfolio, _, _, _ = parse_imported_data(data)
        portfolios = {"Main": portfolio}
    if not portfolios:
        raise ValueError("Expected a JSON object with portfolios or a portfolio list")
    watchlists: Dict[str, List[str]] = {}
    if isinstance(data.get("watchlists"), dict):
        watchlists = _clean_watchlists(data.get("watchlists"), [])
    if not watchlists:
        _, watchlist, _, _ = parse_imported_data({"portfolio": [], "watchlist": data.get("watchlist", []) or []})
        watchlists = {"Main": watchlist} if watchlist else {"Main": []}
    _, _, notes, manual = parse_imported_data({"portfolio": [], "watchlist": [], "notes": data.get("notes", {}), "manual": data.get("manual", {})})
    cash: Dict[str, float] = {}
    raw_cash = data.get("portfolio_cash", {}) or {}
    if isinstance(raw_cash, dict):
        for name, v in raw_cash.items():
            f = _to_float(v)
            if isinstance(name, str) and f is not None:
                cash[name.strip()] = max(0.0, f)
    return {
        "portfolios": portfolios, "watchlists": watchlists, "portfolio_cash": cash,
        "notes": notes, "manual": manual,
        "scan_presets": data.get("scan_presets", {}) if isinstance(data.get("scan_presets"), dict) else {},
        "signal_log": data.get("signal_log", []) if isinstance(data.get("signal_log"), list) else [],
    }
