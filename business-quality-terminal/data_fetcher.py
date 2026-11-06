"""
data_fetcher.py

Pulls annual financial-statement history from SEC EDGAR's free XBRL
"companyfacts" API (data.sec.gov) -- NOT yfinance, because Yahoo's free
statement feed only exposes ~4 years of annual history. yfinance is used
ONLY for price history (current price, split history, and the historical
prices needed to compute a multi-year P/E), since EDGAR has no market data.

EDGAR pitfalls handled on purpose:
  - SEC requires a descriptive User-Agent (app name + real contact email)
    on every request, and asks for <= 10 requests/second (throttled here).
  - XBRL facts live under different unit keys ("USD", "USD/shares",
    "shares") -- mapped per field, never just "USD".
  - Tags can live under different taxonomy namespaces (us-gaap, dei, ...).
  - Only 10-K / 10-K/A facts for annual history; duration facts only
    ~300-400 day spans. 10-Q / 10-Q/A facts (~50-140 day spans) feed
    trailing-12-month values only — the Sticker base, never the Big Five.
  - Restatements: the most recently *filed* value for a period wins.
  - Fiscal year-ends are matched to years_ago (0/1/3/5/10) by nearest date
    within a tolerance window, not by column position.
  - EDGAR has no "Invested Capital" tag -- derived as Equity + Debt - Cash.
  - STOCK SPLITS: per-share figures (EPS, dividends/share) and share counts
    are reported as originally filed, so they are NOT comparable across a
    split. They are adjusted here using each fact's *filed* date and the
    split history from yfinance.

Caching: a small TTL pickle cache on disk (survives restarts, so a long
multi-hour scan can resume). Failed lookups (None) are never cached.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import shutil
import statistics
import threading
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEC_USER_AGENT = "BusinessQualityTerminal Jakob Freele jakobfreele@gmail.com"

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_TICKER_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

VALID_FORMS = {"10-K", "10-K/A"}
DURATION_MIN_DAYS = 300
DURATION_MAX_DAYS = 400

# Quarterly facts (10-Q / 10-Q/A) are used ONLY for trailing-12-month
# values — the Sticker base. A quarter is ~90 days; the wide window
# tolerates 13-week quarters and odd short periods after a year-end change.
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
QUARTER_DURATION_MIN_DAYS = 50
QUARTER_DURATION_MAX_DAYS = 140

TARGET_HORIZONS = [0, 1, 3, 5, 10]
# Flat window (not scaled by horizon): fiscal year-ends are defined by a
# calendar rule, so drift does not accumulate over the years. A wide window
# would silently match a ~8.5-year-old figure as "10 years ago".
HORIZON_TOLERANCE_DAYS = 60

CACHE_TTL_SECONDS = 6 * 60 * 60
DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# SEC asks for <= 10 requests/sec; stay well under it, across all threads.
SEC_MIN_INTERVAL_SECONDS = 0.12

TAG_CANDIDATES: Dict[str, List[str]] = {
    "Net_Income": ["NetIncomeLoss", "ProfitLoss"],
    "EPS": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "Sales": ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "Operating_Cash_Flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Cap_Ex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
    ],
    "Depreciation_Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "Stockholders_Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Long_Term_Debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "Short_Term_Debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "Shares_Outstanding": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
    "Dividends_Per_Share": [
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ],
    # Display-only accuracy metrics (never feed the Sticker, Moat, or a
    # signal): stock-based compensation, interest expense, operating
    # income. SBC is the cash-flow add-back tag most filers report.
    "Share_Based_Compensation": [
        "ShareBasedCompensation",
        "ShareBasedCompensationExpense",
    ],
    "Interest_Expense": ["InterestExpense", "InterestAndDebtExpense"],
    "Operating_Income": ["OperatingIncomeLoss"],
}

METRIC_UNIT: Dict[str, str] = {
    "Net_Income": "USD",
    "EPS": "USD/shares",
    "Sales": "USD",
    "Operating_Cash_Flow": "USD",
    "Cap_Ex": "USD",
    "Depreciation_Amortization": "USD",
    "Stockholders_Equity": "USD",
    "Long_Term_Debt": "USD",
    "Short_Term_Debt": "USD",
    "Cash": "USD",
    "Shares_Outstanding": "shares",
    "Dividends_Per_Share": "USD/shares",
    "Share_Based_Compensation": "USD",
    "Interest_Expense": "USD",
    "Operating_Income": "USD",
}

DURATION_METRICS = {
    "Net_Income",
    "EPS",
    "Sales",
    "Operating_Cash_Flow",
    "Cap_Ex",
    "Depreciation_Amortization",
    "Shares_Outstanding",
    "Dividends_Per_Share",
    "Share_Based_Compensation",
    "Interest_Expense",
    "Operating_Income",
}
INSTANT_METRICS = {"Stockholders_Equity", "Long_Term_Debt", "Short_Term_Debt", "Cash"}

# Metrics that are NOT comparable across a stock split unless adjusted.
PER_SHARE_METRICS = {"EPS", "Dividends_Per_Share"}  # divide by later splits
SHARE_COUNT_METRICS = {"Shares_Outstanding"}  # multiply by later splits


# ---------------------------------------------------------------------------
# Disk cache (pickle: preserves int keys; None results are never cached)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(DISK_CACHE_DIR, f"{digest}.pkl")


def cache_get_with_age(key: str) -> Optional[Tuple[Any, float]]:
    """Returns (data, age_in_seconds) or None if absent/unreadable."""
    try:
        with open(_cache_path(key), "rb") as f:
            payload = pickle.load(f)
        return payload["data"], time.time() - payload["cached_at"]
    except Exception:
        return None


def cache_get(key: str, ttl: float = CACHE_TTL_SECONDS) -> Optional[Any]:
    hit = cache_get_with_age(key)
    if hit is None:
        return None
    data, age = hit
    return data if age <= ttl else None


def cache_set(key: str, data: Any) -> None:
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        path = _cache_path(key)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"cached_at": time.time(), "data": data}, f)
        os.replace(tmp, path)
    except Exception:
        pass  # a cache write failure must never break a fetch


def clear_cache() -> None:
    shutil.rmtree(DISK_CACHE_DIR, ignore_errors=True)


def cached(ttl_seconds: float = CACHE_TTL_SECONDS):
    """TTL disk-cache decorator. Results of None are never cached, so a
    transient network failure can't get stuck for the whole TTL."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            hit = cache_get(key, ttl_seconds)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            if result is not None:
                cache_set(key, result)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# SEC access (throttled), ticker -> CIK
# ---------------------------------------------------------------------------

_sec_lock = threading.Lock()
_sec_last_request = [0.0]


def sec_get(url: str, headers: Dict[str, str], timeout: float = 30):
    """requests.get with a global (cross-thread) SEC rate limit."""
    with _sec_lock:
        wait = SEC_MIN_INTERVAL_SECONDS - (time.monotonic() - _sec_last_request[0])
        if wait > 0:
            time.sleep(wait)
        _sec_last_request[0] = time.monotonic()
    return requests.get(url, headers=headers, timeout=timeout)


def normalize_ticker(ticker: str) -> str:
    """'brk.b' -> 'BRK-B' (the form both SEC and Yahoo use)."""
    return str(ticker or "").strip().upper().replace(".", "-")


@cached()
def _fetch_ticker_map_raw() -> Optional[Dict[str, Any]]:
    try:
        resp = sec_get(TICKER_MAP_URL, SEC_TICKER_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


_cik_map_memo: Dict[str, Any] = {"map": None, "at": 0.0}


def _get_ticker_to_cik() -> Optional[Dict[str, int]]:
    """ticker -> CIK dict, built once from SEC's map and memoized."""
    now = time.time()
    if _cik_map_memo["map"] is not None and now - _cik_map_memo["at"] < CACHE_TTL_SECONDS:
        return _cik_map_memo["map"]
    raw = _fetch_ticker_map_raw()
    if raw is None:
        return None
    mapping: Dict[str, int] = {}
    for entry in raw.values():
        try:
            mapping.setdefault(normalize_ticker(entry.get("ticker", "")), int(entry["cik_str"]))
        except Exception:
            continue
    _cik_map_memo["map"] = mapping
    _cik_map_memo["at"] = now
    return mapping


def get_cik_for_ticker(ticker: str) -> Optional[int]:
    mapping = _get_ticker_to_cik()
    if mapping is None:
        return None
    return mapping.get(normalize_ticker(ticker))


def fetch_raw_companyfacts(ticker: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (payload, status). status is "ok", "not_covered" (SEC has no
    XBRL company facts for this ticker -- permanent), or "transient"
    (network/HTTP problem -- worth retrying later).
    """
    try:
        mapping = _get_ticker_to_cik()
        if mapping is None:
            return None, "transient"
        cik = mapping.get(normalize_ticker(ticker))
        if cik is None:
            return None, "not_covered"
        resp = sec_get(COMPANYFACTS_URL.format(cik=cik), SEC_HEADERS, timeout=30)
        if resp.status_code == 404:
            return None, "not_covered"
        if resp.status_code != 200:
            return None, "transient"
        return resp.json(), "ok"
    except Exception:
        return None, "transient"


# ---------------------------------------------------------------------------
# XBRL extraction helpers
# ---------------------------------------------------------------------------

def _days_between(start_str: str, end_str: str) -> Optional[int]:
    try:
        y1, m1, d1 = (int(x) for x in start_str.split("-"))
        y2, m2, d2 = (int(x) for x in end_str.split("-"))
        return (date(y2, m2, d2) - date(y1, m1, d1)).days
    except Exception:
        return None


def _iter_taxonomy_facts(raw_companyfacts: Dict[str, Any], tag: str):
    """Yield (taxonomy, fact_entry) for `tag` across ALL namespaces."""
    for taxonomy_name, taxonomy_facts in raw_companyfacts.get("facts", {}).items():
        if tag in taxonomy_facts:
            yield taxonomy_name, taxonomy_facts[tag]


def _extract_tag_series_full(
    raw_companyfacts: Dict[str, Any], tag: str, expected_unit: str, is_duration: bool,
    forms: Optional[set] = None, dur_min: int = DURATION_MIN_DAYS, dur_max: int = DURATION_MAX_DAYS,
) -> Dict[str, Tuple[float, str]]:
    """{period_end: (value, filed_date)} after every filter in the module
    docstring (unit key, namespaces, 10-K forms, annual duration, latest
    filing wins). Pass quarterly forms/duration for 10-Q facts."""
    _forms = forms if forms is not None else VALID_FORMS
    best: Dict[str, Dict[str, Any]] = {}
    for _taxonomy, fact_entry in _iter_taxonomy_facts(raw_companyfacts, tag):
        entries = fact_entry.get("units", {}).get(expected_unit)
        if not entries:
            continue
        for item in entries:
            if item.get("form", "") not in _forms:
                continue
            end_date = item.get("end")
            if not end_date:
                continue
            if is_duration:
                start_date = item.get("start")
                if not start_date:
                    continue
                span = _days_between(start_date, end_date)
                if span is None or not (dur_min <= span <= dur_max):
                    continue
            filed = item.get("filed", "")
            existing = best.get(end_date)
            if existing is None or filed > existing.get("filed", ""):
                best[end_date] = item
    return {
        end: (fact["val"], fact.get("filed", ""))
        for end, fact in best.items()
        if isinstance(fact.get("val"), (int, float)) and not isinstance(fact.get("val"), bool)
    }


def _extract_tag_series(
    raw_companyfacts: Dict[str, Any], tag: str, expected_unit: str, is_duration: bool
) -> Dict[str, float]:
    full = _extract_tag_series_full(raw_companyfacts, tag, expected_unit, is_duration)
    return {end: val for end, (val, _filed) in full.items()}


def _split_factor(filed_date: str, splits: Optional[List[Tuple[str, float]]]) -> float:
    """Product of split ratios that happened AFTER `filed_date`."""
    factor = 1.0
    for split_date, ratio in splits or []:
        if ratio > 0 and split_date > filed_date:
            factor *= ratio
    return factor


def _get_best_tag_series_with_tag(
    raw_companyfacts: Dict[str, Any],
    metric_key: str,
    splits: Optional[List[Tuple[str, float]]] = None,
    quarterly: bool = False,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Best-tag series, split-adjusted where the metric is per-share or
    a share count. Returns (series, primary_tag). With quarterly=True
    the series is built from 10-Q facts instead of 10-K facts.

    BUG FIX (2026-11-04): the old version returned the FIRST candidate
    tag with any data. That silently truncated history when the first
    tag only exists for recent years — e.g. AAPL Sales under "Revenues"
    exists only from ~2015, while "SalesRevenueNet" has the full history,
    so early Sales horizons looked missing and Big Five passes / Moat
    were understated. Now every candidate tag's full series is built,
    tags are ranked by coverage (most fiscal year-ends, then latest end,
    then candidate order), and values are merged date-by-date with the
    fuller tag winning each date and thinner tags filling gaps only.
    The primary tag (most coverage) is returned so the UI can show
    which SEC tag supplied each metric.
    """
    expected_unit = METRIC_UNIT[metric_key]
    is_duration = metric_key in DURATION_METRICS
    _forms = QUARTERLY_FORMS if quarterly else VALID_FORMS
    _dur_min = QUARTER_DURATION_MIN_DAYS if quarterly else DURATION_MIN_DAYS
    _dur_max = QUARTER_DURATION_MAX_DAYS if quarterly else DURATION_MAX_DAYS

    per_tag: List[Tuple[str, Dict[str, float]]] = []
    for tag in TAG_CANDIDATES[metric_key]:
        full = _extract_tag_series_full(raw_companyfacts, tag, expected_unit, is_duration, _forms, _dur_min, _dur_max)
        if not full:
            continue
        series: Dict[str, float] = {}
        for end, (val, filed) in full.items():
            factor = _split_factor(filed, splits)
            if metric_key in PER_SHARE_METRICS:
                series[end] = val / factor
            elif metric_key in SHARE_COUNT_METRICS:
                series[end] = val * factor
            else:
                series[end] = val
        per_tag.append((tag, series))
    if not per_tag:
        return {}, None
    # Rank: most year-ends first, then latest end date, then candidate order.
    order = {tag: i for i, tag in enumerate(TAG_CANDIDATES[metric_key])}
    per_tag.sort(key=lambda ts: (-len(ts[1]), max(ts[1]) if ts[1] else "", order.get(ts[0], 99)))
    primary_tag = per_tag[0][0]
    merged: Dict[str, float] = {}
    for _tag, series in per_tag:
        for end, val in series.items():
            merged.setdefault(end, val)
    return merged, primary_tag


def _get_best_tag_series(
    raw_companyfacts: Dict[str, Any],
    metric_key: str,
    splits: Optional[List[Tuple[str, float]]] = None,
    quarterly: bool = False,
) -> Dict[str, float]:
    """Best-tag series (see _get_best_tag_series_with_tag)."""
    series, _tag = _get_best_tag_series_with_tag(raw_companyfacts, metric_key, splits, quarterly)
    return series


def _ttm_from_series(
    annual_by_date: Dict[str, float], quarterly_by_date: Dict[str, float]
) -> Tuple[Optional[float], Optional[str], bool]:
    """Trailing-12-month value from an annual {end: value} series and a
    quarterly {end: value} series. Returns (value, as_of_end, used_ttm).
    TTM = latest annual + latest quarter − same quarter a year earlier
    (end date within 45 days of latest-quarter-minus-365-days). When the
    latest quarter is not past the latest annual year-end — the 10-K was
    just filed — TTM is just the annual value (used_ttm False). None when
    the year-ago quarter is missing, so a partial TTM never ships."""
    if not annual_by_date:
        return None, None, False
    _a_end = max(annual_by_date)
    _a_val = annual_by_date[_a_end]
    if not quarterly_by_date:
        return _a_val, _a_end, False
    _q_end = max(quarterly_by_date)
    if _q_end <= _a_end:
        return _a_val, _a_end, False
    try:
        _q_end_d = date.fromisoformat(_q_end)
    except ValueError:
        return _a_val, _a_end, False
    _target = _q_end_d - timedelta(days=365)
    _best: Optional[str] = None
    _best_diff: Optional[int] = None
    for _d in quarterly_by_date:
        try:
            _dd = date.fromisoformat(_d)
        except ValueError:
            continue
        _diff = abs((_dd - _target).days)
        if _diff <= 45 and (_best_diff is None or _diff < _best_diff):
            _best, _best_diff = _d, _diff
    if _best is None:
        return _a_val, _a_end, False
    return _a_val + quarterly_by_date[_q_end] - quarterly_by_date[_best], _q_end, True


def _map_series_to_horizons(series: Dict[str, float]) -> Dict[int, float]:
    """{end_date: value} -> {years_ago: value} by nearest fiscal year-end
    within HORIZON_TOLERANCE_DAYS (not by raw column position)."""
    if not series:
        return {}
    sorted_dates = sorted(series.keys(), reverse=True)
    most_recent = sorted_dates[0]
    result: Dict[int, float] = {0: series[most_recent]}

    for target_years in TARGET_HORIZONS:
        if target_years == 0:
            continue
        target_days = target_years * 365.25
        best_date, best_diff = None, None
        for candidate in sorted_dates:
            days_back = _days_between(candidate, most_recent)
            if days_back is None:
                continue
            diff = abs(days_back - target_days)
            if diff <= HORIZON_TOLERANCE_DAYS and (best_diff is None or diff < best_diff):
                best_diff, best_date = diff, candidate
        if best_date is not None:
            result[target_years] = series[best_date]
    return result


def _combine_horizon_dicts(a: Dict[int, float], b: Dict[int, float], op) -> Dict[int, float]:
    result = {}
    for years_ago, a_val in a.items():
        b_val = b.get(years_ago)
        if b_val is None:
            continue
        try:
            combined = op(a_val, b_val)
            if combined is not None:
                result[years_ago] = combined
        except Exception:
            continue
    return result


def _build_financial_history(
    raw_companyfacts: Dict[str, Any], splits: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    """
    raw companyfacts payload -> the financial_history dict gatekeeper.py
    consumes. Pass `splits` ([(iso_date, ratio)]) to split-adjust EPS,
    dividends/share and share counts.
    """
    raw_series: Dict[str, Dict[str, float]] = {}
    tag_used: Dict[str, str] = {}
    for metric in TAG_CANDIDATES:
        series, tag = _get_best_tag_series_with_tag(raw_companyfacts, metric, splits)
        raw_series[metric] = series
        if tag is not None:
            tag_used[metric] = tag
    by_horizon = {m: _map_series_to_horizons(s) for m, s in raw_series.items()}

    net_income = by_horizon["Net_Income"]
    eps = by_horizon["EPS"]
    sales = by_horizon["Sales"]
    op_cash_flow = by_horizon["Operating_Cash_Flow"]
    cap_ex = by_horizon["Cap_Ex"]
    dep_amort = by_horizon["Depreciation_Amortization"]
    equity = by_horizon["Stockholders_Equity"]
    lt_debt = by_horizon["Long_Term_Debt"]
    st_debt = by_horizon["Short_Term_Debt"]
    cash = by_horizon["Cash"]
    shares = by_horizon["Shares_Outstanding"]

    free_cash_flow = _combine_horizon_dicts(op_cash_flow, cap_ex, lambda a, b: a - b)
    ni_plus_da = _combine_horizon_dicts(net_income, dep_amort, lambda a, b: a + b)
    owner_earnings = _combine_horizon_dicts(ni_plus_da, cap_ex, lambda a, b: a - b)

    total_debt = {y: v + st_debt.get(y, 0.0) for y, v in lt_debt.items()}
    equity_plus_debt = _combine_horizon_dicts(equity, total_debt, lambda a, b: a + b)
    invested_capital = _combine_horizon_dicts(equity_plus_debt, cash, lambda a, b: a - b)

    # Non-positive invested capital (e.g. buyback-driven negative equity at
    # MCD/SBUX) makes ROIC meaningless, so those horizons are dropped —
    # but they are now RECORDED in _roic_dropped instead of vanishing
    # silently, so the UI can say why ROIC (and the Moat points it feeds)
    # is missing for those horizons.
    roic: Dict[int, float] = {}
    roic_dropped: Dict[int, str] = {}
    for y, ni in net_income.items():
        ic = invested_capital.get(y)
        if ic is not None and ic > 0:
            roic[y] = ni / ic
        elif ic is not None:
            roic_dropped[y] = (
                "invested capital ≤ 0 (equity + debt − cash), so ROIC is meaningless — "
                "common at buyback-heavy companies with negative equity"
            )
        else:
            roic_dropped[y] = (
                "invested capital unavailable (missing equity, debt, or cash in SEC filings)"
            )

    equity_per_share = _combine_horizon_dicts(equity, shares, lambda a, b: a / b if b else None)

    # Display-only accuracy metrics (never feed the Sticker, Moat, or a
    # signal). SBC-adjusted FCF subtracts stock-based compensation from
    # free cash flow — SBC is a real cost to owners (dilution), even
    # though it never leaves as cash. Interest coverage is operating
    # income / interest expense. Maintenance capex is estimated as D&A
    # (a rough proxy, not a reported number — the app labels it as such).
    sbc = by_horizon["Share_Based_Compensation"]
    interest = by_horizon["Interest_Expense"]
    op_income = by_horizon["Operating_Income"]
    sbc_adjusted_fcf = _combine_horizon_dicts(free_cash_flow, sbc, lambda a, b: a - b)
    interest_coverage = _combine_horizon_dicts(
        op_income, interest, lambda a, b: a / b if b and b > 0 else None
    )

    # Audit series: full {fiscal_year_end: value} series (split-adjusted)
    # for the 10-year audit table — the horizon dicts above only keep
    # 0/1/3/5/10 years ago. ROIC needs invested capital per date.
    def _combine_date_series(a: Dict[str, float], b: Dict[str, float], op) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for end, a_val in (a or {}).items():
            b_val = (b or {}).get(end)
            if b_val is None:
                continue
            try:
                val = op(a_val, b_val)
            except Exception:
                continue
            if val is not None:
                out[end] = val
        return out

    _ic_by_date: Dict[str, float] = {}
    for end, eq in (raw_series.get("Stockholders_Equity") or {}).items():
        lt = (raw_series.get("Long_Term_Debt") or {}).get(end, 0.0)
        st = (raw_series.get("Short_Term_Debt") or {}).get(end, 0.0)
        cs = (raw_series.get("Cash") or {}).get(end, 0.0)
        try:
            _ic_by_date[end] = float(eq) + float(lt) + float(st) - float(cs)
        except (TypeError, ValueError):
            continue
    _roic_by_date = {
        end: ni / _ic_by_date[end]
        for end, ni in (raw_series.get("Net_Income") or {}).items()
        if end in _ic_by_date and _ic_by_date[end] > 0
    }

    # Trailing-12-month values from 10-Q facts: the Sticker base (current
    # EPS, owner earnings, FCF) uses the latest four quarters instead of
    # the last completed fiscal year, so a Sticker isn't a year stale the
    # day after a 10-K. Big Five, growth, and P/E stay annual. Each
    # component's TTM is computed from its own quarterly series, so the
    # windows can differ by a quarter when a filing is missing — the
    # headline `used` flag tracks the EPS TTM, which drives the Sticker.
    _ttm_metrics = ("EPS", "Net_Income", "Depreciation_Amortization", "Cap_Ex", "Operating_Cash_Flow")
    _q_series = {m: _get_best_tag_series(raw_companyfacts, m, splits, quarterly=True) for m in _ttm_metrics}
    _ttm_vals: Dict[str, Any] = {}
    _ttm_asof: Optional[str] = None
    _ttm_used = False
    for _m in _ttm_metrics:
        _val, _asof, _used = _ttm_from_series(raw_series.get(_m) or {}, _q_series.get(_m) or {})
        _ttm_vals[_m] = _val
        if _m == "EPS":
            _ttm_used = _used
            _ttm_asof = _asof
    _ttm_oe: Optional[float] = None
    if (_ttm_vals.get("Net_Income") is not None and _ttm_vals.get("Depreciation_Amortization") is not None
            and _ttm_vals.get("Cap_Ex") is not None):
        _ttm_oe = _ttm_vals["Net_Income"] + _ttm_vals["Depreciation_Amortization"] - _ttm_vals["Cap_Ex"]
    _ttm_fcf: Optional[float] = None
    if _ttm_vals.get("Operating_Cash_Flow") is not None and _ttm_vals.get("Cap_Ex") is not None:
        _ttm_fcf = _ttm_vals["Operating_Cash_Flow"] - _ttm_vals["Cap_Ex"]

    return {
        "ROIC": roic,
        "Equity_Per_Share": equity_per_share,
        "EPS": eps,
        "Sales": sales,
        "Free_Cash_Flow": free_cash_flow,
        "Long_Term_Debt": {0: lt_debt[0]} if 0 in lt_debt else {},
        "Owner_Earnings": {0: owner_earnings[0]} if 0 in owner_earnings else {},
        "Cap_Ex": {0: cap_ex[0]} if 0 in cap_ex else {},
        "Net_Income": {0: net_income[0]} if 0 in net_income else {},
        "Share_Based_Compensation": {0: sbc[0]} if 0 in sbc else {},
        "Interest_Expense": {0: interest[0]} if 0 in interest else {},
        "Operating_Income": {0: op_income[0]} if 0 in op_income else {},
        "SBC_Adjusted_FCF": {0: sbc_adjusted_fcf[0]} if 0 in sbc_adjusted_fcf else {},
        "Total_Debt": {0: total_debt[0]} if 0 in total_debt else {},
        "_sbc_adjusted_fcf_history": sbc_adjusted_fcf,
        "_interest_coverage_history": interest_coverage,
        "_total_debt_history": total_debt,
        "_maintenance_capex_history": dep_amort,
        "_audit_series": {
            "ROIC": _roic_by_date,
            "Equity_Per_Share": _combine_date_series(
                raw_series.get("Stockholders_Equity"), raw_series.get("Shares_Outstanding"),
                lambda a, b: a / b if b else None,
            ),
            "EPS": raw_series.get("EPS", {}),
            "Sales": raw_series.get("Sales", {}),
            "Free_Cash_Flow": _combine_date_series(
                raw_series.get("Operating_Cash_Flow"), raw_series.get("Cap_Ex"),
                lambda a, b: a - b,
            ),
        },
        "_owner_earnings_history": owner_earnings,
        "_cap_ex_history": cap_ex,
        "_net_income_history": net_income,
        "_eps_history": eps,
        "_shares_outstanding_history": shares,
        "_invested_capital_history": invested_capital,
        "_roic_dropped": roic_dropped,
        "_tag_used": tag_used,
        # Trailing-12-month Sticker inputs (see above) and the latest
        # fiscal year-end — how old the annual numbers are.
        "_ttm": {
            "eps": _ttm_vals.get("EPS"),
            "owner_earnings": _ttm_oe,
            "free_cash_flow": _ttm_fcf,
            "net_income": _ttm_vals.get("Net_Income"),
            "used": _ttm_used,
            "as_of": _ttm_asof,
            "annual_eps": (by_horizon.get("EPS") or {}).get(0),
        },
        "_last_fy_end": max(raw_series["EPS"].keys()) if raw_series.get("EPS") else None,
        # Full {fiscal_year_end: value} series (split-adjusted), not
        # horizon-mapped: dividend streaks need every year, and the
        # historical P/E needs each year's EPS paired with its own price.
        "_dividends_per_share_history": raw_series.get("Dividends_Per_Share", {}),
        "_eps_history_by_date": raw_series.get("EPS", {}),
    }


def get_financial_history(ticker: str) -> Optional[Dict[str, Any]]:
    """Convenience wrapper (no split adjustment). Never raises."""
    raw, status = fetch_raw_companyfacts(ticker)
    if raw is None:
        return None
    try:
        return _build_financial_history(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prices (yfinance): history + splits, latest price, historical P/E
# ---------------------------------------------------------------------------

PRICE_HISTORY_YEARS = 16  # enough to cover splits since a 10-year-old filing


def _dividends_from_history(hist: Any) -> List[Tuple[str, float]]:
    """Dividend payments (date, $/share) from a yfinance history frame,
    most recent last; keeps the last 12. The frame already carries the
    Dividends column when actions=True, so this costs no extra fetch."""
    out: List[Tuple[str, float]] = []
    try:
        if "Dividends" not in hist.columns:
            return out
        for idx, v in zip(hist.index, hist["Dividends"]):
            v = float(v)
            if v == v and v > 0:
                out.append((idx.strftime("%Y-%m-%d"), v))
    except Exception:
        return out
    return out[-12:]


def _ttm_dividends(payments: List[Tuple[str, float]]) -> Optional[float]:
    """Trailing-12-month dividends per share from [(date, $/share)].
    The payment frequency is read from the median gap between recent
    payments (monthly -> last 12, quarterly -> last 4, semiannual ->
    last 2, annual -> last 1); when the frequency can't be established
    the sum of payments in the 365 days up to the latest payment is
    used. A plain 365-day window would catch five quarterly payments
    whenever the year-ago distribution lands on the boundary, so the
    count is set by frequency, not by the window. None when there are
    no payments."""
    if not payments:
        return None
    _parsed: List[Tuple[Any, float]] = []
    for d, v in payments:
        try:
            _parsed.append((date.fromisoformat(d), v))
        except ValueError:
            continue
    if not _parsed:
        return None
    _parsed.sort()
    if len(_parsed) >= 3:
        _gaps = sorted((_parsed[i + 1][0] - _parsed[i][0]).days for i in range(len(_parsed) - 1))
        _med = _gaps[len(_gaps) // 2]
        _n: Optional[int] = None
        if 20 <= _med <= 40:
            _n = 12
        elif 75 <= _med <= 110:
            _n = 4
        elif 150 <= _med <= 200:
            _n = 2
        elif 300 <= _med <= 400:
            _n = 1
        if _n is not None:
            return sum(v for _d, v in _parsed[-_n:])
    latest = _parsed[-1][0]
    cutoff = latest - timedelta(days=365)
    total = sum(v for d, v in _parsed if d > cutoff)
    return total if total > 0 else None


def get_price_history(ticker: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    One yfinance call -> daily (split-adjusted, NOT dividend-adjusted)
    closes + the stock-split history. Returns (bundle, status) with status
    "ok", "empty" (Yahoo has no data -- delisted/unknown) or "error".
    bundle = {"closes": {date: close}, "splits": [(date, ratio)],
              "last_close": float, "last_date": str,
              "dividends": [(date, $/share)], "ttm_dividends": float|None}
    """
    try:
        import yfinance as yf

        start = (date.today() - timedelta(days=365 * PRICE_HISTORY_YEARS)).isoformat()
        hist = yf.Ticker(normalize_ticker(ticker)).history(
            start=start, auto_adjust=False, actions=True
        )
        if hist is None or hist.empty:
            return None, "empty"
        closes = {
            idx.strftime("%Y-%m-%d"): float(c)
            for idx, c in zip(hist.index, hist["Close"])
            if c == c  # drop NaN
        }
        if not closes:
            return None, "empty"
        splits: List[Tuple[str, float]] = []
        if "Stock Splits" in hist.columns:
            for idx, v in zip(hist.index, hist["Stock Splits"]):
                v = float(v)
                if v == v and v > 0 and v != 1.0:
                    splits.append((idx.strftime("%Y-%m-%d"), v))
        dividends = _dividends_from_history(hist)
        last_date = max(closes)
        return (
            {"closes": closes, "splits": splits, "dividends": dividends,
             "ttm_dividends": _ttm_dividends(dividends),
             "last_close": closes[last_date], "last_date": last_date},
            "ok",
        )
    except Exception:
        return None, "error"


def get_latest_price(ticker: str) -> Optional[Tuple[float, str]]:
    """Cheap latest close -> (price, date), or None on failure."""
    try:
        import yfinance as yf

        hist = yf.Ticker(normalize_ticker(ticker)).history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        closes = [(idx.strftime("%Y-%m-%d"), float(c)) for idx, c in zip(hist.index, hist["Close"]) if c == c]
        if not closes:
            return None
        last_date, price = max(closes)
        return price, last_date
    except Exception:
        return None


def _nearest_price_on_or_before(
    price_by_date: Dict[str, float], target_date_str: str, max_lookback_days: int = 10
) -> Optional[float]:
    """Close on target date or the nearest prior trading day."""
    try:
        y, m, d = (int(x) for x in target_date_str.split("-"))
        target = date(y, m, d)
        for offset in range(0, max_lookback_days + 1):
            candidate = (target - timedelta(days=offset)).isoformat()
            if candidate in price_by_date:
                return price_by_date[candidate]
        return None
    except Exception:
        return None


def historical_median_pe(
    closes: Dict[str, float], eps_by_date: Dict[str, float]
) -> Optional[float]:
    """
    Multi-year historical P/E: each past fiscal year's (split-adjusted) EPS
    paired with the close on/near THAT year-end, then the median ratio.
    Median (not mean) so one year of near-zero EPS can't blow the average
    up. Loss years are skipped. None if no year can be paired.

    Why not today's P/E: if "historical PE" is today's price/EPS, Sticker
    Price becomes proportional to today's price (while under the growth
    cap) and "% undervalued" stops depending on price at all.
    """
    ratios = []
    for end_date, eps_val in (eps_by_date or {}).items():
        try:
            eps_val = float(eps_val)
        except (TypeError, ValueError):
            continue
        if eps_val <= 0:
            continue
        price = _nearest_price_on_or_before(closes, end_date)
        if price is not None and price > 0:
            ratios.append(price / eps_val)
    if not ratios:
        return None
    return statistics.median(ratios)
