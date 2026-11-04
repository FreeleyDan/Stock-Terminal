"""
valuation.py

The single place every tab gets its numbers from.

  build_snapshot(ticker)  -> everything that does NOT depend on the
      discount-rate / growth-cap settings: latest price, split-adjusted
      current EPS, historical growth, multi-year median P/E, Gatekeeper
      results, dividend-growth streak. Cached on disk, so a long market
      scan can be interrupted and resumed.
  valuation_from_snapshot(snapshot, discount_rate, max_growth_rate)
      -> Sticker Price / MOS / % undervalued / action, computed instantly
      from a snapshot, so moving a slider never triggers a refetch.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

from calculations import (
    calculate_margin_of_safety,
    calculate_sticker_price,
    estimate_growth_rate_from_eps_history,
)
from data_fetcher import (
    DURATION_MAX_DAYS,
    DURATION_MIN_DAYS,
    _build_financial_history,
    _days_between,
    cache_get,
    cache_set,
    fetch_raw_companyfacts,
    get_latest_price,
    get_price_history,
    historical_median_pe,
    normalize_ticker,
)
from gatekeeper import determine_action, evaluate_business_quality

SNAPSHOT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_GROWTH_RATE = 0.15

PERMANENT = "permanent"
TRANSIENT = "transient"


def consecutive_dividend_growth_years(dividend_series: Dict[str, float]) -> int:
    """
    Consecutive year-over-year dividend-per-share increases, newest first.
    Stops at the first non-increase or the first gap that isn't a normal
    ~1-fiscal-year span (a missing year is never skipped over). 0 for a
    company with no (or only one year of) dividend history.
    """
    if not dividend_series or len(dividend_series) < 2:
        return 0
    dates = sorted(dividend_series.keys(), reverse=True)
    streak = 0
    for i in range(len(dates) - 1):
        newer, older = dates[i], dates[i + 1]
        gap = _days_between(older, newer)
        if gap is None or not (DURATION_MIN_DAYS <= gap <= DURATION_MAX_DAYS):
            break
        if dividend_series[newer] > dividend_series[older]:
            streak += 1
        else:
            break
    return streak


def _dividend_metrics(
    dps_series: Dict[str, float],
    current_eps: Optional[float],
    current_price: Optional[float],
    fcf_history: Dict[Any, float],
    shares_history: Dict[Any, float],
) -> Dict[str, Any]:
    """
    Dividend-growth metrics used ONLY by the dividend tab (Miller /
    Van Knapp layer). All values are None when the SEC data does not
    support them — callers must show "—" for None, never 0.

    - dividend_per_share: latest annual DPS from SEC filings.
    - dividend_yield: latest DPS / current price (trailing yield).
    - dividend_growth_5yr / _10yr: CAGR of DPS over ~5 / ~10 years,
      using the filing nearest the target date (within 200 days).
    - payout_ratio: DPS / EPS (Miller: <60% non-utilities, <70-85%
      utilities; the user's screener uses <65% generally).
    - chowder: yield + 5yr dividend growth (Van Knapp / Chowder rule:
      >=8% is favored as likely to deliver ~8% annualized returns).
    - years_to_10pct_yield_on_cost: years for yield-on-cost to reach
      10% at the current 5yr growth rate (Van Knapp "10 by 10").
    - fcf_dividend_coverage: free cash flow per share / DPS.
    """
    out: Dict[str, Any] = {
        "dividend_per_share": None,
        "dividend_yield": None,
        "dividend_growth_5yr": None,
        "dividend_growth_10yr": None,
        "payout_ratio": None,
        "chowder": None,
        "years_to_10pct_yield_on_cost": None,
        "fcf_dividend_coverage": None,
    }
    if not dps_series:
        return out
    dates = sorted(dps_series.keys())
    latest_date_str = dates[-1]
    latest_dps = dps_series[latest_date_str]
    if latest_dps is None or latest_dps <= 0:
        return out
    out["dividend_per_share"] = latest_dps
    latest_days = _days_between("1970-01-01", latest_date_str)
    if latest_days is None:
        return out

    def _cagr_years_ago(years: float) -> Optional[float]:
        target = years * 365.25
        best = None
        best_diff = None
        for d in dates[:-1]:
            days = _days_between("1970-01-01", d)
            if days is None:
                continue
            elapsed = latest_days - days
            if elapsed <= 0:
                continue
            diff = abs(elapsed - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = (d, elapsed)
        if best is None or best_diff is None or best_diff > 200:
            return None
        past = dps_series[best[0]]
        if past is None or past <= 0:
            return None
        actual_years = best[1] / 365.25
        if actual_years <= 0:
            return None
        return (latest_dps / past) ** (1.0 / actual_years) - 1.0

    out["dividend_growth_5yr"] = _cagr_years_ago(5)
    out["dividend_growth_10yr"] = _cagr_years_ago(10)
    if current_eps is not None and current_eps > 0:
        out["payout_ratio"] = latest_dps / current_eps
    if current_price is not None and current_price > 0:
        out["dividend_yield"] = latest_dps / current_price
    if out["dividend_yield"] is not None and out["dividend_growth_5yr"] is not None:
        out["chowder"] = out["dividend_yield"] + out["dividend_growth_5yr"]
        if out["dividend_yield"] >= 0.10:
            out["years_to_10pct_yield_on_cost"] = 0.0
        elif out["dividend_growth_5yr"] > 0:
            out["years_to_10pct_yield_on_cost"] = math.log(0.10 / out["dividend_yield"]) / math.log(1.0 + out["dividend_growth_5yr"])
    fcf0 = (fcf_history or {}).get(0)
    shares0 = (shares_history or {}).get(0)
    if fcf0 is not None and shares0:
        out["fcf_dividend_coverage"] = (fcf0 / shares0) / latest_dps
    return out


def _error(ticker: str, message: str, kind: str) -> Dict[str, Any]:
    return {"ticker": ticker, "error": message, "error_kind": kind}


def _evaluate(ticker: str, prune_unvaluable: bool) -> Tuple[Dict[str, Any], Optional[str]]:
    """Returns (snapshot, where_to_cache) with where_to_cache in
    {"full", "pruned", None}. Transient failures are never cached."""
    raw, status = fetch_raw_companyfacts(ticker)
    if raw is None:
        if status == "not_covered":
            return _error(ticker, "No SEC filings found (not a US 10-K filer, or ticker unknown)", PERMANENT), "full"
        return _error(ticker, "Couldn't reach SEC EDGAR (try again shortly)", TRANSIENT), None

    try:
        unadjusted = _build_financial_history(raw)
    except Exception:
        return _error(ticker, "Couldn't parse SEC data", PERMANENT), "full"

    eps0 = unadjusted["_eps_history"]
    if not eps0:
        return _error(ticker, "No annual EPS in SEC filings", PERMANENT), "full"

    if prune_unvaluable:
        # Scan mode: skip the Yahoo call for companies that can't get a
        # Sticker Price anyway (loss-making, or no positive EPS growth).
        # Splits don't change the sign of EPS or of its growth.
        cur = eps0.get(0)
        growth0 = estimate_growth_rate_from_eps_history(eps0)
        if cur is None or cur <= 0:
            return _error(ticker, "EPS is zero or negative", PERMANENT), "pruned"
        if growth0 is None or growth0 <= 0:
            return _error(ticker, "No positive EPS growth history", PERMANENT), "pruned"

    bundle, pstatus = get_price_history(ticker)
    if bundle is None:
        if pstatus == "empty":
            return _error(ticker, "No price data on Yahoo (delisted, or Yahoo throttling)", PERMANENT), "full"
        return _error(ticker, "Couldn't reach Yahoo Finance (try again shortly)", TRANSIENT), None

    try:
        hist = _build_financial_history(raw, bundle["splits"])
        eps = hist["_eps_history"]
        current_eps = eps.get(0)
        growth_rate = estimate_growth_rate_from_eps_history(eps)
        hist_pe = historical_median_pe(bundle["closes"], hist["_eps_history_by_date"])
        gatekeeper = evaluate_business_quality(hist)
        streak = consecutive_dividend_growth_years(hist["_dividends_per_share_history"])
    except Exception:
        return _error(ticker, "Couldn't compute valuation inputs", PERMANENT), "full"

    div_metrics = _dividend_metrics(
        hist.get("_dividends_per_share_history") or {},
        current_eps,
        bundle["last_close"],
        hist.get("Free_Cash_Flow") or {},
        hist.get("_shares_outstanding_history") or {},
    )
    snapshot = {
        "ticker": ticker,
        "name": raw.get("entityName"),
        "error": None,
        "error_kind": None,
        "current_price": bundle["last_close"],
        "price_date": bundle["last_date"],
        "price_asof": time.time(),
        "current_eps": current_eps,
        "growth_rate": growth_rate,
        "historical_pe": hist_pe,
        "moat_score": gatekeeper["moat_score"],
        "gatekeeper": gatekeeper,
        "dividend_streak": streak,
        "split_count": len(bundle["splits"]),
        # Extra numbers for the ticker card (missing from older cached
        # snapshots — callers must use .get() with a fallback):
        "owner_earnings": (hist.get("_owner_earnings_history") or {}).get(0),
        "shares_history": hist.get("_shares_outstanding_history") or {},
        # Dividend-growth metrics — dividend tab only (Miller / Van Knapp).
        # Missing from older cached snapshots; callers must use .get().
        **div_metrics,
    }
    return snapshot, "full"


def _refresh_price_if_stale(snapshot: Dict[str, Any], key: str, price_max_age: float) -> Dict[str, Any]:
    if snapshot.get("error"):
        return snapshot
    if time.time() - snapshot.get("price_asof", 0) <= price_max_age:
        return snapshot
    latest = get_latest_price(snapshot["ticker"])
    if latest is None:
        return snapshot  # keep the older price rather than fail
    refreshed = dict(snapshot)
    refreshed["current_price"], refreshed["price_date"] = latest
    refreshed["price_asof"] = time.time()
    cache_set(key, refreshed)
    return refreshed


def build_snapshot(
    ticker: str, price_max_age: float = 3600, prune_unvaluable: bool = False
) -> Dict[str, Any]:
    """
    Fundamentals are cached for 24h; the price is re-fetched when older than
    `price_max_age` seconds (cheap call). Never raises. On failure returns
    {"ticker", "error", "error_kind"} -- error_kind "transient" means worth
    retrying, "permanent" means the company can't be valued.
    """
    ticker = normalize_ticker(ticker)
    if not ticker:
        return _error("", "Empty ticker", PERMANENT)
    full_key = f"snapshot:{ticker}"
    pruned_key = f"snapshot_pruned:{ticker}"

    try:
        hit = cache_get(full_key, SNAPSHOT_TTL_SECONDS)
        if hit is not None:
            return _refresh_price_if_stale(hit, full_key, price_max_age)
        if prune_unvaluable:
            pruned = cache_get(pruned_key, SNAPSHOT_TTL_SECONDS)
            if pruned is not None:
                return pruned

        snapshot, store_as = _evaluate(ticker, prune_unvaluable)
        if store_as == "full":
            cache_set(full_key, snapshot)
        elif store_as == "pruned":
            cache_set(pruned_key, snapshot)
        return snapshot
    except Exception:
        return _error(ticker, "Unexpected error while fetching data", TRANSIENT)


def valuation_from_snapshot(
    snapshot: Dict[str, Any],
    discount_rate: float = 0.15,
    max_growth_rate: float = DEFAULT_MAX_GROWTH_RATE,
) -> Dict[str, Any]:
    """
    Sticker/MOS from a snapshot. The growth rate fed to the Sticker Price is
    min(historical EPS CAGR, max_growth_rate): extrapolating a 40% EPS CAGR
    ten years out produces absurd Sticker Prices, and in a market-wide scan
    those outliers would crowd the top of the "most undervalued" list.
    """
    out: Dict[str, Any] = {
        "sticker_price": None,
        "margin_of_safety_price": None,
        "undervalued_pct": None,
        "growth_used": None,
        "action": None,
        "note": None,
    }
    price = snapshot.get("current_price")
    eps = snapshot.get("current_eps")
    growth = snapshot.get("growth_rate")
    pe = snapshot.get("historical_pe")

    if eps is None or eps <= 0:
        out["note"] = "EPS is zero or negative, so no Sticker Price"
    elif growth is None:
        out["note"] = "Not enough EPS history to estimate growth"
    elif growth <= 0:
        out["note"] = "EPS hasn't been growing, so no Sticker Price"
    elif pe is None:
        out["note"] = "No historical P/E (couldn't pair past EPS with past prices)"
    else:
        used = min(growth, max_growth_rate)
        sticker = calculate_sticker_price(eps, used, pe, discount_rate)
        if sticker is None:
            out["note"] = "Sticker Price couldn't be computed"
        else:
            out["growth_used"] = used
            out["sticker_price"] = sticker
            out["margin_of_safety_price"] = calculate_margin_of_safety(sticker)
            if price is not None and sticker > 0:
                out["undervalued_pct"] = (sticker - price) / sticker * 100.0

    out["action"] = determine_action(price, out["sticker_price"], out["margin_of_safety_price"])
    return out
