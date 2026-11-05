"""
app.py

Business Quality Terminal -- Rule #1 Investing / Buffett-style valuation.

Tabs:
  1. Ticker Lookup        any SEC-registered company, full Gatekeeper checklist
  2. Portfolio            holdings you track, with BUY / NEAR BUY / HOLD /
                          REVIEW / SELL signals (BUY = add to a holding)
  3. Watchlist            tickers you're waiting on, with BUY / NEAR BUY signals
  4. Top 25 Undervalued   market scan (S&P 500, every NYSE+Nasdaq stock, or custom)
  5. Top 25 + Dividend Growers   same scan, filtered by consecutive years of
                          dividend growth

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from calculations import calculate_ddm_value, calculate_sticker_price

try:
    from calculations import median_yoy_growth
except ImportError:  # deployed calculations.py can lag the repo behind app.py
    def median_yoy_growth(series_by_date: Dict[str, float]) -> Optional[float]:
        """
        Median year-over-year growth across consecutive fiscal year-ends.

        Local copy of calculations.median_yoy_growth, used only when the
        deployed calculations.py predates it — so a stale deploy can never
        break app startup again. Display only: it never feeds the Sticker,
        Moat, or a signal.
        """
        try:
            dated: List[tuple] = []
            for end, val in (series_by_date or {}).items():
                y, m, d = (int(x) for x in str(end).split("-"))
                dated.append((date(y, m, d), float(val)))
            dated.sort()
            steps: List[float] = []
            for (d1, v1), (d2, v2) in zip(dated, dated[1:]):
                days = (d2 - d1).days
                if days < 200 or days > 550 or v1 <= 0 or v2 <= 0:
                    continue
                steps.append((v2 / v1) ** (365.25 / days) - 1.0)
            if len(steps) < 2:
                return None
            return statistics.median(steps)
        except (TypeError, ValueError):
            return None
from data_fetcher import _build_financial_history, clear_cache, fetch_raw_companyfacts, get_price_history, normalize_ticker
from holdings import NON_13F, check_holdings
from gatekeeper import BIG_FIVE_METRICS, HORIZONS
from screener import big_five_pass_count, dividend_safety_score, rank_snapshots, scan_snapshots, stepped_rank_snapshots
from signals import BUY_RANK, PORTFOLIO_RANK, buy_signal, portfolio_signal, sell_signal
from storage import (
    ACCOUNT_TYPES,
    MANUAL_SECTOR_OPTIONS,
    append_sale,
    append_signal_log,
    clear_manual,
    delete_portfolio,
    delete_scan_preset,
    delete_watchlist,
    last_saved_display,
    load_last_scan,
    load_user_data,
    save_account_type,
    save_cash,
    save_last_scan,
    save_manual,
    save_note,
    save_portfolio,
    save_scan_preset,
    save_tax_rates,
    save_user_data,
    save_watchlist,
)
from universe import get_all_us_listed_tickers, get_sp500_tickers
from valuation import DEFAULT_MAX_GROWTH_RATE, build_snapshot, valuation_from_snapshot

try:
    from storage import save_dividend_goal
except ImportError:  # deployed storage.py can lag the repo behind app.py
    def save_dividend_goal(goal: float) -> bool:
        """Local copy of storage.save_dividend_goal, used only when the
        deployed storage.py predates it — so a stale deploy can never
        break app startup again. load/save with no path use storage's
        own default file, the same one the real function uses."""
        _data = load_user_data()
        _data["dividend_income_goal"] = max(0.0, float(goal or 0.0))
        return save_user_data(_data)

try:
    from storage import record_dividend_goal_month, save_dividend_watch
except ImportError:  # deployed storage.py can lag the repo behind app.py
    def save_dividend_watch(watch: Dict[str, Any]) -> bool:
        """Local copy of storage.save_dividend_watch (see above)."""
        _data = load_user_data()
        _data["dividend_watch"] = watch or {}
        return save_user_data(_data)

    def record_dividend_goal_month(month: str, income: float, goal: float, buys: float = 0.0, raises: float = 0.0) -> bool:
        """Local copy of storage.record_dividend_goal_month (see above)."""
        _data = load_user_data()
        _history = [e for e in (_data.get("dividend_goal_history") or []) if e.get("month") != month]
        _history.append({"month": month, "income": max(0.0, float(income or 0.0)), "goal": max(0.0, float(goal or 0.0)),
                         "buys": float(buys or 0.0), "raises": float(raises or 0.0)})
        _history.sort(key=lambda e: e.get("month", ""))
        _data["dividend_goal_history"] = _history[-24:]
        return save_user_data(_data)

try:
    from storage import save_holdings_baseline
except ImportError:  # deployed storage.py can lag the repo behind app.py
    def save_holdings_baseline(baseline: Dict[str, Any]) -> bool:
        """Local copy of storage.save_holdings_baseline (see above)."""
        _data = load_user_data()
        _data["holdings_baseline"] = baseline or {}
        return save_user_data(_data)

st.set_page_config(page_title="Business Quality Terminal", layout="wide")

# Restrained styling: no animation, color only for valuation / signal state.
st.markdown(
    """
    <style>
    .metric-box { border: 1px solid #333; border-radius: 6px; padding: 14px 18px; text-align: center; }
    .metric-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 600; font-family: "Courier New", monospace; }
    .metric-green { color: #2e7d32; }
    .metric-amber { color: #b98600; }
    .metric-red { color: #c62828; }
    .metric-neutral { color: inherit; }
    </style>
    """,
    unsafe_allow_html=True,
)

METRIC_DISPLAY_NAMES = {
    "ROIC": "ROIC",
    "Equity_Per_Share": "Equity/Share",
    "EPS": "EPS",
    "Sales": "Sales",
    "Free_Cash_Flow": "Free Cash Flow",
}
ACTION_COLOR_CLASS = {
    "UNDERVALUED_BUY": "metric-green",
    "FAIR_VALUE_HOLD": "metric-amber",
    "OVERPRICED_AVOID": "metric-red",
}
SIGNAL_COLORS = {
    "BUY": "#2e7d32",
    "NEAR BUY": "#b98600",
    "CHEAP, WEAK MOAT": "#b98600",
    "SELL": "#c62828",
    "REVIEW": "#b98600",
    "HOLD": "#2e7d32",
}

MOAT_TYPE_OPTIONS = [
    "Not rated yet",
    "Brand",
    "Switching costs",
    "Network effect",
    "Low-cost advantage",
    "Regulatory / patent",
    "None / unclear",
]

LOOKUP_PRICE_MAX_AGE = 60 * 60
TRACKING_PRICE_MAX_AGE = 15 * 60  # portfolio / watchlist: fresher prices

DISCLAIMER = (
    "Signals are mechanical screens on this model's numbers, not financial "
    "advice \u2014 check the reasons and do your own research."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ticker_list(raw_text: str) -> List[str]:
    tickers = [normalize_ticker(c) for c in raw_text.replace("\n", ",").split(",")]
    return list(dict.fromkeys(t for t in tickers if t))


def _cell(value: Any) -> Optional[Any]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value


def _num(value: Any) -> Optional[float]:
    value = _cell(value)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def money(value: Optional[float]) -> str:
    return "\u2014" if value is None else f"${value:,.2f}"


def _signed_money(value: Optional[float]) -> str:
    """$1,234.56 with an explicit +/\u2212 sign (for income attribution)."""
    if value is None:
        return "\u2014"
    if value > 0:
        return f"+${value:,.2f}"
    if value < 0:
        return f"\u2212${abs(value):,.2f}"
    return "$0.00"


def status_icon(status: str) -> str:
    return {"pass": "\u2705", "fail": "\u274c", "not_yet_determined": "\u2014"}.get(status, "\u2014")


def render_metric_box(label: str, value_str: str, color_class: str) -> str:
    return (
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {color_class}">{value_str}</div></div>'
    )


def price_color_class(price, sticker, mos) -> str:
    if price is None or sticker is None or mos is None:
        return "metric-neutral"
    if price <= mos:
        return "metric-green"
    if price <= sticker:
        return "metric-amber"
    return "metric-red"


def _longest_cagr(gatekeeper: Dict[str, Any], metric: str) -> Optional[float]:
    """Longest-horizon CAGR for a Big Five metric, or None."""
    horizons = (gatekeeper.get("big_five") or {}).get(metric) or {}
    for h in (10, 5, 3, 1):
        entry = horizons.get(h) or {}
        if entry.get("cagr") is not None:
            return entry["cagr"]
    return None


def _share_change_pct(shares_history: Dict[Any, Any], years_ago: int) -> Optional[float]:
    """% change in shares outstanding from `years_ago` to now.

    Positive = dilution (more shares), negative = buybacks (fewer shares).
    Keys may be ints (from a fresh snapshot) or strings (from a reload).
    """
    if not shares_history:
        return None
    def _get(years: int) -> Optional[float]:
        for key in (years, str(years)):
            if key in shares_history and shares_history[key]:
                return float(shares_history[key])
        return None
    now, past = _get(0), _get(years_ago)
    if now is None or past is None or past == 0:
        return None
    return (now / past - 1.0) * 100.0


def _notes_for(ticker: str) -> Dict[str, Any]:
    try:
        return (load_user_data().get("notes") or {}).get(normalize_ticker(ticker), {}) or {}
    except Exception:
        return {}


def _manual_for(ticker: str) -> Dict[str, Any]:
    """Manual dividend data for a ticker, or {} — never raises."""
    try:
        return (load_user_data().get("manual") or {}).get(normalize_ticker(ticker), {}) or {}
    except Exception:
        return {}


def _apply_manual_to_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of a snapshot with the user's manual dividend fields
    applied. Manual fields are tracked separately in the copy as
    "manual_fields" (a list of field names) so the table can flag them
    with ✏️. The original session-state scan is never mutated.
    """
    manual = _manual_for(snap.get("ticker", ""))
    if not manual:
        out = dict(snap)
        out["manual_fields"] = []
        return out
    out = dict(snap)
    out["manual_fields"] = sorted(manual.keys())
    price = out.get("current_price")
    eps = out.get("current_eps")
    if "dividend_per_share" in manual:
        out["dividend_per_share"] = manual["dividend_per_share"]
        if price:
            out["dividend_yield"] = manual["dividend_per_share"] / price
        if "payout_ratio" not in manual and eps:
            out["payout_ratio"] = manual["dividend_per_share"] / eps
    if "payout_ratio" in manual:
        out["payout_ratio"] = manual["payout_ratio"]
    if "dividend_growth_5yr" in manual:
        out["dividend_growth_5yr"] = manual["dividend_growth_5yr"]
    if "dividend_streak" in manual:
        out["dividend_streak"] = manual["dividend_streak"]
    if "fcf_dividend_coverage" in manual:
        out["fcf_dividend_coverage"] = manual["fcf_dividend_coverage"]
    if "debt_ratio" in manual:
        out["debt_ratio_manual"] = manual["debt_ratio"]
    if "sector" in manual:
        out["sector_manual"] = manual["sector"]
    # Recompute Chowder / 10-by-10 from the (possibly manual) inputs.
    yld = out.get("dividend_yield")
    dgr = out.get("dividend_growth_5yr")
    if yld is not None and dgr is not None:
        out["chowder"] = yld + dgr
        if yld >= 0.10:
            out["years_to_10pct_yield_on_cost"] = 0.0
        elif dgr > 0:
            out["years_to_10pct_yield_on_cost"] = math.log(0.10 / yld) / math.log(1.0 + dgr)
    return out


def _manual_label(manual_fields: List[str], field: str) -> str:
    """' ✏️' if this field was entered manually, else ''."""
    return " ✏️" if field in (manual_fields or []) else ""


def _deterioration_flags(snap: Dict[str, Any]) -> List[str]:
    """
    Trend flags from data already in the snapshot — no new fetch.
    A flag is a prompt to look closer, not a sell signal on its own.
    - Big Five metric passing at 10yr but failing at 1yr or 3yr.
    - Dividend did not increase in the latest year (streak 0 with a
      dividend history of 2+ years — Miller's "not increased in 12
      months" trigger).
    - EPS growing much faster than free cash flow (>5 pp gap).
    - Share count diluting: shares up >5% over 5 years.
    Debt trend is NOT flagged: only the latest debt ratio is stored.
    """
    flags: List[str] = []
    gk = snap.get("gatekeeper") or {}
    big_five = gk.get("big_five") or {}
    deteriorating = []
    for metric in BIG_FIVE_METRICS:
        horizons = big_five.get(metric) or {}
        ten = (horizons.get(10) or {}).get("status")
        recent_fail = (horizons.get(1) or {}).get("status") == "fail" or (horizons.get(3) or {}).get("status") == "fail"
        if ten == "pass" and recent_fail:
            deteriorating.append(metric)
    if deteriorating:
        flags.append(f"Big Five deteriorating (10yr pass, recent fail): {', '.join(deteriorating)}")
    # Snapshots don't carry the full DPS date series (only the streak
    # and growth rates), so use the streak itself as the trigger:
    if snap.get("dividend_per_share") and snap.get("dividend_streak") == 0:
        flags.append("Dividend did not increase in the latest year (streak 0) — check for a cut or freeze")
    eps_cagr, fcf_cagr = _longest_cagr(gk, "EPS"), _longest_cagr(gk, "Free_Cash_Flow")
    if eps_cagr is not None and fcf_cagr is not None and (eps_cagr - fcf_cagr) * 100.0 > 5.0:
        flags.append("EPS growing >5 pp faster than free cash flow — earnings quality check")
    ch5 = _share_change_pct(snap.get("shares_history") or {}, 5)
    if ch5 is not None and ch5 > 5.0:
        flags.append(f"Shares outstanding up {ch5:+.1f}% over 5 yrs (dilution)")
    return flags


def _data_completeness(gatekeeper: Dict[str, Any]) -> Dict[str, Any]:
    """
    How much Big Five history is actually determined, vs "not yet
    determined" (missing history). The Moat Score excludes undetermined
    horizons from its denominator, so a thin-data moat can look like a
    thick-data moat without this check.
    Returns {"metrics": n/5 with >=1 determined horizon,
             "horizons": n/20 determined horizon checks,
             "missing_metrics": [display names with no determined horizon]}.
    """
    big_five = (gatekeeper or {}).get("big_five") or {}
    metrics = 0
    horizons = 0
    missing: List[str] = []
    for metric in BIG_FIVE_METRICS:
        hs = big_five.get(metric) or {}
        determined = [h for h in HORIZONS if (hs.get(h) or {}).get("status") in ("pass", "fail")]
        horizons += len(determined)
        if determined:
            metrics += 1
        else:
            missing.append(METRIC_DISPLAY_NAMES.get(metric, metric))
    return {"metrics": metrics, "horizons": horizons, "missing_metrics": missing}


def _latest_shares(snap: Dict[str, Any]) -> Optional[float]:
    shares = snap.get("shares_history") or {}
    if not isinstance(shares, dict):
        return None
    v = shares.get(0)
    if v is None:
        v = shares.get("0")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _market_cap(snap: Dict[str, Any]) -> Optional[float]:
    """Approximate market cap: current price × latest shares outstanding.
    No new fetch — shares come from SEC filings already in the snapshot."""
    price = snap.get("current_price")
    shares = _latest_shares(snap)
    if price is None or shares is None:
        return None
    try:
        return float(price) * float(shares)
    except (TypeError, ValueError):
        return None


def _format_market_cap(value: Optional[float]) -> str:
    if value is None:
        return "\u2014"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return money(value)


def _parse_purchase_date(text: Any) -> Optional[date]:
    """Parse a YYYY-MM-DD purchase date from the editor, or None."""
    if text is None or str(text).strip() in ("", "nan", "None"):
        return None
    try:
        return datetime.strptime(str(text).strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _tax_term(purchase_date_str: Optional[str], sale: Optional[date] = None) -> Dict[str, Any]:
    """Long-term vs short-term for a sale date (default today).
    Long-term means held MORE than one year: the first long-term-
    eligible sale date is the one-year anniversary plus one day
    (e.g. bought 2024-01-01, first LT sale date 2025-01-02).
    Returns {"term": "Long-term"/"Short-term"/"—", "days_to_lt": int|None,
             "lt_date": date|None}. Single-lot assumption — see UI note."""
    purchased = _parse_purchase_date(purchase_date_str)
    if purchased is None:
        return {"term": "\u2014", "days_to_lt": None, "lt_date": None}
    sale_date = sale or datetime.now().date()
    try:
        anniversary = purchased.replace(year=purchased.year + 1)
    except ValueError:
        anniversary = purchased.replace(year=purchased.year + 1, day=28)
    lt_date = anniversary + timedelta(days=1)
    if sale_date >= lt_date:
        return {"term": "Long-term", "days_to_lt": None, "lt_date": lt_date}
    return {"term": "Short-term", "days_to_lt": (lt_date - sale_date).days, "lt_date": lt_date}


def _estimated_tax(gain: Optional[float], term: str, rates: Dict[str, Any]) -> Optional[float]:
    """Estimated tax on a sale, using rates the user entered (percent).
    None if there is no gain, no term, or no rates entered — a loss
    owes no tax on sale (it may offset other gains), and a missing
    rate is never guessed. This is an estimate, not a tax calculation."""
    if gain is None or gain <= 0:
        return None
    lt = float(rates.get("lt_pct") or 0.0)
    st_rate = float(rates.get("st_pct") or 0.0)
    ca = float(rates.get("ca_pct") or 0.0)
    if term == "Long-term":
        if lt <= 0 and ca <= 0:
            return None
        return gain * (lt + ca) / 100.0
    if term == "Short-term":
        if st_rate <= 0 and ca <= 0:
            return None
        return gain * (st_rate + ca) / 100.0
    return None


def _owner_earnings_yield(snap: Dict[str, Any]) -> Optional[float]:
    """Owner earnings (latest year, company total from SEC filings) /
    market cap. A Buffett-style earnings yield, comparable across
    holdings. None if either side is missing."""
    oe = snap.get("owner_earnings")
    cap = _market_cap(snap)
    if oe is None or cap is None or cap <= 0:
        return None
    try:
        return float(oe) / float(cap)
    except (TypeError, ValueError):
        return None


def _crosscheck_normalized(snap: Dict[str, Any], val: Optional[Dict[str, Any]], discount_rate: float) -> Dict[str, Any]:
    """Eye-check only: normalized owner-earnings sticker + ROIC level.

    Base = mean owner earnings per share across the snapshot's
    available horizons (0/1/3/5 years ago), where OE/share =
    total owner earnings (from companyfacts, split-invariant totals)
    divided by the snapshot's split-adjusted shares at that horizon.
    Growth and historical P/E are the main valuation's, so only the
    base changes — this is a cross-check, not a second valuation
    system, and it never feeds a signal, a Top 25 rank, or MOS.
    Returns {'error': ...} when the history is not available. The
    companyfacts fetch happens only when the user clicks the
    cross-check button, never on every Lookup render.
    """
    try:
        raw, status = fetch_raw_companyfacts(snap["ticker"])
        if raw is None:
            return {"error": "Couldn't reach SEC EDGAR for the cross-check"}
        hist = _build_financial_history(raw)
        oe_total = hist.get("_owner_earnings_history") or {}
        roic = hist.get("ROIC") or {}
        shares = snap.get("shares_history") or {}
        oe_ps: Dict[Any, float] = {}
        for horizon, total in oe_total.items():
            sh = shares.get(horizon)
            if sh:
                try:
                    oe_ps[horizon] = float(total) / float(sh)
                except Exception:
                    continue
        recent = [oe_ps[h] for h in (0, 1, 3, 5) if h in oe_ps and oe_ps[h] is not None]
        if len(recent) < 2:
            return {"error": "Not enough owner-earnings history for a normalized base"}
        base = sum(recent) / len(recent)
        if base is None or base <= 0:
            return {"error": "Normalized owner earnings are not positive"}
        growth = (val or {}).get("growth_used") or snap.get("growth_rate")
        pe = snap.get("historical_pe")
        sticker = calculate_sticker_price(base, growth, pe, discount_rate) if growth and pe else None
        roic_recent = [roic[h] for h in (0, 1, 3, 5) if h in roic and roic[h] is not None]
        roic_mean = (sum(roic_recent) / len(roic_recent)) if roic_recent else None
        roic_latest = roic.get(0)
        return {
            "base_oe_ps": base, "oe_ps_by_horizon": {str(k): v for k, v in oe_ps.items()},
            "sticker": sticker, "mos": sticker * 0.5 if sticker else None,
            "growth_used": growth, "roic_mean": roic_mean, "roic_latest": roic_latest,
            "roic_ok": bool(roic_mean is not None and roic_mean >= 0.15 and roic_latest is not None and roic_latest >= 0.10),
        }
    except Exception:
        return {"error": "Cross-check couldn't be computed from SEC data"}


def _audit_table_rows(audit_series: Dict[str, Any], gatekeeper: Dict[str, Any]) -> Dict[str, Any]:
    """Build the 10-year audit table from the snapshot's audit series.
    Returns {"years": [2015, ..., 2025], "rows": [{metric, values,
    median_yoy, cagr10, is_pct}]}. Median YoY is the typical year's
    growth; the 10-yr CAGR is the gatekeeper's end-to-end figure — when
    they disagree, a base year is doing the work. Display only.
    """
    series = audit_series or {}
    labels = {
        "ROIC": "ROIC",
        "Equity_Per_Share": "Equity / share",
        "EPS": "EPS",
        "Sales": "Sales",
        "Free_Cash_Flow": "Free cash flow",
    }
    years: List[int] = sorted({int(str(d)[:4]) for s in series.values() for d in (s or {})})
    years = years[-11:]
    rows = []
    big_five = (gatekeeper or {}).get("big_five") or {}
    for key, label in labels.items():
        s = series.get(key) or {}
        by_year: Dict[int, float] = {}
        for d, v in s.items():
            try:
                by_year[int(str(d)[:4])] = float(v)
            except (TypeError, ValueError):
                continue
        cagr10 = ((big_five.get(key) or {}).get(10) or {}).get("cagr")
        rows.append({
            "metric": label,
            "values": [by_year.get(y) for y in years],
            "median_yoy": median_yoy_growth(s),
            "cagr10": cagr10,
            "is_pct": key == "ROIC",
        })
    return {"years": years, "rows": rows}


def _splits_after_purchase(splits: Any, purchase_date_str: Optional[str]) -> List[Dict[str, Any]]:
    """Splits that happened after a purchase date, as [{date, ratio}].
    A split after the purchase means the shares/cost entered may still
    be pre-split. Empty when there is no purchase date or no split list
    (older cached snapshots lack it).
    """
    purchased = _parse_purchase_date(purchase_date_str)
    if purchased is None:
        return []
    out: List[Dict[str, Any]] = []
    for item in splits or []:
        try:
            sdate, ratio = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if ratio and ratio != 1.0 and str(sdate) > purchased.isoformat():
            out.append({"date": str(sdate), "ratio": ratio})
    return out


def _short_why(text: str, limit: int = 100) -> str:
    """Table cells must stay short — long Why text runs off the page
    and cannot be scrolled to on some clients. The full reasons live in
    the 'Why, in full' expander under each table."""
    if not text:
        return "\u2014"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _project_drip_income(div_income: float, dps: float, dgr: float, price: float, shares: float, years: int) -> float:
    """DRIP projection, flat-price assumption (stated in the UI): each
    year the dividend is paid on current shares at that year's DPS
    (DPS grown at dgr), then reinvested at today's price, increasing
    shares for the next year. Returns that year's income."""
    shares_sim, dps_sim, income = shares, dps, div_income
    for _ in range(years):
        dps_sim *= (1.0 + dgr)
        income = shares_sim * dps_sim
        if price and price > 0:
            shares_sim += income / price
    return income


def _projected_income_at_year(items: List[Dict[str, Any]], years: int) -> float:
    """Total dividend income in `years` years: holdings with a usable
    dividend growth rate compound at it (DRIP holdings also grow shares
    via reinvestment at today's price, flat-price assumption); holdings
    without one stay flat. Illustration only, not a forecast."""
    total = 0.0
    for it in items:
        dgr = it.get("dgr")
        if dgr is not None and dgr >= 0:
            if it.get("drip") and it.get("price") and it.get("shares") and it.get("dps"):
                total += _project_drip_income(it["income"], it["dps"], dgr, it["price"], it["shares"], years)
            else:
                total += it["income"] * ((1.0 + dgr) ** years)
        else:
            total += it["income"]
    return total


def _years_to_goal(items: List[Dict[str, Any]], goal: float, current_income: float) -> Any:
    """Rough years until projected dividend income reaches `goal`.
    Returns (years, weighted_dgr): years is 0 when already at goal, None
    when not reached within 50 years; weighted_dgr is the income-weighted
    average dividend growth rate used, or None when no holding has one."""
    if current_income >= goal:
        return (0, None)
    _w = [(it["income"], it["dgr"]) for it in items
          if it.get("dgr") is not None and it["dgr"] >= 0 and it.get("income")]
    _rate = (sum(i * g for i, g in _w) / sum(i for i, _ in _w)) if _w else None
    for _y in range(1, 51):
        if _projected_income_at_year(items, _y) >= goal:
            return (_y, _rate)
    return (None, _rate)


def _income_attribution(shares_now: Any, dps_now: Any, shares_prev: Any, dps_prev: Any, split_ratio: float = 1.0) -> Any:
    """Split an income change into new shares vs dividend raises ($/yr).
    income = shares x DPS, so the change is (new shares x old DPS) + (all
    shares x DPS change). The previous shares/DPS are split-adjusted, so
    a split alone attributes $0 to both. Returns (buys, raises); (0, 0)
    on first sighting (no baseline) or missing data. New shares includes
    DRIP reinvestment on DRIP holdings — the app cannot tell a DRIP share
    from a bought share by count alone."""
    if shares_now is None or dps_now is None or shares_prev is None or dps_prev is None:
        return (0.0, 0.0)
    _prev_shares = shares_prev * split_ratio
    _prev_dps = dps_prev / split_ratio if split_ratio else dps_prev
    _buys = (shares_now - _prev_shares) * _prev_dps
    _raises = shares_now * (dps_now - _prev_dps)
    return (_buys, _raises)


def _observed_income_growth(history: List[Dict[str, Any]]) -> Optional[float]:
    """Median month-over-month income growth from the goal history,
    annualized and capped at 50%/yr. Needs at least 2 usable steps, so
    one big first month (e.g. entering the starting portfolio) cannot
    set the pace. None when there is not enough history yet."""
    _rates: List[float] = []
    for _a, _b in zip(history, history[1:]):
        _ia, _ib = _a.get("income") or 0.0, _b.get("income") or 0.0
        if _ia > 0 and _ib > 0:
            _rates.append(_ib / _ia - 1.0)
    if len(_rates) < 2:
        return None
    _med = statistics.median(_rates)
    if _med <= 0:
        return None
    return min((1.0 + _med) ** 12 - 1.0, 0.50)


def _years_to_goal_observed(current_income: float, goal: float, annual_rate: Optional[float]) -> Optional[int]:
    """Years to goal at an observed annual income growth rate (buys +
    raises + DRIP together). 0 when already at goal, None when the rate
    is missing/zero or the goal is 50+ years out."""
    if current_income >= goal:
        return 0
    if not annual_rate or annual_rate <= 0:
        return None
    _y = math.log(goal / current_income) / math.log(1.0 + annual_rate)
    return math.ceil(_y) if _y <= 50 else None


def _dividend_payment_cut(payments: Any) -> Optional[str]:
    """Flag a cut from the actual payment history: the latest dividend
    payment vs the one before it. Returns the flag string, or None.
    A one-off special dividend can make the next regular payment look
    like a cut — the UI says so where this flag is shown."""
    if not payments or len(payments) < 2:
        return None
    (prev_date, prev_amt), (last_date, last_amt) = payments[-2], payments[-1]
    if prev_amt and prev_amt > 0 and last_amt < prev_amt * 0.995:
        _drop = (prev_amt - last_amt) / prev_amt
        return (
            f"Dividend cut: last payment ${last_amt:.2f} on {last_date} "
            f"vs ${prev_amt:.2f} on {prev_date} (\u2212{_drop:.0%})"
        )
    return None


def _payout_jump_check(payout: Any, prev_payout: Any) -> Optional[str]:
    """Flag a payout-ratio jump vs the stored baseline: +10 points, or
    crossing above 100%. Returns the flag string, or None."""
    if payout is None or prev_payout is None:
        return None
    if payout - prev_payout >= 0.10:
        return f"Payout ratio jumped {prev_payout:.0%} \u2192 {payout:.0%}"
    if payout > 1.0 and prev_payout <= 1.0:
        return "Payout above 100% \u2014 dividend may not be covered by earnings"
    return None


def _table_downloads(df: pd.DataFrame, key_prefix: str, filename_base: str) -> None:
    """CSV always; Excel when openpyxl is installed (it is added to
    requirements.txt, so Cloud gets it on redeploy — if it is missing
    locally, only CSV is offered rather than an error)."""
    st.download_button(
        "Download CSV", data=df.to_csv(index=False), file_name=f"{filename_base}.csv",
        mime="text/csv", key=f"{key_prefix}_csv",
    )
    try:
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        st.download_button(
            "Download Excel (.xlsx)", data=buf.getvalue(),
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx",
        )
    except Exception:
        st.caption("Excel export needs openpyxl (added to requirements.txt) — CSV is available now.")


def _prices_as_of_caption(dates: List[Any]) -> str:
    """One-line prices-as-of note from snapshot price_dates. A cached
    scan or a tracking-price refresh can leave prices a day (or more)
    old, so the date must be visible, not assumed to be live."""
    uniq = sorted({str(d) for d in dates if d})
    if not uniq:
        return ""
    if len(uniq) == 1:
        return f"Prices as of {uniq[0]} (last close in the snapshots)."
    return f"Prices as of {uniq[0]} – {uniq[-1]} (varies by ticker; each price is its last close)."


def _peg_ratio(snap: Dict[str, Any]) -> Optional[float]:
    """PEG = current P/E ÷ EPS growth rate (as a whole number, e.g. 12
    for 12%). Uses historical EPS growth, not the capped growth used
    for the sticker. None if EPS <= 0 or growth <= 0."""
    price = snap.get("current_price")
    eps = snap.get("current_eps")
    growth = snap.get("growth_rate")
    if not price or not eps or eps <= 0 or not growth or growth <= 0:
        return None
    try:
        pe = float(price) / float(eps)
        return pe / (float(growth) * 100.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def build_checklist_dataframe(gatekeeper: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for metric in BIG_FIVE_METRICS:
        row = {"Metric": METRIC_DISPLAY_NAMES.get(metric, metric)}
        for h in HORIZONS:
            entry = gatekeeper["big_five"].get(metric, {}).get(h, {"status": "not_yet_determined"})
            cagr = entry.get("cagr")
            # Numbers, not just checkmarks: "12.4% ✅" tells you how close
            # a pass or fail actually is.
            row[f"{h}yr"] = f"{cagr:.1%} {status_icon(entry['status'])}" if cagr is not None else status_icon(entry["status"])
        rows.append(row)
    for label, key in (
        ("Debt Test (LT Debt / Owner Earnings \u2264 3)", "debt_test"),
        ("Capital Allocation (Cap-Ex / Net Income \u2264 0.50)", "capital_allocation_test"),
    ):
        row = {"Metric": label}
        for h in HORIZONS:
            row[f"{h}yr"] = ""
        row["1yr"] = status_icon(gatekeeper[key]["status"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["Metric"] + [f"{h}yr" for h in HORIZONS])


def style_signal_column(df: pd.DataFrame):
    def color(value):
        c = SIGNAL_COLORS.get(value)
        return f"color: {c}; font-weight: 600" if c else ""

    styler = df.style
    apply_fn = getattr(styler, "map", None) or styler.applymap
    return apply_fn(color, subset=["Signal"])


def _add_ticker_to_portfolio(ticker: str, pf_name: str) -> str:
    """Add one ticker to a named portfolio, blank shares/cost for the
    user to fill in on the Portfolio tab. Returns "added", "already",
    or "failed". The cached portfolio editor frame is dropped so the
    Portfolio tab rebuilds it from the saved file (the editor
    auto-saves on every change, so no in-progress edit is lost)."""
    ticker = normalize_ticker(ticker or "")
    if not ticker:
        return "failed"
    data = load_user_data()
    portfolios = dict(data.get("portfolios") or {})
    holdings = list(portfolios.get(pf_name, []))
    if any(normalize_ticker(h.get("ticker", "")) == ticker for h in holdings):
        return "already"
    holdings.append({"ticker": ticker, "shares": None, "avg_cost": None})
    if not save_portfolio(holdings, name=pf_name):
        return "failed"
    st.session_state.pop(f"portfolio_df_{pf_name}", None)
    st.session_state.pop(f"portfolio_saved_{pf_name}", None)
    return "added"


def _add_ticker_to_watchlist(ticker: str, wl_name: str) -> str:
    """Add one ticker to a named watchlist. Returns "added", "already",
    or "failed". Drops the cached watchlist editor frame so the
    Watchlist tab rebuilds it from the saved file."""
    ticker = normalize_ticker(ticker or "")
    if not ticker:
        return "failed"
    data = load_user_data()
    lists = dict(data.get("watchlists") or {})
    wl = list(lists.get(wl_name, []))
    if ticker in wl:
        return "already"
    wl.append(ticker)
    if not save_watchlist(wl, name=wl_name):
        return "failed"
    st.session_state.pop(f"watchlist_df_{wl_name}", None)
    st.session_state.pop(f"watchlist_saved_{wl_name}", None)
    return "added"


def _render_scan_diff(prev: Optional[Dict[str, float]], cur: Dict[str, float]) -> None:
    """What changed since the last scan: new names in the top 25,
    names that dropped out, and the biggest moves in % below Sticker.
    `prev`/`cur` are {ticker: undervalued_pct} from the previous and
    current scans, ranked with the same filters."""
    if not prev:
        return
    new = [t for t in cur if t not in prev]
    dropped = [t for t in prev if t not in cur]
    movers = sorted(
        ((t, cur[t] - prev[t]) for t in cur if t in prev and abs(cur[t] - prev[t]) >= 0.05),
        key=lambda kv: -abs(kv[1]),
    )[:5]
    if not new and not dropped and not movers:
        st.caption("No change since the last scan — same top 25.")
        return
    with st.expander("What changed since the last scan", expanded=False):
        if new:
            st.write("**New in the top 25:** " + ", ".join(f"{t} ({cur[t]:.1f}% below Sticker)" for t in new))
        if dropped:
            st.write("**Dropped out:** " + ", ".join(f"{t} (was {prev[t]:.1f}% below Sticker)" for t in dropped))
        if movers:
            st.write("**Biggest moves:** " + "; ".join(f"{t} {d:+.1f} pp" for t, d in movers))
        st.caption("Compared with the top 25 from the previous scan, ranked with the same filters.")


def _add_to_portfolio_controls(tickers: List[str], key_suffix: str) -> None:
    """Ticker + portfolio pickers and an Add button, shared by the
    Watchlist tab and the ranked (Top 25 / Dividend) tables."""
    if not tickers:
        return
    data = load_user_data()
    portfolio_names = list((data.get("portfolios") or {"Main": []}).keys()) or ["Main"]
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        pick_ticker = st.selectbox("Ticker", tickers, key=f"add_pf_ticker_{key_suffix}")
    with c2:
        pick_pf = st.selectbox("Add to portfolio", portfolio_names, key=f"add_pf_name_{key_suffix}")
    with c3:
        if st.button("Add to portfolio", key=f"add_pf_btn_{key_suffix}"):
            result = _add_ticker_to_portfolio(pick_ticker, pick_pf)
            if result == "added":
                st.success(f"Added {pick_ticker} to '{pick_pf}' — set shares and cost on the Portfolio tab.")
            elif result == "already":
                st.warning(f"{pick_ticker} is already in '{pick_pf}'.")
            else:
                st.warning("Couldn't write user_data.json — changes won't persist after a restart.")


def _pct_or_dash(v: Any) -> str:
    return "\u2014" if v is None else f"{v:.1%}"



def render_ticker_card(ticker: str, snap: Dict[str, Any], val: Optional[Dict[str, Any]], max_growth: float, discount_rate: float = 0.15) -> None:
    st.subheader(ticker + (f" \u2014 {snap['name']}" if snap.get("name") else ""))
    if snap.get("error"):
        st.warning(f"{ticker}: {snap['error']}")
        st.divider()
        return

    price, sticker, mos = snap["current_price"], val["sticker_price"], val["margin_of_safety_price"]
    cols = st.columns(4)
    with cols[0]:
        st.markdown(render_metric_box("Current Price", money(price), price_color_class(price, sticker, mos)), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(render_metric_box("Sticker Price", money(sticker), "metric-neutral"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(render_metric_box("Margin of Safety Price", money(mos), "metric-neutral"), unsafe_allow_html=True)
    with cols[3]:
        action = val["action"]
        st.markdown(render_metric_box("Action", action.replace("_", " "), ACTION_COLOR_CLASS.get(action, "metric-neutral")), unsafe_allow_html=True)

    st.markdown(f"**Moat Score:** {snap['moat_score']} / 100")
    comp = _data_completeness(snap["gatekeeper"])
    st.caption(
        f"Data completeness: {comp['metrics']}/5 Big Five metrics with determined history "
        f"({comp['horizons']}/20 horizon checks)"
        + (f" — missing: {', '.join(comp['missing_metrics'])}" if comp["missing_metrics"] else "")
    )
    if comp["metrics"] <= 3:
        st.warning(
            "Thin data — this Moat Score is based on limited history. Treat it with less confidence "
            "than a score with 5/5 metrics determined."
        )
    # ROIC dropouts, no longer silent: if ROIC is missing for a horizon,
    # say why — a capped Moat is not the same as a bad business.
    roic_dropped = snap.get("roic_dropped") or {}
    if roic_dropped:
        def _roic_year_label(y: Any) -> str:
            try:
                yi = int(y)
            except Exception:
                return str(y)
            return "latest year" if yi == 0 else f"{yi} yrs ago"
        st.warning(
            "ROIC not computed for: "
            + "; ".join(f"{_roic_year_label(y)} — {reason}" for y, reason in sorted(roic_dropped.items(), key=lambda kv: str(kv[0])))
            + ". Those horizons contribute no ROIC to the Big Five / Moat, which caps the score — it does not mean ROIC is low."
        )
    details = [f"Last close {snap.get('price_date', '')}"]
    if val["growth_used"] is not None:
        g = snap["growth_rate"]
        details.append(f"EPS growth {g:.1%}" + (f" (capped at {max_growth:.0%} for valuation)" if g > max_growth else ""))
    if snap.get("historical_pe") is not None:
        details.append(f"median historical P/E {snap['historical_pe']:.1f}")
    peg_card = _peg_ratio(snap)
    if peg_card is not None:
        details.append(f"PEG {peg_card:.2f} (current P/E ÷ EPS growth)")
    details.append(f"Market cap ~{_format_market_cap(_market_cap(snap))}")
    oey_card = _owner_earnings_yield(snap)
    if oey_card is not None:
        details.append(f"Owner earnings yield {oey_card:.1%} (owner earnings ÷ market cap)")
    if snap.get("dividend_streak"):
        details.append(f"dividend raised {snap['dividend_streak']} yrs in a row")
    st.caption(" \u00b7 ".join(details))
    if val["note"]:
        st.caption(val["note"])
    st.dataframe(build_checklist_dataframe(snap["gatekeeper"]), hide_index=True, use_container_width=True)

    # One-tap add from the card, same as the tables: pick a portfolio /
    # watchlist and the ticker lands there with blank shares/cost.
    st.caption("Add this ticker to a portfolio or watchlist — shares and cost stay blank until you set them on the Portfolio tab.")
    card_c1, card_c2 = st.columns(2)
    with card_c1:
        pf_names_card = list((load_user_data().get("portfolios") or {"Main": []}).keys()) or ["Main"]
        pick_pf_card = st.selectbox("Portfolio", pf_names_card, key=f"card_pf_{ticker}")
        if st.button("Add to portfolio", key=f"card_add_pf_{ticker}"):
            card_pf_result = _add_ticker_to_portfolio(ticker, pick_pf_card)
            if card_pf_result == "added":
                st.success(f"Added {ticker} to '{pick_pf_card}'.")
            elif card_pf_result == "already":
                st.warning(f"{ticker} is already in '{pick_pf_card}'.")
            else:
                st.warning("Couldn't write user_data.json — changes won't persist after a restart.")
    with card_c2:
        wl_names_card = list((load_user_data().get("watchlists") or {"Main": []}).keys()) or ["Main"]
        pick_wl_card = st.selectbox("Watchlist", wl_names_card, key=f"card_wl_{ticker}")
        if st.button("Add to watchlist", key=f"card_add_wl_{ticker}"):
            card_wl_result = _add_ticker_to_watchlist(ticker, pick_wl_card)
            if card_wl_result == "added":
                st.success(f"Added {ticker} to watchlist '{pick_wl_card}'.")
            elif card_wl_result == "already":
                st.warning(f"{ticker} is already in watchlist '{pick_wl_card}'.")
            else:
                st.warning("Couldn't write user_data.json — changes won't persist after a restart.")

    # Price vs Sticker chart — lazy, so a lookup doesn't pay for a
    # Yahoo history call it didn't ask for. Monthly closes (16 years is
    # ~200 points, not ~4,000 daily), with today's Sticker and MOS as
    # horizontal lines. The result is kept in session state for the
    # ticker, so re-renders don't re-fetch.
    if val and val.get("sticker_price") is not None:
        chart_key = f"price_chart_{ticker}"
        if st.button("Show price vs Sticker chart", key=f"chart_btn_{ticker}"):
            with st.spinner("Fetching price history..."):
                bundle, pstatus = get_price_history(ticker)
            if bundle and bundle.get("closes"):
                st.session_state[chart_key] = bundle["closes"]
            else:
                st.session_state[chart_key] = {"error": "Couldn't fetch price history"}
        chart_closes = st.session_state.get(chart_key)
        if isinstance(chart_closes, dict) and chart_closes.get("error"):
            st.warning(str(chart_closes["error"]))
        elif isinstance(chart_closes, dict) and chart_closes:
            monthly: Dict[str, float] = {}
            for d in sorted(chart_closes):
                monthly[d[:7]] = chart_closes[d]
            chart_frame = pd.DataFrame({"Price": list(monthly.values())}, index=pd.to_datetime(list(monthly.keys())))
            chart_frame["Sticker"] = val["sticker_price"]
            chart_frame["MOS Price"] = val["margin_of_safety_price"]
            st.line_chart(chart_frame)
            st.caption(
                "Monthly closes (split-adjusted, not dividend-adjusted). Sticker and MOS lines are today's values, "
                "not historical — the chart shows where price has been relative to today's buy levels."
            )

    # Deterioration flags: trend prompts, not sell signals on their own.
    det_flags = _deterioration_flags(snap)
    if det_flags:
        with st.expander(f"⚠️ Deterioration flags ({len(det_flags)})", expanded=False):
            for flag in det_flags:
                st.write(f"- {flag}")
            st.caption("A flag is a prompt to look closer — e.g. a Big Five metric that passed at 10 years but fails recently, or a dividend that didn't increase this year. It does not by itself mean sell.")

    # Sticker sensitivity: the sticker is a point estimate built from
    # growth and discount guesses. Vary both around the current
    # assumptions so the range — and the MOS range — is visible.
    if val and val.get("growth_used") is not None and snap.get("current_eps") and snap.get("historical_pe"):
        with st.expander("Sticker sensitivity — what if growth / discount is wrong?", expanded=False):
            growth_centre = val["growth_used"]
            growths = sorted({max(0.01, growth_centre - 0.03), growth_centre, growth_centre + 0.03})
            discounts = sorted({max(0.06, discount_rate - 0.03), discount_rate, min(0.20, discount_rate + 0.03)})
            sens_rows = []
            for g in growths:
                row_s: Dict[str, Any] = {"Growth": f"{g:.1%}" + (" ← current" if abs(g - growth_centre) < 1e-9 else "")}
                for d in discounts:
                    s = calculate_sticker_price(snap["current_eps"], g, snap["historical_pe"], d)
                    row_s[f"Discount {d:.0%}" + (" ← current" if abs(d - discount_rate) < 1e-9 else "")] = money(s)
                sens_rows.append(row_s)
            st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)
            stickers = [
                calculate_sticker_price(snap["current_eps"], g, snap["historical_pe"], d)
                for g in growths for d in discounts
            ]
            stickers = [s for s in stickers if s is not None]
            if stickers:
                st.caption(
                    f"Sticker range in this table: {money(min(stickers))} – {money(max(stickers))}; "
                    f"MOS range (50%): {money(min(stickers) * 0.5)} – {money(max(stickers) * 0.5)}. "
                    f"The main sticker caps growth at {max_growth:.0%}; sensitivity varies growth around the growth actually used."
                )

    # Numbers behind the checklist: debt, owner earnings, dilution,
    # and EPS-vs-FCF divergence (cash vs accounting — a Buffett check).
    gk = snap["gatekeeper"]
    with st.expander("Numbers behind the checklist", expanded=False):
        debt_ratio = (gk.get("debt_test") or {}).get("ratio")
        cap_ratio = (gk.get("capital_allocation_test") or {}).get("ratio")
        st.write(f"Debt Test: LT debt / owner earnings = {debt_ratio:.2f}x" if debt_ratio is not None else "Debt Test: \u2014 (needs owner earnings)")
        st.write(f"Capital allocation: Cap-Ex / net income = {cap_ratio:.1%}" if cap_ratio is not None else "Capital allocation: \u2014")
        st.write(f"Owner earnings (latest year): {money(snap.get('owner_earnings'))}")
        tag_used = snap.get("tag_used") or {}
        if tag_used:
            st.caption(
                "SEC tags used (fullest candidate tag per metric, gaps filled from other tags): "
                + "; ".join(f"{k}: {v}" for k, v in sorted(tag_used.items()))
            )
        ch10 = _share_change_pct(snap.get("shares_history") or {}, 10)
        ch5 = _share_change_pct(snap.get("shares_history") or {}, 5)
        if ch10 is not None:
            st.write(f"Shares outstanding: {ch10:+.1f}% over 10 yrs" + (" (dilution)" if ch10 > 0 else " (buybacks)"))
        if ch5 is not None:
            st.write(f"Shares outstanding: {ch5:+.1f}% over 5 yrs" + (" (dilution)" if ch5 > 0 else " (buybacks)"))
        if ch10 is None and ch5 is None:
            st.write("Shares outstanding trend: \u2014 (not in SEC history)")
        eps_cagr, fcf_cagr = _longest_cagr(gk, "EPS"), _longest_cagr(gk, "Free_Cash_Flow")
        if eps_cagr is not None and fcf_cagr is not None:
            gap_pp = (eps_cagr - fcf_cagr) * 100.0
            st.write(f"EPS CAGR {eps_cagr:.1%} vs free-cash-flow CAGR {fcf_cagr:.1%} (gap {gap_pp:+.1f} pp)")
            if gap_pp > 5.0:
                st.warning("EPS is growing much faster than free cash flow — check earnings quality before trusting the EPS growth rate.")
        else:
            st.write("EPS vs free-cash-flow comparison: \u2014")

    # 10-year audit — the actual year-by-year Big Five values, so a
    # CAGR can't hide behind one weird base year. Display only.
    _audit = _audit_table_rows(snap.get("audit_series") or {}, snap.get("gatekeeper") or {})
    with st.expander("10-year audit — Big Five values by year (median YoY vs 10-yr CAGR)", expanded=False):
        if not _audit["years"]:
            st.caption("Audit data appears after the snapshot refreshes (fundamentals are cached 24 hours).")
        else:
            _a_cols = ["Metric"] + [str(y) for y in _audit["years"]] + ["Median YoY", "10-yr CAGR"]
            _a_rows = []
            for _ar in _audit["rows"]:
                _vals = []
                for _v in _ar["values"]:
                    if _v is None:
                        _vals.append("\u2014")
                    elif _ar["is_pct"]:
                        _vals.append(f"{_v:.1%}")
                    elif _ar["metric"] in ("EPS", "Equity / share"):
                        _vals.append(money(_v))
                    else:
                        _vals.append(_format_market_cap(_v))
                _a_rows.append(
                    {"Metric": _ar["metric"], **dict(zip([str(y) for y in _audit["years"]], _vals)),
                     "Median YoY": "\u2014" if _ar["median_yoy"] is None else f"{_ar['median_yoy']:.1%}",
                     "10-yr CAGR": "\u2014" if _ar["cagr10"] is None else f"{_ar['cagr10']:.1%}"}
                )
            st.dataframe(pd.DataFrame(_a_rows, columns=_a_cols), hide_index=True, use_container_width=True)
            st.caption("CAGR is end-to-end — one bad base year flatters it; median YoY is the typical year's growth. When they disagree, look at the yearly values. Display only — nothing here changes the Sticker, Moat, or any signal.")

    # Buffett notes: Meaning / circle of competence, moat type,
    # Management, thesis, and sell trigger. XBRL cannot see these.
    note = _notes_for(ticker)
    with st.expander(f"Buffett notes for {ticker}", expanded=False):
        meaning = st.slider("Meaning — how well do you understand this business? (1 = not at all, 5 = deeply)", 1, 5, int(note.get("meaning") or 3), key=f"notes_meaning_{ticker}")
        management = st.slider("Management — trust in capital allocation (1 = low, 5 = high)", 1, 5, int(note.get("management") or 3), key=f"notes_mgmt_{ticker}")
        moat_type = st.selectbox("Moat type", MOAT_TYPE_OPTIONS, index=MOAT_TYPE_OPTIONS.index(note.get("moat_type")) if note.get("moat_type") in MOAT_TYPE_OPTIONS else 0, key=f"notes_moat_{ticker}")
        thesis = st.text_area("Thesis — why own this?", value=note.get("thesis") or "", key=f"notes_thesis_{ticker}", height=80)
        sell_trigger = st.text_area("Sell trigger — what would make you sell?", value=note.get("sell_trigger") or "", key=f"notes_sell_{ticker}", height=80)
        if st.button("Save notes", key=f"notes_save_{ticker}"):
            if save_note(ticker, {"meaning": meaning, "management": management, "moat_type": moat_type, "thesis": thesis, "sell_trigger": sell_trigger}):
                st.success(f"Notes saved for {ticker}.")
            else:
                st.warning("Couldn't save notes — they won't persist after a restart on Cloud. Use the Portfolio export as a backup.")

    # Cross-checks — eye check only. Nothing here changes the main
    # Sticker Price, MOS, Top 25 ranking, or any BUY/SELL signal.
    with st.expander("Cross-checks — eye check only (normalized sticker + ROIC level)", expanded=False):
        st.caption(
            "The normalized sticker reuses the main growth rate and historical P/E and only changes the earnings base "
            "to mean owner earnings per share across recent horizons. It is rougher than backtest v6's annual mean "
            "(the snapshot gives horizons, not every year), so treat a disagreement as a prompt to look closer, not as a new valuation."
        )
        if st.button("Compute cross-checks", key=f"cross_{ticker}"):
            st.session_state.setdefault("crosscheck", {})[ticker] = _crosscheck_normalized(snap, val, discount_rate)
        cross = (st.session_state.get("crosscheck") or {}).get(ticker)
        if cross:
            if cross.get("error"):
                st.warning(cross["error"])
            else:
                st.write(f"Normalized owner earnings/share (mean of recent horizons): {money(cross['base_oe_ps'])}")
                st.write(f"Cross-check Sticker: {money(cross['sticker'])} — MOS: {money(cross['mos'])} (main Sticker: {money(sticker)})")
                if cross.get("roic_mean") is not None:
                    st.write(
                        f"ROIC level: mean {cross['roic_mean']:.1%}, latest {cross['roic_latest']:.1%}" if cross.get("roic_latest") is not None else f"ROIC level: mean {cross['roic_mean']:.1%}"
                        + (" — meets the v6 ROIC-level bar (mean ≥15%, latest ≥10%)" if cross.get("roic_ok") else " — below the v6 ROIC-level bar")
                    )
                else:
                    st.write("ROIC level: — (not in SEC history)")
        st.markdown("**More eye checks — from this snapshot, no extra fetch**")
        _fcf = snap.get("free_cash_flow")
        _sbc = snap.get("sbc")
        _sbc_fcf = snap.get("sbc_adjusted_fcf")
        if _fcf is not None and _sbc is not None and _sbc_fcf is not None:
            st.write(f"Free cash flow (latest year): {money(_fcf)} — after stock-based compensation {money(_sbc)}: {money(_sbc_fcf)}")
        elif _fcf is not None:
            st.write(f"Free cash flow (latest year): {money(_fcf)} — SBC: \u2014 (not in SEC filings)")
        else:
            st.write("Free cash flow: \u2014")
        _oe = snap.get("owner_earnings")
        _ni = snap.get("net_income")
        _maint = snap.get("maintenance_capex_est")
        if _oe is not None and _ni is not None and _maint is not None:
            st.write(f"Owner earnings (latest year, total capex): {money(_oe)} — with maintenance capex \u2248 D&A ({money(_maint)}): {money(_ni)}")
        elif _oe is not None:
            st.write(f"Owner earnings (latest year, total capex): {money(_oe)}")
        st.caption("Maintenance capex is estimated as depreciation & amortization — a rough proxy, not a reported number, so the maintenance-capex owner earnings equal net income by construction. SBC is a real cost to owners (dilution) even though no cash leaves.")
        _ic = snap.get("interest_coverage")
        if _ic is not None:
            st.write(f"Interest coverage (operating income / interest expense): {_ic:.1f}x")
        else:
            st.write("Interest coverage: \u2014 (no interest expense in SEC filings, or operating income missing)")
        st.write(f"Total debt (long-term + short-term, latest year): {money(snap.get('total_debt'))}")

    # Value-investor holdings — eye check only, from SEC filings.
    # One expander, one button, one table: the 13F managers plus Phil
    # Town's Rule One Fund (N-PORT, not a 13F filer), marked by Source.
    with st.expander("Value-investor holdings — eye check only (SEC filings, lagged)", expanded=False):
        st.caption(
            "Holdings are quarterly and can be ~4 months old. 13F filings are due up to 45 days after quarter-end and "
            "exclude shorts, cash, bonds, and non-US positions; Phil Town's Rule One Fund files N-PORT instead (only "
            "public filings for a fiscal quarter-end month, up to 60 days after quarter-end, common stock only — "
            "options/derivatives and the money-market position excluded). Matching is by issuer name, not CUSIP — class "
            "shares (e.g. GOOG vs GOOGL) are flagged ambiguous instead of guessed. A holding here is never a buy signal. "
            "The first check for a session scans the Rule One trust's recent filings for its series and can take a while; "
            "results are then cached for 24 hours."
        )
        if st.button("Check investor holdings", key=f"holdings_{ticker}"):
            st.session_state.setdefault("holdings_check", {})[ticker] = check_holdings(ticker, snap.get("name"))
        held = (st.session_state.get("holdings_check") or {}).get(ticker)
        if held:
            rows_h = []
            for r in held:
                if r.get("error"):
                    rows_h.append({"Investor": r["person"], "Manager": r["manager"], "Source": r.get("source") or "—", "Holds?": "— (fetch failed)", "Shares": "—", "Value": "—", "% of portfolio": "—", "Report date": r.get("report_date") or "—", "Filing date": r.get("filing_date") or "—", "Issuer as filed": r["error"]})
                elif not r["holds"]:
                    rows_h.append({"Investor": r["person"], "Manager": r["manager"], "Source": r.get("source") or "—", "Holds?": "No (by name)", "Shares": "—", "Value": "—", "% of portfolio": "—", "Report date": r.get("report_date") or "—", "Filing date": r.get("filing_date") or "—", "Issuer as filed": "—"})
                else:
                    for m in r["matches"]:
                        rows_h.append({
                            "Investor": r["person"], "Manager": r["manager"], "Source": r.get("source") or "—",
                            "Holds?": "Yes — class ambiguous, check CUSIP" if r["ambiguous"] else "Yes",
                            "Shares": f"{m['shares']:,.0f}", "Value": money(m["value"]),
                            "% of portfolio": f"{m['pct_portfolio']:.1%}" if m.get("pct_portfolio") is not None else "—",
                            "Report date": r.get("report_date") or "—", "Filing date": r.get("filing_date") or "—",
                            "Issuer as filed": f"{m['issuer']} (CUSIP {m['cusip']})",
                        })
            st.dataframe(pd.DataFrame(rows_h), hide_index=True, use_container_width=True)
            for r in held:
                if r.get("source") == "N-PORT" and r.get("net_assets"):
                    st.caption(f"Rule One Fund net assets at report date: {money(r['net_assets'])}")
        for nf in NON_13F:
            st.caption(f"{nf['person']}: {nf['note']}")
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Ticker Search")
    st.caption("Any ticker, comma or newline separated.")
    ticker_text = st.text_area("Tickers", value="AAPL, MCD, SBUX", height=100, label_visibility="collapsed")
    lookup_tickers = parse_ticker_list(ticker_text)

    st.divider()
    discount_rate = st.slider(
        "Discount rate (minimum acceptable return)", min_value=0.06, max_value=0.20, value=0.15, step=0.01, format="%.2f"
    )
    max_growth = st.slider(
        "Growth cap for Sticker Price", min_value=0.05, max_value=0.30, value=DEFAULT_MAX_GROWTH_RATE, step=0.01, format="%.2f"
    )
    st.caption(
        "Sticker Price uses the lower of a company's historical EPS growth and this cap, so one "
        "hot streak can't be extrapolated 10 years. Both settings apply to every tab."
    )
    st.divider()
    if st.button("Refresh data", key="refresh_data"):
        clear_cache()
        st.caption("Cache cleared \u2014 everything will be re-fetched.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.title("Business Quality Terminal")
st.caption("Rule #1 Investing / Buffett-style Gatekeeper checklist and valuation.")

tab_dashboard, tab_lookup, tab_portfolio, tab_watchlist, tab_undervalued, tab_dividends = st.tabs(
    ["Dashboard", "Ticker Lookup", "Portfolio", "Watchlist", "Top 25 Undervalued", "Top 25 + Dividend Growers"]
)

# --- 0. Dashboard -------------------------------------------------------
# One screen: what you own, what it's worth, and what needs attention.
# It checks every holding and watchlist ticker once, reusing the cached
# snapshots (fundamentals 24h, prices refreshed only if older than
# 15 minutes), so after the first check of the day it's mostly cache
# hits. Signal thresholds come from the Portfolio/Watchlist sliders —
# their session values if you've set them, otherwise the defaults.
with tab_dashboard:
    st.caption(
        "What you own, what it's worth, and what needs attention — across every portfolio and watchlist, "
        "without opening each tab."
    )
    _dash_data = load_user_data()
    _dash_portfolios = _dash_data.get("portfolios") or {}
    _dash_watchlists = _dash_data.get("watchlists") or {}
    _dash_cash = _dash_data.get("portfolio_cash") or {}
    _dash_div = 0.0
    _dash_div_items: List[Dict[str, Any]] = []
    _dash_checked = 0
    _dash_watch: Dict[str, Any] = dict(_dash_data.get("dividend_watch") or {})
    _dash_watch_changed = False
    _dash_has_cut = False
    _dash_baseline: Dict[str, Any] = dict(_dash_data.get("holdings_baseline") or {})
    _dash_baseline_changed = False
    _dash_attr_buys = 0.0
    _dash_attr_raises = 0.0
    if not any(_dash_portfolios.values()) and not any(_dash_watchlists.values()):
        st.info("Nothing to summarize yet — add a holding on the Portfolio tab or a ticker on the Watchlist tab.")
    else:
        _approach = float(st.session_state.get("pf_approach", 10))
        _min_moat_hold = float(st.session_state.get("pf_moat", 40))
        _max_loss = float(st.session_state.get("pf_loss", 0))
        _min_moat_buy_pf = float(st.session_state.get("pf_buy_moat", 60))
        _near_buy_pf = float(st.session_state.get("pf_near", 15))
        _min_moat_wl = float(st.session_state.get("wl_moat", 60))
        _near_buy_wl = float(st.session_state.get("wl_near", 15))
        _dash_value = _dash_cost = _dash_div = _dash_cash_total = 0.0
        _pf_counts: Counter = Counter()
        _wl_counts: Counter = Counter()
        _attention: List[Dict[str, Any]] = []
        _dash_price_dates: List[Any] = []
        _pf_rows: List[Dict[str, Any]] = []
        with st.spinner("Checking your holdings and watchlist..."):
            for _pf_name, _holdings in _dash_portfolios.items():
                _pf_cash = float(_dash_cash.get(_pf_name, 0.0))
                _dash_cash_total += _pf_cash
                _pf_value = _pf_cost = _pf_div = 0.0
                _pf_sig_counts: Counter = Counter()
                for _h in _holdings:
                    _snap = build_snapshot(_h["ticker"], price_max_age=TRACKING_PRICE_MAX_AGE)
                    if _snap.get("error"):
                        _pf_counts["NO DATA"] += 1
                        _pf_sig_counts["NO DATA"] += 1
                        _attention.append({"Where": _pf_name, "Ticker": _h["ticker"], "Signal": "NO DATA", "Why": _short_why(_snap["error"])})
                        continue
                    if _snap.get("price_date"):
                        _dash_price_dates.append(_snap["price_date"])
                    _val = valuation_from_snapshot(_snap, discount_rate, max_growth)
                    _sig = portfolio_signal(
                        _snap, _val, _h.get("avg_cost"), _approach, _min_moat_hold, _max_loss,
                        _min_moat_buy_pf, _near_buy_pf,
                    )
                    _pf_counts[_sig["signal"]] += 1
                    _pf_sig_counts[_sig["signal"]] += 1
                    _price = _snap["current_price"]
                    _value = _price * _h["shares"] if _h.get("shares") else None
                    if _value is not None:
                        _pf_value += _value
                    if _h.get("shares") and _h.get("avg_cost"):
                        _pf_cost += _h["shares"] * _h["avg_cost"]
                    _manual_dash = _manual_for(_h["ticker"])
                    _dps_dash = _manual_dash.get("dividend_per_share") if "dividend_per_share" in _manual_dash else _snap.get("dividend_per_share")
                    _div_dash = _h["shares"] * _dps_dash if (_h.get("shares") and _dps_dash) else None
                    if _div_dash is not None:
                        _pf_div += _div_dash
                    _dash_checked += 1
                    # Years-to-goal inputs: this holding's income, DPS, 5-yr
                    # dividend growth, price, shares, DRIP flag.
                    _dgr_dash = _manual_dash.get("dividend_growth_5yr") if "dividend_growth_5yr" in _manual_dash else _snap.get("dividend_growth_5yr")
                    if _div_dash is not None:
                        _dash_div_items.append({"income": _div_dash, "dps": _dps_dash, "dgr": _dgr_dash, "price": _price, "shares": _h.get("shares"), "drip": bool(_h.get("drip"))})
                    # Dividend-cut watch: the cut flag compares the last two
                    # actual payments in the snapshot (not the last check).
                    # The payout jump compares against the stored baseline;
                    # the snapshot's own payout is used, not a manual
                    # override — a user editing manual data is not a
                    # company action.
                    _cut_flag = _dividend_payment_cut(_snap.get("dividend_payments"))
                    _payout_raw = _snap.get("payout_ratio")
                    _prev_pay = (_dash_watch.get(_h["ticker"]) or {}).get("payout")
                    _pay_flag = _payout_jump_check(_payout_raw, _prev_pay)
                    _watch_flags = [f for f in (_cut_flag, _pay_flag) if f]
                    if _watch_flags:
                        if _cut_flag:
                            _dash_has_cut = True
                        _attention.append({
                            "Where": _pf_name, "Ticker": _h["ticker"],
                            "Signal": "DIV CUT" if _cut_flag else "PAYOUT",
                            "Why": _short_why("; ".join(_watch_flags)),
                        })
                    _new_watch = {"payout": _payout_raw, "date": datetime.now().strftime("%Y-%m-%d")}
                    _prev_watch = _dash_watch.get(_h["ticker"])
                    if _prev_watch is None or _prev_watch.get("payout") != _new_watch["payout"]:
                        _dash_watch[_h["ticker"]] = _new_watch
                        _dash_watch_changed = True
                    # Income attribution: compare this holding against its
                    # stored baseline (first sighting just sets it).
                    # Splits since the baseline are adjusted out.
                    _prev_base = _dash_baseline.get(_h["ticker"])
                    _base_date = (_prev_base or {}).get("date")
                    _split_ratio = 1.0
                    if _base_date:
                        for _sp in _splits_after_purchase(_snap.get("splits"), _base_date):
                            _split_ratio *= _sp["ratio"]
                    _attr_b, _attr_r = _income_attribution(
                        _h.get("shares"), _dps_dash,
                        (_prev_base or {}).get("shares"), (_prev_base or {}).get("dps"),
                        _split_ratio,
                    )
                    _dash_attr_buys += _attr_b
                    _dash_attr_raises += _attr_r
                    _new_base = {"shares": _h.get("shares"), "dps": _dps_dash, "income": _div_dash,
                                 "date": datetime.now().strftime("%Y-%m-%d")}
                    if (_prev_base is None or _prev_base.get("shares") != _new_base["shares"]
                            or _prev_base.get("dps") != _new_base["dps"]):
                        _dash_baseline[_h["ticker"]] = _new_base
                        _dash_baseline_changed = True
                    if _sig["signal"] in ("SELL", "REVIEW"):
                        _attention.append({
                            "Where": _pf_name, "Ticker": _h["ticker"], "Signal": _sig["signal"],
                            "Why": _short_why("; ".join(_sig["reasons"])),
                        })
                _dash_value += _pf_value
                _dash_cost += _pf_cost
                _dash_div += _pf_div
                _pf_rows.append({
                    "Portfolio": _pf_name, "Holdings": len(_holdings),
                    "Value": money(_pf_value), "Cash": money(_pf_cash),
                    "Gain/Loss": money(_pf_value - _pf_cost) if _pf_cost else "—",
                    "Div income/yr": money(_pf_div) if _pf_div else "—",
                    "BUY": _pf_sig_counts.get("BUY", 0), "REVIEW": _pf_sig_counts.get("REVIEW", 0),
                    "SELL": _pf_sig_counts.get("SELL", 0),
                })
            for _wl_name, _tickers in _dash_watchlists.items():
                for _t in _tickers:
                    _snap = build_snapshot(_t, price_max_age=TRACKING_PRICE_MAX_AGE)
                    if _snap.get("error"):
                        _wl_counts["NO DATA"] += 1
                        continue
                    if _snap.get("price_date"):
                        _dash_price_dates.append(_snap["price_date"])
                    _val = valuation_from_snapshot(_snap, discount_rate, max_growth)
                    _sig = buy_signal(_snap, _val, _min_moat_wl, _near_buy_wl)
                    _wl_counts[_sig["signal"]] += 1
                    if _sig["signal"] in ("BUY", "NEAR BUY"):
                        _attention.append({
                            "Where": f"Watchlist '{_wl_name}'", "Ticker": _t, "Signal": _sig["signal"],
                            "Why": _short_why("; ".join(_sig["reasons"])),
                        })
        if _dash_watch_changed:
            save_dividend_watch(_dash_watch)
        if _dash_baseline_changed:
            save_holdings_baseline(_dash_baseline)
        _m1, _m2, _m3, _m4 = st.columns(4)
        with _m1:
            st.metric("Total account value", money(_dash_value + _dash_cash_total))
        with _m2:
            _gain = _dash_value - _dash_cost
            _gain_pct = (_gain / _dash_cost * 100) if _dash_cost else None
            st.metric("Total gain/loss", money(_gain) if _dash_cost else "—", delta=None if _gain_pct is None else f"{_gain_pct:+.1f}% vs cost")
        with _m3:
            st.metric("Cash", money(_dash_cash_total))
        with _m4:
            st.metric("Dividend income/yr", money(_dash_div) if _dash_div else "—")
        if _pf_rows:
            st.dataframe(pd.DataFrame(_pf_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Portfolio signals: " + (" · ".join(f"{n} {s}" for s, n in _pf_counts.most_common()) if _pf_counts else "no holdings")
            + "  |  Watchlist signals: " + (" · ".join(f"{n} {s}" for s, n in _wl_counts.most_common()) if _wl_counts else "no tickers")
        )
        if _attention:
            st.subheader("Needs attention")
            st.dataframe(pd.DataFrame(_attention), hide_index=True, use_container_width=True)
        else:
            st.success("Nothing needs attention right now — no SELL or REVIEW holdings, and no watchlist ticker at its buy price.")
        if _dash_has_cut:
            st.caption("Dividend-cut flags compare the last two actual payments — a one-off special dividend can make the next regular payment look like a cut.")
        _as_of_dash = _prices_as_of_caption(_dash_price_dates)
        if _as_of_dash:
            st.caption(_as_of_dash)
        st.caption(
            "Signals use your Portfolio/Watchlist slider settings (or their defaults if you haven't changed them "
            "this session). Gain/loss and dividend income count only holdings with shares and a cost / dividend "
            "per share recorded. Totals include cash across all portfolios."
        )
    # --- Dividend income goal -------------------------------------------
    # One overall annual goal, saved in user_data.json. Progress uses the
    # same projected dividend income/yr as the metric above (shares x
    # dividend per share across every portfolio).
    _div_goal = float(_dash_data.get("dividend_income_goal") or 0.0)
    st.subheader("Dividend income goal")
    _goal_in = st.number_input(
        "Annual dividend income goal ($/yr)", min_value=0.0, value=_div_goal,
        step=100.0, key="div_goal_input",
        help="The yearly dividend income you want your portfolios to reach. "
        "Progress below uses the same projected income as the Dividend income/yr metric above.",
    )
    if float(_goal_in) != _div_goal:
        save_dividend_goal(float(_goal_in))
        _div_goal = float(_goal_in)
    if _div_goal > 0:
        _div_pct = _dash_div / _div_goal
        st.progress(min(_div_pct, 1.0))
        if _div_pct >= 1.0:
            st.write(f"**{money(_dash_div)} of {money(_div_goal)}/yr — {_div_pct:.0%} of goal. Goal reached.**")
        else:
            st.write(
                f"**{money(_dash_div)} of {money(_div_goal)}/yr — {_div_pct:.0%} of goal** "
                f"· {money(_div_goal - _dash_div)}/yr to go"
            )
    else:
        st.caption("Set a goal above and this shows how close your current dividend income is to it.")
    if _dash_checked > 0:
        _hist_month = datetime.now().strftime("%Y-%m")
        _stored_month = next((e for e in (_dash_data.get("dividend_goal_history") or []) if e.get("month") == _hist_month), None)
        _month_buys = (float((_stored_month or {}).get("buys") or 0.0) + _dash_attr_buys)
        _month_raises = (float((_stored_month or {}).get("raises") or 0.0) + _dash_attr_raises)
        _history = list(_dash_data.get("dividend_goal_history") or [])
        _history = [e for e in _history if e.get("month") != _hist_month]
        _history.append({"month": _hist_month, "income": _dash_div, "goal": _div_goal, "buys": _month_buys, "raises": _month_raises})
        _history.sort(key=lambda e: e.get("month", ""))
        _history = _history[-24:]
        record_dividend_goal_month(_hist_month, _dash_div, _div_goal, buys=_month_buys, raises=_month_raises)
        if _div_goal > 0 and _dash_div < _div_goal:
            _obs_rate = _observed_income_growth(_history)
            if _obs_rate is not None:
                _yrs_obs = _years_to_goal_observed(_dash_div, _div_goal, _obs_rate)
                if _yrs_obs is not None:
                    st.write(f"Roughly **{_yrs_obs} year{'s' if _yrs_obs != 1 else ''}** to reach the goal at your actual pace (~{_obs_rate:.0%}/yr income growth over the last {len(_history)} months).")
                else:
                    st.write("50+ years to reach the goal at your actual pace — growth is too slow to get there soon.")
                st.caption("Estimate only: your actual pace includes weekly buys, raises, and DRIP, so it changes as you do.")
            else:
                _yrs, _rate = _years_to_goal(_dash_div_items, _div_goal, _dash_div)
                if _yrs is not None:
                    _rate_txt = f" (dividends growing ~{_rate:.1%}/yr)" if _rate is not None else ""
                    st.write(f"Roughly **{_yrs} year{'s' if _yrs != 1 else ''}** to reach the goal at the current pace{_rate_txt}.")
                    st.caption("Estimate only: assumes dividend growth continues, no new money in or out, and flat prices for DRIP reinvestment. Once there are a few months of history, this switches to your actual pace.")
                elif _rate is not None:
                    st.write("50+ years to reach the goal at the current pace — growth is too slow to get there soon.")
                else:
                    st.write("Not on pace at current growth — no holding has a dividend growth rate to project from, so income stays flat.")
        if _month_buys or _month_raises:
            st.write(f"This month: {_signed_money(_month_buys)} from new shares · {_signed_money(_month_raises)} from dividend raises")
            st.caption("New shares includes DRIP reinvestment on DRIP holdings — the app cannot tell a DRIP share from a bought share by count alone.")
        if len(_history) >= 2:
            _hist_df = pd.DataFrame(_history).set_index("month")
            _hist_cols = ["income"] + (["goal"] if any(e.get("goal") for e in _history) else [])
            st.subheader("Goal history")
            st.line_chart(_hist_df[_hist_cols].rename(columns={"income": "Income", "goal": "Goal"}))
            st.caption("One point per month, updated each time you open the Dashboard.")
        else:
            st.caption("Goal history starts this month — check back next month for a trend.")


# --- 1. Ticker Lookup ----------------------------------------------------
with tab_lookup:
    st.caption("Works for any company that files 10-Ks with the SEC \u2014 not limited to the S&P 500.")
    # Lazy fetch: the page must render before any SEC/Yahoo call runs.
    # Auto-fetching on first load is what left Cloud stuck on a spinner
    # when Yahoo throttled / timed out from a shared Cloud IP.
    if st.button("Fetch tickers", key="fetch_lookup"):
        st.session_state["lookup_requested"] = lookup_tickers
    lookup_to_fetch = st.session_state.get("lookup_requested", [])
    if not lookup_to_fetch:
        st.info("Enter one or more tickers in the sidebar, then click Fetch tickers.")
    fetched_snaps: Dict[str, Any] = {}
    for ticker in lookup_to_fetch:
        with st.spinner(f"Fetching {ticker}..."):
            snap = build_snapshot(ticker, price_max_age=LOOKUP_PRICE_MAX_AGE)
        val = None if snap.get("error") else valuation_from_snapshot(snap, discount_rate, max_growth)
        fetched_snaps[ticker] = snap
        render_ticker_card(ticker, snap, val, max_growth, discount_rate)
    # Side-by-side compare: same numbers, one table, up to 4 tickers.
    comparable = [t for t, s in fetched_snaps.items() if not s.get("error")]
    if len(comparable) >= 2:
        st.subheader("Compare tickers")
        compare_picks = st.multiselect(
            "Pick 2–4 tickers to compare", comparable, default=comparable[:4], max_selections=4, key="compare_picks"
        )
        if len(compare_picks) >= 2:
            compare_rows: List[Dict[str, Any]] = []
            for label, fmt in [
                ("Price", lambda s, v: money(s.get("current_price"))),
                ("Sticker Price", lambda s, v: money(v.get("sticker_price")) if v else "—"),
                ("MOS Price", lambda s, v: money(v.get("margin_of_safety_price")) if v else "—"),
                ("% Below Sticker", lambda s, v: "—" if v is None or v.get("undervalued_pct") is None else f"{v['undervalued_pct']:.1f}%"),
                ("Growth Used", lambda s, v: "—" if v is None or v.get("growth_used") is None else f"{v['growth_used']:.1%}"),
                ("Historical P/E", lambda s, v: "—" if s.get("historical_pe") is None else f"{s['historical_pe']:.1f}"),
                ("Moat Score", lambda s, v: str(s.get("moat_score"))),
                ("Big Five passes", lambda s, v: f"{big_five_pass_count(s, 0.10)}/5"),
                ("Dividend streak (yrs)", lambda s, v: str(s.get("dividend_streak") or 0)),
                ("Dividend yield", lambda s, v: _pct_or_dash(s.get("dividend_yield"))),
                ("Payout ratio", lambda s, v: _pct_or_dash(s.get("payout_ratio"))),
                ("Market cap", lambda s, v: _format_market_cap(_market_cap(s))),
                ("PEG", lambda s, v: "—" if _peg_ratio(s) is None else f"{_peg_ratio(s):.2f}"),
                ("Owner earnings yield", lambda s, v: "—" if _owner_earnings_yield(s) is None else f"{_owner_earnings_yield(s):.1%}"),
                ("Signal (Moat ≥60, near-buy 15%)", lambda s, v: buy_signal(s, v, 60.0, 15.0)["signal"] if v else "—"),
            ]:
                row: Dict[str, Any] = {"Metric": label}
                for t in compare_picks:
                    s = fetched_snaps[t]
                    v = valuation_from_snapshot(s, discount_rate, max_growth)
                    row[t] = fmt(s, v)
                compare_rows.append(row)
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
            st.caption("Signal uses the watchlist defaults (Moat ≥60, near-buy band 15%) for all tickers, so the comparison is apples-to-apples.")
        elif compare_picks:
            st.info("Pick at least 2 tickers to compare.")


# --- 2. Portfolio ---------------------------------------------------------
def clean_portfolio_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for _, r in df.iterrows():
        ticker = normalize_ticker(_cell(r.get("Ticker")) or "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        target = _num(r.get("Target %"))
        drip_raw = r.get("DRIP")
        rows.append({
            "ticker": ticker, "shares": _num(r.get("Shares")), "avg_cost": _num(r.get("Avg Cost")),
            "target_pct": max(0.0, min(100.0, target)) if target is not None else None,
            "drip": bool(drip_raw) if drip_raw is not None and str(drip_raw) not in ("", "nan") else False,
            "purchase_date": _parse_purchase_date(r.get("Purchase Date")).strftime("%Y-%m-%d") if _parse_purchase_date(r.get("Purchase Date")) else None,
        })
    return rows


with tab_portfolio:
    st.caption(
        "Add your holdings (shares and average cost are optional but give you gain/loss). "
        "Signals are BUY (add to a holding) / NEAR BUY / HOLD / REVIEW / SELL. "
        "Changes save automatically to user_data.json — on Streamlit Cloud that file "
        "can be wiped on sleep/restart, so use the export below for anything you want to keep. " + DISCLAIMER
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        approach_pct = st.slider("Flag when within X% of Sticker Price", 0, 30, 10, key="pf_approach")
    with c2:
        min_moat_hold = st.slider("Flag when Moat Score is below", 0, 100, 40, key="pf_moat")
    with c3:
        max_loss_pct = st.number_input("Flag when down more than X% vs cost (0 = off)", 0, 90, 0, key="pf_loss")
    c4, c5 = st.columns(2)
    with c4:
        min_moat_buy_pf = st.slider("Minimum Moat Score to call it a BUY (add)", 0, 100, 60, key="pf_buy_moat")
    with c5:
        near_buy_pct_pf = st.slider("'Near buy' band: within X% above the buy price", 0, 50, 15, key="pf_near")

    # Named portfolios: each name is its own holdings list (e.g. Roth
    # vs taxable). Legacy single-portfolio files migrate to "Main".
    saved_all = load_user_data()
    portfolio_names = list((saved_all.get("portfolios") or {"Main": []}).keys()) or ["Main"]
    # BUG FIX (2026-11-04): pf_selected_name used to be the selectbox's own
    # widget key, and the Add/Delete/Import handlers below set it AFTER the
    # selectbox had already rendered in the same run. Streamlit forbids
    # changing a widget's session-state value after instantiation, so the
    # first click errored — even though the save before it had already
    # succeeded, which is why a second tap then "worked" (the name was
    # already in the list). Fix: pf_selected_name is now plain state, not
    # a widget key; the selectbox is driven by index, and the user's
    # dropdown pick is written back to the plain key afterwards.
    if "pf_selected_name" not in st.session_state or st.session_state["pf_selected_name"] not in portfolio_names:
        st.session_state["pf_selected_name"] = portfolio_names[0]
    pf_name = st.selectbox("Portfolio", portfolio_names, index=portfolio_names.index(st.session_state["pf_selected_name"]))
    st.session_state["pf_selected_name"] = pf_name
    _last_saved = last_saved_display()
    st.caption(
        ("Saved on this server: " + _last_saved + ". " if _last_saved else "No saved file on this server yet. ")
        + "Streamlit Cloud storage is ephemeral — a restart or sleep can wipe portfolios, watchlists, and notes. "
        "Use Export (below) after important changes to keep a permanent copy."
    )
    new_pf_col1, new_pf_col2, new_pf_col3 = st.columns(3)
    with new_pf_col1:
        new_pf_name = st.text_input("New portfolio name", key="pf_new_name").strip()
    with new_pf_col2:
        if st.button("Add portfolio", key="pf_add_btn") and new_pf_name:
            if new_pf_name in portfolio_names:
                st.warning(f"A portfolio named {new_pf_name} already exists.")
            elif save_portfolio([], name=new_pf_name):
                st.session_state["pf_selected_name"] = new_pf_name
                st.rerun()
    with new_pf_col3:
        if len(portfolio_names) > 1 and st.button(f"Delete portfolio '{pf_name}'", key="pf_del_btn"):
            st.warning("Deleting cannot be undone — export first if you want a backup.")
            if delete_portfolio(pf_name):
                st.session_state["pf_selected_name"] = portfolio_names[0]
                for k in list(st.session_state.keys()):
                    if k.startswith("portfolio_df_") or k.startswith("portfolio_saved_"):
                        st.session_state.pop(k, None)
                st.rerun()
    cash_balance = st.number_input(
        "Cash balance ($) — counts toward total account value and weights",
        0.0, 1e12, float((saved_all.get("portfolio_cash") or {}).get(pf_name, 0.0)), key=f"pf_cash_{pf_name}",
    )
    if cash_balance != float((saved_all.get("portfolio_cash") or {}).get(pf_name, 0.0)):
        save_cash(pf_name, cash_balance)
    # Account type + tax rates. Tax columns, estimates, wash-sale
    # warnings, and rebalance tax costs appear ONLY for Taxable
    # portfolios — sales inside an IRA are not taxed at sale, so
    # showing tax estimates there would be wrong.
    pf_account_type = (saved_all.get("portfolio_account_types") or {}).get(pf_name, "Not set")
    if pf_account_type not in ACCOUNT_TYPES:
        pf_account_type = "Not set"
    pf_account_type_new = st.selectbox(
        "Account type — tax features apply only to Taxable portfolios",
        list(ACCOUNT_TYPES), index=list(ACCOUNT_TYPES).index(pf_account_type), key=f"pf_acct_{pf_name}",
    )
    if pf_account_type_new != pf_account_type:
        save_account_type(pf_name, pf_account_type_new)
        pf_account_type = pf_account_type_new
    is_taxable_pf = pf_account_type == "Taxable"
    with st.expander("Tax settings — your marginal rates (estimates only)", expanded=False):
        st.caption(
            "Enter YOUR marginal rates as percentages (e.g. 15 for 15%). These are used only for the "
            "estimated-tax columns and rebalance tax costs. Leave them at 0 and estimates show —, "
            "never a guessed rate. Estimates are not tax advice and do not replace your broker's "
            "1099 or a tax professional — your actual rate depends on total income, filing status, "
            "and what else you sold that year."
        )
        saved_rates = saved_all.get("tax_rates") or {}
        lt_rate_in = st.number_input("Federal long-term rate (%)", 0.0, 50.0, float(saved_rates.get("lt_pct") or 0.0), key="tax_lt_rate")
        st_rate_in = st.number_input("Federal short-term rate (%)", 0.0, 50.0, float(saved_rates.get("st_pct") or 0.0), key="tax_st_rate")
        ca_rate_in = st.number_input("California rate (%)", 0.0, 20.0, float(saved_rates.get("ca_pct") or 0.0), key="tax_ca_rate")
        tax_rates = {"lt_pct": float(lt_rate_in), "st_pct": float(st_rate_in), "ca_pct": float(ca_rate_in)}
        if tax_rates != {"lt_pct": float(saved_rates.get("lt_pct") or 0.0), "st_pct": float(saved_rates.get("st_pct") or 0.0), "ca_pct": float(saved_rates.get("ca_pct") or 0.0)}:
            save_tax_rates(tax_rates["lt_pct"], tax_rates["st_pct"], tax_rates["ca_pct"])
    if not is_taxable_pf:
        st.caption("Estimated tax, wash-sale warnings, and rebalance tax costs appear only for Taxable portfolios. Long/short-term status is shown for any portfolio with a purchase date, since holding period is factual — sales inside an IRA are not taxed at sale, so no tax estimate is shown there.")

    pf_df_key = f"portfolio_df_{pf_name}"
    pf_saved_key = f"portfolio_saved_{pf_name}"
    if pf_df_key not in st.session_state:
        saved_pf = (load_user_data().get("portfolios") or {}).get(pf_name, [])
        frame = pd.DataFrame(
            [{"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": h["avg_cost"],
              "Target %": h.get("target_pct"), "DRIP": bool(h.get("drip")),
              "Purchase Date": h.get("purchase_date") or ""} for h in saved_pf],
            columns=["Ticker", "Shares", "Avg Cost", "Target %", "DRIP", "Purchase Date"],
        )
        frame["Shares"] = pd.to_numeric(frame["Shares"], errors="coerce").astype("float64")
        frame["Avg Cost"] = pd.to_numeric(frame["Avg Cost"], errors="coerce").astype("float64")
        frame["Target %"] = pd.to_numeric(frame["Target %"], errors="coerce").astype("float64")
        st.session_state[pf_df_key] = frame
        st.session_state[pf_saved_key] = saved_pf

    # The holdings editor lives in a collapsed expander so the input
    # grid doesn't clog the page — open it only to add, remove, or
    # change a holding. Everything below it (signals, totals, tax)
    # reads the same saved holdings.
    with st.expander("Edit holdings — add, remove, or change tickers, shares, cost, targets, purchase dates", expanded=False):
        edited = st.data_editor(
            st.session_state[pf_df_key],
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"portfolio_editor_{pf_name}",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Shares": st.column_config.NumberColumn("Shares", min_value=0.0),
                "Avg Cost": st.column_config.NumberColumn("Avg Cost", min_value=0.0, format="$%.2f"),
                "Target %": st.column_config.NumberColumn("Target % of account (blank = no target)", min_value=0.0, max_value=100.0),
                "DRIP": st.column_config.CheckboxColumn("DRIP (dividends reinvested)"),
                "Purchase Date": st.column_config.TextColumn("Purchase Date (YYYY-MM-DD, blank = unknown)"),
            },
        )
        st.caption("Purchase Date is a single-lot assumption: if you bought a holding in multiple lots at different times, one date cannot split long-term vs short-term per lot. Tax columns below use this date, and only for Taxable portfolios.")
    holdings = clean_portfolio_rows(edited)
    if holdings != st.session_state[pf_saved_key]:
        if save_portfolio(holdings, name=pf_name):
            st.session_state[pf_saved_key] = holdings
        else:
            st.warning("Couldn't write user_data.json \u2014 changes won't persist after a restart.")

    if not holdings:
        st.info("Add a ticker above to see its buy/hold/sell signal.")
    else:
        rows, total_value, total_cost, total_div_income = [], 0.0, 0.0, 0.0
        review_items: List[Dict[str, Any]] = []
        full_why_items: List[Dict[str, Any]] = []
        projection_items: List[Dict[str, Any]] = []
        projection_excluded: List[str] = []
        projection_cost_included = 0.0
        portfolio_price_dates: List[Any] = []
        with st.spinner("Checking your holdings..."):
            for h in holdings:
                snap = build_snapshot(h["ticker"], price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": "\u2014",
                                 "Value": "\u2014", "Gain/Loss": "\u2014", "Gain %": "\u2014", "Sticker": "\u2014", "Moat": "\u2014",
                                 "Meaning": "\u2014", "Moat type": "\u2014", "Weight": "\u2014", "Target %": "\u2014", "Drift": "\u2014",
                                 "Div Income/yr": "\u2014",
                                 "% of Div Income": "\u2014",
                                 "Yield on Cost": "\u2014", "Flags": "\u2014", "Mkt Cap": "\u2014", "PEG": "\u2014",
                                 "Owner Yield": "\u2014", "Data": "\u2014", "Tax Term": "\u2014", "Days to LT": "\u2014", "Est Tax": "\u2014",
                                 "Signal": "NO DATA", "Lean": "\u2014", "Why": _short_why(snap["error"]), "_rank": 9, "_value": 0.0, "_moat": 0.0})
                    review_items.append({"Ticker": h["ticker"], "Signal": "NO DATA", "Lean": "\u2014", "Reason": snap["error"], "_urgency": 0})
                    continue
                val = valuation_from_snapshot(snap, discount_rate, max_growth)
                if snap.get("price_date"):
                    portfolio_price_dates.append(snap["price_date"])
                sig = portfolio_signal(
                    snap, val, h["avg_cost"], approach_pct, min_moat_hold, float(max_loss_pct),
                    min_moat_buy_pf, float(near_buy_pct_pf),
                )
                note_pf = _notes_for(h["ticker"])
                price = snap["current_price"]
                value = price * h["shares"] if h["shares"] else None
                gain = (price - h["avg_cost"]) * h["shares"] if (h["shares"] and h["avg_cost"]) else None
                gain_pct = (price / h["avg_cost"] - 1) * 100 if h["avg_cost"] else None
                if value is not None:
                    total_value += value
                if h["shares"] and h["avg_cost"]:
                    total_cost += h["shares"] * h["avg_cost"]
                # Dividend income: pulled DPS, or manual DPS (✏️) if the
                # pull missed it. No DPS = no income estimate, not $0.
                manual_pf = _manual_for(h["ticker"])
                dps_pf = manual_pf.get("dividend_per_share") if "dividend_per_share" in manual_pf else snap.get("dividend_per_share")
                dps_label = " ✏️" if "dividend_per_share" in manual_pf else ""
                div_income = h["shares"] * dps_pf if (h["shares"] and dps_pf) else None
                if div_income is not None:
                    total_div_income += div_income
                yoc = (dps_pf / h["avg_cost"]) if (dps_pf and h["avg_cost"]) else None
                det = _deterioration_flags(snap)
                comp_pf = _data_completeness(snap["gatekeeper"])
                peg_pf = _peg_ratio(snap)
                # Review queue reasons for this holding.
                review_reasons: List[str] = []
                review_urgency = 2
                if sig["signal"] in ("SELL", "REVIEW"):
                    review_reasons.append(f"Signal: {sig['signal']}")
                    review_urgency = 0 if sig["signal"] == "SELL" else 1
                if det:
                    review_reasons.extend(det)
                if comp_pf["metrics"] <= 3:
                    review_reasons.append(f"Thin data: {comp_pf['metrics']}/5 Big Five metrics determined")
                # Split warning: bought before a split, the shares/cost
                # entered may still be pre-split (Yahoo splits, from
                # the snapshot). Post-split shares are x the ratio and
                # cost per share is / the ratio.
                for _sp in _splits_after_purchase(snap.get("splits"), h.get("purchase_date")):
                    review_reasons.append(
                        f"Split {_sp['ratio']:g}-for-1 on {_sp['date']} after your {h.get('purchase_date')} purchase — "
                        f"check shares (x{_sp['ratio']:g}) and cost (÷{_sp['ratio']:g}) are post-split"
                    )
                    review_urgency = min(review_urgency, 1)
                # Dividend safety / jeopardy, only for holdings with a
                # known dividend (non-dividend holdings are not scored).
                payout_pf = manual_pf.get("payout_ratio") if "payout_ratio" in manual_pf else snap.get("payout_ratio")
                fcf_pf = manual_pf.get("fcf_dividend_coverage") if "fcf_dividend_coverage" in manual_pf else snap.get("fcf_dividend_coverage")
                dgr_pf = manual_pf.get("dividend_growth_5yr") if "dividend_growth_5yr" in manual_pf else snap.get("dividend_growth_5yr")
                if dps_pf is not None or payout_pf is not None:
                    debt_pf = manual_pf.get("debt_ratio") if "debt_ratio" in manual_pf else ((snap.get("gatekeeper") or {}).get("debt_test") or {}).get("ratio")
                    try:
                        bf10_pf = big_five_pass_count(snap, 0.10)
                    except Exception:
                        bf10_pf = None
                    safety_pf = dividend_safety_score(
                        payout_pf, fcf_pf, debt_pf, snap.get("dividend_streak"), dgr_pf, bf10_pf,
                        sector=manual_pf.get("sector"),
                    )
                    if safety_pf["score"] < 40:
                        review_reasons.append(f"Safety Score {safety_pf['score']} {safety_pf['band']}")
                        review_urgency = min(review_urgency, 2)
                    if (payout_pf is not None and payout_pf > 1.0) or (fcf_pf is not None and fcf_pf < 1.0):
                        review_reasons.append("Dividend in jeopardy (payout >100% or FCF cover <1x)")
                        review_urgency = min(review_urgency, 1)
                lean_pf = sig.get("lean", "\u2014")
                lean_note_pf = sig.get("lean_note", "")
                if sig["signal"] == "REVIEW":
                    why_short_pf = _short_why(f"Lean {lean_pf}: {lean_note_pf}") if lean_note_pf else _short_why("; ".join(sig["reasons"]))
                else:
                    why_short_pf = _short_why("; ".join(sig["reasons"]))
                full_why_items.append({
                    "Ticker": h["ticker"], "Signal": sig["signal"], "Lean": lean_pf,
                    "Why": "; ".join(sig["reasons"]) + (f" | Lean: {lean_note_pf}" if lean_note_pf else ""),
                })
                if review_reasons:
                    review_items.append({"Ticker": h["ticker"], "Signal": sig["signal"], "Lean": lean_pf, "Reason": "; ".join(review_reasons), "_urgency": review_urgency})
                # Dividend income projection: current income grown at
                # the 5-year dividend growth rate. DRIP holdings also
                # grow shares: each year's dividend is reinvested at
                # today's price (flat-price assumption, stated in the
                # UI). Illustration only, not a forecast.
                if div_income is not None and dgr_pf is not None and dgr_pf >= 0:
                    if h.get("drip") and price and h["shares"] and dps_pf:
                        inc5 = _project_drip_income(div_income, dps_pf, dgr_pf, price, h["shares"], 5)
                        inc10 = _project_drip_income(div_income, dps_pf, dgr_pf, price, h["shares"], 10)
                    else:
                        inc5 = div_income * ((1.0 + dgr_pf) ** 5)
                        inc10 = div_income * ((1.0 + dgr_pf) ** 10)
                    yoc10 = (inc10 / (h["shares"] * h["avg_cost"])) if (h["shares"] and h["avg_cost"]) else None
                    projection_items.append({
                        "Ticker": h["ticker"], "DRIP": "Yes" if h.get("drip") else "No",
                        "Income now": money(div_income),
                        "Income in 5 yrs": money(inc5), "Income in 10 yrs": money(inc10),
                        "Yield on Cost in 10 yrs": "\u2014" if yoc10 is None else f"{yoc10:.1%}",
                        "_inc10": inc10,
                    })
                    if h["shares"] and h["avg_cost"]:
                        projection_cost_included += h["shares"] * h["avg_cost"]
                elif div_income is not None:
                    projection_excluded.append(h["ticker"])
                oey_pf = _owner_earnings_yield(snap)
                tax_info_pf = _tax_term(h.get("purchase_date"))
                est_tax_pf = _estimated_tax(gain, tax_info_pf["term"], tax_rates) if is_taxable_pf else None
                rows.append({
                    "Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": money(price),
                    "Value": money(value), "Gain/Loss": money(gain),
                    "Gain %": "\u2014" if gain_pct is None else f"{gain_pct:+.1f}%",
                    "Sticker": money(val["sticker_price"]), "Moat": snap["moat_score"],
                    "Meaning": note_pf.get("meaning") or "\u2014", "Moat type": note_pf.get("moat_type") or "\u2014",
                    "Weight": "\u2014", "Target %": "\u2014" if h.get("target_pct") is None else f"{h['target_pct']:.1f}%", "Drift": "\u2014",
                    "Div Income/yr": money(div_income) + dps_label if div_income is not None else "\u2014",
                    "% of Div Income": "\u2014",
                    "Yield on Cost": "\u2014" if yoc is None else f"{yoc:.1%}",
                    "Flags": "; ".join(det) if det else "\u2014",
                    "Mkt Cap": _format_market_cap(_market_cap(snap)),
                    "PEG": "\u2014" if peg_pf is None else f"{peg_pf:.2f}",
                    "Owner Yield": "\u2014" if oey_pf is None else f"{oey_pf:.1%}",
                    "Data": f"{comp_pf['metrics']}/5" + (" ⚠️" if comp_pf["metrics"] <= 3 else ""),
                    "Tax Term": tax_info_pf["term"],
                    "Days to LT": "\u2014" if tax_info_pf["days_to_lt"] is None else str(tax_info_pf["days_to_lt"]),
                    "Est Tax": "\u2014" if est_tax_pf is None else money(est_tax_pf),
                    "Signal": sig["signal"], "Lean": lean_pf, "Why": why_short_pf,
                    "_rank": PORTFOLIO_RANK.get(sig["signal"], 9),
                    "_value": value or 0.0, "_moat": snap["moat_score"] or 0.0, "_target": h.get("target_pct"),
                    "_ticker_for_log": h["ticker"], "_price_for_log": price, "_sticker_for_log": val["sticker_price"],
                    "_gain": gain, "_term": tax_info_pf["term"], "_shares": h["shares"], "_avg_cost": h["avg_cost"],
                    "_div_income": div_income or 0.0,
                })
        # Weights are % of total account value (holdings + cash), so a
        # cash balance makes weights real instead of holdings-only.
        total_account = total_value + cash_balance
        for r in rows:
            if total_account > 0 and r["_value"]:
                r["Weight"] = f"{r['_value'] / total_account:.1%}"
                if r.get("_target") is not None:
                    r["Drift"] = f"{(r['_value'] / total_account * 100.0 - r['_target']):+.1f}%"
        for r in rows:
            if total_div_income > 0 and r.get("_div_income"):
                r["% of Div Income"] = f"{r['_div_income'] / total_div_income:.1%}"
        weighted_moat = (sum(r["_value"] * r["_moat"] for r in rows) / total_value) if total_value > 0 else None
        largest = max(rows, key=lambda r: r["_value"]) if rows else None
        rows.sort(key=lambda r: r["_rank"])
        # Signal log: one entry per ticker per day (today's entry is
        # updated on re-check, past days are never rewritten).
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_entries = [
            {"date": today_str, "ticker": r["_ticker_for_log"], "price": r["_price_for_log"],
             "sticker": r["_sticker_for_log"], "signal": r["Signal"], "lean": r["Lean"]}
            for r in rows if r.get("_ticker_for_log")
        ]
        if log_entries:
            append_signal_log(log_entries)
        # Review queue first: only holdings needing attention, sorted
        # by urgency (SELL, then REVIEW/jeopardy, then other flags).
        if review_items:
            review_items.sort(key=lambda r: r["_urgency"])
            st.markdown("**Review queue — holdings needing attention**")
            st.dataframe(pd.DataFrame(review_items).drop(columns=["_urgency"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Review queue: no holdings need attention right now.")
        df = pd.DataFrame(rows).drop(columns=["_rank", "_value", "_moat", "_target", "_ticker_for_log", "_price_for_log", "_sticker_for_log", "_gain", "_term", "_shares", "_avg_cost", "_div_income"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        as_of_pf = _prices_as_of_caption(portfolio_price_dates)
        if as_of_pf:
            st.caption(as_of_pf)
        # Full Why, in wrapping text — the table's Why column is kept
        # short on purpose, because long Why text ran off the page.
        with st.expander("Why, in full — every holding's reasons and lean", expanded=False):
            for item in full_why_items:
                st.markdown(f"**{item['Ticker']} — {item['Signal']} (Lean: {item['Lean']})** — {item['Why']}")
            st.caption("Lean is a reading of the reasons, not a new signal. SELL still means sell, BUY still means add.")
        if total_cost > 0:
            total_gain = total_value - total_cost
            st.markdown(
                f"**Portfolio totals:** value {money(total_value)} \u00b7 cost basis {money(total_cost)} \u00b7 "
                f"gain/loss {money(total_gain)} ({total_gain / total_cost:+.1%})"
            )
        if cash_balance > 0 or total_account > 0:
            st.markdown(
                f"**Cash:** {money(cash_balance)}"
                + (f" ({cash_balance / total_account:.1%} of account)" if total_account > 0 else "")
                + f" \u00b7 **Total account value:** {money(total_account)} (holdings + cash)"
            )
        if total_div_income > 0:
            st.markdown(
                f"**Dividend income:** {money(total_div_income)}/yr from holdings with a known dividend per share"
                + (f" \u00b7 yield on cost {total_div_income / total_cost:.1%}" if total_cost > 0 else "")
                + " \u00b7 holdings with no DPS show \u2014, not $0, so this total may understate income."
            )
        if weighted_moat is not None:
            st.markdown(f"**Weighted Moat Score:** {weighted_moat:.1f} / 100 (by position value)")
        if largest is not None and total_account > 0 and largest["_value"]:
            st.markdown(f"**Largest holding:** {largest['Ticker']} at {largest['_value'] / total_account:.1%} of account value")
        # Dividend income projection, if growth continues at the 5-year
        # rate. This is an illustration, not a forecast — growth rates
        # change, dividends can be cut, and holdings without a known
        # DPS or growth rate are excluded (not counted as $0).
        if projection_items:
            st.markdown("**Dividend income projection — if dividend growth continues at the 5-year rate**")
            proj_df = pd.DataFrame(projection_items).drop(columns=["_inc10"])
            st.dataframe(proj_df, hide_index=True, use_container_width=True)
            total_inc10 = sum(p["_inc10"] for p in projection_items)
            st.caption(
                f"Projected total in 10 yrs: {money(total_inc10)}/yr"
                + (f" \u00b7 projected yield on cost (included holdings only): {total_inc10 / projection_cost_included:.1%}" if projection_cost_included > 0 else "")
                + (f" \u00b7 excluded (no DPS or growth rate): {', '.join(projection_excluded)}" if projection_excluded else "")
                + " \u00b7 DRIP holdings assume price stays at today's price, so reinvested dividends buy shares at today's price."
                + " \u00b7 Illustration only, not a forecast."
            )
        counts = Counter(r["Signal"] for r in rows)
        st.caption(" \u00b7 ".join(f"{n} {s}" for s, n in counts.most_common()))
        _table_downloads(df, f"pf_{pf_name}", f"{pf_name}_portfolio")
        # Wash-sale watch (Taxable only): a ticker sold at a loss in
        # the last 30 days that is now held again. This catches only
        # re-adds recorded here — purchases outside the app are not
        # known — and it is a risk flag, not a determination. Your
        # broker's wash-sale reporting and a tax professional govern.
        if is_taxable_pf:
            today_w = datetime.now().date()
            held_tickers = {h["ticker"] for h in holdings}
            wash_hits = []
            for entry in (load_user_data().get("sales_log") or []):
                if entry.get("gain") is None or entry["gain"] >= 0 or entry.get("ticker") not in held_tickers:
                    continue
                try:
                    sale_d = datetime.strptime(entry["date"], "%Y-%m-%d").date()
                except Exception:
                    continue
                if 0 <= (today_w - sale_d).days <= 30:
                    wash_hits.append(entry)
            if wash_hits:
                st.warning(
                    "Wash-sale risk — sold at a loss in the last 30 days and now held again: "
                    + ", ".join(f"{e['ticker']} (sold {e['date']}, loss {money(e['gain'])})" for e in wash_hits)
                    + ". A loss repurchased within 30 days may be disallowed (added to the new shares' basis instead)."
                )
        # Record a sale + tax-year summary.
        with st.expander("Record a sale — and tax-year summary", expanded=False):
            if "sale_msg" in st.session_state:
                st.success(st.session_state.pop("sale_msg"))
            st.caption(
                "Recording a sale here does not place a trade — it only logs a sale you already made, "
                "so the tax-year summary and wash-sale watch can use it. It also updates the holding "
                "(removes it, or reduces shares for a partial sale)."
            )
            holdings_map = {h["ticker"]: h for h in holdings}
            sale_ticker = st.selectbox("Ticker sold", list(holdings_map.keys()), key=f"sale_ticker_{pf_name}")
            sale_holding = holdings_map.get(sale_ticker, {}) if sale_ticker else {}
            sale_price_map = {r["Ticker"]: r["_price_for_log"] for r in rows if r.get("_price_for_log")}
            sale_shares = st.number_input("Shares sold", 0.0, float(sale_holding.get("shares") or 0.0), float(sale_holding.get("shares") or 0.0), key=f"sale_shares_{pf_name}")
            sale_price = st.number_input("Sale price per share ($)", 0.0, 1e9, float(sale_price_map.get(sale_ticker) or sale_holding.get("avg_cost") or 0.0), key=f"sale_price_{pf_name}")
            sale_date = st.date_input("Sale date", datetime.now().date(), key=f"sale_date_{pf_name}")
            purchase_unknown = st.checkbox("Purchase date unknown (term will show —)", key=f"sale_purch_unknown_{pf_name}")
            default_purch = _parse_purchase_date(sale_holding.get("purchase_date")) or sale_date
            sale_purch_date = None if purchase_unknown else st.date_input("Purchase date", default_purch, key=f"sale_purch_{pf_name}")
            if st.button("Record sale", key=f"sale_btn_{pf_name}"):
                if not sale_ticker or sale_shares <= 0 or sale_price <= 0:
                    st.warning("Enter a ticker, shares sold, and sale price.")
                elif sale_holding.get("shares") and sale_shares > sale_holding["shares"]:
                    st.warning("Shares sold cannot exceed shares held.")
                else:
                    avg_c = sale_holding.get("avg_cost") or 0.0
                    sale_gain = (sale_price - avg_c) * sale_shares
                    sale_term = _tax_term(sale_purch_date.strftime("%Y-%m-%d") if sale_purch_date else None, sale_date)["term"]
                    if append_sale({
                        "date": sale_date.strftime("%Y-%m-%d"), "ticker": sale_ticker, "portfolio": pf_name,
                        "shares": sale_shares, "sale_price": sale_price, "avg_cost": avg_c, "gain": sale_gain,
                        "term": sale_term, "purchase_date": sale_purch_date.strftime("%Y-%m-%d") if sale_purch_date else None,
                    }):
                        new_holdings = []
                        for h in holdings:
                            if h["ticker"] != sale_ticker:
                                new_holdings.append(h)
                            elif sale_holding.get("shares") and sale_shares < sale_holding["shares"]:
                                nh = dict(h)
                                nh["shares"] = sale_holding["shares"] - sale_shares
                                new_holdings.append(nh)
                        save_portfolio(new_holdings, name=pf_name)
                        st.session_state.pop(pf_df_key, None)
                        st.session_state.pop(pf_saved_key, None)
                        st.session_state["sale_msg"] = f"Recorded sale: {sale_ticker} {sale_shares:g} shares at {money(sale_price)} — gain/loss {money(sale_gain)} ({sale_term})."
                        st.rerun()
                    else:
                        st.error("Could not save the sale.")
            st.markdown("**Tax-year summary — from sales recorded here only**")
            st.caption("This is not a 1099: it covers only sales you recorded in this app, not sales outside it. Losses may offset gains; the general $3,000/year limit on net losses against ordinary income and carryforward rules are outside this summary — check with a tax professional.")
            all_sales = load_user_data().get("sales_log") or []
            sale_pf_options = ["All portfolios"] + sorted({e.get("portfolio") or "" for e in all_sales if e.get("portfolio")})
            sale_pf_filter = st.selectbox("Portfolio filter", sale_pf_options, key=f"sale_filter_{pf_name}")
            filtered_sales = [e for e in all_sales if sale_pf_filter == "All portfolios" or e.get("portfolio") == sale_pf_filter]
            if not filtered_sales:
                st.caption("No sales recorded yet.")
            else:
                summary_rows = []
                for year in sorted({e["date"][:4] for e in filtered_sales}):
                    year_sales = [e for e in filtered_sales if e["date"][:4] == year]
                    lt_sum = sum(e["gain"] for e in year_sales if e.get("term") == "Long-term" and e.get("gain") is not None)
                    st_sum = sum(e["gain"] for e in year_sales if e.get("term") == "Short-term" and e.get("gain") is not None)
                    other_sum = sum(e["gain"] for e in year_sales if e.get("term") not in ("Long-term", "Short-term") and e.get("gain") is not None)
                    summary_rows.append({"Year": year, "Long-term gain/loss": money(lt_sum), "Short-term gain/loss": money(st_sum), "Unknown-term gain/loss": money(other_sum), "Net": money(lt_sum + st_sum + other_sum), "Sales": len(year_sales)})
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
                st.dataframe(pd.DataFrame(filtered_sales[::-1]), hide_index=True, use_container_width=True)
                st.download_button(
                    "Download sales log (CSV)", data=pd.DataFrame(filtered_sales).to_csv(index=False),
                    file_name="sales_log.csv", mime="text/csv", key=f"sale_log_csv_{pf_name}",
                )
        # Tax-aware rebalance trades: only holdings with a target set.
        with st.expander("Rebalance trades — to reach your target weights", expanded=False):
            st.caption(
                "Trades at current prices to reach the Target % values in the editor. Prices move, so these are "
                "indicative, not orders. For Taxable portfolios, sell trades show estimated tax cost (from your "
                "rates and the holding's term); for other account types that column is —. Estimates only."
            )
            rebalance_rows = []
            targeted_value = 0.0
            for r in rows:
                if r.get("_target") is None or not r.get("_price_for_log"):
                    continue
                target_value = r["_target"] / 100.0 * total_account
                targeted_value += target_value
                trade_value = target_value - r["_value"]
                trade_shares = trade_value / r["_price_for_log"] if r["_price_for_log"] else None
                est_tax_cost = None
                if trade_value < 0 and is_taxable_pf and r.get("_avg_cost") and r["_price_for_log"] > r["_avg_cost"]:
                    gain_sold = (r["_price_for_log"] - r["_avg_cost"]) * abs(trade_shares) if trade_shares else 0.0
                    est_tax_cost = _estimated_tax(gain_sold, r.get("_term", "\u2014"), tax_rates)
                rebalance_rows.append({
                    "Ticker": r["Ticker"], "Current Value": money(r["_value"]),
                    "Target %": f"{r['_target']:.1f}%", "Target Value": money(target_value),
                    "Trade $": money(trade_value),
                    "Trade Shares": "\u2014" if trade_shares is None else f"{trade_shares:+.2f} ({'Buy' if trade_shares > 0 else 'Sell'})",
                    "Est Tax Cost": "\u2014" if est_tax_cost is None else money(est_tax_cost),
                })
            if not rebalance_rows:
                st.caption("No holdings have a Target % set yet — set one in the editor above.")
            else:
                st.dataframe(pd.DataFrame(rebalance_rows), hide_index=True, use_container_width=True)
                unallocated_pct = 100.0 - sum(r["_target"] for r in rows if r.get("_target") is not None)
                st.caption(f"Unallocated (cash + untargeted holdings): {unallocated_pct:.1f}% / {money(total_account - targeted_value)}. Targets need not sum to 100%.")
        # Signal log: what past signals said, and what prices did after.
        with st.expander("Signal log — past signals and what prices did after", expanded=False):
            signal_log = (load_user_data().get("signal_log") or [])
            if not signal_log:
                st.caption("No logged signals yet — today's check is logged above, once per ticker per day.")
            else:
                recent_log = signal_log[-50:][::-1]
                st.dataframe(pd.DataFrame(recent_log), hide_index=True, use_container_width=True)
                # Track record: BUY entries older than 30 days for tickers
                # still held, vs their current price. Price change since
                # the signal, NOT a realized return — you may not have bought.
                # Raw prices live on rows as _price_for_log (dropped from
                # the displayed df, still present in the rows list).
                current_prices = {r["Ticker"]: r["_price_for_log"] for r in rows if r.get("_price_for_log")}
                buy_tracked = []
                for entry in signal_log:
                    if entry.get("signal") != "BUY" or entry.get("ticker") not in current_prices:
                        continue
                    try:
                        entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
                    except Exception:
                        continue
                    days = (datetime.now().date() - entry_date).days
                    if days >= 30 and entry.get("price"):
                        buy_tracked.append((current_prices[entry["ticker"]] / entry["price"] - 1.0))
                if buy_tracked:
                    winners = sum(1 for x in buy_tracked if x > 0)
                    st.caption(
                        f"BUY signals older than 30 days (still-held tickers): {len(buy_tracked)} entries, "
                        f"average price change since signal {sum(buy_tracked) / len(buy_tracked):+.1%}, "
                        f"{winners}/{len(buy_tracked)} positive. Price change since signal, not a realized return."
                    )
                else:
                    st.caption("No BUY signals older than 30 days for currently held tickers yet.")
                log_df = pd.DataFrame(signal_log)
                st.download_button(
                    "Download signal log (CSV)", data=log_df.to_csv(index=False),
                    file_name="signal_log.csv", mime="text/csv", key="pf_log_csv",
                )
        export_data = load_user_data()
        export_data["portfolios"] = {pf_name: holdings}
        export_data["portfolio"] = holdings
        st.download_button(
            "Download portfolio + notes (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="portfolio.json",
            mime="application/json",
            key="pf_export_btn",
        )
        st.caption("Keep this file as your backup — on Streamlit Cloud the saved file can be wiped on sleep/restart, and this download is how you restore it (Import above).")


# --- 3. Watchlist ---------------------------------------------------------
with tab_watchlist:
    st.caption(
        "Tickers you're waiting to buy. BUY needs both a margin-of-safety price and a strong "
        "enough business. " + DISCLAIMER
    )
    w1, w2 = st.columns(2)
    with w1:
        min_moat_buy = st.slider("Minimum Moat Score to call it a BUY", 0, 100, 60, key="wl_moat")
    with w2:
        near_buy_pct = st.slider("'Near buy' band: within X% above the buy price", 0, 50, 15, key="wl_near")

    # Named watchlists, same pattern as named portfolios.
    saved_wl_all = load_user_data()
    watchlist_names = list((saved_wl_all.get("watchlists") or {"Main": []}).keys()) or ["Main"]
    # Same bug fix as portfolios above: wl_selected_name is plain state,
    # not the selectbox's widget key, so Add/Delete can set it and rerun
    # without Streamlit's post-instantiation widget-state error.
    if "wl_selected_name" not in st.session_state or st.session_state["wl_selected_name"] not in watchlist_names:
        st.session_state["wl_selected_name"] = watchlist_names[0]
    wl_name = st.selectbox("Watchlist", watchlist_names, index=watchlist_names.index(st.session_state["wl_selected_name"]))
    st.session_state["wl_selected_name"] = wl_name
    _last_saved_wl = last_saved_display()
    st.caption(
        ("Saved on this server: " + _last_saved_wl + ". " if _last_saved_wl else "No saved file on this server yet. ")
        + "Streamlit Cloud storage is ephemeral — a restart or sleep can wipe portfolios, watchlists, and notes. "
        "Use Export on the Portfolio tab after important changes to keep a permanent copy."
    )
    wl_c1, wl_c2, wl_c3 = st.columns(3)
    with wl_c1:
        new_wl_name = st.text_input("New watchlist name", key="wl_new_name").strip()
    with wl_c2:
        if st.button("Add watchlist", key="wl_add_btn") and new_wl_name:
            if new_wl_name in watchlist_names:
                st.warning(f"A watchlist named {new_wl_name} already exists.")
            elif save_watchlist([], name=new_wl_name):
                st.session_state["wl_selected_name"] = new_wl_name
                st.rerun()
    with wl_c3:
        if len(watchlist_names) > 1 and st.button(f"Delete watchlist '{wl_name}'", key="wl_del_btn"):
            st.warning("Deleting cannot be undone — export first if you want a backup.")
            if delete_watchlist(wl_name):
                st.session_state["wl_selected_name"] = watchlist_names[0]
                for k in list(st.session_state.keys()):
                    if k.startswith("watchlist_df_") or k.startswith("watchlist_saved_"):
                        st.session_state.pop(k, None)
                st.rerun()

    wl_df_key = f"watchlist_df_{wl_name}"
    wl_saved_key = f"watchlist_saved_{wl_name}"
    if wl_df_key not in st.session_state:
        saved_wl = (load_user_data().get("watchlists") or {}).get(wl_name, [])
        st.session_state[wl_df_key] = pd.DataFrame({"Ticker": pd.Series(saved_wl, dtype="object")})
        st.session_state[wl_saved_key] = saved_wl

    edited_w = st.data_editor(
        st.session_state[wl_df_key],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"watchlist_editor_{wl_name}",
        column_config={"Ticker": st.column_config.TextColumn("Ticker")},
    )
    watch = list(dict.fromkeys(
        t for t in (normalize_ticker(_cell(v) or "") for v in edited_w["Ticker"]) if t
    ))
    if watch != st.session_state[wl_saved_key]:
        if save_watchlist(watch, name=wl_name):
            st.session_state[wl_saved_key] = watch
        else:
            st.warning("Couldn't write user_data.json \u2014 changes won't persist after a restart.")

    if not watch:
        st.info("Add a ticker above to see its buy signal.")
    else:
        rows = []
        watch_full_why: List[Dict[str, Any]] = []
        watch_price_dates: List[Any] = []
        with st.spinner("Checking your watchlist..."):
            for t in watch:
                snap = build_snapshot(t, price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": t, "Price": "\u2014", "Sticker": "\u2014", "Buy Below": "\u2014", "% Above Buy Price": "\u2014",
                                 "Moat": "\u2014", "Mkt Cap": "\u2014", "PEG": "\u2014", "Owner Yield": "\u2014", "Data": "\u2014", "Flags": "\u2014",
                                 "Signal": "NO DATA", "Why": _short_why(snap["error"]), "_rank": 9, "_pct": 1e9})
                    continue
                val = valuation_from_snapshot(snap, discount_rate, max_growth)
                sig = buy_signal(snap, val, float(min_moat_buy), float(near_buy_pct))
                pct = sig["pct_above_buy"]
                comp_w = _data_completeness(snap["gatekeeper"])
                peg_w = _peg_ratio(snap)
                oey_w = _owner_earnings_yield(snap)
                det_w = _deterioration_flags(snap)
                if snap.get("price_date"):
                    watch_price_dates.append(snap["price_date"])
                watch_full_why.append({"Ticker": t, "Signal": sig["signal"], "Why": "; ".join(sig["reasons"])})
                rows.append({
                    "Ticker": t, "Price": money(snap["current_price"]), "Sticker": money(val["sticker_price"]),
                    "Buy Below": money(val["margin_of_safety_price"]),
                    "% Above Buy Price": "\u2014" if pct is None else f"{pct:+.1f}%",
                    "Moat": snap["moat_score"],
                    "Mkt Cap": _format_market_cap(_market_cap(snap)),
                    "PEG": "\u2014" if peg_w is None else f"{peg_w:.2f}",
                    "Owner Yield": "\u2014" if oey_w is None else f"{oey_w:.1%}",
                    "Data": f"{comp_w['metrics']}/5" + (" ⚠️" if comp_w["metrics"] <= 3 else ""),
                    "Flags": "; ".join(det_w) if det_w else "\u2014",
                    "Signal": sig["signal"], "Why": _short_why("; ".join(sig["reasons"])),
                    "_rank": BUY_RANK.get(sig["signal"], 9), "_pct": pct if pct is not None else 1e9,
                })
        rows.sort(key=lambda r: (r["_rank"], r["_pct"]))
        df = pd.DataFrame(rows).drop(columns=["_rank", "_pct"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        as_of_w = _prices_as_of_caption(watch_price_dates)
        if as_of_w:
            st.caption(as_of_w)
        with st.expander("Why, in full", expanded=False):
            for item in watch_full_why:
                st.markdown(f"**{item['Ticker']} — {item['Signal']}** — {item['Why']}")
        _table_downloads(df, f"wl_{wl_name}", f"{wl_name}_watchlist")
        counts = Counter(r["Signal"] for r in rows)
        st.caption(" \u00b7 ".join(f"{n} {s}" for s, n in counts.most_common()))
        st.caption("Add a watchlist ticker to a portfolio — shares and cost stay blank until you set them on the Portfolio tab.")
        _add_to_portfolio_controls([r["Ticker"] for r in rows], f"wl_{wl_name}")
        # Wash-sale watch for watchlist tickers too: if you buy a
        # ticker in a taxable account within 30 days of selling it at
        # a loss, the loss may be disallowed.
        today_wl = datetime.now().date()
        wl_wash = []
        for entry in (load_user_data().get("sales_log") or []):
            if entry.get("gain") is None or entry["gain"] >= 0 or entry.get("ticker") not in watch:
                continue
            try:
                sale_d_wl = datetime.strptime(entry["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if 0 <= (today_wl - sale_d_wl).days <= 30:
                wl_wash.append(entry)
        if wl_wash:
            st.warning(
                "Wash-sale risk on this watchlist — sold at a loss in the last 30 days: "
                + ", ".join(f"{e['ticker']} (sold {e['date']})" for e in wl_wash)
                + ". Buying it in a taxable account within 30 days of that sale may disallow the loss."
            )


# --- 4 & 5. Market scanner --------------------------------------------------
def render_ranked_table(scan: Dict[str, Any], min_dividend_years: int, min_moat: float, quality_first: bool = True, download_key: str = "scan_csv", dividend_mode: bool = False, max_payout: Optional[float] = None, min_div_growth: Optional[float] = None, min_chowder: Optional[float] = None, min_market_cap_b: float = 0.0) -> None:
    # Dividend tab only (Miller / Van Knapp layer): filter snapshots
    # before ranking. A missing metric is shown as "—" and does NOT
    # exclude a stock — SEC dividend-per-share coverage is imperfect,
    # so unknown is not treated as failing.
    snapshots_for_rank = scan["snapshots"]
    # Market-cap filter (all scan tabs): unknown market cap does NOT
    # exclude a stock, for the same reason as above.
    if min_market_cap_b and min_market_cap_b > 0:
        cap_filtered = []
        for snap in snapshots_for_rank:
            if snap.get("error"):
                cap_filtered.append(snap)
                continue
            cap = _market_cap(snap)
            if cap is not None and cap < min_market_cap_b * 1e9:
                continue
            cap_filtered.append(snap)
        snapshots_for_rank = cap_filtered
    as_of_scan = _prices_as_of_caption([s.get("price_date") for s in snapshots_for_rank if not s.get("error")])
    if as_of_scan:
        st.caption(as_of_scan + " A cached scan keeps its prices until you re-scan.")
    if dividend_mode:
        # Apply manual dividend data first, so manual values count for
        # filters, Chowder, and the safety score — and are flagged ✏️.
        snapshots_for_rank = [_apply_manual_to_snapshot(s) for s in snapshots_for_rank]
        filtered = []
        for snap in snapshots_for_rank:
            if snap.get("error"):
                filtered.append(snap)
                continue
            payout = snap.get("payout_ratio")
            if max_payout is not None and payout is not None and payout > max_payout / 100.0:
                continue
            dgr = snap.get("dividend_growth_5yr")
            if min_div_growth is not None and dgr is not None and dgr < min_div_growth / 100.0:
                continue
            chow = snap.get("chowder")
            if min_chowder is not None and chow is not None and chow < min_chowder / 100.0:
                continue
            filtered.append(snap)
        snapshots_for_rank = filtered
    if quality_first:
        stepped = stepped_rank_snapshots(
            snapshots_for_rank, discount_rate, max_growth, min_dividend_years=min_dividend_years, min_moat_score=min_moat, top_n=25
        )
        ranked = stepped["ranked"]
        hurdle_text = (
            f"Big Five hurdle used: {stepped['hurdle_used']:.0%} (stepped down from 10% in 1% steps)"
            if stepped["stepped_down"] and stepped["hurdle_used"] is not None
            else f"Big Five hurdle used: {stepped['hurdle_used']:.0%} — wonderful businesses first, then price"
            if stepped["hurdle_used"] is not None
            else "No stock passed 4 of 5 Big Five even at 5%"
        )
    else:
        ranked = rank_snapshots(
            snapshots_for_rank, discount_rate, max_growth, min_dividend_years=min_dividend_years, min_moat_score=min_moat, top_n=25
        )
        hurdle_text = "Quality-first off — ranking by price alone (moat filter only)"
    st.caption(
        f"{scan['label']}: scanned {scan['done']} of {scan['total']} tickers \u2014 {len(scan['snapshots'])} had usable "
        f"data, {ranked['valued_count']} could be valued, {ranked['matched_count']} match your filters. {hurdle_text}."
    )
    if dividend_mode:
        st.caption(
            "Dividend tab only — Lowell Miller, The Single Best Investment, plus David Van Knapp, plus your "
            "10-criteria screener: payout generally under 65% (Miller: under 60% non-utilities), dividend growth "
            "at least 4% (Miller minimum), Chowder = yield + 5-year dividend growth (Van Knapp / Chowder rule, "
            "8%+ favored), and 10-by-10 = years for yield-on-cost to reach 10%. Van Knapp's full Quality Snapshots "
            "score uses paid data (Value Line, Morningstar moat, S&P ratings, Simply Safe Dividends) that SEC filings "
            "don't carry, so this tab uses the app's Moat Score / Debt Test / free-cash-flow checks as its quality proxy, "
            "not his exact score. Stocks with a missing dividend metric show \u2014 and are not excluded for it. "
            "Safety Score is the app's own 0–100 score (payout/FCF cover 25, balance sheet 20, dividend history 20, "
            "dividend growth 20, earnings quality 15; 80+ Very Safe, 60+ Safe, 40+ Borderline, 20+ Unsafe). "
            "A * on the Safety Score means a component scored 0 for missing data, not necessarily for risk — check the "
            "Manual column. Any cell with \u270f\ufe0f is a manual value you entered, not pulled data."
        )
    if scan.get("aborted"):
        st.warning(scan["aborted"])
    if quality_first and stepped["schd_fallback"]:
        st.info(
            "No candidates passed 4 of 5 Big Five at 10% down to 5% — per your locked rule, "
            "SCHD is the fallback for an empty / disqualified list. SCHD is an ETF, not a "
            "10-K filer, so it has no Sticker Price here; this is a fallback message, not a valuation."
        )
    if not ranked["top_results"]:
        st.info("Nothing matched. Loosen the filters, or run a wider scan.")
    else:
        rows = []
        snap_map_all = {s.get("ticker"): s for s in snapshots_for_rank}
        for i, r in enumerate(ranked["top_results"], start=1):
            note_scan = _notes_for(r["ticker"])
            snap_row = snap_map_all.get(r["ticker"], {})
            comp_row = _data_completeness(snap_row.get("gatekeeper") or {}) if snap_row else {"metrics": 0, "horizons": 0, "missing_metrics": []}
            peg_row = _peg_ratio(snap_row) if snap_row else None
            oey_row = _owner_earnings_yield(snap_row) if snap_row else None
            row: Dict[str, Any] = {
                "#": i, "Ticker": r["ticker"], "Company": r["name"] or "", "Price": money(r["current_price"]),
                "Sticker": money(r["sticker_price"]), "MOS Price": money(r["margin_of_safety_price"]),
                "% Below Sticker": f"{r['undervalued_pct']:.1f}%", "Growth Used": f"{r['growth_used']:.1%}",
                "Moat": r["moat_score"], "Big Five": f"{r.get('big_five_passes', 0)}/5",
                "Meaning": note_scan.get("meaning") or "\u2014", "Moat type": note_scan.get("moat_type") or "\u2014",
                "Action": r["action"].replace("_", " "), "Div. Streak (yrs)": r["dividend_streak"],
                "Mkt Cap": _format_market_cap(_market_cap(snap_row)) if snap_row else "\u2014",
                "PEG": "\u2014" if peg_row is None else f"{peg_row:.2f}",
                "Owner Yield": "\u2014" if oey_row is None else f"{oey_row:.1%}",
                "Data": f"{comp_row['metrics']}/5" + (" ⚠️" if comp_row["metrics"] <= 3 else ""),
            }
            if dividend_mode:
                payout = r.get("payout_ratio")
                fcf_cover = r.get("fcf_dividend_coverage")
                if payout is not None and payout > 1.0:
                    div_flag = "\u26a0 Dividend in jeopardy (payout >100%)"
                elif fcf_cover is not None and fcf_cover < 1.0:
                    div_flag = "\u26a0 FCF doesn't cover dividend"
                elif payout is not None and payout > 0.65:
                    div_flag = "Payout high"
                else:
                    div_flag = "\u2014"
                years_to_10 = r.get("years_to_10pct_yield_on_cost")
                # Safety score from the (manual-applied) snapshot, with
                # Big Five passes measured at a fixed 10% hurdle — not
                # the stepped hurdle used for ranking — so the score
                # doesn't inflate when the hurdle steps down.
                snap_for_score = {s.get("ticker"): s for s in snapshots_for_rank}.get(r["ticker"], {})
                manual_fields = snap_for_score.get("manual_fields", []) or []
                debt_ratio = snap_for_score.get("debt_ratio_manual")
                if debt_ratio is None:
                    debt_ratio = ((snap_for_score.get("gatekeeper") or {}).get("debt_test") or {}).get("ratio")
                try:
                    bf_passes_10 = big_five_pass_count(snap_for_score, 0.10) if snap_for_score else None
                except Exception:
                    bf_passes_10 = None
                sector_manual = snap_for_score.get("sector_manual")
                safety = dividend_safety_score(
                    payout, fcf_cover, debt_ratio, r.get("dividend_streak"), r.get("dividend_growth_5yr"), bf_passes_10,
                    sector=sector_manual,
                )
                chowder_label = _manual_label(manual_fields, "dividend_growth_5yr") or _manual_label(manual_fields, "dividend_per_share")
                row["Div. Streak (yrs)"] = f"{r['dividend_streak']}{_manual_label(manual_fields, 'dividend_streak')}"
                row.update(
                    {
                        "Yield": _pct_or_dash(r.get("dividend_yield")) + _manual_label(manual_fields, "dividend_per_share"),
                        "Div Growth 5yr": _pct_or_dash(r.get("dividend_growth_5yr")) + _manual_label(manual_fields, "dividend_growth_5yr"),
                        "Chowder": _pct_or_dash(r.get("chowder")) + chowder_label,
                        "Payout": _pct_or_dash(payout) + _manual_label(manual_fields, "payout_ratio"),
                        "FCF Cover": ("\u2014" if fcf_cover is None else f"{fcf_cover:.1f}x") + _manual_label(manual_fields, "fcf_dividend_coverage"),
                        "Yrs to 10% YoC": "\u2014" if years_to_10 is None else ("Already \u226510%" if years_to_10 == 0 else f"{years_to_10:.0f} yrs"),
                        "Div. flag": div_flag,
                        "Safety Score": f"{safety['score']} {safety['band']}" + (" *" if safety["missing"] else ""),
                        "Manual": "\u270f\ufe0f " + ", ".join(manual_fields) if manual_fields else "\u2014",
                    }
                )
                # DDM + P/E cross-check, dividend tab only. DDM values
                # only the dividend stream, so it is usually below the
                # EPS sticker — agreement / disagreement is the signal,
                # not an average of the two values.
                dps_row = r.get("dividend_per_share")
                dgr_row = r.get("dividend_growth_5yr")
                ddm = calculate_ddm_value(dps_row, dgr_row, discount_rate) if (dps_row is not None and dgr_row is not None) else None
                eps_row = snap_for_score.get("current_eps")
                price_row = r.get("current_price")
                current_pe = (price_row / eps_row) if (price_row and eps_row and eps_row > 0) else None
                hist_pe = snap_for_score.get("historical_pe")
                if ddm is not None and price_row is not None:
                    # Fair-value comparison on both sides: price below
                    # the Rule #1 sticker vs price below the DDM value.
                    rule1_under = price_row <= (r.get("sticker_price") or 0)
                    ddm_under = price_row <= ddm
                    agreement = "Agree" if rule1_under == ddm_under else "\u26a0 Disagree"
                else:
                    agreement = "\u2014"
                row.update(
                    {
                        "DDM Value": money(ddm),
                        "DDM vs Price": "\u2014" if (ddm is None or not price_row) else f"{(ddm - price_row) / ddm:+.1f}%",
                        "Current P/E": "\u2014" if current_pe is None else f"{current_pe:.1f}",
                        "Hist P/E": "\u2014" if hist_pe is None else f"{hist_pe:.1f}",
                        "P/E vs Hist": "\u2014" if (current_pe is None or not hist_pe) else f"{current_pe / hist_pe - 1:+.1%}",
                        "Rule1/DDM": agreement,
                        "Sector": (sector_manual + " ✏️") if sector_manual else "\u2014",
                    }
                )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        _table_downloads(pd.DataFrame(rows), download_key, "top25")
    if scan["errors"]:
        reasons = Counter(scan["errors"].values())
        with st.expander(f"{len(scan['errors'])} tickers skipped \u2014 why"):
            for reason, n in reasons.most_common():
                sample = [t for t, m in scan["errors"].items() if m == reason][:12]
                st.write(f"**{n}** \u00d7 {reason}  \u2014  e.g. {', '.join(sample)}")
    # Tickers that had usable data but still couldn't be valued — the
    # gap between "had usable data" and "could be valued" in the caption
    # above. Same four branches as valuation_from_snapshot, grouped so
    # the reason is visible instead of a silent absence from the table.
    # Computed over this tab's filtered snapshot set, matching the
    # ranking above.
    unvalued: Dict[str, List[str]] = {}
    for snap in snapshots_for_rank:
        if snap.get("error"):
            continue
        eps = snap.get("current_eps")
        growth = snap.get("growth_rate")
        pe = snap.get("historical_pe")
        if eps is None or eps <= 0:
            reason = "EPS is zero or negative, so no Sticker Price"
        elif growth is None:
            reason = "Not enough EPS history to estimate growth"
        elif growth <= 0:
            reason = "EPS hasn't been growing, so no Sticker Price"
        elif pe is None:
            reason = "No historical P/E (couldn't pair past EPS with past prices)"
        else:
            continue
        unvalued.setdefault(reason, []).append(snap.get("ticker") or "?")
    if unvalued:
        total_unvalued = sum(len(v) for v in unvalued.values())
        with st.expander(f"{total_unvalued} tickers had data but couldn't be valued \u2014 why"):
            st.caption(
                "These are not fetch failures (those are in the skipped-tickers expander above) — the filings "
                "came back, but the Rule #1 Sticker can't be computed for them. A loss-maker, a company with no "
                "EPS growth history, or one whose past EPS can't be paired with past prices simply has no Sticker "
                "here; that is a limitation of the method, not a buy or sell signal."
            )
            for reason, tickers in sorted(unvalued.items(), key=lambda kv: -len(kv[1])):
                st.write(f"**{len(tickers)}** \u00d7 {reason}  \u2014  e.g. {', '.join(tickers[:12])}")
    if ranked["top_results"]:
        st.caption("Add a result to a portfolio — shares and cost stay blank until you set them on the Portfolio tab.")
        _add_to_portfolio_controls([r["ticker"] for r in ranked["top_results"]], download_key)
    return ranked


with tab_undervalued:
    st.caption(
        "Scans a universe and ranks wonderful businesses first, then by how far each stock trades below its "
        "Sticker Price. Results are cached, so you can stop a scan and click Scan again to resume. " + DISCLAIMER
    )
    st.caption(
        "Read this list as a valuation discipline, not a buy signal on its own. Backtests of this exact method "
        "on the S&P 500, 2010\u20132020 (local, 2026-11-04): a Sticker at a 50% margin of safety, with the Big Five "
        "quality gate, did not beat the S&P 500 \u2014 it landed a few points a year behind it, and concentrating on "
        "the 3 most undervalued names made it no better. The names here are worth a closer look only if you "
        "understand the business behind them."
    )
    # Restore the last saved scan on a fresh session (Cloud may still wipe
    # the file on sleep/restart — the CSV download below is the durable copy).
    if "scan" not in st.session_state:
        last_scan, last_saved_at = load_last_scan()
        if last_scan is not None:
            st.session_state["scan"] = last_scan
            if last_saved_at:
                partial_note = ""
                if last_scan.get("aborted") and "Partial scan" in str(last_scan.get("aborted")):
                    partial_note = f" — partial scan ({last_scan.get('done', 0)} of {last_scan.get('total', 0)} tickers). Click Scan to resume it."
                st.caption(f"Loaded last saved scan from {datetime.fromtimestamp(last_saved_at).strftime('%Y-%m-%d %H:%M')}.{partial_note} Click Scan to refresh it.")
    choice = st.radio("Universe", ["S&P 500", "All NYSE + Nasdaq stocks", "Custom list"], horizontal=True, key="scan_universe_choice")
    custom_text = ""
    if choice == "Custom list":
        custom_text = st.text_area("Tickers (comma or newline separated)", height=100, key="scan_custom")
    if choice == "All NYSE + Nasdaq stocks":
        st.caption(
            "Every common-stock listing on NYSE and Nasdaq (~6,000 tickers). The first full scan makes thousands of SEC "
            "and Yahoo requests and can take an hour or more; after that it's cached for a day."
        )

    if st.button("Scan", key="scan_button"):
        if choice == "S&P 500":
            universe = get_sp500_tickers()
        elif choice == "All NYSE + Nasdaq stocks":
            universe = get_all_us_listed_tickers()
        else:
            universe = parse_ticker_list(custom_text)
        if choice != "Custom list" and len(universe) <= 60:
            st.warning(f"Only got {len(universe)} tickers for this universe \u2014 the live list couldn't be fetched, so this is a small fallback list.")
        if not universe:
            st.warning("Enter at least one ticker.")
        else:
            bar = st.progress(0.0)
            status = st.empty()

            def on_progress(done: int, total: int, ticker: str) -> None:
                bar.progress(done / total if total else 1.0)
                status.caption(f"Checked {done}/{total} (latest: {ticker})")

            def on_partial(partial: Dict[str, Any]) -> None:
                # Saved to disk every 25 tickers, so closing the browser
                # mid-scan no longer loses the ranked table.
                partial_with_label = dict(partial)
                partial_with_label["label"] = choice
                save_last_scan(partial_with_label)

            # Keep the top 25 the user was looking at, so the diff below
            # can compare the new scan against it (same filters, new data).
            # prev_top25 is refreshed after every render of the Top 25
            # tab, so it always holds the list as last displayed.
            if st.session_state.get("prev_top25"):
                st.session_state["scan_diff_prev"] = dict(st.session_state["prev_top25"])
            else:
                st.session_state.pop("scan_diff_prev", None)
            result = scan_snapshots(universe, progress_callback=on_progress, partial_callback=on_partial)
            bar.empty()
            status.empty()
            result["label"] = choice
            st.session_state["scan"] = result
            save_last_scan(result)

    quality_first_scan = st.checkbox(
        "Wonderful businesses first: require 4 of 5 Big Five at the hurdle, stepping down 1% from 10% to 5% if empty, SCHD fallback if still empty",
        value=True, key="scan_quality_first",
    )
    min_moat_scan = st.slider("Minimum Moat Score (filter)", 0, 100, 0, key="scan_moat")
    min_cap_scan = st.number_input("Minimum market cap ($ billions, 0 = no minimum; unknown cap does not exclude)", 0.0, 10000.0, 0.0, key="scan_mincap")
    # Saved filter presets (Undervalued tab only): selecting one
    # overrides the sliders above, so a favourite filter set is one
    # click instead of re-setting every slider.
    with st.expander("Filter presets", expanded=False):
        uv_presets = (load_user_data().get("scan_presets") or {}).get("undervalued", {}) or {}
        uv_preset_name = st.selectbox("Preset (overrides sliders while selected)", ["(sliders)"] + list(uv_presets.keys()), key="uv_preset_sel")
        uv_new_name = st.text_input("Save current filters as preset", key="uv_preset_new").strip()
        if st.button("Save preset", key="uv_preset_save") and uv_new_name:
            if save_scan_preset("undervalued", uv_new_name, {"quality_first": bool(quality_first_scan), "min_moat": float(min_moat_scan), "min_cap_b": float(min_cap_scan)}):
                st.success(f"Saved preset '{uv_new_name}'.")
                st.rerun()
        if uv_preset_name != "(sliders)" and st.button("Delete preset", key="uv_preset_del"):
            if delete_scan_preset("undervalued", uv_preset_name):
                st.rerun()
    if uv_preset_name != "(sliders)":
        uv_p = uv_presets.get(uv_preset_name, {}) or {}
        quality_first_scan = bool(uv_p.get("quality_first", True))
        min_moat_scan = float(uv_p.get("min_moat", 0))
        min_cap_scan = float(uv_p.get("min_cap_b", 0))
        st.caption(f"Using preset '{uv_preset_name}' — sliders are ignored until you switch back to (sliders).")
    if "scan" in st.session_state:
        ranked_now = render_ranked_table(st.session_state["scan"], 0, float(min_moat_scan), quality_first=quality_first_scan, download_key="scan_csv", min_market_cap_b=float(min_cap_scan))
        # Refresh the stored top 25 for the next scan's diff, then show
        # the diff against the previous scan (if this session has one).
        st.session_state["prev_top25"] = {r["ticker"]: r["undervalued_pct"] for r in ranked_now["top_results"]}
        _render_scan_diff(st.session_state.get("scan_diff_prev"), st.session_state["prev_top25"])
    else:
        st.info("Pick a universe and click Scan.")

with tab_dividends:
    st.caption(
        "The same scan as the previous tab (run it there first), keeping only companies that have raised their "
        "dividend per share for at least this many years in a row. This tab ONLY adds the dividend-growth layer: "
        "Lowell Miller, The Single Best Investment, and David Van Knapp's dividend-growth teachings, plus your "
        "10-criteria screener. The other tabs do not use Miller / Van Knapp."
    )
    div_years = st.slider("Minimum consecutive years of dividend growth (5+ required, 10+ preferred in your screener)", 1, 25, 5, key="div_years")
    min_moat_div = st.slider("Minimum Moat Score (filter)", 0, 100, 0, key="div_moat")
    max_payout_div = st.slider("Maximum payout ratio — DPS / EPS (your screener: generally under 65%; Miller: under 60% non-utilities)", 0, 100, 65, key="div_payout")
    min_div_growth_div = st.slider("Minimum 5-year dividend growth (Miller minimum: 4%)", 0, 20, 4, key="div_growth")
    min_chowder_div = st.slider("Minimum Chowder — yield + 5yr dividend growth (8%+ favored; 0 = show all, highlight in table)", 0, 15, 0, key="div_chowder")
    min_cap_div = st.number_input("Minimum market cap ($ billions, 0 = no minimum; unknown cap does not exclude)", 0.0, 10000.0, 0.0, key="div_mincap")
    quality_first_div = st.checkbox(
        "Wonderful businesses first (same hurdle step-down as the Undervalued tab)",
        value=True, key="div_quality_first",
    )
    with st.expander("Filter presets", expanded=False):
        div_presets = (load_user_data().get("scan_presets") or {}).get("dividend", {}) or {}
        div_preset_name = st.selectbox("Preset (overrides sliders while selected)", ["(sliders)"] + list(div_presets.keys()), key="div_preset_sel")
        div_new_name = st.text_input("Save current filters as preset", key="div_preset_new").strip()
        if st.button("Save preset", key="div_preset_save") and div_new_name:
            if save_scan_preset("dividend", div_new_name, {
                "div_years": int(div_years), "min_moat": float(min_moat_div), "max_payout": float(max_payout_div),
                "min_div_growth": float(min_div_growth_div), "min_chowder": float(min_chowder_div),
                "min_cap_b": float(min_cap_div), "quality_first": bool(quality_first_div),
            }):
                st.success(f"Saved preset '{div_new_name}'.")
                st.rerun()
        if div_preset_name != "(sliders)" and st.button("Delete preset", key="div_preset_del"):
            if delete_scan_preset("dividend", div_preset_name):
                st.rerun()
    if div_preset_name != "(sliders)":
        dv_p = div_presets.get(div_preset_name, {}) or {}
        div_years = int(dv_p.get("div_years", 5))
        min_moat_div = float(dv_p.get("min_moat", 0))
        max_payout_div = float(dv_p.get("max_payout", 65))
        min_div_growth_div = float(dv_p.get("min_div_growth", 4))
        min_chowder_div = float(dv_p.get("min_chowder", 0))
        min_cap_div = float(dv_p.get("min_cap_b", 0))
        quality_first_div = bool(dv_p.get("quality_first", True))
        st.caption(f"Using preset '{div_preset_name}' — sliders are ignored until you switch back to (sliders).")
    with st.expander("\u270f\ufe0f Manual dividend data — fill in a field the SEC pull missed"):
        st.caption(
            "Use this when a metric shows \u2014 in the table but you found it yourself (annual report, company "
            "investor page, broker). Leave a field blank to keep the pulled value. Anything you enter here is "
            "flagged \u270f\ufe0f in the table and counts for the filters and the Safety Score — it is never shown "
            "as pulled data. Manual data is saved with your portfolio export/import, so back it up the same way."
        )
        manual_ticker = st.text_input("Ticker", key="manual_ticker").strip().upper()
        existing_manual = _manual_for(manual_ticker) if manual_ticker else {}
        if existing_manual:
            st.caption(f"Already saved for {manual_ticker}: " + ", ".join(f"{k}={v}" for k, v in existing_manual.items()))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            manual_dps = st.text_input("Dividend per share, latest annual ($)", key="manual_dps", placeholder="e.g. 1.61")
            manual_payout = st.text_input("Payout ratio (%)", key="manual_payout", placeholder="e.g. 54 for 54%")
            manual_growth = st.text_input("5-year dividend growth (%)", key="manual_growth", placeholder="e.g. 10 for 10%")
        with col_m2:
            manual_streak = st.text_input("Consecutive years of dividend increases", key="manual_streak", placeholder="e.g. 12")
            manual_fcf = st.text_input("FCF dividend coverage (x)", key="manual_fcf", placeholder="e.g. 1.8")
            manual_debt = st.text_input("Debt ratio — LT debt / owner earnings (x)", key="manual_debt", placeholder="e.g. 1.5")
            manual_sector_options = ["Not set"] + list(MANUAL_SECTOR_OPTIONS)
            existing_sector = existing_manual.get("sector") if existing_manual else None
            manual_sector = st.selectbox(
                "Sector tag — only changes safety-score payout tiers (Utility up to 70–85%, REIT up to 85–95% on EPS; REIT EPS payout is not FFO, so treat with caution)",
                manual_sector_options,
                index=manual_sector_options.index(existing_sector) if existing_sector in manual_sector_options else 0,
                key="manual_sector",
            )

        def _parse_manual_pct(text: str) -> Optional[float]:
            try:
                return float(text) / 100.0
            except (TypeError, ValueError):
                return None

        def _parse_manual_num(text: str) -> Optional[float]:
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        if st.button("Save manual dividend data", key="manual_save_btn"):
            if not manual_ticker:
                st.warning("Enter a ticker first.")
            else:
                fields: Dict[str, Any] = {}
                if manual_dps.strip():
                    v = _parse_manual_num(manual_dps)
                    if v is not None:
                        fields["dividend_per_share"] = v
                if manual_payout.strip():
                    v = _parse_manual_pct(manual_payout)
                    if v is not None:
                        fields["payout_ratio"] = v
                if manual_growth.strip():
                    v = _parse_manual_pct(manual_growth)
                    if v is not None:
                        fields["dividend_growth_5yr"] = v
                if manual_streak.strip():
                    try:
                        fields["dividend_streak"] = int(manual_streak)
                    except ValueError:
                        pass
                if manual_fcf.strip():
                    v = _parse_manual_num(manual_fcf)
                    if v is not None:
                        fields["fcf_dividend_coverage"] = v
                if manual_debt.strip():
                    v = _parse_manual_num(manual_debt)
                    if v is not None:
                        fields["debt_ratio"] = v
                if manual_sector != "Not set":
                    fields["sector"] = manual_sector
                if not fields:
                    st.warning("No valid numbers entered — nothing saved.")
                elif save_manual(manual_ticker, fields):
                    st.success(f"Saved manual data for {manual_ticker} — it will show \u270f\ufe0f in the table. Re-run or refresh the table to see it.")
                else:
                    st.error("Couldn't save manual data.")
        if st.button("Clear manual data for this ticker", key="manual_clear_btn"):
            if manual_ticker and clear_manual(manual_ticker):
                st.success(f"Cleared manual data for {manual_ticker}.")
            else:
                st.warning("Enter a ticker with saved manual data first.")
    if "scan" in st.session_state:
        render_ranked_table(
            st.session_state["scan"], int(div_years), float(min_moat_div), quality_first=quality_first_div,
            download_key="div_csv", dividend_mode=True, max_payout=float(max_payout_div),
            min_div_growth=float(min_div_growth_div), min_chowder=float(min_chowder_div),
            min_market_cap_b=float(min_cap_div),
        )
        st.caption(
            "Note: dividend metrics come from fresh SEC filings. A scan cached before this update won't have them "
            "(shows \u2014) — click Refresh data on the Ticker Lookup tab, or wait for the 24-hour cache to expire, "
            "then re-run the scan."
        )
    else:
        st.info("Run a scan on the Top 25 Undervalued tab first; this tab reuses its results.")
