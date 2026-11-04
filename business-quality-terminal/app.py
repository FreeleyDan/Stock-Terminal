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

import json
import math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from calculations import calculate_ddm_value, calculate_sticker_price
from data_fetcher import clear_cache, normalize_ticker
from gatekeeper import BIG_FIVE_METRICS, HORIZONS
from screener import big_five_pass_count, dividend_safety_score, rank_snapshots, scan_snapshots, stepped_rank_snapshots
from signals import BUY_RANK, PORTFOLIO_RANK, buy_signal, portfolio_signal, sell_signal
from storage import (
    MANUAL_SECTOR_OPTIONS,
    clear_manual,
    load_last_scan,
    load_user_data,
    parse_imported_data,
    save_last_scan,
    save_manual,
    save_note,
    save_portfolio,
    save_user_data,
    save_watchlist,
)
from universe import get_all_us_listed_tickers, get_sp500_tickers
from valuation import DEFAULT_MAX_GROWTH_RATE, build_snapshot, valuation_from_snapshot

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
    if snap.get("dividend_streak"):
        details.append(f"dividend raised {snap['dividend_streak']} yrs in a row")
    st.caption(" \u00b7 ".join(details))
    if val["note"]:
        st.caption(val["note"])
    st.dataframe(build_checklist_dataframe(snap["gatekeeper"]), hide_index=True, use_container_width=True)

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

tab_lookup, tab_portfolio, tab_watchlist, tab_undervalued, tab_dividends = st.tabs(
    ["Ticker Lookup", "Portfolio", "Watchlist", "Top 25 Undervalued", "Top 25 + Dividend Growers"]
)

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
    for ticker in lookup_to_fetch:
        with st.spinner(f"Fetching {ticker}..."):
            snap = build_snapshot(ticker, price_max_age=LOOKUP_PRICE_MAX_AGE)
        val = None if snap.get("error") else valuation_from_snapshot(snap, discount_rate, max_growth)
        render_ticker_card(ticker, snap, val, max_growth, discount_rate)


# --- 2. Portfolio ---------------------------------------------------------
def clean_portfolio_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for _, r in df.iterrows():
        ticker = normalize_ticker(_cell(r.get("Ticker")) or "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        rows.append({"ticker": ticker, "shares": _num(r.get("Shares")), "avg_cost": _num(r.get("Avg Cost"))})
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

    with st.expander("Import a portfolio file (JSON)", expanded=False):
        st.caption("Import replaces the saved portfolio, watchlist, and notes with the file's contents. Export first if you want a backup. Manual dividend data is replaced only if the file contains it — otherwise your existing manual data is kept.")
        uploaded_pf = st.file_uploader("Portfolio JSON file", type=["json"], key="pf_import_file")
        if uploaded_pf is not None and st.button("Import this file", key="pf_import_btn"):
            try:
                imported_portfolio, imported_watchlist, imported_notes, imported_manual = parse_imported_data(json.load(uploaded_pf))
                existing_manual = (load_user_data().get("manual") or {}) if not imported_manual else imported_manual
                save_user_data({"portfolio": imported_portfolio, "watchlist": imported_watchlist, "notes": imported_notes, "manual": existing_manual})
                frame_imported = pd.DataFrame(
                    [{"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": h["avg_cost"]} for h in imported_portfolio],
                    columns=["Ticker", "Shares", "Avg Cost"],
                )
                frame_imported["Shares"] = pd.to_numeric(frame_imported["Shares"], errors="coerce").astype("float64")
                frame_imported["Avg Cost"] = pd.to_numeric(frame_imported["Avg Cost"], errors="coerce").astype("float64")
                st.session_state["portfolio_df"] = frame_imported
                st.session_state["portfolio_saved"] = imported_portfolio
                st.session_state.pop("watchlist_df", None)
                st.session_state.pop("watchlist_saved", None)
                st.success(f"Imported {len(imported_portfolio)} holdings and {len(imported_watchlist)} watchlist tickers.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    if "portfolio_df" not in st.session_state:
        saved = load_user_data()
        frame = pd.DataFrame(
            [{"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": h["avg_cost"]} for h in saved["portfolio"]],
            columns=["Ticker", "Shares", "Avg Cost"],
        )
        frame["Shares"] = pd.to_numeric(frame["Shares"], errors="coerce").astype("float64")
        frame["Avg Cost"] = pd.to_numeric(frame["Avg Cost"], errors="coerce").astype("float64")
        st.session_state["portfolio_df"] = frame
        st.session_state["portfolio_saved"] = saved["portfolio"]

    edited = st.data_editor(
        st.session_state["portfolio_df"],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="portfolio_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0.0),
            "Avg Cost": st.column_config.NumberColumn("Avg Cost", min_value=0.0, format="$%.2f"),
        },
    )
    holdings = clean_portfolio_rows(edited)
    if holdings != st.session_state["portfolio_saved"]:
        if save_portfolio(holdings):
            st.session_state["portfolio_saved"] = holdings
        else:
            st.warning("Couldn't write user_data.json \u2014 changes won't persist after a restart.")

    if not holdings:
        st.info("Add a ticker above to see its buy/hold/sell signal.")
    else:
        rows, total_value, total_cost, total_div_income = [], 0.0, 0.0, 0.0
        review_items: List[Dict[str, Any]] = []
        projection_items: List[Dict[str, Any]] = []
        projection_excluded: List[str] = []
        projection_cost_included = 0.0
        with st.spinner("Checking your holdings..."):
            for h in holdings:
                snap = build_snapshot(h["ticker"], price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": "\u2014",
                                 "Value": "\u2014", "Gain/Loss": "\u2014", "Gain %": "\u2014", "Sticker": "\u2014", "Moat": "\u2014",
                                 "Meaning": "\u2014", "Moat type": "\u2014", "Weight": "\u2014", "Div Income/yr": "\u2014",
                                 "Yield on Cost": "\u2014", "Flags": "\u2014", "Mkt Cap": "\u2014", "PEG": "\u2014", "Data": "\u2014",
                                 "Signal": "NO DATA", "Why": snap["error"], "_rank": 9, "_value": 0.0, "_moat": 0.0})
                    review_items.append({"Ticker": h["ticker"], "Signal": "NO DATA", "Reason": snap["error"], "_urgency": 0})
                    continue
                val = valuation_from_snapshot(snap, discount_rate, max_growth)
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
                if review_reasons:
                    review_items.append({"Ticker": h["ticker"], "Signal": sig["signal"], "Reason": "; ".join(review_reasons), "_urgency": review_urgency})
                # Dividend income projection: current income grown at
                # the 5-year dividend growth rate. Illustration only.
                if div_income is not None and dgr_pf is not None and dgr_pf >= 0:
                    inc5 = div_income * ((1.0 + dgr_pf) ** 5)
                    inc10 = div_income * ((1.0 + dgr_pf) ** 10)
                    yoc10 = ((dps_pf * ((1.0 + dgr_pf) ** 10)) / h["avg_cost"]) if (dps_pf and h["avg_cost"]) else None
                    projection_items.append({
                        "Ticker": h["ticker"], "Income now": money(div_income),
                        "Income in 5 yrs": money(inc5), "Income in 10 yrs": money(inc10),
                        "Yield on Cost in 10 yrs": "\u2014" if yoc10 is None else f"{yoc10:.1%}",
                        "_inc10": inc10,
                    })
                    if h["shares"] and h["avg_cost"]:
                        projection_cost_included += h["shares"] * h["avg_cost"]
                elif div_income is not None:
                    projection_excluded.append(h["ticker"])
                rows.append({
                    "Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": money(price),
                    "Value": money(value), "Gain/Loss": money(gain),
                    "Gain %": "\u2014" if gain_pct is None else f"{gain_pct:+.1f}%",
                    "Sticker": money(val["sticker_price"]), "Moat": snap["moat_score"],
                    "Meaning": note_pf.get("meaning") or "\u2014", "Moat type": note_pf.get("moat_type") or "\u2014",
                    "Weight": "\u2014", "Div Income/yr": money(div_income) + dps_label if div_income is not None else "\u2014",
                    "Yield on Cost": "\u2014" if yoc is None else f"{yoc:.1%}",
                    "Flags": "; ".join(det) if det else "\u2014",
                    "Mkt Cap": _format_market_cap(_market_cap(snap)),
                    "PEG": "\u2014" if peg_pf is None else f"{peg_pf:.2f}",
                    "Data": f"{comp_pf['metrics']}/5" + (" ⚠️" if comp_pf["metrics"] <= 3 else ""),
                    "Signal": sig["signal"], "Why": "; ".join(sig["reasons"]), "_rank": PORTFOLIO_RANK.get(sig["signal"], 9),
                    "_value": value or 0.0, "_moat": snap["moat_score"] or 0.0,
                })
        # Weights and weighted moat need the total first.
        for r in rows:
            if total_value > 0 and r["_value"]:
                r["Weight"] = f"{r['_value'] / total_value:.1%}"
        weighted_moat = (sum(r["_value"] * r["_moat"] for r in rows) / total_value) if total_value > 0 else None
        largest = max(rows, key=lambda r: r["_value"]) if rows else None
        rows.sort(key=lambda r: r["_rank"])
        # Review queue first: only holdings needing attention, sorted
        # by urgency (SELL, then REVIEW/jeopardy, then other flags).
        if review_items:
            review_items.sort(key=lambda r: r["_urgency"])
            st.markdown("**Review queue — holdings needing attention**")
            st.dataframe(pd.DataFrame(review_items).drop(columns=["_urgency"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Review queue: no holdings need attention right now.")
        df = pd.DataFrame(rows).drop(columns=["_rank", "_value", "_moat"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        if total_cost > 0:
            total_gain = total_value - total_cost
            st.markdown(
                f"**Portfolio totals:** value {money(total_value)} \u00b7 cost basis {money(total_cost)} \u00b7 "
                f"gain/loss {money(total_gain)} ({total_gain / total_cost:+.1%})"
            )
        if total_div_income > 0:
            st.markdown(
                f"**Dividend income:** {money(total_div_income)}/yr from holdings with a known dividend per share"
                + (f" \u00b7 yield on cost {total_div_income / total_cost:.1%}" if total_cost > 0 else "")
                + " \u00b7 holdings with no DPS show \u2014, not $0, so this total may understate income."
            )
        if weighted_moat is not None:
            st.markdown(f"**Weighted Moat Score:** {weighted_moat:.1f} / 100 (by position value)")
        if largest is not None and total_value > 0 and largest["_value"]:
            st.markdown(f"**Largest holding:** {largest['Ticker']} at {largest['_value'] / total_value:.1%} of portfolio value")
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
                + " \u00b7 Illustration only, not a forecast."
            )
        counts = Counter(r["Signal"] for r in rows)
        st.caption(" \u00b7 ".join(f"{n} {s}" for s, n in counts.most_common()))
        export_data = load_user_data()
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

    if "watchlist_df" not in st.session_state:
        saved = load_user_data()
        st.session_state["watchlist_df"] = pd.DataFrame({"Ticker": pd.Series(saved["watchlist"], dtype="object")})
        st.session_state["watchlist_saved"] = saved["watchlist"]

    edited_w = st.data_editor(
        st.session_state["watchlist_df"],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="watchlist_editor",
        column_config={"Ticker": st.column_config.TextColumn("Ticker")},
    )
    watch = list(dict.fromkeys(
        t for t in (normalize_ticker(_cell(v) or "") for v in edited_w["Ticker"]) if t
    ))
    if watch != st.session_state["watchlist_saved"]:
        if save_watchlist(watch):
            st.session_state["watchlist_saved"] = watch
        else:
            st.warning("Couldn't write user_data.json \u2014 changes won't persist after a restart.")

    if not watch:
        st.info("Add a ticker above to see its buy signal.")
    else:
        rows = []
        with st.spinner("Checking your watchlist..."):
            for t in watch:
                snap = build_snapshot(t, price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": t, "Price": "\u2014", "Sticker": "\u2014", "Buy Below": "\u2014", "% Above Buy Price": "\u2014",
                                 "Moat": "\u2014", "Signal": "NO DATA", "Why": snap["error"], "_rank": 9, "_pct": 1e9})
                    continue
                val = valuation_from_snapshot(snap, discount_rate, max_growth)
                sig = buy_signal(snap, val, float(min_moat_buy), float(near_buy_pct))
                pct = sig["pct_above_buy"]
                rows.append({
                    "Ticker": t, "Price": money(snap["current_price"]), "Sticker": money(val["sticker_price"]),
                    "Buy Below": money(val["margin_of_safety_price"]),
                    "% Above Buy Price": "\u2014" if pct is None else f"{pct:+.1f}%",
                    "Moat": snap["moat_score"], "Signal": sig["signal"], "Why": "; ".join(sig["reasons"]),
                    "_rank": BUY_RANK.get(sig["signal"], 9), "_pct": pct if pct is not None else 1e9,
                })
        rows.sort(key=lambda r: (r["_rank"], r["_pct"]))
        df = pd.DataFrame(rows).drop(columns=["_rank", "_pct"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        counts = Counter(r["Signal"] for r in rows)
        st.caption(" \u00b7 ".join(f"{n} {s}" for s, n in counts.most_common()))


# --- 4 & 5. Market scanner --------------------------------------------------
def _pct_or_dash(v: Any) -> str:
    return "\u2014" if v is None else f"{v:.1%}"


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
            row: Dict[str, Any] = {
                "#": i, "Ticker": r["ticker"], "Company": r["name"] or "", "Price": money(r["current_price"]),
                "Sticker": money(r["sticker_price"]), "MOS Price": money(r["margin_of_safety_price"]),
                "% Below Sticker": f"{r['undervalued_pct']:.1f}%", "Growth Used": f"{r['growth_used']:.1%}",
                "Moat": r["moat_score"], "Big Five": f"{r.get('big_five_passes', 0)}/5",
                "Meaning": note_scan.get("meaning") or "\u2014", "Moat type": note_scan.get("moat_type") or "\u2014",
                "Action": r["action"].replace("_", " "), "Div. Streak (yrs)": r["dividend_streak"],
                "Mkt Cap": _format_market_cap(_market_cap(snap_row)) if snap_row else "\u2014",
                "PEG": "\u2014" if peg_row is None else f"{peg_row:.2f}",
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
        st.download_button(
            "Download this Top 25 (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="top25.csv",
            mime="text/csv",
            key=download_key,
        )
    if scan["errors"]:
        reasons = Counter(scan["errors"].values())
        with st.expander(f"{len(scan['errors'])} tickers skipped \u2014 why"):
            for reason, n in reasons.most_common():
                sample = [t for t, m in scan["errors"].items() if m == reason][:12]
                st.write(f"**{n}** \u00d7 {reason}  \u2014  e.g. {', '.join(sample)}")


with tab_undervalued:
    st.caption(
        "Scans a universe and ranks wonderful businesses first, then by how far each stock trades below its "
        "Sticker Price. Results are cached, so you can stop a scan and click Scan again to resume. " + DISCLAIMER
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
    if "scan" in st.session_state:
        render_ranked_table(st.session_state["scan"], 0, float(min_moat_scan), quality_first=quality_first_scan, download_key="scan_csv", min_market_cap_b=float(min_cap_scan))
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
