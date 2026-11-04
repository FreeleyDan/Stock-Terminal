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
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from data_fetcher import clear_cache, normalize_ticker
from gatekeeper import BIG_FIVE_METRICS, HORIZONS
from screener import rank_snapshots, scan_snapshots, stepped_rank_snapshots
from signals import BUY_RANK, PORTFOLIO_RANK, buy_signal, portfolio_signal, sell_signal
from storage import (
    load_last_scan,
    load_user_data,
    parse_imported_data,
    save_last_scan,
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


def render_ticker_card(ticker: str, snap: Dict[str, Any], val: Optional[Dict[str, Any]], max_growth: float) -> None:
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
    details = [f"Last close {snap.get('price_date', '')}"]
    if val["growth_used"] is not None:
        g = snap["growth_rate"]
        details.append(f"EPS growth {g:.1%}" + (f" (capped at {max_growth:.0%} for valuation)" if g > max_growth else ""))
    if snap.get("historical_pe") is not None:
        details.append(f"median historical P/E {snap['historical_pe']:.1f}")
    if snap.get("dividend_streak"):
        details.append(f"dividend raised {snap['dividend_streak']} yrs in a row")
    st.caption(" \u00b7 ".join(details))
    if val["note"]:
        st.caption(val["note"])
    st.dataframe(build_checklist_dataframe(snap["gatekeeper"]), hide_index=True, use_container_width=True)

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
        render_ticker_card(ticker, snap, val, max_growth)


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
        st.caption("Import replaces the saved portfolio, watchlist, and notes with the file's contents. Export first if you want a backup.")
        uploaded_pf = st.file_uploader("Portfolio JSON file", type=["json"], key="pf_import_file")
        if uploaded_pf is not None and st.button("Import this file", key="pf_import_btn"):
            try:
                imported_portfolio, imported_watchlist, imported_notes = parse_imported_data(json.load(uploaded_pf))
                save_user_data({"portfolio": imported_portfolio, "watchlist": imported_watchlist, "notes": imported_notes})
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
        rows, total_value, total_cost = [], 0.0, 0.0
        with st.spinner("Checking your holdings..."):
            for h in holdings:
                snap = build_snapshot(h["ticker"], price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": "\u2014",
                                 "Value": "\u2014", "Gain/Loss": "\u2014", "Gain %": "\u2014", "Sticker": "\u2014", "Moat": "\u2014",
                                 "Meaning": "\u2014", "Moat type": "\u2014",
                                 "Signal": "NO DATA", "Why": snap["error"], "_rank": 9})
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
                rows.append({
                    "Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": money(price),
                    "Value": money(value), "Gain/Loss": money(gain),
                    "Gain %": "\u2014" if gain_pct is None else f"{gain_pct:+.1f}%",
                    "Sticker": money(val["sticker_price"]), "Moat": snap["moat_score"],
                    "Meaning": note_pf.get("meaning") or "\u2014", "Moat type": note_pf.get("moat_type") or "\u2014",
                    "Signal": sig["signal"], "Why": "; ".join(sig["reasons"]), "_rank": PORTFOLIO_RANK.get(sig["signal"], 9),
                })
        rows.sort(key=lambda r: r["_rank"])
        df = pd.DataFrame(rows).drop(columns=["_rank"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        if total_cost > 0:
            st.markdown(f"**Positions with shares and cost:** value of all positions {money(total_value)} \u00b7 cost basis {money(total_cost)}")
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


def render_ranked_table(scan: Dict[str, Any], min_dividend_years: int, min_moat: float, quality_first: bool = True, download_key: str = "scan_csv", dividend_mode: bool = False, max_payout: Optional[float] = None, min_div_growth: Optional[float] = None, min_chowder: Optional[float] = None) -> None:
    # Dividend tab only (Miller / Van Knapp layer): filter snapshots
    # before ranking. A missing metric is shown as "—" and does NOT
    # exclude a stock — SEC dividend-per-share coverage is imperfect,
    # so unknown is not treated as failing.
    snapshots_for_rank = scan["snapshots"]
    if dividend_mode:
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
            "not his exact score. Stocks with a missing dividend metric show \u2014 and are not excluded for it."
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
        for i, r in enumerate(ranked["top_results"], start=1):
            note_scan = _notes_for(r["ticker"])
            row: Dict[str, Any] = {
                "#": i, "Ticker": r["ticker"], "Company": r["name"] or "", "Price": money(r["current_price"]),
                "Sticker": money(r["sticker_price"]), "MOS Price": money(r["margin_of_safety_price"]),
                "% Below Sticker": f"{r['undervalued_pct']:.1f}%", "Growth Used": f"{r['growth_used']:.1%}",
                "Moat": r["moat_score"], "Big Five": f"{r.get('big_five_passes', 0)}/5",
                "Meaning": note_scan.get("meaning") or "\u2014", "Moat type": note_scan.get("moat_type") or "\u2014",
                "Action": r["action"].replace("_", " "), "Div. Streak (yrs)": r["dividend_streak"],
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
                row.update(
                    {
                        "Yield": _pct_or_dash(r.get("dividend_yield")),
                        "Div Growth 5yr": _pct_or_dash(r.get("dividend_growth_5yr")),
                        "Chowder": _pct_or_dash(r.get("chowder")),
                        "Payout": _pct_or_dash(payout),
                        "FCF Cover": "\u2014" if fcf_cover is None else f"{fcf_cover:.1f}x",
                        "Yrs to 10% YoC": "\u2014" if years_to_10 is None else ("Already \u226510%" if years_to_10 == 0 else f"{years_to_10:.0f} yrs"),
                        "Div. flag": div_flag,
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
                st.caption(f"Loaded last saved scan from {datetime.fromtimestamp(last_saved_at).strftime('%Y-%m-%d %H:%M')}. Click Scan to refresh it.")
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

            result = scan_snapshots(universe, progress_callback=on_progress)
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
    if "scan" in st.session_state:
        render_ranked_table(st.session_state["scan"], 0, float(min_moat_scan), quality_first=quality_first_scan, download_key="scan_csv")
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
    quality_first_div = st.checkbox(
        "Wonderful businesses first (same hurdle step-down as the Undervalued tab)",
        value=True, key="div_quality_first",
    )
    if "scan" in st.session_state:
        render_ranked_table(
            st.session_state["scan"], int(div_years), float(min_moat_div), quality_first=quality_first_div,
            download_key="div_csv", dividend_mode=True, max_payout=float(max_payout_div),
            min_div_growth=float(min_div_growth_div), min_chowder=float(min_chowder_div),
        )
        st.caption(
            "Note: dividend metrics come from fresh SEC filings. A scan cached before this update won't have them "
            "(shows \u2014) — click Refresh data on the Ticker Lookup tab, or wait for the 24-hour cache to expire, "
            "then re-run the scan."
        )
    else:
        st.info("Run a scan on the Top 25 Undervalued tab first; this tab reuses its results.")
