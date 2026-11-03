"""
app.py

Business Quality Terminal -- Rule #1 Investing / Buffett-style valuation.

Tabs:
  1. Ticker Lookup        any SEC-registered company, full Gatekeeper checklist
  2. Portfolio            holdings you track, with SELL / REVIEW / HOLD signals
  3. Watchlist            tickers you're waiting on, with BUY / NEAR BUY signals
  4. Top 25 Undervalued   market scan (S&P 500, every NYSE+Nasdaq stock, or custom)
  5. Top 25 + Dividend Growers   same scan, filtered by consecutive years of
                          dividend growth

Run with:  streamlit run app.py
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from data_fetcher import clear_cache, normalize_ticker
from gatekeeper import BIG_FIVE_METRICS, HORIZONS
from screener import rank_snapshots, scan_snapshots
from signals import BUY_RANK, SELL_RANK, buy_signal, sell_signal
from storage import load_user_data, save_portfolio, save_watchlist
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
}

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


def build_checklist_dataframe(gatekeeper: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for metric in BIG_FIVE_METRICS:
        row = {"Metric": METRIC_DISPLAY_NAMES.get(metric, metric)}
        for h in HORIZONS:
            entry = gatekeeper["big_five"].get(metric, {}).get(h, {"status": "not_yet_determined"})
            row[f"{h}yr"] = status_icon(entry["status"])
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
        "Discount rate (minimum acceptable return)", min_value=0.06, max_value=0.20, value=0.10, step=0.01, format="%.2f"
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
    if not lookup_tickers:
        st.info("Enter one or more tickers in the sidebar to begin.")
    for ticker in lookup_tickers:
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
        "Changes save automatically to user_data.json. " + DISCLAIMER
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        approach_pct = st.slider("Flag when within X% of Sticker Price", 0, 30, 10, key="pf_approach")
    with c2:
        min_moat_hold = st.slider("Flag when Moat Score is below", 0, 100, 40, key="pf_moat")
    with c3:
        max_loss_pct = st.number_input("Flag when down more than X% vs cost (0 = off)", 0, 90, 0, key="pf_loss")

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
        st.info("Add a ticker above to see its sell signal.")
    else:
        rows, total_value, total_cost = [], 0.0, 0.0
        with st.spinner("Checking your holdings..."):
            for h in holdings:
                snap = build_snapshot(h["ticker"], price_max_age=TRACKING_PRICE_MAX_AGE)
                if snap.get("error"):
                    rows.append({"Ticker": h["ticker"], "Shares": h["shares"], "Avg Cost": money(h["avg_cost"]), "Price": "\u2014",
                                 "Value": "\u2014", "Gain/Loss": "\u2014", "Gain %": "\u2014", "Sticker": "\u2014", "Moat": "\u2014",
                                 "Signal": "NO DATA", "Why": snap["error"], "_rank": 9})
                    continue
                val = valuation_from_snapshot(snap, discount_rate, max_growth)
                sig = sell_signal(snap, val, h["avg_cost"], approach_pct, min_moat_hold, float(max_loss_pct))
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
                    "Signal": sig["signal"], "Why": "; ".join(sig["reasons"]), "_rank": SELL_RANK.get(sig["signal"], 9),
                })
        rows.sort(key=lambda r: r["_rank"])
        df = pd.DataFrame(rows).drop(columns=["_rank"])
        st.dataframe(style_signal_column(df), hide_index=True, use_container_width=True)
        if total_cost > 0:
            st.markdown(f"**Positions with shares and cost:** value of all positions {money(total_value)} \u00b7 cost basis {money(total_cost)}")
        counts = Counter(r["Signal"] for r in rows)
        st.caption(" \u00b7 ".join(f"{n} {s}" for s, n in counts.most_common()))


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
def render_ranked_table(scan: Dict[str, Any], min_dividend_years: int, min_moat: float) -> None:
    ranked = rank_snapshots(
        scan["snapshots"], discount_rate, max_growth, min_dividend_years=min_dividend_years, min_moat_score=min_moat, top_n=25
    )
    st.caption(
        f"{scan['label']}: scanned {scan['done']} of {scan['total']} tickers \u2014 {len(scan['snapshots'])} had usable "
        f"data, {ranked['valued_count']} could be valued, {ranked['matched_count']} match your filters."
    )
    if scan.get("aborted"):
        st.warning(scan["aborted"])
    if not ranked["top_results"]:
        st.info("Nothing matched. Loosen the filters, or run a wider scan.")
    else:
        rows = [
            {
                "#": i, "Ticker": r["ticker"], "Company": r["name"] or "", "Price": money(r["current_price"]),
                "Sticker": money(r["sticker_price"]), "MOS Price": money(r["margin_of_safety_price"]),
                "% Below Sticker": f"{r['undervalued_pct']:.1f}%", "Growth Used": f"{r['growth_used']:.1%}",
                "Moat": r["moat_score"], "Action": r["action"].replace("_", " "), "Div. Streak (yrs)": r["dividend_streak"],
            }
            for i, r in enumerate(ranked["top_results"], start=1)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if scan["errors"]:
        reasons = Counter(scan["errors"].values())
        with st.expander(f"{len(scan['errors'])} tickers skipped \u2014 why"):
            for reason, n in reasons.most_common():
                sample = [t for t, m in scan["errors"].items() if m == reason][:12]
                st.write(f"**{n}** \u00d7 {reason}  \u2014  e.g. {', '.join(sample)}")


with tab_undervalued:
    st.caption(
        "Scans a universe and ranks by how far each stock trades below its Sticker Price. Results are cached, "
        "so you can stop a scan and click Scan again to resume. " + DISCLAIMER
    )
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

    min_moat_scan = st.slider("Minimum Moat Score (filter)", 0, 100, 0, key="scan_moat")
    if "scan" in st.session_state:
        render_ranked_table(st.session_state["scan"], 0, float(min_moat_scan))
    else:
        st.info("Pick a universe and click Scan.")

with tab_dividends:
    st.caption(
        "The same scan as the previous tab (run it there first), keeping only companies that have raised their "
        "dividend per share for at least this many years in a row."
    )
    div_years = st.slider("Minimum consecutive years of dividend growth", 1, 25, 5, key="div_years")
    min_moat_div = st.slider("Minimum Moat Score (filter)", 0, 100, 0, key="div_moat")
    if "scan" in st.session_state:
        render_ranked_table(st.session_state["scan"], int(div_years), float(min_moat_div))
    else:
        st.info("Run a scan on the Top 25 Undervalued tab first; this tab reuses its results.")
