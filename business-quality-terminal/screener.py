"""
screener.py

Market scanner. Two separate steps so that nothing is ever refetched just
because a slider moved:

  scan_snapshots(tickers)   -- fetch a snapshot for every ticker (threaded,
      SEC-throttled, disk-cached so an interrupted scan resumes where it
      stopped).
  rank_snapshots(snapshots, ...) -- instant: apply discount rate, growth
      cap and filters, sort by how far below Sticker Price each trades.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from gatekeeper import BIG_FIVE_METRICS
from valuation import TRANSIENT, build_snapshot, valuation_from_snapshot


def big_five_pass_count(snapshot: Dict[str, Any], hurdle: float) -> int:
    """How many of the Big Five pass at `hurdle`, using stored CAGRs.

    For each metric, take the longest horizon with a CAGR (10yr, then
    5yr, 3yr, 1yr) and count it as a pass if that CAGR is above the
    hurdle. This reuses the CAGRs already in the snapshot's gatekeeper
    result, so stepping the hurdle down does not require a refetch.
    """
    big_five = (snapshot.get("gatekeeper") or {}).get("big_five") or {}
    passes = 0
    for metric in BIG_FIVE_METRICS:
        horizons = big_five.get(metric) or {}
        for h in (10, 5, 3, 1):
            entry = horizons.get(h) or {}
            cagr = entry.get("cagr")
            if cagr is None:
                continue
            if cagr > hurdle:
                passes += 1
            break
    return passes

SCAN_WORKERS = 4
SCAN_PRICE_MAX_AGE_SECONDS = 24 * 60 * 60  # a scan uses prices up to a day old
MAX_CONSECUTIVE_TRANSIENT_FAILURES = 8
TRANSIENT_RETRY_DELAY_SECONDS = 2.0


def scan_snapshots(
    tickers: List[str],
    workers: int = SCAN_WORKERS,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_consecutive_transient: int = MAX_CONSECUTIVE_TRANSIENT_FAILURES,
    retry_delay: float = TRANSIENT_RETRY_DELAY_SECONDS,
) -> Dict[str, Any]:
    """
    Returns {"snapshots": [valid snapshots], "errors": {ticker: message},
             "total": n, "done": n, "aborted": None | str}.

    If SEC or Yahoo starts failing in a row (throttling), the scan stops
    early with `aborted` set instead of hammering them; re-run later and it
    resumes, because finished tickers are already cached. Never raises
    (except KeyboardInterrupt/Streamlit rerun, which stop the pool cleanly).
    """
    unique = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    total = len(unique)
    snapshots: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    stop = threading.Event()

    def task(ticker: str):
        if stop.is_set():
            return ticker, None
        snap = build_snapshot(ticker, price_max_age=SCAN_PRICE_MAX_AGE_SECONDS, prune_unvaluable=True)
        if snap.get("error") and snap.get("error_kind") == TRANSIENT and not stop.is_set():
            time.sleep(retry_delay)
            snap = build_snapshot(ticker, price_max_age=SCAN_PRICE_MAX_AGE_SECONDS, prune_unvaluable=True)
        return ticker, snap

    done = 0
    consecutive_transient = 0
    aborted: Optional[str] = None
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [executor.submit(task, t) for t in unique]
        for future in as_completed(futures):
            ticker, snap = future.result()
            if snap is None:  # skipped after an abort
                continue
            done += 1
            if snap.get("error"):
                errors[ticker] = snap["error"]
                if snap.get("error_kind") == TRANSIENT:
                    consecutive_transient += 1
                else:
                    consecutive_transient = 0
            else:
                snapshots.append(snap)
                consecutive_transient = 0

            if progress_callback is not None:
                try:
                    progress_callback(done, total, ticker)
                except Exception:
                    pass

            if consecutive_transient >= max_consecutive_transient and aborted is None:
                aborted = (
                    f"Stopped after {consecutive_transient} failures in a row \u2014 SEC or Yahoo "
                    "looks like it's throttling. Wait a few minutes and click Scan again; "
                    "finished tickers are cached, so it resumes where it left off."
                )
                stop.set()
                executor.shutdown(wait=False, cancel_futures=True)
                break
    finally:
        # Also runs when Streamlit interrupts the script mid-scan: don't
        # leave thousands of queued fetches running in the background.
        stop.set()
        executor.shutdown(wait=False, cancel_futures=True)

    return {"snapshots": snapshots, "errors": errors, "total": total, "done": done, "aborted": aborted}


def rank_snapshots(
    snapshots: List[Dict[str, Any]],
    discount_rate: float = 0.15,
    max_growth_rate: float = 0.15,
    min_dividend_years: int = 0,
    min_moat_score: float = 0.0,
    top_n: int = 25,
    min_big_five_passes: int = 0,
    big_five_hurdle: Optional[float] = None,
) -> Dict[str, Any]:
    """Most undervalued first, i.e. largest (Sticker - Price) / Sticker.

    Quality filters (dividend streak, moat, Big Five passes) are applied
    BEFORE sorting by price, so a cheap weak business cannot outrank a
    wonderful one on price alone. When `big_five_hurdle` is given, a row
    must pass at least `min_big_five_passes` of the Big Five at that
    hurdle; when it is None, no Big Five filter is applied.
    """
    valued = 0
    rows: List[Dict[str, Any]] = []
    for snap in snapshots:
        val = valuation_from_snapshot(snap, discount_rate, max_growth_rate)
        if val["sticker_price"] is None or val["undervalued_pct"] is None:
            continue
        valued += 1
        if snap.get("dividend_streak", 0) < min_dividend_years:
            continue
        if snap.get("moat_score", 0.0) < min_moat_score:
            continue
        passes = big_five_pass_count(snap, big_five_hurdle) if big_five_hurdle is not None else big_five_pass_count(snap, 0.10)
        if big_five_hurdle is not None and passes < min_big_five_passes:
            continue
        rows.append(
            {
                "ticker": snap["ticker"],
                "name": snap.get("name"),
                "current_price": snap["current_price"],
                "sticker_price": val["sticker_price"],
                "margin_of_safety_price": val["margin_of_safety_price"],
                "undervalued_pct": val["undervalued_pct"],
                "growth_used": val["growth_used"],
                "moat_score": snap.get("moat_score", 0.0),
                "action": val["action"],
                "dividend_streak": snap.get("dividend_streak", 0),
                "big_five_passes": passes,
                # Dividend-growth metrics (dividend tab only; None for
                # older cached snapshots — the tab shows "—" for those).
                "dividend_per_share": snap.get("dividend_per_share"),
                "dividend_yield": snap.get("dividend_yield"),
                "dividend_growth_5yr": snap.get("dividend_growth_5yr"),
                "dividend_growth_10yr": snap.get("dividend_growth_10yr"),
                "payout_ratio": snap.get("payout_ratio"),
                "chowder": snap.get("chowder"),
                "years_to_10pct_yield_on_cost": snap.get("years_to_10pct_yield_on_cost"),
                "fcf_dividend_coverage": snap.get("fcf_dividend_coverage"),
            }
        )
    rows.sort(key=lambda r: r["undervalued_pct"], reverse=True)
    return {"top_results": rows[:top_n], "matched_count": len(rows), "valued_count": valued}


def stepped_rank_snapshots(
    snapshots: List[Dict[str, Any]],
    discount_rate: float = 0.15,
    max_growth_rate: float = 0.15,
    min_dividend_years: int = 0,
    min_moat_score: float = 0.0,
    top_n: int = 25,
    min_big_five_passes: int = 4,
    start_hurdle: float = 0.10,
    floor_hurdle: float = 0.05,
) -> Dict[str, Any]:
    """Rank with the Rule #1 hurdle stepped down 1% at a time if empty.

    Starts at `start_hurdle` (10% by default). If no stock passes
    `min_big_five_passes` of the Big Five at that hurdle (plus the other
    filters), the hurdle drops exactly 1 percentage point and the ranking
    is retried, down to `floor_hurdle`. If nothing passes even at the
    floor, the result carries `schd_fallback=True`: per the user's locked
    rule, SCHD is the fallback for an empty / disqualified list. SCHD
    itself is an ETF, not a 10-K filer, so it is never given a fake
    Sticker Price here — the fallback is a message, not a row.
    """
    hurdle = start_hurdle
    last_ranked: Dict[str, Any] = {"top_results": [], "matched_count": 0, "valued_count": 0}
    while hurdle >= floor_hurdle - 1e-9:
        hurdle = round(hurdle, 2)
        ranked = rank_snapshots(
            snapshots,
            discount_rate,
            max_growth_rate,
            min_dividend_years,
            min_moat_score,
            top_n,
            min_big_five_passes=min_big_five_passes,
            big_five_hurdle=hurdle,
        )
        last_ranked = ranked
        if ranked["matched_count"] > 0:
            return {
                "ranked": ranked,
                "hurdle_used": hurdle,
                "stepped_down": hurdle < start_hurdle,
                "schd_fallback": False,
            }
        hurdle -= 0.01
    return {
        "ranked": last_ranked,
        "hurdle_used": None,
        "stepped_down": True,
        "schd_fallback": True,
    }
