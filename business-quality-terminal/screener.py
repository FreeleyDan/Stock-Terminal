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

from valuation import TRANSIENT, build_snapshot, valuation_from_snapshot

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
    discount_rate: float = 0.10,
    max_growth_rate: float = 0.15,
    min_dividend_years: int = 0,
    min_moat_score: float = 0.0,
    top_n: int = 25,
) -> Dict[str, Any]:
    """Most undervalued first, i.e. largest (Sticker - Price) / Sticker."""
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
            }
        )
    rows.sort(key=lambda r: r["undervalued_pct"], reverse=True)
    return {"top_results": rows[:top_n], "matched_count": len(rows), "valued_count": valued}
