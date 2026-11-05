"""Offline tests for the 2026-11-04 improvements (items 1, 2, 5).

1. SEC tag merge: the fullest candidate tag wins per metric, so a metric
   whose first candidate tag only exists for recent years still gets its
   full history from a later candidate tag (AAPL Sales: "Revenues" from
   ~2015 vs "SalesRevenueNet" with full history).
2. ROIC dropouts are recorded, not silently dropped: negative invested
   capital (equity + debt - cash <= 0) appears in _roic_dropped.
5. last_saved_display: returns a timestamp for an existing file, None
   for a missing file.

Run: python3 test_tag_roic.py (no network, no Streamlit needed).
"""
import os
import tempfile

import data_fetcher as df_mod
import storage


def _fact(end, val, start=None):
    return {
        "form": "10-K",
        "start": start or (end[:4] + "-01-01"),
        "end": end,
        "filed": str(int(end[:4]) + 1) + "-03-01",
        "val": val,
    }


def _series(years_vals, unit="USD"):
    return {"units": {unit: [_fact(e, v) for e, v in years_vals]}}


def _raw():
    years = list(range(2015, 2026))
    return {"entityName": "Test Co", "facts": {"us-gaap": {
        "Revenues": _series([("2024-12-31", 500), ("2025-12-31", 550)]),
        "SalesRevenueNet": _series([(f"{y}-12-31", 100 + y) for y in years]),
        "NetIncomeLoss": _series([(f"{y}-12-31", 50) for y in years]),
        "EarningsPerShareDiluted": _series(
            [(f"{y}-12-31", 2.0) for y in years], unit="USD/shares"),
        "StockholdersEquity": _series([(f"{y}-12-31", -100) for y in years]),
        "LongTermDebtNoncurrent": _series([(f"{y}-12-31", 50) for y in years]),
        "CashAndCashEquivalentsAtCarryingValue": _series(
            [(f"{y}-12-31", 10) for y in years]),
    }}}


def main():
    raw = _raw()
    series, tag = df_mod._get_best_tag_series_with_tag(raw, "Sales")
    assert tag == "SalesRevenueNet", f"primary tag should be SalesRevenueNet, got {tag}"
    assert "2015-12-31" in series, "full-history tag should supply 2015 Sales"
    assert len(series) == 11, f"merged Sales should have 11 ends, got {len(series)}"
    hist = df_mod._build_financial_history(raw)
    assert sorted(hist["Sales"].keys()) == [0, 1, 3, 5, 10], \
        f"Sales horizons should be 0/1/3/5/10, got {sorted(hist['Sales'].keys())}"
    assert hist["_tag_used"].get("Sales") == "SalesRevenueNet"
    assert hist["ROIC"] == {}, "ROIC should be empty with negative invested capital"
    assert 0 in hist["_roic_dropped"], "latest ROIC dropout should be recorded"
    assert "≤ 0" in hist["_roic_dropped"][0], hist["_roic_dropped"][0]
    assert storage.last_saved_display("/tmp/does-not-exist-xyz.json") is None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(b"{}")
        path = f.name
    try:
        assert storage.last_saved_display(path) is not None
    finally:
        os.unlink(path)
    print("ALL TAG/ROIC/PERSISTENCE TESTS PASSED")


if __name__ == "__main__":
    main()
