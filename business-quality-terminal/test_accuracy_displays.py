"""Offline tests for the 2026-11-05 accuracy batch (items 1-4).

1. 10-year audit table + median YoY: median YoY is the typical year's
   growth, unlike the end-to-end CAGR (one bad base year flatters it).
2. Split warning: splits after a purchase date are listed.
3. SBC-adjusted FCF, maintenance-capex estimate, interest coverage,
   total debt: new SEC tags flow through _build_financial_history.
4. (Same as 3 — interest coverage / total obligations display.)

Run: python3 test_accuracy_displays.py (no network, no Streamlit needed
for the data_fetcher/calculations parts; the app.py helpers are tested
with a stubbed streamlit module).
"""

import data_fetcher as df_mod
from calculations import median_yoy_growth


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
        "NetIncomeLoss": _series([(f"{y}-12-31", 100) for y in years]),
        "EarningsPerShareDiluted": _series(
            [(f"{y}-12-31", 2.0) for y in years], unit="USD/shares"),
        "SalesRevenueNet": _series([(f"{y}-12-31", 1000 + 100 * (y - 2015)) for y in years]),
        "NetCashProvidedByUsedInOperatingActivities": _series(
            [(f"{y}-12-31", 200) for y in years]),
        "PaymentsToAcquirePropertyPlantAndEquipment": _series(
            [(f"{y}-12-31", 50) for y in years]),
        "DepreciationDepletionAndAmortization": _series(
            [(f"{y}-12-31", 40) for y in years]),
        "StockholdersEquity": _series([(f"{y}-12-31", 1000) for y in years]),
        "LongTermDebtNoncurrent": _series([(f"{y}-12-31", 300) for y in years]),
        "LongTermDebtCurrent": _series([(f"{y}-12-31", 100) for y in years]),
        "CashAndCashEquivalentsAtCarryingValue": _series(
            [(f"{y}-12-31", 50) for y in years]),
        "WeightedAverageNumberOfDilutedSharesOutstanding": _series(
            [(f"{y}-12-31", 100) for y in years], unit="shares"),
        # New accuracy tags.
        "ShareBasedCompensation": _series([(f"{y}-12-31", 30) for y in years]),
        "InterestExpense": _series([(f"{y}-12-31", 20) for y in years]),
        "OperatingIncomeLoss": _series([(f"{y}-12-31", 160) for y in years]),
    }}}


def main():
    # 1. Median YoY: steady 10% growth -> ~10%.
    steady = {f"{y}-12-31": 100 * (1.10 ** (y - 2015)) for y in range(2015, 2026)}
    med = median_yoy_growth(steady)
    assert med is not None and abs(med - 0.10) < 0.001, f"steady median YoY should be 10%, got {med}"
    # One terrible base year: 100 -> 50 -> 55 -> 60.5 -> 66.55.
    # YoY steps: -50%, +10%, +10%, +10% -> median +10%, while the
    # end-to-end CAGR is negative — the exact case the audit table is for.
    base_year = {"2021-12-31": 100.0, "2022-12-31": 50.0, "2023-12-31": 55.0,
                 "2024-12-31": 60.5, "2025-12-31": 66.55}
    med2 = median_yoy_growth(base_year)
    assert med2 is not None and abs(med2 - 0.10) < 0.001, f"median YoY should be 10%, got {med2}"
    from calculations import calculate_cagr
    assert calculate_cagr(100.0, 66.55, 4) < 0, "end-to-end CAGR should be negative here"
    assert median_yoy_growth({"2025-12-31": 5.0}) is None, "single point -> None"
    assert median_yoy_growth({}) is None, "empty series -> None"

    # 3/4. New tags flow through the history builder.
    hist = df_mod._build_financial_history(_raw())
    assert hist["Share_Based_Compensation"].get(0) == 30, hist["Share_Based_Compensation"]
    assert hist["Interest_Expense"].get(0) == 20, hist["Interest_Expense"]
    assert hist["Operating_Income"].get(0) == 160, hist["Operating_Income"]
    # FCF = OCF 200 - CapEx 50 = 150; SBC-adjusted = 150 - 30 = 120.
    assert hist["SBC_Adjusted_FCF"].get(0) == 120, hist["SBC_Adjusted_FCF"]
    # Interest coverage = 160 / 20 = 8x; total debt = 300 + 100 = 400.
    assert hist["_interest_coverage_history"].get(0) == 8.0, hist["_interest_coverage_history"]
    assert hist["Total_Debt"].get(0) == 400, hist["Total_Debt"]
    # Maintenance capex estimate is D&A (40); owner earnings (total
    # capex) = NI 100 + D&A 40 - CapEx 50 = 90.
    assert hist["_maintenance_capex_history"].get(0) == 40, hist["_maintenance_capex_history"]
    assert hist["Owner_Earnings"].get(0) == 90, hist["Owner_Earnings"]
    # Audit series: 11 year-ends each; Sales 2025 = 2000; ROIC latest =
    # NI 100 / (equity 1000 + debt 400 - cash 50) = 100/1350.
    audit = hist["_audit_series"]
    assert len(audit["Sales"]) == 11, f"audit Sales should have 11 ends, got {len(audit['Sales'])}"
    assert audit["Sales"]["2025-12-31"] == 2000, audit["Sales"]["2025-12-31"]
    assert abs(audit["ROIC"]["2025-12-31"] - 100 / 1350) < 1e-9, audit["ROIC"]
    assert audit["EPS"]["2025-12-31"] == 2.0, audit["EPS"]
    assert audit["Free_Cash_Flow"]["2025-12-31"] == 150, audit["Free_Cash_Flow"]
    assert audit["Equity_Per_Share"]["2025-12-31"] == 10.0, audit["Equity_Per_Share"]

    # 2 + audit-table helper: the real function definitions, extracted
    # from app.py (importing all of app.py would run the whole Streamlit
    # script). They depend only on _parse_purchase_date, median_yoy_growth,
    # and the standard library.
    import ast
    from typing import Any, Dict, List, Optional
    from datetime import date as _date, datetime as _datetime
    src = open("app.py").read()
    tree = ast.parse(src)
    wanted = {"_parse_purchase_date", "_audit_table_rows", "_splits_after_purchase",
              "_project_drip_income", "_projected_income_at_year", "_years_to_goal",
              "_dividend_payment_cut", "_payout_jump_check",
              "_income_attribution", "_observed_income_growth", "_years_to_goal_observed",
              "_record_goal_month", "_display_purchase_date", "_scan_for_universe"}
    import statistics as _statistics
    import math as _math
    ns: Dict[str, Any] = {
        "median_yoy_growth": median_yoy_growth,
        "List": List, "Dict": Dict, "Optional": Optional, "Any": Any,
        "date": _date, "datetime": _datetime,
        "statistics": _statistics, "math": _math,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, "app.py", "exec"), ns)

    class _M:
        pass
    m = _M()
    m._splits_after_purchase = ns["_splits_after_purchase"]
    m._audit_table_rows = ns["_audit_table_rows"]
    m._years_to_goal = ns["_years_to_goal"]
    m._dividend_payment_cut = ns["_dividend_payment_cut"]
    m._payout_jump_check = ns["_payout_jump_check"]
    m._income_attribution = ns["_income_attribution"]
    m._observed_income_growth = ns["_observed_income_growth"]
    m._years_to_goal_observed = ns["_years_to_goal_observed"]
    m._record_goal_month = ns["_record_goal_month"]
    m._parse_purchase_date = ns["_parse_purchase_date"]
    m._display_purchase_date = ns["_display_purchase_date"]
    m._scan_for_universe = ns["_scan_for_universe"]
    # Purchase dates: MM-DD-YYYY is what the editor shows, YYYY-MM-DD is
    # what storage holds; both must parse, storage stays ISO.
    assert m._parse_purchase_date("01-15-2024") == _date(2024, 1, 15), "MM-DD-YYYY parses"
    assert m._parse_purchase_date("1-5-2024") == _date(2024, 1, 5), "no leading zeros parse"
    assert m._parse_purchase_date("2024-01-15") == _date(2024, 1, 15), "ISO still parses"
    assert m._parse_purchase_date("13-01-2024") is None, "month 13 is not a date"
    assert m._parse_purchase_date("02-30-2024") is None, "Feb 30 is not a date"
    assert m._parse_purchase_date("") is None and m._parse_purchase_date(None) is None
    assert m._display_purchase_date("2024-01-15") == "01-15-2024", "stored ISO displays MM-DD-YYYY"
    assert m._display_purchase_date("01-15-2024") == "01-15-2024", "typed MM-DD-YYYY displays unchanged"
    assert m._display_purchase_date(None) == "" and m._display_purchase_date("") == ""
    # Universe filter: switching the radio filters the stored scan.
    _all_scan = {"label": "All NYSE + Nasdaq stocks", "done": 4, "total": 4,
                 "snapshots": [{"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "TSLA"}, {"ticker": "XYZ"}],
                 "errors": {"XYZ": "no data", "AAPL": "stale"}}
    ns["get_sp500_tickers"] = lambda: ["AAPL", "MSFT"] + [f"SP{i}" for i in range(100)]
    ns["parse_ticker_list"] = lambda raw: [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
    _view, _note, _filt = m._scan_for_universe(_all_scan, "S&P 500")
    assert _filt is True and [s["ticker"] for s in _view["snapshots"]] == ["AAPL", "MSFT"], _view
    assert _view["errors"] == {"AAPL": "stale"}, _view["errors"]
    assert "S&P 500 members" in _note and "2 of 4" in _note, _note
    assert len(_all_scan["snapshots"]) == 4, "stored scan must not be mutated"
    _view2, _note2, _filt2 = m._scan_for_universe(_all_scan, "All NYSE + Nasdaq stocks")
    assert _view2 is _all_scan and _note2 is None and _filt2 is False, "matching universe shows as-is"
    _sp_scan = dict(_all_scan, label="S&P 500")
    _view3, _note3, _filt3 = m._scan_for_universe(_sp_scan, "All NYSE + Nasdaq stocks")
    assert _view3 is _sp_scan and _filt3 is False and "full All NYSE + Nasdaq scan" in _note3, _note3
    _view4, _note4, _filt4 = m._scan_for_universe(_all_scan, "Custom list", "AAPL, XYZ")
    assert _filt4 is True and [s["ticker"] for s in _view4["snapshots"]] == ["AAPL", "XYZ"], _view4
    _view5, _note5, _filt5 = m._scan_for_universe(_all_scan, "Custom list", "")
    assert _view5 is _all_scan and _filt5 is False and "Enter tickers" in _note5, _note5
    _nolabel = dict(_all_scan)
    _nolabel.pop("label")
    _view6, _note6, _filt6 = m._scan_for_universe(_nolabel, "S&P 500")
    assert _view6 is _nolabel and _note6 is None and _filt6 is False, "no label -> no filter"
    ns["get_sp500_tickers"] = lambda: ["AAPL", "MSFT"]
    _view7, _note7, _filt7 = m._scan_for_universe(_all_scan, "S&P 500")
    assert _view7 is _all_scan and _filt7 is False and "Couldn't fetch" in _note7, "tiny fallback list must not filter"
    # Income attribution: buys + raises must sum to the income change.
    _b, _r = m._income_attribution(110.0, 2.0, 100.0, 2.0)
    assert (_b, _r) == (20.0, 0.0), (_b, _r)
    _b, _r = m._income_attribution(100.0, 2.2, 100.0, 2.0)
    assert abs(_b) < 1e-9 and abs(_r - 20.0) < 1e-9, (_b, _r)
    _b, _r = m._income_attribution(100.0, 1.8, 100.0, 2.0)
    assert abs(_b) < 1e-9 and abs(_r + 20.0) < 1e-9, (_b, _r)
    _b, _r = m._income_attribution(200.0, 1.0, 100.0, 2.0, split_ratio=2.0)
    assert (_b, _r) == (0.0, 0.0), (_b, _r)
    assert m._income_attribution(100.0, 2.0, None, None) == (0.0, 0.0), "first sighting -> no attribution"
    # Observed growth: median MoM, annualized, capped at 50%.
    assert m._observed_income_growth([{"income": 1000.0}, {"income": 1010.0}, {"income": 1020.1}]) is not None
    _obs = m._observed_income_growth([{"income": 1000.0}, {"income": 1010.0}, {"income": 1020.1}])
    assert abs(_obs - 0.1268) < 0.001, _obs
    assert m._observed_income_growth([{"income": 1000.0}, {"income": 1100.0}, {"income": 1210.0}]) == 0.50, "10%/mo caps at 50%/yr"
    assert m._observed_income_growth([{"income": 1000.0}, {"income": 1100.0}]) is None, "one step is not enough"
    assert m._observed_income_growth([{"income": 1000.0}]) is None, "one point is not enough"
    # Years to goal at an observed rate.
    assert m._years_to_goal_observed(1000.0, 2000.0, 0.10) == 8, "10%/yr doubles in ~8 years"
    assert m._years_to_goal_observed(1000.0, 900.0, 0.10) == 0, "already at goal"
    assert m._years_to_goal_observed(1000.0, 2000.0, None) is None, "no rate -> no projection"
    assert m._years_to_goal_observed(1000.0, 2000.0, 0.0) is None, "zero rate -> no projection"
    # _record_goal_month must survive a stale storage.py: the old
    # record_dividend_goal_month takes only (month, income, goal), so it
    # imports fine but rejects the buys/raises keywords with TypeError.
    _calls = []
    def _new_rec(month, income, goal, buys=0.0, raises=0.0):
        _calls.append((month, income, goal, buys, raises))
        return True
    ns["record_dividend_goal_month"] = _new_rec
    assert m._record_goal_month("2026-11", 5200.0, 12000.0, buys=300.0, raises=50.0) is True
    assert _calls == [("2026-11", 5200.0, 12000.0, 300.0, 50.0)], _calls
    def _old_rec(month, income, goal):
        _calls.append((month, income, goal))
        return True
    ns["record_dividend_goal_month"] = _old_rec
    _calls.clear()
    assert m._record_goal_month("2026-11", 5200.0, 12000.0, buys=300.0, raises=50.0) is True, "stale storage must fall back, not crash"
    assert _calls == [("2026-11", 5200.0, 12000.0)], _calls
    # Years to goal: $1,000 growing at 10% reaches $2,000 in 8 years
    # (1.1^7 = 1.95x, 1.1^8 = 2.14x); DRIP gets there sooner; flat stays put.
    _grow = [{"income": 1000.0, "dps": 2.0, "dgr": 0.10, "price": 50.0, "shares": 500.0, "drip": False}]
    assert m._years_to_goal(_grow, 2000.0, 1000.0) == (8, 0.10), m._years_to_goal(_grow, 2000.0, 1000.0)
    _drip = [dict(_grow[0], drip=True)]
    _yrs_drip, _ = m._years_to_goal(_drip, 2000.0, 1000.0)
    assert 1 <= _yrs_drip < 8, _yrs_drip
    _flat = [{"income": 1000.0, "dps": 2.0, "dgr": None, "price": 50.0, "shares": 500.0, "drip": False}]
    assert m._years_to_goal(_flat, 2000.0, 1000.0) == (None, None), "no growth rate -> no projection"
    assert m._years_to_goal(_grow, 900.0, 1000.0) == (0, None), "already at goal"
    # Payment cut: last payment vs the one before it, not vs the last check.
    _cut = m._dividend_payment_cut([("2026-06-15", 2.00), ("2026-09-15", 1.80)])
    assert _cut and "1.80" in _cut and "2.00" in _cut, _cut
    assert m._dividend_payment_cut([("2026-06-15", 2.00), ("2026-09-15", 2.00)]) is None, "flat is not a cut"
    assert m._dividend_payment_cut([("2026-09-15", 2.00)]) is None, "one payment is not enough"
    assert m._dividend_payment_cut([]) is None, "no payments -> no flag"
    # Payout jump: +10 points, or crossing above 100%.
    assert m._payout_jump_check(0.65, 0.50) is not None, "50% -> 65% is a jump"
    assert m._payout_jump_check(0.55, 0.50) is None, "small move is not a jump"
    assert m._payout_jump_check(1.05, 0.95) is not None, "crossing 100% flags"
    assert m._payout_jump_check(1.05, 1.02) is None, "already above 100% does not re-flag"
    # Splits after a purchase date are listed; earlier splits are not.
    splits = [("2020-08-31", 4.0), ("2026-01-15", 2.0)]
    after = m._splits_after_purchase(splits, "2019-06-01")
    assert [s["date"] for s in after] == ["2020-08-31", "2026-01-15"], after
    assert m._splits_after_purchase(splits, "2021-01-01") == [after[1]], "only later splits"
    assert m._splits_after_purchase(splits, None) == [], "no purchase date -> none"
    assert m._splits_after_purchase(None, "2019-06-01") == [], "no split list -> none"
    assert m._splits_after_purchase([("2020-08-31", 1.0)], "2019-06-01") == [], "1:1 is not a split"
    # Audit table rows: years, values, median YoY, 10-yr CAGR.
    gk = {"big_five": {"Sales": {10: {"cagr": 0.072, "status": "fail"}}}}
    table = m._audit_table_rows(audit, gk)
    assert table["years"] == list(range(2015, 2026)), table["years"]
    sales_row = next(r for r in table["rows"] if r["metric"] == "Sales")
    assert sales_row["values"][-1] == 2000 and sales_row["values"][0] == 1000, sales_row["values"]
    # Sales YoY steps run 10% down to ~5.3%; their median is ~6.9%.
    assert abs(sales_row["median_yoy"] - 0.069) < 0.003, sales_row["median_yoy"]
    assert sales_row["cagr10"] == 0.072, sales_row["cagr10"]
    print("ALL ACCURACY-DISPLAY TESTS PASSED")


if __name__ == "__main__":
    main()
