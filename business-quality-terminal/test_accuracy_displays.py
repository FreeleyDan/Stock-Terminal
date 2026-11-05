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
    wanted = {"_parse_purchase_date", "_audit_table_rows", "_splits_after_purchase"}
    ns: Dict[str, Any] = {
        "median_yoy_growth": median_yoy_growth,
        "List": List, "Dict": Dict, "Optional": Optional, "Any": Any,
        "date": _date, "datetime": _datetime,
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
