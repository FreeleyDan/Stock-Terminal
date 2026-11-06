"""
test_synthetic.py

Builds a synthetic-but-structurally-real SEC EDGAR companyfacts payload
and runs the actual extraction/calculation pipeline against it (no
network). Deliberately includes every pitfall called out in the original
request: nested taxonomies, correct unit keys, multiple fiscal years, a
restated figure, a stray quarterly-length duration fact, and a 10-Q
mixed in. Also importable by other test scripts for its
build_mock_companyfacts() helper.
"""

import json

import pandas as pd

from calculations import (
    calculate_cagr,
    calculate_margin_of_safety,
    calculate_owner_earnings,
    calculate_sticker_price,
)
from data_fetcher import _build_financial_history, _dividends_from_history, _ttm_from_series
from gatekeeper import evaluate_business_quality


def _duration_fact(start, end, val, form="10-K", filed="2024-11-01", fy=2024, fp="FY", accn="0001"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed, "fy": fy, "fp": fp, "accn": accn}


def _instant_fact(end, val, form="10-K", filed="2024-11-01", fy=2024, fp="FY", accn="0001"):
    return {"end": end, "val": val, "form": form, "filed": filed, "fy": fy, "fp": fp, "accn": accn}


def build_mock_companyfacts():
    """
    Fiscal year-ends: FY2024-09-30 (year 0), FY2023, FY2021 (3yr),
    FY2019 (5yr), FY2014 (10yr) -- roughly, with realistic drift.
    Values grow at roughly 12%/year so growth tests should PASS.
    """

    def eps_val(years_ago, base=6.0, rate=0.12):
        return round(base / ((1 + rate) ** years_ago), 4)

    def money_val(base, years_ago, rate=0.12):
        return round(base / ((1 + rate) ** years_ago), 2)

    fy_ends = {
        0: "2024-09-30",
        1: "2023-09-30",
        3: "2021-09-30",
        5: "2019-09-30",
        10: "2014-09-30",
    }

    net_income_entries = []
    eps_entries = []
    sales_entries = []
    op_cash_flow_entries = []
    cap_ex_entries = []
    dep_amort_entries = []
    shares_entries = []

    for years_ago, end_date in fy_ends.items():
        start_year = int(end_date[:4]) - 1
        start_date = f"{start_year}-10-01"

        net_income_entries.append(
            _duration_fact(start_date, end_date, money_val(1_000_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        )
        eps_entries.append(
            _duration_fact(start_date, end_date, eps_val(years_ago), filed=f"{end_date[:4]}-11-15")
        )
        sales_entries.append(
            _duration_fact(start_date, end_date, money_val(10_000_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        )
        op_cash_flow_entries.append(
            _duration_fact(start_date, end_date, money_val(1_500_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        )
        cap_ex_entries.append(
            _duration_fact(start_date, end_date, money_val(300_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        )
        dep_amort_entries.append(
            _duration_fact(start_date, end_date, money_val(200_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        )
        shares_entries.append(
            _duration_fact(start_date, end_date, money_val(150_000_000, years_ago, rate=-0.01), filed=f"{end_date[:4]}-11-15")
        )

    # --- restatement trap: FY2023 Net Income has TWO filings; the later
    # filed one (a restatement, lower value) must win over the earlier one.
    net_income_entries.append(
        _duration_fact("2022-10-01", "2023-09-30", 999_999_999, filed="2023-11-10", accn="0000-early")
    )
    net_income_entries.append(
        _duration_fact("2022-10-01", "2023-09-30", 915_000_000, filed="2024-02-01", accn="0000-restated")
    )

    # --- stray quarterly-length duration fact mixed into the annual tag
    net_income_entries.append(
        _duration_fact("2024-07-01", "2024-09-30", 260_000_000, filed="2024-11-01", accn="0000-q4")
    )

    # --- a 10-Q form mixed in (must be dropped by the form filter)
    net_income_entries.append(
        _duration_fact("2024-01-01", "2024-06-30", 500_000_000, form="10-Q", filed="2024-08-01", accn="0000-10q")
    )

    # --- 10-Q entries for the TTM test: Q1 FY2025 and the same quarter
    # a year earlier. TTM EPS = 6.00 + 1.60 − 1.40 = 6.20.
    eps_entries.append(_duration_fact("2024-10-01", "2024-12-31", 1.60, form="10-Q", filed="2025-02-01", fy=2025, fp="Q1", accn="0000-q1-25"))
    eps_entries.append(_duration_fact("2023-10-01", "2023-12-31", 1.40, form="10-Q", filed="2024-02-01", fy=2024, fp="Q1", accn="0000-q1-24"))
    net_income_entries.append(_duration_fact("2024-10-01", "2024-12-31", 266_666_667, form="10-Q", filed="2025-02-01", fy=2025, fp="Q1", accn="0000-q1-25"))
    net_income_entries.append(_duration_fact("2023-10-01", "2023-12-31", 233_333_333, form="10-Q", filed="2024-02-01", fy=2024, fp="Q1", accn="0000-q1-24"))
    op_cash_flow_entries.append(_duration_fact("2024-10-01", "2024-12-31", 400_000_000, form="10-Q", filed="2025-02-01", fy=2025, fp="Q1", accn="0000-q1-25"))
    op_cash_flow_entries.append(_duration_fact("2023-10-01", "2023-12-31", 350_000_000, form="10-Q", filed="2024-02-01", fy=2024, fp="Q1", accn="0000-q1-24"))
    cap_ex_entries.append(_duration_fact("2024-10-01", "2024-12-31", 80_000_000, form="10-Q", filed="2025-02-01", fy=2025, fp="Q1", accn="0000-q1-25"))
    cap_ex_entries.append(_duration_fact("2023-10-01", "2023-12-31", 70_000_000, form="10-Q", filed="2024-02-01", fy=2024, fp="Q1", accn="0000-q1-24"))
    dep_amort_entries.append(_duration_fact("2024-10-01", "2024-12-31", 55_000_000, form="10-Q", filed="2025-02-01", fy=2025, fp="Q1", accn="0000-q1-25"))
    dep_amort_entries.append(_duration_fact("2023-10-01", "2023-12-31", 50_000_000, form="10-Q", filed="2024-02-01", fy=2024, fp="Q1", accn="0000-q1-24"))

    equity_entries = [
        _instant_fact(end_date, money_val(4_000_000_000, years_ago), filed=f"{end_date[:4]}-11-15")
        for years_ago, end_date in fy_ends.items()
    ]
    lt_debt_entries = [
        _instant_fact(fy_ends[0], 1_800_000_000, filed="2024-11-15"),
    ]
    cash_entries = [
        _instant_fact(fy_ends[0], 900_000_000, filed="2024-11-15"),
    ]

    companyfacts = {
        "cik": 1234567,
        "entityName": "Synthetic Test Co",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": net_income_entries}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps_entries}},
                "Revenues": {"units": {"USD": sales_entries}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": op_cash_flow_entries}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": cap_ex_entries}},
                "DepreciationDepletionAndAmortization": {"units": {"USD": dep_amort_entries}},
                "StockholdersEquity": {"units": {"USD": equity_entries}},
                "LongTermDebtNoncurrent": {"units": {"USD": lt_debt_entries}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": cash_entries}},
            },
            "dei": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": shares_entries}},
            },
        },
    }
    return companyfacts


def main():
    print("=" * 70)
    print("STEP 1: Synthetic mock EDGAR payload extraction test")
    print("=" * 70)

    mock = build_mock_companyfacts()
    financial_history = _build_financial_history(mock)

    errors = []
    if not financial_history["EPS"]:
        errors.append("FAIL: EPS series is empty")
    if not financial_history["Equity_Per_Share"]:
        errors.append("FAIL: Equity_Per_Share is empty")

    net_income_history = financial_history.get("_net_income_history", {})
    if net_income_history.get(1) != 915_000_000:
        errors.append(f"FAIL: restatement handling got {net_income_history.get(1)}")

    gatekeeper_result = evaluate_business_quality(financial_history)
    eps_1yr = gatekeeper_result["big_five"]["EPS"][1]
    if eps_1yr["status"] != "pass":
        errors.append(f"FAIL: EPS 1yr growth check: {eps_1yr}")

    # Dividend payments are extracted from the same yfinance history frame
    # (actions=True) that carries splits — no extra fetch. Zeros and NaN are
    # not payments; only the last 12 are kept.
    hist = pd.DataFrame(
        {"Dividends": [0.0, 0.25, float("nan"), 0.26, 0.26, 0.27]},
        index=pd.to_datetime(["2025-03-14", "2025-06-13", "2025-09-12", "2025-12-12", "2026-03-13", "2026-06-12"]),
    )
    got_divs = _dividends_from_history(hist)
    if got_divs != [("2025-06-13", 0.25), ("2025-12-12", 0.26), ("2026-03-13", 0.26), ("2026-06-12", 0.27)]:
        errors.append(f"FAIL: dividends extraction got {got_divs}")
    if _dividends_from_history(pd.DataFrame({"Close": [1.0]})) != []:
        errors.append("FAIL: no Dividends column -> empty list")

    # Trailing-12-month values from the 10-Q entries: annual + latest Q
    # − same Q last year. EPS 6.00 + 1.60 − 1.40 = 6.20.
    ttm = financial_history.get("_ttm") or {}
    if ttm.get("eps") is None or abs(ttm["eps"] - 6.20) > 1e-9:
        errors.append(f"FAIL: TTM EPS got {ttm.get('eps')}, expected 6.20")
    if ttm.get("used") is not True or ttm.get("as_of") != "2024-12-31":
        errors.append(f"FAIL: TTM flag/as_of got used={ttm.get('used')} as_of={ttm.get('as_of')}")
    if ttm.get("annual_eps") != 6.0:
        errors.append(f"FAIL: annual EPS got {ttm.get('annual_eps')}")
    if abs((ttm.get("owner_earnings") or 0) - 928_333_334) > 1:
        errors.append(f"FAIL: TTM owner earnings got {ttm.get('owner_earnings')}")
    if abs((ttm.get("free_cash_flow") or 0) - 1_240_000_000) > 1:
        errors.append(f"FAIL: TTM FCF got {ttm.get('free_cash_flow')}")
    if financial_history.get("_last_fy_end") != "2024-09-30":
        errors.append(f"FAIL: last FY end got {financial_history.get('_last_fy_end')}")
    # Annual horizons must be unchanged by the 10-Q entries.
    if financial_history["EPS"].get(0) != 6.0:
        errors.append(f"FAIL: annual EPS changed by 10-Q entries: {financial_history['EPS'].get(0)}")

    # _ttm_from_series edge cases (direct).
    _v, _a, _u = _ttm_from_series({"2024-09-30": 6.0}, {"2024-12-31": 1.6, "2023-12-31": 1.4})
    if abs(_v - 6.2) > 1e-9 or _a != "2024-12-31" or _u is not True:
        errors.append(f"FAIL: _ttm_from_series basic got {(_v, _a, _u)}")
    _v, _a, _u = _ttm_from_series({"2024-09-30": 6.0}, {"2024-06-30": 1.5, "2023-06-30": 1.3})
    if _v != 6.0 or _u is not False:
        errors.append(f"FAIL: quarter before year-end must fall back to annual: {(_v, _a, _u)}")
    _v, _a, _u = _ttm_from_series({"2024-09-30": 6.0}, {"2024-12-31": 1.6})
    if _v != 6.0 or _u is not False:
        errors.append(f"FAIL: missing year-ago quarter must fall back to annual: {(_v, _a, _u)}")
    if _ttm_from_series({}, {"2024-12-31": 1.6}) != (None, None, False):
        errors.append("FAIL: no annual data -> no TTM")

    print(f"Moat Score: {gatekeeper_result['moat_score']} / 100")
    if errors:
        print(f"{len(errors)} FAILURE(S):")
        for e in errors:
            print(" -", e)
    else:
        print("ALL SYNTHETIC TESTS PASSED.")
    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
