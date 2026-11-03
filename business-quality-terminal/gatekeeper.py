"""
gatekeeper.py

Applies Phil Town's "Gatekeeper" business-quality checklist to a financial
history dict produced by data_fetcher.py, using the pure math in
calculations.py.

Public entry point: evaluate_business_quality(financial_history)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from calculations import calculate_cagr

# The Big Five Moat Metrics, in the order they're evaluated/displayed.
BIG_FIVE_METRICS: List[str] = [
    "ROIC",
    "Equity_Per_Share",
    "EPS",
    "Sales",
    "Free_Cash_Flow",
]

HORIZONS: List[int] = [1, 3, 5, 10]

GROWTH_THRESHOLD = 0.10  # >10%/year required to pass a Big Five growth test
DEBT_TEST_MAX_RATIO = 3.0  # Long_Term_Debt / Owner_Earnings <= 3
CAP_ALLOC_MAX_RATIO = 0.50  # Cap_Ex / Net_Income <= 0.50

NOT_YET_DETERMINED = "not_yet_determined"
PASS = "pass"
FAIL = "fail"

ACTION_UNDERVALUED_BUY = "UNDERVALUED_BUY"
ACTION_FAIR_VALUE_HOLD = "FAIR_VALUE_HOLD"
ACTION_OVERPRICED_AVOID = "OVERPRICED_AVOID"


def _series_value_n_years_ago(
    financial_history: Dict[str, Any], metric_key: str, years_ago: int
) -> Optional[float]:
    """
    Look up metric_key's value at `years_ago` from financial_history.

    Expected shape of financial_history[metric_key]: a dict mapping
    years_ago (int, e.g. 0/1/3/5/10) -> numeric value, as produced by
    data_fetcher.py's nearest-fiscal-year matching. Returns None if the
    series or that particular horizon isn't present.
    """
    try:
        series = financial_history.get(metric_key)
        if not series:
            return None
        value = series.get(years_ago)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return None


def _evaluate_metric_growth(
    financial_history: Dict[str, Any], metric_key: str
) -> Dict[int, Dict[str, Any]]:
    """
    For a single Big Five metric, compute CAGR and pass/fail/not-yet-
    determined status across all horizons.

    Returns: {horizon: {"cagr": float|None, "status": "pass"|"fail"|"not_yet_determined"}}
    """
    result: Dict[int, Dict[str, Any]] = {}
    current_value = _series_value_n_years_ago(financial_history, metric_key, 0)

    for horizon in HORIZONS:
        past_value = _series_value_n_years_ago(financial_history, metric_key, horizon)

        if current_value is None or past_value is None:
            result[horizon] = {"cagr": None, "status": NOT_YET_DETERMINED}
            continue

        cagr = calculate_cagr(past_value, current_value, horizon)
        if cagr is None:
            # e.g. sign change (negative-to-positive equity) makes CAGR
            # undefined -- treat as not yet determined rather than a hard
            # fail, since it's a data-shape limitation, not evidence of a
            # weak moat.
            result[horizon] = {"cagr": None, "status": NOT_YET_DETERMINED}
            continue

        status = PASS if cagr > GROWTH_THRESHOLD else FAIL
        result[horizon] = {"cagr": cagr, "status": status}

    return result


def _evaluate_debt_test(financial_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debt Test: Long_Term_Debt / Owner_Earnings <= 3.

    Rationale: measures how many years of full owner earnings it would
    take to pay off all long-term debt. Rule #1 treats anything beyond
    ~3 years as a red flag on balance-sheet resilience, since it leaves
    little room to survive a downturn without diluting shareholders or
    distress-selling assets.
    """
    long_term_debt = _series_value_n_years_ago(financial_history, "Long_Term_Debt", 0)
    owner_earnings = _series_value_n_years_ago(financial_history, "Owner_Earnings", 0)

    if long_term_debt is None or owner_earnings is None:
        return {"ratio": None, "status": NOT_YET_DETERMINED}

    if owner_earnings <= 0:
        # Any debt against zero-or-negative owner earnings is an automatic
        # fail (a payoff-in-years ratio isn't meaningful, but the business
        # clearly can't service debt from its own cash generation).
        if long_term_debt <= 0:
            # No debt and no/negative owner earnings -- no debt burden to
            # fail on.
            return {"ratio": 0.0, "status": PASS}
        return {"ratio": None, "status": FAIL}

    ratio = long_term_debt / owner_earnings
    status = PASS if ratio <= DEBT_TEST_MAX_RATIO else FAIL
    return {"ratio": ratio, "status": status}


def _evaluate_capital_allocation_test(
    financial_history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Capital Allocation Test: Cap_Ex / Net_Income <= 0.50.

    Rationale: a business that must reinvest more than half its net
    income just to fund capital expenditures has weak "owner earnings"
    economics even if its income statement looks fine -- little is left
    over for shareholders, debt paydown, or opportunistic reinvestment.
    """
    cap_ex = _series_value_n_years_ago(financial_history, "Cap_Ex", 0)
    net_income = _series_value_n_years_ago(financial_history, "Net_Income", 0)

    if cap_ex is None or net_income is None:
        return {"ratio": None, "status": NOT_YET_DETERMINED}

    if net_income <= 0:
        # Can't fund cap-ex out of net income at all -- automatic fail,
        # unless there's simply no cap-ex either.
        if cap_ex <= 0:
            return {"ratio": 0.0, "status": PASS}
        return {"ratio": None, "status": FAIL}

    ratio = cap_ex / net_income
    status = PASS if ratio <= CAP_ALLOC_MAX_RATIO else FAIL
    return {"ratio": ratio, "status": status}


def _compute_moat_score(
    big_five_results: Dict[str, Dict[int, Dict[str, Any]]],
    debt_test: Dict[str, Any],
    cap_alloc_test: Dict[str, Any],
) -> float:
    """
    Weighted Moat Score, 0-100.

    Weighting rationale: the Big Five growth checks are the heart of the
    "wonderful business" thesis (moat durability shown through
    compounding), so they carry 80% of the score collectively (16% each
    across 5 metrics), split evenly across whichever horizons actually
    have a determined pass/fail (not-yet-determined horizons are excluded
    from the denominator rather than counted as failures, since missing
    history is a data limitation, not a quality signal). The Debt Test
    and Capital Allocation Test are balance-sheet/capital-discipline
    checks layered on top by Rule #1 -- necessary but secondary to
    the growth-quality thesis -- so they split the remaining 20%.
    """
    metric_weight_total = 80.0
    per_metric_weight = metric_weight_total / len(BIG_FIVE_METRICS)

    earned = 0.0

    for metric_key in BIG_FIVE_METRICS:
        horizon_results = big_five_results.get(metric_key, {})
        determined = [
            h for h in HORIZONS if horizon_results.get(h, {}).get("status") in (PASS, FAIL)
        ]
        if not determined:
            continue
        per_horizon_weight = per_metric_weight / len(determined)
        for h in determined:
            if horizon_results[h]["status"] == PASS:
                earned += per_horizon_weight

    # Debt test + capital allocation test: 10 points each, all-or-nothing.
    if debt_test.get("status") == PASS:
        earned += 10.0
    if cap_alloc_test.get("status") == PASS:
        earned += 10.0

    return round(earned, 1)


def _determine_action(
    current_price: Optional[float],
    sticker_price: Optional[float],
    margin_of_safety_price: Optional[float],
) -> str:
    """
    UNDERVALUED_BUY if price <= MOS price.
    FAIR_VALUE_HOLD if MOS price < price <= sticker price.
    OVERPRICED_AVOID if price > sticker price.

    If price/sticker/MOS aren't all available, defaults to the most
    conservative call, OVERPRICED_AVOID, rather than guessing.
    """
    if current_price is None or sticker_price is None or margin_of_safety_price is None:
        return ACTION_OVERPRICED_AVOID

    if current_price <= margin_of_safety_price:
        return ACTION_UNDERVALUED_BUY
    if current_price <= sticker_price:
        return ACTION_FAIR_VALUE_HOLD
    return ACTION_OVERPRICED_AVOID


def evaluate_business_quality(financial_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full Gatekeeper checklist against a single company's
    financial_history dict (as produced by data_fetcher.py).

    Expected financial_history shape:
        {
            "ROIC": {0: ..., 1: ..., 3: ..., 5: ..., 10: ...},
            "Equity_Per_Share": {...},
            "EPS": {...},
            "Sales": {...},
            "Free_Cash_Flow": {...},
            "Long_Term_Debt": {0: ...},
            "Owner_Earnings": {0: ...},
            "Cap_Ex": {0: ...},
            "Net_Income": {0: ...},
            # Valuation inputs, all optional -- degrade gracefully if absent:
            "current_price": float,
            "sticker_price": float,
            "margin_of_safety_price": float,
        }

    Returns:
        {
            "big_five": {metric: {horizon: {"cagr":..., "status":...}}},
            "debt_test": {"ratio":..., "status":...},
            "capital_allocation_test": {"ratio":..., "status":...},
            "moat_score": float (0-100),
            "action": "UNDERVALUED_BUY" | "FAIR_VALUE_HOLD" | "OVERPRICED_AVOID",
        }

    Never raises: malformed/missing financial_history sections degrade to
    "not_yet_determined" rather than throwing.
    """
    if not isinstance(financial_history, dict):
        financial_history = {}

    big_five_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for metric_key in BIG_FIVE_METRICS:
        big_five_results[metric_key] = _evaluate_metric_growth(financial_history, metric_key)

    debt_test = _evaluate_debt_test(financial_history)
    cap_alloc_test = _evaluate_capital_allocation_test(financial_history)

    moat_score = _compute_moat_score(big_five_results, debt_test, cap_alloc_test)

    action = _determine_action(
        financial_history.get("current_price"),
        financial_history.get("sticker_price"),
        financial_history.get("margin_of_safety_price"),
    )

    return {
        "big_five": big_five_results,
        "debt_test": debt_test,
        "capital_allocation_test": cap_alloc_test,
        "moat_score": moat_score,
        "action": action,
    }


# Public alias: valuation.py / signals.py classify price vs. sticker/MOS the
# same way the Gatekeeper does, without reaching for a private name.
determine_action = _determine_action
