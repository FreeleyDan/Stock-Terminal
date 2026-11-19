"""
gatekeeper.py

Applies Phil Town's "Gatekeeper" business-quality checklist to a financial
history dict produced by data_fetcher.py, using the pure math in
calculations.py.

Public entry point: evaluate_business_quality(financial_history)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from calculations import calculate_cagr

# The Big Five Quality Metrics, in the order they're evaluated/displayed.
# (Phil Town calls these the moat metrics; the app renamed the 80/10/10
# score they feed to "Quality Score" in Nov 2026 because the number
# measures compounding quality, not competitive position — the separate
# Moat Score below measures that.)
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
            # fail, since it's a data-shape limitation, not evidence of
            # weak quality.
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


def _compute_quality_score(
    big_five_results: Dict[str, Dict[int, Dict[str, Any]]],
    debt_test: Dict[str, Any],
    cap_alloc_test: Dict[str, Any],
) -> float:
    """
    Weighted Quality Score, 0-100.

    Weighting rationale: the Big Five growth checks are the heart of the
    "wonderful business" thesis (quality shown through compounding), so
    they carry 80% of the score collectively (16% each across 5 metrics),
    split evenly across whichever horizons actually have a determined
    pass/fail (not-yet-determined horizons are excluded from the
    denominator rather than counted as failures, since missing history
    is a data limitation, not a quality signal). The Debt Test and
    Capital Allocation Test are balance-sheet/capital-discipline checks
    layered on top by Rule #1 -- necessary but secondary to the
    growth-quality thesis -- so they split the remaining 20%.
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


# ---------------------------------------------------------------------------
# Moat Score (competitive position, from 10-K history only)
# ---------------------------------------------------------------------------

# A component with fewer than 3 usable observations scores 0 and is named
# in "missing" — the UI flags it with *, the same convention as the
# Dividend Safety Score.
MOAT_MIN_OBSERVATIONS = 3
MOAT_ROIC_LEVEL_CAP = 0.30  # median ROIC at/above this earns full level points
MOAT_ROIC_PERSISTENCE_BAR = 0.15  # a horizon year counts as high-ROIC above this
MOAT_GROSS_MARGIN_CAP = 0.60  # median gross margin at/above this earns full level points
MOAT_REVENUE_GROWTH_STDEV_CAP = 0.30  # revenue-growth stdev at/above this earns 0 stability points


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _stdev(values: List[float]) -> Optional[float]:
    """Population standard deviation; None with fewer than 2 values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _horizon_values(series: Any) -> List[float]:
    """Numeric values from a {years_ago: value} horizon dict, oldest
    first. Non-numeric entries are skipped."""
    if not isinstance(series, dict):
        return []
    out: List[float] = []
    for years_ago in sorted(series):
        try:
            out.append(float(series[years_ago]))
        except (TypeError, ValueError):
            continue
    return out


def _margin_values(numerator: Any, denominator: Any) -> List[float]:
    """Per-horizon margin (numerator / denominator) where both exist and
    the denominator is positive, oldest horizon first."""
    if not isinstance(numerator, dict) or not isinstance(denominator, dict):
        return []
    out: List[float] = []
    for years_ago in sorted(denominator):
        try:
            den = float(denominator[years_ago])
            num = float(numerator[years_ago])
        except (TypeError, ValueError, KeyError):
            continue
        if den > 0:
            out.append(num / den)
    return out


def _annualized_growth_rates(series: Any) -> List[float]:
    """Annualized growth between each pair of consecutive available
    horizons (0->1, 1->3, 3->5, 5->10). Pairs where either value is
    <= 0 are skipped — a growth rate isn't meaningful there."""
    if not isinstance(series, dict):
        return []
    years = sorted(series)
    rates: List[float] = []
    for a, b in zip(years, years[1:]):
        try:
            va = float(series[a])
            vb = float(series[b])
        except (TypeError, ValueError):
            continue
        span = b - a
        if va > 0 and vb > 0 and span > 0:
            rates.append((va / vb) ** (1.0 / span) - 1.0)
    return rates


def _cv_stability(values: List[float]) -> Optional[float]:
    """1 - coefficient of variation (stdev / median), clamped to [0, 1].
    None when the median is not positive — stability is meaningless
    without a positive base level."""
    med = _median(values)
    if med is None or med <= 0:
        return None
    sd = _stdev(values)
    if sd is None:
        return None
    return _clamp01(1.0 - sd / med)


def _compute_moat_score(financial_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Competitive-moat footprint score, 0-100, from 10-K history only.

    Filings carry no market share, pricing power vs competitors, or
    switching costs, so this is a footprint, not a Morningstar-style
    rating: a business with a real moat tends to leave these traces,
    and a business without one rarely fakes all four at once.
      ROIC level (30): median ROIC over the horizon years; 30%+ is full.
      ROIC persistence (25): share of horizon years with ROIC above 15%.
      Gross margin (25): level 15 (median margin; 60%+ is full) +
        stability 10 (1 - coefficient of variation, floored at 0).
      Revenue + operating-margin stability (20): revenue stability 10
        (1 - stdev of annualized horizon growth rates / 30%, floored
        at 0) + operating-margin stability 10 (1 - CV of operating
        margins, floored at 0).
    Every component is computed from the {0,1,3,5,10}-year horizon dicts
    data_fetcher builds; a component with fewer than 3 usable
    observations scores 0 and is named in "missing".
    """
    components: Dict[str, float] = {}
    missing: List[str] = []

    roic_values = _horizon_values(financial_history.get("ROIC"))
    if len(roic_values) >= MOAT_MIN_OBSERVATIONS:
        med = _median(roic_values) or 0.0
        components["roic_level"] = round(_clamp01(med / MOAT_ROIC_LEVEL_CAP) * 30.0, 1)
        high = sum(1 for v in roic_values if v > MOAT_ROIC_PERSISTENCE_BAR)
        components["roic_persistence"] = round(high / len(roic_values) * 25.0, 1)
    else:
        components["roic_level"] = 0.0
        components["roic_persistence"] = 0.0
        missing.append("roic")

    gross_margins = _margin_values(financial_history.get("Gross_Profit"), financial_history.get("Sales"))
    if len(gross_margins) >= MOAT_MIN_OBSERVATIONS:
        med_gm = _median(gross_margins) or 0.0
        level = _clamp01(med_gm / MOAT_GROSS_MARGIN_CAP) * 15.0
        stability = _cv_stability(gross_margins)
        components["gross_margin"] = round(level + (_clamp01(stability) * 10.0 if stability is not None else 0.0), 1)
    else:
        components["gross_margin"] = 0.0
        missing.append("gross_margin")

    growth_rates = _annualized_growth_rates(financial_history.get("Sales"))
    if len(growth_rates) >= MOAT_MIN_OBSERVATIONS:
        sd = _stdev(growth_rates) or 0.0
        components["revenue_stability"] = round(_clamp01(1.0 - sd / MOAT_REVENUE_GROWTH_STDEV_CAP) * 10.0, 1)
    else:
        components["revenue_stability"] = 0.0
        missing.append("revenue_stability")

    op_margins = _margin_values(financial_history.get("Operating_Income"), financial_history.get("Sales"))
    if len(op_margins) >= MOAT_MIN_OBSERVATIONS:
        stability = _cv_stability(op_margins)
        components["operating_margin_stability"] = round(_clamp01(stability) * 10.0 if stability is not None else 0.0, 1)
    else:
        components["operating_margin_stability"] = 0.0
        missing.append("operating_margin_stability")

    return {"score": round(sum(components.values()), 1), "components": components, "missing": missing}


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
            "Gross_Profit": {...},       # full horizon dict (Moat Score)
            "Operating_Income": {...},   # full horizon dict (Moat Score)
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
            "quality_score": float (0-100; the 80/10/10 Big Five + Debt + CapEx score),
            "moat_score": float (0-100; the competitive-moat footprint score),
            "moat_detail": {"score":..., "components":{...}, "missing":[...]},
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

    quality_score = _compute_quality_score(big_five_results, debt_test, cap_alloc_test)
    moat = _compute_moat_score(financial_history)

    action = _determine_action(
        financial_history.get("current_price"),
        financial_history.get("sticker_price"),
        financial_history.get("margin_of_safety_price"),
    )

    return {
        "big_five": big_five_results,
        "debt_test": debt_test,
        "capital_allocation_test": cap_alloc_test,
        "quality_score": quality_score,
        "moat_score": moat["score"],
        "moat_detail": moat,
        "action": action,
    }


# Public alias: valuation.py / signals.py classify price vs. sticker/MOS the
# same way the Gatekeeper does, without reaching for a private name.
determine_action = _determine_action
