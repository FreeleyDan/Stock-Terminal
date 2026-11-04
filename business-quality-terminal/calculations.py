"""
calculations.py

Pure, side-effect-free financial math for a Rule #1 Investing (Phil Town) /
Buffett-style "business quality" analysis. No I/O, no external calls.

Every function is wrapped in try/except and returns None on any bad input
(non-numeric, zero/negative where undefined, missing data) so callers
(gatekeeper.py, app.py) never need to guard against exceptions bubbling up
from here -- they only need to check for None.
"""

from __future__ import annotations

from typing import Dict, Optional


def calculate_cagr(
    start_value: float, end_value: float, periods: float
) -> Optional[float]:
    """
    Compound Annual Growth Rate.

    Rationale (Rule #1 / Phil Town "Big Five" growth test):
    Every Big Five metric (ROIC, Equity/Share, EPS, Sales, Free Cash Flow)
    is judged by its historical CAGR over 1/3/5/10-year windows. A wonderful
    business compounds these metrics at >=10%/year -- roughly the long-run
    return Buffett considers the minimum worth the risk of owning a single
    business instead of an index fund. CAGR (not simple average growth) is
    used because it correctly accounts for compounding over multi-year
    windows.

    Formula: (end_value / start_value) ** (1 / periods) - 1

    Returns None if start_value <= 0, end_value <= 0 (CAGR is undefined /
    not meaningful across a sign change, e.g. negative-to-positive equity),
    periods <= 0, or inputs are non-numeric.
    """
    try:
        start_value = float(start_value)
        end_value = float(end_value)
        periods = float(periods)
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return None
        return (end_value / start_value) ** (1.0 / periods) - 1.0
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def calculate_owner_earnings(
    net_income: float, depreciation_amortization: float, cap_ex: float
) -> Optional[float]:
    """
    Owner Earnings (Buffett's preferred substitute for reported EPS/net
    income as a measure of true cash-generating power).

    Rationale: In his 1986 shareholder letter, Buffett defined owner
    earnings as the cash a business could distribute to owners without
    impairing its competitive position or unit volume -- net income plus
    non-cash depreciation/amortization charges, less the capital
    expenditure required to maintain the business. It strips out the
    accounting non-cash add-backs and nets out the reinvestment a business
    MUST make just to stand still, giving a cleaner picture than GAAP net
    income of what a business could actually pay out.

    Formula: net_income + depreciation_amortization - cap_ex

    Note: cap_ex is expected as a positive number representing cash spent
    on capital expenditures (this module does not assume a sign
    convention from the raw XBRL fact -- normalize sign in data_fetcher.py
    before calling this).

    Returns None on non-numeric input.
    """
    try:
        net_income = float(net_income)
        depreciation_amortization = float(depreciation_amortization)
        cap_ex = float(cap_ex)
        return net_income + depreciation_amortization - cap_ex
    except (TypeError, ValueError):
        return None


def calculate_sticker_price(
    current_eps: float,
    estimated_growth_rate: float,
    historical_pe: float,
    discount_rate: float = 0.15,
) -> Optional[float]:
    """
    Sticker Price -- Phil Town's "what is this business worth today"
    number, built directly from Rule #1 Investing's "Four Ms" valuation
    method (itself a simplified, retail-friendly version of a discounted
    cash flow on EPS).

    Method:
      1. Project EPS 10 years forward at estimated_growth_rate.
      2. Apply a projected future P/E equal to
         min(historical_pe, 2 * growth_rate_as_a_whole_number).
         Rationale: Phil Town caps the exit multiple at 2x the growth
         rate (e.g. a 15% grower gets at most a 30 P/E) as a sanity
         check against paying a "story premium" for growth that a
         business's own historical trading multiple wouldn't support --
         you use whichever of the two is MORE conservative (the min).
      3. Future stock value = projected EPS (year 10) * future PE.
      4. Discount that future value back to the present at
         discount_rate (Phil Town's default "minimum acceptable rate of
         return" is 15%, and the app now defaults to 15% per the user's
         choice; callers should still pass their own required return).

    growth_rate_as_whole_number: estimated_growth_rate is expected as a
    decimal (0.12 for 12%); internally converted to 12 for the 2x-cap
    comparison against historical_pe, which is itself expressed as a
    plain multiple (e.g. 18.5).

    Returns None if current_eps <= 0 (the method is not meaningful for a
    business with no current earnings), estimated_growth_rate <= -1
    (would imply negative-or-worse compounding), historical_pe <= 0, or
    discount_rate <= -1, or on any non-numeric input.
    """
    try:
        current_eps = float(current_eps)
        estimated_growth_rate = float(estimated_growth_rate)
        historical_pe = float(historical_pe)
        discount_rate = float(discount_rate)

        if current_eps <= 0:
            return None
        if estimated_growth_rate <= -1.0:
            return None
        if historical_pe <= 0:
            return None
        if discount_rate <= -1.0:
            return None

        periods = 10
        projected_eps = current_eps * ((1.0 + estimated_growth_rate) ** periods)

        growth_rate_as_whole_number = estimated_growth_rate * 100.0
        future_pe = min(historical_pe, 2.0 * growth_rate_as_whole_number)
        if future_pe <= 0:
            # A non-positive growth estimate collapses the 2x-growth cap
            # to <= 0, which makes the future-value step meaningless.
            return None

        future_value = projected_eps * future_pe
        sticker_price = future_value / ((1.0 + discount_rate) ** periods)
        return sticker_price
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def calculate_ddm_value(
    dividend_per_share: float,
    dividend_growth_rate: float,
    discount_rate: float,
) -> Optional[float]:
    """
    Gordon growth Dividend Discount Model value — dividend tab only.

    Value = DPS * (1 + growth) / (discount - growth). This is a
    cross-check on the Rule #1 EPS sticker, not a replacement for it:
    it values only the dividend stream, so it will usually be lower
    than an EPS sticker for a company that retains most earnings.
    Returns None if DPS <= 0, growth < 0, or discount <= growth
    (the formula explodes when growth meets or exceeds the discount
    rate, which is a sign the growth assumption is not sustainable).
    """
    try:
        dps = float(dividend_per_share)
        g = float(dividend_growth_rate)
        r = float(discount_rate)
        if dps <= 0 or g < 0 or r <= g:
            return None
        return dps * (1.0 + g) / (r - g)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def calculate_margin_of_safety(sticker_price: float) -> Optional[float]:
    """
    Margin of Safety (MOS) Price.

    Rationale: Phil Town's Rule #1 -- "don't lose money" -- is operationalized
    by never paying more than half of a business's calculated intrinsic
    (sticker) value. The 50% discount is a deliberate margin of error: it
    cushions against optimistic growth-rate assumptions, forecasting error,
    and the risk that a wonderful business's moat erodes faster than
    expected. Buying at or below the MOS price aims to roughly double your
    money over the payback of the valuation even if the sticker price
    estimate itself was too high.

    Formula: sticker_price * 0.50

    Returns None if sticker_price is None, non-numeric, or <= 0.
    """
    try:
        sticker_price = float(sticker_price)
        if sticker_price <= 0:
            return None
        return sticker_price * 0.50
    except (TypeError, ValueError):
        return None


def estimate_growth_rate_from_eps_history(eps_history: Dict[int, float]) -> Optional[float]:
    """
    Estimates a forward growth rate for the Sticker Price calculation
    from historical EPS CAGR, preferring the longest available horizon
    (10yr) and falling back to shorter ones (5/3/1yr) if longer history
    isn't available.

    Rationale: a simplification of Rule #1 Investing's growth-rate step.
    A full workflow also weighs analyst estimates and Sales/Equity
    growth, but EPS CAGR is a defensible single proxy when only SEC
    EDGAR history is wired up (no analyst-estimate data source here).

    eps_history: {years_ago: eps_value}, where 0 is the most recent
    fiscal year. Returns None if the current year's EPS is missing or
    non-positive, or if no prior horizon has usable data.
    """
    try:
        current_eps = eps_history.get(0)
        if current_eps is None or current_eps <= 0:
            return None
        for years_ago in (10, 5, 3, 1):
            past_eps = eps_history.get(years_ago)
            if past_eps is None:
                continue
            cagr = calculate_cagr(past_eps, current_eps, years_ago)
            if cagr is not None:
                return cagr
        return None
    except (TypeError, AttributeError):
        return None
