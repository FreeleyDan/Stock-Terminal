"""
signals.py

Rule-based BUY / SELL flags. Pure functions (no I/O) built on a snapshot
and its valuation. These are mechanical screens on the model's numbers,
not financial advice.

BUY (Rule #1: a wonderful business at a margin-of-safety price):
  BUY        price <= Margin of Safety price AND Moat Score >= minimum
  NEAR BUY   quality OK and price within `near_buy_pct`% above the MOS price
  CHEAP, WEAK MOAT   price <= MOS price but Moat Score below the minimum
  WAIT       otherwise
  NO VALUATION  no Sticker Price could be computed

SELL (Rule #1: sell when price reaches fair value, or the business changes):
  SELL       price >= Sticker Price
  REVIEW     price within `approach_pct`% below Sticker, OR Moat Score
             below `min_moat_hold`, OR the Debt Test failed, OR (optional)
             down more than `max_loss_pct`% vs your cost
  HOLD       none of the above

PORTFOLIO (for something you already own — BUY means "add", not a first buy):
  BUY        owned, price <= MOS price, Moat Score >= buy minimum, and no
             quality REVIEW flag (low moat / failed Debt Test)
  NEAR BUY   owned, quality OK, price within `near_buy_pct`% above MOS price
  SELL / REVIEW / HOLD  as in SELL above
Each signal comes with plain-English reasons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BUY = "BUY"
NEAR_BUY = "NEAR BUY"
WEAK = "CHEAP, WEAK MOAT"
WAIT = "WAIT"
NO_VALUATION = "NO VALUATION"

SELL = "SELL"
REVIEW = "REVIEW"
HOLD = "HOLD"

BUY_RANK = {BUY: 0, NEAR_BUY: 1, WEAK: 2, WAIT: 3, NO_VALUATION: 4}
SELL_RANK = {SELL: 0, REVIEW: 1, HOLD: 2, NO_VALUATION: 3}
# Portfolio rows sort with the most actionable first: SELL, then BUY/ADD,
# then NEAR BUY, then REVIEW, then HOLD.
PORTFOLIO_RANK = {SELL: 0, BUY: 1, NEAR_BUY: 2, REVIEW: 3, WEAK: 3, HOLD: 4, NO_VALUATION: 5}


def _money(v: Optional[float]) -> str:
    return "\u2014" if v is None else f"${v:,.2f}"


def buy_signal(
    snapshot: Dict[str, Any],
    valuation: Dict[str, Any],
    min_moat_score: float = 60.0,
    near_buy_pct: float = 15.0,
) -> Dict[str, Any]:
    price = snapshot.get("current_price")
    mos = valuation.get("margin_of_safety_price")
    sticker = valuation.get("sticker_price")
    moat = snapshot.get("moat_score", 0.0)

    if price is None or mos is None or sticker is None:
        return {"signal": NO_VALUATION, "reasons": [valuation.get("note") or "No valuation available"], "pct_above_buy": None}

    pct_above = (price / mos - 1.0) * 100.0
    quality_ok = moat >= min_moat_score
    reasons: List[str] = []

    if price <= mos:
        reasons.append(f"Price {_money(price)} is at/below the margin-of-safety price {_money(mos)}")
        if quality_ok:
            reasons.append(f"Moat Score {moat:g} meets the {min_moat_score:g} minimum")
            signal = BUY
        else:
            reasons.append(f"But Moat Score {moat:g} is below the {min_moat_score:g} minimum")
            signal = WEAK
    elif quality_ok and pct_above <= near_buy_pct:
        reasons.append(f"Price is {pct_above:.1f}% above the {_money(mos)} buy price (within {near_buy_pct:g}%)")
        signal = NEAR_BUY
    else:
        reasons.append(f"Price is {pct_above:.1f}% above the {_money(mos)} buy price")
        if not quality_ok:
            reasons.append(f"Moat Score {moat:g} is below the {min_moat_score:g} minimum")
        signal = WAIT
    return {"signal": signal, "reasons": reasons, "pct_above_buy": pct_above}


def sell_signal(
    snapshot: Dict[str, Any],
    valuation: Dict[str, Any],
    avg_cost: Optional[float] = None,
    approach_pct: float = 10.0,
    min_moat_hold: float = 40.0,
    max_loss_pct: float = 0.0,
) -> Dict[str, Any]:
    price = snapshot.get("current_price")
    sticker = valuation.get("sticker_price")
    moat = snapshot.get("moat_score", 0.0)
    gk = snapshot.get("gatekeeper") or {}

    signal = HOLD
    reasons: List[str] = []

    def escalate(new: str) -> None:
        nonlocal signal
        if SELL_RANK[new] < SELL_RANK[signal]:
            signal = new

    if price is None or sticker is None:
        escalate(REVIEW)
        reasons.append((valuation.get("note") or "No valuation available") + " \u2014 can't judge price vs fair value")
    elif price >= sticker:
        escalate(SELL)
        reasons.append(f"Price {_money(price)} is at/above Sticker Price {_money(sticker)} (fair value reached)")
    elif price >= sticker * (1 - approach_pct / 100.0):
        escalate(REVIEW)
        reasons.append(f"Price is within {approach_pct:g}% of Sticker Price {_money(sticker)}")

    if moat < min_moat_hold:
        escalate(REVIEW)
        reasons.append(f"Moat Score {moat:g} is below {min_moat_hold:g}")

    if (gk.get("debt_test") or {}).get("status") == "fail":
        escalate(REVIEW)
        reasons.append("Debt Test failed (long-term debt > 3x owner earnings)")

    if max_loss_pct > 0 and avg_cost and price is not None and avg_cost > 0:
        loss_pct = (avg_cost - price) / avg_cost * 100.0
        if loss_pct >= max_loss_pct:
            escalate(REVIEW)
            reasons.append(f"Down {loss_pct:.1f}% vs your cost (your {max_loss_pct:g}% limit)")

    if not reasons:
        reasons.append("Below fair value with acceptable quality")
    return {"signal": signal, "reasons": reasons}


def portfolio_signal(
    snapshot: Dict[str, Any],
    valuation: Dict[str, Any],
    avg_cost: Optional[float] = None,
    approach_pct: float = 10.0,
    min_moat_hold: float = 40.0,
    max_loss_pct: float = 0.0,
    min_moat_buy: float = 60.0,
    near_buy_pct: float = 15.0,
) -> Dict[str, Any]:
    """BUY / NEAR BUY / HOLD / REVIEW / SELL for a holding you already own.

    Priority, in order: SELL (fair value reached) > REVIEW for a quality
    problem (low moat or failed Debt Test — never add to a deteriorating
    business just because it is cheap) > BUY / NEAR BUY (wonderful business
    at or near its MOS price) > REVIEW for price approaching Sticker or
    for a loss vs cost beyond the limit > HOLD.

    A loss vs cost alone does not block a BUY: being down is not a
    business-quality signal, and a wonderful business that fell to its
    MOS price can still be an add. It is still noted in the reasons.
    """
    price = snapshot.get("current_price")
    sticker = valuation.get("sticker_price")
    mos = valuation.get("margin_of_safety_price")
    moat = snapshot.get("moat_score", 0.0)
    gk = snapshot.get("gatekeeper") or {}
    debt_failed = (gk.get("debt_test") or {}).get("status") == "fail"
    quality_review = moat < min_moat_hold or debt_failed

    if price is None or sticker is None or mos is None:
        sell = sell_signal(snapshot, valuation, avg_cost, approach_pct, min_moat_hold, max_loss_pct)
        return {"signal": REVIEW, "reasons": sell["reasons"]}

    if price >= sticker:
        return {
            "signal": SELL,
            "reasons": [f"Price {_money(price)} is at/above Sticker Price {_money(sticker)} (fair value reached)"],
        }

    if quality_review:
        reasons = []
        if moat < min_moat_hold:
            reasons.append(f"Moat Score {moat:g} is below {min_moat_hold:g} — do not add")
        if debt_failed:
            reasons.append("Debt Test failed (long-term debt > 3x owner earnings) — do not add")
        return {"signal": REVIEW, "reasons": reasons}

    buy = buy_signal(snapshot, valuation, min_moat_buy, near_buy_pct)
    if buy["signal"] == BUY:
        return {"signal": BUY, "reasons": ["Owned holding — add signal:"] + buy["reasons"]}
    if buy["signal"] == NEAR_BUY:
        return {"signal": NEAR_BUY, "reasons": ["Owned holding — near-add signal:"] + buy["reasons"]}
    if buy["signal"] == WEAK:
        return {"signal": REVIEW, "reasons": buy["reasons"] + ["Do not add — moat is too weak"]}

    reasons: List[str] = []
    if price >= sticker * (1 - approach_pct / 100.0):
        reasons.append(f"Price is within {approach_pct:g}% of Sticker Price {_money(sticker)}")
    if max_loss_pct > 0 and avg_cost and price is not None and avg_cost > 0:
        loss_pct = (avg_cost - price) / avg_cost * 100.0
        if loss_pct >= max_loss_pct:
            reasons.append(f"Down {loss_pct:.1f}% vs your cost (your {max_loss_pct:g}% limit)")
    if reasons:
        return {"signal": REVIEW, "reasons": reasons}
    return {"signal": HOLD, "reasons": ["Below fair value with acceptable quality"]}
