"""
test_all.py -- offline tests (no network) for the data layer, valuation,
signals, storage, scanner and universe modules. Run: python3 test_all.py
"""
import json
import os
import tempfile
import threading
import time

import data_fetcher as df_mod
import gatekeeper
import screener
import signals
import storage
import universe
import valuation
from calculations import calculate_sticker_price
from test_synthetic import _duration_fact, _instant_fact, build_mock_companyfacts

# Isolate the disk cache from any real cache on this machine.
TMP = tempfile.mkdtemp()
df_mod.DISK_CACHE_DIR = os.path.join(TMP, "cache")

failures = []


def check(label, actual, expected=True, approx=None):
    ok = (abs(actual - expected) <= approx) if approx is not None else (actual == expected)
    print(f"{'OK  ' if ok else 'FAIL'} {label}" + ("" if ok else f"  -> got {actual!r}, expected {expected!r}"))
    if not ok:
        failures.append(label)


def section(title):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
section("Horizon matching tolerance (was +/-600 days at 10yr)")
series = {"2024-09-30": 10.0, "2023-09-30": 9.0, "2016-03-31": 2.0}  # 8.5yr-old value only
m = df_mod._map_series_to_horizons(series)
check("an 8.5-year-old figure is NOT matched as '10 years ago'", 10 in m, False)
exact = df_mod._map_series_to_horizons({"2024-09-30": 10.0, "2014-10-05": 2.0})
check("a genuine ~10-year-old figure (5 days drift) still matches", exact.get(10), 2.0)

# --------------------------------------------------------------------------
section("Stock-split adjustment (EDGAR reports per-share data as originally filed)")


def split_payload():
    # 4:1 split on 2023-06-01. FY2022 EPS/DPS were filed BEFORE the split
    # (pre-split shares); FY2023+ were filed after (post-split shares).
    eps = [
        _duration_fact("2021-10-01", "2022-09-30", 4.00, filed="2022-11-01"),
        _duration_fact("2022-10-01", "2023-09-30", 1.20, filed="2023-11-01"),
        _duration_fact("2023-10-01", "2024-09-30", 1.50, filed="2024-11-01"),
    ]
    dps = [
        _duration_fact("2021-10-01", "2022-09-30", 2.00, filed="2022-11-01"),
        _duration_fact("2022-10-01", "2023-09-30", 0.60, filed="2023-11-01"),
        _duration_fact("2023-10-01", "2024-09-30", 0.70, filed="2024-11-01"),
    ]
    shares = [
        _duration_fact("2021-10-01", "2022-09-30", 100e6, filed="2022-11-01"),
        _duration_fact("2022-10-01", "2023-09-30", 400e6, filed="2023-11-01"),
    ]
    return {"facts": {"us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
        "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": dps}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": shares}},
    }}}


SPLITS = [("2023-06-01", 4.0)]
raw_hist = df_mod._build_financial_history(split_payload())
adj_hist = df_mod._build_financial_history(split_payload(), SPLITS)
check("unadjusted FY2022 EPS is the raw 4.00 (the trap)", raw_hist["_eps_history_by_date"]["2022-09-30"], 4.00)
check("adjusted FY2022 EPS = 4.00 / 4 = 1.00", adj_hist["_eps_history_by_date"]["2022-09-30"], 1.00)
check("EPS filed AFTER the split is left alone", adj_hist["_eps_history_by_date"]["2023-09-30"], 1.20)
check("dividends/share adjusted too (2.00 -> 0.50)", adj_hist["_dividends_per_share_history"]["2022-09-30"], 0.50)
check("pre-split share count scaled up (100M -> 400M)",
      df_mod._get_best_tag_series(split_payload(), "Shares_Outstanding", SPLITS)["2022-09-30"], 400e6)
check("dividend streak is 2 yrs once split-adjusted (0.50 -> 0.60 -> 0.70)",
      valuation.consecutive_dividend_growth_years(adj_hist["_dividends_per_share_history"]), 2)
check("...but unadjusted, the split looks like a dividend CUT (2.00 -> 0.60) and the streak is cut short",
      valuation.consecutive_dividend_growth_years(raw_hist["_dividends_per_share_history"]), 1)
check("reverse split (1:10 = 0.1) multiplies per-share values",
      df_mod._split_factor("2022-01-01", [("2023-01-01", 0.1)]), 0.1, approx=1e-12)
check("split before the filing date is ignored", df_mod._split_factor("2024-01-01", [("2023-01-01", 4.0)]), 1.0)

# --------------------------------------------------------------------------
section("Historical P/E (median, paired with each year's own price)")
closes = {"2024-09-30": 100.0, "2023-09-29": 80.0, "2022-09-30": 60.0, "2021-09-30": 5000.0}
eps = {"2024-09-30": 5.0, "2023-09-30": 4.0, "2022-09-30": 3.0, "2021-09-30": 0.01, "2020-09-30": -2.0}
pe = df_mod.historical_median_pe(closes, eps)
check("median of [20, 20, 20, 500000] ignores the near-zero-EPS outlier", pe, 20.0, approx=1e-9)
check("loss years skipped; no pairable year -> None", df_mod.historical_median_pe({"2010-01-01": 5.0}, {"2024-09-30": 5.0}), None)
check("weekend/holiday: falls back to prior trading day", df_mod._nearest_price_on_or_before({"2024-09-27": 148.5}, "2024-09-30"), 148.5)

# --------------------------------------------------------------------------
section("Disk cache: None is never cached, int keys survive")
calls = {"n": 0}


@df_mod.cached(ttl_seconds=60)
def flaky(x):
    calls["n"] += 1
    return None if calls["n"] == 1 else {0: "ok", 1: "int keys"}


check("first call fails (None)", flaky(1), None)
check("second call retries (failure wasn't cached)", flaky(1), {0: "ok", 1: "int keys"})
check("third call served from cache", (flaky(1), calls["n"]), ({0: "ok", 1: "int keys"}, 2))
df_mod.cache_set("k", {"a": 1})
check("expired entries are ignored", df_mod.cache_get("k", ttl=-1), None)
check("... unexpired are returned", df_mod.cache_get("k", ttl=60), {"a": 1})
df_mod.clear_cache()
check("clear_cache removes everything", df_mod.cache_get("k", ttl=60), None)

# --------------------------------------------------------------------------
section("Ticker handling / SEC fetch statuses")
check("BRK.B normalizes to BRK-B", df_mod.normalize_ticker(" brk.b "), "BRK-B")


class Resp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError("http")

    def json(self):
        return self._p


MAP = {"0": {"ticker": "AAPL", "cik_str": 320193}, "1": {"ticker": "BRK-B", "cik_str": 1067983}}
gets = {"map": 0, "facts": 0}
facts_status = {"code": 200}


def fake_sec_get(url, headers, timeout=30):
    if "company_tickers.json" in url:
        gets["map"] += 1
        return Resp(200, MAP)
    gets["facts"] += 1
    return Resp(facts_status["code"], {"facts": {}})


df_mod.sec_get = fake_sec_get
df_mod._cik_map_memo.update({"map": None, "at": 0.0})
check("CIK found for BRK.B written the Wikipedia way", df_mod.get_cik_for_ticker("BRK.B"), 1067983)
df_mod.get_cik_for_ticker("AAPL"); df_mod.get_cik_for_ticker("AAPL")
check("ticker map downloaded once for many lookups", gets["map"], 1)
check("unknown ticker -> 'not_covered'", df_mod.fetch_raw_companyfacts("ZZZZ")[1], "not_covered")
facts_status["code"] = 404
check("EDGAR 404 -> 'not_covered' (permanent)", df_mod.fetch_raw_companyfacts("AAPL")[1], "not_covered")
facts_status["code"] = 503
check("EDGAR 503 -> 'transient' (retry later)", df_mod.fetch_raw_companyfacts("AAPL")[1], "transient")
facts_status["code"] = 200
check("EDGAR 200 -> ok", df_mod.fetch_raw_companyfacts("AAPL")[1], "ok")

# --------------------------------------------------------------------------
section("Snapshots (build_snapshot) with faked SEC + Yahoo")
MOCK = build_mock_companyfacts()  # ~12% grower, EPS 6.0 in FY2024
PRICE_BUNDLE = {
    "closes": {"2024-09-30": 100.0, "2023-09-29": 90.0, "2021-09-30": 70.0, "2019-09-30": 50.0, "2014-09-30": 20.0},
    "splits": [], "last_close": 100.0, "last_date": "2024-10-01",
}
state = {"raw": (MOCK, "ok"), "price": (PRICE_BUNDLE, "ok"), "raw_calls": 0, "price_calls": 0, "latest_calls": 0, "latest": (111.0, "2024-10-02")}


def fake_raw(t):
    state["raw_calls"] += 1
    return state["raw"]


def fake_price(t):
    state["price_calls"] += 1
    return state["price"]


def fake_latest(t):
    state["latest_calls"] += 1
    return state["latest"]


valuation.fetch_raw_companyfacts = fake_raw
valuation.get_price_history = fake_price
valuation.get_latest_price = fake_latest

snap = valuation.build_snapshot("TEST", price_max_age=3600)
check("snapshot builds with no error", snap["error"], None)
check("price comes from the price history", snap["current_price"], 100.0)
check("EPS growth ~12%", snap["growth_rate"], 0.12, approx=0.01)
check("historical P/E is a real median of year-end price/EPS", snap["historical_pe"] is not None and snap["historical_pe"] > 0)
check("Quality Score computed", snap["quality_score"] > 0)
check("Moat Score computed (mock: debt/cash only at year 0, so ROIC has 1 horizon value; no gross/operating margin tags — only revenue stability scores)",
      snap["moat_score"], 10.0)

before = state["raw_calls"]
valuation.build_snapshot("TEST", price_max_age=3600)
check("second call served from cache (no refetch)", state["raw_calls"], before)

# stale price refresh
cached_snap = df_mod.cache_get("snapshot:TEST", 3600)
cached_snap["price_asof"] -= 7200
df_mod.cache_set("snapshot:TEST", cached_snap)
refreshed = valuation.build_snapshot("TEST", price_max_age=3600)
check("stale price is refreshed with the cheap latest-price call", (refreshed["current_price"], state["latest_calls"]), (111.0, 1))
check("...without refetching fundamentals", state["raw_calls"], before)
state["latest"] = None
cached_snap = df_mod.cache_get("snapshot:TEST", 3600); cached_snap["price_asof"] -= 99999; df_mod.cache_set("snapshot:TEST", cached_snap)
check("price refresh failure keeps the older price instead of erroring",
      valuation.build_snapshot("TEST", price_max_age=3600)["current_price"], 111.0)

# permanent vs transient
state["raw"] = (None, "not_covered")
s = valuation.build_snapshot("NOCOV")
check("not covered by EDGAR + Yahoo data -> fund snapshot, not an error",
      (bool(s["error"]), s.get("is_fund"), s["current_price"]), (False, True, 100.0))
n = state["raw_calls"]
valuation.build_snapshot("NOCOV")
check("fund snapshots are cached (no refetch)", state["raw_calls"], n)
state["price"] = (None, "empty")
s = valuation.build_snapshot("NOCOV2")
check("not covered by EDGAR + no Yahoo data -> permanent error", (bool(s["error"]), s["error_kind"]), (True, "permanent"))
state["price"] = (PRICE_BUNDLE, "ok")

state["raw"] = (None, "transient")
s = valuation.build_snapshot("FLAKY")
check("SEC outage -> transient error", s["error_kind"], "transient")
n = state["raw_calls"]
valuation.build_snapshot("FLAKY")
check("transient errors are NOT cached (retried)", state["raw_calls"], n + 1)

state["raw"] = (MOCK, "ok")
state["price"] = (None, "error")
check("Yahoo exception -> transient", valuation.build_snapshot("YFAIL")["error_kind"], "transient")
state["price"] = (None, "empty")
check("Yahoo has no data -> permanent", valuation.build_snapshot("YEMPTY")["error_kind"], "permanent")
state["price"] = (PRICE_BUNDLE, "ok")

# loss-maker + prune mode
LOSS = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
    _duration_fact("2023-10-01", "2024-09-30", -2.0, filed="2024-11-01"),
    _duration_fact("2022-10-01", "2023-09-30", -1.0, filed="2023-11-01")]}}}}}
state["raw"] = (LOSS, "ok")
pc = state["price_calls"]
s = valuation.build_snapshot("LOSER", prune_unvaluable=True)
check("scan mode prunes a loss-maker without calling Yahoo", (bool(s["error"]), state["price_calls"]), (True, pc))
s = valuation.build_snapshot("LOSER")
check("lookup mode still returns a full snapshot for a loss-maker", s["error"], None)
val = valuation.valuation_from_snapshot(s)
check("...with no Sticker Price and a plain-English note", (val["sticker_price"], "negative" in val["note"]), (None, True))
state["raw"] = ({"facts": {"us-gaap": {}}}, "ok")
check("company with no EPS facts -> permanent error", valuation.build_snapshot("NOEPS")["error_kind"], "permanent")
state["raw"] = (MOCK, "ok")

# --------------------------------------------------------------------------
section("valuation_from_snapshot")
base = {"current_price": 50.0, "current_eps": 5.0, "growth_rate": 0.12, "historical_pe": 18.0, "quality_score": 70.0}
v1 = valuation.valuation_from_snapshot(dict(base, current_price=50.0))
v2 = valuation.valuation_from_snapshot(dict(base, current_price=150.0))
check("Sticker Price does NOT move with today's price", v1["sticker_price"], v2["sticker_price"])
check("so % undervalued now differs between cheap and expensive", v1["undervalued_pct"] > 0 > v2["undervalued_pct"])
hot = dict(base, growth_rate=0.40)
check("growth capped at 15% by default", valuation.valuation_from_snapshot(hot)["growth_used"], 0.15)
check("cap is adjustable", valuation.valuation_from_snapshot(hot, max_growth_rate=0.25)["growth_used"], 0.25)
check("capped sticker equals a direct 15% calculation",
      valuation.valuation_from_snapshot(hot)["sticker_price"], calculate_sticker_price(5.0, 0.15, 18.0, 0.15), approx=1e-9)  # app default discount is 15% since c83e977, not 10%)
check("action UNDERVALUED_BUY when price <= MOS",
      valuation.valuation_from_snapshot(dict(base, current_price=1.0))["action"], "UNDERVALUED_BUY")
check("no growth history -> note", "history" in valuation.valuation_from_snapshot(dict(base, growth_rate=None))["note"])
check("flat/negative growth -> no sticker", valuation.valuation_from_snapshot(dict(base, growth_rate=-0.02))["sticker_price"], None)
check("missing historical P/E -> no sticker", valuation.valuation_from_snapshot(dict(base, historical_pe=None))["sticker_price"], None)

# --------------------------------------------------------------------------
section("Buy signals")
gk_ok = {"debt_test": {"status": "pass"}}


def mk(price, quality=70.0, sticker=100.0):
    snap_ = {"current_price": price, "quality_score": quality, "gatekeeper": gk_ok}
    val_ = {"sticker_price": sticker, "margin_of_safety_price": sticker / 2 if sticker else None, "note": "n/a"}
    return snap_, val_


check("BUY: at MOS + strong quality", signals.buy_signal(*mk(45.0))["signal"], "BUY")
check("BUY boundary: exactly at MOS", signals.buy_signal(*mk(50.0))["signal"], "BUY")
check("CHEAP, WEAK QUALITY: at MOS but quality below minimum", signals.buy_signal(*mk(45.0, quality=30.0))["signal"], "CHEAP, WEAK QUALITY")
check("NEAR BUY: 10% above MOS, good quality", signals.buy_signal(*mk(55.0))["signal"], "NEAR BUY")
check("WAIT: 40% above MOS", signals.buy_signal(*mk(70.0))["signal"], "WAIT")
check("WAIT (not NEAR BUY) when near but quality weak", signals.buy_signal(*mk(55.0, quality=30.0))["signal"], "WAIT")
check("NO VALUATION when there's no sticker", signals.buy_signal(*mk(55.0, sticker=None))["signal"], "NO VALUATION")
check("thresholds are adjustable", signals.buy_signal(*mk(45.0, quality=55.0), min_moat_score=50)["signal"], "BUY")
check("reasons always provided", len(signals.buy_signal(*mk(45.0))["reasons"]) >= 1)
# Pre-rename snapshots carry only "moat_score" (which WAS the quality
# score) — signals must still gate on it.
check("old snapshot shape (moat_score only) still gates on quality",
      signals.buy_signal({"current_price": 45.0, "moat_score": 30.0, "gatekeeper": gk_ok},
                         {"sticker_price": 100.0, "margin_of_safety_price": 50.0, "note": "n/a"})["signal"],
      "CHEAP, WEAK QUALITY")
check("new snapshot shape prefers quality_score over moat_score",
      signals.quality_score({"quality_score": 70.0, "moat_score": 20.0}), 70.0)
check("quality_score reads 0.0 for a fund snapshot (both None)", signals.quality_score({"quality_score": None, "moat_score": None}), 0.0)

# --------------------------------------------------------------------------
section("Sell signals")
check("SELL: price above Sticker", signals.sell_signal(*mk(120.0))["signal"], "SELL")
check("SELL boundary: exactly at Sticker", signals.sell_signal(*mk(100.0))["signal"], "SELL")
check("REVIEW: within 10% of Sticker", signals.sell_signal(*mk(95.0))["signal"], "REVIEW")
check("HOLD: well below Sticker with good quality", signals.sell_signal(*mk(60.0))["signal"], "HOLD")
check("REVIEW: weak quality even when cheap", signals.sell_signal(*mk(60.0, quality=20.0))["signal"], "REVIEW")
check("SELL outranks REVIEW (weak quality + above sticker)", signals.sell_signal(*mk(120.0, quality=20.0))["signal"], "SELL")
s_dbt = mk(60.0)[0]; s_dbt["gatekeeper"] = {"debt_test": {"status": "fail"}}
check("REVIEW: failed Debt Test", signals.sell_signal(s_dbt, mk(60.0)[1])["signal"], "REVIEW")
check("loss limit off by default", signals.sell_signal(*mk(60.0), avg_cost=100.0)["signal"], "HOLD")
check("REVIEW: down 40% vs cost with a 25% limit", signals.sell_signal(*mk(60.0), avg_cost=100.0, max_loss_pct=25)["signal"], "REVIEW")
check("loss limit ignores a missing cost basis", signals.sell_signal(*mk(60.0), avg_cost=None, max_loss_pct=25)["signal"], "HOLD")
check("REVIEW (not HOLD) when no valuation can be made", signals.sell_signal(*mk(60.0, sticker=None))["signal"], "REVIEW")

# --------------------------------------------------------------------------
section("Moat Score (competitive-moat footprint, from 10-K history)")


def _grow_series(base, rate=0.12):
    return {y: round(base / ((1 + rate) ** y), 2) for y in (0, 1, 3, 5, 10)}


_sales = _grow_series(10_000_000_000)
_perfect_hist = {
    "ROIC": {y: 0.30 for y in (0, 1, 3, 5, 10)},
    "Sales": _sales,
    "Gross_Profit": {y: round(_sales[y] * 0.60, 2) for y in _sales},
    "Operating_Income": {y: round(_sales[y] * 0.20, 2) for y in _sales},
}
check("perfect moat scores 100", gatekeeper._compute_moat_score(_perfect_hist)["score"], 100.0)
check("perfect moat has no missing components", gatekeeper._compute_moat_score(_perfect_hist)["missing"], [])
check("perfect moat components add up",
      gatekeeper._compute_moat_score(_perfect_hist)["components"],
      {"roic_level": 30.0, "roic_persistence": 25.0, "gross_margin": 25.0,
       "revenue_stability": 10.0, "operating_margin_stability": 10.0})
_weak_sales = {0: 10_000_000_000, 1: 13_000_000_000, 3: 8_000_000_000, 5: 12_000_000_000, 10: 9_000_000_000}
_weak_hist = {
    "ROIC": {y: 0.05 for y in (0, 1, 3, 5, 10)},
    "Sales": _weak_sales,
    "Gross_Profit": {y: round(_weak_sales[y] * 0.20, 2) for y in _weak_sales},
    "Operating_Income": {y: round(_weak_sales[y] * 0.05, 2) for y in _weak_sales},
}
check("weak moat scores low (constant low margins still earn stability points)",
      gatekeeper._compute_moat_score(_weak_hist)["score"], 33.2)
check("no data scores 0 and flags every component",
      gatekeeper._compute_moat_score({}),
      {"score": 0.0,
       "components": {"roic_level": 0.0, "roic_persistence": 0.0, "gross_margin": 0.0,
                      "revenue_stability": 0.0, "operating_margin_stability": 0.0},
       "missing": ["roic", "gross_margin", "revenue_stability", "operating_margin_stability"]})
check("thin data (2 ROIC points) scores only the computable component",
      gatekeeper._compute_moat_score({"ROIC": {0: 0.25, 1: 0.25}, "Sales": _sales})["score"], 10.0)
# The full Gatekeeper return carries both scores: quality (the renamed
# 80/10/10) and moat (the new footprint). ROIC is flat here so its Big
# Five CAGRs fail — quality 84, moat 100.
_full_hist = dict(_perfect_hist)
for _m in ("Equity_Per_Share", "EPS", "Free_Cash_Flow"):
    _full_hist[_m] = {y: 100.0 / ((1.12) ** y) for y in (0, 1, 3, 5, 10)}
_full_hist["Long_Term_Debt"] = {0: 0.0}
_full_hist["Owner_Earnings"] = {0: 1_000_000_000}
_full_hist["Cap_Ex"] = {0: 100_000_000}
_full_hist["Net_Income"] = {0: 1_000_000_000}
_gk = gatekeeper.evaluate_business_quality(_full_hist)
check("gatekeeper returns the renamed Quality Score", _gk["quality_score"], 84.0)
check("gatekeeper returns the new Moat Score", _gk["moat_score"], 100.0)
check("gatekeeper returns the moat detail", _gk["moat_detail"]["missing"], [])

# --------------------------------------------------------------------------
section("Storage")
p = os.path.join(TMP, "user_data.json")
d_missing = storage.load_user_data(p)
check("missing file -> empty portfolio", d_missing["portfolio"], [])
check("missing file -> empty watchlist", d_missing["watchlist"], [])
check("save portfolio", storage.save_portfolio([{"ticker": "AAPL", "shares": 10.0, "avg_cost": 150.0}, {"ticker": "KO", "shares": None, "avg_cost": None}], path=p))
check("save watchlist (keeps the portfolio)", storage.save_watchlist(["MCD", "SBUX"], path=p))
d = storage.load_user_data(p)
check("round trip portfolio", d["portfolio"][0], {"ticker": "AAPL", "shares": 10.0, "avg_cost": 150.0, "drip": False})
check("blank shares/cost stay None", d["portfolio"][1], {"ticker": "KO", "shares": None, "avg_cost": None, "drip": False})
check("round trip watchlist", d["watchlist"], ["MCD", "SBUX"])
with open(p, "w") as f:
    f.write("{not json")
d_corrupt = storage.load_user_data(p)
check("corrupt file doesn't crash; returns empty portfolio", d_corrupt["portfolio"], [])
check("corrupt file doesn't crash; returns empty watchlist", d_corrupt["watchlist"], [])
check("...and is moved aside, not deleted", os.path.exists(p + ".corrupt"))
with open(p, "w") as f:
    json.dump({"portfolio": [{"ticker": "brk.b", "shares": "abc", "avg_cost": float("nan")}, "junk", {"ticker": ""}], "watchlist": ["a", "A", "b.c", ""]}, f)
d = storage.load_user_data(p)
check("sanitizes junk rows / NaN / bad numbers", d["portfolio"], [{"ticker": "BRK-B", "shares": None, "avg_cost": None, "drip": False}])
check("watchlist deduped + normalized", d["watchlist"], ["A", "B-C"])
check("save to an unwritable path returns False, doesn't raise", storage.save_user_data({}, "/nonexistent-dir/x.json"), False)
check("save dividend goal", storage.save_dividend_goal(12000.0, path=p))
d = storage.load_user_data(p)
check("dividend goal round-trips", d.get("dividend_income_goal"), 12000.0)
check("negative dividend goal clamps to 0", storage.save_dividend_goal(-5.0, path=p) and storage.load_user_data(p).get("dividend_income_goal"), 0.0)
check("missing file -> dividend goal defaults to 0", storage.load_user_data(os.path.join(TMP, "nope.json")).get("dividend_income_goal"), 0.0)
check("save dividend watch", storage.save_dividend_watch({"KO": {"payout": 0.65, "date": "2026-11-05"}}, path=p))
d = storage.load_user_data(p)
check("dividend watch round-trips", d.get("dividend_watch"), {"KO": {"payout": 0.65, "date": "2026-11-05"}})
check("record goal month", storage.record_dividend_goal_month("2026-11", 5000.0, 12000.0, path=p))
check("record same month twice keeps one entry, latest wins",
      storage.record_dividend_goal_month("2026-11", 5200.0, 12000.0, path=p) and storage.load_user_data(p).get("dividend_goal_history"),
      [{"month": "2026-11", "income": 5200.0, "goal": 12000.0, "buys": 0.0, "raises": 0.0}])
_months = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01"]
for _m in _months:
    storage.record_dividend_goal_month(_m, 100.0, 12000.0, path=p)
d = storage.load_user_data(p)
check("goal history keeps only the last 24 months", len(d.get("dividend_goal_history")), 24)
check("goal history drops the oldest months", d["dividend_goal_history"][0]["month"], "2024-03")
check("save holdings baseline", storage.save_holdings_baseline({"KO": {"shares": 100.0, "dps": 2.0, "income": 200.0, "date": "2026-11-05"}}, path=p))
d = storage.load_user_data(p)
check("holdings baseline round-trips", d.get("holdings_baseline"), {"KO": {"shares": 100.0, "dps": 2.0, "income": 200.0, "date": "2026-11-05"}})
check("parse_date accepts MM-DD-YYYY, normalizes to ISO", storage._parse_date("01-15-2024"), "2024-01-15")
check("parse_date keeps YYYY-MM-DD", storage._parse_date("2024-01-15"), "2024-01-15")
check("parse_date rejects a bad month", storage._parse_date("13-01-2024"), None)
check("record goal month with attribution",
      storage.record_dividend_goal_month("2026-11", 5200.0, 12000.0, buys=300.0, raises=50.0, path=p)
      and storage.load_user_data(p).get("dividend_goal_history")[-1],
      {"month": "2026-11", "income": 5200.0, "goal": 12000.0, "buys": 300.0, "raises": 50.0})

# --------------------------------------------------------------------------
section("Sheets backend")


class _FakeSheet:
    """In-memory stand-in for a gspread worksheet."""
    def __init__(self):
        self.rows = []
        self.writes = 0

    def get_all_values(self):
        return [list(r) for r in self.rows]

    def clear(self):
        self.rows = []

    def update(self, _range, values, value_input_option="RAW"):
        self.rows = [list(r) for r in values]
        self.writes += 1


_fake = _FakeSheet()
_orig_config = storage._sheets_config
_orig_worksheet = storage._sheets_worksheet
_orig_data_file = storage.DATA_FILE
_sheets_tmp = os.path.join(TMP, "sheets_user_data.json")
storage.DATA_FILE = _sheets_tmp
storage._sheets_config = lambda: ("fake-sheet-id", {"type": "service_account"})
storage._sheets_worksheet = lambda _sid, _creds: _fake
try:
    check("sheets chunk boundary 40000 -> 1", len(storage._sheets_split("x" * 40000)), 1)
    check("sheets chunk boundary 40001 -> 2", len(storage._sheets_split("x" * 40001)), 2)
    _big = {"signal_log": [{"date": "2026-01-01", "ticker": "AAPL", "price": 1.0, "sticker": 2.0, "signal": "HOLD", "lean": ""} for _ in range(600)]}
    check("sheets write + read round-trips >50k chars",
          storage._sheets_write(_fake, _big) and storage._sheets_read(_fake)[0] == _big, True)
    check("sheets write split into 2+ chunks", len(_fake.rows) - 1 >= 2, True)
    _fake.rows = []
    check("sheets empty sheet reads as empty", storage._sheets_read(_fake)[2], "empty")
    _fake.rows = [[storage.SHEETS_MARKER, "2026-01-01T00:00:00+00:00", "1"], ["{bad json"]]
    check("sheets corrupt JSON reads as corrupt, never ok", storage._sheets_read(_fake)[2], "corrupt")
    # Migration: empty sheet + local data -> local uploads to the sheet.
    _fake.rows = []
    storage._reset_sheets_read_cache()
    with open(_sheets_tmp, "w") as f:
        json.dump({"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0}]}}, f)
    d = storage.load_user_data(_sheets_tmp)
    check("sheets migration returns the local data", d["portfolios"]["Main"][0]["ticker"], "SCHD")
    check("sheets migration uploads local to the sheet", storage._sheets_read(_fake)[2], "ok")
    # Sheet wins when local is missing.
    _fake.rows = []
    storage._sheets_write(_fake, {"portfolios": {"Main": [{"ticker": "VOO", "shares": 5.0, "avg_cost": 500.0, "drip": False}]}})
    if os.path.exists(_sheets_tmp):
        os.remove(_sheets_tmp)
    d = storage.load_user_data(_sheets_tmp)
    check("sheets data loads when the local file is gone", d["portfolios"]["Main"][0]["ticker"], "VOO")
    check("sheets load caches the sheet copy locally", os.path.exists(_sheets_tmp), True)
    # Local-newer heal: sheet old, local written after it -> local wins and heals the sheet.
    _fake.rows = []
    storage._sheets_write(_fake, {"portfolios": {"Main": [{"ticker": "KO", "shares": 1.0, "avg_cost": 60.0, "drip": False}]}})
    time.sleep(0.05)
    with open(_sheets_tmp, "w") as f:
        json.dump({"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0}]}}, f)
    d = storage.load_user_data(_sheets_tmp)
    check("local-newer load returns the local data", d["portfolios"]["Main"][0]["ticker"], "SCHD")
    check("local-newer load heals the sheet", storage._sheets_read(_fake)[0]["portfolios"]["Main"][0]["ticker"], "SCHD")
    # Outage: worksheet open raises -> load/save fall back to local, never raise.
    def _raising(_sid, _creds):
        raise RuntimeError("sheets down")
    storage._sheets_worksheet = _raising
    # The cache would mask the outage (a hit returns without opening),
    # so forget it — this load must actually hit the failing worksheet.
    storage._reset_sheets_read_cache()
    with open(_sheets_tmp, "w") as f:
        json.dump({"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0}]}}, f)
    d = storage.load_user_data(_sheets_tmp)
    check("sheets outage load falls back to local", d["portfolios"]["Main"][0]["ticker"], "SCHD")
    check("sheets outage save still lands locally",
          storage.save_user_data({"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0, "drip": False}]}}, _sheets_tmp)
          and storage.load_user_data(_sheets_tmp)["portfolios"]["Main"][0]["ticker"], "SCHD")
    storage._sheets_worksheet = lambda _sid, _creds: _fake
    # A non-default path never touches the sheet, even when configured.
    _writes_before = _fake.writes
    _other = os.path.join(TMP, "other_user_data.json")
    storage.save_user_data({"portfolios": {"Main": [{"ticker": "AAPL", "shares": 1.0, "avg_cost": 1.0, "drip": False}]}}, _other)
    storage.load_user_data(_other)
    check("non-default path never touches the sheet", _fake.writes, _writes_before)
    # Read cache: one sheet read serves repeated loads for a minute, so
    # dozens of load_user_data() calls per rerun cost one API read.
    _open_count = {"n": 0}
    def _counting(_sid, _creds):
        _open_count["n"] += 1
        return _fake
    storage._sheets_worksheet = _counting
    storage._sheets_write(_fake, {"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0, "drip": False}]}})
    if os.path.exists(_sheets_tmp):
        os.remove(_sheets_tmp)
    storage._reset_sheets_read_cache()
    storage.load_user_data(_sheets_tmp)
    storage.load_user_data(_sheets_tmp)
    storage.load_user_data(_sheets_tmp)
    check("sheets read cache: repeated loads make one API read", _open_count["n"], 1)
    # A save warms the cache: the next load needs no API read either.
    storage._reset_sheets_read_cache()
    storage.save_user_data({"portfolios": {"Main": [{"ticker": "VOO", "shares": 5.0, "avg_cost": 500.0, "drip": False}]}}, _sheets_tmp)
    _open_count["n"] = 0
    d = storage.load_user_data(_sheets_tmp)
    check("sheets save warms the read cache", (_open_count["n"], d["portfolios"]["Main"][0]["ticker"]), (0, "VOO"))
    # The cache expires: with a 0.05s TTL the next load re-reads.
    _open_count["n"] = 0
    _orig_ttl = storage._SHEETS_READ_TTL_SECONDS
    storage._SHEETS_READ_TTL_SECONDS = 0.05
    time.sleep(0.1)
    storage.load_user_data(_sheets_tmp)
    check("sheets read cache expires after its TTL", _open_count["n"], 1)
    storage._SHEETS_READ_TTL_SECONDS = _orig_ttl
    storage._sheets_worksheet = lambda _sid, _creds: _fake
    # Wipe guard (Nov 18): a save that looks wiped never overwrites a
    # Sheet or local file that still holds the user's data.
    _full = {"portfolios": {"Main": [{"ticker": "SCHD", "shares": 10.0, "avg_cost": 80.0, "drip": False}]},
             "watchlists": {"Main": ["AAPL"]},
             "notes": {"SCHD": {"meaning": "div"}},
             "manual": {},
             "sales_log": [],
             "signal_log": [{"date": "2026-11-18", "ticker": "SCHD", "price": 80.0, "sticker": 100.0, "signal": "HOLD", "lean": ""}],
             "dividend_income_goal": 100000.0,
             "holdings_baseline": {"SCHD": {"shares": 10.0, "dps": 2.5, "income": 25.0, "date": "2026-11-18"}},
             "dividend_goal_history": [{"month": "2026-11", "income": 25.0, "goal": 100000.0, "buys": 0.0, "raises": 0.0}]}
    # The Nov 18 wiped shape: everything a wipe loses is empty, but the
    # goal, baseline and history survive — so they must NOT save the write.
    _wiped = dict(_full)
    _wiped.update({"portfolios": {"Main": []}, "watchlists": {"Main": []}, "notes": {}, "signal_log": []})
    check("wiped shape is detected", storage._looks_wiped(_wiped), True)
    check("full shape is not wiped", storage._looks_wiped(_full), False)
    check("delete-all-holdings (signal log kept) is not wiped",
          storage._looks_wiped(dict(_full, portfolios={"Main": []})), False)
    # Sheet + local both hold content -> the wiped save is refused, and
    # neither store changes.
    storage._sheets_write(_fake, _full)
    with open(_sheets_tmp, "w") as f:
        json.dump(_full, f)
    storage._reset_sheets_read_cache()
    _writes_before = _fake.writes
    check("wiped save is refused when the Sheet holds data", storage.save_user_data(_wiped, _sheets_tmp), False)
    check("refused save does not touch the Sheet", _fake.writes, _writes_before)
    check("refused save leaves the Sheet data intact",
          storage._sheets_read(_fake)[0]["portfolios"]["Main"][0]["ticker"], "SCHD")
    check("refused save leaves the local file intact",
          storage.load_user_data(_sheets_tmp)["portfolios"]["Main"][0]["ticker"], "SCHD")
    # Nothing to protect (sheet + local both empty) -> the save is allowed.
    _fake.rows = []
    storage._reset_sheets_read_cache()
    if os.path.exists(_sheets_tmp):
        os.remove(_sheets_tmp)
    check("wiped save is allowed when nothing is stored", storage.save_user_data(storage._empty(), _sheets_tmp), True)
    # Heal guard: a wiped local file newer than the Sheet never heals it.
    storage._sheets_write(_fake, _full)
    time.sleep(0.05)
    with open(_sheets_tmp, "w") as f:
        json.dump(_wiped, f)
    storage._reset_sheets_read_cache()
    d = storage.load_user_data(_sheets_tmp)
    check("wiped local file does not heal (overwrite) the Sheet",
          d["portfolios"]["Main"][0]["ticker"], "SCHD")
    check("load repairs the wiped local file from the Sheet",
          json.load(open(_sheets_tmp))["portfolios"]["Main"][0]["ticker"], "SCHD")
finally:
    storage._sheets_config = _orig_config
    storage._sheets_worksheet = _orig_worksheet
    storage.DATA_FILE = _orig_data_file
    storage._reset_sheets_config_cache()
    storage._reset_sheets_read_cache()


# --------------------------------------------------------------------------
section("Scanner")


def good(t, price=50.0, streak=0, quality=70.0, eps=5.0):
    return {"ticker": t, "name": t + " Inc", "error": None, "error_kind": None, "current_price": price, "current_eps": eps,
            "growth_rate": 0.12, "historical_pe": 18.0, "quality_score": quality, "dividend_streak": streak, "gatekeeper": gk_ok}


fake_db = {f"T{i}": good(f"T{i}", price=10 + i) for i in range(20)}
fake_db["PERM"] = {"ticker": "PERM", "error": "No SEC filings found", "error_kind": "permanent"}
counter = {"n": 0}
lock = threading.Lock()


def fake_build(t, price_max_age=0, prune_unvaluable=False):
    with lock:
        counter["n"] += 1
    return fake_db[t]


screener.build_snapshot = fake_build
res = screener.scan_snapshots(list(fake_db) + ["T3", " t4 "], workers=4)
check("scan dedupes and normalizes tickers", res["total"], 21)
check("valid snapshots collected", len(res["snapshots"]), 20)
check("permanent failures recorded with their reason", res["errors"], {"PERM": "No SEC filings found"})
check("not aborted", res["aborted"], None)
seen = []
screener.scan_snapshots(["T1", "T2", "T3"], workers=2, progress_callback=lambda d, t, tk: seen.append((d, t)))
check("progress callback reports done/total", sorted(seen), [(1, 3), (2, 3), (3, 3)])
check("a crashing progress callback doesn't kill the scan",
      screener.scan_snapshots(["T1"], progress_callback=lambda *a: 1 / 0)["done"], 1)

# throttling: everything transient -> abort early instead of hammering
call_log = []


def always_transient(t, price_max_age=0, prune_unvaluable=False):
    call_log.append(t)
    time.sleep(0.01)
    return {"ticker": t, "error": "Couldn't reach Yahoo", "error_kind": "transient"}


screener.build_snapshot = always_transient
res = screener.scan_snapshots([f"X{i}" for i in range(400)], workers=2, max_consecutive_transient=6, retry_delay=0)
check("scan aborts early when SEC/Yahoo keeps failing", res["aborted"] is not None)
check("...and didn't grind through all 400 tickers", len(call_log) < 100)

# transient is retried once and recovers
tries = {}


def recovers(t, price_max_age=0, prune_unvaluable=False):
    tries[t] = tries.get(t, 0) + 1
    return {"ticker": t, "error": "x", "error_kind": "transient"} if tries[t] == 1 else good(t)


screener.build_snapshot = recovers
res = screener.scan_snapshots(["A1", "A2"], retry_delay=0)
check("a transient failure is retried once and recovers", (len(res["snapshots"]), res["errors"]), (2, {}))

# ranking
snaps = [good("CHEAP", price=20.0, streak=10), good("MID", price=60.0, streak=2), good("PRICEY", price=500.0, streak=30),
         good("NODIV", price=10.0, streak=0, quality=20.0), dict(good("LOSS"), current_eps=-1.0)]
r = screener.rank_snapshots(snaps)
order = [x["ticker"] for x in r["top_results"]]
check("ranked most-undervalued first", order, ["NODIV", "CHEAP", "MID", "PRICEY"])
check("loss-maker excluded (no valuation)", ("LOSS" in order, r["valued_count"]), (False, 4))
r = screener.rank_snapshots(snaps, min_dividend_years=5)
check("dividend filter keeps only >=5-year streaks", [x["ticker"] for x in r["top_results"]], ["CHEAP", "PRICEY"])
r = screener.rank_snapshots(snaps, min_dividend_years=11)
check("a stricter slider value keeps fewer", [x["ticker"] for x in r["top_results"]], ["PRICEY"])
r = screener.rank_snapshots(snaps, min_moat_score=50)
check("quality filter drops weak-quality names", "NODIV" in [x["ticker"] for x in r["top_results"]], False)
many = [good(f"M{i}", price=10 + i) for i in range(60)]
check("top_n limits to 25 but counts all matches", (len(screener.rank_snapshots(many)["top_results"]), screener.rank_snapshots(many)["matched_count"]), (25, 60))
cheap_pct = screener.rank_snapshots(snaps, discount_rate=0.10)["top_results"][0]["undervalued_pct"]
check("changing discount rate re-ranks instantly from the same snapshots",
      screener.rank_snapshots(snaps, discount_rate=0.15)["top_results"][0]["undervalued_pct"] != cheap_pct)

# --------------------------------------------------------------------------
section("Universe")
EXCH = {"fields": ["cik", "name", "ticker", "exchange"],
        "data": [[i, f"Co{i}", f"TK{i}", "Nasdaq" if i % 2 else "NYSE"] for i in range(1500)]
        + [[9001, "Pink", "PINK", "OTC"], [9002, "Brk", "BRK.B", "NYSE"], [9003, "Cboe", "CBOE1", "CBOE"]]}


def fake_sec_get_exch(url, headers, timeout=30):
    if "exchange" in url:
        return Resp(200, EXCH)
    return Resp(200, MAP)


universe.sec_get = fake_sec_get_exch
df_mod.clear_cache()
allt = universe.get_all_us_listed_tickers()
check("NYSE + Nasdaq only (OTC and CBOE excluded)", ("PINK" in allt, "CBOE1" in allt, "TK1" in allt, "TK2" in allt), (False, False, True, True))
check("BRK.B normalized to BRK-B", "BRK-B" in allt)
check("returns the full ~1500 (not capped at the S&P 500)", len(allt), 1501)


def broken_sec_get(url, headers, timeout=30):
    raise RuntimeError("network down")


universe.sec_get = broken_sec_get
df_mod.clear_cache()
df_mod._cik_map_memo["map"] = {"AAPL": 1, "MSFT": 2}
check("exchange file unavailable -> falls back to SEC's ticker map", universe.get_all_us_listed_tickers(), ["AAPL", "MSFT"])
df_mod._cik_map_memo.update({"map": None, "at": 0.0})
df_mod.sec_get = broken_sec_get
df_mod.clear_cache()
check("everything unavailable -> small fallback list, no crash", len(universe.get_all_us_listed_tickers()) >= 50)

# --------------------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print("  -", f)
else:
    print("ALL TESTS PASSED")
print("=" * 60)
raise SystemExit(1 if failures else 0)
