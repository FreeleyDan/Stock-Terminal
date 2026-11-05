"""Value-investor 13F holdings — eye check only.

Reads the latest Form 13F-HR for a small list of value managers directly
from SEC EDGAR (no paid API, no third-party scraper). A holding here is
never a buy signal and never changes a Sticker, MOS, Top 25 rank, or
portfolio signal in this app.

Limits, stated in the UI wherever these numbers are shown:
- 13F is quarterly and due up to 45 days after quarter-end, so the
  latest filing can be ~4 months old. Trades since quarter-end are
  invisible. Shorts, cash, bonds, and non-US positions are excluded.
- Matching is by issuer NAME, not CUSIP, because SEC companyfacts does
  not give a ticker->CUSIP map. Class shares (e.g. GOOG vs GOOGL, or
  BRK-A vs BRK-B) share an issuer name and are flagged as ambiguous
  instead of guessed. A "not found" means "not found by name in the
  latest 13F", not a proof the manager owns nothing in the company.
- Phil Town is NOT a 13F filer — he manages the Rule One Fund (RULRX),
  a mutual fund that files N-PORT. That row is produced by
  fetch_latest_nport / check_nport below and is included in the SAME
  table as the 13F managers (same check_holdings call, same button),
  marked with source "N-PORT" so the filing type is never hidden.
- Nomad Investment Partnership (wound down in 2014) has no current
  filing and is listed in NON_13F, never as a holding.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

try:
    import streamlit as st
    _cache = st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
except Exception:  # pragma: no cover - plain-python test import
    def _cache(func):  # type: ignore
        return func

SEC_USER_AGENT = "BusinessQualityTerminal Jakob Freele jakobfreele@gmail.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Filing entities verified against SEC EDGAR / SEC filing URLs in research
# 2026-11-04. Berkshire and Dalal Street CIKs appear in SEC Archives URLs;
# Himalaya and Aquamarine Zurich CIKs appear in multiple 13F trackers.
# If a submissions fetch fails, that manager shows an error, never a
# fabricated "not held".
MANAGERS: List[Dict[str, Any]] = [
    {"person": "Warren Buffett", "manager": "Berkshire Hathaway Inc", "cik": 1067983},
    {"person": "Li Lu", "manager": "Himalaya Capital Management LLC", "cik": 1709323},
    {"person": "Mohnish Pabrai", "manager": "Dalal Street, LLC", "cik": 1549575},
    {"person": "Guy Spier", "manager": "Aquamarine Zurich AG", "cik": 1953324},
    # Added 2026-11-04 at the user's request. CIKs verified against SEC
    # submissions (data.sec.gov/submissions/CIK....json) the same day:
    # each name below matches the filer and each has a 13F-HR for report
    # 2026-06-30. Note the accession prefix can differ from the CIK
    # (e.g. Weitz 0001821268-26-000133, Southeastern 0001398344-26-014641)
    # — the filing directory still uses the CIK, and fetch_latest_13f
    # reads the accession from submissions, so it fetches correctly.
    {"person": "Bill Nygren", "manager": "Harris Associates L.P.", "cik": 813917},
    {"person": "Christopher Davis", "manager": "Davis Selected Advisers", "cik": 1036325},
    {"person": "Francois Rochon", "manager": "Giverny Capital Inc.", "cik": 1641864},
    {"person": "Wallace Weitz", "manager": "Weitz Investment Management, Inc.", "cik": 883965},
    {"person": "Mason Hawkins", "manager": "Southeastern Asset Management Inc.", "cik": 807985},
]

NON_13F: List[Dict[str, str]] = [
    # Phil Town used to live here — he now has a row in the main table
    # (source "N-PORT", Rule One Fund, World Funds Trust CIK 1396092,
    # series S000065131; manager since 2019). Only Nomad remains here.
    {"person": "Nomad Investment Partnership", "note": "Closed — wound down in 2014 and capital returned, so there is no current 13F to check."},
]

_SUFFIXES = {
    "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LTD", "LIMITED",
    "PLC", "LP", "LLP", "DEL", "DE", "NEW", "NV", "SA", "AG", "GROUP",
}


def normalize_issuer_name(name: str) -> str:
    """'Berkshire Hathaway Inc Del' -> 'BERKSHIRE HATHAWAY'."""
    if not name:
        return ""
    text = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    words = [w for w in text.split() if w not in _SUFFIXES]
    return " ".join(words).strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_infotable_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Parse a 13F informationTable XML document into holdings."""
    root = ET.fromstring(xml_text)
    holdings: List[Dict[str, Any]] = []
    for node in root.iter():
        if _local(node.tag) != "infoTable":
            continue
        fields: Dict[str, str] = {}
        for child in node.iter():
            key = _local(child.tag)
            if key in ("nameOfIssuer", "cusip", "value", "sshPrnamt", "putCall") and child.text:
                fields[key] = child.text.strip()
        try:
            # Verified against Dalal Street Q2 2026: XML value is already
            # dollars (517,194 AMR shares x ~$165 = $85,305,978, exactly the
            # XML value). Do NOT multiply by 1,000 — that gave $85B.
            value = int(float(fields.get("value", "0")))
        except Exception:
            value = 0
        try:
            shares = int(float(fields.get("sshPrnamt", "0")))
        except Exception:
            shares = 0
        holdings.append({
            "issuer": fields.get("nameOfIssuer", ""),
            "cusip": fields.get("cusip", ""),
            "value": value,
            "shares": shares,
            "put_call": fields.get("putCall", ""),
        })
    return holdings


def _submissions(cik: int) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            headers=SEC_HEADERS, timeout=30,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _latest_13f_filing(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    recent = sub.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form") or []
    best = None
    for i, form in enumerate(forms):
        if form not in ("13F-HR", "13F-HR/A"):
            continue
        try:
            report = recent["reportDate"][i]
            filed = recent["filingDate"][i]
            accession = recent["accessionNumber"][i]
            primary = recent["primaryDocument"][i]
        except Exception:
            continue
        # Prefer the latest report date, then the latest filing date;
        # an amendment (HR/A) for the same report date wins on filing date.
        key = (report or "", filed or "", 1 if form == "13F-HR/A" else 0)
        if best is None or key > best[0]:
            best = (key, {"accession": accession, "filing_date": filed,
                          "report_date": report, "primary_document": primary, "form": form})
    return best[1] if best else None


@_cache
def fetch_latest_13f(cik: int) -> Dict[str, Any]:
    """Latest 13F holdings for one manager, or {'error': ...}."""
    sub = _submissions(cik)
    if not sub:
        return {"error": "Couldn't reach SEC EDGAR for this manager"}
    filing = _latest_13f_filing(sub)
    if not filing:
        return {"error": "No 13F-HR filing found for this manager"}
    accession_nodash = filing["accession"].replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}"
    # Information-table filename varies by filer: Dalal Street uses
    # infotable.xml, Berkshire uses a numeric file (56757.xml in Q2 2026).
    # So list the filing directory and try each XML file (other than the
    # cover primary_doc) until one parses into holdings.
    candidates: List[str] = []
    try:
        idx = requests.get(f"{base}/index.json", headers=SEC_HEADERS, timeout=30)
        if idx.status_code == 200:
            for item in (idx.json().get("directory", {}).get("item") or []):
                name = item.get("name", "")
                if name.lower().endswith(".xml") and "index" not in name.lower():
                    candidates.append(name)
            # Prefer a file literally named infotable.xml, then the rest.
            candidates.sort(key=lambda n: (0 if n.lower() == "infotable.xml" else 1, n))
    except Exception:
        candidates = []
    if filing.get("primary_document"):
        candidates.append(filing["primary_document"])
    if "infotable.xml" not in candidates:
        candidates.append("infotable.xml")
    holdings: List[Dict[str, Any]] = []
    last_error = "Couldn't download the 13F information table"
    for name in candidates:
        url = name if name.startswith("http") else f"{base}/{name}"
        # primary_document may be a path like xslForm13F_X02/primary_doc.xml
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if resp.status_code != 200:
                continue
            parsed = parse_infotable_xml(resp.text)
            if parsed:
                holdings = parsed
                break
        except Exception:
            continue
    if not holdings:
        return {"error": last_error, **filing}
    # Unit heuristic: some filers report XML value in dollars (Berkshire,
    # Dalal Street, Himalaya — verified: shares x price = XML value),
    # others in thousands as the form specifies (Aquamarine: BRK-B
    # 97,600 shares, XML value 48,838 — $0.50/share in dollars, $500 in
    # thousands). If the median implied price is under $10, the filing
    # is in thousands, so scale all values by 1,000. This is a filing-level
    # heuristic, not a per-holding guess.
    import statistics as _statistics
    implied = [h["value"] / h["shares"] for h in holdings if h["shares"] > 0 and h["value"] > 0]
    if implied and _statistics.median(implied) < 10:
        for h in holdings:
            h["value"] *= 1000
    total = sum(h["value"] for h in holdings)
    for h in holdings:
        h["pct_portfolio"] = (h["value"] / total) if total else None
    return {"holdings": holdings, "total_value": total, **filing}


def check_holdings(ticker: str, company_name: Optional[str]) -> List[Dict[str, Any]]:
    """One row per manager: holds? (by issuer name), matches, or error.

    This is the single entry point the UI calls — one button, one table.
    It covers the 13F managers in MANAGERS plus Phil Town's Rule One
    Fund, whose row comes from check_nport and is marked source "N-PORT"
    (a mutual-fund filing, not a 13F) instead of living in a separate
    expander. Never returns a bare 'not held' for a manager whose filing
    could not be fetched — that row carries an error instead.
    """
    target = normalize_issuer_name(company_name or ticker)
    rows: List[Dict[str, Any]] = []
    for mgr in MANAGERS:
        data = fetch_latest_13f(int(mgr["cik"]))
        row: Dict[str, Any] = {
            "person": mgr["person"], "manager": mgr["manager"], "source": "13F",
            "report_date": data.get("report_date"), "filing_date": data.get("filing_date"),
            "holds": False, "matches": [], "ambiguous": False, "error": data.get("error"),
        }
        if data.get("error") or "holdings" not in data:
            rows.append(row)
            continue
        matches_by_cusip: Dict[str, Dict[str, Any]] = {}
        for h in data["holdings"]:
            issuer_norm = normalize_issuer_name(h["issuer"])
            if not target or not issuer_norm:
                continue
            if issuer_norm == target or issuer_norm.startswith(target) or target.startswith(issuer_norm):
                key = f"{h['cusip']}|{h['put_call']}"
                if key in matches_by_cusip:
                    agg = matches_by_cusip[key]
                    agg["shares"] += h["shares"]
                    agg["value"] += h["value"]
                    agg["pct_portfolio"] = (agg["pct_portfolio"] or 0) + (h["pct_portfolio"] or 0)
                else:
                    matches_by_cusip[key] = dict(h)
        # Berkshire's 13F lists the same issuer many times (other
        # managers); aggregating by CUSIP gives one row per class.
        matches = list(matches_by_cusip.values())
        row["matches"] = matches
        row["holds"] = bool(matches)
        # More than one CUSIP under the same issuer name = class shares:
        # do not guess which class the ticker is.
        row["ambiguous"] = len({m["cusip"] for m in matches}) > 1
        rows.append(row)
    # Phil Town in the same table: same shape as a 13F row, source
    # "N-PORT" so the UI can label the filing type.
    nport_row = check_nport(ticker, company_name)
    nport_row["source"] = "N-PORT"
    rows.append(nport_row)
    return rows


# ---------------------------------------------------------------------------
# N-PORT — Rule One Fund (Phil Town), a mutual fund, not a 13F filer.
# Verified 2026-11-04: World Funds Trust CIK 1396092, series
# S000065131 "Rule One Fund", latest public NPORT-P at the time was
# report 2026-06-30, filed 2026-08-25, accession 0001396092-26-000449,
# net assets $189,470,810.80, 41 invstOrSec entries (12 common-stock
# EC, plus derivatives DE and a money-market STIV).
# Only assetCat == "EC" (common stock) entries count as holdings here:
# derivatives on the same issuer (e.g. CALL/PUT Netflix) and the
# Federated money-market position are excluded, and a "Default" name
# falls back to the entry's title, which is the real issuer name.
# The raw XML is the filing-directory primary_doc.xml — NOT the
# xslFormNPORT-P_X01/primary_doc.xml path, which SEC serves as a
# rendered HTML view with no usable holding tags.
RULE_ONE_CIK = 1396092
RULE_ONE_SERIES_ID = "S000065131"


def parse_nport_xml(xml_text: str) -> Dict[str, Any]:
    """Parse a raw NPORT-P XML document into its fund + EC holdings."""
    root = ET.fromstring(xml_text)
    series_name = series_id = report_date = ""
    net_assets: Optional[float] = None
    holdings: List[Dict[str, Any]] = []
    for el in root.iter():
        key = _local(el.tag)
        if key == "seriesName" and el.text:
            series_name = el.text.strip()
        elif key == "seriesId" and el.text and not series_id:
            series_id = el.text.strip()
        elif key == "repPdDate" and el.text:
            report_date = el.text.strip()
        elif key == "netAssets" and el.text:
            try:
                net_assets = float(el.text)
            except Exception:
                net_assets = None
    for node in root.iter():
        if _local(node.tag) != "invstOrSec":
            continue
        fields: Dict[str, str] = {}
        for child in node:
            key = _local(child.tag)
            if key in ("name", "title", "cusip", "balance", "valUSD", "pctVal", "assetCat") and child.text:
                fields[key] = child.text.strip()
        if fields.get("assetCat") != "EC":
            continue
        name = fields.get("name", "") or ""
        if name == "Default":
            name = fields.get("title", "") or ""
        try:
            shares = float(fields.get("balance", "0"))
        except Exception:
            shares = 0.0
        try:
            value = float(fields.get("valUSD", "0"))
        except Exception:
            value = 0.0
        try:
            pct = float(fields.get("pctVal", "0")) / 100.0
        except Exception:
            pct = None
        if not name or fields.get("cusip") in ("", "N/A"):
            continue
        holdings.append({
            "issuer": name, "title": fields.get("title", ""),
            "cusip": fields.get("cusip", ""), "shares": shares,
            "value": value, "pct_portfolio": pct,
        })
    return {
        "series_name": series_name, "series_id": series_id,
        "report_date": report_date, "net_assets": net_assets,
        "holdings": holdings,
    }


@_cache
def fetch_latest_nport(cik: int = RULE_ONE_CIK, series_id: str = RULE_ONE_SERIES_ID) -> Dict[str, Any]:
    """Latest public NPORT-P for one series, or {'error': ...}.

    Walks the trust's recent NPORT-P filings newest-first and returns
    the first whose parsed seriesId matches. Each candidate is one
    small XML fetch; the whole walk is cached for 24h like the 13Fs.
    """
    sub = _submissions(cik)
    if not sub:
        return {"error": "Couldn't reach SEC EDGAR for the Rule One Fund"}
    recent = sub.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form") or []
    for i, form in enumerate(forms):
        if form not in ("NPORT-P", "NPORT-P/A"):
            continue
        try:
            accession = recent["accessionNumber"][i]
            filed = recent["filingDate"][i]
            report = recent["reportDate"][i]
        except Exception:
            continue
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        try:
            resp = requests.get(f"{base}/primary_doc.xml", headers=SEC_HEADERS, timeout=30)
            if resp.status_code != 200:
                continue
            parsed = parse_nport_xml(resp.text)
        except Exception:
            continue
        if parsed.get("series_id") != series_id:
            continue
        return {
            "accession": accession, "filing_date": filed, "report_date": report,
            "form": form, **parsed,
        }
    return {"error": "No public N-PORT filing found for the Rule One Fund in the trust's recent filings"}


def check_nport(ticker: str, company_name: Optional[str]) -> Dict[str, Any]:
    """Does the Rule One Fund hold this ticker, by issuer name?

    Helper used by check_holdings to build the Phil Town row in the
    same table as the 13F managers — not a separate UI function.
    Returns the same shape as one 13F row, plus the fund's net assets
    so the UI can show the position as a % of the fund.
    """
    target = normalize_issuer_name(company_name or ticker)
    data = fetch_latest_nport()
    row: Dict[str, Any] = {
        "person": "Phil Town", "manager": "Rule One Fund (RULRX)",
        "report_date": data.get("report_date"), "filing_date": data.get("filing_date"),
        "holds": False, "matches": [], "ambiguous": False, "error": data.get("error"),
        "net_assets": data.get("net_assets"), "series_name": data.get("series_name"),
    }
    if data.get("error") or "holdings" not in data:
        return row
    matches_by_cusip: Dict[str, Dict[str, Any]] = {}
    for h in data["holdings"]:
        issuer_norm = normalize_issuer_name(h["issuer"])
        if not target or not issuer_norm:
            continue
        if issuer_norm == target or issuer_norm.startswith(target) or target.startswith(issuer_norm):
            key = h["cusip"]
            if key in matches_by_cusip:
                agg = matches_by_cusip[key]
                agg["shares"] += h["shares"]
                agg["value"] += h["value"]
                agg["pct_portfolio"] = (agg["pct_portfolio"] or 0) + (h["pct_portfolio"] or 0)
            else:
                matches_by_cusip[key] = dict(h)
    matches = list(matches_by_cusip.values())
    row["matches"] = matches
    row["holds"] = bool(matches)
    row["ambiguous"] = len({m["cusip"] for m in matches}) > 1
    return row
