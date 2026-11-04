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
- Two names the user asked about are NOT 13F filers and are listed
  separately in NON_13F, never as holdings:
  Phil Town (not a fund manager, no 13F) and Nomad Investment
  Partnership (wound down in 2014, no current 13F).
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
]

NON_13F: List[Dict[str, str]] = [
    # Correction 2026-11-04, after the user pointed to Valuesider:
    # Phil Town IS a portfolio manager — of the Rule One Fund (RULRX),
    # via Rule One Partners LLC, since 2019 (SEC prospectus, World Funds
    # Trust, CIK 1396092). That fund files N-PORT (mutual-fund holdings),
    # not 13F-HR, so it is a separate source type, not a 13F row.
    # Valuesider Q2 2026: $188.5M, 40 holdings, filed 25 Aug 2026;
    # top stock holdings NFLX, CNSWF, SFM, CRM (plus GOIXX, a
    # government money-market fund, at 33.35% — cash-like, not a pick).
    {"person": "Phil Town", "note": "Not a 13F filer — portfolio manager of the Rule One Fund (RULRX) since 2019, which files N-PORT holdings instead. See the N-PORT eye-check, not the 13F table."},
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

    Never returns a bare 'not held' for a manager whose filing could not
    be fetched — that row carries an error instead.
    """
    target = normalize_issuer_name(company_name or ticker)
    rows: List[Dict[str, Any]] = []
    for mgr in MANAGERS:
        data = fetch_latest_13f(int(mgr["cik"]))
        row: Dict[str, Any] = {
            "person": mgr["person"], "manager": mgr["manager"],
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
    return rows
