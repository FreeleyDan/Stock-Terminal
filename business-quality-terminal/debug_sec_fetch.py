"""
debug_sec_fetch.py

Run this standalone (same folder as data_fetcher.py): python3 debug_sec_fetch.py

Hits the two SEC endpoints directly with NOTHING caught/swallowed, so
whatever is actually going wrong (403, timeout, SSL error, JSON shape
mismatch, etc.) prints plainly instead of disappearing into app.py's
"No EDGAR data found" message. Paste the full output back for diagnosis.
"""

import sys

import requests

from data_fetcher import SEC_HEADERS, SEC_TICKER_HEADERS, SEC_USER_AGENT, TICKER_MAP_URL, COMPANYFACTS_URL

print("=" * 70)
print(f"User-Agent being sent: {SEC_USER_AGENT!r}")
print("=" * 70)

# --- Step 1: ticker -> CIK lookup -------------------------------------
print("\n[1] Fetching ticker->CIK map:", TICKER_MAP_URL)
try:
    resp = requests.get(TICKER_MAP_URL, headers=SEC_TICKER_HEADERS, timeout=15)
    print(f"    status_code: {resp.status_code}")
    print(f"    content-length: {len(resp.content)}")
    if resp.status_code != 200:
        print(f"    body (first 500 chars): {resp.text[:500]!r}")
        sys.exit(1)
    data = resp.json()
    print(f"    entries in map: {len(data)}")
    cik = None
    for _, entry in data.items():
        if entry.get("ticker", "").upper() == "AAPL":
            cik = int(entry["cik_str"])
            break
    print(f"    AAPL CIK resolved to: {cik}")
    if cik is None:
        print("    !! AAPL not found in the ticker map -- unexpected, map shape may have changed.")
        sys.exit(1)
except Exception as e:
    print(f"    !! EXCEPTION during ticker map fetch: {type(e).__name__}: {e}")
    sys.exit(1)

# --- Step 2: companyfacts fetch for that CIK ---------------------------
url = COMPANYFACTS_URL.format(cik=cik)
print(f"\n[2] Fetching companyfacts: {url}")
try:
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    print(f"    status_code: {resp.status_code}")
    print(f"    content-length: {len(resp.content)}")
    for k, v in resp.headers.items():
        if k.lower() in ("x-deny-reason", "content-type", "www-authenticate"):
            print(f"    header {k}: {v}")
    if resp.status_code != 200:
        print(f"    body (first 1000 chars): {resp.text[:1000]!r}")
        sys.exit(1)
    payload = resp.json()
    print(f"    entityName: {payload.get('entityName')}")
    print(f"    taxonomy namespaces present: {list(payload.get('facts', {}).keys())}")
    print(f"    us-gaap tags available: {len(payload.get('facts', {}).get('us-gaap', {}))}")
    has_eps = "EarningsPerShareDiluted" in payload.get("facts", {}).get("us-gaap", {})
    has_ni = "NetIncomeLoss" in payload.get("facts", {}).get("us-gaap", {})
    print(f"    has EarningsPerShareDiluted tag: {has_eps}")
    print(f"    has NetIncomeLoss tag: {has_ni}")
except Exception as e:
    print(f"    !! EXCEPTION during companyfacts fetch: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("SUCCESS -- both SEC endpoints reachable and returned usable data.")
print("If the Streamlit app is still showing 'No EDGAR data found', the")
print("bug is downstream in _build_financial_history, not in network access.")
print("=" * 70)
