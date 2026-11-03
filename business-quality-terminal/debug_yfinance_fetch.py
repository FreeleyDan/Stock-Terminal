"""
debug_yfinance_fetch.py

Run standalone: python3 debug_yfinance_fetch.py

Hits yfinance directly for AAPL with nothing caught, so whatever is
actually failing (rate limit, outdated library, changed Yahoo response
shape, etc.) prints plainly instead of disappearing into
get_current_price_and_pe's silent (None, None).
"""

import sys

try:
    import yfinance as yf
    print(f"yfinance version: {yf.__version__}")
except Exception as e:
    print(f"!! Could not import yfinance: {type(e).__name__}: {e}")
    print("   Run: pip install --upgrade yfinance")
    sys.exit(1)

print("\n[1] Ticker('AAPL').info (what the app tries first)")
try:
    t = yf.Ticker("AAPL")
    info = t.info
    print(f"    info type: {type(info)}, keys: {len(info) if info else 0}")
    print(f"    currentPrice: {info.get('currentPrice')}")
    print(f"    regularMarketPrice: {info.get('regularMarketPrice')}")
    print(f"    trailingPE: {info.get('trailingPE')}")
except Exception as e:
    print(f"    !! EXCEPTION: {type(e).__name__}: {e}")

print("\n[2] Ticker('AAPL').history(period='5d') (the app's fallback for price)")
try:
    hist = t.history(period="5d")
    print(f"    rows returned: {len(hist)}")
    if not hist.empty:
        print(f"    last Close: {hist['Close'].iloc[-1]}")
    else:
        print("    !! history returned EMPTY dataframe -- this is the likely root cause.")
except Exception as e:
    print(f"    !! EXCEPTION: {type(e).__name__}: {e}")

print("\n[3] fast_info (lighter-weight alternative, often more reliable than .info)")
try:
    fi = t.fast_info
    print(f"    last_price: {fi.get('lastPrice', fi.get('last_price', 'n/a'))}")
except Exception as e:
    print(f"    !! EXCEPTION: {type(e).__name__}: {e}")

print("\nDone. Paste this full output back.")
