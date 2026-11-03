# Business Quality Terminal

Rule #1 / Buffett-style valuation. SEC EDGAR for financials, Yahoo (yfinance) for prices only.

    pip install -r requirements.txt
    streamlit run app.py

Tabs: Ticker Lookup | Portfolio (sell signals) | Watchlist (buy signals) |
Top 25 Undervalued | Top 25 + Dividend Growers.

* Portfolio / watchlist save automatically to `user_data.json` (next to app.py).
* Fetched data is cached in `.cache/` (fundamentals 24h, portfolio/watchlist prices 15 min).
  The sidebar "Refresh data" button clears it. A long market scan can be stopped and resumed.
* `python3 test_all.py` runs the offline test suite (no network needed).
* `debug_sec_fetch.py` / `debug_yfinance_fetch.py` show the raw SEC / Yahoo error if a fetch fails.
