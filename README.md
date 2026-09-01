# TRADEHUB

A personal options-trading research dashboard for US equities, built with
[Streamlit](https://streamlit.io/). It surfaces dealer-hedging "magnet pin"
targets, options-derived analytics (gamma exposure, IV rank/percentile,
put/call ratio, volatility skew), intraday breakout/reclaim setups, and a
pre-market context view — plus a trade journal and optional Telegram alerts.

> **Not financial advice.** This is a personal research tool for educational
> use. Options trading involves substantial risk of loss. Nothing here is a
> recommendation to buy or sell any security. Do your own diligence.

---

## Features

- **🎯 Magnet Pins** — max-pain / dealer-hedging targets per watchlist ticker,
  with OI-confidence tiering and computed trade mechanics (suggested strike,
  estimated delta, stop, profit target, trade-readiness gate).
- **Options analytics** (per magnet ticker, from one cached option-chain fetch):
  - **GEX** — net gamma exposure, gamma-flip level, and POSITIVE/NEGATIVE regime.
  - **IV Rank + IV Percentile** — logged daily to build a real 52-week history;
    shows a labelled realized-vol *proxy* (`~`) until ~60 days accumulate.
  - **Put/Call Ratio** — put vs call volume for the nearest expiry, with
    EXTREME-put / EXTREME-call flags.
  - **Skew + Skew Trend** — 25-delta put IV minus 25-delta call IV, plus a
    5-day steepening/flattening trend built from logged history.
- **⏱ Intraday** — opening-range-breakout (ORB) and VWAP-reclaim setups.
- **🌅 Pre-Market** — overnight gap, gap-vs-typical proxy, prior-day levels,
  VIX regime, and sector-ETF context.
- **🌐 Futures** — a GREEN/AMBER/RED field-condition gauge from index futures
  (ES/NQ/YM/RTY), crude, gold, Treasuries, the dollar index, and VIX.
- **📓 Journal & Trades** — log/track trades, performance analytics, and an
  active-trade tracker.
- **Telegram alerts** — optional push notifications for qualifying signals.

Backtest tooling ships as standalone scripts (`alpha_backtest.py`,
`contraction_backtest.py`, `premarket_report.py`).

---

## Requirements

- Python 3.10+
- A free [Polygon.io](https://polygon.io/) API key (used for last-trade prices
  and intraday aggregates)
- (Optional) A Telegram bot token + chat ID for alerts

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Copy the secrets template and fill in your own values:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml`:

   ```toml
   POLYGON_KEY = "your_polygon_api_key_here"
   TELEGRAM_BOT_TOKEN = "your_telegram_bot_token_here"
   TELEGRAM_CHAT_ID = "your_telegram_chat_id_here"
   ```

   `secrets.toml` is git-ignored, so your keys are never committed. Telegram
   values can be left blank if you don't want alerts.

---

## Running

```bash
streamlit run alpha_scanner.py
```

On Windows you can also double-click **`tradehub.bat`**, which launches the
dashboard on port 8501 from this folder.

---

## Data sources & honest limitations

- Market data comes from **yfinance** (option chains, quotes) and **Polygon**
  (last-trade prices, intraday minute bars).
- Free/entitlement-limited data means some analytics are **approximations**,
  and they are labelled as such in the UI:
  - There is **no free source of historical implied volatility**, so IV Rank /
    Percentile are logged daily and shown as a realized-vol proxy (`~`) until a
    real history builds. Skew trend works the same way.
  - yfinance option **volume is same-day cumulative**, so Put/Call ratio reads
    `N/A` before the open and firms up during the session.
  - Pre-market **volume is reported as zero** by Yahoo, so the pre-market view
    uses a move-based gap proxy instead.
  - True **unusual-options-activity** (trade-level sweep classification) is not
    available on the free Polygon tier.

---

## Project layout

```
AlphaScanner/
  alpha_scanner.py          # the Streamlit dashboard (main app)
  alpha_backtest.py         # ORB / VWAP backtest engine
  contraction_backtest.py   # a rejected thesis, kept for reference
  premarket_report.py       # standalone CLI pre-market report
  requirements.txt
  tradehub.bat              # Windows launcher
  .streamlit/
    secrets.toml.example    # copy to secrets.toml and fill in
```

Runtime files (`iv_history.json`, `skew_history.json`, `scanner_prefs.json`,
`trade_journal.json`, `last_alert.json`) are created locally as you use the app
and are git-ignored.
