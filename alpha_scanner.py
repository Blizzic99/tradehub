# AlphaAssistant-0.1 - Hybrid Dealer Magnet and Blow-Off Top Scanner
# Data: Polygon.io (prices) + Yahoo Finance (options) | Streamlit Dashboard
# Level 2: Pre-Market Context Engine (now with market status)
# Level 3: Volume & Open Interest Confirmation
# Level 4: Editable Trade Journal
# Level 4.5: Active Trade Tracker
# Level 5: Telegram Alerts (new tradeable signals)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf
import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from math import log, sqrt
from statistics import NormalDist

# DST-aware US market timezone (shared by market-status + intraday scanners)
try:
    from zoneinfo import ZoneInfo
    ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    ET_ZONE = timezone(timedelta(hours=-4))  # fallback: EDT

# Streamlit script context — attached to worker threads so @st.cache_data works inside the
# thread pool (and no "missing ScriptRunContext" warnings). Optional: falls back gracefully.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:
    add_script_run_ctx = None
    get_script_run_ctx = None

# --- Yahoo Finance access hardening ---------------------------------------------------------------
# Yahoo blocks/throttles datacenter IPs (Streamlit Community Cloud is one), which is why the hosted
# app can't pull option chains while a local run works. A curl_cffi session that impersonates Chrome
# gives a real-browser TLS fingerprint + persistent crumb/cookies, which often gets past that
# throttling. Created once, reused everywhere; falls back to a plain Ticker if curl_cffi is absent.
try:
    from curl_cffi import requests as _cffi_requests
    _YF_SESSION = _cffi_requests.Session(impersonate="chrome")
except Exception:
    _YF_SESSION = None

def _yf_ticker(ticker):
    """yf.Ticker on the browser-impersonating session when available (helps past cloud throttling)."""
    if _YF_SESSION is not None:
        try:
            return yf.Ticker(ticker, session=_YF_SESSION)
        except Exception:
            pass
    return yf.Ticker(ticker)

def _yf_options(stock, retries=1, pause=0.7):
    """stock.options with a light retry — under load Yahoo intermittently returns an empty list or
    raises; one short retry recovers many of those without meaningfully slowing a healthy scan."""
    for attempt in range(retries + 1):
        try:
            exps = stock.options
            if exps:
                return exps
        except Exception:
            pass
        if attempt < retries:
            time.sleep(pause)
    return ()

SCAN_WORKERS = 6   # concurrent yfinance/Polygon fetches per scan phase. Kept moderate: too many
                   # simultaneous yfinance calls (esp. from a shared Streamlit Cloud IP) trip Yahoo's
                   # bot-detection -> empty frames / HTTP 429. Bump toward 8-10 for a faster LOCAL scan.

def _parallel_map(fn, items):
    """Thread-pool map preserving input order, propagating the Streamlit context so cached
    fetches work inside workers. Each item is ISOLATED — a worker that raises yields None for THAT
    item instead of collapsing the whole phase to serial (which would re-run everything and re-raise
    on the bad item). One bad ticker never aborts the scan; callers treat None as a failed item."""
    items = list(items)
    if not items:
        return []
    ctx = get_script_run_ctx() if get_script_run_ctx else None
    def _wrap(item):
        try:
            if add_script_run_ctx and ctx is not None:
                add_script_run_ctx(ctx=ctx)
            return fn(item)
        except Exception:
            return None          # isolate this item's failure; caller handles None
    try:
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
            return list(ex.map(_wrap, items))
    except Exception:
        return [_wrap(x) for x in items]   # pool couldn't run — serial, still per-item safe

# ============================================================
# CONFIG
# ============================================================
def _secret(name, default=""):
    """Read a secret: Streamlit secrets first (this is how Streamlit CLOUD injects them, and how a
    LOCAL run picks up .streamlit/secrets.toml when launched from the project dir), then the
    environment, then the .streamlit/secrets.toml NEXT TO THIS SCRIPT (and ~/.streamlit/secrets.toml)
    read directly. That last, file-anchored fallback matters: st.secrets only finds secrets.toml
    RELATIVE TO the directory `streamlit run` was launched from, so launching the app from any other
    working dir (or a different copy of it) silently blanked POLYGON_KEY and made every options fetch
    fail — the '49/49 no options data' symptom. Anchoring to __file__ finds the key regardless of CWD.
    Keeps API keys/tokens OUT of source so the .py can be shared/version-controlled safely."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    v = os.environ.get(name)
    if v:
        return v
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".streamlit", "secrets.toml"),
                 os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")):
        if not os.path.exists(path):
            continue
        try:                                        # preferred: real TOML parse (stdlib 3.11+)
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if name in data:
                return data[name]
        except Exception:                           # fallback: minimal KEY = "value" line parse
            try:
                for line in open(path, encoding="utf-8"):
                    k, _, val = line.partition("=")
                    if k.strip() == name:
                        return val.strip().strip('"').strip("'")
            except Exception:
                pass
    return default

POLYGON_KEY = _secret("POLYGON_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# Telegram alert settings (loaded from secrets/env — not hard-coded)
TELEGRAM_BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")   # from BotFather
TELEGRAM_CHAT_ID = _secret("TELEGRAM_CHAT_ID")       # numeric ID from userinfobot (str is fine for the API)
ALERTS_ENABLED = True                                # master toggle; can also be changed in sidebar
LAST_ALERT_FILE = "last_alert.json"

# Sidebar preference persistence — remembers toggle choices across refreshes AND full page reloads.
PREFS_FILE = "scanner_prefs.json"

def load_prefs():
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_prefs(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass

# ============================================================
# MARKET HOLIDAYS 2026 (US stock market closed)
# ============================================================
HOLIDAYS_2026 = [
    datetime(2026, 1, 1),    # New Year's Day
    datetime(2026, 1, 19),   # Martin Luther King Jr. Day
    datetime(2026, 2, 16),   # Presidents' Day
    datetime(2026, 4, 3),    # Good Friday (approximate)
    datetime(2026, 5, 25),   # Memorial Day
    datetime(2026, 6, 19),   # Juneteenth
    datetime(2026, 7, 3),    # Independence Day observed (Friday)
    datetime(2026, 9, 7),    # Labor Day
    datetime(2026, 11, 26),  # Thanksgiving
    datetime(2026, 12, 25),  # Christmas
]

def is_market_open():
    now = datetime.now(timezone.utc).astimezone(ET_ZONE)
    today = now.date()
    for holiday in HOLIDAYS_2026:
        if holiday.date() == today:
            next_day = now + timedelta(days=1)
            while next_day.weekday() >= 5 or next_day.date() in [h.date() for h in HOLIDAYS_2026]:
                next_day += timedelta(days=1)
            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            return "🔴 Market Closed – Holiday", f"Next open: {next_open.strftime('%A %b %d, %I:%M %p ET')}"
    if now.weekday() >= 5:
        days_until_monday = 7 - now.weekday()
        next_monday = now + timedelta(days=days_until_monday)
        next_monday = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
        return "🔴 Market Closed – Weekend", f"Next open: Monday {next_monday.strftime('%b %d, %I:%M %p ET')}"
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now <= market_close:
        return "🟢 Market Open", "Closes at 4:00 PM ET"
    if now < market_open:
        return "🟡 Pre-Market", f"Opens today at 9:30 AM ET"
    next_open = now + timedelta(days=1)
    while next_open.weekday() >= 5 or next_open.date() in [h.date() for h in HOLIDAYS_2026]:
        next_open += timedelta(days=1)
    next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
    return "🔴 Market Closed – After Hours", f"Next open: {next_open.strftime('%A %b %d, %I:%M %p ET')}"

# ============================================================
# TELEGRAM ALERTS (Level 5)
# ============================================================
def send_telegram_alert(message):
    """Send a message via Telegram bot. Returns True if successful."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except:
        return False

def load_last_alert():
    """Load the last alerted trade from a local file."""
    if os.path.exists(LAST_ALERT_FILE):
        try:
            with open(LAST_ALERT_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_last_alert(alert_dict):
    """Save the last alerted trade to a local file."""
    with open(LAST_ALERT_FILE, "w") as f:
        json.dump(alert_dict, f, indent=2)

def should_send_alert(best_signal):
    """Return True if the current best tradeable signal is different from the last alert."""
    last = load_last_alert()
    if not last:
        return True
    # Compare key fields: ticker, direction, max pain, expiry
    same = (
        last.get("ticker") == best_signal["Ticker"] and
        last.get("direction") == best_signal["Direction"] and
        last.get("max_pain") == best_signal["Max Pain Strike"] and
        last.get("expiry") == best_signal["Expiry"]
    )
    return not same

def generate_alert_message(best_signal):
    """Create a formatted Telegram message for the best trade candidate."""
    direction = best_signal["Direction"]
    action = "BUY CALL 📈" if direction == "MAGNET UP" else "BUY PUT 📉"
    msg = f"""
🚨 <b>New Tradeable Signal</b> 🚨
<b>{best_signal['Ticker']}</b> — {direction}
Magnet: {best_signal['Max Pain Strike']}
Current Price: {best_signal['Current Price']}
Distance: {best_signal['Distance %']}
OI Confidence: {best_signal['OI Confidence']}
Action: {action}
Expiry: {best_signal['Expiry']}
"""
    return msg.strip()

# --- Blow-off top alerts (Level 5, extended) ---
LAST_BLOWOFF_FILE = "last_blowoff_alert.json"

def load_blowoff_alerts():
    if os.path.exists(LAST_BLOWOFF_FILE):
        try:
            with open(LAST_BLOWOFF_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_blowoff_alerts(state):
    with open(LAST_BLOWOFF_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_new_blowoff_tickers(current_tickers):
    """Return (new_tickers, state). Dedupes per calendar day so the same blow-off isn't re-alerted on every refresh."""
    today = datetime.now().strftime("%Y-%m-%d")
    state = load_blowoff_alerts()
    if state.get("date") != today:
        state = {"date": today, "tickers": []}
    already = set(state.get("tickers", []))
    new = [t for t in current_tickers if t not in already]
    return new, state

def generate_blowoff_message(rows):
    """Format a Telegram message for one or more newly-detected blow-off tops."""
    lines = ["⚠️ <b>Blow-Off Top Detected</b> ⚠️"]
    for r in rows:
        lines.append(
            f"<b>{r['Ticker']}</b> {r['Price']} | {r['Detachment from 20MA']} above 20MA | vol {r['Volume Ratio']}\n"
            f"Action: SHORT / BUY PUTS (stop above today's high)"
        )
    return "\n".join(lines)

# ============================================================
# TRADE JOURNAL STORAGE (Level 4)
# ============================================================
JOURNAL_FILE = "trade_journal.json"

def load_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_journal(journal):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2)

def add_trade(ticker="", direction="PUT", strike="", expiry="", entry_price="", exit_price="", result="OPEN", notes="", setup="", stop="", target=""):
    journal = load_journal()
    trade = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "direction": direction,
        "strike": strike,
        "expiry": expiry,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "result": result,
        "setup": setup,
        "stop": stop,
        "target": target,
        "notes": notes
    }
    journal.append(trade)
    save_journal(journal)
    return journal

# Canonical journal field order (new setup/stop/target are backward-compatible: older
# entries without them load fine and are normalized to "" before editing/display).
JOURNAL_FIELDS = ["date", "ticker", "direction", "strike", "expiry", "entry_price",
                  "exit_price", "result", "setup", "stop", "target", "notes"]

def update_journal(updated_df):
    new_journal = []
    for _, row in updated_df.iterrows():
        trade = {
            "date": str(row.get("date", "")),
            "ticker": str(row.get("ticker", "")),
            "direction": str(row.get("direction", "")),
            "strike": str(row.get("strike", "")),
            "expiry": str(row.get("expiry", "")),
            "entry_price": str(row.get("entry_price", "")),
            "exit_price": str(row.get("exit_price", "")),
            "result": str(row.get("result", "OPEN")),
            "setup": str(row.get("setup", "")),
            "stop": str(row.get("stop", "")),
            "target": str(row.get("target", "")),
            "notes": str(row.get("notes", ""))
        }
        new_journal.append(trade)
    save_journal(new_journal)

def delete_trade(index):
    journal = load_journal()
    if 0 <= index < len(journal):
        journal.pop(index)
        save_journal(journal)

# 1 US equity option contract = 100 shares; entry/exit prices are premiums, so dollar P&L per
# contract = (premium move) * 100. Named so the journal + analytics stay consistent.
OPTION_CONTRACT_MULTIPLIER = 100

def get_journal_stats(journal):
    if not journal:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}
    total = len(journal)
    wins = sum(1 for t in journal if t["result"] == "WIN")
    losses = sum(1 for t in journal if t["result"] == "LOSS")
    closed = wins + losses          # win rate excludes OPEN trades from the denominator
    win_rate = round((wins / closed) * 100, 1) if closed > 0 else 0
    total_pnl = 0
    for t in journal:
        try:
            entry = float(t["entry_price"])
            exit_p = float(t["exit_price"])
            if t["direction"] == "CALL":
                total_pnl += (exit_p - entry) * OPTION_CONTRACT_MULTIPLIER
            else:
                total_pnl += (entry - exit_p) * OPTION_CONTRACT_MULTIPLIER
        except:
            pass
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": round(total_pnl, 2)
    }

def get_performance_analytics(journal):
    """Profit factor, max drawdown, and an equity curve from closed trades. Returns None if no priced trades."""
    pnls = []
    for t in journal:
        try:
            entry = float(t["entry_price"])
            exit_p = float(t["exit_price"])
        except:
            continue
        if t.get("direction") == "CALL":
            pnls.append((exit_p - entry) * OPTION_CONTRACT_MULTIPLIER)
        else:
            pnls.append((entry - exit_p) * OPTION_CONTRACT_MULTIPLIER)
    if not pnls:
        return None
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    equity = []
    run = 0.0
    for p in pnls:
        run += p
        equity.append(round(run, 2))
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "closed_trades": len(pnls),
        "profit_factor": profit_factor,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "equity_curve": equity
    }

# ============================================================
# ACTIVE TRADE TRACKER (Level 4.5)
# ============================================================
def get_open_trades():
    journal = load_journal()
    return [t for t in journal if t["result"] == "OPEN"]

def get_option_price(ticker, strike, direction, expiry):
    try:
        stock = _yf_ticker(ticker)
        exp_date = None
        for exp in _yf_options(stock):
            if exp >= expiry:
                exp_date = exp
                break
        if exp_date is None:
            return None
        chain = stock.option_chain(exp_date)
        if direction == "CALL":
            options = chain.calls
        else:
            options = chain.puts
        strike_num = float(strike)
        match = options[options["strike"] == strike_num]
        if match.empty:
            options["diff"] = abs(options["strike"] - strike_num)
            match = options.nsmallest(1, "diff")
        if not match.empty:
            return match["lastPrice"].iloc[0]
        return None
    except:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_stock_price(ticker):
    try:
        stock = _yf_ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1]
        return None
    except:
        return None

def time_until_market_close(expiry_str):
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        # Anchor to 3:50 PM ET explicitly. Without tz, datetime.now() is the server clock (UTC on
        # Streamlit Cloud, local time elsewhere) and the countdown is off by hours.
        close_time = expiry_date.replace(hour=15, minute=50, second=0, tzinfo=ET_ZONE)
        now = datetime.now(ET_ZONE)
        diff = close_time - now
        if diff.total_seconds() < 0:
            return "EXPIRED - CLOSE IMMEDIATELY"
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        if diff.days > 0:
            return str(diff.days) + "d " + str(hours) + "h " + str(minutes) + "m"
        return str(hours) + "h " + str(minutes) + "m"
    except:
        return "N/A"

def track_active_trades():
    open_trades = get_open_trades()
    if not open_trades:
        return []
    tracked = []
    for trade in open_trades:
        ticker = trade["ticker"]
        direction = trade["direction"]
        strike = trade["strike"]
        expiry = trade["expiry"]
        entry = trade["entry_price"]
        stock_price = get_stock_price(ticker)
        option_price = get_option_price(ticker, strike, direction, expiry) if stock_price else None
        time_left = time_until_market_close(expiry)
        try:
            entry_num = float(entry)
            if option_price is not None:
                option_num = float(option_price)
                pnl = round(option_num - entry_num, 2)
                pnl_pct = round(((option_num - entry_num) / entry_num) * 100, 1) if entry_num > 0 else 0
            else:
                pnl = None
                pnl_pct = None
        except:
            pnl = None
            pnl_pct = None
        status = "NEUTRAL"
        if pnl_pct is not None:
            if pnl_pct > 20:
                status = "STRONG GAIN"
            elif pnl_pct > 5:
                status = "GAIN"
            elif pnl_pct < -50:
                status = "CRITICAL LOSS"
            elif pnl_pct < -20:
                status = "LOSS"
        if "EXPIRED" in str(time_left):
            status = "EXIT NOW"
        tracked.append({
            "Ticker": ticker,
            "Direction": direction,
            "Strike": "$" + str(strike),
            "Entry Price": "$" + str(entry),
            "Current Option Price": "$" + str(round(option_price, 2)) if option_price else "N/A",
            "Stock Price": "$" + str(round(stock_price, 2)) if stock_price else "N/A",
            "P&L ($)": "$" + str(pnl) if pnl is not None else "N/A",
            "P&L (%)": str(pnl_pct) + "%" if pnl_pct is not None else "N/A",
            "Time Left": time_left,
            "Status": status
        })
    return tracked

# ============================================================
# CORE WATCHLIST
# ============================================================
CORE_WATCHLIST = [
    # === CURRENT CORE ===
    "SPY", "QQQ", "AAPL", "TSLA", "NVDA",
    "AMZN", "META", "MSFT", "GOOGL", "AMD",
    "IWM", "COIN", "PLTR", "SMCI",

    # === MANUAL ADDS ===
    "SOLS",   # Solstice Advanced Materials
    "LEU",    # Centrus Energy (nuclear fuel)
    "AMTM",   # Amentum Holdings (defense/gov services)

    # === AI DATA LAYER ===
    "SNOW",   # Snowflake — cloud data platform, AI backbone
    "MDB",    # MongoDB — flexible database for AI apps
    "ORCL",   # Oracle — cloud infra + AI data, OCI growth
    "CRM",    # Salesforce — enterprise AI platform (Agentforce)

    # === CHIP SUPPLY CHAIN ===
    "TSM",    # TSMC — manufactures advanced AI chips
    "ASML",   # Only maker of EUV lithography machines
    "AMAT",   # Applied Materials — chip fab equipment
    "LRCX",   # Lam Research — etch/deposition equipment
    "KLAC",   # KLA Corp — process control / inspection
    "AVGO",   # Broadcom — custom AI chips + networking
    "MRVL",   # Marvell — custom AI silicon + optical interconnect
    "MU",     # Micron — HBM memory for AI accelerators
    "INTC",   # Intel — foundry services + data center chips

    # === AI CLOUD INFRASTRUCTURE ===
    "CRWV",   # CoreWeave — GPU cloud for AI workloads
    "NET",    # Cloudflare — AI inference edge + networking

    # === HIGH OPTIONS VOLUME ETFs ===
    "XLF",    # Financials ETF
    "XLE",    # Energy ETF
    "GLD",    # Gold ETF — macro hedge
    "TLT",    # 20yr Treasury ETF — rates plays
    "SMH",    # Semiconductor ETF
    "XLK",    # Technology sector ETF
    "XBI",    # Biotech ETF — high volatility

    # === SECTOR ETFs ===
    "XLV",    # Healthcare
    "XLI",    # Industrials
    "XLY",    # Consumer Discretionary
    "XLP",    # Consumer Staples
    "ARKK",   # ARK Innovation — high volatility

    # === HIGH VOLATILITY / MEME ===
    "GME",    # GameStop
    "AMC",    # AMC Entertainment
    "MARA",   # Marathon Digital (crypto mining)
    "RIOT",   # Riot Platforms (crypto mining)
    "HOOD",   # Robinhood — retail trading proxy
]

# ============================================================
# FUTURES CONTEXT (Level 7) — risk/reward regime from Yahoo Finance futures.
# Separate fetch from the pre-market cache. Not a signal — a FIELD-CONDITION gauge.
# ============================================================
FUTURES_WATCHLIST = {
    "ES": "ES=F",      # S&P 500 E-mini
    "NQ": "NQ=F",      # Nasdaq-100 E-mini
    "YM": "YM=F",      # Dow E-mini
    "RTY": "RTY=F",    # Russell 2000 E-mini
    "CL": "CL=F",      # Crude Oil WTI
    "GC": "GC=F",      # Gold
    "ZB": "ZB=F",      # 30-yr Treasury Bond
    "ZN": "ZN=F",      # 10-yr Treasury Note
    "DX": "DX-Y.NYB",  # US Dollar Index (ICE spot index — Yahoo's DX=F future 404s)
}

# Risk/reward thresholds — tuned for overnight / pre-market futures (change % vs prior close).
FUTURES_RISK_OFF = {
    "ES": -0.50,       # S&P futures down > 0.5%
    "NQ": -0.75,       # Nasdaq down > 0.75%
    "CL": 1.50,        # Oil up > 1.5% (inflation / geopolitical shock)
    "VIX": 22.0,       # VIX above 22 = fear regime
}
FUTURES_RISK_ON = {
    "ES": 0.25,        # S&P futures up > 0.25%
    "NQ": 0.35,        # Nasdaq up > 0.35%
    "CL": -0.50,       # Oil down or flat (no energy stress)
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_top_option_volume_tickers(limit=50):
    try:
        candidate_tickers = [
            "SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMZN", "META",
            "MSFT", "GOOGL", "AMD", "COIN", "PLTR", "SMCI", "GME", "AMC",
            "BA", "NFLX", "DIS", "PYPL", "UBER", "SNAP", "RIVN", "LCID",
            "BABA", "SOFI", "MARA", "RIOT", "TLT", "EEM", "F", "XOM",
            "CVX", "WMT", "JPM", "BAC", "C", "V", "MA", "PFE", "MRNA"
        ]
        def _vol_for(ticker):
            try:
                stock = _yf_ticker(ticker)
                opt_dates = _yf_options(stock)
                if not opt_dates:
                    return (ticker, None)
                chain = stock.option_chain(opt_dates[0])
                total_vol = chain.calls["volume"].sum() + chain.puts["volume"].sum()
                return (ticker, int(total_vol)) if total_vol > 0 else (ticker, None)
            except Exception:
                return (ticker, None)
        pairs = _parallel_map(_vol_for, candidate_tickers)   # concurrent — this is the slowest scan phase
        volume_map = {t: v for t, v in pairs if v}
        sorted_tickers = sorted(volume_map, key=volume_map.get, reverse=True)
        return sorted_tickers[:limit]
    except Exception:
        # Cached function — fail quietly; the dashboard still has the core watchlist
        return []

@st.cache_data(ttl=120, show_spinner=False)
def _spot_price(ticker):
    """Current-ish underlying price. Polygon real-time last-trade needs a paid STOCK plan (403 on the
    free stock tier), so fall back to yfinance's latest close (one light call via the browser session,
    cloud-safe) and finally Polygon's prev close (EOD, entitled) as a stale-but-present backstop."""
    try:
        pr = requests.get(POLYGON_BASE_URL + "/v2/last/trade/" + ticker,
                          params={"apiKey": POLYGON_KEY}, timeout=5).json()
        if pr.get("status") == "OK" and (pr.get("results") or {}).get("p"):
            return float(pr["results"]["p"])
    except Exception:
        pass
    try:
        h = _yf_ticker(ticker).history(period="1d")
        if h is not None and not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    try:
        pv = requests.get(POLYGON_BASE_URL + "/v2/aggs/ticker/" + ticker + "/prev",
                          params={"apiKey": POLYGON_KEY, "adjusted": "true"}, timeout=6).json()
        res = pv.get("results") or []
        if res and res[0].get("c"):
            return float(res[0]["c"])
    except Exception:
        pass
    return None

@st.cache_data(ttl=120, show_spinner=False)
def _polygon_chain(ticker):
    """Nearest-expiry option chain from POLYGON (entitled options snapshot) — the single source of
    truth for all options analytics (max-pain, GEX, IV, P/C, skew, strike/delta). Normalized to:
    {expiry, dte_years, spot, strikes[all], call_oi/put_oi, call_iv/put_iv, call_vol_total,
    put_vol_total}. Paginates the snapshot; drops deep-ITM/OTM garbage IV (>=500%). Real OI + IV,
    and unlike Yahoo it isn't blocked from datacenter IPs (fixes the cloud outage). None on failure.
    Cached 120s so get_max_pain_strike AND get_option_chain_snapshot share ONE fetch per ticker."""
    try:
        if not POLYGON_KEY:
            return None
        today = datetime.now().date().isoformat()
        rr = requests.get(POLYGON_BASE_URL + "/v3/reference/options/contracts",
                          params={"apiKey": POLYGON_KEY, "underlying_ticker": ticker,
                                  "expiration_date.gte": today, "expired": "false",
                                  "sort": "expiration_date", "order": "asc", "limit": 1}, timeout=8)
        exps = rr.json().get("results") or []
        exp_str = exps[0].get("expiration_date") if exps else None
        if not exp_str:
            return None
        call_iv, put_iv, call_oi, put_oi = {}, {}, {}, {}
        call_vol = put_vol = 0.0
        strikes = set()
        url = POLYGON_BASE_URL + "/v3/snapshot/options/" + ticker
        params = {"apiKey": POLYGON_KEY, "expiration_date": exp_str, "limit": 250}
        pages = 0
        while url and pages < 10:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            j = resp.json()
            for c in (j.get("results") or []):
                det = c.get("details") or {}
                try:
                    k = float(det.get("strike_price"))
                except Exception:
                    continue
                strikes.add(k)
                oi = c.get("open_interest")
                iv = c.get("implied_volatility")
                vol = (c.get("day") or {}).get("volume") or 0.0
                if det.get("contract_type") == "call":
                    if oi and oi > 0: call_oi[k] = float(oi)
                    if iv and 0 < iv < 5: call_iv[k] = float(iv)     # <500% drops deep-strike garbage IV
                    call_vol += float(vol or 0)
                else:
                    if oi and oi > 0: put_oi[k] = float(oi)
                    if iv and 0 < iv < 5: put_iv[k] = float(iv)
                    put_vol += float(vol or 0)
            nxt = j.get("next_url")
            url, params, pages = (nxt, {"apiKey": POLYGON_KEY}, pages + 1) if nxt else (None, params, pages)
        if not strikes:
            return None
        spot = _spot_price(ticker)
        exp_close = datetime.strptime(exp_str, "%Y-%m-%d").replace(hour=16, minute=0)
        dte_years = max((exp_close - datetime.now()).total_seconds(), 3600) / (365.0 * 24 * 3600)
        return {"expiry": exp_str, "dte_years": dte_years, "spot": spot,
                "strikes": sorted(strikes), "call_iv": call_iv, "put_iv": put_iv,
                "call_oi": call_oi, "put_oi": put_oi,
                "call_vol_total": call_vol, "put_vol_total": put_vol}
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def _atm_iv_30d(ticker):
    """~30-DAY CONSTANT-MATURITY at-the-money IV from the Polygon snapshot. This is the reference fed
    to IV Rank/Percentile and to the daily IV log, so 'current IV' matches the ~30-day (standard
    monthly, 3rd-Friday) tenor of the backfilled history (alpha_backtest.backfill_iv_history) instead
    of drifting with the nearest weekly expiry that _polygon_chain returns. Targets the nearest
    monthly >= 21 DTE, reads the ATM strike's call+put implied_vol and returns their mean. None on
    failure -- the caller then SKIPS logging that day rather than mixing a weekly reading into a
    monthly series. NOTE: the trade's stop/target (compute_trade_mechanics._atm_iv) still uses the
    nearest-expiry IV, which is correct there since the trade is on that expiry."""
    try:
        if not POLYGON_KEY:
            return None
        spot = _spot_price(ticker)
        if spot is None:
            return None
        cutoff = (datetime.now() + timedelta(days=21)).date()   # nearest monthly >= 21 DTE
        y, m, exp = cutoff.year, cutoff.month, None
        for _ in range(4):
            wd = datetime(y, m, 1).weekday()
            tf = datetime(y, m, 1 + (4 - wd) % 7 + 14).date()   # 1st Friday + 2 weeks = 3rd Friday
            if tf >= cutoff:
                exp = tf
                break
            m = m + 1 if m < 12 else 1
            y = y if m != 1 else y + 1
        if exp is None:
            return None
        r = requests.get(POLYGON_BASE_URL + "/v3/snapshot/options/" + ticker,
                         params={"apiKey": POLYGON_KEY, "expiration_date": exp.isoformat(),
                                 "strike_price.gte": round(spot * 0.97, 2),
                                 "strike_price.lte": round(spot * 1.03, 2), "limit": 250}, timeout=12)
        if r.status_code != 200:
            return None
        by_strike = {}
        for c in (r.json().get("results") or []):
            det = c.get("details") or {}
            try:
                k = float(det.get("strike_price"))
            except Exception:
                continue
            iv = c.get("implied_volatility")
            if iv and 0 < iv < 5:                               # <500% drops deep-strike garbage IV
                by_strike.setdefault(k, {})[det.get("contract_type")] = float(iv)
        if not by_strike:
            return None
        nk = min(by_strike, key=lambda k: abs(k - spot))        # ATM strike (nearest to spot)
        ivs = [by_strike[nk][t] for t in ("call", "put") if t in by_strike[nk]]
        return sum(ivs) / len(ivs) if ivs else None
    except Exception:
        return None

def get_max_pain_strike(ticker):
    """Returns (max_pain_strike, current_price, expiry_date, total_oi_at_pain, atm_iv, skew) from the
    POLYGON chain (real OI + IV) via the shared _polygon_chain (cached 120s). Max-pain is computed
    over strikes that actually carry OI; ATM IV + 25-delta skew come off the same chain."""
    try:
        ch = _polygon_chain(ticker)
        if not ch:
            return None, None, None, 0, None, None
        call_oi, put_oi = ch["call_oi"], ch["put_oi"]
        oi_strikes = sorted(set(call_oi) | set(put_oi))
        if not oi_strikes:
            return None, None, None, 0, None, None
        min_pain, max_pain_strike = float("inf"), None
        for candidate in oi_strikes:
            pain = 0.0
            for k in oi_strikes:
                if k > candidate:
                    pain += (k - candidate) * call_oi.get(k, 0.0)
                elif k < candidate:
                    pain += (candidate - k) * put_oi.get(k, 0.0)
            if pain < min_pain:
                min_pain, max_pain_strike = pain, candidate
        if max_pain_strike is None:
            return None, None, None, 0, None, None
        total_oi_at_pain = int(call_oi.get(max_pain_strike, 0) + put_oi.get(max_pain_strike, 0))
        current_price = ch.get("spot")
        expiry_date = datetime.strptime(ch["expiry"], "%Y-%m-%d").date()
        # ATM IV (daily IV log): strike nearest spot, mean of its call & put IV.
        atm_iv = None
        try:
            if current_price:
                k = min(ch["strikes"], key=lambda x: abs(x - current_price))
                ivs = [v for v in (ch["call_iv"].get(k), ch["put_iv"].get(k)) if v and v > 0]
                if ivs:
                    atm_iv = sum(ivs) / len(ivs)
        except Exception:
            atm_iv = None
        # 25-delta skew (daily skew log) off the same chain.
        skew = None
        try:
            skew = compute_skew({"call_iv": ch["call_iv"], "put_iv": ch["put_iv"],
                                 "dte_years": ch["dte_years"]}, current_price)
        except Exception:
            skew = None
        return max_pain_strike, current_price, expiry_date, total_oi_at_pain, atm_iv, skew
    except Exception:
        return None, None, None, 0, None, None

@st.cache_data(ttl=300, show_spinner=False)
def check_parabolic_condition(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, session=_YF_SESSION)
        if df.empty or len(df) < 20:
            return False, None, None, None
        close = df["Close"].values
        volume = df["Volume"].values
        ma20 = pd.Series(close).rolling(20).mean().values
        current_price = close[-1]
        current_ma20 = ma20[-1]
        current_vol = volume[-1]
        avg_vol_20 = np.mean(volume[-20:])
        if current_ma20 is None or np.isnan(current_ma20):
            return False, None, None, None
        detachment_pct = ((current_price - current_ma20) / current_ma20) * 100
        if len(close) >= 25:
            recent_slope = (close[-1] - close[-5]) / 5
            prior_slope = (close[-5] - close[-25]) / 20
            angle_acceleration = recent_slope / prior_slope if prior_slope > 0 else float("inf")
        else:
            angle_acceleration = 1
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        is_parabolic = (
            detachment_pct > 40 and
            vol_ratio > 2.5 and
            angle_acceleration > 3
        )
        return is_parabolic, detachment_pct, vol_ratio, current_price
    except Exception:
        return False, None, None, None

def get_oi_tier(total_oi):
    """Pure tiering of the OI sitting at the max-pain strike. No network — the OI is
    already returned by get_max_pain_strike, which avoids a second option-chain fetch.
    > 100k contracts = HIGH conviction pin, > 25k = MEDIUM, else LOW (untradeable)."""
    if total_oi > 100000:
        return "HIGH"
    elif total_oi > 25000:
        return "MEDIUM"
    return "LOW"

# ============================================================
# INTRADAY DATA SOURCE (for VWAP Reclaim + Opening Range Breakout)
# Swap INTRADAY_SOURCE to "polygon" once on a paid Polygon plan that
# includes intraday aggregates. The polygon path below is implemented
# and validated against historical bars; it is just gated behind the
# flag so a future switch is a one-line change.
# ============================================================
INTRADAY_SOURCE = "yfinance"          # "yfinance" (free, current) | "polygon" (paid plan later)
ORB_RANGE_MINUTES = 15                 # opening range = high/low of the first N minutes
VWAP_OPENING_WINDOW_MINUTES = 30       # look for the dip-below-VWAP within the first N minutes
INTRADAY_VOL_CONFIRM = 1.5             # reclaim / breakout bar volume must exceed this x avg minute volume
# --- ORB tightening (added 2026-06-29: a strong gap-up day was flagging ~every name) ---
ORB_MIN_RANGE_PCT = 0.15               # opening range must span >= this % of price (skip microscopic ranges)
ORB_BREAK_BUFFER_FRAC = 0.25           # close must clear the OR high/low by >= this fraction of the OR height
                                       # (raised 0.10->0.25 on 2026-07-01: a $0.05 poke past a tight range was
                                       #  firing on ~every SPY session — see backtest replay)
ORB_ENTRY_WINDOW_MINUTES = 90          # a breakout must fire within the first N min of the session (opening-range
                                       # breakouts are a MORNING setup; a 1pm break of the 9:30 range isn't the thesis)
ORB_REQUIRE_HOLD = True                # breakout invalid if price has fallen back inside the range by the latest bar
ORB_MAX_RESULTS = 15                    # display only the top-N breakouts (ranked by volume conviction)
# --- VWAP-reclaim regime/timing gates ---
VWAP_MAX_VIX = 25.0                    # skip VWAP-reclaim signals when VIX is above this (high-fear regime)
# (Reclaim timing is enforced directly in _vwap_reclaim_row: dip AND reclaim must both be inside the
#  first VWAP_OPENING_WINDOW_MINUTES, i.e. 9:30-10:00 ET. No separate cutoff-hour constant.)
# ET_ZONE (DST-aware) is defined once near the top of the file.

def _normalize_intraday(df):
    """Return OHLCV (lowercase cols), most recent session only, regular hours, ET index. Empty df if unusable."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    need = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()
    df = df[need].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_ZONE)
    else:
        df.index = df.index.tz_convert(ET_ZONE)
    df = df.sort_index()
    last_date = df.index[-1].date()
    df = df[[ts.date() == last_date for ts in df.index]]
    df = df.between_time("09:30", "16:00")
    return df

def _intraday_yfinance(ticker):
    try:
        stock = _yf_ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        return _normalize_intraday(df)
    except Exception:
        return pd.DataFrame()

def _intraday_polygon(ticker):
    """Polygon 1-minute aggregates. Requires a paid plan that includes intraday timeframes."""
    try:
        day = datetime.now(ET_ZONE).date()
        for _ in range(5):
            while day.weekday() >= 5:
                day -= timedelta(days=1)
            ds = day.strftime("%Y-%m-%d")
            url = POLYGON_BASE_URL + "/v2/aggs/ticker/" + ticker + "/range/1/minute/" + ds + "/" + ds
            r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}, timeout=15)
            j = r.json()
            if j.get("resultsCount"):
                rows = j["results"]
                idx = pd.to_datetime([b["t"] for b in rows], unit="ms", utc=True).tz_convert(ET_ZONE)
                df = pd.DataFrame({
                    "open": [b["o"] for b in rows],
                    "high": [b["h"] for b in rows],
                    "low": [b["l"] for b in rows],
                    "close": [b["c"] for b in rows],
                    "volume": [b["v"] for b in rows],
                }, index=idx)
                return _normalize_intraday(df)
            day -= timedelta(days=1)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_intraday_minute_bars(ticker):
    """Return normalized 1-min bars for the latest session from the configured source.
    Cached 60s: yfinance bars are ~15-min delayed anyway, so a 1-min cache costs no
    freshness but spares ~49 fetches on incidental Streamlit reruns."""
    if INTRADAY_SOURCE == "polygon":
        df = _intraday_polygon(ticker)
        if df.empty:
            df = _intraday_yfinance(ticker)   # graceful fallback if plan blocks the timeframe
    else:
        df = _intraday_yfinance(ticker)
    # Only ever present TODAY's session. period="1d" returns Friday's bars on a weekend, and the
    # Polygon path walks back to the last day WITH data — either would make the VWAP/ORB scanners
    # compute on stale bars and label them as live. If the latest bar isn't today (ET), show nothing.
    if not df.empty and df.index[-1].date() != datetime.now(ET_ZONE).date():
        return pd.DataFrame()
    return df

def compute_session_vwap(df):
    """Running session VWAP aligned to df index (typical price * volume, cumulative)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum()
    cum_pv = (tp * df["volume"]).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)

def _vwap_reclaim_row(ticker, df, vix=None):
    """Return a result dict only for a genuine EARLY-SESSION dip-then-reclaim sequence, else None.
    The signal is NOT "price is currently above VWAP" — it requires, in order:
      1. price traded BELOW VWAP within the opening window, 9:30-10:00 ET (a real dip),
      2. a later candle CLOSED back above VWAP — the reclaim — ALSO within 9:30-10:00 ET,
      3. price is STILL holding above VWAP on the latest bar,
      4. the reclaim candle's volume beat the running average,
    plus a VIX regime gate. A reclaim after 10:00 ET is rejected (returns None) — a late
    reclaim like 10:59 or 11:35 is not this setup."""
    # Gate A: high-fear regime — mean-reversion-style intraday setups are unreliable when VIX is elevated.
    if isinstance(vix, (int, float)) and vix > VWAP_MAX_VIX:
        return None
    vwap = compute_session_vwap(df)
    # Anchor the opening window to the ACTUAL 9:30 ET open (robust to a missing first bar) rather
    # than df.index[0]. BOTH the dip AND the reclaim must fall inside this window (9:30-10:00 ET).
    _sd = df.index[0].date()
    market_open = pd.Timestamp(_sd.year, _sd.month, _sd.day, 9, 30, tz=df.index.tz)
    opening_cutoff = market_open + timedelta(minutes=VWAP_OPENING_WINDOW_MINUTES)   # 10:00 ET
    opening_mask = df.index < opening_cutoff
    below = df["close"] < vwap
    # Step 1: price must have actually traded below VWAP during the opening window (the dip).
    first_dip_pos = None
    for i in range(len(df)):
        if opening_mask[i] and below.iloc[i]:
            first_dip_pos = i
            break
    if first_dip_pos is None:
        return None
    # Step 2: the FIRST candle after the dip that CLOSES back above VWAP is the reclaim candle.
    reclaim_pos = None
    for i in range(first_dip_pos + 1, len(df)):
        if df["close"].iloc[i] > vwap.iloc[i]:
            reclaim_pos = i
            break
    if reclaim_pos is None:
        return None
    # Gate B: the RECLAIM candle itself must fall INSIDE the opening window (9:30-10:00 ET). A dip
    # early followed by a reclaim at 10:59 or 11:35 is NOT the setup — only genuine early-session
    # reclaims qualify. (Stricter than the old 1:00 PM cutoff, which let late reclaims through — the
    # bug that flagged ASML @ 11:35 and KLAC @ 10:59 as valid LONGs.)
    if df.index[reclaim_pos] >= opening_cutoff:
        return None
    # Step 3: must still be holding above VWAP now (a reclaim that already failed back below is dead).
    if df["close"].iloc[-1] <= vwap.iloc[-1]:
        return None
    avg_vol = df["volume"].iloc[:reclaim_pos + 1].mean()
    reclaim_vol = df["volume"].iloc[reclaim_pos]
    vol_ratio = reclaim_vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio < INTRADAY_VOL_CONFIRM:
        return None
    return {
        "Ticker": ticker,
        "Current Price": "$" + str(round(df["close"].iloc[-1], 2)),
        "VWAP": "$" + str(round(vwap.iloc[-1], 2)),
        "Reclaim Time": df.index[reclaim_pos].strftime("%H:%M ET"),
        "Reclaim Vol vs Avg": str(round(vol_ratio, 1)) + "x",
        "Setup": "LONG (VWAP reclaim)",
        "_price": df["close"].iloc[-1],
        "_is_long": True,
        "_level": vwap.iloc[-1],
        "_signal_type": "vwap",
        "_magnet_strike": None,
        "_oi_tier": None,
        "_distance": None,
        "_vol_ratio": vol_ratio,
        "_trigger_dt": df.index[reclaim_pos],
    }

def _orb_row(ticker, df):
    """Return a result dict if ticker broke out of the opening range with conviction, else None.
    Tightened so a strong trend day doesn't flag every name: (1) the opening range must be
    non-trivial, (2) the breakout bar must CLOSE clear of the range by a buffer, on volume
    above a causal (no-lookahead) running average, and (3) price must STILL be holding the
    breakout on the latest bar — a poke that failed back into the range doesn't count."""
    session_start = df.index[0]
    or_cutoff = session_start + timedelta(minutes=ORB_RANGE_MINUTES)
    or_mask = df.index < or_cutoff
    or_bars = df[or_mask]
    post = df[~or_mask]
    if or_bars.empty or post.empty:
        return None
    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    or_range = or_high - or_low
    # (1) Skip a microscopic opening range — on gap days a tiny range makes everything "break out".
    if or_high <= 0 or (or_range / or_high) * 100 < ORB_MIN_RANGE_PCT:
        return None
    buffer = or_range * ORB_BREAK_BUFFER_FRAC
    n_or = len(or_bars)
    last_close = df["close"].iloc[-1]
    entry_cutoff = session_start + timedelta(minutes=ORB_ENTRY_WINDOW_MINUTES)
    breakout = None
    for i in range(len(post)):
        if post.index[i] >= entry_cutoff:
            break   # past the morning entry window — an opening-range break this late isn't the setup
        c = post["close"].iloc[i]
        v = post["volume"].iloc[i]
        # (2) volume vs a causal running average (bars up to & including this one — no lookahead)
        avg_vol = df["volume"].iloc[:n_or + i + 1].mean()
        if avg_vol <= 0 or v <= avg_vol * INTRADAY_VOL_CONFIRM:
            continue
        if c > or_high + buffer:
            # (3) must still be holding above the range now, not a failed poke
            if ORB_REQUIRE_HOLD and last_close <= or_high:
                continue
            breakout = ("UP (breakout)", or_high, post.index[i], v / avg_vol)
            break
        if c < or_low - buffer:
            if ORB_REQUIRE_HOLD and last_close >= or_low:
                continue
            breakout = ("DOWN (breakdown)", or_low, post.index[i], v / avg_vol)
            break
    if breakout is None:
        return None
    direction, level, btime, vr = breakout
    return {
        "Ticker": ticker,
        "Direction": direction,
        "OR High": "$" + str(round(or_high, 2)),
        "OR Low": "$" + str(round(or_low, 2)),
        "Breakout Level": "$" + str(round(level, 2)),
        "Current Price": "$" + str(round(df["close"].iloc[-1], 2)),
        "Breakout Time": btime.strftime("%H:%M ET"),
        "Vol vs Avg": str(round(vr, 1)) + "x",
        "_price": df["close"].iloc[-1],
        "_is_long": direction.startswith("UP"),
        "_level": level,
        "_signal_type": "orb",
        "_magnet_strike": None,
        "_oi_tier": None,
        "_distance": None,
        "_vol_ratio": vr,
        "_trigger_dt": btime,
    }

def scan_intraday_setups(tickers, vix=None):
    """One pass over the watchlist computing both VWAP reclaim and ORB. Returns (vwap_rows, orb_rows, tickers_with_data).
    Minute-bar fetches run concurrently; vix is forwarded to the VWAP-reclaim gate so high-fear sessions suppress those signals.
    ORB rows are ranked by volume conviction (strongest first) so the display can show a shortlist."""
    tickers = list(tickers)
    dfs = _parallel_map(get_intraday_minute_bars, tickers)   # concurrent fetch
    vwap_rows = []
    orb_rows = []
    tickers_with_data = 0
    for ticker, df in zip(tickers, dfs):
        if df is None or df.empty or len(df) < 5:
            continue
        tickers_with_data += 1
        if len(df) >= VWAP_OPENING_WINDOW_MINUTES + 2:
            row = _vwap_reclaim_row(ticker, df, vix=vix)
            if row:
                vwap_rows.append(row)
        if len(df) > ORB_RANGE_MINUTES + 1:
            row = _orb_row(ticker, df)
            if row:
                orb_rows.append(row)
    # Rank ORB breakouts by volume conviction (the "N.Nx" field) so the strongest float to the top.
    orb_rows.sort(key=lambda r: float(str(r.get("Vol vs Avg", "0x")).rstrip("x")), reverse=True)
    return vwap_rows, orb_rows, tickers_with_data

# ============================================================
# PRE-MARKET CONTEXT ENGINE (Level 2) + MARKET STATUS
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def get_market_context():
    context = {}
    try:
        vix = _yf_ticker("^VIX")
        vix_hist = vix.history(period="2d")
        if not vix_hist.empty and len(vix_hist) >= 1:
            current_vix = vix_hist["Close"].iloc[-1]
            prev_vix = vix_hist["Close"].iloc[0] if len(vix_hist) > 1 else current_vix
            vix_change = ((current_vix - prev_vix) / prev_vix * 100) if prev_vix != 0 else 0
            context["vix"] = round(current_vix, 2)
            context["vix_change"] = round(vix_change, 2)
        else:
            context["vix"] = "N/A"
            context["vix_change"] = "N/A"

        spy = _yf_ticker("SPY")
        spy_hist = spy.history(period="5d")
        if not spy_hist.empty and len(spy_hist) >= 2:
            closes = spy_hist["Close"].dropna()
            if len(closes) >= 2:
                prev_close = closes.iloc[-2]
                last_price = closes.iloc[-1]
                spy_change = ((last_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
                context["spy_change"] = round(spy_change, 2)
            else:
                context["spy_change"] = "N/A"
        else:
            context["spy_change"] = "N/A"
    except Exception:
        context["vix"] = "N/A"
        context["vix_change"] = "N/A"
        context["spy_change"] = "N/A"
    return context

def get_market_bias(context):
    bias = "NEUTRAL"
    reasons = []
    try:
        vix = context.get("vix", "N/A")
        spy_change = context.get("spy_change", "N/A")
        if isinstance(vix, (int, float)):
            if vix > 30:
                bias = "HOSTILE"
                reasons.append("VIX above 30 - high fear, magnet pins unreliable")
            elif vix > 25:
                bias = "NEUTRAL"
                reasons.append("VIX elevated at " + str(vix) + " - proceed with caution")
            elif vix < 15:
                bias = "FAVORABLE"
                reasons.append("VIX low at " + str(vix) + " - calm markets favor magnet pins")
            else:
                reasons.append("VIX normal at " + str(vix))
        if isinstance(spy_change, (int, float)):
            if abs(spy_change) > 1:
                bias = "HOSTILE"
                reasons.append("SPY moved " + str(spy_change) + "% - large overnight moves weaken the pin")
            elif abs(spy_change) > 0.5:
                if bias == "FAVORABLE":
                    bias = "NEUTRAL"
                reasons.append("SPY moved " + str(spy_change) + "% - moderate overnight move")
    except Exception:
        bias = "NEUTRAL"
        reasons.append("Could not assess market - proceed manually")
    return bias, reasons

# ============================================================
# TRADE MECHANICS (Phase 1) — per-signal suggested strike, Black-Scholes delta,
# stop, and profit target. Also fetches a nearest-expiry option-chain snapshot
# that Phase 3 (IV / IV-Rank) reuses. Every helper is defensive: any failure
# yields "N/A" rather than breaking a scan.
# ============================================================
RISK_FREE_RATE = 0.045       # annual risk-free rate (per spec)
STOP_BUFFER = 0.0075         # FALLBACK flat stop offset (used only when IV/DTE unavailable)
MAGNET_TARGET_OFFSET = 0.30  # exit ~$0.30 before the max-pain pin ($0.20-$0.50 band)
# DTE-aware stop/target sizing: distances scale with the option's expected 1-sigma move,
# E = price * ATM_IV * sqrt(DTE_years). A 2DTE trade gets a tight stop/target; a 30DTE a wide one.
STOP_K = 0.5                 # stop at 0.5 x expected move (adverse)
TARGET_K = 1.0               # target at 1.0 x expected move (favorable)
_NORM = NormalDist()

@st.cache_data(ttl=60, show_spinner=False)
def get_futures_data():
    """Latest daily quote per FUTURES_WATCHLIST contract. Uses ONE batched Yahoo download (fewer
    requests = far less rate-limit risk than 9 individual calls) through the browser session, then
    retries individually for any contract the batch drops. Returns
    {short: {price, prev_close, change_pct, trend}} — value is None per contract on failure."""
    out = {short: None for short in FUTURES_WATCHLIST}
    tickers = list(FUTURES_WATCHLIST.values())

    def _parse(df):
        if df is None or df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        closes = close.dropna()
        if len(closes) < 2:
            return None
        prev_close, last = float(closes.iloc[-2]), float(closes.iloc[-1])
        change_pct = ((last - prev_close) / prev_close) * 100 if prev_close else 0.0
        return {"price": last, "prev_close": prev_close, "change_pct": change_pct,
                "trend": "UP" if last >= prev_close else "DOWN"}

    try:
        raw = yf.download(tickers, period="5d", interval="1d", group_by="ticker",
                          progress=False, auto_adjust=False, threads=True, session=_YF_SESSION)
    except Exception:
        raw = None
    cols0 = []
    if raw is not None and hasattr(raw, "columns"):
        try:
            cols0 = list(raw.columns.get_level_values(0))
        except Exception:
            cols0 = list(raw.columns)
    for short, tk in FUTURES_WATCHLIST.items():
        if tk in cols0:
            try:
                out[short] = _parse(raw[tk])
            except Exception:
                out[short] = None
    for short, tk in FUTURES_WATCHLIST.items():        # individual retry for anything the batch dropped
        if out[short] is None:
            try:
                out[short] = _parse(yf.download(tk, period="5d", interval="1d",
                                                progress=False, session=_YF_SESSION))
            except Exception:
                out[short] = None
    return out

def get_futures_regime(futures_data, vix=None):
    """Classify field conditions from futures + VIX. Returns (regime, risks) where regime is
    GREEN / AMBER / RED and risks is a list of the specific reasons that drove it."""
    risks = []
    regime = "GREEN"

    # --- VIX gate ---
    if vix is not None and vix >= FUTURES_RISK_OFF["VIX"]:
        risks.append(f"VIX {vix:.1f} >= {FUTURES_RISK_OFF['VIX']:.1f}")
        regime = "RED"

    # --- Equity futures stress ---
    es = futures_data.get("ES")
    nq = futures_data.get("NQ")
    if es and es.get("change_pct", 0) <= FUTURES_RISK_OFF["ES"]:
        risks.append(f"ES {es['change_pct']:.2f}% <= {FUTURES_RISK_OFF['ES']}%")
        regime = "RED"
    if nq and nq.get("change_pct", 0) <= FUTURES_RISK_OFF["NQ"]:
        risks.append(f"NQ {nq['change_pct']:.2f}% <= {FUTURES_RISK_OFF['NQ']}%")
        regime = "RED"

    # --- Crude oil shock ---
    cl = futures_data.get("CL")
    if cl and cl.get("change_pct", 0) >= FUTURES_RISK_OFF["CL"]:
        risks.append(f"Oil +{cl['change_pct']:.2f}% >= +{FUTURES_RISK_OFF['CL']}%")
        regime = "RED"

    # --- Defensive rotation (Treasuries + Gold strong while equity flat) ---
    zb = futures_data.get("ZB")
    zn = futures_data.get("ZN")
    gc = futures_data.get("GC")
    if regime != "RED" and es is not None and abs(es.get("change_pct", 0)) < 0.2:
        if (zb and zb.get("trend") == "UP") or (zn and zn.get("trend") == "UP"):
            if gc and gc.get("trend") == "UP":
                risks.append("Flight to safety: Treasuries + Gold UP while ES flat")
                regime = "AMBER"

    # --- Strong risk-on note (GREEN only) ---
    if regime == "GREEN":
        if es and es.get("change_pct", 0) >= FUTURES_RISK_ON["ES"]:
            if nq and nq.get("change_pct", 0) >= FUTURES_RISK_ON["NQ"]:
                risks.append("Risk-on: ES + NQ both up")
            else:
                risks.append("ES up, NQ lagging — mixed")

    return regime, risks

def get_option_chain_snapshot(ticker):
    """Nearest-expiry option chain for the computed columns (GEX / P/C / skew / strike+delta / IV).
    Now a thin view over the shared Polygon _polygon_chain (cached there, 120s) — same output keys as
    the old yfinance version, so every downstream consumer is unchanged. Returns None on failure."""
    ch = _polygon_chain(ticker)
    if not ch or not ch.get("strikes"):
        return None
    return {"expiry": ch["expiry"], "dte_years": ch["dte_years"], "strikes": ch["strikes"],
            "call_iv": ch["call_iv"], "put_iv": ch["put_iv"],
            "call_oi": ch["call_oi"], "put_oi": ch["put_oi"],
            "call_vol_total": ch["call_vol_total"], "put_vol_total": ch["put_vol_total"]}

def _bs_delta(S, K, T, sigma, r, is_call):
    """Black-Scholes delta. Returns None on degenerate inputs."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        nd1 = _NORM.cdf(d1)
        return nd1 if is_call else nd1 - 1.0
    except Exception:
        return None

# ============================================================
# GAMMA EXPOSURE (GEX) — Phase 1. Convention: calls POSITIVE, puts NEGATIVE, i.e. dealers are
# assumed net LONG call gamma / SHORT put gamma (the standard SqueezeMetrics-style dealer
# positioning). That assumption is exactly what makes POSITIVE GEX = price-dampening, which the
# magnet-pin thesis relies on; it is an assumption (dealer books aren't observable). Data (per
# strike IV + OI) rides the same get_option_chain_snapshot fetch — no extra network per ticker.
# ============================================================
def _bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma (identical for calls & puts): phi(d1) / (S*sigma*sqrt(T)).
    Returns None on degenerate inputs."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (log(S / K) + (RISK_FREE_RATE + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        return _NORM.pdf(d1) / (S * sigma * sqrt(T))
    except Exception:
        return None

def _net_gex_at(snap, S):
    """Net dealer GEX at a hypothetical spot S: sum over strikes of Gamma*OI*100*S, calls
    POSITIVE and puts NEGATIVE. Each strike keeps its OWN implied vol (we have no vol surface to
    reprice it at other spots — the standard simplification). Units: $ gamma per $1 spot move."""
    if S <= 0:
        return None
    T = snap.get("dte_years")
    if not T or T <= 0:
        return 0.0
    civ, coi = snap.get("call_iv") or {}, snap.get("call_oi") or {}
    piv, poi = snap.get("put_iv") or {}, snap.get("put_oi") or {}
    total = 0.0
    for k in snap.get("strikes") or []:
        if k in coi and k in civ:
            g = _bs_gamma(S, k, T, civ[k])
            if g is not None:
                total += g * coi[k] * 100.0 * S
        if k in poi and k in piv:
            g = _bs_gamma(S, k, T, piv[k])
            if g is not None:
                total -= g * poi[k] * 100.0 * S
    return total

def compute_gex(snap, spot):
    """Return {'net','flip','regime'} or None. net = net GEX at current spot; flip = the spot
    level where net GEX crosses zero (found by sweeping +/-20% around spot, linear-interpolated);
    regime = POSITIVE if spot is ABOVE the flip (dealers dampen -> magnet-pin thesis holds),
    NEGATIVE if BELOW (dealers amplify -> thesis weakened). Falls back to the sign of net GEX at
    spot when there is no zero-crossing within +/-20%."""
    if not snap or not spot or spot <= 0:
        return None
    if not (snap.get("call_oi") or snap.get("put_oi")):
        return None                                   # no open interest -> GEX not computable
    net = _net_gex_at(snap, spot)
    if net is None:
        return None
    lo, hi, steps = 0.80 * spot, 1.20 * spot, 80
    flip, prev_s, prev_g = None, None, None
    for i in range(steps + 1):
        s = lo + (hi - lo) * i / steps
        g = _net_gex_at(snap, s)
        if g is None:
            continue
        if prev_g is not None and prev_g != g and ((prev_g <= 0.0 <= g) or (prev_g >= 0.0 >= g)):
            flip = prev_s + (s - prev_s) * (0.0 - prev_g) / (g - prev_g)   # interpolate zero
            break
        prev_s, prev_g = s, g
    if flip is not None:
        regime = "POSITIVE" if spot > flip else "NEGATIVE"
    else:
        regime = "POSITIVE" if net > 0 else "NEGATIVE"
    return {"net": net, "flip": flip, "regime": regime}

def _fmt_gex(x):
    """Compact $ formatting with B/M suffix and explicit sign."""
    if x is None:
        return "N/A"
    a = abs(x)
    sign = "-" if x < 0 else "+"
    if a >= 1e9:
        return "%s$%.2fB" % (sign, a / 1e9)
    if a >= 1e6:
        return "%s$%.0fM" % (sign, a / 1e6)
    return "%s$%.0f" % (sign, a)

def color_gex_regime(val):
    """Cell style for the GEX Regime column — the negative-GEX 'flag' (red) on magnet rows."""
    v = str(val)
    if "POSITIVE" in v:
        return "background-color: " + C_GREEN_BG + "; color: " + C_GREEN + "; font-weight: 600"
    if "NEGATIVE" in v:
        return "background-color: " + C_RED_BG + "; color: " + C_RED + "; font-weight: 700"
    return ""

# ============================================================
# PUT/CALL RATIO (Phase 3) — total put volume / total call volume for the nearest expiry, from the
# same snapshot. EXTREME flags: >= PCR_EXTREME_HI = put-heavy (possible bullish contrarian);
# <= PCR_EXTREME_LO = call-heavy (possible bearish contrarian). yfinance option volume is same-day
# cumulative, so it is thin/zero before the open and firms up during the session.
# ============================================================
PCR_EXTREME_HI = 1.5
PCR_EXTREME_LO = 0.5
PCR_COLS = ["P/C Ratio", "P/C Flag"]
PCR_CAPTION = ("P/C Ratio = total put volume / total call volume (nearest expiry). EXTREME-PUT "
               ">=1.5 (heavy put buying -> possible bullish contrarian); EXTREME-CALL <=0.5 (heavy "
               "call buying -> possible bearish contrarian). yfinance option volume is same-day "
               "cumulative, so pre-open it reads 'N/A' and firms up during the session.")

def compute_pcr(snap):
    """(ratio, flag). flag = 'EXTREME-PUT' (ratio>=1.5), 'EXTREME-CALL' (ratio<=0.5), or ''.
    ratio is None when call volume is zero/unavailable (e.g. pre-market before volume prints)."""
    if not snap:
        return None, ""
    cv = snap.get("call_vol_total") or 0.0
    pv = snap.get("put_vol_total") or 0.0
    if cv <= 0:
        return None, ""
    ratio = pv / cv
    if ratio >= PCR_EXTREME_HI:
        return ratio, "EXTREME-PUT"
    if ratio <= PCR_EXTREME_LO:
        return ratio, "EXTREME-CALL"
    return ratio, ""

def color_pcr_flag(val):
    """Amber highlight on EXTREME put/call readings (contrarian-sentiment flag)."""
    if "EXTREME" in str(val):
        return "background-color: " + C_AMBER_BG + "; color: " + C_ACCENT + "; font-weight: 700"
    return ""

# ============================================================
# VOLATILITY SKEW (Phase 4) — 25-delta put IV minus 25-delta call IV, in vol points. Positive =
# puts richer than calls = downside hedging demand / fear (normal for equity indices). The 5-day
# TREND needs history no free source has, so — like IV — we LOG daily skew per ticker to
# skew_history.json and report steepening/flattening once >= SKEW_TREND_MIN_DAYS days are logged.
#   skew  = IV(nearest -0.25-delta put) - IV(nearest +0.25-delta call), x100 vol points
#   trend = today's skew - skew ~5 trading days ago  (STEEPENING = fear rising, early warning)
# ============================================================
SKEW_HISTORY_FILE = "skew_history.json"
SKEW_HISTORY_CAP = 260
SKEW_HISTORY_WINDOW_DAYS = 365
SKEW_TREND_MIN_DAYS = 5
SKEW_TREND_DEADBAND = 0.5          # vol points; |delta| below this reads "flat"
SKEW_COLS = ["Skew", "Skew Trend"]
SKEW_CAPTION = ("Skew = 25-delta put IV minus 25-delta call IV (vol points); positive = puts "
                "richer = downside hedging/fear. Skew Trend = today vs ~5 trading days ago from "
                "logged history: STEEPENING (fear rising — early warning, red) or flattening "
                "(green); reads 'building N/5d' until >=5 days of skew are logged.")

def compute_skew(snap, spot):
    """25d-put IV minus 25d-call IV in vol points, or None. Uses each strike's own IV to find the
    strike whose CALL delta is nearest +0.25 and whose PUT delta is nearest -0.25."""
    if not snap or not spot or spot <= 0:
        return None
    T = snap.get("dte_years")
    civ, piv = snap.get("call_iv") or {}, snap.get("put_iv") or {}
    if not T or T <= 0 or not civ or not piv:
        return None
    best_c = best_p = None
    ec = ep = 1e9
    for k, iv in civ.items():
        d = _bs_delta(spot, k, T, iv, RISK_FREE_RATE, True)
        if d is not None and abs(d - 0.25) < ec:
            ec, best_c = abs(d - 0.25), iv
    for k, iv in piv.items():
        d = _bs_delta(spot, k, T, iv, RISK_FREE_RATE, False)
        if d is not None and abs(d + 0.25) < ep:
            ep, best_p = abs(d + 0.25), iv
    if best_c is None or best_p is None:
        return None
    return (best_p - best_c) * 100.0

def load_skew_history():
    if os.path.exists(SKEW_HISTORY_FILE):
        try:
            with open(SKEW_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_skew_history(h):
    try:
        with open(SKEW_HISTORY_FILE, "w") as f:
            json.dump(h, f)
    except Exception:
        pass

def record_skew(ticker, skew):
    """Append (or update) today's skew for a ticker; deduped per day, capped ~1yr. Main thread."""
    if ticker is None or skew is None:
        return
    try:
        skew = float(skew)
    except Exception:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    h = load_skew_history()
    lst = h.get(ticker, [])
    for e in lst:
        if e.get("date") == today:
            e["skew"] = skew
            break
    else:
        lst.append({"date": today, "skew": skew})
    h[ticker] = lst[-SKEW_HISTORY_CAP:]
    save_skew_history(h)

# ------------------------------------------------------------
# MAGNET-PIN FORWARD TEST (Level 8) — historical option OI isn't available on the free tier, so
# max-pain pins can't be backtested retrospectively. Instead we LOG each day's pin prediction here
# and evaluate convergence AFTER each expiry passes (no lookahead). Consumed by
# alpha_backtest.magnet_forward_report(). Measures the UNDERLYING's move toward the pin, NOT option
# P&L (no theta/spread modeled) — same caveat as the ORB/VWAP backtest.
# ------------------------------------------------------------
MAGNET_HISTORY_FILE = "magnet_history.json"
MAGNET_HISTORY_CAP = 400          # snapshots kept per ticker (~1.5yr of daily entries)

def load_magnet_history():
    if os.path.exists(MAGNET_HISTORY_FILE):
        try:
            with open(MAGNET_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_magnet_history(h):
    try:
        with open(MAGNET_HISTORY_FILE, "w") as f:
            json.dump(h, f)
    except Exception:
        pass

def record_magnet(ticker, max_pain, spot, expiry):
    """Log today's magnet-pin prediction: the max-pain pin, the spot at log time, and the nearest
    expiry it targets. Deduped per day (last scan of the day wins), capped. Main thread. Evaluated
    later (once the expiry has passed) by alpha_backtest.magnet_forward_report — strictly no lookahead."""
    if ticker is None or max_pain is None or spot is None or expiry is None:
        return
    try:
        max_pain = float(max_pain); spot = float(spot)
    except Exception:
        return
    if max_pain <= 0 or spot <= 0:
        return
    exp = expiry.strftime("%Y-%m-%d") if hasattr(expiry, "strftime") else str(expiry)
    today = datetime.now().strftime("%Y-%m-%d")
    h = load_magnet_history()
    lst = h.get(ticker, [])
    for e in lst:
        if e.get("date") == today:
            e["spot"] = spot; e["max_pain"] = max_pain; e["expiry"] = exp
            break
    else:
        lst.append({"date": today, "spot": spot, "max_pain": max_pain, "expiry": exp})
    h[ticker] = lst[-MAGNET_HISTORY_CAP:]
    save_magnet_history(h)

def _skew_history_obs(ticker):
    lst = load_skew_history().get(ticker, [])
    cutoff = datetime.now().date() - timedelta(days=SKEW_HISTORY_WINDOW_DAYS)
    out = []
    for e in lst:
        try:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
            out.append((d, float(e["skew"]))) if d >= cutoff else None
        except Exception:
            pass
    return sorted(out)

def _skew_trend(ticker, cur):
    """(delta, label). delta = cur - skew ~5 trading days ago; label steepening/flattening/flat,
    or (None, 'building N/5d') while fewer than SKEW_TREND_MIN_DAYS days are logged."""
    if cur is None:
        return None, "n/a"
    obs = _skew_history_obs(ticker)
    if len(obs) < SKEW_TREND_MIN_DAYS:
        return None, "building %d/%dd" % (len(obs), SKEW_TREND_MIN_DAYS)
    today = datetime.now().date()
    target = today - timedelta(days=7)                     # ~5 trading days back
    past = [(d, s) for d, s in obs if d <= today - timedelta(days=3)]
    if not past:
        return None, "building"
    _, ref = min(past, key=lambda x: abs((x[0] - target).days))
    delta = cur - ref
    if delta > SKEW_TREND_DEADBAND:
        return delta, "steepening"
    if delta < -SKEW_TREND_DEADBAND:
        return delta, "flattening"
    return delta, "flat"

def color_skew_trend(val):
    """Steepening (fear rising) = red; flattening = green; flat/building = muted."""
    v = str(val)
    if "steepening" in v:
        return "background-color: " + C_RED_BG + "; color: " + C_RED + "; font-weight: 700"
    if "flattening" in v:
        return "background-color: " + C_GREEN_BG + "; color: " + C_GREEN + "; font-weight: 600"
    return "color: " + C_MUTED

GEX_COLS = ["Net GEX", "Gamma Flip", "GEX Regime"]
GEX_CAPTION = ("GEX = sum of Gamma*OI*100*spot (calls +, puts -; dealers assumed long-call / "
               "short-put gamma). POSITIVE regime = spot ABOVE the gamma-flip level -> dealers "
               "dampen moves (magnet-pin thesis holds); NEGATIVE (red) = spot BELOW flip -> dealers "
               "amplify -> magnet thesis is weakened on that row (flag only; strength/filtering "
               "unchanged). Net GEX is approx $ gamma per $1 spot move; Gamma Flip = zero-gamma level.")

def _nearest_otm_strike(strikes, price, is_long):
    """ATM-or-one-strike-OTM: the nearest strike that is not in-the-money for the
    trade's direction (>= price for a call/long, <= price for a put/short)."""
    if not strikes or price is None:
        return None
    if is_long:
        above = [s for s in strikes if s >= price]
        return min(above) if above else max(strikes)
    below = [s for s in strikes if s <= price]
    return max(below) if below else min(strikes)

def _iv_at(snap, strike, is_long):
    m = snap.get("call_iv" if is_long else "put_iv", {}) or {}
    if not m:
        return None
    if strike in m:
        return m[strike]
    return m[min(m.keys(), key=lambda x: abs(x - strike))]

def _atm_iv(snap, price):
    """Direction-neutral at-the-money IV (mean of nearest call & put IV). Used as a
    stable reference for IV-Rank history so the series isn't polluted by the trade's
    call/put side flipping day to day."""
    if price is None:
        return None
    ivs = []
    for m in (snap.get("call_iv") or {}, snap.get("put_iv") or {}):
        if m:
            k = min(m.keys(), key=lambda x: abs(x - price))
            ivs.append(m[k])
    return sum(ivs) / len(ivs) if ivs else None

def _fmt_strike(s):
    if s is None:
        return "N/A"
    return "$" + (str(int(s)) if float(s).is_integer() else str(round(s, 2)))

def compute_trade_mechanics(ticker, price, is_long, signal_type, level, magnet_strike=None):
    """Phase-1 computed columns for one signal. Stop/target are DTE-AWARE: sized to the option's
    expected 1-sigma move E = price * ATM_IV * sqrt(DTE_years) -- stop at STOP_K*E (adverse), target
    at TARGET_K*E (favorable) -- so a 2DTE trade gets a tight stop/target and a 30DTE a wide one,
    instead of a flat %. Magnet targets the pin and flags '>1sigma' when the pin is more than one
    expected move away (unlikely to be reached by expiry). Falls back to the flat STOP_BUFFER when
    IV/DTE are unavailable. Returns display strings + private fields (_iv/_sugg_strike/_atm_iv).
    Never raises."""
    out = {"Suggested Strike": "N/A", "Est. Delta": "N/A",
           "Stop Price": "N/A", "Profit Target": "N/A",
           "_iv": None, "_sugg_strike": None, "_atm_iv": None}
    try:
        snap = get_option_chain_snapshot(ticker)
        atm_iv = _atm_iv(snap, price) if snap else None
        dte_years = snap.get("dte_years") if snap else None
        # Expected 1-sigma underlying move ($) over the option's life; None if IV/DTE unavailable.
        E = (price * atm_iv * sqrt(dte_years)) if (atm_iv and dte_years and price and price > 0
                                                   and dte_years > 0) else None

        if isinstance(level, (int, float)) and level > 0:
            if E is not None:
                stop = level - STOP_K * E if is_long else level + STOP_K * E     # DTE-aware
            else:
                stop = level * (1 - STOP_BUFFER) if is_long else level * (1 + STOP_BUFFER)  # fallback
            out["Stop Price"] = "$" + str(round(stop, 2))

        if signal_type == "magnet" and isinstance(magnet_strike, (int, float)):
            tgt = magnet_strike - MAGNET_TARGET_OFFSET if is_long else magnet_strike + MAGNET_TARGET_OFFSET
            far = E is not None and abs(magnet_strike - price) > E    # pin more than one expected move away
            out["Profit Target"] = "$" + str(round(tgt, 2)) + (" (>1sd)" if far else "")   # (>1sd)=pin >1 expected move away, unlikely by expiry
        elif isinstance(level, (int, float)) and level > 0 and E is not None:
            tgt = level + TARGET_K * E if is_long else level - TARGET_K * E      # DTE-aware ~1σ target
            out["Profit Target"] = "$" + str(round(tgt, 2))
        else:
            out["Profit Target"] = "Stop only"   # no IV/DTE and not a magnet -> momentum exit only

        if snap:
            strike = _nearest_otm_strike(snap["strikes"], price, is_long)
            out["_sugg_strike"] = strike
            out["Suggested Strike"] = _fmt_strike(strike)
            iv = _iv_at(snap, strike, is_long) if strike is not None else None
            out["_iv"] = iv
            out["_atm_iv"] = atm_iv
            if strike is not None and iv:
                d = _bs_delta(price, strike, snap["dte_years"], iv, RISK_FREE_RATE, is_long)
                if d is not None:
                    out["Est. Delta"] = round(d, 2)
    except Exception:
        pass
    return out

PHASE1_COLS = ["Suggested Strike", "Est. Delta", "Stop Price", "Profit Target"]
PHASE2_COLS = ["Trade Readiness", "Readiness Note"]

# Trade-readiness gates I already apply manually (Phase 2). VIX and signal-age are
# global; the OI / distance / volume gates apply only to the noted signal types.
READINESS_MAX_VIX = 25.0
READINESS_MAX_AGE_MIN = 30
READINESS_MIN_VOL = 1.5
READINESS_MAX_DIST_PCT = 1.0

def compute_readiness(sig_type, oi_tier, distance, vol_ratio, vix, trigger_dt, now_et):
    """Return (verdict, note). PASS only if every applicable gate holds; otherwise
    FAIL with a semicolon-list of the specific gates that failed."""
    fails = []
    if sig_type == "magnet":
        if oi_tier not in ("HIGH", "MEDIUM"):
            fails.append("OI not HIGH/MED")
        if distance is None or abs(distance) > READINESS_MAX_DIST_PCT:
            fails.append("dist >1% from pin")
    if sig_type in ("orb", "vwap"):
        if vol_ratio is None or vol_ratio < READINESS_MIN_VOL:
            fails.append("vol <1.5x avg")
    # global gate: VIX must be confirmably calm (unknown VIX fails safe)
    if isinstance(vix, (int, float)):
        if vix >= READINESS_MAX_VIX:
            fails.append("VIX >=25")
    else:
        fails.append("VIX unknown")
    # global gate: signal freshness (< 30 min old). Magnet pins are recomputed each
    # scan so their trigger time is ~now and this passes; intraday uses the real
    # breakout / reclaim time.
    age_min = None
    try:
        if trigger_dt is not None and now_et is not None:
            age_min = (now_et - trigger_dt).total_seconds() / 60.0
    except Exception:
        age_min = None
    if age_min is None:
        fails.append("age unknown")
    elif age_min >= READINESS_MAX_AGE_MIN:
        fails.append(str(int(age_min)) + "m old (>30m)")
    return ("PASS", "All checks passed") if not fails else ("FAIL", "; ".join(fails))

def color_readiness(val):
    """Cell style for the Trade Readiness column. The C_* theme tokens it references
    exist by the time any table renders."""
    v = str(val)
    if v == "PASS":
        return "background-color: " + C_GREEN_BG + "; color: " + C_GREEN + "; font-weight: 700"
    if v == "FAIL":
        return "background-color: " + C_RED_BG + "; color: " + C_RED + "; font-weight: 600"
    return ""

# ------------------------------------------------------------
# IV RANK & IV PERCENTILE (Phase 2 of the analytics expansion). No free source of historical IV
# exists (Polygon options = not entitled; yfinance = today's snapshot only), so two tracks run in
# parallel: (1) we LOG today's ATM IV per ticker to iv_history.json every scan, capped to ~a year,
# so real 52-week Rank/Percentile become genuinely accurate over time; (2) until at least
# IV_HISTORY_MIN_REAL trading days are logged we show a realized-vol PROXY (marked '~', labelled
# "approx." in the caption) — NOT true IV Rank. Once a ticker has >= the minimum it switches to its
# real logged history (window grows toward 52 weeks).
#   IV Rank       = (IV_now - min) / (max - min) * 100          over the window
#   IV Percentile = 100 * (# window observations below IV_now) / (window size)
# Proxy uses the same two formulas but ranks IV_now against a rolling realized-vol distribution
# (IV usually runs above realized vol, so proxy readings skew HIGH — noted in the caption).
# ------------------------------------------------------------
PHASE3_COLS = ["IV", "IV Rank", "IV Pctl"]
IV_HISTORY_FILE = "iv_history.json"
IV_HISTORY_CAP = 260            # ~a year of trading days (52 weeks ~= 252) + a little buffer
IV_HISTORY_WINDOW_DAYS = 365    # calendar cutoff for the "past year" window
IV_HISTORY_MIN_REAL = 60        # min logged trading days before real Rank/Pctl replace the proxy
IV_CAPTION = ("IV = implied vol of the suggested-strike option. IV Rank & IV Pctl: a trailing '~' "
              "means REALIZED-VOL PROXY (approx.) — NOT true IV Rank — shown until >=60 days of "
              "daily IV are logged, then they switch to real logged IV history (window grows toward "
              "52 weeks). The proxy ranks IV vs realized vol, which usually runs lower, so proxy "
              "readings skew high.")

def load_iv_history():
    if os.path.exists(IV_HISTORY_FILE):
        try:
            with open(IV_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_iv_history(h):
    try:
        with open(IV_HISTORY_FILE, "w") as f:
            json.dump(h, f)
    except Exception:
        pass

def record_iv(ticker, iv):
    """Append (or update) today's ATM IV observation for a ticker. Deduped per day
    so repeated reruns don't inflate the series; capped to the most recent entries."""
    if ticker is None or iv is None:
        return
    try:
        iv = float(iv)
    except Exception:
        return
    if iv <= 0:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    h = load_iv_history()
    lst = h.get(ticker, [])
    for e in lst:
        if e.get("date") == today:
            e["iv"] = iv
            break
    else:
        lst.append({"date": today, "iv": iv})
    h[ticker] = lst[-IV_HISTORY_CAP:]
    save_iv_history(h)

def _iv_history_obs(ticker):
    """Sorted [(date, iv)] for a ticker within the past IV_HISTORY_WINDOW_DAYS (one obs/day)."""
    lst = load_iv_history().get(ticker, [])
    cutoff = datetime.now().date() - timedelta(days=IV_HISTORY_WINDOW_DAYS)
    out = []
    for e in lst:
        try:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
            iv = float(e["iv"])
            if d >= cutoff and iv > 0:
                out.append((d, iv))
        except Exception:
            pass
    return sorted(out)

def _iv_stats_from_history(ticker, cur):
    """(rank, percentile) from REAL logged IV history, or None if fewer than IV_HISTORY_MIN_REAL
    trading days are logged. Rank = min/max position; Percentile = % of days strictly below."""
    if cur is None:
        return None
    try:
        cur = float(cur)
    except Exception:
        return None
    vals = [iv for _, iv in _iv_history_obs(ticker)]
    if len(vals) < IV_HISTORY_MIN_REAL:
        return None
    lo, hi = min(vals), max(vals)
    rank = 50.0 if hi <= lo else max(0.0, min(100.0, (cur - lo) / (hi - lo) * 100.0))
    pct = 100.0 * sum(1 for v in vals if v < cur) / len(vals)
    return rank, pct

@st.cache_data(ttl=3600, show_spinner=False)
def realized_vol_series(ticker):
    """Sorted rolling-21-day annualized realized-vol values over ~the last year — the distribution
    the PROXY ranks current IV against while real IV history is still building. [] on failure."""
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, session=_YF_SESSION)
        if df is None or df.empty or len(df) < 30:
            return []
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rets = np.log(close / close.shift(1)).dropna()
        roll = (rets.rolling(21).std() * np.sqrt(252)).dropna()
        if len(roll) < 10:
            return []
        return sorted(float(x) for x in roll.tail(252))
    except Exception:
        return []

def compute_iv_metrics(ticker, atm_iv):
    """Returns (iv_rank, iv_percentile, source). source = 'real' (>= IV_HISTORY_MIN_REAL days of
    logged IV), 'proxy' (realized-vol distribution stand-in), or None. Rank/Pctl are 0-100 ints;
    None when nothing is computable."""
    stats = _iv_stats_from_history(ticker, atm_iv)
    if stats is not None:
        return int(round(stats[0])), int(round(stats[1])), "real"
    if atm_iv is None:
        return None, None, None
    series = realized_vol_series(ticker)
    if series:
        try:
            cur = float(atm_iv)
            lo, hi = series[0], series[-1]
            rank = 50.0 if hi <= lo else max(0.0, min(100.0, (cur - lo) / (hi - lo) * 100.0))
            pct = 100.0 * sum(1 for v in series if v < cur) / len(series)
            return int(round(rank)), int(round(pct)), "proxy"
        except Exception:
            return None, None, None
    return None, None, None

def iv_days_logged():
    """Most trading days of IV logged for any single ticker — drives the switchover status line."""
    try:
        return max((len(v) for v in load_iv_history().values()), default=0)
    except Exception:
        return 0

def attach_computed_columns(df, vix=None, with_gex=False):
    """Add computed columns (Phases 1-3, plus GEX when with_gex=True) to a signal table using the
    private underscore fields each row carries, then strip those private fields. GEX is
    magnet-only (with_gex=True) so the ORB / VWAP tables are unchanged. Safe on empty/None input."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    now_et = datetime.now(ET_ZONE)
    out_cols = PHASE1_COLS + PHASE2_COLS + PHASE3_COLS + (GEX_COLS + PCR_COLS + SKEW_COLS if with_gex else [])
    cols = {c: [] for c in out_cols}
    for _, r in df.iterrows():
        ticker = r.get("Ticker")
        m = compute_trade_mechanics(
            ticker, r.get("_price"), bool(r.get("_is_long")),
            r.get("_signal_type"), r.get("_level"), magnet_strike=r.get("_magnet_strike"))
        for c in PHASE1_COLS:
            cols[c].append(m[c])
        verdict, note = compute_readiness(
            r.get("_signal_type"), r.get("_oi_tier"), r.get("_distance"),
            r.get("_vol_ratio"), vix, r.get("_trigger_dt"), now_et)
        cols["Trade Readiness"].append(verdict)
        cols["Readiness Note"].append(note)
        # Phase 2: IV column = IV of the actual traded (nearest-expiry) strike. IV Rank/Percentile
        # rank the ~30-day CONSTANT-MATURITY ATM IV against the ~30-day backfilled+logged history, so
        # the comparison is apples-to-apples on tenor (not a 0-7 DTE weekly vs a monthly series).
        # Daily IV logging happens once per ticker in the magnet loop (full watchlist, main thread) —
        # NOT here, which would double-log and race across the parallel tables.
        atm = _atm_iv_30d(ticker)
        cols["IV"].append(str(round(m["_iv"] * 100, 1)) + "%" if m.get("_iv") else "N/A")
        rank, pct, src = compute_iv_metrics(ticker, atm)
        mark = "~" if src == "proxy" else ""
        cols["IV Rank"].append("N/A" if rank is None else str(rank) + mark)
        cols["IV Pctl"].append("N/A" if pct is None else str(pct) + mark)
        if with_gex:   # GEX + P/C ride the cached chain snapshot compute_trade_mechanics fetched
            snap = get_option_chain_snapshot(ticker)
            gx = compute_gex(snap, r.get("_price"))
            if gx:
                cols["Net GEX"].append(_fmt_gex(gx["net"]))
                cols["Gamma Flip"].append(_fmt_strike(gx["flip"]) if gx["flip"] is not None else ">+/-20%")
                cols["GEX Regime"].append(gx["regime"])
            else:
                cols["Net GEX"].append("N/A")
                cols["Gamma Flip"].append("N/A")
                cols["GEX Regime"].append("N/A")
            ratio, pflag = compute_pcr(snap)
            cols["P/C Ratio"].append("%.2f" % ratio if ratio is not None else "N/A")
            cols["P/C Flag"].append(pflag if pflag else "—")
            # Skew uses the _skew raw field (computed once in get_max_pain_strike) + logged history
            sk = r.get("_skew")
            sdelta, slabel = _skew_trend(ticker, sk)
            cols["Skew"].append(("%+.1f" % sk) if sk is not None else "N/A")
            cols["Skew Trend"].append(("%+.1f %s" % (sdelta, slabel)) if sdelta is not None else slabel)
    for c in out_cols:
        df[c] = cols[c]
    drop = [c for c in df.columns if isinstance(c, str) and c.startswith("_")]
    return df.drop(columns=drop)

# ============================================================
# PRE-MARKET CONTEXT (Level 6) — overnight gaps, unusual-move proxy, key levels, sector read
# Ported from premarket_report.py. NOTE: Yahoo returns pre-market VOLUME as a hard zero on every
# bar (verified across tickers/intervals), so "pre-market volume vs 20-day typical" is NOT
# computable; GAPx substitutes today's |gap| vs the ticker's OWN trailing-20-session average
# |gap| — a move-based proxy for "is this morning unusual", labelled as such in the UI.
# Uses the live scanner's real yfinance + compute_session_vwap + VWAP_MAX_VIX (no drift).
# ============================================================
PM_DISCLAIMER = "Context only — not a validated signal. Cross-reference with news/catalyst before acting."
PM_START, PM_END = "04:00", "09:29"
PM_RTH_START, PM_RTH_END = "09:30", "16:00"
PM_GAP_SESSIONS = 20
PM_UNUSUAL_HI, PM_UNUSUAL_LO = 2.0, 0.5
PM_SECTOR_MAP = {
    "NVDA": "SMH", "AMD": "SMH", "TSM": "SMH", "ASML": "SMH", "AMAT": "SMH", "LRCX": "SMH",
    "KLAC": "SMH", "AVGO": "SMH", "MRVL": "SMH", "MU": "SMH", "INTC": "SMH", "SMCI": "SMH",
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK", "ORCL": "XLK", "CRM": "XLK",
    "SNOW": "XLK", "MDB": "XLK", "NET": "XLK", "PLTR": "XLK", "CRWV": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "GME": "XLY", "AMC": "XLY",
    "COIN": "XLF", "HOOD": "XLF",
    "LEU": "XLE", "AMTM": "XLI", "SOLS": "XLI",
}
PM_WEAK_PROXY = {"MARA", "RIOT", "COIN", "GLD", "TLT"}   # no honest sector proxy in this universe

def _pm_norm(df):
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols)
    df = df[cols].dropna(subset=["close"]).copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_ZONE)
    else:
        df.index = df.index.tz_convert(ET_ZONE)
    return df.sort_index()

def _pm_bt(df, a, b):
    """between_time() tolerant of empty/non-datetime frames (raw between_time raises on empty,
    which would let one dataless ticker take down the whole tab)."""
    if df is None or len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return df.between_time(a, b)

def _pm_download_one(tk, **kw):
    try:
        d = yf.download(tk, progress=False, auto_adjust=False, session=_YF_SESSION, **kw)
    except Exception:
        return pd.DataFrame()
    if d is None or len(d) == 0:
        return pd.DataFrame()
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    return d

@st.cache_data(ttl=120, show_spinner=False)
def pm_fetch(tickers):
    """Batch 5-min prepost (60d) + daily (30d), retrying any ticker the batch drops (a dropped
    sector ETF would silently gut the with-sector/ALONE read). Cached 120s; REFRESH clears it.
    Returns (intra, daily) dicts of normalized frames."""
    tickers = list(dict.fromkeys(tickers))
    intra, daily = {}, {}
    try:
        raw5 = yf.download(tickers, period="60d", interval="5m", prepost=True, group_by="ticker",
                           progress=False, auto_adjust=False, threads=True, session=_YF_SESSION)
    except Exception:
        raw5 = None
    try:
        rawd = yf.download(tickers, period="30d", interval="1d", group_by="ticker",
                           progress=False, auto_adjust=False, threads=True, session=_YF_SESSION)
    except Exception:
        rawd = None
    for tk in tickers:
        try:
            intra[tk] = _pm_norm(raw5[tk].copy())
        except Exception:
            intra[tk] = pd.DataFrame()
        try:
            d = _pm_norm(rawd[tk].copy())
            if not d.empty:
                d.index = pd.DatetimeIndex([t.date() for t in d.index])
            daily[tk] = d
        except Exception:
            daily[tk] = pd.DataFrame()
    for tk in tickers:                          # individual retry for anything the batch dropped
        if intra[tk].empty:
            intra[tk] = _pm_norm(_pm_download_one(tk, period="60d", interval="5m", prepost=True))
        if daily[tk].empty:
            d = _pm_norm(_pm_download_one(tk, period="30d", interval="1d"))
            if not d.empty:
                d.index = pd.DatetimeIndex([t.date() for t in d.index])
            daily[tk] = d
    return intra, daily

def _pm_last_and_date(df5):
    if df5.empty:
        return None, None
    pre = _pm_bt(df5, PM_START, PM_END)
    if pre.empty:
        return None, None
    d = max(t.date() for t in pre.index)
    day = pre[[t.date() == d for t in pre.index]]
    if day.empty:
        return None, None
    return d, float(day["close"].iloc[-1])

def _pm_gap_pct(df5, daily, d):
    """gap% for session d: last pre-market print vs the prior session's official close.
    date-vs-DatetimeIndex must go through pd.Timestamp (== against a Timestamp is silently False)."""
    pre = _pm_bt(df5, PM_START, PM_END)
    day = pre[[t.date() == d for t in pre.index]]
    if day.empty or daily.empty:
        return None
    prior = daily[daily.index < pd.Timestamp(d)]
    if prior.empty:
        return None
    pc = float(prior["close"].iloc[-1])
    if pc <= 0:
        return None
    return (float(day["close"].iloc[-1]) - pc) / pc * 100.0

def _pm_typical_abs_gap(df5, daily, upto):
    pre = _pm_bt(df5, PM_START, PM_END)
    days = sorted({t.date() for t in pre.index if t.date() < upto})[-PM_GAP_SESSIONS:]
    gaps = [abs(g) for g in (_pm_gap_pct(df5, daily, dd) for dd in days) if g is not None]
    return (sum(gaps) / len(gaps)) if gaps else None

def _pm_prior_levels(df5, daily, pm_date):
    hi = lo = cl = vw = None
    prior = daily[daily.index < pd.Timestamp(pm_date)] if not daily.empty else pd.DataFrame()
    if not prior.empty:
        r = prior.iloc[-1]
        hi, lo, cl = float(r["high"]), float(r["low"]), float(r["close"])
        pd_date = prior.index[-1].date()
        rth = _pm_bt(df5, PM_RTH_START, PM_RTH_END)
        sess = rth[[t.date() == pd_date for t in rth.index]]
        if not sess.empty and float(sess["volume"].sum()) > 0:
            try:
                vw = float(compute_session_vwap(sess).iloc[-1])    # reused, not reimplemented
            except Exception:
                vw = None
    return hi, lo, cl, vw

def pm_regime(daily):
    d = daily.get("^VIX", pd.DataFrame())
    if d.empty or len(d) < 2:
        return None, None, None, None
    cur = float(d["close"].iloc[-1]); prev = float(d["close"].iloc[-2])
    chg = cur - prev
    return cur, chg, (chg / prev * 100.0 if prev else None), cur > VWAP_MAX_VIX

def pm_build_rows(tickers, intra, daily):
    """Display-ready rows sorted by |gap| desc; each is a dict of formatted strings + a Flag."""
    sect_gap = {}
    for etf in sorted(set(PM_SECTOR_MAP.values()) | {"SPY"}):
        df5 = intra.get(etf, pd.DataFrame())
        d, _ = _pm_last_and_date(df5)
        sect_gap[etf] = _pm_gap_pct(df5, daily.get(etf, pd.DataFrame()), d) if d else None

    def money(x):
        return ("$%.2f" % x) if x is not None else "—"

    rows = []
    for tk in tickers:
        df5, dfd = intra.get(tk, pd.DataFrame()), daily.get(tk, pd.DataFrame())
        d, pm_last = _pm_last_and_date(df5)
        if not d or dfd.empty:
            continue
        gap_pct = _pm_gap_pct(df5, dfd, d)
        prior = dfd[dfd.index < pd.Timestamp(d)]
        pc = float(prior["close"].iloc[-1]) if not prior.empty else None
        gap_usd = (pm_last - pc) if (pm_last is not None and pc) else None
        typ = _pm_typical_abs_gap(df5, dfd, d)
        gapx = (abs(gap_pct) / typ) if (gap_pct is not None and typ and typ > 0) else None
        etf = PM_SECTOR_MAP.get(tk, "SPY")
        sg = sect_gap.get(etf)
        rel = "n/a"
        if gap_pct is not None and sg is not None:
            ex = gap_pct - sg
            rel = "with-sector" if abs(ex) < 0.5 else ("ALONE %+.1f" % ex if abs(ex) >= 1.0 else "mixed %+.1f" % ex)
        hi, lo, cl, vw = _pm_prior_levels(df5, dfd, d)
        flag = "UNUSUAL" if (gapx is not None and gapx > PM_UNUSUAL_HI) else \
               ("quiet" if (gapx is not None and gapx < PM_UNUSUAL_LO) else "")
        rows.append({
            "Ticker": tk + ("*" if tk in PM_WEAK_PROXY else ""),
            "PM Last": money(pm_last),
            "Gap $": ("%+.2f" % gap_usd) if gap_usd is not None else "—",
            "Gap %": ("%+.2f%%" % gap_pct) if gap_pct is not None else "—",
            "GAPx": ("%.1fx" % gapx) if gapx is not None else "—",
            "Sector": etf,
            "Sector %": ("%+.2f%%" % sg) if sg is not None else "—",
            "Rel": rel,
            "PD High": money(hi), "PD Low": money(lo), "PD Close": money(cl), "PD VWAP": money(vw),
            "Flag": flag,
            "_gap": abs(gap_pct) if gap_pct is not None else -1.0,
        })
    rows.sort(key=lambda r: -r["_gap"])
    for r in rows:
        del r["_gap"]
    return rows

# ============================================================
# STREAMLIT DASHBOARD
# ============================================================
st.set_page_config(page_title="Alpha Scanner", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# THEME — OLED dark, Inter, semantic status tokens (self-contained, no config.toml
# so other Streamlit apps on the Desktop are unaffected). See ui-ux-pro-max design system.
# ============================================================
# Semantic palette reused by the table cell-color functions below for one consistent language.
C_BG        = "#0A0B0E"   # app background (near-OLED)
C_SURFACE   = "#14161B"   # cards / metrics
C_SURFACE_2 = "#1B1E26"   # raised surface
C_BORDER    = "#262B36"
C_TEXT      = "#E6E9EF"
C_MUTED     = "#8B93A7"
C_PRIMARY   = "#3B82F6"   # blue (calls / up / info)
C_ACCENT    = "#D97706"   # amber (warnings / moderate)
C_GREEN     = "#16A34A"   # bullish / strong / gain
C_RED       = "#DC2626"   # bearish / loss / blow-off
C_GREEN_BG  = "#0E2A1A"   # muted green fill for table cells
C_AMBER_BG  = "#2C2208"   # muted amber fill
C_RED_BG    = "#2A0E0E"   # muted red fill

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {C_BG}; color: {C_TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.01em; color: {C_TEXT}; }}
h2 {{ font-size: 1.25rem !important; font-weight: 600 !important; margin-top: 0.4rem !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
[data-testid="stSidebar"] {{ background: {C_SURFACE}; border-right: 1px solid {C_BORDER}; }}
[data-testid="stMetric"] {{
  background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 10px;
  padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {C_MUTED}; font-size: 0.78rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.04em; }}
[data-testid="stMetricValue"] {{ font-weight: 700; font-size: 1.5rem; }}
[data-testid="stCaptionContainer"], .stCaption, small {{ color: {C_MUTED} !important; }}
hr {{ border-color: {C_BORDER} !important; }}
.stButton button {{ border-radius: 8px; border: 1px solid {C_BORDER}; font-weight: 600;
  transition: all 160ms ease; }}
.stButton button:hover {{ border-color: {C_PRIMARY}; color: {C_PRIMARY}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {C_BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {C_MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {C_TEXT}; }}
/* Custom header band */
.alpha-header {{ display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px; }}
.alpha-header .title {{ font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }}
.alpha-header .ver {{ font-size: 0.72rem; font-weight: 600; color: {C_PRIMARY};
  border: 1px solid {C_PRIMARY}; border-radius: 6px; padding: 2px 7px; }}
.alpha-sub {{ color: {C_MUTED}; font-size: 0.85rem; margin-bottom: 0.2rem; }}
/* Verdict scorecard */
.verdict {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 10px;
  padding: 14px 16px; height: 100%; }}
.verdict .vlabel {{ color: {C_MUTED}; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; }}
.verdict .vmain {{ font-size: 1.35rem; font-weight: 700; margin: 3px 0; line-height: 1.2; }}
.verdict .vmeta {{ color: {C_MUTED}; font-size: 0.84rem; }}
/* Compact market status strip */
.mkt-strip {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 2px 0 4px; }}
.pill {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 999px;
  padding: 5px 13px; font-size: 0.88rem; color: {C_TEXT}; font-weight: 600; }}
.pill .k {{ color: {C_MUTED}; font-weight: 500; margin-right: 6px; text-transform: uppercase;
  font-size: 0.7rem; letter-spacing: 0.05em; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="alpha-header"><span class="title">Alpha&nbsp;Scanner</span>'
    '<span class="ver">v0.1 · HYBRID</span></div>'
    '<div class="alpha-sub">Magnet pins · blow-off tops · VWAP / ORB · trade tracker · Telegram alerts</div>',
    unsafe_allow_html=True
)

# Pre-market context
market_context = get_market_context()
market_bias, bias_reasons = get_market_bias(market_context)
market_status, market_next = is_market_open()

# Compact market status strip (replaces the old 4-metric block for a cleaner glance).
_vix_val = market_context.get("vix", "N/A")
_spy_val = market_context.get("spy_change", "N/A")
_spy_txt = (("+" if _spy_val >= 0 else "") + str(_spy_val) + "%") if isinstance(_spy_val, (int, float)) else "N/A"
_bias_color = C_GREEN if market_bias == "FAVORABLE" else (C_RED if market_bias == "HOSTILE" else C_ACCENT)
st.markdown(
    '<div class="mkt-strip">'
    f'<span class="pill"><span class="k">Market</span>{market_status}</span>'
    f'<span class="pill"><span class="k">VIX</span>{_vix_val}</span>'
    f'<span class="pill"><span class="k">SPY</span>{_spy_txt}</span>'
    f'<span class="pill" style="border-color:{_bias_color};"><span class="k">Bias</span>'
    f'<span style="color:{_bias_color};font-weight:700;">{market_bias}</span></span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption(market_next)
with st.expander("Why this bias?"):
    for reason in bias_reasons:
        st.caption("• " + reason)

# --- Active Trade Tracker (now rendered inside the Journal tab; defined here) ---
def render_active_tracker():
    open_trades = track_active_trades()
    if open_trades:
        df_active = pd.DataFrame(open_trades)
        def color_status(val):
            if "STRONG GAIN" in str(val) or "GAIN" in str(val):
                return f"background-color: {C_GREEN_BG}; color: {C_GREEN}; font-weight: 600"
            elif "LOSS" in str(val) or "EXIT" in str(val):
                return f"background-color: {C_RED_BG}; color: {C_RED}; font-weight: 600"
            return f"background-color: {C_AMBER_BG}; color: {C_ACCENT}"
        styled_active = df_active.style.map(color_status, subset=["Status"])
        st.dataframe(styled_active, width="stretch", hide_index=True)
        st.caption("Refresh (REFRESH SCAN) to update live P&L.")
    else:
        st.info("No active trades. Log a trade below with Result = OPEN to track it live here.")

# Sidebar
# Saved preferences seed the widget defaults (used on a fresh page load; within a session the
# widget key preserves the choice). We re-save after reading so reloads remember the last state.
_prefs = load_prefs()
st.sidebar.header("Scan Controls")
tradeable_only = st.sidebar.checkbox("Show tradeable signals only (STRONG + good OI)",
                                     value=_prefs.get("tradeable_only", True), key="tradeable_only_sidebar")
# Intraday VWAP/ORB is one 1-min fetch per watchlist name. Default it ON only during the
# regular session (when the setups are live); off-hours you can still tick it on to review
# the day's VWAP/ORB without paying the fetch cost on every refresh.
_market_is_open = "Market Open" in market_status
run_intraday = st.sidebar.checkbox(
    "Run intraday VWAP/ORB scan",
    value=_prefs.get("run_intraday", _market_is_open),
    key="run_intraday_sidebar",
    help="One 1-min fetch per watchlist ticker. Auto-on during market hours; tick on after-hours to review the day's session."
)
# Pre-market context is a heavier fetch (5-min pre/post bars, 60d). Auto-on only pre-market.
_is_premarket = "Pre-Market" in market_status
run_premarket = st.sidebar.checkbox(
    "Run pre-market context",
    value=_prefs.get("run_premarket", _is_premarket),
    key="run_premarket_sidebar",
    help="Overnight gaps, unusual-move (GAPx), prior-day levels + closing VWAP, and sector context. ~15-30s fetch; auto-on during pre-market, off during the session by default."
)
re_scan = st.sidebar.button("REFRESH SCAN")

# Alert settings — tucked into an expander to declutter the sidebar.
st.sidebar.divider()
with st.sidebar.expander("📲 Telegram alerts", expanded=False):
    enable_alerts = st.checkbox("Enable alerts for new signals",
                                value=_prefs.get("enable_alerts", ALERTS_ENABLED), key="enable_alerts_sidebar")
    bot_token = st.text_input("Bot Token", value=TELEGRAM_BOT_TOKEN, type="password", key="bot_token_input")
    chat_id = st.text_input("Chat ID", value=TELEGRAM_CHAT_ID, key="chat_id_input")
    # Update globals from the inputs so scanning + the test button use the live values.
    TELEGRAM_BOT_TOKEN = bot_token
    TELEGRAM_CHAT_ID = chat_id
    last_alert = load_last_alert()
    if last_alert:
        st.caption(f"Last alert: {last_alert.get('ticker','?')} {last_alert.get('direction','?')} on {last_alert.get('timestamp','?')}")
    if st.button("🧪 Send Test Alert"):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            st.error("Bot token and Chat ID are required.")
        elif send_telegram_alert("✅ Alpha Scanner test alert! If you see this, Telegram alerts are working."):
            st.success("Test alert sent! Check your Telegram.")
        else:
            st.error("Failed to send test alert. Verify your token and chat ID.")

# Persist the toggle choices so a page reload restores them.
save_prefs({"tradeable_only": tradeable_only, "run_intraday": run_intraday,
            "enable_alerts": enable_alerts, "run_premarket": run_premarket})

with st.sidebar.expander("Watchlist (" + str(len(CORE_WATCHLIST)) + " tickers)", expanded=False):
    st.write(", ".join(CORE_WATCHLIST))
    st.caption("Edit CORE_WATCHLIST in the script to modify.")
st.sidebar.caption("📓 Log & edit trades in the Journal tab.")

# ============================================================
# AUTO SCAN ON STARTUP (or on refresh button)
# ============================================================
if "initial_scan_done" not in st.session_state:
    st.session_state.initial_scan_done = False

if not st.session_state.initial_scan_done or re_scan:
    st.session_state.initial_scan_done = True
    if re_scan:
        # Explicit refresh = fresh data: drop all cached fetches so we re-hit the network.
        # (Incidental reruns — checkbox toggles, trade logging — keep using the cache.)
        st.cache_data.clear()
    with st.spinner("Fetching top option volume tickers..."):
        top_tickers = get_top_option_volume_tickers(50)
    all_tickers = list(set(CORE_WATCHLIST + top_tickers))
    st.caption("Scanned " + str(len(all_tickers)) + " tickers · " + str(len(CORE_WATCHLIST))
               + " core + " + str(len(top_tickers)) + " top option-volume")

    # Results the top scorecard needs — initialized so the verdict renders even if a section is empty
    magnet_results = []
    parabolic_results = []
    vwap_rows, orb_rows, intraday_data_count = [], [], 0
    best_signal = None
    magnet_display_df = None   # the displayed magnet table, reused for journal prefill

    # Top-of-page verdict strip (a placeholder filled after the scan computes below)
    scorecard = st.container()

    tab_premarket, tab_futures, tab_magnet, tab_intraday, tab_blowoff, tab_journal = st.tabs(
        ["🌅 Pre-Market", "🌐 Futures", "🎯 Magnet Pins", "⏱ Intraday", "🔥 Blow-Offs", "📓 Journal & Trades"]
    )

    # ============================================================
    # TAB 0 — PRE-MARKET CONTEXT
    # ============================================================
    with tab_premarket:
        st.subheader("Pre-Market Context")
        st.caption(PM_DISCLAIMER)
        if not run_premarket:
            st.info("Pre-market context is off (heavier fetch). Tick **Run pre-market context** in the "
                    "sidebar, then REFRESH SCAN — it auto-enables during pre-market hours.")
        else:
            with st.spinner("Pulling pre-market context (5-min pre/post bars)..."):
                pm_need = sorted(set(CORE_WATCHLIST) | set(PM_SECTOR_MAP.values()) | {"SPY", "^VIX"})
                pm_intra, pm_daily = pm_fetch(pm_need)
                pm_rows = pm_build_rows(CORE_WATCHLIST, pm_intra, pm_daily)

            vix, vchg, vpct, above = pm_regime(pm_daily)
            if vix is not None:
                pc1, pc2, pc3 = st.columns([2, 2, 6])
                pc1.metric("VIX", "%.2f" % vix, "%+.2f (%+.1f%%)" % (vchg, vpct if vpct is not None else 0.0))
                pc2.metric("vs VWAP_MAX_VIX", ("ABOVE %.0f" % VWAP_MAX_VIX) if above else ("below %.0f" % VWAP_MAX_VIX))
                pc3.caption("VIX is informational here — not a trade gate. Yahoo reports pre-market VOLUME as "
                            "zero, so **GAPx** substitutes: today's |gap| ÷ this ticker's own 20-session average "
                            "|gap| (>2.0x = UNUSUAL, <0.5x = quiet). '*' = weak/no sector proxy.")

            if pm_rows:
                df_pm = pd.DataFrame(pm_rows)

                def _pm_color_gap(v):
                    s = str(v)
                    if s.startswith("+"): return f"color: {C_GREEN}; font-weight: 600"
                    if s.startswith("-"): return f"color: {C_RED}; font-weight: 600"
                    return ""
                def _pm_color_flag(v):
                    if "UNUSUAL" in str(v): return f"background-color: {C_AMBER_BG}; color: {C_ACCENT}; font-weight: 700"
                    if "quiet" in str(v): return f"color: {C_MUTED}"
                    return ""
                def _pm_color_rel(v):
                    if "ALONE" in str(v): return f"background-color: {C_AMBER_BG}; color: {C_ACCENT}"
                    if "with-sector" in str(v): return f"color: {C_MUTED}"
                    return ""

                styled_pm = (df_pm.style
                             .map(_pm_color_gap, subset=["Gap %"])
                             .map(_pm_color_flag, subset=["Flag"])
                             .map(_pm_color_rel, subset=["Rel"]))
                st.dataframe(styled_pm, width="stretch", hide_index=True)
                st.caption("Gap = last pre-market print vs prior official close. Levels = prior day H/L/C + prior "
                           "session closing VWAP (compute_session_vwap). 'ALONE' (gap ≥1pp from its sector ETF) "
                           "usually means company-specific news — find the catalyst. " + PM_DISCLAIMER)
            else:
                st.info("No pre-market data available (weekend / holiday / before ~4:00 AM ET, or Yahoo has no "
                        "pre-market prints yet). Yahoo serves pre-market prices only during/after the session.")

    # ============================================================
    # TAB 0.5 — FUTURES REGIME (risk/reward gauge)
    # ============================================================
    with tab_futures:
        st.subheader("Futures Regime")
        st.caption("Field-condition gauge, not a trade signal. The risk/reward of every magnet pin "
                   "and intraday setup depends on whether futures are calm (GREEN), stressed "
                   "(AMBER), or hostile (RED).")

        with st.spinner("Fetching futures..."):
            futures_data = get_futures_data()

        if futures_data and any(v for v in futures_data.values()):
            regime, risks = get_futures_regime(futures_data, vix=market_context.get("vix"))
            regime_text = C_GREEN if regime == "GREEN" else (C_ACCENT if regime == "AMBER" else C_RED)
            # Banner detail is regime-appropriate: for GREEN, 'risks' holds the risk-on / calm note;
            # for AMBER/RED it holds the stress reasons. Either way, list them plainly (no misleading
            # "favorable" label on a hostile tape).
            detail = ", ".join(risks) if risks else "No stress signals — conditions are calm."

            st.markdown(f"""
            <div style="background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:10px;
                        padding:16px; margin-bottom:16px; border-left:4px solid {regime_text};">
              <div style="font-size:1.2rem; font-weight:700; color:{regime_text};">{regime} LIGHT</div>
              <div style="color:{C_MUTED}; font-size:0.9rem; margin-top:4px;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

            rows = []
            for short, data in futures_data.items():
                if data:
                    rows.append({
                        "Contract": short,
                        "Price": f"${data['price']:.2f}",
                        "Prev Close": f"${data['prev_close']:.2f}",
                        "Change %": f"{data['change_pct']:+.2f}%",
                        "Trend": data["trend"],
                    })
            if rows:
                df_futures = pd.DataFrame(rows)

                def _color_futures_change(v):
                    s = str(v)
                    if s.startswith("+"): return f"color: {C_GREEN}; font-weight: 600"
                    if s.startswith("-"): return f"color: {C_RED}; font-weight: 600"
                    return ""

                st.dataframe(df_futures.style.map(_color_futures_change, subset=["Change %"]),
                             width="stretch", hide_index=True)
            st.caption("Change % is the latest daily bar vs the prior close. ES/NQ/YM/RTY = equity "
                       "index futures; CL = crude, GC = gold, ZB/ZN = Treasuries, DX = dollar index.")
        else:
            st.info("No futures data right now — Yahoo returned too few bars (holiday/off-hours) or is "
                    "throttling. Try REFRESH SCAN; it reloads once Yahoo responds with 2+ sessions.")

    # ============================================================
    # TAB 1 — MAGNET PINS
    # ============================================================
    with tab_magnet:
        st.subheader("Magnet Pin Signals (Dealer Hedging Targets)")
        if not POLYGON_KEY:
            st.error("**Polygon API key not found — all options data is unavailable.** "
                     "Every options metric (max pain, OI, IV, GEX, skew, P/C) comes from Polygon, so "
                     "the whole watchlist shows no data without it. Fix: set `POLYGON_KEY` in "
                     "`.streamlit/secrets.toml` next to alpha_scanner.py (local runs — now resolved no "
                     "matter which folder you launch from), or in the app's **Secrets** on Streamlit "
                     "Cloud (deployed). This is NOT a rate-limit or a market-hours issue.")
        with st.spinner("Calculating max pain for watchlist..."):
            mp_results = _parallel_map(get_max_pain_strike, CORE_WATCHLIST)   # concurrent fetch
            iv30_map = dict(zip(CORE_WATCHLIST, _parallel_map(_atm_iv_30d, CORE_WATCHLIST)))  # ~30d constant-maturity IV, in parallel
            failed_tickers = []   # tickers with no usable options data (fetch failure / rate-limit / none listed)
            for ticker, res in zip(CORE_WATCHLIST, mp_results):
                if res is None:                       # worker raised (isolated by _parallel_map)
                    failed_tickers.append(ticker)
                    continue
                max_pain, current_price, expiry, total_oi, atm_iv, skew = res
                record_iv(ticker, iv30_map.get(ticker))   # daily IV log = ~30d constant-maturity (matches backfilled history); skips ticker if None (main thread — no file race)
                record_skew(ticker, skew)     # full-watchlist daily skew logging (main thread)
                if max_pain and current_price:
                    record_magnet(ticker, max_pain, current_price, expiry)   # forward-test log (no lookahead)
                    distance_pct = ((current_price - max_pain) / max_pain) * 100
                    signal_strength = "STRONG" if abs(distance_pct) < 2 else ("MODERATE" if abs(distance_pct) < 4 else "WEAK")
                    oi_conf = get_oi_tier(total_oi)
                    if oi_conf == "LOW" and signal_strength == "STRONG":
                        signal_strength = "MODERATE"
                    magnet_results.append({
                        "Ticker": ticker,
                        "Max Pain Strike": "$" + str(round(max_pain, 2)),
                        "Current Price": "$" + str(round(current_price, 2)),
                        "Distance %": str(round(distance_pct, 2)) + "%",
                        "Direction": "MAGNET UP" if distance_pct < 0 else "MAGNET DOWN",
                        "OI Confidence": oi_conf + " (" + str(total_oi) + ")",
                        "Expiry": expiry.strftime("%Y-%m-%d") if expiry else "N/A",
                        "Signal Strength": signal_strength,
                        # --- raw fields for computed columns (dropped before display) ---
                        "_price": current_price,
                        "_is_long": distance_pct < 0,
                        "_level": max_pain,
                        "_signal_type": "magnet",
                        "_magnet_strike": max_pain,
                        "_oi_tier": oi_conf,
                        "_distance": distance_pct,
                        "_vol_ratio": None,
                        "_trigger_dt": datetime.now(ET_ZONE),
                        "_skew": skew,
                    })
                else:
                    failed_tickers.append(ticker)   # fetch returned no max-pain/price for this ticker

        if failed_tickers:
            if POLYGON_KEY and len(failed_tickers) == len(CORE_WATCHLIST):
                # Key present but EVERY ticker failed -> not per-ticker noise; a Polygon-wide problem.
                st.warning("⚠️ Options data failed for ALL " + str(len(CORE_WATCHLIST)) + " tickers "
                           "despite a configured key — likely a Polygon outage, an expired/downgraded "
                           "subscription (options snapshot needs the entitled plan), or a rate-limit. "
                           "Try REFRESH SCAN; if it persists, check the Polygon dashboard.")
            elif POLYGON_KEY:
                st.caption("⚠️ No options data for " + str(len(failed_tickers)) + "/"
                           + str(len(CORE_WATCHLIST)) + ": " + ", ".join(failed_tickers)
                           + " — timeout, rate-limit, or no listed options (not a signal-quality issue).")
            # (key-missing case already shown as a prominent st.error above)

        if magnet_results:
            df_magnet = pd.DataFrame(magnet_results)
            if tradeable_only:
                def is_tradeable(row):
                    strength = row["Signal Strength"]
                    oi = row["OI Confidence"]
                    if strength == "STRONG" and ("HIGH" in str(oi) or "MEDIUM" in str(oi)):
                        return True
                    if strength == "MODERATE" and "HIGH" in str(oi):
                        return True
                    return False
                df_magnet = df_magnet[df_magnet.apply(is_tradeable, axis=1)]

            strength_order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
            df_magnet["strength_sort"] = df_magnet["Signal Strength"].map(strength_order)
            df_magnet = df_magnet.sort_values(["strength_sort", "Distance %"])
            df_magnet = df_magnet.drop("strength_sort", axis=1)

            def color_strength(val):
                if val == "STRONG": return f"background-color: {C_GREEN_BG}; color: {C_GREEN}; font-weight: 700"
                elif val == "MODERATE": return f"background-color: {C_AMBER_BG}; color: {C_ACCENT}; font-weight: 600"
                return f"background-color: {C_RED_BG}; color: {C_MUTED}"
            def color_oi(val):
                if "HIGH" in str(val): return f"background-color: {C_GREEN_BG}; color: {C_GREEN}; font-weight: 600"
                elif "MEDIUM" in str(val): return f"background-color: {C_AMBER_BG}; color: {C_ACCENT}"
                return f"background-color: {C_RED_BG}; color: {C_MUTED}"

            if not df_magnet.empty:
                df_show = attach_computed_columns(df_magnet, vix=market_context.get("vix"), with_gex=True)
                magnet_display_df = df_show   # reused by the Journal tab's prefill selector
                styled_magnet = (df_show.style
                                 .map(color_strength, subset=["Signal Strength"])
                                 .map(color_oi, subset=["OI Confidence"])
                                 .map(color_readiness, subset=["Trade Readiness"])
                                 .map(color_gex_regime, subset=["GEX Regime"])
                                 .map(color_pcr_flag, subset=["P/C Flag"])
                                 .map(color_skew_trend, subset=["Skew Trend"]))
                st.dataframe(styled_magnet, width="stretch", hide_index=True)
                st.caption(IV_CAPTION)
                _ivdays = iv_days_logged()
                st.caption(("✅ Real logged IV history active (%d days)." % _ivdays)
                           if _ivdays >= IV_HISTORY_MIN_REAL else
                           ("⏳ IV history logging: %d/%d days accumulated — IV Rank/Pctl are a "
                            "realized-vol PROXY ('~') until then." % (_ivdays, IV_HISTORY_MIN_REAL)))
                st.caption(GEX_CAPTION)
                st.caption(PCR_CAPTION)
                st.caption(SKEW_CAPTION)
                best = df_show.iloc[0]
                best_signal = best   # expose the top pick to the scorecard above

                # --- TELEGRAM ALERT LOGIC ---
                if enable_alerts and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    oi_str = str(best.get("OI Confidence", ""))
                    oi_qualifies = "HIGH" in oi_str or "MEDIUM" in oi_str
                    # Only alert on a genuinely tradeable pin — within 1% of the magnet, matching the
                    # Trade Readiness gate (compute_readiness). Was OI-only, so a MODERATE pin up to
                    # ~4% away could still ping.
                    try:
                        dist_pct = abs(float(str(best.get("Distance %", "100%")).rstrip("%")))
                    except Exception:
                        dist_pct = 100.0
                    dist_qualifies = dist_pct < READINESS_MAX_DIST_PCT
                    if oi_qualifies and dist_qualifies and should_send_alert(best):
                        msg = generate_alert_message(best)
                        success = send_telegram_alert(msg)
                        if success:
                            save_last_alert({
                                "ticker": best["Ticker"],
                                "direction": best["Direction"],
                                "max_pain": best["Max Pain Strike"],
                                "expiry": best["Expiry"],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                            st.success("🔔 Alert sent to Telegram!")
                        else:
                            st.warning("Alert failed. Check your bot token and chat ID.")
                    else:
                        st.caption("No new signal to alert (same as last alert).")
            else:
                st.warning("No tradeable signals found. Toggle off 'tradeable signals only' in the sidebar to see all signals.")
        else:
            st.warning("No magnet data retrieved.")

    # ============================================================
    # TAB 2 — PARABOLIC BLOW-OFF TOPS
    # ============================================================
    with tab_blowoff:
        st.subheader("Parabolic Blow-Off Alerts (Short Candidates)")
        parabolic_subset = all_tickers[:75]
        with st.spinner("Scanning all tickers for parabolic conditions..."):
            pb_results = _parallel_map(check_parabolic_condition, parabolic_subset)   # concurrent fetch
            for ticker, (is_parabolic, detachment, vol_ratio, price) in zip(parabolic_subset, pb_results):
                if is_parabolic:
                    parabolic_results.append({
                        "Ticker": ticker,
                        "Price": "$" + str(round(price, 2)) if price else "N/A",
                        "Detachment from 20MA": str(round(detachment, 1)) + "%" if detachment else "N/A",
                        "Volume Ratio": str(round(vol_ratio, 1)) + "x" if vol_ratio else "N/A",
                        "Action": "SHORT / BUY PUTS",
                        "Stop Loss": "Above today's high"
                    })
        if parabolic_results:
            df_parabolic = pd.DataFrame(parabolic_results)
            st.dataframe(df_parabolic, width="stretch", hide_index=True)
            st.error("BLOW-OFF TOPS DETECTED — high-risk short plays active. Use hard stops above today's high.")

            # --- TELEGRAM ALERT: blow-off tops (new tickers only, deduped per day) ---
            if enable_alerts and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                current_blowoff_tickers = [r["Ticker"] for r in parabolic_results]
                new_blowoff, bo_state = get_new_blowoff_tickers(current_blowoff_tickers)
                if new_blowoff:
                    new_rows = [r for r in parabolic_results if r["Ticker"] in new_blowoff]
                    bo_msg = generate_blowoff_message(new_rows)
                    if send_telegram_alert(bo_msg):
                        bo_state["tickers"] = sorted(set(bo_state.get("tickers", [])) | set(new_blowoff))
                        save_blowoff_alerts(bo_state)
                        st.success("🔔 Blow-off alert sent to Telegram!")
                    else:
                        st.warning("Blow-off alert failed. Check your bot token and chat ID.")
        else:
            st.info("No parabolic blow-off tops detected currently.")

    # ============================================================
    # TAB 3 — INTRADAY (VWAP RECLAIM + OPENING RANGE BREAKOUT)
    # ============================================================
    with tab_intraday:
        if run_intraday:
            with st.spinner("Scanning watchlist for intraday VWAP / opening-range setups..."):
                vwap_rows, orb_rows, intraday_data_count = scan_intraday_setups(
                    CORE_WATCHLIST, vix=market_context.get("vix"))

        st.subheader("VWAP Reclaim Setups")
        st.caption("Genuine early-session dip-then-reclaim: BOTH the dip below VWAP and the reclaim "
                   "candle that CLOSES back above it must occur within the first "
                   + str(VWAP_OPENING_WINDOW_MINUTES) + " min (9:30-10:00 ET), on volume > "
                   + str(INTRADAY_VOL_CONFIRM) + "x average, with price STILL holding above. Reclaims "
                   "after 10:00 ET are rejected and not shown. Skipped when VIX > "
                   + str(VWAP_MAX_VIX) + ". Source: " + INTRADAY_SOURCE + ".")
        if not run_intraday:
            st.info("Intraday scan is off (saves one fetch per ticker). Enable 'Run intraday VWAP/ORB scan' in the sidebar.")
        elif intraday_data_count == 0:
            st.info("Not enough intraday data yet (market closed, pre-market, or the opening range hasn't formed). Check back during/after the session.")
        elif vwap_rows:
            df_v = attach_computed_columns(pd.DataFrame(vwap_rows), vix=market_context.get("vix"))
            st.dataframe(df_v.style.map(color_readiness, subset=["Trade Readiness"]),
                         width="stretch", hide_index=True)
            st.caption(IV_CAPTION)
            st.success("VWAP reclaim setups detected — long bias above VWAP.")
        else:
            st.info("No VWAP reclaim setups on the watchlist right now.")

        st.divider()
        st.subheader("Opening Range Breakout (15-min)")
        st.caption("High/low of the first " + str(ORB_RANGE_MINUTES)
                   + "-min range. Flagged only when a bar within the first " + str(ORB_ENTRY_WINDOW_MINUTES)
                   + " min CLOSES beyond it by ≥" + str(int(ORB_BREAK_BUFFER_FRAC * 100)) + "% of the range, on > "
                   + str(INTRADAY_VOL_CONFIRM) + "x running avg volume, and price is STILL holding "
                   + "the breakout. Opening range must span ≥" + str(ORB_MIN_RANGE_PCT)
                   + "% of price. Source: " + INTRADAY_SOURCE + ".")
        if not run_intraday:
            st.info("Intraday scan is off (saves one fetch per ticker). Enable 'Run intraday VWAP/ORB scan' in the sidebar.")
        elif intraday_data_count == 0:
            st.info("Not enough intraday data yet (market closed, pre-market, or the opening range hasn't formed). Check back during/after the session.")
        elif orb_rows:
            shown = orb_rows[:ORB_MAX_RESULTS]
            df_o = attach_computed_columns(pd.DataFrame(shown), vix=market_context.get("vix"))
            st.dataframe(df_o.style.map(color_readiness, subset=["Trade Readiness"]),
                         width="stretch", hide_index=True)
            st.caption(IV_CAPTION)
            if len(orb_rows) > ORB_MAX_RESULTS:
                st.success("Showing top " + str(ORB_MAX_RESULTS) + " of " + str(len(orb_rows))
                           + " opening-range breakouts (ranked by volume conviction).")
            else:
                st.success("Opening-range breakouts detected.")
        else:
            st.info("No opening-range breakouts on the watchlist right now.")

    # ============================================================
    # TAB 4 — TRADE JOURNAL
    # ============================================================
    with tab_journal:
        # 1) Active trades — live P&L on OPEN positions (moved here from the top of the page).
        st.subheader("Active Trades")
        render_active_tracker()
        st.divider()

        # 2) Log a trade — spacious form with one-click prefill from a scanned magnet signal.
        st.subheader("Log a Trade")
        _setups = ["", "magnet", "orb", "vwap", "blowoff", "other"]
        prefill_opts = ["✍️ Manual entry"]
        prefill_map = {}
        if magnet_display_df is not None and not magnet_display_df.empty:
            for _, prow in magnet_display_df.iterrows():
                lbl = str(prow["Ticker"]) + " · " + str(prow["Direction"]) + " · " + str(prow.get("Suggested Strike", ""))
                prefill_opts.append(lbl)
                prefill_map[lbl] = prow
        psel = st.selectbox("Prefill from a scanned magnet signal", prefill_opts,
                            help="Auto-fills ticker, direction, strike, expiry, stop and target from the Magnet Pins table.")
        d_ticker, d_dir, d_strike, d_expiry, d_stop, d_target, d_setup = "SPY", "CALL", "", "", "", "", ""
        if psel in prefill_map:
            pr = prefill_map[psel]
            d_ticker = str(pr.get("Ticker", ""))
            d_dir = "CALL" if str(pr.get("Direction", "")) == "MAGNET UP" else "PUT"
            d_strike = str(pr.get("Suggested Strike", "")).replace("$", "")
            d_expiry = str(pr.get("Expiry", ""))
            d_stop = str(pr.get("Stop Price", "")).replace("$", "")
            d_target = str(pr.get("Profit Target", "")).replace("$", "")
            d_setup = "magnet"
        with st.form("trade_form_journal", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                j_ticker = st.text_input("Ticker", value=d_ticker)
                j_direction = st.selectbox("Direction", ["CALL", "PUT"], index=0 if d_dir == "CALL" else 1)
                j_setup = st.selectbox("Setup", _setups, index=_setups.index(d_setup) if d_setup in _setups else 0)
            with fc2:
                j_strike = st.text_input("Strike", value=d_strike)
                j_expiry = st.text_input("Expiry (YYYY-MM-DD)", value=d_expiry)
                j_result = st.selectbox("Result", ["OPEN", "WIN", "LOSS"])
            with fc3:
                j_entry = st.text_input("Entry $", value="")
                j_exit = st.text_input("Exit $ (blank if open)", value="")
                j_stop = st.text_input("Stop $", value=d_stop)
                j_target = st.text_input("Target $", value=d_target)
            j_notes = st.text_area("Notes", value="")
            if st.form_submit_button("＋ Log Trade"):
                add_trade(j_ticker, j_direction, j_strike, j_expiry, j_entry, j_exit,
                          j_result, j_notes, setup=j_setup, stop=j_stop, target=j_target)
                st.success("Trade logged!")
                st.rerun()
        st.divider()

        # 3) Performance + editable history
        st.subheader("Performance")
        journal = load_journal()
        stats = get_journal_stats(journal)
        if stats["total_trades"] > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", stats["total_trades"])
            col2.metric("Wins", stats["wins"])
            col3.metric("Losses", stats["losses"])
            col4.metric("Win Rate", str(stats["win_rate"]) + "%")
            col5.metric("Total P&L", "$" + str(stats["total_pnl"]))

            analytics = get_performance_analytics(journal)
            if analytics:
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Win Rate", str(stats["win_rate"]) + "%")
                pf = analytics["profit_factor"]
                a2.metric("Profit Factor", "∞" if pf == float("inf") else str(round(pf, 2)))
                a3.metric("Max Drawdown", "$" + str(analytics["max_drawdown"]))
                a4.metric("Closed Trades", analytics["closed_trades"])
                st.caption("Profit factor = gross profit / gross loss. Max drawdown = largest peak-to-trough drop in cumulative P&L. "
                           "Avg win $" + str(analytics["avg_win"]) + " / avg loss $" + str(analytics["avg_loss"]) + ".")
                eq_df = pd.DataFrame({"Equity ($)": analytics["equity_curve"]})
                eq_df.index = range(1, len(analytics["equity_curve"]) + 1)
                eq_df.index.name = "Trade #"
                st.line_chart(eq_df)
            else:
                st.caption("Analytics appear once trades have both entry and exit prices.")

            st.subheader("Trade History")
            st.caption("Edit any cell inline · add rows at the bottom · then Save. Setup / Stop / Target are captured per trade.")
            df_journal = pd.DataFrame(journal)
            # Normalize to the canonical field set so older entries (missing the new
            # setup/stop/target columns) still edit cleanly and never render as NaN.
            for _c in JOURNAL_FIELDS:
                if _c not in df_journal.columns:
                    df_journal[_c] = ""
            df_journal = df_journal[JOURNAL_FIELDS].fillna("")
            edited_df = st.data_editor(
                df_journal,
                width="stretch",
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "date": "Date",
                    "ticker": "Ticker",
                    "direction": st.column_config.SelectboxColumn("Direction", options=["CALL", "PUT"]),
                    "strike": "Strike",
                    "expiry": "Expiry",
                    "entry_price": "Entry $",
                    "exit_price": "Exit $",
                    "result": st.column_config.SelectboxColumn("Result", options=["OPEN", "WIN", "LOSS"]),
                    "setup": st.column_config.SelectboxColumn("Setup", options=_setups),
                    "stop": "Stop $",
                    "target": "Target $",
                    "notes": "Notes",
                },
            )
            if st.button("💾 Save Changes"):
                update_journal(edited_df)
                st.success("Journal saved!")
                st.rerun()
            with st.expander("Delete a trade by row number"):
                max_idx = len(journal) - 1 if len(journal) > 0 else 0
                delete_index = st.number_input("Row number (0 = first row)", min_value=0, max_value=max_idx, value=0, step=1)
                if st.button("Delete Trade"):
                    delete_trade(delete_index)
                    st.success("Trade deleted!")
                    st.rerun()
        else:
            st.caption("No trades logged yet — use the form above (or prefill from a signal) to add your first.")

    # ============================================================
    # FILL THE TOP SCORECARD (verdict + signal counts) now that the scan is done
    # ============================================================
    strong_magnets = [m for m in magnet_results if m["Signal Strength"] == "STRONG"]
    with scorecard:
        sc_left, sc_right = st.columns([5, 4])
        with sc_left:
            if best_signal is not None:
                up = best_signal["Direction"] == "MAGNET UP"
                dir_color = C_GREEN if up else C_RED
                action = "Buy CALL" if up else "Buy PUT"
                arrow = "▲" if up else "▼"
                st.markdown(f"""
                <div class="verdict" style="border-left:3px solid {dir_color};">
                  <div class="vlabel">BEST TRADE CANDIDATE</div>
                  <div class="vmain">{best_signal['Ticker']}
                    <span style="color:{dir_color};">{arrow} {action}</span></div>
                  <div class="vmeta">Magnet {best_signal['Max Pain Strike']} ·
                    Price {best_signal['Current Price']} · Dist {best_signal['Distance %']} ·
                    OI {best_signal['OI Confidence']} · Exp {best_signal['Expiry']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict" style="border-left:3px solid {C_MUTED};">
                  <div class="vlabel">BEST TRADE CANDIDATE</div>
                  <div class="vmain" style="color:{C_MUTED};">Stand down</div>
                  <div class="vmeta">No tradeable magnet setup right now.</div>
                </div>
                """, unsafe_allow_html=True)
        with sc_right:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Strong Magnets", len(strong_magnets))
            m2.metric("Blow-Offs", len(parabolic_results))
            m3.metric("VWAP", len(vwap_rows))
            m4.metric("ORB", len(orb_rows))
        st.caption("Only trade STRONG magnets with HIGH/MEDIUM OI. Short only confirmed parabolic setups with hard stops.")
        st.divider()

else:
    st.info("The scanner is ready. Use the sidebar REFRESH SCAN button to update data.")