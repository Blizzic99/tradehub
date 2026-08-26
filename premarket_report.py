"""premarket_report.py — single-screen pre-market CONTEXT for the CORE_WATCHLIST.

This is a context sheet, NOT a signal generator. Nothing here has been validated as an edge —
in fact the contraction-breakout work in contraction_backtest.py is a standing reminder that a
number can look tradeable and be worthless. Read this as "what is different this morning", then
go find the catalyst.

Parity: CORE_WATCHLIST, compute_session_vwap and VWAP_MAX_VIX are IMPORTED from alpha_scanner.py
(via alpha_backtest's stub loader) rather than redefined, so this file can never drift from the
live scanner's definitions.

DATA REALITY (probed 2026-07-16 — do not re-litigate without re-probing):
  * Yahoo serves pre-market PRICES (5m/1m, prepost=True) but reports pre-market VOLUME as a hard
    ZERO on every bar/ticker/interval tested (AAPL 1980/1980 pre-market bars zero; same for NVDA,
    SPY; RTH volume is fine, so the feed itself is healthy). Polygon can't cover it either: its
    free tier blocks same-day intraday, which is precisely why the scanner uses yfinance.
  * CONSEQUENCE: "pre-market cumulative volume vs a trailing-20d same-time-of-day average" is NOT
    COMPUTABLE here. Rather than print a fake zero or silently drop the check, this report
    substitutes GAPx: today's |gap| against this ticker's OWN trailing-20-session average |gap|.
    It answers the same question ("is this morning unusual for this name?") from the move instead
    of the tape. It is a PROXY, and it is labelled as one everywhere it appears.
  * 1-minute history only reaches ~7 days, so the 20-session baseline uses 5-minute bars (60d).
"""

import sys
import os
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpha_backtest as ab      # stub-imports alpha_scanner offline; gives parity constants

ET = ab.ET_ZONE
DISCLAIMER = "Context only - not a validated signal. Cross-reference with news/catalyst before acting."
PM_START, PM_END = "04:00", "09:29"          # pre-market window (RTH opens 09:30)
RTH_START, RTH_END = "09:30", "16:00"
GAP_BASELINE_SESSIONS = 20
UNUSUAL_HI, UNUSUAL_LO = 2.0, 0.5            # GAPx thresholds: >2x unusual-high, <0.5x unusually quiet

# Sector proxy per name, so a single-name gap can be read against its group. Names with no clean
# proxy in the watchlist (crypto miners track BTC, not a sector) fall back to SPY and are marked
# with a weak-proxy note rather than pretending XLF explains them.
SECTOR_MAP = {
    # semis
    "NVDA": "SMH", "AMD": "SMH", "TSM": "SMH", "ASML": "SMH", "AMAT": "SMH", "LRCX": "SMH",
    "KLAC": "SMH", "AVGO": "SMH", "MRVL": "SMH", "MU": "SMH", "INTC": "SMH", "SMCI": "SMH",
    # mega-cap tech / software / AI data layer
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK", "ORCL": "XLK", "CRM": "XLK",
    "SNOW": "XLK", "MDB": "XLK", "NET": "XLK", "PLTR": "XLK", "CRWV": "XLK",
    # consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "GME": "XLY", "AMC": "XLY",
    # financials-adjacent
    "COIN": "XLF", "HOOD": "XLF",
    # energy / nuclear fuel, industrials / defense / materials
    "LEU": "XLE", "AMTM": "XLI", "SOLS": "XLI",
}
WEAK_PROXY = {"MARA", "RIOT", "COIN", "GLD", "TLT"}   # no honest sector proxy in this universe


def _norm(df):
    """Yahoo frame -> tz-aware ET index, lowercase OHLCV, sorted. Empty frame if unusable."""
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols)
    df = df[cols].dropna(subset=["close"]).copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df.sort_index()


def _download_one(yf, tk, **kw):
    """Single-ticker download, flattening yfinance's ('Close','XLK') MultiIndex to 'Close'.
    Note the column order differs from a group_by='ticker' batch (which yields ('XLK','Close'))."""
    try:
        d = yf.download(tk, progress=False, auto_adjust=False, **kw)
    except Exception:
        return pd.DataFrame()
    if d is None or len(d) == 0:
        return pd.DataFrame()
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    return d


def fetch(tickers):
    """Batch-pull 5-minute prepost bars (60d) + daily bars (30d), then RETRY anything the batch
    dropped. Returns (intraday, daily) dicts.

    The retry is not optional: a transient batch failure is silent apart from yfinance's one-line
    'N Failed downloads', and if the casualty is a SECTOR ETF it quietly guts the with-sector /
    ALONE read for every name mapped to it (observed live: XLK dropped once and took 11 names'
    sector comparison down to 'n/a'). Retried individually, XLK returns ~11k bars fine.
    """
    yf = ab._load_real_yfinance()
    tickers = list(dict.fromkeys(tickers))
    intra, daily = {}, {}
    try:
        raw5 = yf.download(tickers, period="60d", interval="5m", prepost=True, group_by="ticker",
                           progress=False, auto_adjust=False, threads=True)
    except Exception:
        raw5 = None
    try:
        rawd = yf.download(tickers, period="30d", interval="1d", group_by="ticker",
                           progress=False, auto_adjust=False, threads=True)
    except Exception:
        rawd = None
    for tk in tickers:
        try:
            intra[tk] = _norm(raw5[tk].copy())
        except Exception:
            intra[tk] = pd.DataFrame()
        try:
            d = _norm(rawd[tk].copy())
            if not d.empty:
                d.index = pd.DatetimeIndex([t.date() for t in d.index])
            daily[tk] = d
        except Exception:
            daily[tk] = pd.DataFrame()

    retried = []
    for tk in tickers:
        if intra[tk].empty:
            intra[tk] = _norm(_download_one(yf, tk, period="60d", interval="5m", prepost=True))
            if not intra[tk].empty:
                retried.append(tk)
        if daily[tk].empty:
            d = _norm(_download_one(yf, tk, period="30d", interval="1d"))
            if not d.empty:
                d.index = pd.DatetimeIndex([t.date() for t in d.index])
            daily[tk] = d
    if retried:
        print("  (recovered %d ticker(s) the batch dropped: %s)" % (len(retried), ", ".join(retried)))
    dead = [tk for tk in tickers if intra[tk].empty]
    if dead:
        print("  WARNING: no intraday data even after retry: %s" % ", ".join(dead))
    return intra, daily


def _bt(df, a, b):
    """between_time() that tolerates empty / non-datetime-indexed frames. Raw between_time raises
    'Index must be DatetimeIndex' on an empty frame, which would let one dataless ticker take down
    the whole report."""
    if df is None or len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return df.between_time(a, b)


def _pm_last_and_date(df5):
    """(pm_date, pm_last_price, pm_bar_count, cutoff_time) for the most recent session that has
    pre-market prices. Returns (None,...) if none."""
    if df5.empty:
        return None, None, 0, None
    pre = _bt(df5, PM_START, PM_END)
    if pre.empty:
        return None, None, 0, None
    d = max(t.date() for t in pre.index)
    day = pre[[t.date() == d for t in pre.index]]
    if day.empty:
        return None, None, 0, None
    return d, float(day["close"].iloc[-1]), len(day), day.index[-1].time()


def _gap_pct_for_session(df5, daily, d):
    """gap% for session `d`: its last pre-market print vs the prior session's official close.
    NOTE: `daily` carries a DatetimeIndex, so date-vs-index comparisons must go through
    pd.Timestamp — comparing a datetime64 index to a plain date raises, and (worse) comparing a
    Timestamp to a date with == is silently always False."""
    pre = _bt(df5, PM_START, PM_END)
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


def _typical_abs_gap(df5, daily, upto_date, n=GAP_BASELINE_SESSIONS):
    """Mean |gap%| over the `n` sessions BEFORE upto_date (the substitute for the unavailable
    pre-market-volume baseline)."""
    pre = _bt(df5, PM_START, PM_END)
    days = sorted({t.date() for t in pre.index if t.date() < upto_date})[-n:]
    gaps = [abs(g) for g in (_gap_pct_for_session(df5, daily, d) for d in days) if g is not None]
    return (sum(gaps) / len(gaps)) if gaps else None


def _prior_levels(df5, daily, pm_date):
    """(high, low, close, closing_vwap) of the session before pm_date. H/L/C from official daily
    bars; VWAP from that session's RTH 5m bars via the scanner's own compute_session_vwap."""
    hi = lo = cl = vw = None
    prior = daily[daily.index < pd.Timestamp(pm_date)] if not daily.empty else pd.DataFrame()
    if not prior.empty:
        r = prior.iloc[-1]
        hi, lo, cl = float(r["high"]), float(r["low"]), float(r["close"])
        pd_date = prior.index[-1].date()          # Timestamp -> date; comparing t.date() to a
        rth = _bt(df5, RTH_START, RTH_END)  # Timestamp is silently ALWAYS False
        sess = rth[[t.date() == pd_date for t in rth.index]]
        if not sess.empty and float(sess["volume"].sum()) > 0:
            try:
                v = ab.compute_session_vwap(sess)     # reused from alpha_scanner, not reimplemented
                vw = float(v.iloc[-1])
            except Exception:
                vw = None
    return hi, lo, cl, vw


def regime(intra, daily):
    """(vix_level, vix_change, vix_change_pct, is_above_gate). VIX daily closes; pre-market VIX
    isn't meaningful, so this is the latest close vs the prior close."""
    d = daily.get("^VIX", pd.DataFrame())
    if d.empty or len(d) < 2:
        return None, None, None, None
    cur = float(d["close"].iloc[-1]); prev = float(d["close"].iloc[-2])
    chg = cur - prev
    return cur, chg, (chg / prev * 100.0 if prev else None), cur > ab.VWAP_MAX_VIX


def build_report(tickers=None):
    tickers = list(tickers or ab.CORE_WATCHLIST)
    need = sorted(set(tickers) | set(SECTOR_MAP.values()) | {"SPY", "^VIX"})
    intra, daily = fetch(need)

    # what session are we actually describing, and is it live?
    now = datetime.now(ET)
    ref_date = None
    for tk in ["SPY"] + tickers:
        d, _, _, _ = _pm_last_and_date(intra.get(tk, pd.DataFrame()))
        if d:
            ref_date = d
            break
    live = bool(ref_date and ref_date == now.date() and now.time() < datetime.strptime("09:30", "%H:%M").time())
    state = "LIVE pre-market" if live else "STALE - showing the last completed pre-market session"

    print("=" * 118)
    print("PRE-MARKET CONTEXT  |  generated %s  |  session %s  [%s]"
          % (now.strftime("%Y-%m-%d %H:%M ET"), ref_date, state))
    print(DISCLAIMER)
    print("=" * 118)

    vix, vchg, vpct, above = regime(intra, daily)
    if vix is not None:
        print("REGIME: VIX %.2f  (%+.2f, %+.1f%% vs prior close)  %s VWAP_MAX_VIX=%.1f   [informational only - not a gate here]"
              % (vix, vchg, vpct, "ABOVE" if above else "below", ab.VWAP_MAX_VIX))
    else:
        print("REGIME: VIX unavailable")

    print("\nDATA NOTE: Yahoo reports pre-market VOLUME as zero on every bar (verified across tickers")
    print("and intervals), so 'PM volume vs 20d typical' is NOT computable. GAPx below substitutes")
    print("today's |gap| vs this ticker's own trailing-%d-session average |gap| - a MOVE-based proxy" % GAP_BASELINE_SESSIONS)
    print("for 'is this morning unusual', not a volume measure.  * = weak/no sector proxy.\n")

    hdr = ("  %-6s %9s %8s %8s %6s %6s %7s %-12s %9s %9s %9s %9s"
           % ("TICKER", "PM LAST", "GAP $", "GAP %", "GAPx", "SECT", "SECT%", "REL", "PD HIGH", "PD LOW", "PD CLOSE", "PD VWAP"))
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    # sector gaps first (each ETF's own overnight gap)
    sect_gap = {}
    for etf in sorted(set(SECTOR_MAP.values()) | {"SPY"}):
        df5 = intra.get(etf, pd.DataFrame())
        d, _, _, _ = _pm_last_and_date(df5)
        sect_gap[etf] = _gap_pct_for_session(df5, daily.get(etf, pd.DataFrame()), d) if d else None

    rows = []
    for tk in tickers:
        df5, dfd = intra.get(tk, pd.DataFrame()), daily.get(tk, pd.DataFrame())
        d, pm_last, nbars, _ = _pm_last_and_date(df5)
        if not d or dfd.empty:
            rows.append((tk, None)); continue
        gap_pct = _gap_pct_for_session(df5, dfd, d)
        prior = dfd[dfd.index < pd.Timestamp(d)]
        pc = float(prior["close"].iloc[-1]) if not prior.empty else None
        gap_usd = (pm_last - pc) if (pm_last is not None and pc) else None
        typ = _typical_abs_gap(df5, dfd, d)
        gapx = (abs(gap_pct) / typ) if (gap_pct is not None and typ and typ > 0) else None
        etf = SECTOR_MAP.get(tk, "SPY")
        sg = sect_gap.get(etf)
        rel = "n/a"
        if gap_pct is not None and sg is not None:
            ex = gap_pct - sg
            rel = "with-sector" if abs(ex) < 0.5 else ("ALONE %+.1f" % ex if abs(ex) >= 1.0 else "mixed %+.1f" % ex)
        hi, lo, cl, vw = _prior_levels(df5, dfd, d)
        rows.append((tk, {"pm": pm_last, "g$": gap_usd, "g%": gap_pct, "gx": gapx, "etf": etf,
                          "sg": sg, "rel": rel, "hi": hi, "lo": lo, "cl": cl, "vw": vw}))

    rows.sort(key=lambda r: -(abs(r[1]["g%"]) if r[1] and r[1]["g%"] is not None else -1))
    def f(x, p="%9.2f"):
        return (p % x) if x is not None else "        -"
    for tk, r in rows:
        if r is None:
            print("  %-6s      no pre-market data" % tk); continue
        star = "*" if tk in WEAK_PROXY else " "
        gx = ("%5.1fx" % r["gx"]) if r["gx"] is not None else "    -"
        flag = ""
        if r["gx"] is not None:
            if r["gx"] > UNUSUAL_HI: flag = " <== UNUSUAL move"
            elif r["gx"] < UNUSUAL_LO: flag = " (quiet)"
        print("  %-6s %9s %8s %7s%% %6s %5s%s %6s %-12s %9s %9s %9s %9s%s"
              % (tk, f(r["pm"]), f(r["g$"], "%8.2f"), ("%+7.2f" % r["g%"]) if r["g%"] is not None else "      -",
                 gx, r["etf"], star, ("%+6.2f" % r["sg"]) if r["sg"] is not None else "     -",
                 r["rel"], f(r["hi"]), f(r["lo"]), f(r["cl"]), f(r["vw"]), flag))

    print("\n  GAPx = |gap| / trailing-%dd average |gap| for that name  (>%.1fx unusual, <%.1fx quiet)."
          % (GAP_BASELINE_SESSIONS, UNUSUAL_HI, UNUSUAL_LO))
    print("  REL: 'with-sector' = gap within 0.5pp of its sector ETF; 'ALONE' = >=1pp away from the")
    print("  sector, which usually means company-specific news - go find the catalyst before acting.")
    print("  PD VWAP = prior session's closing VWAP (alpha_scanner.compute_session_vwap, 5m bars).")
    print("  " + DISCLAIMER)


if __name__ == "__main__":
    build_report()
