"""alpha_backtest.py — historical data layer for backtesting the Alpha Scanner setups.

Stage 1 (this file): just the Polygon minute-bar fetcher + session grouping, so the fetch
can be verified before any strategy logic is built on top of it.

Design mirrors `_intraday_polygon` / `_normalize_intraday` in alpha_scanner.py:
  - endpoint  /v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}
  - request   adjusted=true, sort=asc, limit=50000, apiKey=...
  - output    tz-aware America/New_York index, lowercase o/h/l/c/v, regular hours 09:30-16:00
Differences: it paginates the FULL date range via Polygon's `next_url`, throttles for the
free tier, and keeps every session in the range (not just the most recent one).
"""

import os
import sys
import time
import math
import json
import types
import tempfile
import importlib.util
from datetime import date, datetime, timedelta, timezone

import pandas as pd

try:
    import requests
except Exception as e:  # pragma: no cover
    raise SystemExit("alpha_backtest requires the 'requests' package: " + str(e))

# DST-aware US market timezone — same choice as alpha_scanner.py.
try:
    from zoneinfo import ZoneInfo
    ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    ET_ZONE = timezone(timedelta(hours=-4))  # fallback: EDT

POLYGON_BASE_URL = "https://api.polygon.io"
REGULAR_HOURS = ("09:30", "16:00")

# Global Polygon request pacing — a single timestamp shared by ALL fetches in this process, so the
# free-tier 5-req/min budget is respected across page boundaries AND ticker boundaries (sleeping
# only between pages of one ticker lets back-to-back tickers blow the window and 429-cascade).
_LAST_POLY_REQ = [0.0]


def _poly_throttle(seconds):
    wait = _LAST_POLY_REQ[0] + seconds - time.time()
    if wait > 0:
        time.sleep(wait)
    _LAST_POLY_REQ[0] = time.time()
# A full RTH day is ~391 one-minute bars (390 + the inclusive 16:00 print). An early-close
# half-day (day after Thanksgiving, Christmas Eve, etc.) closes ~13:00 ET and yields ~211.
# 350 sits comfortably between the two, so it's a robust, simple half-day detector.
HALF_DAY_MIN_BARS = 350


# ------------------------------------------------------------------
# Polygon key resolution — mirrors alpha_scanner.py's _secret() so the
# same secrets.toml/env works, WITHOUT importing the Streamlit app.
# ------------------------------------------------------------------
def _secret(name, default=""):
    """Read a secret: env var first, then .streamlit/secrets.toml NEXT TO THIS SCRIPT (and the
    user's ~/.streamlit/secrets.toml). We read the TOML file directly rather than via st.secrets,
    because this backtest import-stubs Streamlit (see _install_import_stubs) — so st.secrets is
    inert here. Anchored to __file__, so the temp-dir chdir during scanner load can't hide it."""
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
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("#") or "=" not in s:
                            continue
                        k, _, val = s.partition("=")
                        if k.strip() == name:
                            return val.strip().strip('"').strip("'")
            except Exception:
                pass
    return default


def _polygon_key():
    """POLYGON_API_KEY env (standalone) wins; otherwise fall back to the shared _secret('POLYGON_KEY')."""
    return os.environ.get("POLYGON_API_KEY") or _secret("POLYGON_KEY")


# ------------------------------------------------------------------
# Normalization (keeps ALL sessions in range, unlike the scanner's last-session-only variant)
# ------------------------------------------------------------------
def _normalize(df):
    """Lowercase OHLCV, tz-aware ET index, sorted, filtered to regular hours 09:30-16:00. Empty df if unusable."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    need = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame(columns=need)
    df = df[need].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_ZONE)
    else:
        df.index = df.index.tz_convert(ET_ZONE)
    df = df.sort_index()
    df = df.between_time(REGULAR_HOURS[0], REGULAR_HOURS[1])
    return df


def fetch_minute_history(ticker, start_date, end_date, throttle_seconds=12, timeout=30, api_key=None):
    """Pull 1-minute bars for `ticker` from Polygon over [start_date, end_date] inclusive.

    Paginates the full range via Polygon's `next_url` (appending the api key to each follow-up
    request, since next_url omits it), so it returns every bar in the window rather than just the
    first 50000. Throttles to `throttle_seconds` between requests (default 12s ≈ 5 req/min) so it
    works on the Polygon free tier.

    start_date / end_date: 'YYYY-MM-DD' strings or date/datetime objects.
    Returns a single DataFrame with a tz-aware America/New_York DatetimeIndex and lowercase
    open/high/low/close/volume columns, filtered to regular hours (matches _normalize_intraday).
    """
    key = api_key or _polygon_key()
    if not key:
        raise RuntimeError(
            "No Polygon API key. Set POLYGON_API_KEY in the environment, or add POLYGON_KEY to "
            "your Streamlit secrets (.streamlit/secrets.toml)."
        )

    def _fmt(d):
        if isinstance(d, str):
            return d
        return d.strftime("%Y-%m-%d")

    frm, to = _fmt(start_date), _fmt(end_date)
    url = POLYGON_BASE_URL + "/v2/aggs/ticker/" + ticker + "/range/1/minute/" + frm + "/" + to
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}

    rows = []
    retries = 0
    while url:
        _poly_throttle(throttle_seconds)   # global pace: every Polygon request in this process
                                           # (first pages, follow-up pages, and other tickers alike)
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:        # rate-limited: back off and retry the SAME page.
            retries += 1                   # (One unhandled 429 used to cascade — each failed ticker
            if retries > 4:                #  fired the next request immediately, so 44/49 died.)
                raise RuntimeError(
                    "Polygon rate limit (429) persisted after %d retries for %s %s..%s"
                    % (retries - 1, ticker, frm, to))
            time.sleep(30 * retries)       # 30/60/90/120s — free-tier windows reset within a minute
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                "Polygon request failed (%s) for %s %s..%s: %s"
                % (resp.status_code, ticker, frm, to, resp.text[:200])
            )
        retries = 0
        j = resp.json()
        rows.extend(j.get("results", []) or [])
        next_url = j.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": key}   # next_url already carries limit/sort/adjusted; only add the key
        else:
            url = None

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    idx = pd.to_datetime([b["t"] for b in rows], unit="ms", utc=True).tz_convert(ET_ZONE)
    df = pd.DataFrame({
        "open":   [b["o"] for b in rows],
        "high":   [b["h"] for b in rows],
        "low":    [b["l"] for b in rows],
        "close":  [b["c"] for b in rows],
        "volume": [b["v"] for b in rows],
    }, index=idx)
    return _normalize(df)


def group_by_session(df, drop_half_days=True, min_session_bars=HALF_DAY_MIN_BARS, verbose=True):
    """Split a multi-day minute frame into {date: session_df}, one entry per trading day (sorted).

    This is the backtest-PREP layer, not the data layer: fetch_minute_history() still returns
    every bar faithfully. When drop_half_days=True, sessions with fewer than `min_session_bars`
    bars — i.e. early-close half-days like the day after Thanksgiving or Christmas Eve (~211 bars
    vs ~391 on a full day) — are EXCLUDED so they don't skew VWAP/ORB stats. The bar-count test is
    used because it's robust and simple (a real half-day closes ~13:00 ET). Drops are printed (when
    verbose) so the exclusion is transparent, never silent. Pass drop_half_days=False for the raw split.
    """
    if df is None or df.empty:
        return {}
    sessions = {d: g for d, g in df.groupby(df.index.date)}
    if not drop_half_days:
        return sessions

    kept = {}
    dropped = []
    for d in sorted(sessions.keys()):
        s = sessions[d]
        if len(s) < min_session_bars:
            dropped.append((d, len(s)))
        else:
            kept[d] = s

    if verbose:
        if dropped:
            print("Dropped %d half-day session(s) (< %d bars):" % (len(dropped), min_session_bars))
            for d, n in dropped:
                print("  %s  %d bars" % (d.strftime("%Y-%m-%d"), n))
        else:
            print("Half-day filter: none to drop (all %d session(s) >= %d bars)." % (len(kept), min_session_bars))
    return kept


# ==================================================================================
# PARITY LAYER — pull thresholds + compute_session_vwap straight from alpha_scanner.py
# so the backtest and the live scanner share ONE source of truth and can never drift.
#
# alpha_scanner.py is a Streamlit app that runs top-to-bottom on import, so importing it
# naively would launch the UI and hit the network. We install lightweight streamlit/yfinance
# stubs (only if those modules aren't already present) and import inside a temp cwd, exactly
# like tests/test_scanner.py — the app's UI/scan/file-writes become inert, but every function
# and constant is defined for real. If alpha_scanner is already imported (e.g. running inside
# the live app), we just reuse it.
# ==================================================================================
class _AnyStub:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _AnyStub()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return _AnyStub()
    def __iter__(self): return iter([])


class _SessionState(dict):
    def __getattr__(self, n):
        try: return self[n]
        except KeyError: return None
    def __setattr__(self, n, v): self[n] = v


def _install_import_stubs():
    if "streamlit" not in sys.modules:
        st = types.ModuleType("streamlit")
        def cache_data(*a, **k):
            if a and callable(a[0]): return a[0]     # passthrough decorator — keep real functions
            return lambda fn: fn
        cache_data.clear = lambda *a, **k: None
        st.cache_data = cache_data
        st.session_state = _SessionState(initial_scan_done=True)   # skip the heavy auto-scan block
        st.secrets = _AnyStub()
        st.column_config = _AnyStub()
        def columns(spec, *a, **k):
            n = len(spec) if isinstance(spec, (list, tuple)) else spec
            return [_AnyStub() for _ in range(n)]
        st.columns = columns
        st.tabs = lambda items, *a, **k: [_AnyStub() for _ in items]
        for name in ["set_page_config", "title", "caption", "markdown", "header", "subheader",
                     "divider", "metric", "dataframe", "line_chart", "success", "warning", "error",
                     "info", "write", "spinner", "container", "empty", "form", "data_editor",
                     "expander", "popover", "toast", "rerun", "stop", "table"]:
            setattr(st, name, _AnyStub())
        # Fallback: any layout/display st.* added to alpha_scanner later that isn't listed
        # above stays inert here (returns an _AnyStub) instead of raising AttributeError and
        # breaking the scanner import. Explicitly-set attrs (button/checkbox/…) take precedence.
        st.__getattr__ = lambda name: _AnyStub()
        st.button = lambda *a, **k: False
        st.form_submit_button = lambda *a, **k: False
        st.checkbox = lambda *a, **k: bool(k.get("value", False))
        st.text_input = lambda *a, **k: str(k.get("value", ""))
        st.text_area = lambda *a, **k: str(k.get("value", ""))
        st.number_input = lambda *a, **k: k.get("value", 0)
        def _selectbox(label, options=(), *a, **k): return options[0] if options else ""
        st.selectbox = _selectbox
        class _Sidebar:
            def __getattr__(self, n):
                if n in ("button", "form_submit_button"): return lambda *a, **k: False
                if n in ("text_input", "text_area"): return lambda *a, **k: str(k.get("value", ""))
                if n == "checkbox": return lambda *a, **k: bool(k.get("value", False))
                if n == "selectbox": return _selectbox
                if n == "number_input": return lambda *a, **k: k.get("value", 0)
                if n == "columns": return columns
                return _AnyStub()
        st.sidebar = _Sidebar()
        sys.modules["streamlit"] = st
    if "yfinance" not in sys.modules:
        yf = types.ModuleType("yfinance")
        class _Hist:
            empty = True
            def __getitem__(self, k): return []
        class _Ticker:
            def __init__(self, *a, **k): self.options = []
            def history(self, *a, **k): return _Hist()
            def option_chain(self, *a, **k): raise Exception("no chain")
        yf.Ticker = _Ticker
        yf.download = lambda *a, **k: _Hist()
        sys.modules["yfinance"] = yf


def _load_scanner_module():
    if "alpha_scanner" in sys.modules:
        return sys.modules["alpha_scanner"]
    _install_import_stubs()
    scanner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha_scanner.py")
    spec = importlib.util.spec_from_file_location("alpha_scanner", scanner_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alpha_scanner"] = mod
    _cwd = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())   # keep the app's prefs/journal writes out of the real dirs
        spec.loader.exec_module(mod)
    finally:
        os.chdir(_cwd)
    return mod


_asc = _load_scanner_module()

# The single source of truth — imported, never redefined.
INTRADAY_VOL_CONFIRM        = _asc.INTRADAY_VOL_CONFIRM
VWAP_OPENING_WINDOW_MINUTES = _asc.VWAP_OPENING_WINDOW_MINUTES
ORB_RANGE_MINUTES           = _asc.ORB_RANGE_MINUTES
ORB_MIN_RANGE_PCT           = _asc.ORB_MIN_RANGE_PCT
ORB_BREAK_BUFFER_FRAC       = _asc.ORB_BREAK_BUFFER_FRAC
ORB_ENTRY_WINDOW_MINUTES    = _asc.ORB_ENTRY_WINDOW_MINUTES
compute_session_vwap        = _asc.compute_session_vwap   # reused, not reimplemented
VWAP_MAX_VIX                = _asc.VWAP_MAX_VIX            # VWAP-reclaim VIX gate (ORB is NOT gated)
CORE_WATCHLIST              = _asc.CORE_WATCHLIST          # default ticker set for the backtest driver


# ==================================================================================
# ENTRY DETECTORS (historical replay)
# ==================================================================================
def detect_vwap_entry(session_df, vwap_series, vol_confirm=None):
    """Detect a VWAP-reclaim ENTRY in one session's minute bars. Returns (fire_pos, "LONG") or None.

    MUST stay logic-identical to the live `_vwap_reclaim_row` in alpha_scanner.py — same thresholds
    (imported, not copied), same VWAP (compute_session_vwap, passed in), same no-lookahead volume.
    `vol_confirm` (None = live INTRADAY_VOL_CONFIRM) exists ONLY for train-window parameter sweeps;
    leaving it None keeps exact live parity.

    ONE deliberate difference from the live version: the live detector asks "is this setup valid as
    of the final bar right now?" and therefore requires price to STILL be holding above VWAP at the
    last bar. This backtest version asks "did the setup FIRE at some bar, and which one?" — so the
    "still holding at the last bar" check is INTENTIONALLY OMITTED. Whether the reclaim later held or
    failed is the trade OUTCOME (the simulator's job), not an entry condition. Keeping that check here
    would silently discard every reclaim that fired and then failed — i.e. cherry-pick winners and
    inflate the win rate. So it is gone on purpose.

    Enforced in order: (1) price dipped below VWAP within the first VWAP_OPENING_WINDOW_MINUTES;
    (2) a later bar CLOSES back above VWAP (the reclaim bar = the fire point); (3) that bar's volume
    exceeds INTRADAY_VOL_CONFIRM x the causal average volume up to & including it; (4) the reclaim bar
    ALSO falls inside the 9:30-10:00 ET opening window (matches the live scanner). The fire bar index
    is returned; the simulator enters at the NEXT bar's open (never on the signal bar itself).
    """
    vc = INTRADAY_VOL_CONFIRM if vol_confirm is None else float(vol_confirm)
    df = session_df
    if df is None or len(df) < VWAP_OPENING_WINDOW_MINUTES + 2:
        return None
    vwap = vwap_series
    # Anchor the opening window to the ACTUAL 9:30 ET open (matches the live _vwap_reclaim_row),
    # not df.index[0] — BOTH the dip and the reclaim must fall inside 9:30-10:00 ET.
    _sd = df.index[0].date()
    market_open = pd.Timestamp(_sd.year, _sd.month, _sd.day, 9, 30, tz=df.index.tz)
    opening_cutoff = market_open + timedelta(minutes=VWAP_OPENING_WINDOW_MINUTES)   # 10:00 ET
    opening_mask = df.index < opening_cutoff
    below = df["close"] < vwap

    # (1) dip below VWAP within the opening window
    first_dip_pos = None
    for i in range(len(df)):
        if opening_mask[i] and below.iloc[i]:
            first_dip_pos = i
            break
    if first_dip_pos is None:
        return None

    # (2) first later bar that CLOSES back above VWAP = the reclaim / fire bar
    reclaim_pos = None
    for i in range(first_dip_pos + 1, len(df)):
        if df["close"].iloc[i] > vwap.iloc[i]:
            reclaim_pos = i
            break
    if reclaim_pos is None:
        return None

    # (4) the RECLAIM must fall inside the 9:30-10:00 ET opening window — keeps this backtest
    # logic-identical to the live _vwap_reclaim_row (was a 1:00 PM cutoff; changed 2026-09-01 so a
    # late reclaim at 10:59/11:35 is not counted as an entry, matching the live scanner).
    if df.index[reclaim_pos] >= opening_cutoff:
        return None

    # (3) reclaim-bar volume vs the CAUSAL average up to & including it (strict no-lookahead)
    avg_vol = df["volume"].iloc[:reclaim_pos + 1].mean()
    reclaim_vol = df["volume"].iloc[reclaim_pos]
    vol_ratio = reclaim_vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio < vc:
        return None

    # NOTE: no "still holding above VWAP at the last bar" check — see docstring. Entry only.
    return reclaim_pos, "LONG"


def detect_orb_entry(session_df, vol_confirm=None, min_range_pct=None, break_buffer_frac=None):
    """Detect an opening-range-breakout ENTRY in one session's minute bars. Returns (fire_pos, "UP"|"DOWN") or None.

    MUST stay logic-identical to the live `_orb_row` in alpha_scanner.py — same thresholds (imported,
    not copied) and same no-lookahead causal volume average.
    The three threshold args (None = live INTRADAY_VOL_CONFIRM / ORB_MIN_RANGE_PCT /
    ORB_BREAK_BUFFER_FRAC) exist ONLY for train-window parameter sweeps; leaving them None keeps
    exact live parity.

    ONE deliberate difference: the live version requires price to STILL be holding beyond the range at
    the last bar (a "valid right now" snapshot). This backtest version returns the FIRST bar that fired,
    with the "still holding" check INTENTIONALLY OMITTED — whether the breakout held or reversed is the
    trade OUTCOME for the simulator, not an entry condition. Keeping it would cherry-pick breakouts that
    happened to persist to 4pm and throw away every one that fired then failed.

    Enforced: opening range = high/low of the first ORB_RANGE_MINUTES bars; the range must be
    >= ORB_MIN_RANGE_PCT of price; the fire point is the first post-range bar — WITHIN the first
    ORB_ENTRY_WINDOW_MINUTES of the session — that CLOSES beyond the range by >= ORB_BREAK_BUFFER_FRAC
    x the range height, on volume > INTRADAY_VOL_CONFIRM x the causal running average. The fire bar
    index is returned; the simulator enters at the NEXT bar's open.
    """
    vc = INTRADAY_VOL_CONFIRM if vol_confirm is None else float(vol_confirm)
    mrp = ORB_MIN_RANGE_PCT if min_range_pct is None else float(min_range_pct)
    bbf = ORB_BREAK_BUFFER_FRAC if break_buffer_frac is None else float(break_buffer_frac)
    df = session_df
    if df is None or len(df) <= ORB_RANGE_MINUTES + 1:
        return None
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
    # opening range must be non-trivial
    if or_high <= 0 or (or_range / or_high) * 100 < mrp:
        return None
    buffer = or_range * bbf
    n_or = len(or_bars)
    entry_cutoff = session_start + timedelta(minutes=ORB_ENTRY_WINDOW_MINUTES)

    for i in range(len(post)):
        if post.index[i] >= entry_cutoff:
            break   # past the morning entry window — an opening-range break this late isn't the setup
        c = post["close"].iloc[i]
        v = post["volume"].iloc[i]
        # causal running average — bars up to & including this one only (strict no-lookahead)
        avg_vol = df["volume"].iloc[:n_or + i + 1].mean()
        if avg_vol <= 0 or v <= avg_vol * vc:
            continue
        if c > or_high + buffer:
            return n_or + i, "UP"      # fire bar index; no "still holding" check (see docstring)
        if c < or_low - buffer:
            return n_or + i, "DOWN"
    return None


# ==================================================================================
# TRADE SIMULATOR (historical replay) — turn a detector fire into a completed trade
# ==================================================================================
def simulate_trade(session_df, vwap_series, entry_pos, signal_type, direction, ticker="SPY"):
    """Simulate ONE trade from a detector fire to its exit — forward-only, no lookahead.

    Required signature is (session_df, vwap_series, entry_pos, signal_type, direction); `ticker`
    is an optional trailing arg purely so the returned row can carry the symbol (defaults to the
    only symbol this backtest replays).

    ENTRY: fill at the OPEN of the bar AFTER entry_pos. The detector returns the SIGNAL bar and you
    cannot realistically fill inside it, so the fill is always next-bar-open, never same-bar. If the
    signal bar is the last bar of the session there is no next bar to fill on -> return None (the
    trade simply can't be taken).

    EXIT (thesis-consistent, strictly forward-only — only bars from the fill bar onward are examined,
    all of which are after the signal bar; each decision uses only that bar's OWN close):
      * VWAP reclaim (LONG)       -> first later bar that CLOSES back BELOW VWAP (the reclaim failed);
      * ORB breakout UP (long)    -> first later bar that CLOSES back inside the range (close <= or_high);
      * ORB breakout DOWN (short) -> first later bar that CLOSES back above the range low (close >= or_low);
      * if none of those ever triggers -> exit at the session's FINAL real close.
    The opening range (or_high/or_low) is recomputed from the same first-ORB_RANGE_MINUTES window the
    detector used. "Final real close" is session_df's last bar; group_by_session already dropped
    half-days, so that last bar is the true session close (half-day handling is respected upstream).

    return_pct is the UNDERLYING's price move as a PERCENT, signed for the trade's direction
    (long: (exit-entry)/entry; short: (entry-exit)/entry — a short profits when price falls, so a
    positive return_pct always means a profitable trade). IMPORTANT: return_pct is NOT option P&L.
    A real options position would add leverage and theta decay that this price-only backtest does
    NOT model, so read these as the direction call's edge, not as an actual options return.
    """
    df = session_df
    n = len(df)
    fill_pos = entry_pos + 1
    if fill_pos >= n:                       # signal fired on the last bar: no next-bar fill possible
        return None

    sig = (signal_type or "").upper()
    direction_u = (direction or "").upper()
    is_long = direction_u in ("LONG", "UP")   # VWAP LONG and ORB UP are long; ORB DOWN is short

    entry_price = float(df["open"].iloc[fill_pos])
    entry_time = df.index[fill_pos]

    # Opening range for ORB exits — same window & (imported) threshold the detector used.
    or_high = or_low = None
    if "ORB" in sig:
        or_cutoff = df.index[0] + timedelta(minutes=ORB_RANGE_MINUTES)
        or_bars = df[df.index < or_cutoff]
        if not or_bars.empty:
            or_high = float(or_bars["high"].max())
            or_low = float(or_bars["low"].min())

    def _exit_triggered(i):
        c = float(df["close"].iloc[i])
        if "VWAP" in sig:
            return c < float(vwap_series.iloc[i])         # LONG reclaim failed: closed back below VWAP
        if "ORB" in sig:
            if direction_u == "UP" and or_high is not None:
                return c <= or_high                       # long breakout closed back inside the range
            if direction_u == "DOWN" and or_low is not None:
                return c >= or_low                        # short breakdown closed back inside the range
        return False

    # Forward-only exit scan: bars fill_pos..end, all strictly after the signal bar (no lookahead).
    exit_pos = n - 1                                      # default: session's final real close
    for i in range(fill_pos, n):
        if _exit_triggered(i):
            exit_pos = i
            break

    exit_price = float(df["close"].iloc[exit_pos])
    exit_time = df.index[exit_pos]

    # return_pct = signed UNDERLYING price move, in percent. NOT option P&L — a real options position
    # would add leverage and theta decay this backtest does not model.
    if is_long:
        return_pct = (exit_price - entry_price) / entry_price * 100.0
    else:                                                # short: profits when the underlying falls
        return_pct = (entry_price - exit_price) / entry_price * 100.0

    return {
        "ticker": ticker,
        "date": df.index[0].strftime("%Y-%m-%d"),
        "signal_type": signal_type,
        "direction": direction,
        "entry_time": entry_time.strftime("%H:%M ET"),
        "entry_price": round(entry_price, 2),
        "exit_time": exit_time.strftime("%H:%M ET"),
        "exit_price": round(exit_price, 2),
        "return_pct": round(return_pct, 3),
        "bars_held": exit_pos - fill_pos + 1,            # inclusive: fill bar through exit bar
    }


# ==================================================================================
# FULL BACKTEST DRIVER — fetch, session-split, detect, and simulate across many tickers
# ==================================================================================
TRADE_FIELDS = ["ticker", "date", "signal_type", "direction", "entry_time",
                "entry_price", "exit_time", "exit_price", "return_pct", "bars_held"]


def _load_real_yfinance():
    """Return the REAL yfinance module. alpha_backtest installs a yfinance STUB at import time
    (so importing alpha_scanner stays offline); the stub is a bare ModuleType with no __file__,
    so detect + evict it and import the real package for the daily VIX pull. alpha_scanner is
    already fully loaded, so swapping yfinance back to the real module now affects nothing."""
    stub = sys.modules.get("yfinance")
    if stub is not None and getattr(stub, "__file__", None) is None:
        del sys.modules["yfinance"]
    import yfinance as yf
    return yf


def _fetch_vix_by_date(start_date, end_date):
    """{date: daily ^VIX close} over the window (padded a few days before start) via yfinance --
    daily VIX needs no Polygon. Returns {} on any failure; a missing VIX simply means nothing gets
    gated, matching the live scanner (its gate only fires when vix is a real number > VWAP_MAX_VIX)."""
    try:
        yf = _load_real_yfinance()

        def _fmt(d):
            return d if isinstance(d, str) else d.strftime("%Y-%m-%d")

        # Pad the start back ~7 calendar days so the FIRST session still has a prior trading day's
        # VIX close available for the (lookahead-free) gate.
        pad_start = (datetime.strptime(_fmt(start_date), "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        end_plus = (datetime.strptime(_fmt(end_date), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        hist = yf.Ticker("^VIX").history(start=pad_start, end=end_plus, interval="1d")
        out = {}
        if hist is not None and not hist.empty and "Close" in hist.columns:
            for ts, close in hist["Close"].items():
                try:
                    out[ts.date()] = float(close)
                except Exception:
                    pass
        return out
    except Exception as e:
        print("  (VIX fetch failed: %s -- VWAP signals will NOT be VIX-gated)" % str(e)[:120])
        return {}


def _fmt_secs(s):
    s = int(max(0, s))
    if s < 90:
        return "%ds" % s
    if s < 5400:
        return "%dm" % (s // 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


# Session open->close returns per ticker, collected by run_backtest while it already holds the
# minute bars — so report()'s no-signal baseline never re-fetches through the free-tier throttle.
_BASELINE_OC = {}   # ticker -> {session_date: open->close return_pct}

# Per-process fetch caches so a parameter sweep (many run_backtest calls over the SAME window)
# pays the throttled Polygon pull exactly once; variations after the first are pure recompute.
_BARS_CACHE = {}    # (ticker, 'start', 'end') -> normalized minute df
_VIX_CACHE = {}     # ('start', 'end') -> {date: vix_close}


def _session_oc_returns(sessions):
    """{date: (close-open)/open %} for each session — the 'just be long intraday' return."""
    out = {}
    for d, s in sessions.items():
        try:
            o = float(s["open"].iloc[0])
            c = float(s["close"].iloc[-1])
            if o > 0:
                out[d] = (c - o) / o * 100.0
        except Exception:
            pass
    return out


def run_backtest(tickers, start_date, end_date, apply_vix_gate=True,
                 vol_confirm=None, orb_min_range_pct=None, orb_break_buffer_frac=None,
                 vwap_max_vix=None, write_csv=True, verbose=True,
                 csv_path="backtest_trades.csv"):
    """Backtest the VWAP-reclaim + ORB entry detectors across `tickers` over
    [start_date, end_date], simulating every fire into a completed trade.

    Threshold overrides (all default None = the live alpha_scanner.py constants, i.e. exact live
    parity): `vol_confirm` (both detectors), `orb_min_range_pct` / `orb_break_buffer_frac` (ORB),
    `vwap_max_vix` (the VWAP prior-day VIX gate). These exist for TRAIN-window sweeps only — they
    never touch the live scanner's constants. simulate_trade consumes none of these (its only
    shared constant is ORB_RANGE_MINUTES, which defines the opening range itself).
    `write_csv=False` skips the CSV write entirely (sweeps use this); `csv_path` names the output
    file — REGRESSION / test runs must pass a non-default path (e.g. backtest_trades_regression.csv)
    so they never clobber the canonical backtest_trades.csv. `verbose=False` silences per-ticker
    progress (fetch failures always print). Minute bars and VIX closes are cached per
    (ticker, window) in-process, so repeat calls over the same window recompute without re-fetching.

    Per ticker: fetch_minute_history -> group_by_session(drop_half_days=True) -> for each session
    compute_session_vwap, run BOTH detectors, and simulate_trade each fire.

    VIX gate (exact parity with the live scanner): daily ^VIX closes are pulled ONCE for the whole
    window (plus a few days before start). When apply_vix_gate is True a VWAP-reclaim fire on
    session date D is skipped iff the PRIOR trading day's VIX close > VWAP_MAX_VIX (strict,
    isinstance-guarded -- same comparison as _vwap_reclaim_row). Using the prior close, not D's own
    close, avoids lookahead: D's close isn't known until 16:00, long after the ~09:45 ET entry.
    ORB fires are NOT VIX-gated, matching _orb_row (which takes no vix).

    Every trade dict (the Prompt-3 fields) is collected, written to backtest_trades.csv, and the
    list is returned. Per-ticker progress with elapsed/ETA is printed as it runs (free-tier
    throttling makes a full-watchlist run slow).
    """
    tickers = list(tickers)
    max_vix = VWAP_MAX_VIX if vwap_max_vix is None else float(vwap_max_vix)
    vkey = (str(start_date), str(end_date))
    if apply_vix_gate:
        if vkey not in _VIX_CACHE:
            _VIX_CACHE[vkey] = _fetch_vix_by_date(start_date, end_date)
        vix_by_date = _VIX_CACHE[vkey]
    else:
        vix_by_date = {}
    # Shift the VIX series one trading day: each session's gate uses the PRIOR trading day's close
    # (the same-day close isn't known until 16:00, so using it for a ~09:45 entry is lookahead).
    # VIX and the sessions share one trading calendar, so each VIX date maps to the previous VIX
    # date's close = "yesterday's close" for that session.
    _vd = sorted(vix_by_date.keys())
    prior_vix = {_vd[i]: vix_by_date[_vd[i - 1]] for i in range(1, len(_vd))}
    if verbose:
        if apply_vix_gate:
            print("VIX gate ON: %d daily ^VIX close(s) loaded; VWAP fires skipped when the PRIOR trading "
                  "day's close > %s. ORB not gated (matches live)." % (len(vix_by_date), max_vix))
        else:
            print("VIX gate OFF (VWAP fires not filtered).")

    all_trades = []
    t0 = time.time()
    n = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        bkey = (ticker, str(start_date), str(end_date))
        df = _BARS_CACHE.get(bkey)
        if df is None:
            try:
                df = fetch_minute_history(ticker, start_date, end_date)
                _BARS_CACHE[bkey] = df
            except Exception as e:
                print("  %-6s FETCH FAILED: %s" % (ticker, str(e)[:120]))
                continue
        sessions = group_by_session(df, drop_half_days=True, verbose=False)
        # Feed the report's no-signal baseline while the bars are in hand (no re-fetch later).
        _BASELINE_OC.setdefault(ticker, {}).update(_session_oc_returns(sessions))
        tk_trades = []
        for d in sorted(sessions.keys()):
            s = sessions[d]
            vwap = compute_session_vwap(s)
            v = detect_vwap_entry(s, vwap, vol_confirm=vol_confirm)   # VWAP reclaim -- VIX-gated
            if v:
                pos, direction = v
                vix = prior_vix.get(d)                  # PRIOR trading day's VIX close (no lookahead)
                gated = apply_vix_gate and isinstance(vix, (int, float)) and vix > max_vix
                if not gated:
                    tr = simulate_trade(s, vwap, pos, "VWAP", direction, ticker=ticker)
                    if tr:
                        tk_trades.append(tr)
            o = detect_orb_entry(s, vol_confirm=vol_confirm,          # ORB breakout -- NOT gated
                                 min_range_pct=orb_min_range_pct,
                                 break_buffer_frac=orb_break_buffer_frac)
            if o:
                pos, direction = o
                tr = simulate_trade(s, vwap, pos, "ORB", direction, ticker=ticker)
                if tr:
                    tk_trades.append(tr)
        all_trades.extend(tk_trades)
        if verbose:
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (n - idx)
            print("  %-6s %2d trade(s) found   [%d/%d | elapsed %s | ~%s left]"
                  % (ticker, len(tk_trades), idx, n, _fmt_secs(elapsed), _fmt_secs(eta)))

    if write_csv:
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                w.writeheader()
                for t in all_trades:
                    w.writerow({k: t.get(k, "") for k in TRADE_FIELDS})
            print("\nWrote %d trade(s) to %s" % (len(all_trades), csv_path))
        except Exception as e:
            print("\nCSV write failed: %s" % e)

    return all_trades


# ==================================================================================
# REPORT — honest performance summary vs a no-signal baseline
# ==================================================================================
def _as_date(d):
    """Coerce str/date/datetime to a date (datetime checked first: it subclasses date)."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def report(trades, baseline_tickers, start_date, end_date):
    """Print performance stats by signal type (VWAP, ORB) and combined (ALL).

    Per bucket: trade count, win rate, average win %, average loss %, expectancy (mean
    return_pct), profit factor (gross wins / gross losses), and max drawdown on an
    equal-weight-per-trade equity curve — trades sorted chronologically by (date, entry_time),
    each contributing its return_pct in percentage points to a running sum.

    Honesty features:
      (1) any bucket with fewer than 30 trades is flagged "SAMPLE TOO SMALL — treat as noise";
      (2) a no-signal BASELINE — the average open->close return of `baseline_tickers` across every
          session scanned — prints alongside expectancy (delta in the 'vs BASE' column). If a
          signal's expectancy is not clearly above the baseline, the signal has no edge over just
          being long the underlying intraday. Baseline sessions reuse run_backtest's cached bars;
          only tickers this process never fetched are pulled fresh.
    """
    sd, ed = _as_date(start_date), _as_date(end_date)

    # ---- no-signal baseline: avg open->close across every session scanned ----
    base_rets = []
    for tk in baseline_tickers:
        oc = _BASELINE_OC.get(tk)
        if oc is None:                               # standalone use (e.g. report over a loaded CSV)
            try:
                df = fetch_minute_history(tk, sd, ed)
                oc = _session_oc_returns(group_by_session(df, drop_half_days=True, verbose=False))
                _BASELINE_OC[tk] = oc
            except Exception as e:
                print("  (baseline fetch failed for %s: %s)" % (tk, str(e)[:100]))
                continue
        base_rets.extend(r for d, r in oc.items() if sd <= d <= ed)
    baseline = (sum(base_rets) / len(base_rets)) if base_rets else None

    def bucket_stats(rows):
        rets = [float(t["return_pct"]) for t in rows]
        n = len(rets)
        if n == 0:
            return None
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r < 0]
        gross_w, gross_l = sum(wins), abs(sum(losses))
        pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
        eq = peak = mdd = 0.0
        for t in sorted(rows, key=lambda t: (str(t.get("date", "")), str(t.get("entry_time", "")))):
            eq += float(t["return_pct"])
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        return {"n": n, "win_rate": 100.0 * len(wins) / n,
                "avg_win": (sum(wins) / len(wins)) if wins else None,
                "avg_loss": (sum(losses) / len(losses)) if losses else None,
                "expectancy": sum(rets) / n, "pf": pf, "mdd": mdd}

    print("\n" + "=" * 96)
    print("BACKTEST REPORT  %s .. %s" % (sd, ed))
    print("=" * 96)
    if baseline is not None:
        print("BASELINE (no signal): %+.4f%%/session -- avg open->close across %d ticker-session(s)."
              % (baseline, len(base_rets)))
        print("A signal only has edge if its expectancy is CLEARLY above this number.\n")
    else:
        print("BASELINE unavailable (no session data) -- 'vs BASE' deltas omitted.\n")

    hdr = "  %-5s %5s %6s %9s %9s %9s %9s %7s %10s" % (
        "SIG", "N", "WIN%", "AVG WIN", "AVG LOSS", "EXPECT", "vs BASE", "PF", "MAXDD(pp)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, rows in (("VWAP", [t for t in trades if t.get("signal_type") == "VWAP"]),
                        ("ORB", [t for t in trades if t.get("signal_type") == "ORB"]),
                        ("ALL", list(trades))):
        st = bucket_stats(rows)
        if st is None:
            print("  %-5s %5d   — no trades —" % (label, 0))
            continue
        # round-then-add-0.0 kills the float-noise negative zero ("-0.0000" would misread as below-baseline)
        vsbase = (round(st["expectancy"] - baseline, 4) + 0.0) if baseline is not None else None
        warn = "   ** SAMPLE TOO SMALL -- treat as noise (n<30) **" if st["n"] < 30 else ""
        print("  %-5s %5d %6.1f %9s %9s %+9.4f %9s %7s %10.2f%s" % (
            label, st["n"], st["win_rate"],
            ("%+.3f" % st["avg_win"]) if st["avg_win"] is not None else "n/a",
            ("%+.3f" % st["avg_loss"]) if st["avg_loss"] is not None else "n/a",
            round(st["expectancy"], 4) + 0.0,
            ("%+.4f" % vsbase) if vsbase is not None else "n/a",
            ("inf" if st["pf"] == float("inf") else "%.2f" % st["pf"]),
            st["mdd"], warn))

    print("\n  Notes: return_pct is the UNDERLYING's move, not option P&L (no leverage/theta modeled).")
    print("  Baseline = buy at open, sell at close, equal-weight over all scanned sessions. It is a")
    print("  LONG benchmark; a SHORT signal's edge vs 'doing nothing' (0%) also matters on its own.")
    print("  MAXDD is percentage points on an equal-weight, one-trade-at-a-time equity curve.")


# ==================================================================================
# MAGNET-PIN FORWARD TEST — evaluate the pins logged by alpha_scanner.record_magnet
# ==================================================================================
def _fetch_daily_closes(ticker, start_date, end_date, throttle_seconds=12, timeout=30):
    """Polygon DAILY closes over [start_date, end_date] inclusive -> {date: close}. One page (a year
    of daily bars is far under the 50000 limit). Shares the global _poly_throttle; retries on 429.
    Uses the UTC date of each bar (Polygon stamps daily bars at 00:00, so the UTC date == trading
    date; converting to ET could shift it back a day). Raises if no API key."""
    key = _polygon_key()
    if not key:
        raise RuntimeError("No Polygon API key. Set POLYGON_API_KEY in the environment, or add "
                           "POLYGON_KEY to your Streamlit secrets (.streamlit/secrets.toml).")
    frm = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
    to = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
    url = POLYGON_BASE_URL + "/v2/aggs/ticker/" + ticker + "/range/1/day/" + frm + "/" + to
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    retries = 0
    while True:
        _poly_throttle(throttle_seconds)
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:
            retries += 1
            if retries > 4:
                raise RuntimeError("Polygon rate limit (429) persisted for %s daily %s..%s" % (ticker, frm, to))
            time.sleep(30 * retries)
            continue
        if resp.status_code != 200:
            raise RuntimeError("Polygon daily request failed (%s) for %s %s..%s: %s"
                               % (resp.status_code, ticker, frm, to, resp.text[:200]))
        break
    out = {}
    for b in (resp.json().get("results", []) or []):
        d = datetime.fromtimestamp(b["t"] / 1000, tz=timezone.utc).date()
        out[d] = float(b["c"])
    return out


def magnet_forward_report(history_path="magnet_history.json", throttle_seconds=12, verbose=True,
                          csv_out=None):
    """Evaluate the magnet-pin forward test logged by alpha_scanner.record_magnet. STRICTLY no
    lookahead: a logged prediction (date, spot, max_pain, expiry) is scored ONLY once its expiry has
    passed, against the underlying's close AT that expiry. Prints the same column layout as the
    ORB/VWAP report (as a convergence trade: long if spot<pin, short if spot>pin) plus a convergence
    block. Measures the UNDERLYING's move toward the pin, NOT option P&L (no theta/spread modeled)."""
    if not os.path.exists(history_path):
        print("No %s yet -- run the scanner (Magnet tab) at least once to start logging predictions." % history_path)
        return []
    with open(history_path) as f:
        hist = json.load(f)
    today = date.today()

    per_ticker, pending, total_logged, earliest_pending = {}, 0, 0, None
    for tk, entries in hist.items():
        for e in entries or []:
            total_logged += 1
            try:
                exp = datetime.strptime(e["expiry"], "%Y-%m-%d").date()
                spot, mp = float(e["spot"]), float(e["max_pain"])
                ldate = datetime.strptime(e["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if exp >= today:                       # expiry not passed -> can't score without lookahead
                pending += 1
                earliest_pending = exp if earliest_pending is None else min(earliest_pending, exp)
                continue
            per_ticker.setdefault(tk, []).append({"ldate": ldate, "spot": spot, "mp": mp, "exp": exp})

    n_eval = sum(len(v) for v in per_ticker.values())
    print("\n" + "=" * 96)
    print("MAGNET-PIN FORWARD TEST  (as of %s)" % today)
    print("=" * 96)
    print("Logged snapshots: %d across %d ticker(s). Evaluable now (expiry passed): %d | still pending: %d."
          % (total_logged, len(hist), n_eval, pending))
    if n_eval == 0:
        if earliest_pending:
            print("No prediction's expiry has passed yet -- earliest pending expiry is %s. Check back after it."
                  % earliest_pending)
        else:
            print("Nothing logged yet -- run the scanner a few times, then re-run this after an expiry passes.")
        print("=" * 96)
        return []

    trades = []
    for tk, preds in per_ticker.items():
        lo = min(p["ldate"] for p in preds)
        hi = max(p["exp"] for p in preds)
        try:
            closes = _fetch_daily_closes(tk, lo - timedelta(days=5), hi + timedelta(days=1),
                                         throttle_seconds=throttle_seconds)
        except Exception as ex:
            if verbose:
                print("  (daily fetch failed for %s: %s)" % (tk, str(ex)[:90]))
            continue
        if not closes:
            continue
        sdays = sorted(closes.keys())
        for p in preds:
            prior = [x for x in sdays if x <= p["exp"]]     # nearest trading-day close <= expiry
            if not prior:
                continue
            cexp = closes[prior[-1]]
            if p["spot"] <= 0 or p["mp"] <= 0:
                continue
            long_ret = (cexp - p["spot"]) / p["spot"] * 100.0        # baseline: long over the horizon
            if p["spot"] < p["mp"]:
                ret, direction = long_ret, "UP"                      # spot below pin -> predict UP (long)
            elif p["spot"] > p["mp"]:
                ret, direction = -long_ret, "DOWN"                   # spot above pin -> predict DOWN (short)
            else:
                continue
            init_dist = abs(p["spot"] - p["mp"]) / p["mp"] * 100.0
            final_dist = abs(cexp - p["mp"]) / p["mp"] * 100.0
            trades.append({"return_pct": ret, "base_ret": long_ret, "direction": direction,
                           "date": p["ldate"].isoformat(), "hold_days": (p["exp"] - p["ldate"]).days,
                           "init_dist": init_dist, "final_dist": final_dist,
                           "converged": final_dist < init_dist,
                           "gap_closed": ((init_dist - final_dist) / init_dist) if init_dist > 0 else 0.0})

    if not trades:
        print("No evaluable predictions after fetching outcomes (all daily fetches failed?).")
        print("=" * 96)
        return []

    baseline = sum(t["base_ret"] for t in trades) / len(trades)

    def _stats(rows):
        r = [t["return_pct"] for t in rows]
        if not r:
            return None
        wins = [x for x in r if x > 0]; losses = [x for x in r if x < 0]
        gw, gl = sum(wins), abs(sum(losses))
        pf = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
        eq = peak = mdd = 0.0
        for t in sorted(rows, key=lambda t: t["date"]):
            eq += t["return_pct"]; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        return {"n": len(r), "win_rate": 100.0 * len(wins) / len(r),
                "avg_win": (sum(wins) / len(wins)) if wins else None,
                "avg_loss": (sum(losses) / len(losses)) if losses else None,
                "expectancy": sum(r) / len(r), "pf": pf, "mdd": mdd}

    print("BASELINE (no signal): %+.4f%% -- avg LONG move over the same horizons (%d prediction(s))."
          % (baseline, len(trades)))
    print("A pin only has edge if its expectancy is CLEARLY above this.\n")
    hdr = "  %-9s %5s %6s %9s %9s %9s %9s %7s %10s" % (
        "SIG", "N", "WIN%", "AVG WIN", "AVG LOSS", "EXPECT", "vs BASE", "PF", "MAXDD(pp)")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, rows in (("MAGNET-UP", [t for t in trades if t["direction"] == "UP"]),
                        ("MAGNET-DN", [t for t in trades if t["direction"] == "DOWN"]),
                        ("ALL", trades)):
        s = _stats(rows)
        if s is None:
            print("  %-9s %5d   -- none --" % (label, 0)); continue
        vsbase = round(s["expectancy"] - baseline, 4) + 0.0
        warn = "   ** n<30: treat as noise **" if s["n"] < 30 else ""
        print("  %-9s %5d %6.1f %9s %9s %+9.4f %9s %7s %10.2f%s" % (
            label, s["n"], s["win_rate"],
            ("%+.3f" % s["avg_win"]) if s["avg_win"] is not None else "n/a",
            ("%+.3f" % s["avg_loss"]) if s["avg_loss"] is not None else "n/a",
            round(s["expectancy"], 4) + 0.0, "%+.4f" % vsbase,
            ("inf" if s["pf"] == float("inf") else "%.2f" % s["pf"]), s["mdd"], warn))

    pct_conv = 100.0 * sum(1 for t in trades if t["converged"]) / len(trades)
    avg_init = sum(t["init_dist"] for t in trades) / len(trades)
    avg_final = sum(t["final_dist"] for t in trades) / len(trades)
    avg_gap = 100.0 * sum(t["gap_closed"] for t in trades) / len(trades)
    print("\n  CONVERGENCE (distance to pin, %d prediction(s)):" % len(trades))
    print("    ended CLOSER to the pin : %.1f%%" % pct_conv)
    print("    avg distance at log     : %.2f%%" % avg_init)
    print("    avg distance at expiry  : %.2f%%" % avg_final)
    print("    avg gap closed          : %+.1f%%  (100%% = landed exactly on the pin; negative = moved away)" % avg_gap)

    print("\n  Notes:")
    print("  - Measures the UNDERLYING's move toward the pin, NOT option P&L. Real option P&L would be")
    print("    WORSE: theta decay + bid/ask spread are not modeled.")
    print("  - Predictions are DAILY snapshots, so entries targeting the same expiry are correlated")
    print("    (not independent) -- N is optimistic for statistical significance.")
    print("  - No lookahead: each prediction is scored only after its expiry, vs the close at expiry.")
    print("=" * 96)
    if csv_out:
        cols = ["date", "direction", "hold_days", "return_pct", "base_ret",
                "init_dist", "final_dist", "converged", "gap_closed"]
        try:
            with open(csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for t in trades:
                    w.writerow({k: t.get(k) for k in cols})
            print("Wrote %d convergence row(s) to %s  (feed to --cost-check)." % (len(trades), csv_out))
        except Exception as e:
            print("  (could not write %s: %s)" % (csv_out, str(e)[:80]))
    return trades


# ==================================================================================
# OPTION-COST STRESS TEST — does an underlying edge survive spread + theta?
# ==================================================================================
def cost_check(csv_path, signal_type=None, iv=0.30, hold_days_override=None, verbose=True):
    """Rough option-cost stress test for ANY signal's trade CSV (ORB, VWAP, or magnet convergence).
    Reads the per-trade SIGNED underlying return (`return_pct` column) and models buying an ATM
    option (delta 0.5, given `iv`): gross option return = leverage x underlying move; NET subtracts a
    round-trip bid/ask SPREAD and THETA (scaled by each trade's real hold). Prints NET expectancy and
    NET win% across a DTE x spread grid, plus the break-even underlying edge. This is a rough SANITY
    check, NOT precise option P&L (delta-only, ATM, one IV) -- but the grid brackets realistic costs.

    Generic across signal shapes:
      * `signal_type` filters a mixed CSV (e.g. "ORB" / "VWAP"); None = all rows.
      * Hold duration for theta is auto-detected per row: `bars_held` (intraday MINUTES -> /390 of a
        trading day) or `hold_days` (already in trading days, e.g. magnet convergence). If neither is
        present, uses `hold_days_override` or assumes 1.0 day.
    """
    if not os.path.exists(csv_path):
        print("No such CSV: %s" % csv_path)
        return
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    if signal_type:
        rows = [r for r in rows if str(r.get("signal_type", "")).upper() == signal_type.upper()]
    data = []
    for r in rows:
        try:
            data.append((r, float(r["return_pct"])))
        except Exception:
            pass
    if not data:
        print("No usable rows (need a numeric 'return_pct' column) in %s%s."
              % (csv_path, " for signal " + signal_type if signal_type else ""))
        return

    def hold_frac(r):
        if hold_days_override is not None:
            return float(hold_days_override)
        v = r.get("bars_held")
        if v not in (None, ""):
            try:
                return float(v) / 390.0                 # intraday minutes -> fraction of a trading day
            except Exception:
                pass
        v = r.get("hold_days")
        if v not in (None, ""):
            try:
                return float(v)                          # already trading days (e.g. magnet)
            except Exception:
                pass
        return 1.0                                       # unknown hold -> assume a full day

    n = len(data)
    rets = [x for _, x in data]
    raw_exp = sum(rets) / n
    wins = sum(1 for x in rets if x > 0)
    avg_hold = sum(hold_frac(r) for r, _ in data) / n

    def lev(dte):
        return 0.5 / (0.4 * iv * math.sqrt(dte / 252.0))   # ATM: L = delta / (premium/S); S cancels
    def theta_daily(dte):
        return 0.5 / dte                                   # ATM daily theta ~ 0.5/DTE of premium

    print("=" * 92)
    print("OPTION-COST STRESS TEST  |  %s%s  |  n=%d"
          % (os.path.basename(csv_path), "  signal=" + signal_type.upper() if signal_type else "", n))
    print("=" * 92)
    print("Raw UNDERLYING expectancy : %+.4f%%/trade   (win%% %.1f  |  avg hold %.2f trading day(s))"
          % (raw_exp, 100.0 * wins / n, avg_hold))
    print("Modeled as buying an ATM option (delta 0.5, IV %.0f%%). Values below are %% of premium/trade."
          % (iv * 100))
    print("Gross = leverage x underlying move; NET subtracts round-trip SPREAD + THETA (theta scaled by")
    print("each trade's real hold). Delta-only (ignores gamma). NOT precise option P&L.\n")
    print("  %-26s %7s %9s %11s %9s" % ("SCENARIO", "LEVER", "GROSS", "NET EXP", "NET WIN%"))
    print("  " + "-" * 66)
    for dte, lbl in [(2, "0-2 DTE (weekly/0DTE)"), (7, "~weekly (7 DTE)"), (30, "~monthly (30 DTE)")]:
        L, td = lev(dte), theta_daily(dte)
        gross_exp = 100.0 * (L * raw_exp / 100.0)
        for sp, spl in [(0.02, "2%"), (0.05, "5%"), (0.10, "10%")]:
            nets = [L * (x / 100.0) - (sp + td * hold_frac(r)) for r, x in data]
            exp = 100.0 * sum(nets) / n
            w = 100.0 * sum(1 for v in nets if v > 0) / n
            print("  %-26s %6.0fx %+8.2f%% %+10.2f%% %8.1f%%" % (lbl + " | sp " + spl, L, gross_exp, exp, w))

    Lw = lev(7)
    be = 0.05 / Lw * 100.0
    print("\n  Break-even: at 7 DTE (%.0fx) you'd need a raw underlying edge >= %.3f%%/trade to cover a 5%%"
          % (Lw, be))
    print("  round-trip spread ALONE (pre-theta). This signal delivers %+.4f%%." % raw_exp)
    print("\n  CAVEATS: delta-only (ignores gamma, which slightly helps long-option winners); assumes ATM,")
    print("  IV %.0f%%, buy-at-ask / sell-at-bid; ignores commissions & slippage beyond the spread. A rough"
          % (iv * 100))
    print("  sanity check -- real results vary with strike / IV / execution.")
    print("=" * 92)


# ==================================================================================
# TRAIN-WINDOW PARAMETER SWEEP — compare threshold variations on TRAIN data only
# ==================================================================================
SWEEP_PARAMS = ("vol_confirm", "orb_min_range_pct", "orb_break_buffer_frac", "vwap_max_vix")


def sweep_train(variations, start, end, tickers=None):
    """Run run_backtest once per variation (a dict of threshold overrides) over the SAME window
    and print a comparison table: label, N, win%, expectancy, PF, vs baseline. Returns
    {label: trades}.

    Each variation dict may set any of vol_confirm / orb_min_range_pct / orb_break_buffer_frac /
    vwap_max_vix (missing keys = the live alpha_scanner.py values) plus an optional 'label'. An
    empty dict {} is the live-defaults control row — include it so variations are compared
    against current behavior.

    Fetch cost: minute bars + VIX are cached per (ticker, window) in-process, so the FIRST
    variation pays the throttled Polygon pull and every further variation is pure recompute
    (seconds). No CSV is written (sweeps must not clobber backtest_trades.csv).

    TRAIN ONLY by design: pass your train window (e.g. 2026-01-10..2026-05-10). Nothing here
    knows about a test window — evaluating the picked candidate on held-out data is a separate,
    manual, ONE-TIME run_backtest call, so the test set can't be overfit by iterated sweeping.
    """
    tickers = list(tickers) if tickers else list(CORE_WATCHLIST)
    results = {}
    order = []
    for i, var in enumerate(variations):
        v = dict(var)
        label = v.pop("label", None) or (
            ", ".join("%s=%s" % kv for kv in sorted(v.items())) if v else "live-defaults")
        unknown = sorted(set(v) - set(SWEEP_PARAMS))
        if unknown:
            raise ValueError("unknown sweep parameter(s) %s in variation %r (valid: %s)"
                             % (unknown, label, list(SWEEP_PARAMS)))
        print("\n--- sweep %d/%d: %s ---" % (i + 1, len(variations), label))
        trades = run_backtest(tickers, start, end, apply_vix_gate=True,
                              write_csv=False, verbose=(i == 0), **v)
        print("  -> %d trade(s)" % len(trades))
        results[label] = trades
        order.append(label)

    # No-signal baseline over the swept window (identical for every variation: same sessions).
    sd, ed = _as_date(start), _as_date(end)
    base_rets = [r for tk in tickers
                 for d, r in _BASELINE_OC.get(tk, {}).items() if sd <= d <= ed]
    baseline = (sum(base_rets) / len(base_rets)) if base_rets else None

    print("\n" + "=" * 92)
    print("TRAIN SWEEP  %s .. %s  (%d ticker(s); pick candidates here, then ONE manual test run)"
          % (sd, ed, len(tickers)))
    print("=" * 92)
    if baseline is not None:
        print("BASELINE (no signal): %+.4f%%/session over %d ticker-session(s)\n"
              % (baseline, len(base_rets)))
    hdr = "  %-38s %6s %6s %9s %7s %9s" % ("VARIATION", "N", "WIN%", "EXPECT", "PF", "vs BASE")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label in order:
        rets = [float(t["return_pct"]) for t in results[label]]
        if not rets:
            print("  %-38s %6d   -- no trades --" % (label[:38], 0))
            continue
        n = len(rets)
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r < 0]
        gross_w, gross_l = sum(wins), abs(sum(losses))
        pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
        expect = sum(rets) / n
        vsbase = (round(expect - baseline, 4) + 0.0) if baseline is not None else None
        small = "  (n<30: noise)" if n < 30 else ""
        print("  %-38s %6d %6.1f %+9.4f %7s %9s%s" % (
            label[:38], n, 100.0 * len(wins) / n, round(expect, 4) + 0.0,
            ("inf" if pf == float("inf") else "%.2f" % pf),
            ("%+.4f" % vsbase) if vsbase is not None else "n/a", small))
    print("\n  NOTE: picking the best row here is in-sample selection — confirm the ONE candidate")
    print("  you pick with a single run_backtest on the held-out test window before believing it.")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Backtest the Alpha Scanner VWAP + ORB entry detectors.")
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated tickers (default: CORE_WATCHLIST from alpha_scanner.py).")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: ~6 months before end).")
    ap.add_argument("--end", default=None,
                    help="End date YYYY-MM-DD (default: yesterday; the free tier blocks same-day intraday).")
    ap.add_argument("--no-vix-gate", action="store_true", help="Disable the VWAP VIX gate.")
    ap.add_argument("--csv", default="backtest_trades.csv",
                    help="Output CSV path. Use backtest_trades_regression.csv for test/regression "
                         "runs so the canonical backtest_trades.csv is never clobbered.")
    ap.add_argument("--throttle", type=int, default=12,
                    help="Per-page throttle seconds used only for the pre-run time estimate (fetcher default is 12).")
    ap.add_argument("--magnet-report", action="store_true",
                    help="Instead of the ORB/VWAP backtest, evaluate the magnet-pin forward test "
                         "(magnet_history.json logged by the scanner) — convergence to the pin by expiry.")
    ap.add_argument("--magnet-csv", default=None,
                    help="With --magnet-report, also write the convergence rows to this CSV (feed to --cost-check).")
    ap.add_argument("--cost-check", default=None, metavar="CSV",
                    help="Instead of a backtest, run the option-cost stress test on any trade CSV "
                         "(ORB/VWAP/magnet): net expectancy + win%% after spread + theta, across a DTE x "
                         "spread grid. Combine with --signal to filter and --iv to set the vol assumption.")
    ap.add_argument("--signal", default=None,
                    help="With --cost-check, filter to one signal_type (e.g. ORB or VWAP).")
    ap.add_argument("--iv", type=float, default=0.30,
                    help="With --cost-check, the assumed ATM implied vol (default 0.30).")
    args = ap.parse_args()

    if args.cost_check:
        cost_check(args.cost_check, signal_type=args.signal, iv=args.iv)
        raise SystemExit(0)

    if args.magnet_report:
        magnet_forward_report(csv_out=args.magnet_csv)
        raise SystemExit(0)

    today = datetime.now(ET_ZONE).date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else (today - timedelta(days=1))
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else (end_date - timedelta(days=182))
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else list(CORE_WATCHLIST)
    apply_gate = not args.no_vix_gate

    # Rough pre-run time estimate -- dominated by free-tier throttle between Polygon pages.
    days = (end_date - start_date).days
    trading_days = max(1, int(days * 5 / 7))
    BARS_PER_DAY_EST = 800                                    # RTH is ~391; Polygon may return some extended-hours bars
    approx_pages = max(1, math.ceil(trading_days * BARS_PER_DAY_EST / 45000))
    # every request is globally throttled now (not just follow-up pages), so cost ~= pages * (throttle + request time)
    est_total = len(tickers) * approx_pages * (args.throttle + 4)
    print("Backtest plan: %d ticker(s) | %s .. %s (%d calendar / ~%d trading days) | VIX gate %s"
          % (len(tickers), start_date, end_date, days, trading_days, "ON" if apply_gate else "OFF"))
    print("Rough time estimate: ~%s  (~%d Polygon page(s)/ticker at %ds throttle). Actual varies with "
          "page counts and rate limits." % (_fmt_secs(est_total), approx_pages, args.throttle))
    print("Starting pull now...\n")

    trades = run_backtest(tickers, start_date, end_date, apply_vix_gate=apply_gate,
                          csv_path=args.csv)

    vwap_n = sum(1 for t in trades if t.get("signal_type") == "VWAP")
    orb_n = sum(1 for t in trades if t.get("signal_type") == "ORB")
    print("\nSUMMARY: %d trade(s) (%d VWAP, %d ORB) | %s .. %s | %d ticker(s) scanned -> %s"
          % (len(trades), vwap_n, orb_n, start_date, end_date, len(tickers), args.csv))

    report(trades, tickers, start_date, end_date)
