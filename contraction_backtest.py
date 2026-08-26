"""contraction_backtest.py — multi-day volatility-contraction breakout backtest on DAILY bars.

A deliberately different thesis from the intraday VWAP/ORB engine in alpha_backtest.py:
volatility compresses over days (ATR% percentile in the bottom of its trailing range), then
price breaks out of the compression range on volume. Daily bars come from yfinance, so this
avoids the Polygon minute-bar throttle entirely.

Reuses alpha_backtest's report() unchanged: simulate_contraction_trade returns the exact
TRADE_FIELDS dict shape, and run_contraction_backtest seeds alpha_backtest's _BASELINE_OC
with daily open->close returns so report()'s no-signal baseline needs no minute fetches.
(In report()'s table, CONTRACTION trades appear in the ALL row; the VWAP/ORB rows print
"no trades" — expected, not a bug.)
"""

import sys
import os
import bisect
from datetime import date, datetime, timedelta
from math import exp, log, sqrt
from statistics import NormalDist, stdev

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpha_backtest as ab   # stub-imports the scanner (offline); gives report/_BASELINE_OC etc.


# ------------------------------------------------------------------
# DATA — daily bars via yfinance (no Polygon, no throttle)
# ------------------------------------------------------------------
def fetch_daily_history(ticker, start_date, end_date):
    """Daily OHLCV for [start_date, end_date] inclusive via yfinance (auto-adjusted).
    Returns a DataFrame with a tz-naive date DatetimeIndex and lowercase
    open/high/low/close/volume columns; empty DataFrame on failure.

    alpha_backtest installs a yfinance STUB at import (to keep the scanner import offline),
    so the real package is loaded via ab._load_real_yfinance() — same trick as its VIX pull."""
    cols = ["open", "high", "low", "close", "volume"]
    try:
        yf = ab._load_real_yfinance()

        def _fmt(d):
            return d if isinstance(d, str) else d.strftime("%Y-%m-%d")

        end_plus = (datetime.strptime(_fmt(end_date), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=_fmt(start_date), end=end_plus,
                                         interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return pd.DataFrame(columns=cols)
        hist = hist.rename(columns={c: str(c).lower() for c in hist.columns})
        if not all(c in hist.columns for c in cols):
            return pd.DataFrame(columns=cols)
        out = hist[cols].copy()
        out = out.dropna(subset=["close"])
        out.index = pd.DatetimeIndex([ts.date() for ts in out.index])   # tz-naive dates
        return out.sort_index()
    except Exception as e:
        print("  (daily fetch failed for %s: %s)" % (ticker, str(e)[:120]))
        return pd.DataFrame(columns=cols)


# ------------------------------------------------------------------
# DETECTOR — contraction flag, then breakout scan (strictly causal)
# ------------------------------------------------------------------
def detect_contraction_breakout(df, atr_window=14, pct_window=100, pct_threshold=20.0,
                                breakout_lookback=14, breakout_buffer_pct=0.5,
                                scan_days=5, vol_confirm=1.5, vol_window=20):
    """Find volatility-contraction -> breakout events in daily bars. Strictly causal: every
    quantity for day X uses only bars up to and including X.

    Per day T (first evaluable T = atr_window + pct_window - 1 = index 113, i.e. ~100 prior
    days of valid ATR plus the TR/ATR warmup):
      1. TR% = true range / same-day close * 100; ATR% = `atr_window`-day rolling mean of TR%.
      2. Percentile rank of ATR%[T] vs the trailing `pct_window` days INCLUSIVE of T, using
         the mid-rank convention: 100 * (# strictly below + 0.5 * # tied, excluding T itself)
         / window size. (Plain strictly-below ranks EVERY day of a flat-ATR stretch as 0th
         percentile — "contracted" forever; mid-rank puts an all-tied window at ~50%. On real
         float ATR values ties are vanishingly rare, so this matches strictly-below in practice.)
      3. Contracted if that percentile < `pct_threshold` (bottom 20% by default).
      4. On a contraction flag at day C (a new episode), the breakout level is FIXED at
         max(high of the last `breakout_lookback` days as of C) * (1 + buffer). Scan forward
         up to `scan_days` trading days for the first day whose CLOSE exceeds the level on
         volume > `vol_confirm` x the `vol_window`-day average volume (causal, through that day).
      5. One cycle, one event at most. After the cycle resolves (breakout found OR the scan
         window expires), no re-flag occurs until the contraction LAPSES — i.e. a
         non-contracted day re-arms the detector, and the next contracted day after that
         starts a NEW episode. ("Skip re-flagging until a new contraction occurs": a long
         unbroken quiet regime is ONE episode, not a fresh signal every day.)

    Returns a list of {"date": breakout DatetimeIndex value, "contraction_start_date":
    flag-day DatetimeIndex value} in chronological order.
    """
    n = len(df)
    first = atr_window + pct_window - 1
    if df is None or n <= first:
        return []

    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    tr_pct = tr / df["close"] * 100.0
    atr_pct = tr_pct.rolling(atr_window).mean()
    atr_vals = atr_pct.values

    def contracted(t):
        window = atr_vals[t - pct_window + 1: t + 1]
        x = atr_vals[t]
        if pd.isna(x) or pd.isna(window).any():
            return False
        less = (window < x).sum()
        tied = (window == x).sum() - 1          # exclude day T itself
        pct_rank = 100.0 * (less + 0.5 * max(tied, 0)) / len(window)
        return pct_rank < pct_threshold

    events = []
    armed = True
    i = first
    while i < n:
        is_c = contracted(i)
        if is_c and armed:
            c_pos = i
            level = float(df["high"].iloc[c_pos - breakout_lookback + 1: c_pos + 1].max())
            level *= (1.0 + breakout_buffer_pct / 100.0)
            fired = None
            last_scan = min(c_pos + scan_days, n - 1)
            for j in range(c_pos + 1, last_scan + 1):
                vol_avg = float(df["volume"].iloc[j - vol_window + 1: j + 1].mean())
                if (float(df["close"].iloc[j]) > level and vol_avg > 0
                        and float(df["volume"].iloc[j]) > vol_confirm * vol_avg):
                    fired = j
                    break
            if fired is not None:
                events.append({"date": df.index[fired],
                               "contraction_start_date": df.index[c_pos]})
            armed = False                                  # cycle consumed
            i = (fired if fired is not None else last_scan) + 1
            continue
        if not is_c:
            armed = True                                   # contraction lapsed -> re-arm
        i += 1
    return events


# ------------------------------------------------------------------
# SIMULATOR — same dict shape as alpha_backtest.simulate_trade
# ------------------------------------------------------------------
def simulate_contraction_trade(df, breakout_date, hold_days=5, ticker="SPY"):
    """Simulate one contraction-breakout trade. Entry = the NEXT trading day's OPEN after
    breakout_date (never a same-bar fill). Exit = the CLOSE exactly `hold_days` trading days
    after entry, or the last available bar's close if the series runs out. Returns None if
    the breakout is the final bar (no next open to fill on).

    Returns the same dict shape as alpha_backtest.simulate_trade (TRADE_FIELDS), with
    signal_type="CONTRACTION", direction="LONG", and dates in the time fields (daily bars
    have no intraday times). bars_held is inclusive fill-bar-through-exit-bar, matching
    simulate_trade, so a full hold is hold_days + 1. return_pct is the UNDERLYING's move —
    not option P&L (no leverage/theta modeled)."""
    try:
        pos = df.index.get_loc(pd.Timestamp(breakout_date))
    except KeyError:
        return None
    e = pos + 1
    if e >= len(df):
        return None
    x = min(e + hold_days, len(df) - 1)
    entry = float(df["open"].iloc[e])
    exit_ = float(df["close"].iloc[x])
    if entry <= 0:
        return None
    return {
        "ticker": ticker,
        "date": df.index[e].strftime("%Y-%m-%d"),
        "signal_type": "CONTRACTION",
        "direction": "LONG",
        "entry_time": df.index[e].strftime("%Y-%m-%d"),
        "entry_price": round(entry, 2),
        "exit_time": df.index[x].strftime("%Y-%m-%d"),
        "exit_price": round(exit_, 2),
        "return_pct": round((exit_ - entry) / entry * 100.0, 3),
        "bars_held": x - e + 1,
    }


# ------------------------------------------------------------------
# DRIVER — mirrors run_backtest so alpha_backtest.report() works unchanged
# ------------------------------------------------------------------
_DAILY_CACHE = {}   # (ticker, 'start', 'end') -> daily df, so repeat runs over one window don't re-fetch


def run_contraction_backtest(tickers, start_date, end_date, hold_days=5, vol_confirm=None,
                             event_start=None, write_csv=True,
                             csv_path="contraction_trades.csv", verbose=True):
    """Loop tickers: fetch daily bars, detect contraction breakouts, simulate each into a
    trade. Returns the trades list (TRADE_FIELDS dicts) and writes contraction_trades.csv
    (own file — never touches backtest_trades.csv). `vol_confirm=None` uses the detector's live
    default (1.5); pass a number to sweep the breakout volume gate. Daily bars are cached per
    (ticker, window) in-process, so running several vol_confirm variations over the same window
    re-fetches nothing.

    `event_start=None` scores every detected event; set it to a date to keep only events whose
    BREAKOUT falls on/after it. This is the walk-forward / out-of-sample knob: fetch from well
    BEFORE the evaluation window so the detector's 113-bar warmup (14 ATR + 100 percentile) is
    fed by PRIOR bars — causal, exactly what live trading does — while only events inside the
    evaluation window are simulated and scored. Without it, a short OOS window is nearly all
    warmup and scores almost nothing.

    Seeds alpha_backtest._BASELINE_OC with each session's daily open->close return — an empty
    dict even for no-data tickers, so ab.report never falls back to a Polygon minute fetch —
    letting ab.report(trades, tickers, start_date, end_date) compute its no-signal baseline
    from these daily bars instead of minute history."""
    all_trades = []
    tickers = list(tickers)
    det_kwargs = {} if vol_confirm is None else {"vol_confirm": vol_confirm}
    ev_start = ab._as_date(event_start) if event_start is not None else None
    for idx, tk in enumerate(tickers, start=1):
        ab._BASELINE_OC.setdefault(tk, {})          # ensure report() never Polygon-fetches this ticker
        ckey = (tk, str(start_date), str(end_date))
        df = _DAILY_CACHE.get(ckey)
        if df is None:
            df = fetch_daily_history(tk, start_date, end_date)
            _DAILY_CACHE[ckey] = df
        if df.empty:
            if verbose:
                print("  %-6s no daily data" % tk)
            continue
        oc = ab._BASELINE_OC[tk]                     # daily open->close, same semantic as intraday baseline
        for i in range(len(df)):
            o = float(df["open"].iloc[i])
            if o > 0:
                oc[df.index[i].date()] = (float(df["close"].iloc[i]) - o) / o * 100.0
        events = detect_contraction_breakout(df, **det_kwargs)
        if ev_start is not None:      # OOS: warmup used prior bars; score only in-window events
            events = [e for e in events if e["date"].date() >= ev_start]
        tk_trades = []
        for ev in events:
            tr = simulate_contraction_trade(df, ev["date"], hold_days=hold_days, ticker=tk)
            if tr:
                tk_trades.append(tr)
        all_trades.extend(tk_trades)
        if verbose:
            print("  %-6s %2d event(s) -> %2d trade(s)   [%d/%d]"
                  % (tk, len(events), len(tk_trades), idx, len(tickers)))
    if write_csv:
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ab.TRADE_FIELDS)
                w.writeheader()
                for t in all_trades:
                    w.writerow({k: t.get(k, "") for k in ab.TRADE_FIELDS})
            print("\nWrote %d trade(s) to %s" % (len(all_trades), csv_path))
        except Exception as e:
            print("\nCSV write failed: %s" % e)
    return all_trades


# ==================================================================================
# UNIVERSE — widen beyond the hand-picked watchlist, with survivorship handled explicitly
# ==================================================================================
SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_universe(window_start=None, include_removed=True):
    """Build a wide US large-cap universe from Wikipedia's S&P 500 tables.
    Returns (current, removed, meta): `current` = today's constituents; `removed` = tickers
    dropped from the index on/after `window_start`, i.e. names that WERE members during the
    backtest window but are not today.

    SURVIVORSHIP: backtesting `current` alone is biased — it keeps only the survivors, and it
    includes names ADDED to the index precisely BECAUSE they had already run, while silently
    dropping everything that failed or lagged its way out. Adding `removed` recovers much of that
    missing tail (Yahoo still serves some delisted names, e.g. FRCB), so `current + removed` is
    materially less biased. It is NOT point-in-time perfect: exact membership per date isn't
    reconstructed, and delisted names Yahoo has purged are unrecoverable. Run BOTH and diff them —
    that difference is an estimate of the bias itself.
    """
    import io as _io
    import requests as _rq
    html = _rq.get(SP500_WIKI, headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=30).text
    tables = pd.read_html(_io.StringIO(html))
    cur = [str(t).strip().upper().replace(".", "-") for t in tables[0][tables[0].columns[0]]]
    removed = []
    if include_removed and len(tables) > 1:
        ch = tables[1]
        dcol = next((c for c in ch.columns if "Effective Date" in str(c)), None)
        rcol = next((c for c in ch.columns if "Removed" in str(c) and "Ticker" in str(c)), None)
        if dcol is not None and rcol is not None:
            ws = ab._as_date(window_start) if window_start else None
            for _, row in ch.iterrows():
                tk = str(row[rcol]).strip().upper()
                if not tk or tk in ("NAN", "-", "—", "NONE"):
                    continue
                try:
                    d = pd.to_datetime(str(row[dcol])).date()
                except Exception:
                    continue
                if ws is None or d >= ws:
                    removed.append(tk.replace(".", "-"))
    cur = sorted(set(t for t in cur if t and t != "NAN"))
    removed = sorted(set(removed) - set(cur))
    return cur, removed, {"n_current": len(cur), "n_removed": len(removed)}


def prefetch_daily_batch(tickers, start, end, chunk=60, verbose=True):
    """Batch-download daily bars straight into _DAILY_CACHE so walk_forward finds everything
    cached (one yfinance request per `chunk` tickers instead of one per ticker — the difference
    between minutes and an hour at universe scale). Tickers with no data are cached as an EMPTY
    frame so walk_forward skips them instead of retrying one-by-one. Returns (ok, failed)."""
    yf = ab._load_real_yfinance()
    fs, fe = ab._as_date(start), ab._as_date(end)
    end_plus = (datetime.combine(fe, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
    cols = ["open", "high", "low", "close", "volume"]
    ok, failed, tickers = [], [], list(tickers)
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, start=fs.strftime("%Y-%m-%d"), end=end_plus, interval="1d",
                              auto_adjust=True, group_by="ticker", progress=False, threads=True)
        except Exception as e:
            if verbose:
                print("  batch %d..%d FAILED: %s" % (i, i + len(part), str(e)[:70]))
            for tk in part:
                _DAILY_CACHE[(tk, str(fs), str(fe))] = pd.DataFrame(columns=cols)
                failed.append(tk)
            continue
        for tk in part:
            d = None
            try:
                d = raw[tk] if len(part) > 1 else raw
                d = d.rename(columns={c: str(c).lower() for c in d.columns})
                d = d[cols].dropna(subset=["close"])
            except Exception:
                d = None
            if d is None or d.empty:
                _DAILY_CACHE[(tk, str(fs), str(fe))] = pd.DataFrame(columns=cols)
                failed.append(tk)
                continue
            d.index = pd.DatetimeIndex([ts.date() for ts in d.index])
            _DAILY_CACHE[(tk, str(fs), str(fe))] = d.sort_index()
            ok.append(tk)
        if verbose:
            print("  prefetch %4d/%d  (ok %d, no-data %d)" % (min(i + chunk, len(tickers)), len(tickers), len(ok), len(failed)))
    return ok, failed


def make_rv_selector(top_n=100, mode="top", min_bars=30):
    """Build a POINT-IN-TIME universe selector: each fold gets the `top_n` tickers ranked by
    realized volatility measured ONLY over that fold's own train window.

    This is the difference between a real finding and a hindsight artifact. A hand-picked list of
    today's high-beta winners embeds the answer; ranking by RV using nothing but bars inside
    [train_start, test_start) is something a live system could have done that morning, with no
    knowledge of what happened next. mode="top" takes the highest-RV names, "bottom" the lowest —
    running both is what separates a genuine volatility interaction from selection restated.
    """
    cache = {}

    def sel(tr_s, te_s, data, first_bar):
        scored = []
        for tk, df in data.items():
            if first_bar[tk] > tr_s:                 # must cover the whole train window
                continue
            if tk not in cache:                      # log-returns computed once per ticker
                c = np.maximum(df["close"].values.astype(float), 1e-9)
                cache[tk] = (df.index[1:], np.diff(np.log(c)))
            idx, lr = cache[tk]
            i0 = idx.searchsorted(pd.Timestamp(tr_s), "left")
            i1 = idx.searchsorted(pd.Timestamp(te_s), "left")   # strictly BEFORE the test window
            seg = lr[i0:i1]
            if len(seg) < min_bars:
                continue
            rv = float(seg.std(ddof=1)) * sqrt(252.0)
            if rv > 0:
                scored.append((rv, tk))
        scored.sort(reverse=(mode == "top"))
        return [tk for _, tk in scored[:top_n]]

    return sel


# ==================================================================================
# WALK-FORWARD VALIDATION — does the thesis generalize across regimes, or only one?
# ==================================================================================
def _add_months(d, m):
    return (pd.Timestamp(d) + pd.DateOffset(months=m)).date()


def _compress(nums):
    """[1,2,3,7,8] -> '1-3,7-8' (compact fold-range reporting)."""
    if not nums:
        return ""
    nums = sorted(nums)
    out, s, p = [], nums[0], nums[0]
    for x in nums[1:]:
        if x == p + 1:
            p = x
            continue
        out.append("%d" % s if s == p else "%d-%d" % (s, p))
        s = p = x
    out.append("%d" % s if s == p else "%d-%d" % (s, p))
    return ",".join(out)


def _perf(rets):
    """report()-compatible stats for a list of return_pct values (None if empty)."""
    if not rets:
        return None
    w = [r for r in rets if r > 0]
    l = [r for r in rets if r < 0]
    gw, gl = sum(w), abs(sum(l))
    pf = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    return {"n": len(rets), "win": 100.0 * len(w) / len(rets),
            "exp": sum(rets) / len(rets), "pf": pf}


def walk_forward(tickers, full_start, full_end, train_months=6, test_months=2,
                 step_months=2, vol_confirm=1.2, hold_days=5, low_conf_n=15,
                 dominant_share=0.30, per_fold_report=False, universe_fn=None):
    """Walk-forward validation of the contraction-breakout thesis.

    Folds: fold i has train = [full_start + i*step_months, +train_months) and test = the
    following test_months, non-overlapping with its own train. Both slide by step_months until a
    test window would run past full_end. With the defaults (test=step=2mo) the test windows tile
    the span contiguously and never overlap, so pooling cannot double-count a trade; if
    step_months < test_months the windows WOULD overlap and pooled stats are invalid — warned.

    NO per-fold re-tuning: one fixed vol_confirm across every fold (default 1.2 = relaxed_vol).
    This asks whether the thesis generalizes at all, not whether it can be curve-fit per regime.

    Ticker coverage: a ticker joins fold k only if its history already covers that fold's TRAIN
    window (first bar <= train_start) — that train window is what feeds the detector's 113-bar
    warmup for the fold's test events. Late-listing names (recent IPOs / spin-offs) are therefore
    skipped for early folds and join once they qualify, rather than erroring or silently thinning
    the sample; a per-fold exclusion report is printed. Each fold's FAIR baseline is computed over
    the SAME participating tickers, so signal and benchmark always see one universe.

    Causality / warmup: bars are fetched once over [full_start, full_end] and the detector runs
    ONCE per ticker across the whole continuous series — what a live system actually does — then
    each fold scores only events whose BREAKOUT falls inside its test window. The detector's
    lookback is bounded (14 ATR + 100 pct = 113 bars), so any fold's flags depend only on bars
    at/before that day; nothing after a signal informs it. A fold whose train window supplies
    fewer than 113 bars is flagged WARMUP-SHORT (train_months=6 ~= 130 bars is the practical
    minimum). Trade exits run their full hold even past a fold boundary (truncating would distort
    the return); a trade belongs to the fold its breakout fired in.

    Robustness: each fold's SHARE of the pooled return sum is reported, and any fold exceeding
    `dominant_share` (default 0.30) is flagged DOMINANT — meaning the pooled result is not robust
    to dropping that single fold. Leave-one-out pooled expectancies are printed for the biggest
    contributors so that dependence is measured, not asserted.

    Returns {"folds": [...], "pooled": {...}, "coverage": {...}}.
    """
    fs, fe = ab._as_date(full_start), ab._as_date(full_end)
    tickers = list(tickers)
    if step_months < test_months:
        print("WARNING: step_months (%d) < test_months (%d) -> test windows OVERLAP; pooled stats "
              "would double-count trades." % (step_months, test_months))

    # ---- one fetch + one continuous detection pass per ticker ----
    data, evs, first_bar, no_data = {}, {}, {}, []
    for tk in tickers:
        ab._BASELINE_OC.setdefault(tk, {})
        ckey = (tk, str(fs), str(fe))
        df = _DAILY_CACHE.get(ckey)
        if df is None:
            df = fetch_daily_history(tk, fs, fe)
            _DAILY_CACHE[ckey] = df
        if df.empty:
            no_data.append(tk)
            continue
        data[tk] = df
        first_bar[tk] = df.index[0].date()
        oc = ab._BASELINE_OC[tk]
        for i in range(len(df)):
            o = float(df["open"].iloc[i])
            if o > 0:
                oc[df.index[i].date()] = (float(df["close"].iloc[i]) - o) / o * 100.0
        evs[tk] = detect_contraction_breakout(df, vol_confirm=vol_confirm)
    if not data:
        print("No daily data for any ticker.")
        return {"folds": [], "pooled": None, "coverage": {"no_data": no_data, "excluded": {}}}

    # hold-matched baseline series per ticker: entry date -> buy-open/sell-close-N-later return
    hold_series = {}
    for tk, df in data.items():
        lst = []
        for i in range(len(df) - hold_days):
            o = float(df["open"].iloc[i])
            if o > 0:
                lst.append((df.index[i].date(),
                            (float(df["close"].iloc[i + hold_days]) - o) / o * 100.0))
        hold_series[tk] = lst

    # ---- fold schedule ----
    folds, i = [], 0
    while True:
        tr_s = _add_months(fs, i * step_months)
        tr_e = _add_months(tr_s, train_months)   # test opens where train closes
        te_e = _add_months(tr_e, test_months)    # exclusive
        if te_e > fe:                            # test would exceed full_end -> stop
            break
        folds.append((tr_s, tr_e, te_e))
        i += 1
    if not folds:
        print("No folds fit in [%s, %s] with train=%dmo test=%dmo." % (fs, fe, train_months, test_months))
        return {"folds": [], "pooled": None, "coverage": {"no_data": no_data, "excluded": {}}}

    # ---- per-fold universe. Default: every ticker whose history covers that fold's train window.
    # `universe_fn(train_start, test_start, data, first_bar)` overrides it with a POINT-IN-TIME
    # rule (see make_rv_selector) — it may only look at bars before test_start, and its output is
    # re-filtered by the same history requirement so a selector can never smuggle in a ticker that
    # lacked warmup. The fold's baseline uses this SAME set, so signal and benchmark never diverge.
    fold_univ = []
    for (tr_s, te_s, te_e) in folds:
        eligible = [tk for tk in data if first_bar[tk] <= tr_s]
        if universe_fn is None:
            fold_univ.append(eligible)
        else:
            elig = set(eligible)
            fold_univ.append([tk for tk in universe_fn(tr_s, te_s, data, first_bar) if tk in elig])
    fold_univ_sets = [set(u) for u in fold_univ]

    # ---- baselines bucketed in ONE pass over the bars, not folds x tickers x bars.
    # At universe scale (~700 tickers x 39 folds) the naive filter is ~48M comparisons; bisect
    # onto the (contiguous, non-overlapping) test windows makes it O(bars). Overlapping windows
    # (step < test) can't be bucketed this way, so that case keeps the exact per-fold filter.
    fold_base = [[] for _ in folds]
    if step_months >= test_months:
        starts = [f[1] for f in folds]                           # test starts, ascending
        for tk, lst in hold_series.items():
            for d, r in lst:
                k0 = bisect.bisect_right(starts, d) - 1
                if k0 < 0:
                    continue
                tr_s, te_s, te_e = folds[k0]
                if d < te_e and tk in fold_univ_sets[k0]:        # te_e is exclusive
                    fold_base[k0].append(r)
    else:
        for k0, (tr_s, te_s, te_e) in enumerate(folds):
            te_last = te_e - timedelta(days=1)
            fold_base[k0] = [r for tk in fold_univ[k0]
                             for d, r in hold_series[tk] if te_s <= d <= te_last]

    # ---- pass 1: build every fold (shares need the pooled total, so print after) ----
    rows, pooled_rets, pooled_base = [], [], []
    excluded = {}                                 # ticker -> [fold numbers it was skipped for]
    for k, (tr_s, te_s, te_e) in enumerate(folds, start=1):
        te_last = te_e - timedelta(days=1)
        part = fold_univ[k - 1]
        for tk in data:
            if first_bar[tk] > tr_s:
                excluded.setdefault(tk, []).append(k)
        trades = []
        for tk in part:
            for e in evs[tk]:
                if te_s <= e["date"].date() <= te_last:
                    tr = simulate_contraction_trade(data[tk], e["date"],
                                                    hold_days=hold_days, ticker=tk)
                    if tr:
                        trades.append(tr)
        rets = [float(t["return_pct"]) for t in trades]
        base = fold_base[k - 1]
        b5 = (sum(base) / len(base)) if base else 0.0
        st = _perf(rets)
        pooled_rets.extend(rets)
        pooled_base.extend(base)
        # worst-case warmup = bars in the train window (a participant listing exactly at tr_s)
        warm_bars = 0
        if part:
            ridx = data[part[0]].index
            warm_bars = int(((ridx >= pd.Timestamp(tr_s)) & (ridx < pd.Timestamp(te_s))).sum())
        rows.append({"fold": k, "test": (te_s, te_last), "stats": st, "base5": b5,
                     "vs5": (round(st["exp"] - b5, 4) + 0.0) if st else None,
                     "rets": rets, "trades": trades, "sum": sum(rets), "n_part": len(part),
                     "warm_bars": warm_bars})

    # ---- shares of the pooled return sum (the dominance test) ----
    total = sum(pooled_rets)
    for r in rows:
        r["share"] = (r["sum"] / total) if abs(total) > 1e-9 else None
        r["dominant"] = bool(r["share"] is not None and r["share"] > dominant_share)

    # ---- print ----
    hdr = "  %-4s %-23s %4s %5s %6s %10s %7s %11s %7s  %s" % (
        "FOLD", "TEST WINDOW", "TK", "N", "WIN%", "EXPECT", "PF", "vs5dBASE", "SHARE%", "FLAG")
    print("\n" + "=" * 122)
    print("WALK-FORWARD  %s .. %s  |  train=%dmo test=%dmo step=%dmo  |  vol_confirm=%s FIXED (no re-tuning)"
          % (fs, fe, train_months, test_months, step_months, vol_confirm))
    print("=" * 122)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        flags = []
        if r["warm_bars"] and r["warm_bars"] < 113:
            flags.append("WARMUP-SHORT(%d<113)" % r["warm_bars"])
        if r["stats"] and r["stats"]["n"] < low_conf_n:
            flags.append("LOW-CONF(n<%d)" % low_conf_n)
        if r["dominant"]:
            flags.append("** DOMINANT FOLD -- pooled result is not robust to removing this fold **")
        win = "%s..%s" % (r["test"][0], r["test"][1])
        sh = ("%6.1f" % (100.0 * r["share"])) if r["share"] is not None else "     -"
        if r["stats"] is None:
            print("  %-4d %-23s %4d %5d    -- no trades --                    %s  %s"
                  % (r["fold"], win, r["n_part"], 0, sh, " ".join(flags)))
            continue
        st = r["stats"]
        print("  %-4d %-23s %4d %5d %6.1f %+10.4f %7s %+11.4f %s  %s" % (
            r["fold"], win, r["n_part"], st["n"], st["win"], round(st["exp"], 4) + 0.0,
            ("inf" if st["pf"] == float("inf") else "%.2f" % st["pf"]), r["vs5"], sh,
            " ".join(flags)))

    # ---- ticker coverage report ----
    print("\n  TICKER COVERAGE (a ticker joins a fold only once its history covers that fold's train window)")
    if no_data:
        shown = ", ".join(no_data[:20]) + (" ... +%d more" % (len(no_data) - 20) if len(no_data) > 20 else "")
        print("    no data in range, excluded from ALL folds (%d): %s" % (len(no_data), shown))
    if excluded:
        order = sorted(excluded, key=lambda t: -len(excluded[t]))
        for tk in order[:15]:
            print("    %-6s first bar %s -> skipped folds %s  (%d of %d)"
                  % (tk, first_bar[tk], _compress(excluded[tk]), len(excluded[tk]), len(folds)))
        if len(order) > 15:
            print("    ... and %d more late-listing ticker(s) skipped for their early folds"
                  % (len(order) - 15))
    if not excluded and not no_data:
        print("    all %d ticker(s) had full history for every fold." % len(data))

    # ---- pooled: the real answer ----
    pst = _perf(pooled_rets)
    pb5 = (sum(pooled_base) / len(pooled_base)) if pooled_base else 0.0
    scored = [r for r in rows if r["stats"]]
    pos = [r for r in scored if r["stats"]["exp"] > 0]
    neg = [r for r in scored if r["stats"]["exp"] < 0]
    beat = [r for r in scored if r["vs5"] is not None and r["vs5"] > 0]
    dom = [r for r in rows if r["dominant"]]
    print("\n" + "-" * 122)
    if pst:
        print("POOLED across %d fold(s) -- THE REAL ANSWER" % len(folds))
        print("  N=%d | win%%=%.1f | expectancy=%+.4f%% | PF=%s | fair %dd baseline=%+.4f%% | vs baseline=%+.4f"
              % (pst["n"], pst["win"], round(pst["exp"], 4) + 0.0,
                 ("inf" if pst["pf"] == float("inf") else "%.2f" % pst["pf"]),
                 hold_days, pb5, round(pst["exp"] - pb5, 4) + 0.0))
        print("  folds net-positive (expectancy>0): %d | net-negative: %d | zero-trade: %d"
              % (len(pos), len(neg), len(folds) - len(scored)))
        print("  folds beating the fair %dd baseline: %d of %d scored" % (hold_days, len(beat), len(scored)))
        if pst["n"] < low_conf_n * 2:
            print("  ** pooled N is small -- treat even the pooled number as weak evidence **")

        # leave-one-out robustness for the biggest contributors
        print("\n  ROBUSTNESS -- pooled expectancy with each top contributor REMOVED:")
        for r in sorted(scored, key=lambda x: -(x["share"] if x["share"] is not None else 0))[:5]:
            loo = [v for o in rows if o["fold"] != r["fold"] for v in o["rets"]]
            lp = _perf(loo)
            lv = (round(lp["exp"], 4) + 0.0) if lp else 0.0
            print("    drop fold %-3d (share %5.1f%%, n=%3d) -> pooled expectancy %+.4f%%  (was %+.4f%%, delta %+.4f)"
                  % (r["fold"], 100.0 * (r["share"] or 0), r["stats"]["n"], lv,
                     round(pst["exp"], 4) + 0.0, round(lv - pst["exp"], 4) + 0.0))
        if dom:
            print("  ** %d DOMINANT fold(s): %s -- the pooled number leans on a single window; NOT robust **"
                  % (len(dom), ", ".join(str(r["fold"]) for r in dom)))
        else:
            print("  no single fold exceeds %.0f%% of the pooled return sum -- no one window carries the result."
                  % (100.0 * dominant_share))
    else:
        print("POOLED: no trades in any fold.")
    print("\n  Baseline is HOLD-MATCHED: buy any day's open, sell the close %d sessions later, over the" % hold_days)
    print("  same test sessions AND the same participating tickers -- the fair benchmark for a %d-day-hold" % hold_days)
    print("  rule. Returns are the UNDERLYING's move: no costs, slippage or option theta.")
    return {"folds": rows,
            "pooled": {"stats": pst, "base5": pb5, "rets": pooled_rets, "base_rets": pooled_base,
                       "trades": [t for r in rows for t in r["trades"]]},
            "data": data,
            "coverage": {"no_data": no_data, "excluded": excluded, "first_bar": first_bar}}


# ==================================================================================
# COST MODEL — does the edge survive commissions, spread and (for options) theta?
# ==================================================================================
def apply_costs(signal_rets, base_rets=None, hold_days=5, label="pooled trades",
                share_costs_bps=(2, 5, 20, 50),
                opt_dtes=(14, 30, 45, 60), opt_ivs=(0.25, 0.40, 0.60),
                opt_delta=0.50, opt_spread_pct=3.0, opt_commission_pct=0.35):
    """Charge realistic costs against a list of UNDERLYING return_pct values.

    Two tracks, deliberately separated by how much they assume:

    1) SHARES (low assumption): net = underlying_move - round_trip_cost, where the round trip is
       the bid/ask spread crossed twice plus commission, quoted in bps of notional. Almost no
       modelling risk — it applies directly to what the backtest measured.

    2) OPTIONS (HIGH assumption, illustrative): the system actually trades options, where a
       positive underlying edge can still lose because theta is a fixed drag. Modelled as
           option_ret = leverage * underlying_ret - theta_drag - spread - commission   (floored -100%)
           premium_pct = 0.4 * IV * sqrt(DTE/365)      (standard ATM premium approximation)
           leverage    = delta / premium_pct           (delta-$ per premium-$)
           theta_drag  = 1 - sqrt((DTE - hold)/DTE)    (sqrt-time decay of extrinsic value)
       Swept over a DTE x IV grid because those two dominate. Known simplifications, ALL of which
       matter: linear delta ignores gamma (understates both big wins and the deceleration of
       losses); it ignores vega, and a volatility-contraction breakout is exactly when IV should
       EXPAND, which would help a long call (so this is conservative); it assumes an ATM strike
       held to exit with no management. Treat the grid as an order-of-magnitude answer, not a
       substitute for replaying real option chains.

    `base_rets` (the hold-matched no-signal returns) is charged the SAME costs, which is the
    honest comparison: random option buying bleeds theta too, so the difference isolates the
    timing edge from the structural drag.
    """
    s = _perf(signal_rets)
    if not s:
        print("No trades to cost.")
        return None
    b = _perf(base_rets) if base_rets else None

    print("\n" + "=" * 112)
    print("COST MODEL  --  %s  (N=%d, hold=%dd)" % (label, s["n"], hold_days))
    print("=" * 112)
    print("GROSS (underlying move, no costs):")
    print("  expectancy %+.4f%% | win %.1f%% | PF %s%s"
          % (round(s["exp"], 4) + 0.0, s["win"],
             ("inf" if s["pf"] == float("inf") else "%.2f" % s["pf"]),
             ("  | fair %dd baseline %+.4f%% | edge %+.4f"
              % (hold_days, b["exp"], round(s["exp"] - b["exp"], 4) + 0.0)) if b else ""))
    print("\nBREAKEVEN (assumption-free):")
    print("  round-trip cost that zeroes the strategy: %.4f%% (%.0f bps)"
          % (round(s["exp"], 4) + 0.0, s["exp"] * 100))
    if b:
        print("  NOTE: the edge vs baseline is ~cost-INVARIANT -- both sides pay the same round trip,")
        print("  so per-trade costs cancel in the difference. Costs decide ABSOLUTE profitability.")

    # ---- shares ----
    print("\nSHARES (round-trip spread + commission, bps of notional):")
    hdr = "  %-18s %8s %11s %7s %8s  %s" % ("SCENARIO", "COST", "NET-EXP", "WIN%", "PF", "VERDICT")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    names = {2: "liquid megacap", 5: "typical", 20: "wide / small-cap", 50: "punitive"}
    for bps in share_costs_bps:
        c = bps / 100.0
        st = _perf([x - c for x in signal_rets])
        verdict = "survives" if st["exp"] > 0 else "DEAD"
        if b and st["exp"] <= (b["exp"] - c):
            verdict += " (but <= baseline)"
        print("  %-18s %8s %+11.4f %7.1f %8s  %s" % (
            names.get(bps, "%d bps" % bps), "%d bps" % bps, round(st["exp"], 4) + 0.0,
            st["win"], ("inf" if st["pf"] == float("inf") else "%.2f" % st["pf"]), verdict))

    # ---- options ----
    print("\nOPTIONS (long ATM call: delta %.2f, spread %.1f%% of premium, commission %.2f%%)"
          % (opt_delta, opt_spread_pct, opt_commission_pct))
    print("  HIGH-ASSUMPTION MODEL -- see docstring; ignores gamma & vega (IV expansion on a")
    print("  breakout would help a long call, so these figures lean conservative).")
    hdr2 = "  %-4s %5s %7s %6s %7s %8s %10s %6s %7s %10s %8s" % (
        "DTE", "IV", "PREM%", "LEV", "DRAG%", "BE-MOVE", "NET-EXP", "WIN%", "PF", "BASE-EXP", "EDGE")
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    grid = []
    for dte in opt_dtes:
        for iv in opt_ivs:
            prem = 0.4 * iv * sqrt(dte / 365.0)                 # fraction of spot
            if prem <= 0 or dte <= hold_days:
                continue
            lev = opt_delta / prem
            theta = (1.0 - sqrt(max(dte - hold_days, 0) / float(dte))) * 100.0
            drag = theta + opt_spread_pct + opt_commission_pct
            be = drag / lev                                     # underlying move needed to break even
            os_ = _perf([max(lev * x - drag, -100.0) for x in signal_rets])
            ob = _perf([max(lev * x - drag, -100.0) for x in base_rets]) if base_rets else None
            edge = (os_["exp"] - ob["exp"]) if ob else None
            grid.append({"dte": dte, "iv": iv, "net": os_["exp"], "edge": edge})
            print("  %-4d %4.0f%% %7.2f %6.1f %7.2f %8.2f %+10.3f %6.1f %7s %10s %8s" % (
                dte, iv * 100, prem * 100, lev, drag, be, round(os_["exp"], 3) + 0.0, os_["win"],
                ("inf" if os_["pf"] == float("inf") else "%.2f" % os_["pf"]),
                ("%+.3f" % ob["exp"]) if ob else "n/a",
                ("%+.3f" % edge) if edge is not None else "n/a"))
    pos = [g for g in grid if g["net"] > 0]
    print("\n  option cells with POSITIVE net expectancy: %d of %d" % (len(pos), len(grid)))
    if grid:
        best = max(grid, key=lambda g: g["net"])
        print("  best cell: DTE %d @ IV %.0f%% -> net %+.3f%%%s"
              % (best["dte"], best["iv"] * 100, best["net"],
                 ("  (edge over random option buying %+.3f)" % best["edge"]) if best["edge"] is not None else ""))
    print("\n  BE-MOVE = underlying move needed just to cover theta+spread+commission. Any signal")
    print("  whose average move is below that line loses money on options even with a real edge.")
    return {"gross": s, "baseline": b, "option_grid": grid}


# ==================================================================================
# OPTION REPRICING — exact Black-Scholes, per-trade IV estimated from MEASURED realized vol
#
# DATA REALITY (checked 2026-07-13, do not re-litigate without re-probing): there are NO
# historical option chains available here. Polygon's options data returns
# "You are not entitled to this data" on this plan (and its free tier caps stock history at
# ~2 years), and yfinance has never exposed historical chains — only a live snapshot. So the
# option leg CANNOT be backtested against real premiums. This module therefore does the next
# most honest thing: it prices a synthetic ATM call with Black-Scholes, taking IV from each
# trade's own MEASURED realized volatility at entry rather than from a guessed grid.
#
# Upgraded from assumption to measurement: per-trade IV level, gamma (exact BS, not linear
# delta), and the actual hold length (truncated trades included correctly).
# Still assumed, irreducibly, without real chains: IV = RV x a risk-premium factor (swept);
# how IV moves over the hold (swept); an exactly-ATM strike; no volatility skew or smile.
# ==================================================================================
_N = NormalDist()


def _bs_call(S, K, T, sigma, r):
    """Black-Scholes call price; falls back to intrinsic on degenerate inputs."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return S * _N.cdf(d1) - K * exp(-r * T) * _N.cdf(d1 - v)


def _realized_vol(df, pos, window=20):
    """Annualized realized vol from daily log returns over the `window` bars ENDING at `pos`.
    Callers must pass the bar BEFORE entry: at the entry OPEN you only know closes through the
    prior session, so using the entry day's own close would be lookahead."""
    if pos < window or pos >= len(df):
        return None
    c = [float(x) for x in df["close"].iloc[pos - window:pos + 1]]
    lr = [log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i] > 0 and c[i - 1] > 0]
    if len(lr) < 5:
        return None
    try:
        return stdev(lr) * sqrt(252.0)
    except Exception:
        return None


def reprice_options(trades, data, dte=30, iv_premium=1.2, iv_expansion=1.0, rv_window=20,
                    spread_pct=3.0, commission_pct=0.35):
    """Reprice each underlying trade as a long ATM call, Black-Scholes at entry and exit.

    IV_entry = (measured realized vol at entry) * iv_premium   [RV is computed causally]
    IV_exit  = IV_entry * iv_expansion                          [vega, made explicit]
    Strike   = the entry price (exactly ATM). Hold length comes from the trade's own bars_held,
    so boundary-truncated trades decay for the right number of days. Returns (net_rets, rvs).
    """
    r = ab._asc.RISK_FREE_RATE          # reuse the scanner's rate, never redefine it
    out, rvs = [], []
    for t in trades:
        df = data.get(t["ticker"])
        if df is None:
            continue
        try:
            pos = df.index.get_loc(pd.Timestamp(t["entry_time"]))
        except Exception:
            continue
        rv = _realized_vol(df, pos - 1, rv_window)      # causal: prior session's close
        if not rv or rv <= 0:
            continue
        S0, S1 = float(t["entry_price"]), float(t["exit_price"])
        hold = max(int(t["bars_held"]) - 1, 0)          # actual sessions held
        p0 = _bs_call(S0, S0, dte / 365.0, rv * iv_premium, r)
        if p0 <= 0:
            continue
        p1 = _bs_call(S1, S0, max(dte - hold, 0) / 365.0, rv * iv_premium * iv_expansion, r)
        out.append(max((p1 - p0) / p0 * 100.0 - spread_pct - commission_pct, -100.0))
        rvs.append(rv)
    return out, rvs


def option_study(trades, data, base_rets_trades=None, dte_list=(30, 45), rv_window=20,
                 iv_premiums=(1.0, 1.2, 1.4), iv_expansions=(0.9, 1.0, 1.2),
                 spread_pct=3.0, commission_pct=0.35, label="pooled trades"):
    """Black-Scholes option study on real trades with per-trade measured-RV-derived IV.
    Prints the measured RV distribution, a base case, an iv_premium x iv_expansion sensitivity
    grid, and a breakdown by entry-RV tercile (which directly tests the 'only low-IV names work'
    conclusion the earlier approximation suggested)."""
    print("\n" + "=" * 112)
    print("OPTION REPRICING (exact Black-Scholes; per-trade IV from MEASURED realized vol)  --  %s" % label)
    print("=" * 112)
    print("DATA REALITY: no historical option chains exist on these sources (Polygon options = NOT")
    print("ENTITLED on this plan; yfinance has no historical chains). Premiums here are MODELLED,")
    print("not observed. Measured now (was assumed): per-trade IV level, gamma, true hold length.")
    print("Still assumed: IV = RV x premium factor, IV drift over the hold, ATM strike, no skew.\n")

    _, rvs = reprice_options(trades, data, dte=dte_list[0], rv_window=rv_window)
    if not rvs:
        print("No trades could be repriced (insufficient history for realized vol).")
        return None
    s = sorted(rvs)
    def pct(p):
        return s[min(int(p / 100.0 * len(s)), len(s) - 1)]
    print("MEASURED entry realized vol across %d repriced trades:" % len(s))
    print("  p10 %.1f%% | median %.1f%% | p90 %.1f%%  (this is what the contraction filter actually selects)"
          % (pct(10) * 100, pct(50) * 100, pct(90) * 100))

    print("\nBASE CASE  (DTE %d, iv_premium 1.2, iv_expansion 1.0, spread %.1f%%, comm %.2f%%):"
          % (dte_list[0], spread_pct, commission_pct))
    rets, _ = reprice_options(trades, data, dte=dte_list[0], iv_premium=1.2, iv_expansion=1.0,
                             rv_window=rv_window, spread_pct=spread_pct, commission_pct=commission_pct)
    st = _perf(rets)
    if st:
        print("  N=%d | net expectancy %+.3f%% | win %.1f%% | PF %s"
              % (st["n"], round(st["exp"], 3) + 0.0, st["win"],
                 ("inf" if st["pf"] == float("inf") else "%.2f" % st["pf"])))

    for dte in dte_list:
        print("\nSENSITIVITY, DTE=%d -- net expectancy %% (rows: IV=RV x premium; cols: IV drift over hold)" % dte)
        h = "  %-14s" % "iv_premium" + "".join("%12s" % ("x%.1f" % e) for e in iv_expansions)
        print(h); print("  " + "-" * (len(h) - 2))
        for ivp in iv_premiums:
            cells = []
            for ive in iv_expansions:
                rr, _ = reprice_options(trades, data, dte=dte, iv_premium=ivp, iv_expansion=ive,
                                        rv_window=rv_window, spread_pct=spread_pct,
                                        commission_pct=commission_pct)
                p = _perf(rr)
                cells.append("%+12.2f" % (p["exp"] if p else 0.0))
            print("  %-14s%s" % ("x%.1f" % ivp, "".join(cells)))

    # ---- does it only work on low-vol names? terciles of measured entry RV ----
    print("\nBY ENTRY-RV TERCILE (DTE %d, iv_premium 1.2, iv_expansion 1.0) -- tests 'only low-IV works':" % dte_list[0])
    rets, rvs = reprice_options(trades, data, dte=dte_list[0], iv_premium=1.2, iv_expansion=1.0,
                                rv_window=rv_window, spread_pct=spread_pct, commission_pct=commission_pct)
    pairs = sorted(zip(rvs, rets))
    n = len(pairs)
    third = max(n // 3, 1)
    buckets = [("low RV", pairs[:third]), ("mid RV", pairs[third:2 * third]), ("high RV", pairs[2 * third:])]
    hdr = "  %-9s %-16s %5s %11s %7s %8s" % ("BUCKET", "RV RANGE", "N", "NET-EXP", "WIN%", "PF")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, grp in buckets:
        if not grp:
            continue
        g = _perf([r for _, r in grp])
        print("  %-9s %-16s %5d %+11.3f %7.1f %8s" % (
            name, "%.0f%%-%.0f%%" % (grp[0][0] * 100, grp[-1][0] * 100), g["n"],
            round(g["exp"], 3) + 0.0, g["win"],
            ("inf" if g["pf"] == float("inf") else "%.2f" % g["pf"])))
    return {"rvs": rvs, "base": st}


if __name__ == "__main__":
    # Detection-only eyeball pass: SPY, last 2 years of daily bars. Prints the raw breakout
    # events (breakout date + contraction start date) and the firing rate — NO simulation —
    # so the event stream can be sanity-checked before trusting the simulator or running wide.
    end = date.today()
    start = end - timedelta(days=730)
    print("Fetching SPY daily bars %s .. %s via yfinance (no Polygon needed)..." % (start, end))
    df = fetch_daily_history("SPY", start, end)
    if df.empty:
        raise SystemExit("No daily data returned.")
    print("Got %d daily bars: %s .. %s\n" % (
        len(df), df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")))

    events = detect_contraction_breakout(df)
    print("CONTRACTION -> BREAKOUT events (detection only, no simulation):\n")
    print("  %-14s %s" % ("BREAKOUT", "CONTRACTION START"))
    print("  " + "-" * 34)
    for ev in events:
        print("  %-14s %s" % (ev["date"].strftime("%Y-%m-%d"),
                              ev["contraction_start_date"].strftime("%Y-%m-%d")))
    years = max((df.index[-1] - df.index[0]).days / 365.25, 0.01)
    print("\n%d event(s) over %.1f years (~%.1f per year). First evaluable day needs "
          "%d bars of warmup (14 ATR + 100 percentile), so detection covers %s onward."
          % (len(events), years, len(events) / years, 14 + 100 - 1,
             df.index[min(113, len(df) - 1)].strftime("%Y-%m-%d")))
