#!/usr/bin/env python
"""live_rules_replay.py -- replay what would have happened if EVERY scanner signal in a date window had
been taken (1 contract each), under the LIVE scanner's exit rules, and tag each trade with the Trade
Readiness verdict the live scanner would have shown at signal time -- plus the proposed corrected rule.

READ-ONLY research tool: never places orders; never overwrites an existing file (every output uses a
new prefix and the run aborts if any target already exists). Existing backtests are untouched -- this
module only IMPORTS alpha_backtest / alpha_scanner primitives so its math can't drift from the live app:
  * detectors, minute-bar fetch, session split, VWAP, VIX  -> alpha_backtest
  * E (_expected_move), delta (_bs_delta), suggested strike (_nearest_otm_strike),
    readiness (compute_readiness), STOP_K / TARGET_K / STOP_BUFFER  -> alpha_scanner
  * implied vol from an option price (_bs_iv, bisection)  -> alpha_backtest

EXIT RULES per trade (first to trigger; same-bar priority stop > target > thesis > time):
  stop    : a bar CLOSES through the stop  -> fill AT the stop (at that bar's close if the stop sat on the
            wrong side of entry -- a price we can actually get; note it still only exits once a bar
            closes through the stop, not unconditionally at the fill bar)
  target  : a bar CLOSES through the target -> fill AT the target (bar close if wrong-side)
  thesis  : ORB closes back inside the range / VWAP reclaim closes back below VWAP (alpha_backtest rule)
  time    : 0-2 DTE only -> exit at the close of the first bar at/after 2:00 PM ET
  else    : session close
Stop/target = anchor -/+ STOP_K*E / anchor +/- TARGET_K*E, E = spot*ATM_IV*sqrt(DTE_years). Two anchors:
  as_live   : the signal LEVEL (OR high/low or VWAP) -- exactly what the scanner displayed (the anchoring
              the user flagged as a bug; wrong-side stops/targets are counted and listed)
  corrected : the SPOT at signal time (stops below spot for longs, above for shorts)
If ATM IV is unavailable the scanner's own fallback applies: flat STOP_BUFFER stop, no target.

IV at signal time is backed out (Black-Scholes) from the actual option minute prints (last print at or
before the signal) for the nearest listed expiry: ATM IV = mean of the ATM call & put IV (as the scanner
does); the suggested strike's own IV gives its delta. Fallback when a strike has no recent print: the
logged ~30d IV from iv_history.json (labelled iv_source=proxy30d).
"""
import argparse
import csv
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, time as dtime, timedelta

import alpha_backtest as ab

asc = ab._asc
ET = ab.ET_ZONE
RFR = asc.RISK_FREE_RATE
HARD_EXIT = dtime(14, 0)
MAX_STALE_MIN = 15
DELTA_MIN = 0.30

# ------------------------------------------------------------------------------ pure helpers (tested)
def occ_ticker(ticker, expiry, is_call, strike):
    """OCC option symbol, e.g. ('NVDA', date(2026,9,24), True, 232.5) -> 'O:NVDA260924C00232500'."""
    return "O:%s%s%s%08d" % (ticker, expiry.strftime("%y%m%d"), "C" if is_call else "P", int(round(strike * 1000)))


def years_to_expiry(t_et, expiry):
    """Years from t_et to the 16:00 ET close on `expiry`, floored at 1h (same as the scanner's chain)."""
    exp_close = datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=ET)
    return max((exp_close - t_et).total_seconds(), 3600) / (365.0 * 24 * 3600)


def stop_target(anchor, is_long, E):
    """Scanner formula: stop = anchor -/+ STOP_K*E, target = anchor +/- TARGET_K*E. With E unknown the
    scanner falls back to a flat STOP_BUFFER stop and no target ('Stop only')."""
    if E is None:
        return (anchor * (1 - asc.STOP_BUFFER) if is_long else anchor * (1 + asc.STOP_BUFFER)), None
    if is_long:
        return anchor - asc.STOP_K * E, anchor + asc.TARGET_K * E
    return anchor + asc.STOP_K * E, anchor - asc.TARGET_K * E


def simulate_live(df, vwap, sig_pos, sig, direction, stop, target, dte_days):
    """Forward-only exit under the live rules (see module docstring). Returns a dict or None if the
    signal fired on the last bar (no next-bar fill)."""
    n = len(df)
    fill = sig_pos + 1
    if fill >= n:
        return None
    is_long = direction.upper() in ("LONG", "UP")
    entry = float(df["open"].iloc[fill])
    stop_ok = stop is None or (stop < entry if is_long else stop > entry)
    tgt_ok = target is None or (target > entry if is_long else target < entry)
    or_hi = or_lo = None
    if sig == "ORB":
        orb = df[df.index < df.index[0] + timedelta(minutes=ab.ORB_RANGE_MINUTES)]
        or_hi, or_lo = float(orb["high"].max()), float(orb["low"].min())

    def done(i, px, why):
        ret = (px - entry) / entry * 100.0 if is_long else (entry - px) / entry * 100.0
        return {"entry_price": entry, "entry_time": df.index[fill], "exit_price": float(px),
                "exit_time": df.index[i], "exit_reason": why, "return_pct": ret, "bars_held": i - fill + 1,
                "stop_wrong_side": not stop_ok, "target_wrong_side": not tgt_ok}

    for i in range(fill, n):
        c = float(df["close"].iloc[i])
        if stop is not None and (c <= stop if is_long else c >= stop):
            return done(i, stop if stop_ok else c, "stop" if stop_ok else "stop_wrongside")
        if target is not None and (c >= target if is_long else c <= target):
            return done(i, target if tgt_ok else c, "target" if tgt_ok else "target_wrongside")
        if sig == "VWAP" and c < float(vwap.iloc[i]):
            return done(i, c, "thesis")
        if sig == "ORB" and ((direction.upper() == "UP" and c <= or_hi) or (direction.upper() == "DOWN" and c >= or_lo)):
            return done(i, c, "thesis")
        if dte_days is not None and dte_days <= 2 and df.index[i].time() >= HARD_EXIT:
            return done(i, c, "time_2pm")
    return done(n - 1, float(df["close"].iloc[-1]), "session_close")


def corrected_readiness(current_verdict, delta, beyond_1sigma=False):
    """Proposed rule: FAIL if the current verdict fails, if the suggested-strike |delta| < 0.30 or is
    unknown (fail-safe), or (magnet) if the target carries the >1-sigma flag."""
    notes = []
    if current_verdict != "PASS":
        notes.append("current FAIL")
    if delta is None:
        notes.append("delta unknown")
    elif abs(delta) < DELTA_MIN:
        notes.append("delta %.2f < 0.30" % abs(delta))
    if beyond_1sigma:
        notes.append("pin >1 sigma")
    return ("FAIL" if notes else "PASS"), "; ".join(notes)


def bucket_stats(rows, baseline=None, key="return_pct"):
    """Same formulas as alpha_backtest.report: N, WIN%, EXPECT, vs BASE, PF, MAXDD (pp, chronological)."""
    rets = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not rets:
        return None
    wins = [x for x in rets if x > 0]
    losses = [x for x in rets if x < 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    eq = peak = mdd = 0.0
    for r in sorted((r for r in rows if r.get(key) not in (None, "")),
                    key=lambda r: (str(r.get("date")), str(r.get("entry_time")))):
        eq += float(r[key])
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    exp = sum(rets) / len(rets)
    return {"n": len(rets), "win": 100.0 * len(wins) / len(rets), "exp": exp,
            "vsbase": (exp - baseline) if baseline is not None else None, "pf": pf, "mdd": mdd}


def fmt_row(label, st):
    if st is None:
        return "  %-34s %4d   -- no trades --" % (label, 0)
    return "  %-34s %4d %6.1f %+9.3f %9s %6s %7.2f%s" % (
        label, st["n"], st["win"], st["exp"], ("%+.3f" % st["vsbase"]) if st["vsbase"] is not None else "n/a",
        "inf" if st["pf"] == float("inf") else "%.2f" % st["pf"], st["mdd"], "  NOISE(n<30)" if st["n"] < 30 else "")


HEADER = "  %-34s %4s %6s %9s %9s %6s %7s" % ("BUCKET", "N", "WIN%", "EXPECT%", "vs BASE", "PF", "MAXDD")


# ------------------------------------------------------------------------------ data (networked)
class OptionData:
    """Polygon options lookups at signal time, cached. Options calls use a light 0.15s pace."""

    def __init__(self, key):
        self.key, self._exp, self._bars = key, {}, {}

    def _get(self, url, params):
        ab._poly_throttle(0.15)
        r = ab._polygon_request(ab.POLYGON_BASE_URL + url, self.key, params=params, timeout=20)
        return r.json() if r.status_code == 200 else {}

    def nearest_expiry(self, ticker, d):
        k = (ticker, d)
        if k not in self._exp:
            best = None
            for expired in ("true", "false"):
                j = self._get("/v3/reference/options/contracts",
                              {"underlying_ticker": ticker, "expiration_date.gte": d.isoformat(),
                               "expired": expired, "sort": "expiration_date", "order": "asc", "limit": 1})
                for c in j.get("results") or []:
                    e = date.fromisoformat(c["expiration_date"])
                    best = e if best is None else min(best, e)
            self._exp[k] = best
        return self._exp[k]

    def strikes(self, ticker, expiry):
        return ab._expiry_strikes(ticker, expiry.isoformat(), self.key)

    def bars(self, occ, d):
        k = (occ, d)
        if k not in self._bars:
            j = self._get("/v2/aggs/ticker/%s/range/1/minute/%s/%s" % (occ, d, d), {"adjusted": "true", "limit": 50000})
            self._bars[k] = [(datetime.fromtimestamp(b["t"] / 1000, tz=ET), float(b["c"])) for b in j.get("results") or []]
        return self._bars[k]

    def price_at(self, occ, d, t, max_stale=MAX_STALE_MIN):
        """Last print at or before t (no lookahead) within max_stale minutes -> (price, age_min) or (None, None)."""
        last = None
        for ts, px in self.bars(occ, d):
            if ts <= t:
                last = (ts, px)
            else:
                break
        if last is None:
            return None, None
        age = (t - last[0]).total_seconds() / 60.0
        return (last[1], age) if age <= max_stale else (None, age)


def load_proxy_iv():
    try:
        return json.load(open("iv_history.json"))
    except Exception:
        return {}


def proxy_iv(proxy, ticker, d):
    """Logged ~30d ATM IV on the latest date <= d (labelled proxy; not nearest-expiry IV)."""
    best = None
    for e in proxy.get(ticker, []):
        if e.get("date") and e["date"] <= d.isoformat():
            best = e
    return float(best["iv"]) if best else None


def vix_intraday(start, end):
    """{ET timestamp: VIX} at 5-minute resolution (what the live scanner saw), plus daily closes as fallback."""
    try:
        yf = ab._load_real_yfinance()
        h = yf.Ticker("^VIX").history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="5m")
        s = h["Close"].dropna()
        s.index = s.index.tz_convert(ET)
        return s
    except Exception:
        return None


def vix_at(series, daily_prior, t):
    if series is not None and len(series):
        prior = series[series.index <= t]
        if len(prior):
            return float(prior.iloc[-1]), "intraday5m"
    v = daily_prior.get(t.date())
    return (v, "prior_close") if v is not None else (None, "unknown")


def option_context(od, proxy, ticker, d, t, spot, is_long):
    """IV/E/delta/suggested strike/premium at signal time t (ET)."""
    ctx = {"expiry": None, "dte_days": None, "atm_iv": None, "iv_source": None, "E": None,
           "sugg_strike": None, "sugg_occ": None, "sugg_iv": None, "delta": None, "premium": None}
    exp = od.nearest_expiry(ticker, d)
    if exp is None:
        return ctx
    ctx["expiry"], ctx["dte_days"] = exp, (exp - d).days
    T = years_to_expiry(t, exp)
    strikes = od.strikes(ticker, exp)
    if not strikes:
        return ctx
    k_atm = min(strikes, key=lambda k: abs(k - spot))
    ivs = []
    for is_call in (True, False):
        px, _ = od.price_at(occ_ticker(ticker, exp, is_call, k_atm), d, t)
        iv = ab._bs_iv(px, spot, k_atm, T, RFR, is_call) if px else None
        if iv:
            ivs.append(iv)
    if ivs:
        ctx["atm_iv"], ctx["iv_source"] = sum(ivs) / len(ivs), "option_prints"
    else:
        ctx["atm_iv"] = proxy_iv(proxy, ticker, d)
        ctx["iv_source"] = "proxy30d" if ctx["atm_iv"] else None
    ctx["E"] = asc._expected_move(spot, ctx["atm_iv"], T)
    k = asc._nearest_otm_strike(strikes, spot, is_long)
    ctx["sugg_strike"] = k
    occ = occ_ticker(ticker, exp, is_long, k)
    ctx["sugg_occ"] = occ
    px, _ = od.price_at(occ, d, t)
    ctx["premium"] = px
    siv = ab._bs_iv(px, spot, k, T, RFR, is_long) if px else None
    ctx["sugg_iv"] = siv or ctx["atm_iv"]
    if ctx["sugg_iv"]:
        ctx["delta"] = asc._bs_delta(spot, k, T, ctx["sugg_iv"], RFR, is_long)
    return ctx


# ------------------------------------------------------------------------------ phase 1: ORB + VWAP
TRADE_FIELDS = ["ticker", "date", "signal_type", "direction", "signal_time", "spot_at_signal", "level",
                "vol_ratio", "vix", "vix_source", "expiry", "dte_days", "atm_iv", "iv_source", "E",
                "sugg_strike", "delta", "premium_entry", "premium_exit", "opt_pnl_usd", "opt_pnl_adj_usd",
                "ready_current", "ready_current_note", "ready_corrected", "ready_corrected_note",
                "variant", "stop", "target", "stop_wrong_side", "target_wrong_side", "entry_time",
                "entry_price", "exit_time", "exit_price", "exit_reason", "return_pct", "bars_held"]
SPREAD_ASSUMPTION = 0.05   # round-trip spread as a fraction of premium (the cost-check grid's middle case)


def replay_intraday(tickers, start, end, od, proxy, log=print):
    vix_s = vix_intraday(start, end)
    vd = ab._fetch_vix_by_date(start, end)
    ks = sorted(vd)
    daily_prior = {ks[i]: vd[ks[i - 1]] for i in range(1, len(ks))}
    trades, base = [], []
    for idx, tk in enumerate(tickers, 1):
        try:
            bars = ab.fetch_minute_history(tk, start, end)
        except Exception as e:
            log("  %-6s FETCH FAILED: %s" % (tk, str(e)[:100]))
            continue
        sessions = ab.group_by_session(bars, drop_half_days=True, verbose=False)
        base.extend(ab._session_oc_returns(sessions).values())
        found = 0
        for d in sorted(sessions):
            s = sessions[d]
            vwap = ab.compute_session_vwap(s)
            fires = []
            v = ab.detect_vwap_entry(s, vwap)
            if v:
                fires.append(("VWAP", v[0], v[1]))
            o = ab.detect_orb_entry(s)
            if o:
                fires.append(("ORB", o[0], o[1]))
            for sig, pos, direction in fires:
                t = s.index[pos]
                is_long = direction.upper() in ("LONG", "UP")
                spot = float(s["close"].iloc[pos])
                vol_ratio = float(s["volume"].iloc[pos]) / float(s["volume"].iloc[:pos + 1].mean())
                vix, vsrc = vix_at(vix_s, daily_prior, t)
                if sig == "VWAP" and isinstance(vix, (int, float)) and vix > asc.VWAP_MAX_VIX:
                    continue            # live VWAP detector skips high-VIX reclaims (signal never shown)
                if sig == "VWAP":
                    level = float(vwap.iloc[pos])
                else:
                    orb = s[s.index < s.index[0] + timedelta(minutes=ab.ORB_RANGE_MINUTES)]
                    level = float(orb["high"].max()) if is_long else float(orb["low"].min())
                cx = option_context(od, proxy, tk, d, t, spot, is_long)
                cur, cur_note = asc.compute_readiness(sig.lower(), None, None, vol_ratio, vix, t, t)
                corr, corr_note = corrected_readiness(cur, cx["delta"])
                for variant, anchor in (("as_live", level), ("corrected", spot)):
                    stp, tgt = stop_target(anchor, is_long, cx["E"])
                    r = simulate_live(s, vwap, pos, sig, direction, stp, tgt, cx["dte_days"])
                    if r is None:
                        continue
                    pe = cx["premium"]
                    px_exit = od.price_at(cx["sugg_occ"], d, r["exit_time"])[0] if cx["sugg_occ"] and pe else None
                    pnl = 100.0 * (px_exit - pe) if (pe and px_exit is not None) else None
                    trades.append({
                        "ticker": tk, "date": d.isoformat(), "signal_type": sig, "direction": direction,
                        "signal_time": t.strftime("%H:%M"), "spot_at_signal": round(spot, 2), "level": round(level, 2),
                        "vol_ratio": round(vol_ratio, 2), "vix": vix, "vix_source": vsrc,
                        "expiry": cx["expiry"].isoformat() if cx["expiry"] else "", "dte_days": cx["dte_days"],
                        "atm_iv": round(cx["atm_iv"], 4) if cx["atm_iv"] else "", "iv_source": cx["iv_source"] or "",
                        "E": round(cx["E"], 3) if cx["E"] else "", "sugg_strike": cx["sugg_strike"],
                        "delta": round(cx["delta"], 3) if cx["delta"] is not None else "",
                        "premium_entry": pe if pe is not None else "", "premium_exit": px_exit if px_exit is not None else "",
                        "opt_pnl_usd": round(pnl, 2) if pnl is not None else "",
                        "opt_pnl_adj_usd": round(pnl - 100.0 * pe * SPREAD_ASSUMPTION, 2) if pnl is not None else "",
                        "ready_current": cur, "ready_current_note": cur_note,
                        "ready_corrected": corr, "ready_corrected_note": corr_note,
                        "variant": variant, "stop": round(stp, 2), "target": round(tgt, 2) if tgt is not None else "",
                        "stop_wrong_side": r["stop_wrong_side"], "target_wrong_side": r["target_wrong_side"],
                        "entry_time": r["entry_time"].strftime("%H:%M"), "entry_price": round(r["entry_price"], 2),
                        "exit_time": r["exit_time"].strftime("%H:%M"), "exit_price": round(r["exit_price"], 2),
                        "exit_reason": r["exit_reason"], "return_pct": round(r["return_pct"], 4),
                        "bars_held": r["bars_held"]})
                found += 1
        log("  %-6s %d signal(s)  [%d/%d]" % (tk, found, idx, len(tickers)))
    baseline = sum(base) / len(base) if base else None
    return trades, baseline


# ------------------------------------------------------------------------------ phase 2: magnet
MAGNET_FIELDS = ["ticker", "log_date", "expiry", "spot", "max_pain", "dist_pct", "direction", "close_at_expiry",
                 "return_pct", "hold_days", "atm_iv", "iv_source", "E", "beyond_1sigma", "sugg_strike", "delta",
                 "ready_current", "ready_current_note", "ready_corrected", "ready_corrected_note"]


def magnet_rows(hist, start, end, od, proxy, vix_daily, log=print):
    """Predictions logged in [start, end] whose expiry <= end, scored exactly like magnet_forward_report
    (entry = spot at log, exit = underlying close AT expiry, direction toward the pin). IV/delta at log
    time are backed out from the options' DAILY closes on the log date (log time-of-day isn't recorded)."""
    out, closes = [], {}
    for tk, entries in hist.items():
        for e in entries or []:
            ld, ex = date.fromisoformat(e["date"]), date.fromisoformat(e["expiry"])
            if not (start <= ld <= end and ex <= end):
                continue
            if tk not in closes:
                closes[tk] = ab._fetch_daily_closes(tk, start - timedelta(days=3), end)
            # same close-at-expiry lookup as magnet_forward_report: last trading-day close on/before expiry
            on_or_before = [d for d in closes[tk] if d <= ex]
            c_exp = closes[tk][max(on_or_before)] if on_or_before else None
            c_log = closes[tk].get(ld)
            spot, mp = float(e["spot"]), float(e["max_pain"])
            if c_exp is None or mp <= 0 or spot == mp:      # the report skips spot == pin (no direction)
                continue
            is_long = spot < mp
            long_ret = (c_exp - spot) / spot * 100.0
            dist = abs(spot - mp) / mp * 100.0
            # T is measured from 16:00 ET on the log date (the log's time of day isn't recorded), which
            # understates T by up to a day for multi-day logs. Same-day logs (log date == expiry) have
            # ~no time value left at the EOD print, so IV/E/delta/1-sigma are UNKNOWN for them.
            same_day = ld >= ex
            t = datetime(ld.year, ld.month, ld.day, 16, 0, tzinfo=ET)
            T = years_to_expiry(t, ex)
            iv, src, E, k, dlt = None, None, None, None, None
            strikes = od.strikes(tk, ex)
            if strikes and c_log and not same_day:
                ka = min(strikes, key=lambda x: abs(x - c_log))
                ivs = []
                for is_call in (True, False):
                    j = od._get("/v2/aggs/ticker/%s/range/1/day/%s/%s" % (occ_ticker(tk, ex, is_call, ka), ld, ld), {"adjusted": "true"})
                    res = j.get("results") or []
                    if res and res[0].get("c"):
                        v = ab._bs_iv(float(res[0]["c"]), c_log, ka, T, RFR, is_call)
                        if v:
                            ivs.append(v)
                if ivs:
                    iv, src = sum(ivs) / len(ivs), "option_eod"
            if iv is None and not same_day:
                iv = proxy_iv(proxy, tk, ld)
                src = "proxy30d" if iv else None
            if same_day:
                src = "unknown(same-day log)"
            E = asc._expected_move(spot, iv, T) if not same_day else None
            if strikes:
                k = asc._nearest_otm_strike(strikes, spot, is_long)
                siv = None                                  # the suggested strike's OWN IV (as the scanner)
                if c_log and not same_day:
                    j = od._get("/v2/aggs/ticker/%s/range/1/day/%s/%s" % (occ_ticker(tk, ex, is_long, k), ld, ld),
                                {"adjusted": "true"})
                    res = j.get("results") or []
                    if res and res[0].get("c"):
                        siv = ab._bs_iv(float(res[0]["c"]), c_log, k, T, RFR, is_long)
                if (siv or iv) and not same_day:
                    dlt = asc._bs_delta(spot, k, T, siv or iv, RFR, is_long)
            beyond = E is not None and abs(mp - spot) > E     # unknown E (same-day) -> not flagged
            vix = vix_daily.get(ld)
            cur, cur_note = asc.compute_readiness("magnet", "MEDIUM", dist if spot >= mp else -dist, None, vix, t, t)
            cur_note = (cur_note + "; " if cur_note else "") + "OI gate NOT reproducible (no historical OI) -> assumed met"
            corr, corr_note = corrected_readiness(cur, dlt, beyond)
            out.append({"ticker": tk, "log_date": ld.isoformat(), "expiry": ex.isoformat(), "spot": spot, "max_pain": mp,
                        "dist_pct": round(dist, 3), "direction": "UP" if is_long else "DOWN", "close_at_expiry": c_exp,
                        "return_pct": round(long_ret if is_long else -long_ret, 4), "hold_days": (ex - ld).days,
                        "atm_iv": round(iv, 4) if iv else "", "iv_source": src or "", "E": round(E, 3) if E else "",
                        "beyond_1sigma": beyond, "sugg_strike": k, "delta": round(dlt, 3) if dlt is not None else "",
                        "ready_current": cur, "ready_current_note": cur_note,
                        "ready_corrected": corr, "ready_corrected_note": corr_note,
                        "date": ld.isoformat(), "entry_time": ""})
    return out


# ------------------------------------------------------------------------------ driver
def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def capture(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tickers", nargs="*")
    a = ap.parse_args()
    start, end, P = date.fromisoformat(a.start), date.fromisoformat(a.end), a.prefix
    if not P.startswith("week_"):
        sys.exit("prefix must start with week_ (never touch existing outputs)")
    existing = [f for f in os.listdir(".") if f.startswith(P)]
    if existing:
        sys.exit("refusing to overwrite existing outputs: %s" % existing)
    key = ab._polygon_key()
    if not key:
        sys.exit("no Polygon key")
    # In-process cache for daily closes (this script only): the two --magnet-report runs and the tagging
    # step all need the same tickers' closes; the free stock tier allows 5 calls/min, so fetch each once
    # over a superset window and slice. alpha_backtest itself is not modified.
    win_lo, win_hi, cache, orig = start - timedelta(days=12), end + timedelta(days=2), {}, ab._fetch_daily_closes

    def cached_closes(ticker, s, e, throttle_seconds=12, timeout=30):
        s, e = ab._as_date(s), ab._as_date(e)
        if s < win_lo or e > win_hi:
            return orig(ticker, s, e, throttle_seconds, timeout)
        if ticker not in cache:
            cache[ticker] = orig(ticker, win_lo, win_hi, throttle_seconds, timeout)
        return {d: c for d, c in cache[ticker].items() if s <= d <= e}

    ab._fetch_daily_closes = cached_closes
    tickers = a.tickers or list(asc.CORE_WATCHLIST)
    od, proxy = OptionData(key), load_proxy_iv()
    rep = []

    print("PHASE 1: ORB/VWAP replay %s..%s over %d tickers" % (start, end, len(tickers)))
    trades, baseline = replay_intraday(tickers, start, end, od, proxy)
    write_csv(P + "orb_vwap_trades.csv", trades, TRADE_FIELDS)
    rep.append("## Phase 1 - ORB / VWAP (all signals, 1 contract each)\n\nBaseline (long open->close, all ticker-sessions): %s\n"
               % ("%+.4f%%" % baseline if baseline is not None else "n/a"))
    for variant in ("as_live", "corrected"):
        rows = [t for t in trades if t["variant"] == variant]
        rep.append("### Exit anchoring: %s  (stop/target wrong-side: %d / %d of %d)\n```\n%s" % (
            variant, sum(1 for t in rows if t["stop_wrong_side"]), sum(1 for t in rows if t["target_wrong_side"]), len(rows), HEADER))
        for sig in ("ORB", "VWAP", "ALL"):
            sub = rows if sig == "ALL" else [t for t in rows if t["signal_type"] == sig]
            rep.append(fmt_row(sig, bucket_stats(sub, baseline)))
            for rule in ("ready_current", "ready_corrected"):
                for v in ("PASS", "FAIL"):
                    rep.append(fmt_row("  %s %s-only" % (rule.split("_")[1], v), bucket_stats([t for t in sub if t[rule] == v], baseline)))
        rep.append("```")
        # cost-check per bucket (raw underlying -> options estimate), IV = bucket median ATM IV
        for sig in ("ORB", "VWAP", "ALL"):
            for rule, v in (("all", None), ("ready_current", "PASS"), ("ready_current", "FAIL"),
                            ("ready_corrected", "PASS"), ("ready_corrected", "FAIL")):
                sub = [t for t in rows if (sig == "ALL" or t["signal_type"] == sig) and (v is None or t[rule] == v)]
                if not sub:
                    continue
                ivs = sorted(float(t["atm_iv"]) for t in sub if t["atm_iv"] != "")
                iv = ivs[len(ivs) // 2] if ivs else 0.30
                path = "%s%s_%s_%s_%s.csv" % (P, variant, sig.lower(), rule.replace("ready_", ""), (v or "all").lower())
                write_csv(path, sub, TRADE_FIELDS)
                txt = capture(ab.cost_check, path, iv=iv)
                rep.append("#### cost-check %s | %s %s | %s (n=%d, IV %.0f%%)\n```\n%s```" % (
                    variant, rule, v or "ALL", sig, len(sub), iv * 100, txt))
    opt = [t for t in trades if t["variant"] == "as_live" and t["opt_pnl_usd"] != ""]

    print("PHASE 2: magnet from the forward log")
    hist = json.load(open("magnet_history.json"))
    filt = {tk: [e for e in es if start.isoformat() <= e["date"] <= end.isoformat() and e["expiry"] <= end.isoformat()]
            for tk, es in hist.items()}
    filt = {k: v for k, v in filt.items() if v}
    mh = P + "magnet_history.json"
    json.dump(filt, open(mh, "w"), indent=1)
    rep.append("## Phase 2 - Magnet pin (forward log, logged %s..%s, expiry <= %s)\n" % (start, end, end))
    for lbl, md in (("le1pct", 1.0), ("all", 100.0)):
        mcsv = "%smagnet_%s.csv" % (P, lbl)
        txt = capture(ab.magnet_forward_report, mh, csv_out=mcsv, max_distance_pct=md)
        rep.append("### --magnet-report  max-distance %s\n```\n%s```" % (md, txt))
        if os.path.exists(mcsv):
            rep.append("#### cost-check magnet %s\n```\n%s```" % (lbl, capture(ab.cost_check, mcsv)))
    vd = ab._fetch_vix_by_date(start, end)
    mrows = magnet_rows(filt, start, end, od, proxy, vd)
    write_csv(P + "magnet_tagged.csv", mrows, MAGNET_FIELDS)

    print("PHASE 3: current vs corrected readiness")
    rep.append("## Phase 3 - Current vs corrected readiness\n```\n" + HEADER)
    groups = (("ORB/VWAP as_live", [t for t in trades if t["variant"] == "as_live"]),
              ("ORB/VWAP corrected-anchor", [t for t in trades if t["variant"] == "corrected"]),
              ("MAGNET <=1%", [m for m in mrows if m["dist_pct"] <= 1.0]), ("MAGNET all", mrows))
    for name, rows in groups:
        rep.append(fmt_row(name + " (all)", bucket_stats(rows)))
        for rule in ("ready_current", "ready_corrected"):
            for v in ("PASS", "FAIL"):
                rep.append(fmt_row("  %s %s" % (rule.split("_")[1], v), bucket_stats([r for r in rows if r[rule] == v])))
    rep.append("```")

    print("PHASE 4: summary")
    live = [t for t in trades if t["variant"] == "as_live"]
    tot_opt = sum(float(t["opt_pnl_usd"]) for t in opt)
    tot_adj = sum(float(t["opt_pnl_adj_usd"]) for t in opt)
    best = max(live, key=lambda t: float(t["return_pct"]), default=None)
    worst = min(live, key=lambda t: float(t["return_pct"]), default=None)
    sig_n = len(live)
    rep.append("## Phase 4 - Summary\n")
    rep.append("- Signals: ORB/VWAP %d (ORB %d, VWAP %d) + magnet predictions %d (<=1%%: %d)" % (
        sig_n, sum(1 for t in live if t["signal_type"] == "ORB"), sum(1 for t in live if t["signal_type"] == "VWAP"),
        len(mrows), sum(1 for m in mrows if m["dist_pct"] <= 1.0)))
    rep.append("- ORB/VWAP underlying, as-live exits: sum %+.3f pp over %d trades" % (sum(float(t["return_pct"]) for t in live), sig_n))
    rep.append("- Option P&L from real prints (1 contract, as-live exits): %d/%d trades priced; raw $%+.2f; "
               "after a %.0f%% round-trip spread $%+.2f" % (len(opt), sig_n, tot_opt, SPREAD_ASSUMPTION * 100, tot_adj))
    if best:
        rep.append("- Best (underlying): %s %s %s %+.3f%% (%s)   Worst: %s %s %s %+.3f%% (%s)" % (
            best["ticker"], best["date"], best["signal_type"], float(best["return_pct"]), best["exit_reason"],
            worst["ticker"], worst["date"], worst["signal_type"], float(worst["return_pct"]), worst["exit_reason"]))
    if opt:
        bo = max(opt, key=lambda t: float(t["opt_pnl_usd"]))
        wo = min(opt, key=lambda t: float(t["opt_pnl_usd"]))
        rep.append("- Best (option $): %s %s %s $%+.2f   Worst: %s %s %s $%+.2f" % (
            bo["ticker"], bo["date"], bo["signal_type"], float(bo["opt_pnl_usd"]),
            wo["ticker"], wo["date"], wo["signal_type"], float(wo["opt_pnl_usd"])))
    nv = [m for m in mrows if m["ticker"] == "NVDA" and m["log_date"] == "2026-09-24"]
    for m in nv:
        rep.append("- NVDA 2026-09-24 magnet prediction (replay): spot %.2f pin %.2f dist %.2f%% -> %s, close@%s %.2f, "
                   "return %+.3f%%; current %s (%s); corrected %s (%s)" % (
                       m["spot"], m["max_pain"], m["dist_pct"], m["direction"], m["expiry"], m["close_at_expiry"],
                       m["return_pct"], m["ready_current"], m["ready_current_note"], m["ready_corrected"], m["ready_corrected_note"]))
    open(P + "summary.md", "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
