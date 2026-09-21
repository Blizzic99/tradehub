#!/usr/bin/env python
"""Phase 6 — headless scan -> analyze -> Telegram alert pipeline.

Runs the magnet scan WITHOUT the Streamlit UI, picks qualifying signals (STRONG/MODERATE strength
with good open interest, within 1% of the max-pain pin), builds the Phase-5 ThinkorSwim MANUAL order
ticket for each, and sends ONE Telegram alert containing them. Designed for an unattended run
(Windows Task Scheduler, cron, or a Claude Code cloud Routine) so alerts fire without the laptop's
Streamlit app open.

HARD CONSTRAINT: this pipeline scans, analyzes, and alerts ONLY. It NEVER places, modifies, or
cancels a live order — every alert ends with a manual TOS ticket the user types and confirms.

Usage:
    python scan_alert.py            # DRY RUN: print the alert to stdout, send nothing
    python scan_alert.py --send     # actually send the Telegram alert (needs creds in secrets)
    python scan_alert.py --max 5    # cap the number of tickets in the alert (default 3)
"""
import argparse
import datetime as _dt
import html
import json
import os
import sys

import alpha_backtest as ab          # stub-loads the scanner offline as ab._asc
asc = ab._asc

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_alert_state.json")


def _signature(signals):
    """Order-independent identity of the current qualifying set (for scheduled-run dedup)."""
    return sorted([r["Ticker"], r["Direction"], str(r.get("Suggested Strike"))] for r in signals)


def _already_alerted(sig):
    try:
        return json.load(open(STATE_FILE)).get("sig") == sig
    except Exception:
        return False


def _remember(sig):
    try:
        json.dump({"sig": sig, "at": _dt.datetime.now(asc.ET_ZONE).isoformat()}, open(STATE_FILE, "w"))
    except Exception:
        pass


def _init():
    """Inject the real Polygon key + yfinance (the offline stub-load blanks them)."""
    asc.POLYGON_KEY = ab._polygon_key()
    try:
        asc.yf = ab._load_real_yfinance()
    except Exception:
        pass
    # Telegram creds resolve at scanner import via _secret (file-anchored); re-affirm here.
    asc.TELEGRAM_BOT_TOKEN = asc._secret("TELEGRAM_BOT_TOKEN")
    asc.TELEGRAM_CHAT_ID = asc._secret("TELEGRAM_CHAT_ID")


def scan_qualifying(max_n=3):
    """Return up to max_n qualifying magnet signals (closest to the pin first), each a dict with the
    display fields + the Phase-5 ticket. Mirrors the Streamlit magnet tab's tradeable filter."""
    wl = asc.CORE_WATCHLIST
    mp = asc._parallel_map(asc.get_max_pain_strike, wl, workers=getattr(asc, "POLYGON_WORKERS", 12))
    out = []
    for tk, res in zip(wl, mp):
        if not res:
            continue
        max_pain, price, expiry, total_oi, _atm_iv, _skew = res
        if not (max_pain and price):
            continue
        dist = (price - max_pain) / max_pain * 100.0
        strength = "STRONG" if abs(dist) < 2 else ("MODERATE" if abs(dist) < 4 else "WEAK")
        oi = asc.get_oi_tier(total_oi)
        tradeable = (strength == "STRONG" and oi in ("HIGH", "MEDIUM")) or (strength == "MODERATE" and oi == "HIGH")
        if not tradeable:
            continue
        is_long = dist < 0                       # price below pin -> expect drift up -> long / CALL
        m = asc.compute_trade_mechanics(tk, price, is_long, "magnet", max_pain, magnet_strike=max_pain)
        row = {
            "Ticker": tk,
            "Direction": "MAGNET UP" if is_long else "MAGNET DOWN",
            "Suggested Strike": m.get("Suggested Strike"),
            "Expiry": expiry.strftime("%Y-%m-%d") if expiry else "N/A",
            "Stop Price": m.get("Stop Price"),
            "Profit Target": m.get("Profit Target"),
            "Exit Window": m.get("Exit Window"),
            "_dist": dist, "_strength": strength, "_oi": oi, "_price": price, "_pin": max_pain,
        }
        row["_ticket"] = asc._tos_ticket_from_row(row, "magnet")
        out.append(row)
    out.sort(key=lambda r: abs(r["_dist"]))
    return out[:max_n]


def build_message(signals):
    """Compose the HTML Telegram message (parse_mode=HTML). Tickets go in <pre> for monospace."""
    now = _dt.datetime.now(asc.ET_ZONE).strftime("%b %d, %I:%M %p ET")
    head = ["\U0001F3AF <b>Alpha Scanner - Magnet Alert</b>",
            "<i>%s - %d qualifying signal(s)</i>" % (now, len(signals))]
    if not signals:
        head.append("\nNo qualifying magnet setups right now. Stand down.")
        return "\n".join(head)
    parts = ["\n".join(head)]
    for r in signals:
        parts.append("\n<b>%s</b> %s - %s - OI %s - %.2f%% from pin"
                     % (r["Ticker"], r["Direction"], r["_strength"], r["_oi"], r["_dist"]))
        parts.append("<pre>%s</pre>" % html.escape(r["_ticket"]))
    parts.append("\n⚠️ Manual entry only - Alpha Scanner never places, modifies, or cancels orders.")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually send the Telegram alert (default: dry run)")
    ap.add_argument("--max", type=int, default=3, help="max tickets in the alert (default 3)")
    ap.add_argument("--force", action="store_true", help="send even if the qualifying set is unchanged since last run")
    args = ap.parse_args()
    _init()
    if not asc.POLYGON_KEY:
        print("No Polygon key resolved - cannot scan. Set POLYGON_KEY in .streamlit/secrets.toml.")
        return 1
    signals = scan_qualifying(max_n=args.max)
    message = build_message(signals)
    print(message.encode("ascii", "replace").decode("ascii"))   # console-safe echo
    print("\n[%d qualifying signal(s)]" % len(signals))
    if not args.send:
        print("DRY RUN - nothing sent. Re-run with --send to deliver via Telegram.")
        return 0
    if not (asc.TELEGRAM_BOT_TOKEN and asc.TELEGRAM_CHAT_ID):
        print("Telegram credentials not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) - not sent.")
        return 2
    sig = _signature(signals)
    if signals and not args.force and _already_alerted(sig):
        print("Qualifying set unchanged since last run - not re-sending (use --force to override).")
        return 0
    ok = asc.send_telegram_alert(message)
    if ok:
        _remember(sig)
    print("Telegram alert sent." if ok else "Telegram send FAILED (check token / chat id / network).")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
