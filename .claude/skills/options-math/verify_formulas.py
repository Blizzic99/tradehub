#!/usr/bin/env python
"""Formula verification for the options-math Skill.

Imports the ACTUAL pricing functions from alpha_scanner.py / alpha_backtest.py and checks each
against the known-correct worked example documented in SKILL.md. The expected values are anchored to
Hull's canonical Black-Scholes example (S=K=100, T=1, r=0.05, sigma=0.20 -> call = 10.4506) and to
hand-computed convention examples (max pain, GEX sign, skew, P/C). Exit 0 = all match; exit 2 = a
mismatch (the PostToolUse hook turns exit 2 into a BLOCK on the edit).

No network: alpha_backtest stub-loads the scanner offline as `_asc`, and every function checked here
is pure (no fetch). Run manually:

    C:/Users/santw/AppData/Local/Python/bin/python.exe .claude/skills/options-math/verify_formulas.py

Demo the catch (no code edit needed): set OPTIONS_MATH_DEMO_FAIL=1 to corrupt ONE expected value and
watch the failure path fire (a FAIL line + exit 2) -- this is the Phase-1 "hook catches a wrong
value" demonstration.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import alpha_backtest as ab            # stub-loads streamlit/yfinance offline; exposes _asc (scanner)
asc = ab._asc

DEMO_FAIL = os.environ.get("OPTIONS_MATH_DEMO_FAIL") == "1"

# --- worked-example fixtures (mirror SKILL.md; do NOT change a value without re-deriving it) ---
SNAP_GEX = {"dte_years": 1.0, "strikes": [100.0],
            "call_iv": {100.0: 0.20}, "put_iv": {100.0: 0.20},
            "call_oi": {100.0: 2000.0}, "put_oi": {100.0: 1000.0}}
SNAP_SKEW = {"dte_years": 0.0833, "call_iv": {110.0: 0.25}, "put_iv": {90.0: 0.30}}

_checks = []
def check(label, actual, expected, tol=1e-6, exact=False):
    if exact:
        ok = (actual == expected)
    else:
        try:
            ok = actual is not None and abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = False
    _checks.append((label, ok, actual, expected))

# Black-Scholes price / N(.) / implied-vol solver  (alpha_backtest) -- Hull canonical example
check("backtest._norm_cdf(0.35) = N(0.35)",          ab._norm_cdf(0.35),                        0.636830651, 1e-6)
check("backtest._bs_call(100,100,1,.05,.20) [Hull]", ab._bs_call(100, 100, 1, 0.05, 0.20),      10.45058357, 1e-4)
check("backtest._bs_iv(10.4506,100,100,1,.05,call)", ab._bs_iv(10.4506, 100, 100, 1, 0.05, True), 0.20000044, 1e-4)

# Greeks  (alpha_scanner)
exp_delta = 0.636830651 + (0.1 if DEMO_FAIL else 0.0)   # DEMO_FAIL corrupts the EXPECTED -> a caught mismatch
check("scanner._bs_delta call = N(d1)",              asc._bs_delta(100, 100, 1, 0.20, 0.05, True),  exp_delta,    1e-6)
check("scanner._bs_delta put  = N(d1)-1",            asc._bs_delta(100, 100, 1, 0.20, 0.05, False), -0.363169349, 1e-6)
check("scanner._bs_gamma (r=RISK_FREE_RATE inside)", asc._bs_gamma(100, 100, 1, 0.20),              0.018920992,  1e-6)

# GEX / volatility skew / put-call ratio  (alpha_scanner)
check("scanner._net_gex_at(asym book, S=100)",       asc._net_gex_at(SNAP_GEX, 100.0),          189209.916, 0.1)
check("scanner.compute_gex regime (spot>flip)",      asc.compute_gex(SNAP_GEX, 100.0)["regime"], "POSITIVE", exact=True)
check("scanner.compute_skew = (0.30-0.25)*100",      asc.compute_skew(SNAP_SKEW, 100.0),         5.0, 1e-6)
_pcr = asc.compute_pcr({"call_vol_total": 1000.0, "put_vol_total": 1500.0})
check("scanner.compute_pcr ratio = 1500/1000",       _pcr[0],                                    1.5, 1e-9)
check("scanner.compute_pcr flag (>=1.5)",            _pcr[1],                                    "EXTREME-PUT", exact=True)

# Phase 4: DTE-aware expected move + best-exit-window (Skill examples 6 & 12)
check("scanner._expected_move(100,0.20,0.25)=1sig",  asc._expected_move(100, 0.20, 0.25),        10.0, 1e-9)
check("scanner._best_exit_window(10) multi-day",     asc._best_exit_window(10),
      "exit/roll by ~5 DTE (~50% of the 10-DTE entry)", exact=True)
check("scanner._best_exit_window(1) 0-2 DTE",        asc._best_exit_window(1),
      "same day: best ~9:30-11:30 AM ET, hard exit by ~2:00 PM ET (0-2 DTE theta cliff)", exact=True)

_fails = [c for c in _checks if not c[1]]
_w = max(len(c[0]) for c in _checks)
for label, ok, actual, expected in _checks:
    line = ("PASS " if ok else "FAIL ") + label.ljust(_w) + "  got=" + repr(actual)
    if not ok:
        line += "  want=" + repr(expected)
    print(line)
print("-" * 72)
print("%d/%d formula checks passed%s" % (len(_checks) - len(_fails), len(_checks),
                                          "  (OPTIONS_MATH_DEMO_FAIL active)" if DEMO_FAIL else ""))
sys.exit(2 if _fails else 0)
