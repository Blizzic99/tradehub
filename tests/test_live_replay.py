"""Known-answer tests for live_rules_replay.py (the week-replay research tool): OCC symbols, time to
expiry, the scanner's stop/target formula under both anchors, the forward-only live exit simulator
(stop / target / thesis / 2 PM / session close, same-bar priority, wrong-side fills, shorts, no-fill),
the corrected readiness rule, and the bucket statistics. Synthetic bars only; no network."""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import live_rules_replay as R

ET = R.ET


def _session(post_closes, or_hi=100.5, or_lo=99.5, day=date(2026, 9, 24)):
    """15 opening-range bars (high or_hi / low or_lo, close 100) then post bars with given closes.
    Each post bar opens at the previous close. Index: 1-minute ET bars from 09:30."""
    t0 = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    rows = [{"open": 100.0, "high": or_hi, "low": or_lo, "close": 100.0, "volume": 1000} for _ in range(15)]
    prev = 100.0
    for c in post_closes:
        rows.append({"open": prev, "high": max(prev, c), "low": min(prev, c), "close": c, "volume": 1000})
        prev = c
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(len(rows))])
    df = pd.DataFrame(rows, index=idx)
    return df, pd.Series(100.0, index=idx)


def test_occ_ticker_known_answer():
    assert R.occ_ticker("NVDA", date(2026, 9, 24), True, 232.5) == "O:NVDA260924C00232500"
    assert R.occ_ticker("SPY", date(2026, 10, 2), False, 650) == "O:SPY261002P00650000"


def test_years_to_expiry_and_one_hour_floor():
    t = datetime(2026, 9, 24, 10, 0, tzinfo=ET)
    assert R.years_to_expiry(t, date(2026, 9, 24)) == pytest.approx(6 * 3600 / (365 * 24 * 3600))
    late = datetime(2026, 9, 24, 16, 30, tzinfo=ET)
    assert R.years_to_expiry(late, date(2026, 9, 24)) == pytest.approx(3600 / (365 * 24 * 3600))


def test_stop_target_formula_both_anchors_and_fallback():
    # scanner: stop = anchor -/+ 0.5E, target = anchor +/- 1.0E
    assert R.stop_target(100.0, True, 2.0) == (99.0, 102.0)
    assert R.stop_target(100.0, False, 2.0) == (101.0, 98.0)
    s, t = R.stop_target(100.0, True, None)            # no IV -> flat STOP_BUFFER stop, no target
    assert s == pytest.approx(100.0 * (1 - 0.0075)) and t is None
    # the NVDA row the user flagged: long, pin 232.5 as anchor, spot 230.69 -> stop ABOVE spot (the bug);
    # anchoring to spot puts it below
    E = 1.56
    assert R.stop_target(232.5, True, E)[0] > 230.69
    assert R.stop_target(230.69, True, E)[0] < 230.69


def test_long_stop_fills_at_stop_level():
    df, vw = _session([101.0, 101.2, 100.6])           # signal bar 15 (close 101), entry = open of bar 16 = 101.0
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=100.7, target=110.0, dte_days=5)
    assert r["exit_reason"] == "stop" and r["exit_price"] == 100.7 and r["entry_price"] == 101.0
    assert r["return_pct"] == pytest.approx((100.7 - 101.0) / 101.0 * 100)


def test_long_target_fills_at_target_level():
    df, vw = _session([101.0, 101.2, 101.6])
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=100.7, target=101.5, dte_days=5)
    assert r["exit_reason"] == "target" and r["exit_price"] == 101.5


def test_thesis_exit_when_back_inside_range():
    df, vw = _session([101.0, 101.2, 100.4])            # 100.4 <= or_high 100.5
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=99.0, target=105.0, dte_days=5)
    assert r["exit_reason"] == "thesis" and r["exit_price"] == 100.4


def test_stop_beats_thesis_on_the_same_bar():
    df, vw = _session([101.0, 101.2, 100.3])            # closes through stop 100.45 AND inside the range
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=100.45, target=105.0, dte_days=5)
    assert r["exit_reason"] == "stop" and r["exit_price"] == 100.45


def test_two_pm_hard_exit_only_for_0_to_2_dte():
    closes = [101.0] * 400                               # never hits stop/target/thesis
    df, vw = _session(closes)
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=90.0, target=110.0, dte_days=2)
    assert r["exit_reason"] == "time_2pm" and r["exit_time"].strftime("%H:%M") == "14:00"
    r3 = R.simulate_live(df, vw, 15, "ORB", "UP", stop=90.0, target=110.0, dte_days=3)
    assert r3["exit_reason"] == "session_close" and r3["exit_time"] == df.index[-1]


def test_wrong_side_stop_exits_at_market_and_is_flagged():
    df, vw = _session([101.0, 101.2, 101.3])
    r = R.simulate_live(df, vw, 15, "ORB", "UP", stop=101.5, target=110.0, dte_days=5)   # stop above entry
    assert r["stop_wrong_side"] is True
    assert r["exit_reason"] == "stop_wrongside" and r["exit_price"] == 101.2          # the bar close, not 101.5


def test_short_mirror():
    df, vw = _session([99.0, 98.8, 99.4])                 # ORB DOWN: signal close 99, entry 99.0
    r = R.simulate_live(df, vw, 15, "ORB", "DOWN", stop=99.3, target=90.0, dte_days=5)
    assert r["exit_reason"] == "stop" and r["exit_price"] == 99.3
    assert r["return_pct"] == pytest.approx((99.0 - 99.3) / 99.0 * 100)
    df2, vw2 = _session([99.0, 98.8, 97.9])
    r2 = R.simulate_live(df2, vw2, 15, "ORB", "DOWN", stop=99.3, target=98.0, dte_days=5)
    assert r2["exit_reason"] == "target" and r2["exit_price"] == 98.0


def test_vwap_thesis_is_close_below_vwap():
    df, _ = _session([101.0, 101.2, 100.9])
    vw = pd.Series(101.0, index=df.index)
    r = R.simulate_live(df, vw, 15, "VWAP", "LONG", stop=99.0, target=110.0, dte_days=5)
    assert r["exit_reason"] == "thesis" and r["exit_price"] == 100.9


def test_signal_on_last_bar_cannot_fill():
    df, vw = _session([101.0])
    assert R.simulate_live(df, vw, len(df) - 1, "ORB", "UP", 99.0, 105.0, 5) is None


@pytest.mark.parametrize("cur,delta,beyond,verdict", [
    ("PASS", 0.13, False, "FAIL"),    # the NVDA row: delta 0.13 must fail
    ("PASS", 0.45, False, "PASS"),
    ("PASS", -0.13, False, "FAIL"),   # put delta: magnitude
    ("PASS", -0.45, False, "PASS"),
    ("PASS", 0.30, False, "PASS"),    # boundary: >= 0.30 passes
    ("PASS", None, False, "FAIL"),    # unknown delta fails safe
    ("PASS", 0.45, True, "FAIL"),     # >1 sigma pin
    ("FAIL", 0.60, False, "FAIL"),    # never upgrades a current FAIL
])
def test_corrected_readiness_rule(cur, delta, beyond, verdict):
    assert R.corrected_readiness(cur, delta, beyond)[0] == verdict


def test_bucket_stats_known_answer():
    rows = [{"return_pct": 1.0, "date": "2026-09-23", "entry_time": "09:50"},
            {"return_pct": -0.5, "date": "2026-09-24", "entry_time": "09:50"},
            {"return_pct": 2.0, "date": "2026-09-25", "entry_time": "09:50"}]
    st = R.bucket_stats(rows, baseline=0.1)
    assert st["n"] == 3 and st["win"] == pytest.approx(200 / 3)
    assert st["exp"] == pytest.approx(2.5 / 3) and st["vsbase"] == pytest.approx(2.5 / 3 - 0.1)
    assert st["pf"] == pytest.approx(6.0) and st["mdd"] == pytest.approx(0.5)
