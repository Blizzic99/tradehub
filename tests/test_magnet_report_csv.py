"""Regression + known-answer test for alpha_backtest.magnet_forward_report's CSV export (--magnet-csv),
which used to crash with "name 'csv' is not defined". Daily closes are faked; no network."""
import csv
import json
from datetime import date


def test_magnet_report_writes_csv_with_known_answers(ab, monkeypatch, tmp_path):
    hist = {"AAA": [{"date": "2026-09-24", "expiry": "2026-09-25", "max_pain": 102.0, "spot": 100.0}],
            "BBB": [{"date": "2026-09-24", "expiry": "2026-09-25", "max_pain": 49.0, "spot": 50.0}]}
    hp = tmp_path / "h.json"
    hp.write_text(json.dumps(hist))
    closes = {"AAA": {date(2026, 9, 24): 100.0, date(2026, 9, 25): 101.0},
              "BBB": {date(2026, 9, 24): 50.0, date(2026, 9, 25): 49.5}}
    monkeypatch.setattr(ab, "_fetch_daily_closes", lambda tk, s, e, *a, **k: closes[tk])
    out = tmp_path / "m.csv"
    trades = ab.magnet_forward_report(str(hp), throttle_seconds=0, csv_out=str(out), max_distance_pct=100)
    assert out.exists(), "the --magnet-csv export was not written"
    rows = {r["direction"]: r for r in csv.DictReader(open(out))}
    # AAA: spot 100 below pin 102 -> UP; close 101 -> +1.0%. BBB: spot 50 above pin 49 -> DOWN; close 49.5 -> +1.0%
    assert float(rows["UP"]["return_pct"]) == 1.0
    assert float(rows["DOWN"]["return_pct"]) == 1.0
    assert len(trades) == 2
