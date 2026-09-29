"""Runs the options-math known-answer verifier (.claude/skills/options-math/verify_formulas.py) under
pytest, so the Hull-anchored worked examples (BS price, N(.), IV solver, delta, gamma, GEX, skew, P/C,
expected move, exit window, magnet target) are part of the single `pytest` run."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFY = os.path.join(ROOT, ".claude", "skills", "options-math", "verify_formulas.py")


def test_options_math_worked_examples_pass():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("OPTIONS_MATH_DEMO_FAIL", None)
    res = subprocess.run([sys.executable, VERIFY], capture_output=True, text=True, env=env, cwd=ROOT,
                         stdin=subprocess.DEVNULL)   # don't inherit a parent stdin that may be invalid (WinError 6)
    tail = "\n".join(res.stdout.splitlines()[-4:])
    assert res.returncode == 0, "known-answer verifier failed:\n" + res.stdout[-3000:]
    assert "formula checks passed" in tail
