#!/usr/bin/env python
"""PostToolUse hook entry for the options-math verifier.

Claude Code pipes the tool call as JSON on stdin. If the edited file is one of the pricing modules
(alpha_scanner.py / alpha_backtest.py), run verify_formulas.py and exit 2 on any mismatch so the
edit is BLOCKED with the discrepancy shown to Claude. Any other file -> exit 0 (skip silently), so
the hook is scoped to exactly the modules that hold pricing/Greeks/IV/GEX/max-pain logic.
"""
import json
import os
import subprocess
import sys

PRICING_MODULES = {"alpha_scanner.py", "alpha_backtest.py"}
HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, "verify_formulas.py")


def main():
    try:
        raw = sys.stdin.buffer.read()
        if raw.startswith(b"\xef\xbb\xbf"):        # strip a UTF-8 BOM (some Windows shells add one)
            raw = raw[3:]
        payload = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return 0                                   # no / unparseable payload -> never block
    file_path = ((payload.get("tool_input") or {}).get("file_path")) or ""
    if os.path.basename(file_path) not in PRICING_MODULES:
        return 0                                   # not a pricing module -> skip

    res = subprocess.run([sys.executable, VERIFY], capture_output=True, text=True,
                         env=os.environ.copy())
    if res.returncode != 0:
        sys.stderr.write(
            "options-math verification FAILED after editing %s.\n"
            "A pricing / Greeks / IV / GEX / skew formula no longer reproduces its known-correct "
            "worked example from the options-math Skill. Either the change introduced a bug, or the "
            "change is intended -- in which case re-derive the value against the cited source (Hull, "
            "etc.) and update BOTH SKILL.md and verify_formulas.py before proceeding.\n\n"
            "%s\n%s\n" % (os.path.basename(file_path), res.stdout, res.stderr))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
