"""Shared fixtures + secret isolation.

ISOLATION (runs at import, before any test imports the app): the app's secret loaders read environment
variables, then <repo>/.streamlit/secrets.toml, then ~/.streamlit/secrets.toml. Tests must never load the
owner's real credentials, so the home directory is pointed at an empty temp dir (Windows' expanduser uses
USERPROFILE) and every secret env var is removed. test_tests_never_load_real_secrets enforces this.

The scanner is imported through alpha_backtest's offline stub-loader (streamlit and yfinance are stubbed,
the auto-scan is skipped), so tests never start Streamlit or touch the network.
"""
import os
import sys
import tempfile

import pytest

SECRET_ENV_VARS = ("POLYGON_KEY", "POLYGON_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                   "TELEGRAM_ALERTS_ENABLED")
ISOLATED_HOME = tempfile.mkdtemp(prefix="alphascanner-tests-home-")
for _var in ("HOME", "USERPROFILE"):
    os.environ[_var] = ISOLATED_HOME
for _var in SECRET_ENV_VARS:
    os.environ.pop(_var, None)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def ab():
    import alpha_backtest
    return alpha_backtest


@pytest.fixture(scope="session")
def asc(ab):
    return ab._asc
