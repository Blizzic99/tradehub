@echo off
cd /d "%~dp0"
echo ============================================================
echo   Alpha Scanner Backtest - VWAP + ORB entry detectors
echo   Default: full watchlist, last ~6 months. It prints a time
echo   estimate first - press Ctrl+C to abort if it's too long.
echo.
echo   For a quicker custom run, edit this file and add args, e.g.
echo     alpha_backtest.py --tickers SPY,QQQ --start 2026-08-01
echo ============================================================
echo.
"C:\Users\santw\AppData\Local\Python\bin\python.exe" alpha_backtest.py
echo.
echo === Backtest finished. Trades saved to backtest_trades.csv ===
pause
