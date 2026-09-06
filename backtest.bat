@echo off
cd /d "%~dp0"
echo ============================================================
echo   Alpha Scanner Backtest - VWAP + ORB entry detectors
echo   Runs a FAST 10-ticker set (indices + megacap tech), last
echo   ~6 months (roughly 8 min). It prints a time estimate first
echo   - press Ctrl+C to abort.
echo.
echo   For the FULL 49-ticker watchlist: delete the --tickers ...
echo   part from the last command line of this file.
echo   Custom: change the tickers, or add --start YYYY-MM-DD.
echo ============================================================
echo.
"C:\Users\santw\AppData\Local\Python\bin\python.exe" alpha_backtest.py --tickers SPY,QQQ,IWM,AAPL,NVDA,MSFT,AMZN,META,TSLA,AMD
echo.
echo === Backtest finished. Trades saved to backtest_trades.csv ===
pause
