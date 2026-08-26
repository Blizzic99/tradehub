@echo off
cd /d "%~dp0"
"C:\Users\santw\AppData\Local\Python\bin\python.exe" -m streamlit run alpha_scanner.py --server.port 8501
pause
