@echo off
REM Daily phishing feed collector.
REM Uses the project's own virtualenv Python so pandas/requests are available.
REM Appends output to logs\accumulate.log so failures are visible after the fact.

cd /d "%~dp0"
if not exist logs mkdir logs

echo. >> logs\accumulate.log
echo ===== %DATE% %TIME% ===== >> logs\accumulate.log
".venv\Scripts\python.exe" "src\accumulate.py" >> logs\accumulate.log 2>&1
