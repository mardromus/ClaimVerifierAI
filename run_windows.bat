@echo off
rem ClaimVerifier AI: set up (first run only) and start the web app. Options are passed on, e.g.
rem   run_windows.bat -Config configs/lite.yaml
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_windows.ps1" %*
if errorlevel 1 pause
