@echo off
rem Start the VMI UI on Windows with the venv made by install.bat.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv here yet -- run install.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" UI.py %*
if errorlevel 1 pause
