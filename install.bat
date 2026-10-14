@echo off
rem Set up the VMI UI on Windows: create .venv next to this script and
rem install requirements.txt into it. Safe to re-run (updates the existing venv).
setlocal
cd /d "%~dp0"

set "CHECK=import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)"

rem Prefer the py launcher (picks the newest installed Python), else python on PATH.
set "PYTHON="
py -3 -c "%CHECK%" >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON (
    python -c "%CHECK%" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
    echo Python 3.13 or newer is required. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer.
    goto :fail
)
for /f "delims=" %%v in ('%PYTHON% --version') do echo Using %%v

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv || goto :fail
)

".venv\Scripts\python.exe" -m pip install --upgrade --disable-pip-version-check pip || goto :fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :fail

echo.
echo Installed. Start the app with run.bat
pause
exit /b 0

:fail
echo.
echo Install failed.
pause
exit /b 1
