@echo off
REM One-time setup on Windows: creates .venv, installs dependencies, creates .env
cd /d "%~dp0.."
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
if not exist .venv (
    echo Creating virtual environment...
    %PY% -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements-dev.txt || goto :error
if not exist .env (
    copy .env.example .env >nul
    echo Created .env from .env.example ^(demo mode - no keys needed^)
)
echo.
echo Setup complete. Next: scripts\run_api.bat  and, in a second terminal, scripts\run_frontend.bat
exit /b 0
:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH ^(https://www.python.org/downloads/^).
exit /b 1
