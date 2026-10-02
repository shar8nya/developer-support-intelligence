@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Run scripts\setup.bat first & exit /b 1)
uvicorn app.main:app --host 127.0.0.1 --port 8000 %*
