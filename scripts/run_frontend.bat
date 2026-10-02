@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Run scripts\setup.bat first & exit /b 1)
streamlit run frontend\app.py %*
