@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Run scripts\setup.bat first & exit /b 1)
python scripts\apply_migrations.py %*
