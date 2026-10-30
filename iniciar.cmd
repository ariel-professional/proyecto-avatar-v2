@echo off
setlocal
set "AVATAR_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%AVATAR_PYTHON%" (
    set "AVATAR_PYTHON=python"
)
"%AVATAR_PYTHON%" "%~dp0main.py" %*
if errorlevel 1 pause
