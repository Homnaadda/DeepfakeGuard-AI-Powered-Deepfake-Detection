@echo off
cd /d "%~dp0"

echo Checking virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Use the python executable inside the venv directly
set VENV_PYTHON=venv\Scripts\python.exe

echo Installing dependencies...
"%VENV_PYTHON%" -m pip install dlib-bin
"%VENV_PYTHON%" -m pip install face-recognition-models click
"%VENV_PYTHON%" -m pip install face-recognition --no-deps
"%VENV_PYTHON%" -m pip install -r requirements.txt

echo Starting server...
"%VENV_PYTHON%" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
