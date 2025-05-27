@echo off

REM Backend Setup Script for Document Verification System
REM Works on Windows

echo 🚀 Setting up Document Verification Backend...

REM Navigate to backend directory
cd backend

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python packages...
pip install -r requirements.txt

echo ✅ Backend setup complete!
echo.
echo To activate the environment in the future, run:
echo   cd backend ^&^& .venv\Scripts\activate.bat
echo.
echo To start the backend server, run:
echo   python -m uvicorn server:app --reload --port 8000

pause 