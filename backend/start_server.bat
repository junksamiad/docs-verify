@echo off

REM Start the Document Verification Backend Server
echo 🚀 Starting Document Verification Backend Server...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start the server
echo 📡 Server starting on http://localhost:8000
python -m uvicorn server:app --reload --port 8000

pause 