@echo off
echo.
echo ========================================
echo 🚀 Document Verification System Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM ===========================================
REM BACKEND SETUP
REM ===========================================
echo 📦 Setting up Backend...
echo.

REM Navigate to backend directory (we're already here)
if not exist .venv (
    echo 🔧 Creating backend virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create backend virtual environment
        pause
        exit /b 1
    )
    echo ✅ Backend virtual environment created
) else (
    echo ✅ Backend virtual environment already exists
)

echo 🔧 Activating backend virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate backend virtual environment
    pause
    exit /b 1
)

echo 📥 Installing backend dependencies...
pip install --upgrade pip
pip install fastapi uvicorn python-multipart python-dotenv openai google-genai pillow pdf2image python-docx lxml requests pillow-heif
if errorlevel 1 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)
echo ✅ Backend dependencies installed

REM Setup backend .env file
if not exist .env (
    if exist .env.example (
        echo 🔧 Creating backend .env from template...
        copy .env.example .env
        echo ✅ Backend .env created from template
        echo ⚠️  Please edit backend\.env and add your API keys
    ) else (
        echo 🔧 Creating backend .env file...
        echo # Backend Environment Variables > .env
        echo OPENAI_API_KEY=your_openai_api_key_here >> .env
        echo GOOGLE_API_KEY=your_google_api_key_here >> .env
        echo ✅ Backend .env created
        echo ⚠️  Please edit backend\.env and add your API keys
    )
) else (
    echo ✅ Backend .env already exists
)

echo.

REM ===========================================
REM FRONTEND SETUP
REM ===========================================
echo 🌐 Setting up Frontend...
echo.

REM Navigate to frontend directory
cd ..\frontend
if errorlevel 1 (
    echo ❌ Frontend directory not found
    pause
    exit /b 1
)

if not exist .venv (
    echo 🔧 Creating frontend virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create frontend virtual environment
        pause
        exit /b 1
    )
    echo ✅ Frontend virtual environment created
) else (
    echo ✅ Frontend virtual environment already exists
)

echo 🔧 Activating frontend virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate frontend virtual environment
    pause
    exit /b 1
)

echo 📥 Installing frontend dependencies...
pip install --upgrade pip
pip install flask requests
if errorlevel 1 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)
echo ✅ Frontend dependencies installed

REM Setup frontend .env file
if not exist .env (
    if exist .env.example (
        echo 🔧 Creating frontend .env from template...
        copy .env.example .env
        echo ✅ Frontend .env created from template
    ) else (
        echo 🔧 Creating frontend .env file...
        echo # Frontend Environment Variables > .env
        echo BACKEND_URL=http://localhost:8000 >> .env
        echo ✅ Frontend .env created
    )
) else (
    echo ✅ Frontend .env already exists
)

echo.

REM ===========================================
REM START SERVERS
REM ===========================================
echo 🚀 Starting Document Verification Servers...
echo.

REM Start backend server in new window
echo 📡 Starting Backend Server (Port 8000)...
cd ..\backend
start "📡 Backend Server - Document Verification" cmd /k "call .venv\Scripts\activate.bat && echo Backend Server Starting... && python -m uvicorn server:app --reload --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server in new window
echo 🌐 Starting Frontend Server (Port 5000)...
cd ..\frontend
start "🌐 Frontend Server - Document Verification" cmd /k "call .venv\Scripts\activate.bat && echo Frontend Server Starting... && python app.py"

echo.
echo ========================================
echo ✅ Setup Complete!
echo ========================================
echo.
echo 🌐 Frontend: http://localhost:5000
echo 📡 Backend:  http://localhost:8000
echo.
echo 📝 Note: If this is your first time running:
echo    1. Edit backend\.env with your API keys
echo    2. Edit frontend\.env if needed
echo    3. Restart the servers after adding API keys
echo.
echo 🔄 Both servers are starting in separate windows...
echo 🚪 You can close this window once servers are running
echo.

timeout /t 5 /nobreak >nul
echo Press any key to exit this setup window...
pause >nul 