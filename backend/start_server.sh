#!/bin/bash

# Start the Document Verification Backend Server
echo "🚀 Starting Document Verification Backend Server..."

# Activate virtual environment
source .venv/bin/activate

# Start the server
echo "📡 Server starting on http://localhost:8000"
python -m uvicorn server:app --reload --port 8000 