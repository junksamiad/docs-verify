#!/bin/bash

# Backend Setup Script for Document Verification System
# Works on both macOS and Linux

echo "🚀 Setting up Document Verification Backend..."

# Navigate to backend directory
cd backend

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

echo "✅ Backend setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  cd backend && source .venv/bin/activate"
echo ""
echo "To start the backend server, run:"
echo "  python -m uvicorn server:app --reload --port 8000" 