# Document Verification System

A multi-modal document verification system that classifies and analyzes various document types using AI.

## Features

- **Document Classification**: Supports images, PDFs, DOCX, and TXT files
- **AI Providers**: Choose between OpenAI and Google Gemini for classification
- **Specialized Analysis**: Detailed analysis for passports, CVs, driving licenses, and bank statements
- **Modern UI**: Clean, responsive web interface
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Quick Setup

### Prerequisites

- **Python 3.8+** installed on your system
- **API Keys** (see Environment Setup below)

### For macOS/Linux Users

```bash
# Clone the repository
git clone <your-repo-url>
cd docs-verify

# Run the setup script
./setup_backend.sh

# Set up environment variables (see Environment Setup below)
```

### For Windows Users

```cmd
# Clone the repository
git clone <your-repo-url>
cd docs-verify

# Run the setup script
setup_backend.bat

# Set up environment variables (see Environment Setup below)
```

## Environment Setup

Create a `.env` file in the `backend/` directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## Running the Application

### Backend (FastAPI)

**Option 1: Using the start script (recommended)**
```bash
cd backend
./start_server.sh          # On macOS/Linux
# OR
start_server.bat           # On Windows
```

**Option 2: Manual start**
```bash
cd backend
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat
python -m uvicorn server:app --reload --port 8000
```

### Frontend (Flask)

```bash
cd frontend
source .venv/bin/activate  # Create frontend venv if needed
python app.py
```

Access the application at: `http://localhost:5001`

## Supported File Types

- **Images**: PNG, JPEG, GIF, WEBP, HEIC/HEIF
- **Documents**: PDF, DOCX, TXT

## Document Types Supported

- Bank Statement
- Utility Bill
- Council Tax Bill
- CV/Resume
- P45, P60
- NI Letter
- Pay Slip
- Passport
- Right To Work Share Code
- DBS Certificate
- Police Check Certificate
- Drivers Licence
- Training Certificate
- Birth Certificate
- HMRC Letter
- DWP Letter
- Other documents

## Architecture

### Two-Stage Pipeline

1. **Classification Stage**: Determines document type
   - OpenAI GPT-4o (for images/PDFs)
   - Google Gemini (for images/PDFs)
   - Text classifier (for DOCX/TXT)

2. **Analysis Stage**: Extracts detailed information
   - **Passport Agent**: Uses OpenAI for verification flags and data extraction
   - **CV Agent**: Uses Google Gemini for personal details and work gap analysis
   - **Driving License Agent**: Uses OpenAI for license details and verification
   - **Bank Statement Agent**: Uses Google Gemini for transaction categorization

## Development

### Project Structure

```
docs-verify/
├── backend/                 # FastAPI backend
│   ├── server.py           # Main server file
│   ├── *_agent.py          # Specialized document agents
│   ├── *_classifier.py     # Document classifiers
│   └── requirements.txt    # Python dependencies
├── frontend/               # Flask frontend
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/           # CSS and assets
└── setup_backend.*       # Setup scripts
```

### Adding New Document Types

1. Add the document type to `POSSIBLE_DOC_TYPES` in `backend/server.py`
2. Create a new agent file (e.g., `new_document_agent.py`)
3. Add routing logic in the specialized agent processing section
4. Update frontend display logic if needed

## Troubleshooting

### PDF Processing Issues

If PDF processing fails, ensure Poppler is installed:

**macOS**: `brew install poppler`
**Ubuntu/Debian**: `sudo apt-get install poppler-utils`
**Windows**: Download from [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/)

### Virtual Environment Issues

- Never commit virtual environments to git
- Use the provided setup scripts to create fresh environments
- Ensure you're using the correct Python version (3.8+)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here] 