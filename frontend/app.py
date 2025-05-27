from flask import Flask, render_template, request, jsonify
import requests # To make requests to the FastAPI backend
import os

app = Flask(__name__)

# Configuration
FASTAPI_BACKEND_URL_CLASSIFY = os.environ.get("FASTAPI_BACKEND_URL_CLASSIFY", "http://127.0.0.1:8000/classify-image/")
FASTAPI_BACKEND_URL_CONFIG = os.environ.get("FASTAPI_BACKEND_URL_MODELS", "http://127.0.0.1:8000/get-model-display-names")

@app.route('/')
def index():
    model_names = {
        "openai": "OpenAI (Default)", # Default/fallback names
        "gemini": "Google Gemini (Default)"
    }
    possible_doc_types = [] # Default empty list
    try:
        response = requests.get(FASTAPI_BACKEND_URL_CONFIG, timeout=5)
        response.raise_for_status()
        config_data = response.json()
        
        model_names["openai"] = config_data.get("openai_model_name", model_names["openai"])
        model_names["gemini"] = config_data.get("gemini_model_name", model_names["gemini"])
        possible_doc_types = config_data.get("possible_doc_types", []) # Get doc types
        
        print(f"Fetched model names: {model_names}")
        print(f"Fetched doc types: {possible_doc_types}")
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch config from backend: {e}. Using defaults.")
    # Pass the model names (fetched or default) to the template
    return render_template('index.html', model_names=model_names, possible_doc_types=possible_doc_types)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    detail = request.form.get('detail', 'auto') # Get detail level from form
    ai_provider = request.form.get('ai_provider', 'openai') # Get AI provider

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        files = {'file': (file.filename, file.stream, file.mimetype)}
        payload = {
            'detail': detail,
            'ai_provider': ai_provider # Add AI provider to payload
        }
        
        try:
            print(f"Forwarding to backend. Detail: {detail}, AI Provider: {ai_provider}, File: {file.filename}")
            response = requests.post(FASTAPI_BACKEND_URL_CLASSIFY, files=files, data=payload, timeout=180) # Use specific URL for classification
            response.raise_for_status()
            
            # Assuming the backend returns JSON with 'document_type' and 'filename'
            return jsonify(response.json()), response.status_code
        
        except requests.exceptions.HTTPError as e:
            # Try to return the error message from the backend if possible
            error_message = f"Backend error: {e.response.status_code}"
            try:
                error_detail = e.response.json().get("detail", e.response.text)
                error_message += f" - {error_detail}"
            except ValueError: # If response is not JSON
                error_message += f" - {e.response.text}"
            print(f"HTTPError when calling backend: {error_message}")
            return jsonify({"error": error_message}), e.response.status_code if e.response is not None else 500
        except requests.exceptions.RequestException as e:
            print(f"RequestException when calling backend: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File upload failed"}), 500

if __name__ == '__main__':
    # Make sure to create 'static' and 'templates' folders in the 'frontend' directory.
    # Your index.html should be in 'frontend/templates/'
    # Your style.css should be in 'frontend/static/'
    app.run(debug=True, port=5002) 