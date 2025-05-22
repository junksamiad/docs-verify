from flask import Flask, render_template, request, jsonify
import requests # To make requests to the FastAPI backend
import os

app = Flask(__name__)

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/classify-image/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    detail = request.form.get('detail', 'auto') # Get detail level from form

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        files = {'file': (file.filename, file.stream, file.mimetype)}
        payload = {'detail': detail}
        
        try:
            response = requests.post(BACKEND_URL, files=files, data=payload, timeout=60) # Added timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            
            # Assuming the backend returns JSON with 'document_type' and 'filename'
            return jsonify(response.json()), response.status_code
        
        except requests.exceptions.HTTPError as e:
            # Try to return the error message from the backend if available
            error_message = "Error from backend"
            try:
                error_message = e.response.json().get("detail", error_message)
            except ValueError: # If response is not JSON
                error_message = e.response.text
            return jsonify({"error": error_message, "status_code": e.response.status_code}), e.response.status_code
        except requests.exceptions.RequestException as e:
            # For network errors, timeouts, etc.
            return jsonify({"error": f"Could not connect to backend: {str(e)}"}), 503
        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return jsonify({"error": "File upload failed"}), 500

if __name__ == '__main__':
    # Make sure to create 'static' and 'templates' folders in the 'frontend' directory.
    # Your index.html should be in 'frontend/templates/'
    # Your style.css should be in 'frontend/static/'
    app.run(debug=True, port=5000) 