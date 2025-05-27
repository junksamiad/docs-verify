from google import genai # Updated import
from google.genai import types # Updated import for types
import os
import json
from PIL import Image # For reading image dimensions if needed, and basic ops
from pydantic import BaseModel, Field # For response schema definition
from typing import Optional, List

# --- Pydantic Model for Gemini Response ---
class GeminiDocTypeResponse(BaseModel):
    predicted_document_type: Optional[str] = Field(default=None)

# --- Model Configuration ---
GEMINI_MODEL_ID = 'gemini-1.5-flash'  # Updated to a common and available model
GEMINI_FRIENDLY_NAME = f"Google Gemini ({GEMINI_MODEL_ID})" # Display name for the UI
# --- End Model Configuration ---

_gemini_client = None

def get_gemini_client() -> Optional[genai.Client]:
    """Initializes and returns the Gemini client if API key is set."""
    global _gemini_client
    if _gemini_client:
        return _gemini_client

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error [GeminiClassifier]: GOOGLE_API_KEY environment variable not set.")
        return None
    try:
        _gemini_client = genai.Client(api_key=google_api_key)
        print("[GeminiClassifier LOG] Gemini client initialized successfully.")
        return _gemini_client
    except Exception as e:
        print(f"Error [GeminiClassifier]: Failed to initialize Gemini client: {e}")
        return None

def classify_document_with_gemini(file_path: str, original_mime_type: str, document_types: list[str]) -> str | None:
    """
    Classifies the type of document (image or PDF) using the Google Gemini API.
    For PDFs, it uploads to the File API. For images, it sends data directly.

    Args:
        file_path: Path to the document file (image or PDF).
        original_mime_type: The MIME type of the file (e.g., 'image/png', 'application/pdf').
        document_types: A list of strings representing the possible document types.

    Returns:
        The predicted document type as a string, or None if an error occurs.
    """
    client = get_gemini_client()
    if not client:
        return None # Error already printed by get_gemini_client
    
    if not os.path.exists(file_path):
        print(f"Error [GeminiClassifier]: File not found at {file_path}")
        return None
    if not document_types:
        print("Error [GeminiClassifier]: Document types list cannot be empty.")
        return None
    if not original_mime_type:
        print(f"Error [GeminiClassifier]: Invalid or missing original MIME type: {original_mime_type}")
        return None

    uploaded_file_for_gemini = None # To store reference to file uploaded to Gemini File API
    
    try:
        file_part = None
        if original_mime_type == "application/pdf":
            print(f"[GeminiClassifier LOG] Uploading PDF to Gemini File API: {file_path}")
            # Ensure correct parameters for upload_file, display_name is optional
            uploaded_file_for_gemini = client.files.upload_file(path=file_path, mime_type=original_mime_type)
            print(f"[GeminiClassifier LOG] PDF uploaded successfully: name='{uploaded_file_for_gemini.name}', uri='{uploaded_file_for_gemini.uri}'")
            # Ensure the part creation matches the latest SDK structure for file URIs
            file_part = types.Part(file_data=types.FileData(mime_type=uploaded_file_for_gemini.mime_type, file_uri=uploaded_file_for_gemini.uri))
        elif original_mime_type.startswith("image/"):
            print(f"[GeminiClassifier LOG] Reading image file: {file_path} with MIME type: {original_mime_type}")
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            file_part = types.Part.from_data(mime_type=original_mime_type, data=image_bytes)
        else:
            print(f"Error [GeminiClassifier]: Unsupported MIME type for Gemini processing: {original_mime_type}")
            return None
        
        prompt = (
            f"Analyze the document and classify its type. "
            f"The document could be an image or a PDF. "
            f"Your response MUST be a single JSON object. "
            f"The JSON object must have exactly one key: 'predicted_document_type'. "
            f"The value for 'predicted_document_type' must be one of the following exact strings: {document_types}. "
            f"If the document type is not clearly one of these, or if it's ambiguous, choose the closest match or 'Unknown' if that is an option. "
            f"Do not include any other text, explanations, or markdown formatting outside of this JSON object."
        )
        
        print(f"[GeminiClassifier LOG] Sending prompt to Gemini ({GEMINI_MODEL_ID}): {prompt[:200]}...")
        contents = [file_part, prompt] 
        
        generation_config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=GeminiDocTypeResponse
        )
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=contents,
            config=generation_config
        )
        
        api_response_text = None
        if response.text:
            api_response_text = response.text
            print(f"[GeminiClassifier LOG] Raw JSON response: {api_response_text}")
            try:
                # With response_schema, the response.text should be the JSON string.
                # Validation against Pydantic model is good practice.
                parsed_data = GeminiDocTypeResponse.model_validate_json(api_response_text)
                predicted_type = parsed_data.predicted_document_type

                if predicted_type and predicted_type in document_types:
                    print(f"[GeminiClassifier LOG] Predicted type: {predicted_type}")
                    return predicted_type
                elif predicted_type:
                    print(f"Warning [GeminiClassifier]: Model returned type '{predicted_type}' not in allowed list: {document_types}. Raw JSON: {api_response_text}")
                    # Optionally, map to "Unknown" if that's a valid type and desired behavior
                    if "Unknown" in document_types:
                        return "Unknown"
                    return None
                else:
                    print(f"Error [GeminiClassifier]: 'predicted_document_type' key missing or null in JSON response: {api_response_text}")
                    return None
            except Exception as pydantic_error:
                print(f"Error [GeminiClassifier]: Pydantic validation/parsing failed: {pydantic_error}. Raw response: {api_response_text}")
                return None
        else: # Fallback to check candidates if response.text is empty
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 api_response_text = response.candidates[0].content.parts[0].text
                 print(f"[GeminiClassifier LOG] Raw JSON response from candidate: {api_response_text}")
                 try:
                    parsed_data = GeminiDocTypeResponse.model_validate_json(api_response_text)
                    predicted_type = parsed_data.predicted_document_type
                    if predicted_type and predicted_type in document_types:
                        print(f"[GeminiClassifier LOG] Predicted type (from candidate): {predicted_type}")
                        return predicted_type
                    # ... (similar handling as above for not in list or missing key)
                 except Exception as pydantic_error:
                    print(f"[GeminiClassifier ERROR] Pydantic validation failed (from candidate): {pydantic_error}. Raw response: {api_response_text}")
                    return None

            print(f"Error [GeminiClassifier]: No valid text in Gemini API response. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason_name = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason
                print(f"Error [GeminiClassifier]: Request was blocked. Reason: {reason_name}")
            elif response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"Error [GeminiClassifier]: Did not finish successfully. Reason: {reason_name}. Details: {safety_ratings_str}")
            return None

    except Exception as e:
        print(f"An error occurred while calling the Gemini API or processing its response: {e}")
        return None
    finally:
        if uploaded_file_for_gemini:
            try:
                print(f"[GeminiClassifier LOG] Deleting uploaded file '{uploaded_file_for_gemini.name}' from Gemini File API.")
                client.files.delete_file(name=uploaded_file_for_gemini.name)
                print(f"[GeminiClassifier LOG] Successfully deleted '{uploaded_file_for_gemini.name}'.")
            except Exception as delete_e:
                print(f"Error [GeminiClassifier]: Failed to delete file '{uploaded_file_for_gemini.name}' from Gemini File API: {delete_e}")

if __name__ == '__main__':
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nSkipping direct tests for gemini_doc_classifier.py: GOOGLE_API_KEY not set.")
    else:
        # Create a dummy PNG file for testing if it doesn't exist
        sample_png_path = "temp_gemini_test_image.png"
        sample_mime_type = "image/png"
        if not os.path.exists(sample_png_path):
            try:
                img = Image.new('RGB', (200, 100), color = 'blue')
                img.save(sample_png_path)
                print(f"Created dummy test image: {sample_png_path}")
            except Exception as e:
                print(f"Could not create dummy image: {e}. Please ensure Pillow is installed.")
        
        doc_types = ["Invoice", "Receipt", "Letter", "ID Card", "Passport", "Report", "Unknown"]

        if os.path.exists(sample_png_path):
            print(f"\n--- Test Case 1: Classifying PNG '{sample_png_path}' ---")
            classification = classify_document_with_gemini(sample_png_path, sample_mime_type, doc_types)
            print(f"Predicted Document Type (Gemini, PNG): {classification}")
            
            print(f"\n--- Test Case 2: Classifying PNG with a more restrictive list ---")
            restricted_types = ["Invoice", "Receipt"]
            classification_restricted = classify_document_with_gemini(sample_png_path, sample_mime_type, restricted_types)
            print(f"Predicted Document Type (Gemini, PNG, restricted): {classification_restricted}")

        # Test with empty document types
        print("\n--- Test Case 3: Empty document types list (PNG, should return None) ---")
        classification_empty = classify_document_with_gemini(sample_png_path, sample_mime_type, [])
        print(f"Predicted Document Type (Gemini, PNG, empty types): {classification_empty}")

        # --- Add PDF Test Case ---
        sample_pdf_path = "temp_gemini_test_document.pdf"
        sample_pdf_mime_type = "application/pdf"
        if not os.path.exists(sample_pdf_path):
            try:
                # Create a very simple dummy PDF for testing (requires reportlab)
                from reportlab.pdfgen import canvas
                c = canvas.Canvas(sample_pdf_path)
                c.drawString(100, 750, "This is a test PDF document for Gemini classification.")
                c.drawString(100, 730, "It could be a CV or an Invoice.")
                c.save()
                print(f"Created dummy test PDF: {sample_pdf_path}")
            except ImportError:
                print(f"Skipping PDF test for Gemini: reportlab not installed. Cannot create dummy PDF.")
                sample_pdf_path = None # Ensure it's None so test doesn't run
            except Exception as e:
                print(f"Could not create dummy PDF: {e}")
                sample_pdf_path = None

        if sample_pdf_path and os.path.exists(sample_pdf_path):
            print(f"\n--- Test Case 4: Classifying PDF '{sample_pdf_path}' ---")
            classification_pdf = classify_document_with_gemini(sample_pdf_path, sample_pdf_mime_type, doc_types)
            print(f"Predicted Document Type (Gemini, PDF): {classification_pdf}")
        # --- End Add PDF Test Case ---

        # Clean up dummy files
        if os.path.exists(sample_png_path):
            try:
                os.remove(sample_png_path)
                print(f"Removed dummy test image: {sample_png_path}")
            except OSError as e:
                print(f"Error removing dummy test image {sample_png_path}: {e}")
        if sample_pdf_path and os.path.exists(sample_pdf_path): # Cleanup PDF
            try:
                os.remove(sample_pdf_path)
                print(f"Removed dummy test PDF: {sample_pdf_path}")
            except OSError as e:
                print(f"Error removing dummy test PDF {sample_pdf_path}: {e}") 