from google import genai
from google.genai.types import Part, GenerateContentConfig, Blob

import os
import json
from PIL import Image # For reading image dimensions if needed, and basic ops
from pydantic import BaseModel, Field # For response schema definition
from typing import Optional, List

# Initialize client (will pick up GOOGLE_API_KEY from env)
client: Optional[genai.Client] = None
if os.environ.get("GOOGLE_API_KEY"):
    try:
        client = genai.Client()
        print("🔑 GEMINI: API client initialized")
    except Exception as e:
        print(f"❌ GEMINI: Client initialization failed: {e}")
        client = None # Ensure client is None if initialization fails
else:
    print("⚠️ GEMINI: GOOGLE_API_KEY not set - client disabled")

# --- Pydantic Model for Gemini Response ---
class GeminiDocTypeResponse(BaseModel):
    predicted_document_type: Optional[str] = Field(default=None)

# --- Model Configuration ---
GEMINI_MODEL_ID = 'gemini-2.5-flash-preview-05-20'  # Ensure this is the desired flash model
GEMINI_FRIENDLY_NAME = f"Google Gemini ({GEMINI_MODEL_ID})" # Display name for the UI
# --- End Model Configuration ---

def classify_document_with_gemini(file_path: str, original_mime_type: str, document_types: list[str]) -> str | None:
    """
    Classifies the type of document shown in an image or PDF using the Google Gemini API.

    Args:
        file_path: Path to the file (image or PDF).
        original_mime_type: The original MIME type of the file (e.g., 'image/png', 'application/pdf').
        document_types: A list of strings representing the possible document types.

    Returns:
        The predicted document type as a string, or None if an error occurs.
    """
    global client
    if not client:
        print("Error [GeminiClassifier]: Gemini client not initialized. Cannot proceed.")
        return None
    
    # Model ID is now defined globally, no need to create genai.GenerativeModel() here
    # The client.models.generate_content takes the model string directly.

    if not os.path.exists(file_path):
        print(f"Error [GeminiClassifier]: File not found at {file_path}")
        return None
    if not document_types:
        print("Error [GeminiClassifier]: Document types list cannot be empty.")
        return None
    
    # Determine the MIME type to use for the API call
    api_mime_type = original_mime_type
    if original_mime_type not in ["application/pdf", "image/png", "image/jpeg", "image/webp", "image/gif", "image/heic", "image/heif"]:
        # Basic fallback for safety, though server.py should provide a valid one.
        # If it's an image that was converted, it should be image/png by now.
        # If it's an unknown image type, this might be an issue.
        print(f"Warning [GeminiClassifier]: Potentially unsupported original_mime_type '{original_mime_type}'. Attempting as octet-stream or will rely on server conversion to PNG for images.")
        # For now, assume if it's not PDF, it's an image type Gemini can handle or has been converted to PNG by server.py
        if not original_mime_type.startswith("image/") and original_mime_type != "application/pdf":
             api_mime_type = "image/png" # Default to png if it was converted by server.py from something exotic

    try:
        print(f"📁 GEMINI: Processing {api_mime_type} file")
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        # Manual Part construction for file data
        file_part = Part(
            inline_data=Blob(
                data=file_bytes,
                mime_type=api_mime_type
            )
        )
        
        prompt_text_content = (
            f"Analyze the document (image or PDF) and classify its type. "
            f"Your response MUST be a single JSON object. "
            f"The JSON object must have exactly one key: 'predicted_document_type'. "
            f"The value for 'predicted_document_type' must be one of the following exact strings: {document_types}. "
            f"If the document type is not clearly one of these, or if it's ambiguous, choose the closest match or 'Unknown' if that is an option. "
            f"Do not include any other text, explanations, or markdown formatting outside of this JSON object."
        )
        prompt_part = Part(text=prompt_text_content) # Manual text Part construction
        
        print(f"🤖 GEMINI: API call sent ({GEMINI_MODEL_ID})")

        contents = [file_part, prompt_part] # File part first, then prompt part
        
        # Use GenerateContentConfig as per google_libraries.md for new SDK client.models.generate_content config
        generation_config_obj = GenerateContentConfig( 
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=4096, # Adjusted for classification (was 50000, then 2048, setting to 4096)
            # response_schema is typically used with genai.GenerativeModel().generate_content(), 
            # For client.models.generate_content(), the schema is often passed as a dict within the config if needed, 
            # or relied upon by the model if response_mime_type is json.
            # Let's omit response_schema here to align with the simpler client.models.generate_content examples.
        )
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID, # Pass model string directly
            contents=contents,
            config=generation_config_obj 
        )
        
        # Log finish reason if candidates exist
        if response.candidates:
            try:
                finish_reason_name = response.candidates[0].finish_reason.name
                print(f"[GeminiClassifier LOG] Gemini API call finish_reason: {finish_reason_name}")
            except AttributeError:
                print("[GeminiClassifier WARNING] Could not retrieve finish_reason name from response candidate.")
        else:
            print("[GeminiClassifier WARNING] No candidates found in Gemini response to log finish_reason.")

        api_response_text = None
        if response.text:
            api_response_text = response.text
            print(f"📝 GEMINI: Raw response: {api_response_text}")
            try:
                # With response_schema, the response.text should be the JSON string.
                # Validation against Pydantic model is good practice.
                parsed_data = GeminiDocTypeResponse.model_validate_json(api_response_text)
                predicted_type = parsed_data.predicted_document_type

                if predicted_type and predicted_type in document_types:
                    print(f"✅ GEMINI: Classification successful → {predicted_type}")
                    return predicted_type
                elif predicted_type:
                    print(f"⚠️ GEMINI: Invalid type '{predicted_type}' not in allowed list")
                    # Optionally, map to "Unknown" if that's a valid type and desired behavior
                    if "Unknown" in document_types:
                        return "Unknown"
                    return None
                else:
                    print(f"❌ GEMINI: Missing 'predicted_document_type' in response")
                    return None
            except Exception as pydantic_error:
                print(f"❌ GEMINI: Validation error: {pydantic_error}")
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
            # Log finish_reason here as well if it's not STOP and not already logged, or if it indicates an issue
            elif response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                # This log might be redundant if already logged above, but good for error path clarity
                reason_name = response.candidates[0].finish_reason.name 
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"Error [GeminiClassifier]: Did not finish successfully. Finish Reason: {reason_name}. Details: {safety_ratings_str}")
            # else if already logged and was STOP, no need to repeat error for STOP.
            return None

    except Exception as e:
        print(f"[GeminiClassifier ERROR] An error occurred while calling the Gemini API or processing its response: {e}")
        import traceback
        print("[GeminiClassifier TRACEBACK]")
        traceback.print_exc() # Print full traceback for more details
        return None

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
            print(f"\n--- Test Case 1: Classifying image '{sample_png_path}' ---")
            classification = classify_document_with_gemini(sample_png_path, sample_mime_type, doc_types)
            print(f"Predicted Document Type (Gemini Image): {classification}")
            
            print(f"\n--- Test Case 2: Classifying image with a more restrictive list ---")
            restricted_types = ["Invoice", "Receipt"]
            classification_restricted = classify_document_with_gemini(sample_png_path, sample_mime_type, restricted_types)
            print(f"Predicted Document Type (Gemini Image, restricted): {classification_restricted}")

        # Test with a dummy PDF
        sample_pdf_path = "temp_gemini_test_doc.pdf"
        sample_pdf_mime_type = "application/pdf"
        if not os.path.exists(sample_pdf_path):
            try:
                from reportlab.pdfgen import canvas
                c = canvas.Canvas(sample_pdf_path)
                c.drawString(100, 750, "This is a test PDF document for classification.")
                c.drawString(100, 730, "It could be an Invoice or a Report.")
                c.save()
                print(f"Created dummy test PDF: {sample_pdf_path}")
            except ImportError:
                print("reportlab is not installed. Cannot create a dummy PDF. pip install reportlab")
            except Exception as e:
                print(f"Could not create dummy PDF: {e}")

        if os.path.exists(sample_pdf_path):
            print(f"\n--- Test Case 3: Classifying PDF '{sample_pdf_path}' ---")
            classification_pdf = classify_document_with_gemini(sample_pdf_path, sample_pdf_mime_type, doc_types)
            print(f"Predicted Document Type (Gemini PDF): {classification_pdf}")

        # Test with empty document types
        print("\n--- Test Case 4: Empty document types list (should return None) ---")
        classification_empty = classify_document_with_gemini(sample_png_path, sample_mime_type, [])
        print(f"Predicted Document Type (Gemini, empty types): {classification_empty}")

        # Clean up dummy files
        if os.path.exists(sample_png_path):
            try:
                os.remove(sample_png_path)
                print(f"Removed dummy test image: {sample_png_path}")
            except OSError as e:
                print(f"Error removing dummy test image {sample_png_path}: {e}")
        if os.path.exists(sample_pdf_path):
            try:
                os.remove(sample_pdf_path)
                print(f"Removed dummy test PDF: {sample_pdf_path}")
            except OSError as e:
                print(f"Error removing dummy test PDF {sample_pdf_path}: {e}") 