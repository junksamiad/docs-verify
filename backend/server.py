from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
# OpenAI Classifier
from .image_doc_classifier import classify_image_document_type as classify_with_openai
from .image_doc_classifier import OPENAI_FRIENDLY_NAME, convert_to_png_if_needed as convert_for_openai_or_gemini
# Gemini Classifier
from .gemini_doc_classifier import classify_image_with_gemini
from .gemini_doc_classifier import GEMINI_FRIENDLY_NAME
from .passport_agent import analyze_passport_image # Import the new passport agent

app = FastAPI()

# Define a list of possible document types your agent can classify
POSSIBLE_DOC_TYPES = ["Bank Statement", "Utility Bill", "Council Tax Bill", "CV", "P45", "P60", "NI Letter", "Pay Slip", "Passport", "Right To Work Share Code", "DBS Certificate", "Police Check Certificate", "Drivers Licence", "Training Certificate", "Birth Certificate", "HMRC Letter", "DWP Letter", "Other"]

# Create a temporary directory for uploads if it doesn't exist
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/get-model-display-names")
async def get_app_config():
    return {
        "openai_model_name": OPENAI_FRIENDLY_NAME,
        "gemini_model_name": GEMINI_FRIENDLY_NAME,
        "possible_doc_types": POSSIBLE_DOC_TYPES
    }

@app.post("/classify-image/")
async def classify_image_endpoint(file: UploadFile = File(...), 
                                detail: str = Form("auto"), 
                                ai_provider: str = Form("openai")):
    """
    Endpoint to classify an uploaded image.
    Accepts an image file, an optional 'detail' level (auto, low, high) for OpenAI,
    and an 'ai_provider' (openai or gemini).
    """
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    classification_result = None
    actual_ai_provider = ai_provider.lower()
    final_response_content = {}
    path_for_further_processing = None # Path to the image that was actually classified (original or converted)

    # --- START: File Type Validation ---
    allowed_mime_types = [
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "application/pdf", 
        "image/heic", "image/heif" # Common MIME types for HEIC/HEIF
    ]
    # Add more HEIC/HEIF variants if needed, e.g., from a comprehensive list
    
    print(f"[SERVER LOG] Received file: '{file.filename}' with content type: '{file.content_type}', requested provider: {ai_provider}") # Log received content type

    if file.content_type not in allowed_mime_types:
        # Fallback check for extensions if MIME type is generic or not in the precise list
        file_ext = os.path.splitext(file.filename)[1].lower()
        if not ( (file.content_type == "application/octet-stream" or not file.content_type) and 
                   file_ext in [".pdf", ".heic", ".heif", ".png", ".jpg", ".jpeg", ".gif", ".webp"] ):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: '{file.content_type if file.content_type else file_ext}'. Please upload a valid image (JPEG, PNG, GIF, WEBP), PDF, or HEIC/HEIF file."
            )
    # --- END: File Type Validation ---

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate detail parameter (though the agent also does this)
        if detail not in ["auto", "low", "high"]:
            raise HTTPException(status_code=400, detail="Invalid detail level. Must be 'auto', 'low', or 'high'.")

        if actual_ai_provider == "openai":
            print("[SERVER LOG] Using OpenAI for initial classification.")
            # classify_with_openai handles its own conversion and uses the original temp_file_path or a converted one.
            # It returns the classification, but we also need the path it processed for potential further steps.
            # The current `classify_with_openai` doesn't explicitly return the path it used internally for classification.
            # We will call the converter first, then pass the path to the classifier.
            
            converted_path, final_mime = convert_for_openai_or_gemini(temp_file_path, file.content_type)
            if not converted_path:
                raise HTTPException(status_code=500, detail="Image conversion failed for OpenAI processing.")
            path_for_further_processing = converted_path
            
            classification_result = classify_with_openai(
                image_path=path_for_further_processing, # Use the potentially converted path
                original_file_mime_type=final_mime, # Use the mime of the processed image
                document_types=POSSIBLE_DOC_TYPES,
                detail=detail
            )
            print(f"[CLASSIFICATION LOG - OpenAI] Initial classification: '{classification_result}' for file: '{file.filename}'")
        
        elif actual_ai_provider == "gemini":
            print("[SERVER LOG] Using Gemini for initial classification.")
            converted_path, final_mime = convert_for_openai_or_gemini(temp_file_path, file.content_type)
            if not converted_path:
                raise HTTPException(status_code=500, detail="Image conversion failed for Gemini processing.")
            path_for_further_processing = converted_path

            classification_result = classify_image_with_gemini(
                image_path=path_for_further_processing,
                image_mime_type=final_mime,
                document_types=POSSIBLE_DOC_TYPES
            )
            print(f"[CLASSIFICATION LOG - Gemini] Initial classification: '{classification_result}' for file: '{file.filename}'")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid AI provider: '{ai_provider}'. Choose 'openai' or 'gemini'.")

        # --- Prepare base response --- 
        if classification_result:
            final_response_content = {
                "filename": file.filename, 
                "document_type": classification_result, 
                "detail_used": detail if actual_ai_provider == 'openai' else 'N/A for Gemini', 
                "ai_provider": actual_ai_provider,
                "passport_analysis": None # Initialize passport analysis field
            }
        else:
            raise HTTPException(status_code=500, detail=f"Could not classify the image with {actual_ai_provider}. Agent returned no result.")

        # --- Specialized Agent Processing (e.g., Passport) ---
        if classification_result == "Passport" and path_for_further_processing:
            print(f"[SERVER LOG] Document classified as Passport. Starting specialized Passport analysis for: {path_for_further_processing}")
            passport_analysis_results = analyze_passport_image(path_for_further_processing)
            if passport_analysis_results:
                final_response_content["passport_analysis"] = passport_analysis_results
                print("[SERVER LOG] Passport analysis completed.")
            else:
                print("[SERVER LOG] Passport analysis returned no results or an error.")
                # Optionally, include a note about failed sub-analysis in the response
                final_response_content["passport_analysis"] = {"error": "Passport specific analysis failed or returned no data."}
        
        return JSONResponse(content=final_response_content)

    except HTTPException as e:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch any other exceptions and return a generic server error
        # Log the error for debugging: print(f"Error during image classification: {e}")
        print(f"[SERVER ERROR] Error during image classification pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the classification pipeline: {str(e)}")
    finally:
        # Clean up the original uploaded file from UPLOAD_DIR
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[SERVER LOG] Cleaned up original uploaded file: {temp_file_path}")
        # Clean up the converted file if it was created by convert_for_openai_or_gemini and is different from temp_file_path
        # This is crucial because path_for_further_processing might be this converted file.
        if path_for_further_processing and path_for_further_processing != temp_file_path and os.path.exists(path_for_further_processing):
            if CONVERTED_TEMP_DIR in path_for_further_processing: # Ensure we only delete from our temp dir
                try:
                    os.remove(path_for_further_processing)
                    print(f"[SERVER LOG] Cleaned up centrally converted file: {path_for_further_processing}")
                except Exception as e_cleanup:
                    print(f"[SERVER LOG] Error cleaning up centrally converted file {path_for_further_processing}: {e_cleanup}") 