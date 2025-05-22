from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
# OpenAI Classifier
from .image_doc_classifier import classify_image_document_type as classify_with_openai
# Gemini Classifier
from .gemini_doc_classifier import classify_image_with_gemini

app = FastAPI()

# Define a list of possible document types your agent can classify
POSSIBLE_DOC_TYPES = ["Bank Statement", "Utility Bill", "Council Tax Bill", "CV", "P45", "P60", "NI Letter", "Pay Slip", "Passport", "Right To Work Share Code", "DBS Certificate", "Police Check Certificate", "Drivers Licence", "Training Certificate", "Birth Certificate", "HMRC Letter", "DWP Letter", "Other"]

# Create a temporary directory for uploads if it doesn't exist
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

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
            print("[SERVER LOG] Using OpenAI classifier.")
            # OpenAI classifier handles its own conversions and uses the original temp_file_path
            # It also needs the original file mime type for its internal conversion logic
            classification_result = classify_with_openai(
                image_path=temp_file_path,
                original_file_mime_type=file.content_type, 
                document_types=POSSIBLE_DOC_TYPES,
                detail=detail
            )
            print(f"[CLASSIFICATION LOG - OpenAI] Agent returned: '{classification_result}' for file: '{file.filename}' with detail: '{detail}'")
        
        elif actual_ai_provider == "gemini":
            print("[SERVER LOG] Using Gemini classifier.")
            # For Gemini, we first need to ensure the file is in a directly usable image format (e.g., PNG).
            # The OpenAI classifier's main function now internally handles conversion and returns the path to a processable image (or None if conversion fails)
            # Let's leverage the conversion part of classify_with_openai, but then call gemini.
            # This is a bit of a workaround. A cleaner way would be to refactor the conversion logic out of classify_with_openai
            # into a shared utility if this pattern becomes common.
            
            # We need a path to a processable image (PNG) for Gemini, and its MIME type.
            # The OpenAI classifier has the conversion logic. We'll call it to get a path to a PNG.
            # This is not ideal as it runs OpenAI-specific schema prep if not careful.
            # For now, we assume classify_with_openai can be called such that it primarily does conversion if needed
            # and returns a path. We need to modify classify_with_openai to better support this or extract conversion.

            # --- Temporary approach for Gemini: Use OpenAI converter, then pass to Gemini --- 
            # This requires classify_with_openai to be robust enough or be refactored.
            # Let's assume for a moment that we have a function `get_converted_png_path` from `image_doc_classifier`
            # For now, we will call the OpenAI one, and if it returns a path that Gemini can use, proceed.
            # The `image_doc_classifier.classify_image_document_type` function handles conversions internally
            # and then uses the converted path. We need to get that converted path or ensure it outputs one.
            
            # Simpler: Assume the file uploaded to server.py (temp_file_path) is what we first attempt to convert if needed.
            # The OpenAI classifier will do this conversion. If it succeeds, it uses the converted path.
            # We need *that* converted path for Gemini.
            
            # Let's call the OpenAI function, but we only really care about its converted image if it made one.
            # This is a bit messy. The ideal way is to have a standalone converter.
            # For this iteration, we'll call OpenAI converter first. It will produce a .png in CONVERTED_TEMP_DIR.
            # If the input was already a PNG/JPG, it would use original path.

            # We need to pass the *original* temp_file_path to the OpenAI function for potential conversion.
            # The OpenAI function (classify_with_openai) now handles conversion internally and returns the *final category*.
            # We need to modify `classify_with_openai` OR `gemini_doc_classifier` to handle the conversion step more cleanly if the input isn't already PNG.

            # The current `classify_with_openai` (formerly image_doc_classifier.py) handles conversion and then classification.
            # For Gemini, we need the *converted image path* first. 
            # Let's modify the structure: image_doc_classifier will have a function to convert & get path.
            # This is done in the latest version of image_doc_classifier.py (convert_to_png_if_needed)
            
            from .image_doc_classifier import convert_to_png_if_needed
            converted_path_for_gemini, final_mime_type_for_gemini = convert_to_png_if_needed(temp_file_path, file.content_type)

            if converted_path_for_gemini:
                classification_result = classify_image_with_gemini(
                    image_path=converted_path_for_gemini, 
                    image_mime_type=final_mime_type_for_gemini, # Should be image/png after conversion
                    document_types=POSSIBLE_DOC_TYPES
                )
                print(f"[CLASSIFICATION LOG - Gemini] Agent returned: '{classification_result}' for file: '{file.filename}'")
                # Clean up the *converted* file for Gemini if it was created and different from original temp_file_path
                if converted_path_for_gemini != temp_file_path and os.path.exists(converted_path_for_gemini):
                    try:
                        os.remove(converted_path_for_gemini)
                        print(f"[SERVER LOG] Cleaned up Gemini's converted file: {converted_path_for_gemini}")
                    except Exception as e_cleanup:
                        print(f"[SERVER LOG] Error cleaning up Gemini's converted file {converted_path_for_gemini}: {e_cleanup}")
            else:
                print(f"[SERVER LOG] File conversion failed for Gemini: {file.filename}")
                raise HTTPException(status_code=500, detail="Image conversion failed, cannot process with Gemini.")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid AI provider: '{ai_provider}'. Choose 'openai' or 'gemini'.")

        if classification_result:
            return JSONResponse(content={"filename": file.filename, "document_type": classification_result, "detail_used": detail if actual_ai_provider == 'openai' else 'N/A for Gemini', "ai_provider": actual_ai_provider})
        else:
            raise HTTPException(status_code=500, detail=f"Could not classify the image with {actual_ai_provider}. Agent returned no result.")

    except HTTPException as e:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch any other exceptions and return a generic server error
        # Log the error for debugging: print(f"Error during image classification: {e}")
        print(f"[SERVER ERROR] Error during image classification with {actual_ai_provider}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with {actual_ai_provider}: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) 
            print(f"[SERVER LOG] Cleaned up original uploaded file: {temp_file_path}") 