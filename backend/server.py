import os # Ensure os is imported if not already
from dotenv import load_dotenv # Removed find_dotenv for now, will use explicit path

# Construct the path to the .env file relative to this script (server.py)
# __file__ is the path to the current script (server.py)
# os.path.dirname(__file__) is the directory of the current script (backend/)
# os.path.join(os.path.dirname(__file__), '.env') is backend/.env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Environment variable status
openai_key_status = "✅" if os.getenv('OPENAI_API_KEY') else "❌"
google_key_status = "✅" if os.getenv('GOOGLE_API_KEY') else "❌"
print(f"🔑 ENV KEYS: OpenAI {openai_key_status} | Google {google_key_status}")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import shutil
import os
import json
import mimetypes # For guessing MIME types
# Classifier Agents
from classifier_agents.openai_doc_classifier import classify_image_document_type as classify_with_openai
from classifier_agents.openai_doc_classifier import OPENAI_FRIENDLY_NAME, convert_to_png_if_needed as convert_image_for_processing
from classifier_agents.gemini_doc_classifier import classify_document_with_gemini
from classifier_agents.gemini_doc_classifier import GEMINI_FRIENDLY_NAME
from classifier_agents.text_doc_classifier import extract_text_from_docx, extract_text_from_document, classify_text_document_type

# Verification Agents (TEMPORARILY DISABLED FOR TESTING)
# from verification_agents.passport_agent import analyze_passport_image
# from verification_agents.cv_agent import analyze_cv_image, analyze_cv_from_text
# from verification_agents.driving_licence_agent import analyze_driving_licence_image
# from verification_agents.bank_statement_agent import analyze_bank_statement_image, analyze_bank_statement_pdf

app = FastAPI()

# Define a list of possible document types your agent can classify
POSSIBLE_DOC_TYPES = ["Bank Statement", "Utility Bill", "Council Tax Bill", "CV", "P45", "P60", "NI Letter", "Pay Slip", "Passport", "Right To Work Share Code", "DBS Certificate", "Police Check Certificate", "Drivers Licence", "Training Certificate", "Birth Certificate", "HMRC Letter", "DWP Letter", "Other", "Letter", "Report", "Invoice"]

# Create a temporary directory for uploads if it doesn't exist
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed MIME types for direct image/PDF processing + text document specific handling
ALLOWED_IMAGE_PDF_MIME_TYPES = [
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "image/heic", "image/heif"
]
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TXT_MIME_TYPE = "text/plain"

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
async def classify_document_endpoint(file: UploadFile = File(...), 
                                   detail: str = Form("auto"), 
                                   ai_provider: str = Form("openai")):
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    classification_result = None
    extracted_text_content = None # For text files (.docx, .txt)
    path_for_further_processing = None # For image/PDF files
    is_text_file = False
    
    # Determine file type for routing
    file_mime_type = file.content_type
    file_ext = os.path.splitext(file.filename)[1].lower()

    print("=" * 80)
    print(f"📁 UPLOAD: {file.filename} | {file_mime_type} | Provider: {ai_provider}")
    print("-" * 80)

    if file_ext in ['.docx', '.txt'] or file_mime_type in [DOCX_MIME_TYPE, TXT_MIME_TYPE]:
        is_text_file = True
        if file_ext == '.docx' and file_mime_type != DOCX_MIME_TYPE:
             print(f"[SERVER WARNING] File '{file.filename}' has .docx extension but unexpected MIME: '{file_mime_type}'. Proceeding as DOCX.")
        elif file_ext == '.txt' and file_mime_type != TXT_MIME_TYPE:
             print(f"[SERVER WARNING] File '{file.filename}' has .txt extension but unexpected MIME: '{file_mime_type}'. Proceeding as TXT.")
    elif file_mime_type not in ALLOWED_IMAGE_PDF_MIME_TYPES:
        # Fallback for generic MIME types if extension matches allowed image/pdf types
        if not ((file_mime_type == "application/octet-stream" or not file_mime_type) and 
                  file_ext in [".pdf", ".heic", ".heif", ".png", ".jpg", ".jpeg", ".gif", ".webp"]):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: '{file_mime_type if file_mime_type else file_ext}'. Upload image, PDF, DOCX, or TXT."
            )
        if not file_mime_type: # If MIME was empty but extension was allowed, try to guess it
            guessed_mime, _ = mimetypes.guess_type(file.filename)
            file_mime_type = guessed_mime if guessed_mime else "application/octet-stream"
            print(f"[SERVER LOG] Guessed MIME for '{file.filename}' as '{file_mime_type}'")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate detail parameter (though the agent also does this)
        if detail not in ["auto", "low", "high"]:
            raise HTTPException(status_code=400, detail="Invalid detail level. Must be 'auto', 'low', or 'high'.")

        # --- Initial Classification Step ---
        if is_text_file:
            print(f"📄 TEXT PROCESSING: {file.filename}")
            extracted_text_content = extract_text_from_document(temp_file_path, file_ext)
            if not extracted_text_content:
                raise HTTPException(status_code=500, detail=f"Failed to extract text from {file_ext.upper()} file.")
            
            text_classification_data = classify_text_document_type(extracted_text_content, POSSIBLE_DOC_TYPES)
            if not text_classification_data or not text_classification_data.get("predicted_document_type"):
                raise HTTPException(status_code=500, detail="Text document classification failed.")
            classification_result = text_classification_data["predicted_document_type"]
            # extracted_text_content is already set from above
            print(f"✅ CLASSIFIED: {file.filename} → {classification_result} (Text Classifier)")
        else:
            print(f"🖼️ IMAGE/PDF PROCESSING: {file.filename}")
            
            # Conditional conversion logic
            if ai_provider.lower() == "openai":
                # OpenAI: Only convert HEIC/HEIF to PNG, PDFs and other images are supported natively
                if file_mime_type in ["image/heic", "image/heif"]:
                    print(f"🔄 CONVERTING: {file_mime_type} → PNG (OpenAI requirement)")
                    converted_path, final_mime = convert_image_for_processing(temp_file_path, file_mime_type)
                    if not converted_path:
                        raise HTTPException(status_code=500, detail="HEIC/HEIF conversion failed for OpenAI processing.")
                    path_for_further_processing = converted_path
                else:
                    # PDFs, PNGs, JPGs, etc. - use natively
                    print(f"🚀 NATIVE SUPPORT: {file_mime_type} (OpenAI)")
                    path_for_further_processing = temp_file_path
                    final_mime = file_mime_type
            else:  # Gemini
                # Gemini: Supports PDF, HEIC, HEIF natively - no conversion needed
                print(f"🚀 NATIVE SUPPORT: {file_mime_type} (Gemini)")
                path_for_further_processing = temp_file_path
                final_mime = file_mime_type
            
            if ai_provider.lower() == "openai":
                classification_result = classify_with_openai(path_for_further_processing, final_mime, POSSIBLE_DOC_TYPES, detail)
                print(f"✅ CLASSIFIED: {file.filename} → {classification_result} (OpenAI)")
            elif ai_provider.lower() == "gemini":
                # Pass the original MIME type if it's a PDF for Gemini, else the final_mime (which would be image/png if converted)
                mime_for_gemini_classifier = file_mime_type if file_mime_type == "application/pdf" else final_mime
                classification_result = classify_document_with_gemini(path_for_further_processing, mime_for_gemini_classifier, POSSIBLE_DOC_TYPES)
                print(f"✅ CLASSIFIED: {file.filename} → {classification_result} (Gemini)")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid AI provider for image/pdf: '{ai_provider}'.")

        if not classification_result:
            raise HTTPException(status_code=500, detail="Document classification failed to return a type.")

        # --- Prepare base response --- 
        final_response_content = {
            "filename": file.filename, 
            "document_type": classification_result, 
            "detail_used": detail if not is_text_file and ai_provider.lower() == 'openai' else 'N/A', 
            "ai_provider": ai_provider if not is_text_file else 'text_classifier', # Indicate text classifier for text files
            "passport_analysis": None,
            "cv_analysis": None,
            "driving_licence_analysis": None, # Add new key for driving licence
            "bank_statement_analysis": None   # Add new key for bank statement
        }

        # --- Specialized Agent Processing (TEMPORARILY DISABLED FOR TESTING) ---
        print(f"⏭️ AGENTS SKIPPED: Testing classification only")
        # TODO: Re-enable agent processing after classification testing is complete

        print("-" * 80)
        print(f"✅ PROCESSING COMPLETE: {file.filename} → {classification_result}")
        print("=" * 80)

        return JSONResponse(content=final_response_content)

    except HTTPException as e:
        print("-" * 80)
        print(f"❌ PROCESSING FAILED: {file.filename if 'file' in locals() else 'Unknown'} → HTTP {e.status_code}")
        print("=" * 80)
        raise e
    except Exception as e:
        print(f"[SERVER ERROR] Unhandled error in classification pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("-" * 80)
        print(f"❌ PROCESSING FAILED: {file.filename if 'file' in locals() else 'Unknown'} → Server Error")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Cleanup for converted image file (path_for_further_processing)
        if path_for_further_processing and path_for_further_processing != temp_file_path and os.path.exists(path_for_further_processing):
             # Ensure we only delete from the OpenAI classifier's specific temp dir if that's how it's structured
             # Or check if it's in a server-managed CONVERTED_TEMP_DIR (if we add one)
            if convert_image_for_processing.__module__ == 'backend.openai_doc_classifier': # Check where convert_image_for_processing comes from
                # Assuming convert_image_for_processing saves to its own CONVERTED_TEMP_DIR (from openai_doc_classifier.py)
                # and that openai_doc_classifier.py handles its own cleanup or server is aware of that dir.
                # For now, let's assume the current structure: openai_doc_classifier.py creates and cleans up its own temp converted files if it made them.
                # The path_for_further_processing IS that temp file if conversion happened.
                # The `classify_with_openai` (which uses convert_image_for_processing) in openai_doc_classifier.py has a finally block for cleanup.
                # So, we might be double-deleting or relying on openai_doc_classifier to clean. 
                # Let's simplify: server should clean up what it creates or gets back from a direct conversion call.
                # The `convert_image_for_processing` (alias for convert_to_png_if_needed from openai_doc_classifier)
                # creates a new file if conversion happens. Server should delete it.
                try:
                    print(f"[SERVER LOG] Attempting to clean up processed file: {path_for_further_processing}")
                    os.remove(path_for_further_processing)
                    print(f"[SERVER LOG] Cleaned up processed file: {path_for_further_processing}")
                except Exception as cleanup_e:
                    print(f"[SERVER LOG] Error cleaning up processed file {path_for_further_processing}: {cleanup_e}")

# Make sure to load dotenv for environment variables
from dotenv import load_dotenv 
# load_dotenv() # Already called at the top

# Add main block for running with uvicorn if needed, but usually run from CLI
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) 