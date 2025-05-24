import os # Ensure os is imported if not already
from dotenv import load_dotenv # Removed find_dotenv for now, will use explicit path

# Construct the path to the .env file relative to this script (server.py)
# __file__ is the path to the current script (server.py)
# os.path.dirname(__file__) is the directory of the current script (backend/)
# os.path.join(os.path.dirname(__file__), '.env') is backend/.env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Debug: Print the API key after attempting to load .env
print(f"[SERVER DOTENV DEBUG] Attempted to load .env from: {dotenv_path}")
print(f"[SERVER DOTENV DEBUG] OPENAI_API_KEY found in environment: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
# If found, you might want to print a masked version for security in real logs, e.g., os.getenv('OPENAI_API_KEY')[:5] + '...'
# For this debug, knowing 'Yes' or 'No' is key.

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import shutil
import os
import json
import mimetypes # For guessing MIME types
# OpenAI Classifier
from .image_doc_classifier import classify_image_document_type as classify_with_openai
from .image_doc_classifier import OPENAI_FRIENDLY_NAME, convert_to_png_if_needed as convert_image_for_processing
# Gemini Classifier
from .gemini_doc_classifier import classify_document_with_gemini
from .gemini_doc_classifier import GEMINI_FRIENDLY_NAME
from .passport_agent import analyze_passport_image # Import the new passport agent
from .cv_agent import analyze_cv_image, analyze_cv_from_text # Import the new CV agent
from .driving_licence_agent import analyze_driving_licence_image # Import the new Driving Licence agent
from .bank_statement_agent import analyze_bank_statement_image, analyze_bank_statement_pdf # Import both bank statement functions

# Text Document Classifier
from .text_doc_classifier import extract_text_from_docx, classify_text_document_type

app = FastAPI()

# Define a list of possible document types your agent can classify
POSSIBLE_DOC_TYPES = ["Bank Statement", "Utility Bill", "Council Tax Bill", "CV", "P45", "P60", "NI Letter", "Pay Slip", "Passport", "Right To Work Share Code", "DBS Certificate", "Police Check Certificate", "Drivers Licence", "Training Certificate", "Birth Certificate", "HMRC Letter", "DWP Letter", "Other", "Letter", "Report", "Invoice"]

# Create a temporary directory for uploads if it doesn't exist
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed MIME types for direct image/PDF processing + .docx specific handling
ALLOWED_IMAGE_PDF_MIME_TYPES = [
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "image/heic", "image/heif"
]
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

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
    extracted_text_content = None # For .docx files
    path_for_further_processing = None # For image/PDF files
    is_docx_file = False
    
    # Determine file type for routing
    file_mime_type = file.content_type
    file_ext = os.path.splitext(file.filename)[1].lower()

    print(f"[SERVER LOG] Received file: '{file.filename}', MIME: '{file_mime_type}', Extension: '{file_ext}', Provider: {ai_provider}")

    if file_ext == '.docx' or file_mime_type == DOCX_MIME_TYPE:
        is_docx_file = True
        if file_mime_type != DOCX_MIME_TYPE:
             print(f"[SERVER WARNING] File '{file.filename}' has .docx extension but unexpected MIME: '{file_mime_type}'. Proceeding as DOCX.")
    elif file_mime_type not in ALLOWED_IMAGE_PDF_MIME_TYPES:
        # Fallback for generic MIME types if extension matches allowed image/pdf types
        if not ((file_mime_type == "application/octet-stream" or not file_mime_type) and 
                  file_ext in [".pdf", ".heic", ".heif", ".png", ".jpg", ".jpeg", ".gif", ".webp"]):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: '{file_mime_type if file_mime_type else file_ext}'. Upload image, PDF, or DOCX."
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
        if is_docx_file:
            print(f"[SERVER LOG] Processing DOCX file: {file.filename}")
            extracted_text_content = extract_text_from_docx(temp_file_path)
            if not extracted_text_content:
                raise HTTPException(status_code=500, detail="Failed to extract text from DOCX file.")
            
            text_classification_data = classify_text_document_type(extracted_text_content, POSSIBLE_DOC_TYPES)
            if not text_classification_data or not text_classification_data.get("predicted_document_type"):
                raise HTTPException(status_code=500, detail="Text document classification failed.")
            classification_result = text_classification_data["predicted_document_type"]
            # extracted_text_content is already set from above
            print(f"[CLASSIFICATION LOG - Text] Classified DOCX '{file.filename}' as: '{classification_result}'")
        else:
            print(f"[SERVER LOG] Processing Image/PDF file: {file.filename}")
            # Image/PDF processing path (existing logic)
            
            # Conditional conversion for OpenAI or if Gemini gets an image that needs it
            if ai_provider.lower() == "openai" or (ai_provider.lower() == "gemini" and not file_mime_type == "application/pdf"):
                print(f"[SERVER LOG] Converting file {file.filename} to PNG for processing.")
                converted_path, final_mime = convert_image_for_processing(temp_file_path, file_mime_type)
                if not converted_path:
                    raise HTTPException(status_code=500, detail="Image/PDF conversion failed for OpenAI/Gemini-Image path.")
                path_for_further_processing = converted_path
            else: # For Gemini with PDF, use the original path
                print(f"[SERVER LOG] Using original PDF {file.filename} for Gemini classification.")
                path_for_further_processing = temp_file_path
                final_mime = file_mime_type # Original MIME type is PDF
            
            if ai_provider.lower() == "openai":
                classification_result = classify_with_openai(path_for_further_processing, final_mime, POSSIBLE_DOC_TYPES, detail)
                print(f"[CLASSIFICATION LOG - OpenAI Image] Classified '{file.filename}' as: '{classification_result}'")
            elif ai_provider.lower() == "gemini":
                # Pass the original MIME type if it's a PDF for Gemini, else the final_mime (which would be image/png if converted)
                mime_for_gemini_classifier = file_mime_type if file_mime_type == "application/pdf" else final_mime
                classification_result = classify_document_with_gemini(path_for_further_processing, mime_for_gemini_classifier, POSSIBLE_DOC_TYPES)
                print(f"[CLASSIFICATION LOG - Gemini Document] Classified '{file.filename}' as: '{classification_result}'")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid AI provider for image/pdf: '{ai_provider}'.")

        if not classification_result:
            raise HTTPException(status_code=500, detail="Document classification failed to return a type.")

        # --- Prepare base response --- 
        final_response_content = {
            "filename": file.filename, 
            "document_type": classification_result, 
            "detail_used": detail if not is_docx_file and ai_provider.lower() == 'openai' else 'N/A', 
            "ai_provider": ai_provider if not is_docx_file else 'text_classifier', # Indicate text classifier for DOCX
            "passport_analysis": None,
            "cv_analysis": None,
            "driving_licence_analysis": None, # Add new key for driving licence
            "bank_statement_analysis": None   # Add new key for bank statement
        }

        # --- Specialized Agent Processing ---
        if classification_result == "Passport" and not is_docx_file and path_for_further_processing:
            # Passport analysis is only for images/PDFs
            passport_data = analyze_passport_image(path_for_further_processing)
            final_response_content["passport_analysis"] = passport_data if passport_data else {"error": "Passport analysis failed or no data."}
        
        elif classification_result == "CV":
            if is_docx_file and extracted_text_content:
                print(f"[SERVER LOG] CV (from DOCX) analysis using text: {file.filename}")
                cv_data = analyze_cv_from_text(extracted_text_content)
            elif not is_docx_file and path_for_further_processing:
                print(f"[SERVER LOG] CV (from Image/PDF) analysis using image: {file.filename}")
                cv_data = analyze_cv_image(path_for_further_processing)
            else:
                cv_data = {"error": "Could not determine input type for CV analysis."}
            final_response_content["cv_analysis"] = cv_data if cv_data else {"error": "CV analysis failed or no data."}
        
        elif classification_result == "Drivers Licence" and not is_docx_file and path_for_further_processing:
            # Driving Licence analysis is only for images/PDFs
            print(f"[SERVER LOG] Driving Licence analysis using image: {file.filename}")
            dl_data = analyze_driving_licence_image(path_for_further_processing)
            final_response_content["driving_licence_analysis"] = dl_data if dl_data else {"error": "Driving licence analysis failed or no data."}

        elif classification_result == "Bank Statement":
            if file_mime_type == "application/pdf" and not is_docx_file: # Check if original upload was PDF
                print(f"[SERVER LOG] Bank Statement (from PDF) analysis using direct PDF: {file.filename}")
                # Use the original temp_file_path for the PDF, not the potentially converted one
                bs_data = analyze_bank_statement_pdf(temp_file_path) 
            elif not is_docx_file and path_for_further_processing: # Fallback to image analysis if not PDF or if it was an image initially
                print(f"[SERVER LOG] Bank Statement (from Image) analysis using image: {file.filename}")
                # Ensure path_for_further_processing is the converted PNG if the original was PDF but processed as image for classification
                # This case should be less common now for bank statements if Gemini is chosen for classification & PDF is passed directly.
                # If classification was OpenAI, path_for_further_processing is already the PNG.
                # If classification was Gemini (PDF direct) but somehow this path is hit for bank statement *image* analysis, 
                # it means the initial file wasn't PDF or we didn't take the direct PDF path for bank statement analysis.
                # The path_for_further_processing should be correct from the classification stage.
                bs_data = analyze_bank_statement_image(path_for_further_processing)
            else:
                bs_data = {"error": "Could not determine input type for Bank Statement analysis or it was a DOCX classified as Bank Statement which is not directly supported by this agent."}
            final_response_content["bank_statement_analysis"] = bs_data if bs_data else {"error": "Bank statement analysis failed or no data."}

        return JSONResponse(content=final_response_content)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[SERVER ERROR] Unhandled error in classification pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Cleanup for converted image file (path_for_further_processing)
        if path_for_further_processing and path_for_further_processing != temp_file_path and os.path.exists(path_for_further_processing):
             # Ensure we only delete from the image classifier's specific temp dir if that's how it's structured
             # Or check if it's in a server-managed CONVERTED_TEMP_DIR (if we add one)
            if convert_image_for_processing.__module__ == 'backend.image_doc_classifier': # Check where convert_image_for_processing comes from
                # Assuming convert_image_for_processing saves to its own CONVERTED_TEMP_DIR (from image_doc_classifier.py)
                # and that image_doc_classifier.py handles its own cleanup or server is aware of that dir.
                # For now, let's assume the current structure: image_doc_classifier.py creates and cleans up its own temp converted files if it made them.
                # The path_for_further_processing IS that temp file if conversion happened.
                # The `classify_with_openai` (which uses convert_image_for_processing) in image_doc_classifier.py has a finally block for cleanup.
                # So, we might be double-deleting or relying on image_doc_classifier to clean. 
                # Let's simplify: server should clean up what it creates or gets back from a direct conversion call.
                # The `convert_image_for_processing` (alias for convert_to_png_if_needed from image_doc_classifier)
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