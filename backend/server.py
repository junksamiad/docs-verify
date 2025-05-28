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

# Verification Agents
from verification_agents.passport_agent import analyze_passport_document
from verification_agents.cv_agent import analyze_cv_image, analyze_cv_from_text, analyze_cv_pdf
from verification_agents.driving_licence_agent import analyze_driving_licence_document
from verification_agents.bank_statement_agent import analyze_bank_statement_image, analyze_bank_statement_pdf

app = FastAPI()

# Define a list of possible document types your agent can classify
POSSIBLE_DOC_TYPES = [
    "Bank Statement", "Credit Card Statement", "Mortgage Statement", "Loan Statement", 
    "Investment Statement", "Utility Bill", "Council Tax Bill", "CV", "P45", "P60", 
    "NI Letter", "Pay Slip", "Passport", "Right To Work Share Code", "DBS Certificate", 
    "Police Check Certificate", "Drivers Licence", "Training Certificate", "Birth Certificate", 
    "HMRC Letter", "DWP Letter", "Receipt", "Invoice", "Contract", "Insurance Document", 
    "Medical Document", "Other", "Letter", "Report"
]

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

        # --- Specialized Agent Processing ---
        print("🔍 AGENT ROUTING: Checking for specialized analysis...")
        
        if classification_result == "Passport":
            print("🛂 PASSPORT AGENT: Starting passport analysis...")
            
            # Passport agent now supports all formats natively with Gemini 2.5
            if is_text_file:
                print("⚠️ PASSPORT AGENT: Text files not supported - requires image/PDF format")
                final_response_content["passport_analysis"] = {
                    "error": "Passport analysis requires image or PDF format",
                    "details": "Text files (.docx, .txt) cannot be processed by the passport agent"
                }
            else:
                # All image formats and PDF supported natively
                print(f"🖼️ PASSPORT AGENT: Processing {file_mime_type} → {path_for_further_processing}")
                passport_analysis = analyze_passport_document(path_for_further_processing, file_mime_type)
                
                if passport_analysis:
                    if "error" in passport_analysis:
                        print(f"❌ PASSPORT AGENT: Analysis failed → {passport_analysis.get('error', 'Unknown error')}")
                    else:
                        # Log extracted passport details
                        extracted_info = passport_analysis.get('extracted_information', {})
                        manual_flags = passport_analysis.get('manual_verification_flags', [])
                        image_quality = passport_analysis.get('image_quality_summary', 'N/A')
                        
                        surname = extracted_info.get('surname', 'N/A') if extracted_info else 'N/A'
                        given_names = extracted_info.get('given_names', 'N/A') if extracted_info else 'N/A'
                        passport_number = extracted_info.get('passport_number', 'N/A') if extracted_info else 'N/A'
                        nationality = extracted_info.get('nationality', 'N/A') if extracted_info else 'N/A'
                        
                        print(f"✅ PASSPORT AGENT: Extracted → {surname}, {given_names} | {passport_number} | {nationality}")
                        print(f"📸 PASSPORT AGENT: Document quality → {image_quality}")
                        print(f"⚠️ PASSPORT AGENT: Manual verification flags: {len(manual_flags)} → {manual_flags}")
                    
                    final_response_content["passport_analysis"] = passport_analysis
                else:
                    print("❌ PASSPORT AGENT: No response from analysis")
                    final_response_content["passport_analysis"] = {
                        "error": "Passport analysis returned no data"
                    }
        elif classification_result == "Bank Statement":
            print("🏦 BANK STATEMENT AGENT: Starting financial analysis...")
            
            # Determine which function to use based on file type
            if is_text_file:
                print("⚠️ BANK STATEMENT AGENT: Text files not supported - requires image/PDF")
                final_response_content["bank_statement_analysis"] = {
                    "error": "Bank statement analysis requires image or PDF format",
                    "details": "Text files (.docx, .txt) cannot be processed by the bank statement agent"
                }
            else:
                # Use appropriate function based on MIME type
                if file_mime_type == "application/pdf":
                    print(f"📄 BANK STATEMENT AGENT: Processing PDF → {path_for_further_processing}")
                    bank_analysis = analyze_bank_statement_pdf(path_for_further_processing)
                else:
                    print(f"🖼️ BANK STATEMENT AGENT: Processing image → {path_for_further_processing}")
                    bank_analysis = analyze_bank_statement_image(path_for_further_processing)
                
                if bank_analysis:
                    if "error" in bank_analysis:
                        print(f"❌ BANK STATEMENT AGENT: Analysis failed → {bank_analysis.get('error', 'Unknown error')}")
                    else:
                        # Count transactions for logging
                        transaction_count = len(bank_analysis.get('transactions', []))
                        total_in = bank_analysis.get('total_paid_in', 0)
                        total_out = bank_analysis.get('total_paid_out', 0)
                        print(f"✅ BANK STATEMENT AGENT: Analysis complete → {transaction_count} transactions, £{total_in:.2f} in, £{total_out:.2f} out")
                    
                    final_response_content["bank_statement_analysis"] = bank_analysis
                else:
                    print("❌ BANK STATEMENT AGENT: No response from analysis")
                    final_response_content["bank_statement_analysis"] = {
                        "error": "Bank statement analysis returned no data"
                    }
        elif classification_result == "CV":
            print("📋 CV AGENT: Starting CV analysis...")
            
            # Determine which function to use based on file type
            if is_text_file:
                print("📄 CV AGENT: Processing text-based CV...")
                print(f"📄 CV AGENT: Text length being analyzed: {len(extracted_text_content)} characters")
                
                # Debug: Show sample of text being passed to CV agent
                if len(extracted_text_content) > 500:
                    print(f"📄 CV AGENT: Text preview (first 200 chars): {extracted_text_content[:200]}")
                    print(f"📄 CV AGENT: Text preview (last 200 chars): ...{extracted_text_content[-200:]}")
                else:
                    print(f"📄 CV AGENT: Full text being analyzed: {extracted_text_content}")
                
                cv_result = analyze_cv_from_text(extracted_text_content)
            elif file_mime_type == "application/pdf":
                print("📋 CV AGENT: Processing PDF CV...")
                cv_result = analyze_cv_pdf(path_for_further_processing)
            else:
                # Image files (PNG, JPG, WEBP, GIF, HEIC, HEIF)
                print("🖼️ CV AGENT: Processing image CV...")
                cv_result = analyze_cv_image(path_for_further_processing)
            
            if cv_result:
                print("📋 CV AGENT: Analysis complete")
                # Add detailed logging for CV results
                if "error" in cv_result:
                    print(f"❌ CV AGENT: Analysis failed → {cv_result.get('error', 'Unknown error')}")
                else:
                    # Log extracted details
                    personal_details = cv_result.get('personal_details', {})
                    work_gaps = cv_result.get('work_experience_gaps', [])
                    other_flags = cv_result.get('other_verification_flags', [])
                    
                    name = personal_details.get('name', 'N/A') if personal_details else 'N/A'
                    phone = personal_details.get('phone_number', 'N/A') if personal_details else 'N/A'
                    email = personal_details.get('email_address', 'N/A') if personal_details else 'N/A'
                    
                    print(f"✅ CV AGENT: Extracted → Name: {name}, Phone: {phone}, Email: {email}")
                    print(f"⚠️ CV AGENT: Work gaps found: {len(work_gaps)} → {work_gaps}")
                    print(f"🔍 CV AGENT: Other flags: {len(other_flags)} → {other_flags}")
                
                final_response_content["cv_analysis"] = cv_result
            else:
                print("⚠️ CV AGENT: Analysis failed or returned no result")
                final_response_content["cv_analysis"] = {"error": "CV analysis failed to return results"}
        elif classification_result == "Drivers Licence":
            print("🚗 DRIVERS LICENCE AGENT: Starting drivers licence analysis...")
            
            # Drivers licence agent now supports all formats natively with Gemini 2.5
            if is_text_file:
                print("⚠️ DRIVERS LICENCE AGENT: Text files not supported - requires visual document")
                agent_result = {"error": "Drivers licence analysis requires visual document (image or PDF), not text file"}
            else:
                # Use unified function for all visual formats (images and PDFs)
                agent_result = analyze_driving_licence_document(temp_file_path, file_mime_type)
            
            if agent_result and "error" not in agent_result:
                print("✅ DRIVERS LICENCE AGENT: Analysis completed successfully")
                print(f"[DLAgent RESULT] Quality: {agent_result.get('image_quality_summary', 'N/A')}")
                print(f"[DLAgent RESULT] Manual flags: {agent_result.get('manual_verification_flags', [])}")
                print(f"[DLAgent RESULT] Licence details: {agent_result.get('licence_details', {})}")
                final_response_content["driving_licence_analysis"] = agent_result
            else:
                print(f"❌ DRIVERS LICENCE AGENT: Analysis failed - {agent_result.get('error', 'Unknown error')}")
                final_response_content["driving_licence_analysis"] = agent_result
        else:
            print(f"⏭️ AGENT ROUTING: No specialized agent for '{classification_result}' - classification only")

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