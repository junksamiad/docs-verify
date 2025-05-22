from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from .image_doc_classifier import classify_image_document_type # Assuming image_doc_classifier is in the same directory

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
async def classify_image_endpoint(file: UploadFile = File(...), detail: str = Form("auto")):
    """
    Endpoint to classify an uploaded image.
    Accepts an image file and an optional 'detail' level (auto, low, high).
    """
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    # --- START: File Type Validation ---
    allowed_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_mime_types:
        # You could also check file.filename extension as a fallback or primary method
        # e.g., if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload a valid image (JPEG, PNG, GIF, WEBP)."
        )
    # --- END: File Type Validation ---

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate detail parameter (though the agent also does this)
        if detail not in ["auto", "low", "high"]:
            raise HTTPException(status_code=400, detail="Invalid detail level. Must be 'auto', 'low', or 'high'.")

        # Call the classification agent
        classification_result = classify_image_document_type(
            image_path=temp_file_path,
            document_types=POSSIBLE_DOC_TYPES,
            detail=detail
        )

        # ---- START: Added print statement for logging ----
        print(f"[CLASSIFICATION LOG] Agent returned: '{classification_result}' for file: '{file.filename}' with detail: '{detail}'")
        # ---- END: Added print statement for logging ----

        if classification_result:
            return JSONResponse(content={"filename": file.filename, "document_type": classification_result, "detail_used": detail})
        else:
            # If the agent returns None, it might be due to an internal error or an unexpected model response
            raise HTTPException(status_code=500, detail="Could not classify the image. The agent returned no result.")

    except HTTPException as e:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch any other exceptions and return a generic server error
        # Log the error for debugging: print(f"Error during image classification: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) 