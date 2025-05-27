from openai import OpenAI
import base64
import os
import mimetypes
import json
from PIL import Image # For image operations
from pdf2image import convert_from_path # For PDF conversion
from pillow_heif import register_heif_opener # For HEIC support
import io # For handling byte streams

register_heif_opener() # Register HEIC opener with Pillow

# Ensure your OPENAI_API_KEY environment variable is set.
# You can set it using: export OPENAI_API_KEY='your_api_key'
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# If the API key is set globally in the environment, this simpler initialization works:
client = OpenAI()

# Define a directory for temporary converted files
CONVERTED_TEMP_DIR = "temp_converted_images"
os.makedirs(CONVERTED_TEMP_DIR, exist_ok=True)

# --- Model Configuration ---
OPENAI_MODEL_ID = 'gpt-4.1' # The actual model ID used for the API call (Responses API compatible)
OPENAI_FRIENDLY_NAME = f"OpenAI ({OPENAI_MODEL_ID})" # Display name for the UI
# --- End Model Configuration ---

def convert_to_png_if_needed(original_path: str, original_mime_type: str | None) -> tuple[str | None, str | None]:
    """
    Converts PDF or HEIC files to PNG format. 
    If conversion occurs, returns the path to the new PNG file and its MIME type.
    Otherwise, returns the original path and MIME type.
    Manages a new temporary file for the converted image.
    """
    filename, extension = os.path.splitext(os.path.basename(original_path))
    converted_image_path = None
    new_mime_type = original_mime_type

    # Poppler path - set to None to use system PATH or install via package manager
    poppler_bin_path = None  # Let pdf2image find poppler in system PATH

    try:
        if original_mime_type == "application/pdf" or extension.lower() == ".pdf":
            print(f"Converting PDF: {original_path} using Poppler from: {'system PATH' if poppler_bin_path is None else poppler_bin_path}")
            if poppler_bin_path:
                images = convert_from_path(original_path, first_page=1, last_page=1, fmt='png', poppler_path=poppler_bin_path)
            else:
                images = convert_from_path(original_path, first_page=1, last_page=1, fmt='png')
            if images:
                converted_image_path = os.path.join(CONVERTED_TEMP_DIR, f"{filename}_converted.png")
                images[0].save(converted_image_path, "PNG")
                new_mime_type = "image/png"
                print(f"PDF converted to: {converted_image_path}")
                return converted_image_path, new_mime_type # Return path of NEWLY created temp file
            else:
                print(f"Error: Could not convert PDF {original_path} to image.")
                return None, None

        elif original_mime_type == "image/heic" or original_mime_type == "image/heif" or \
             extension.lower() in [".heic", ".heif"]:
            print(f"Converting HEIC/HEIF: {original_path}")
            img = Image.open(original_path)
            converted_image_path = os.path.join(CONVERTED_TEMP_DIR, f"{filename}_converted.png")
            img.save(converted_image_path, "PNG")
            new_mime_type = "image/png"
            print(f"HEIC converted to: {converted_image_path}")
            return converted_image_path, new_mime_type # Return path of NEWLY created temp file

    except Exception as e:
        print(f"Error during conversion of {original_path}: {e}")
        # If conversion created a temp file but failed before returning, clean it up.
        if converted_image_path and os.path.exists(converted_image_path):
            try:
                os.remove(converted_image_path)
            except Exception as cleanup_e:
                print(f"Error cleaning up partially converted file {converted_image_path}: {cleanup_e}")
        return None, None # Indicate conversion failure

    # If no conversion was needed or done
    return original_path, original_mime_type

def encode_image_to_base64(image_path: str, mime_type: str | None) -> str | None:
    """
    Encodes an image file to a Base64 string using the provided MIME type.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path} for encoding.")
        return None
    
    if not mime_type or not mime_type.startswith("image"):
        print(f"Error: Invalid or missing MIME type ('{mime_type}') for encoding {image_path}. Defaulting to image/png.")
        # Forcing PNG as it's our target conversion format and widely supported.
        mime_type = "image/png" 

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def classify_image_document_type(image_path: str, original_file_mime_type: str | None, document_types: list[str], detail: str = "auto") -> str | None:
    """
    Classifies the type of document shown in an image or PDF (supports native PDF input for OpenAI).
    """
    if not image_path:
        print("Error: Image path cannot be empty.")
        return None
    if not document_types:
        print("Error: Document types list cannot be empty.")
        return None

    path_to_process = image_path
    current_mime_type = original_file_mime_type
    converted_temp_file_to_delete = None

    # Handle different file types
    if original_file_mime_type == "application/pdf":
        # For PDFs, use native PDF support in OpenAI Responses API
        print(f"📄 OPENAI: Native PDF processing")
        path_to_process = image_path
        current_mime_type = "application/pdf"
    elif original_file_mime_type in ["image/heic", "image/heif"] or \
         image_path.lower().endswith((".heic", ".heif")):
        # Convert HEIC/HEIF to PNG (still needed as OpenAI doesn't support these)
        print(f"🔄 OPENAI: Converting {original_file_mime_type} → PNG")
        converted_path, new_mime = convert_to_png_if_needed(image_path, original_file_mime_type)
        if converted_path and converted_path != image_path:
            path_to_process = converted_path
            current_mime_type = new_mime
            converted_temp_file_to_delete = converted_path
        elif not converted_path:
            print(f"❌ OPENAI: Conversion failed for {image_path}")
            return None
    else:
        print(f"🖼️ OPENAI: Native image processing ({original_file_mime_type})")
    # For regular images (PNG, JPG, etc.), use as-is
    
    valid_details = ["auto", "low", "high"]
    if detail not in valid_details:
        print(f"Warning: Invalid detail level '{detail}'. Defaulting to 'auto'. Valid options are: {valid_details}")
        detail = "auto"
    
    # Define the JSON schema for structured outputs
    output_schema = {
        "type": "object",
        "properties": {
            "predicted_document_type": {
                "type": "string",
                "enum": document_types,
                "description": "The classified type of the document from the provided list."
            }
        },
        "required": ["predicted_document_type"],
        "additionalProperties": False
    }

    prompt_text = (
        f"Analyze the document and classify its type. "
        f"The 'predicted_document_type' field in your JSON response must be one of these exact values: {', '.join(document_types)}."
    )

    api_response_text = None
    try:
        # Prepare input based on file type
        if current_mime_type == "application/pdf":
            # Use native PDF support with Base64 encoding
            try:
                with open(path_to_process, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                import base64
                base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
                
                # Use the new Responses API with native PDF support
                response = client.responses.create(
                    model=OPENAI_MODEL_ID,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text", 
                                    "text": prompt_text
                                },
                                {
                                    "type": "input_file",
                                    "filename": os.path.basename(path_to_process),
                                    "file_data": f"data:application/pdf;base64,{base64_pdf}"
                                }
                            ]
                        }
                    ],
                    temperature=0.2,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "document_classification",
                            "schema": output_schema,
                            "strict": True
                        }
                    }
                )
                print(f"🤖 OPENAI: API call sent ({OPENAI_MODEL_ID})")
            except Exception as e:
                print(f"[OpenAI ERROR] Failed to process PDF natively: {e}")
                return None
        else:
            # Use image processing for non-PDF files
            base64_image = encode_image_to_base64(path_to_process, current_mime_type)
            if not base64_image:
                if converted_temp_file_to_delete and os.path.exists(converted_temp_file_to_delete):
                    try: 
                        os.remove(converted_temp_file_to_delete)
                    except Exception:
                        pass
                return None

            # Use the new Responses API with image input
            response = client.responses.create(
                model=OPENAI_MODEL_ID,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text", 
                                "text": prompt_text
                            },
                            {
                                "type": "input_image",
                                "image_url": base64_image,
                                "detail": detail
                            }
                        ]
                    }
                ],
                temperature=0.2,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "document_classification",
                        "schema": output_schema,
                        "strict": True
                    }
                }
            )
            print(f"🤖 OPENAI: API call sent ({OPENAI_MODEL_ID})")
        
        # Handle the response (same for both PDF and image)
        if response.status == "completed" and response.output:
            api_response_text = response.output_text
            print(f"📝 OPENAI: Raw response: {api_response_text}")
            
            try:
                parsed_json = json.loads(api_response_text)
                predicted_type = parsed_json.get("predicted_document_type")

                if predicted_type and predicted_type in document_types:
                    print(f"✅ OPENAI: Classification successful → {predicted_type}")
                    return predicted_type
                elif predicted_type:
                    print(f"⚠️ OPENAI: Invalid type '{predicted_type}' not in allowed list")
                    return None 
                else:
                    print(f"❌ OPENAI: Missing 'predicted_document_type' in response")
                    return None
            except json.JSONDecodeError as e:
                print(f"❌ OPENAI: JSON decode error: {e}")
                return None
        else:
            # Handle incomplete responses or refusals
            if response.status == "incomplete":
                print(f"[OpenAI LOG] Incomplete response. Reason: {response.incomplete_details.reason if response.incomplete_details else 'Unknown'}")
            
            # Check for refusals in the output
            if response.output:
                for output_item in response.output:
                    if output_item.type == "message" and output_item.content:
                        for content_item in output_item.content:
                            if content_item.type == "refusal":
                                print(f"[OpenAI LOG] Request was refused: {content_item.refusal}")
                                return None
            
            print(f"[OpenAI LOG] No valid content in API response. Status: {response.status}")
            return None

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API or processing its response: {e}. API response text was: {api_response_text if api_response_text else 'N/A'}")
        return None
    finally:
        # Clean up the temporary *converted* file if one was created (only for HEIC/HEIF)
        if converted_temp_file_to_delete and os.path.exists(converted_temp_file_to_delete):
            try:
                os.remove(converted_temp_file_to_delete)
                print(f"Cleaned up temporary converted file: {converted_temp_file_to_delete}")
            except Exception as e:
                print(f"Error cleaning up temporary converted file {converted_temp_file_to_delete}: {e}")

if __name__ == "__main__":
    # Test with a placeholder for an actual image, PDF, or HEIC file
    # Replace with valid paths to test different types
    sample_files_to_test = {
        # "image": "path/to/your/sample_document_image.jpg", 
        # "pdf": "path/to/your/sample_document.pdf",
        # "heic": "path/to/your/sample_document.heic",
    }
    detail_level = "auto"
    possible_doc_types = ["Invoice", "Receipt", "Letter", "Form", "ID Card", "Passport", "Driver License", "Report", "Unknown"]

    for file_type, file_path_placeholder in sample_files_to_test.items():
        if not os.path.exists(file_path_placeholder):
            print(f"Skipping '{file_type}' example: Sample file not found at '{file_path_placeholder}'. Update path.")
            continue
        
        # Mimic getting MIME type as server would
        mime_type, _ = mimetypes.guess_type(file_path_placeholder)
        if file_path_placeholder.lower().endswith(".pdf") and not mime_type: # mimetypes might not always guess PDF
            mime_type = "application/pdf"
        elif file_path_placeholder.lower().endswith( (".heic", ".heif")) and not mime_type:
            mime_type = "image/heic" # or image/heif

        print(f"\nClassifying '{file_type}' document: '{file_path_placeholder}' (MIME: {mime_type}) with detail: '{detail_level}'")
        classification = classify_image_document_type(file_path_placeholder, mime_type, possible_doc_types, detail_level)
        print(f"Predicted Document Type for {file_type}: {classification}")

    # Example for a file type that doesn't need conversion (e.g., PNG)
    # Ensure you have a PNG file at this path for this test to run.
    png_test_path = "path/to/your/sample_image.png"
    if os.path.exists(png_test_path):
        print(f"\nClassifying PNG document: '{png_test_path}' (MIME: image/png) with detail: '{detail_level}'")
        classification_png = classify_image_document_type(png_test_path, "image/png", possible_doc_types, detail_level)
        print(f"Predicted Document Type for PNG: {classification_png}")
    else:
        print(f"\nSkipping PNG test: Sample file not found at '{png_test_path}'. Update path.")

    print("\nTesting with non-existent image path:")
    classification_non_existent = classify_image_document_type("path/to/non_existent_image.png", "image/png", possible_doc_types, "high")
    print(f"Predicted Document Type (non-existent image): {classification_non_existent}") 