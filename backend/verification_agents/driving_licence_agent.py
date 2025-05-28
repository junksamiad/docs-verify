import os
import json
from typing import Optional

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Part, Blob

# --- Model Configuration ---
DRIVING_LICENCE_AGENT_MODEL_ID = 'gemini-2.5-pro-preview-05-06'  # Using Gemini 2.5 for native PDF/HEIC support
# --- End Model Configuration ---

def get_gemini_client_and_model() -> Optional[genai.Client]:
    """Initializes and returns the Gemini client if API key is set."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("[DLAgent ERROR] GOOGLE_API_KEY environment variable not set.")
        return None
    try:
        client = genai.Client(api_key=google_api_key)
        return client
    except Exception as e:
        print(f"[DLAgent ERROR] Failed to initialize Gemini client: {e}")
        return None

def analyze_driving_licence_document(file_path: str, file_mime_type: str) -> Optional[dict]:
    """
    Analyzes a driving licence document (image or PDF) for genuineness and extracts information.
    Supports all image formats (PNG, JPEG, HEIC, HEIF, etc.) and PDF natively.
    """
    print(f"[DLAgent LOG] Analyzing driving licence document: {file_path} ({file_mime_type})")
    
    client = get_gemini_client_and_model()
    if not client:
        return {"error": "Gemini client not initialized (GOOGLE_API_KEY missing or invalid)."}

    if not os.path.exists(file_path):
        print(f"[DLAgent ERROR] File not found at {file_path}")
        return {"error": f"File not found: {file_path}"}

    try:
        with open(file_path, "rb") as file:
            file_bytes = file.read()
        
        # Create file part using Gemini's native format support
        file_part = Part(
            inline_data=Blob(
                data=file_bytes,
                mime_type=file_mime_type
            )
        )
    except Exception as e:
        print(f"[DLAgent ERROR] Error reading file {file_path}: {e}")
        return {"error": f"Could not read file: {e}"}

    prompt_text = """You are an AI assistant specialized in analyzing driving licence documents (images or PDFs).
Your goal is to assess quality, identify potential issues for manual review, and extract key information.
Carefully examine the provided driving licence document.

Analyze the following:

1. **Document Quality Assessment**: Provide a brief summary of the document quality (e.g., clarity, lighting, readability of text, obstructions, glare, scan quality).
   Example: "Document is clear, well-lit, all text is legible" or "Document is slightly blurry with glare over the address field."

2. **Manual Verification Flags**: Identify and list any specific visual characteristics, anomalies, or inconsistencies that might suggest the document warrants manual human verification.
   Examples: "Font on date of birth appears different from other text", "Edges of the photograph look uneven or tampered with", "Hologram is not visible or looks incorrect".
   If no specific flags, use an empty array.

3. **Extract Driving Licence Information**: Extract all visible details including:
   - licence_number, surname, given_names, date_of_birth, address
   - date_of_issue, date_of_expiry, issuing_authority, country_of_issue
   - categories_entitlements (vehicle categories holder can drive, e.g., ["B", "C1"])
   - restrictions (any restrictions if present, e.g., "01 - eyesight correction")
   
   If certain information is not visible or legible, use null for that field.

Your response MUST be a single, valid JSON object with these exact top-level keys:
- "image_quality_summary": (string) your assessment of the document quality
- "manual_verification_flags": (array of strings) reasons for potential manual review, or empty array if none
- "licence_details": (object) containing all extracted driving licence fields

Do not include any text outside of this JSON object."""

    contents = [file_part, prompt_text]

    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        max_output_tokens=32000
    )

    api_response_text = None
    try:
        print(f"[DLAgent LOG] Sending request to Gemini model: {DRIVING_LICENCE_AGENT_MODEL_ID}")
        response = client.models.generate_content(
            model=DRIVING_LICENCE_AGENT_MODEL_ID,
            contents=contents,
            config=generation_config
        )
        
        if response.text:
            api_response_text = response.text
            print(f"[DLAgent LOG] Raw JSON response: {api_response_text}")
            try:
                parsed_json = json.loads(api_response_text)
                # Validate required keys
                required_keys = ["image_quality_summary", "manual_verification_flags", "licence_details"]
                if all(key in parsed_json for key in required_keys):
                    # Validate manual_verification_flags is a list
                    if not isinstance(parsed_json.get("manual_verification_flags"), list):
                        print("[DLAgent WARNING] 'manual_verification_flags' is not a list. Converting to list.")
                        flags = parsed_json.get("manual_verification_flags", [])
                        parsed_json["manual_verification_flags"] = [flags] if isinstance(flags, str) else []
                    # Validate licence_details is a dict
                    if not isinstance(parsed_json.get("licence_details"), dict):
                        print("[DLAgent WARNING] 'licence_details' is not an object/dictionary.")
                        parsed_json["licence_details"] = {}
                    return parsed_json
                else:
                    print("[DLAgent ERROR] Returned JSON is missing required keys.")
                    print(f"[DLAgent LOG] Expected: {required_keys}")
                    print(f"[DLAgent LOG] Received: {list(parsed_json.keys())}")
                    return {"error": "Driving licence analysis returned incomplete data structure.", "details": api_response_text}

            except json.JSONDecodeError as e:
                print(f"[DLAgent ERROR] JSON decode error: {e}. Raw response: {api_response_text}")
                return {"error": "Failed to decode driving licence analysis JSON.", "details": api_response_text}
        else:
            # Fallback to check candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                api_response_text = response.candidates[0].content.parts[0].text
                print(f"[DLAgent LOG] Raw JSON response from candidate: {api_response_text}")
                try:
                    parsed_json = json.loads(api_response_text)
                    required_keys = ["image_quality_summary", "manual_verification_flags", "licence_details"]
                    if all(key in parsed_json for key in required_keys):
                        return parsed_json
                    else:
                        return {"error": "Driving licence analysis returned incomplete data structure.", "details": api_response_text}
                except json.JSONDecodeError as e:
                    print(f"[DLAgent ERROR] JSON decode error (from candidate): {e}")
                    return {"error": "Failed to decode driving licence analysis JSON.", "details": api_response_text}
            
            print(f"[DLAgent ERROR] No valid text in Gemini API response. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                reason_name = reason.name if hasattr(reason, 'name') else str(reason)
                print(f"[DLAgent Safety] Prompt blocked. Reason: {reason_name}")
                return {"error": f"Prompt blocked for driving licence analysis. Reason: {reason_name}"}
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[DLAgent Safety] Analysis finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Driving licence analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from driving licence analysis API (Gemini)."}

    except Exception as e:
        print(f"[DLAgent ERROR] An error occurred during Gemini driving licence analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during driving licence analysis (Gemini): {str(e)}"}

# Legacy function for backward compatibility
def analyze_driving_licence_image(image_path: str) -> Optional[dict]:
    """
    Legacy function for backward compatibility.
    Now routes to the unified analyze_driving_licence_document function.
    """
    # Determine MIME type based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    mime_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
        '.pdf': 'application/pdf'
    }
    
    file_mime_type = mime_type_map.get(file_ext, 'image/png')  # Default to PNG
    return analyze_driving_licence_document(image_path, file_mime_type)

if __name__ == '__main__':
    # Test the driving licence agent with Gemini
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nSkipping Driving Licence Agent tests: GOOGLE_API_KEY not set.")
    else:
        # Test with a sample image
        sample_image_path = "temp_sample_driving_licence.png"
        if not os.path.exists(sample_image_path):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (400, 300), color='green')
                d = ImageDraw.Draw(img)
                d.text((10, 10), "Sample Driving Licence for Testing", fill=(255, 255, 255))
                d.text((10, 50), "Name: Jane Smith", fill=(255, 255, 255))
                d.text((10, 70), "Licence No: SMITH123456AB9CD", fill=(255, 255, 255))
                d.text((10, 90), "Categories: B, BE", fill=(255, 255, 255))
                img.save(sample_image_path)
                print(f"Created dummy driving licence image at {sample_image_path}")
            except ImportError:
                print("Pillow not installed. Cannot create dummy image.")
            except Exception as e:
                print(f"Could not create dummy image: {e}")

        if os.path.exists(sample_image_path):
            print(f"\n--- Testing Driving Licence Agent (Gemini) with: {sample_image_path} ---")
            analysis_result = analyze_driving_licence_document(sample_image_path, "image/png")
            if analysis_result:
                print("--- Driving Licence Analysis Result (Gemini) ---")
                try:
                    print(json.dumps(analysis_result, indent=4))
                except (json.JSONDecodeError, TypeError):
                    print(analysis_result)
            else:
                print("--- Driving Licence Analysis (Gemini) Failed ---")
        else:
            print(f"Skipping Driving Licence Agent test: Sample image not found at {sample_image_path}")

        # Clean up dummy file
        if sample_image_path == "temp_sample_driving_licence.png" and os.path.exists(sample_image_path):
            try:
                os.remove(sample_image_path)
                print(f"Removed dummy driving licence image at {sample_image_path}")
            except Exception as e:
                print(f"Error removing dummy image: {e}") 