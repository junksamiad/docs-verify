import os
import json
from typing import Optional

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Part, Blob

# --- Model Configuration ---
PASSPORT_AGENT_MODEL_ID = 'gemini-2.5-pro-preview-05-06'  # Using Gemini 2.5 for native PDF/HEIC support
# --- End Model Configuration ---

def get_gemini_client_and_model() -> Optional[genai.Client]:
    """Initializes and returns the Gemini client if API key is set."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("[PassportAgent ERROR] GOOGLE_API_KEY environment variable not set.")
        return None
    try:
        client = genai.Client(api_key=google_api_key)
        return client
    except Exception as e:
        print(f"[PassportAgent ERROR] Failed to initialize Gemini client: {e}")
        return None

def analyze_passport_document(file_path: str, file_mime_type: str) -> Optional[dict]:
    """
    Analyzes a passport document (image or PDF) for genuineness and extracts information.
    Supports all image formats (PNG, JPEG, HEIC, HEIF, etc.) and PDF natively.
    """
    print(f"[PassportAgent LOG] Analyzing passport document: {file_path} ({file_mime_type})")
    
    client = get_gemini_client_and_model()
    if not client:
        return {"error": "Gemini client not initialized (GOOGLE_API_KEY missing or invalid)."}

    if not os.path.exists(file_path):
        print(f"[PassportAgent ERROR] File not found at {file_path}")
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
        print(f"[PassportAgent ERROR] Error reading file {file_path}: {e}")
        return {"error": f"Could not read file: {e}"}

    prompt_text = """You are an AI assistant specialized in analyzing passport documents (images or PDFs).
Your goal is to objectively describe features, assess quality, identify potential issues, and extract information.
Carefully examine the provided passport document.

Analyze the following:

1. **Document Quality Assessment**: Provide a brief summary of the document quality (e.g., clarity, lighting, obstructions, glare, scan quality). 
   Example: "Document is clear, well-lit, with no obstructions" or "Document is slightly blurry with some glare over the date of birth."

2. **Manual Verification Flags**: Identify and list any specific visual characteristics, anomalies, or inconsistencies that might suggest the document warrants manual human verification. 
   Examples: "Unusual font detected in the 'Date of Issue' field", "Edge of the photograph appears to be digitally manipulated", "MRZ checksum appears inconsistent".
   If no specific flags, use an empty array.

3. **Observable Visual Characteristics**: Describe observable visual characteristics of the document. Note any standard passport features visible (e.g., watermarks if clearly discernible, font types, photo integration method, security features).
   Do not speculate on authenticity beyond flagging potential issues for manual review.

4. **Extract Information**: Extract all visible textual information from the passport, including:
   - surname, given_names, passport_number, nationality, date_of_birth, sex
   - date_of_issue, date_of_expiry, issuing_authority, place_of_birth
   - MRZ (Machine Readable Zone) lines if present and legible
   
   If certain information is not visible or legible, use null for that field.

Your response MUST be a single, valid JSON object with these exact top-level keys:
- "image_quality_summary": (string) your assessment of the document quality
- "manual_verification_flags": (array of strings) reasons for potential manual review, or empty array if none
- "observed_features": (string) description of visible document characteristics and standard features
- "extracted_information": (object) containing all extracted fields with null for unreadable/missing data

Do not include any text outside of this JSON object."""

    contents = [file_part, prompt_text]

    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        max_output_tokens=32000
    )

    api_response_text = None
    try:
        print(f"[PassportAgent LOG] Sending request to Gemini model: {PASSPORT_AGENT_MODEL_ID}")
        response = client.models.generate_content(
            model=PASSPORT_AGENT_MODEL_ID,
            contents=contents,
            config=generation_config
        )
        
        if response.text:
            api_response_text = response.text
            print(f"[PassportAgent LOG] Raw JSON response: {api_response_text}")
            try:
                parsed_json = json.loads(api_response_text)
                # Validate required keys
                required_keys = ["image_quality_summary", "manual_verification_flags", "observed_features", "extracted_information"]
                if all(key in parsed_json for key in required_keys):
                    # Validate manual_verification_flags is a list
                    if not isinstance(parsed_json.get("manual_verification_flags"), list):
                        print("[PassportAgent WARNING] 'manual_verification_flags' is not a list. Converting to list.")
                        flags = parsed_json.get("manual_verification_flags", [])
                        parsed_json["manual_verification_flags"] = [flags] if isinstance(flags, str) else []
                    return parsed_json
                else:
                    print("[PassportAgent ERROR] Returned JSON is missing required keys.")
                    print(f"[PassportAgent LOG] Expected: {required_keys}")
                    print(f"[PassportAgent LOG] Received: {list(parsed_json.keys())}")
                    return {"error": "Passport analysis returned incomplete data structure.", "details": api_response_text}

            except json.JSONDecodeError as e:
                print(f"[PassportAgent ERROR] JSON decode error: {e}. Raw response: {api_response_text}")
                return {"error": "Failed to decode passport analysis JSON.", "details": api_response_text}
        else:
            # Fallback to check candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                api_response_text = response.candidates[0].content.parts[0].text
                print(f"[PassportAgent LOG] Raw JSON response from candidate: {api_response_text}")
                try:
                    parsed_json = json.loads(api_response_text)
                    required_keys = ["image_quality_summary", "manual_verification_flags", "observed_features", "extracted_information"]
                    if all(key in parsed_json for key in required_keys):
                        return parsed_json
                    else:
                        return {"error": "Passport analysis returned incomplete data structure.", "details": api_response_text}
                except json.JSONDecodeError as e:
                    print(f"[PassportAgent ERROR] JSON decode error (from candidate): {e}")
                    return {"error": "Failed to decode passport analysis JSON.", "details": api_response_text}
            
            print(f"[PassportAgent ERROR] No valid text in Gemini API response. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                reason_name = reason.name if hasattr(reason, 'name') else str(reason)
                print(f"[PassportAgent Safety] Prompt blocked. Reason: {reason_name}")
                return {"error": f"Prompt blocked for passport analysis. Reason: {reason_name}"}
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[PassportAgent Safety] Analysis finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Passport analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from passport analysis API (Gemini)."}

    except Exception as e:
        print(f"[PassportAgent ERROR] An error occurred during Gemini passport analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during passport analysis (Gemini): {str(e)}"}

# Legacy function for backward compatibility
def analyze_passport_image(image_path: str) -> Optional[dict]:
    """
    Legacy function for backward compatibility.
    Now routes to the unified analyze_passport_document function.
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
    return analyze_passport_document(image_path, file_mime_type)

if __name__ == '__main__':
    # Test the passport agent with Gemini
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nSkipping Passport Agent tests: GOOGLE_API_KEY not set.")
    else:
        # Test with a sample image
        sample_image_path = "temp_sample_passport.png"
        if not os.path.exists(sample_image_path):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (400, 300), color='blue')
                d = ImageDraw.Draw(img)
                d.text((10, 10), "Sample Passport for Testing", fill=(255, 255, 255))
                d.text((10, 50), "Name: John Doe", fill=(255, 255, 255))
                d.text((10, 70), "Passport No: 123456789", fill=(255, 255, 255))
                d.text((10, 90), "Nationality: British", fill=(255, 255, 255))
                img.save(sample_image_path)
                print(f"Created dummy passport image at {sample_image_path}")
            except ImportError:
                print("Pillow not installed. Cannot create dummy image.")
            except Exception as e:
                print(f"Could not create dummy image: {e}")

        if os.path.exists(sample_image_path):
            print(f"\n--- Testing Passport Agent (Gemini) with: {sample_image_path} ---")
            analysis_result = analyze_passport_document(sample_image_path, "image/png")
            if analysis_result:
                print("--- Passport Analysis Result (Gemini) ---")
                try:
                    print(json.dumps(analysis_result, indent=4))
                except (json.JSONDecodeError, TypeError):
                    print(analysis_result)
            else:
                print("--- Passport Analysis (Gemini) Failed ---")
        else:
            print(f"Skipping Passport Agent test: Sample image not found at {sample_image_path}")

        # Clean up dummy file
        if sample_image_path == "temp_sample_passport.png" and os.path.exists(sample_image_path):
            try:
                os.remove(sample_image_path)
                print(f"Removed dummy passport image at {sample_image_path}")
            except Exception as e:
                print(f"Error removing dummy image: {e}") 