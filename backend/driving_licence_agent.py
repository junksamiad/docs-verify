import openai
import os
import json
import base64

# --- Model Configuration ---
DRIVING_LICENCE_AGENT_MODEL_ID = 'gpt-4o-2024-08-06'
# --- End Model Configuration ---

client = openai.OpenAI()

def encode_image_to_base64_for_dl(image_path: str) -> str | None:
    if not os.path.exists(image_path):
        print(f"[DLAgent ERROR] Image file not found at {image_path} for encoding.")
        return None
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}" # Assuming PNG
    except Exception as e:
        print(f"[DLAgent ERROR] Error encoding image {image_path}: {e}")
        return None

def analyze_driving_licence_image(image_path: str) -> dict | None:
    print(f"[DLAgent LOG] Analyzing driving licence image: {image_path}")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[DLAgent ERROR] OPENAI_API_KEY environment variable not set.")
        return None

    base64_image = encode_image_to_base64_for_dl(image_path)
    if not base64_image:
        return None

    prompt_text = (
        "You are an AI assistant specialized in analyzing driving licence images. "
        "Carefully examine the provided driving licence image. Your goal is to assess its quality, identify potential issues for manual review, and extract key information. \n\n"
        "1. Image Quality Assessment: Provide a brief summary of the image quality (e.g., clarity, lighting, readability of text, obstructions, glare). Example: \"Image is clear, well-lit, all text is legible.\" or \"Image is slightly blurry with glare over the address field.\".\n"
        "2. Manual Verification Flags: Identify and list any specific visual characteristics, anomalies, or inconsistencies that might suggest the document warrants manual human verification. Examples: \"Font on date of birth appears different from other text.\", \"Edges of the photograph look uneven or tampered with.\", \"Hologram is not visible or looks incorrect (if one is expected and discernible).\". If no specific flags, state \"No specific flags for manual verification noted from visual inspection.\".\n"
        "3. Extracted Driving Licence Information: Extract the following details. If a section or detail is not present or legible, omit the key or use null for its value.\n"
        "   - licence_number: (string)\n"
        "   - surname: (string)\n"
        "   - given_names: (string, all given names)\n"
        "   - date_of_birth: (string, e.g., DD MM YYYY or MM/DD/YYYY as it appears)\n"
        "   - address: (string, full address if present and legible)\n"
        "   - date_of_issue: (string, as it appears)\n"
        "   - date_of_expiry: (string, as it appears)\n"
        "   - issuing_authority: (string, e.g., DVLA, DMV California)\n"
        "   - categories_entitlements: (array of strings or a single string, listing vehicle categories the holder is entitled to drive, e.g., [\"B\", \"C1\"] or \"B, BE, C1\")\n"
        "   - country_of_issue: (string, if discernible or part of issuing authority, e.g., \"UK\", \"USA\")\n"
        "   - restrictions: (string or array of strings, listing any restrictions if present, e.g., \"01 - eyesight correction\")\n\n"
        "Your response MUST be a single, valid JSON object. Do not include any text outside of this JSON object. "
        "The JSON object should have the following top-level keys: \n"
        "  - \"image_quality_summary\": (string, your assessment of the image quality itself)\n"
        "  - \"manual_verification_flags\": (array of strings, each string being a reason for potential manual review; or an empty array if none noted)\n"
        "  - \"licence_details\": (an object containing all the extracted driving licence fields like licence_number, surname, etc.)\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image, "detail": "high"} # Use high detail
                }
            ]
        }
    ]

    api_response_content = None
    try:
        print(f"[DLAgent LOG] Sending request to OpenAI model: {DRIVING_LICENCE_AGENT_MODEL_ID}")
        response = client.chat.completions.create(
            model=DRIVING_LICENCE_AGENT_MODEL_ID,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            api_response_content = response.choices[0].message.content
            print(f"[DLAgent LOG] Raw JSON response content: {api_response_content}")
            try:
                parsed_json = json.loads(api_response_content)
                required_keys = ["image_quality_summary", "manual_verification_flags", "licence_details"]
                if all(key in parsed_json for key in required_keys):
                    if not isinstance(parsed_json.get("manual_verification_flags"), list):
                        print("[DLAgent WARNING] 'manual_verification_flags' is not a list.")
                    if not isinstance(parsed_json.get("licence_details"), dict):
                        print("[DLAgent WARNING] 'licence_details' is not an object/dictionary.")
                    return parsed_json
                else:
                    print(f"[DLAgent ERROR] Returned JSON missing required keys. Expected: {required_keys}, Got: {list(parsed_json.keys())}")
                    return {"error": "Driving licence analysis returned incomplete data.", "details": api_response_content}

            except json.JSONDecodeError as e:
                print(f"[DLAgent ERROR] Decoding JSON response from API: {e}. Raw: {api_response_content}")
                return {"error": "Failed to decode driving licence analysis JSON.", "details": api_response_content}
        else:
            print(f"[DLAgent ERROR] No valid content in API response. Full response: {response}")
            return {"error": "No content from driving licence analysis API."}

    except Exception as e:
        print(f"[DLAgent ERROR] An error occurred: {e}. API response content: {api_response_content if api_response_content else 'N/A'}")
        return {"error": f"Exception during driving licence analysis: {str(e)}"}

if __name__ == '__main__':
    if os.environ.get("OPENAI_API_KEY"):
        sample_dl_image_path = "path/to/your/sample_driving_licence.png"
        if os.path.exists(sample_dl_image_path):
            print(f"--- Testing Driving Licence Agent with: {sample_dl_image_path} ---")
            analysis_result = analyze_driving_licence_image(sample_dl_image_path)
            if analysis_result:
                print("--- Driving Licence Analysis Result ---")
                print(json.dumps(analysis_result, indent=4))
            else:
                print("--- Driving Licence Analysis Failed ---")
        else:
            print(f"Skipping DL Agent test: Sample image not found at {sample_dl_image_path}. Please update path.")
    else:
        print("Skipping DL Agent test: OPENAI_API_KEY not set.") 