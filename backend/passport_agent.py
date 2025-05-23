import openai
import os
import json
import base64

# --- Model Configuration ---
PASSPORT_AGENT_MODEL_ID = 'gpt-4o-2024-08-06'  # Or your preferred capable GPT-4 model
# --- End Model Configuration ---

# Initialize OpenAI client (picks up API key from environment OPENAI_API_KEY)
# Ensure OPENAI_API_KEY is set in your environment (e.g., via .env file loaded by server.py)
client = openai.OpenAI()

def encode_image_to_base64_for_passport(image_path: str) -> str | None:
    """
    Encodes an image file to a Base64 string, assuming image_path is a PNG.
    Returns a data URL.
    """
    if not os.path.exists(image_path):
        print(f"[PassportAgent ERROR] Image file not found at {image_path} for encoding.")
        return None
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Assuming PNG as it's the target format from any prior conversion
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"[PassportAgent ERROR] Error encoding image {image_path}: {e}")
        return None

def analyze_passport_image(image_path: str) -> dict | None:
    """
    Analyzes a passport image for genuineness and extracts information.
    image_path is expected to be a path to a processable image (e.g., PNG).
    """
    print(f"[PassportAgent LOG] Analyzing passport image: {image_path}")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[PassportAgent ERROR] OPENAI_API_KEY environment variable not set.")
        return None

    base64_image = encode_image_to_base64_for_passport(image_path)
    if not base64_image:
        return None

    # Define the desired JSON output structure for the prompt
    # Note: Complex schemas are better, but for now, we guide with a detailed text description.
    # For more robust JSON, use the JSON mode with a schema if the model version supports it well with vision.
    # GPT-4o supports response_format={ "type": "json_object" }
    
    prompt_text = (
        "You are an AI assistant tasked with analyzing passport images. "
        "Carefully examine the provided passport image. Your goal is to objectively describe its features and extract information. \n"
        "1. Describe observable visual characteristics of the document. Note any standard passport features visible (e.g., specific watermarks if clearly discernible, type of font if it appears unusual, photo integration method if visible). Do not speculate on authenticity. \n"
        "2. Extract all visible textual information from the passport. This includes, but is not limited to: "
        "surname, given names, passport number, nationality, date of birth, sex, date of issue, date of expiry, issuing authority, place of birth, and any MRZ (Machine Readable Zone) lines if present and legible.\n"
        "Your response MUST be a single, valid JSON object. Do not include any text outside of this JSON object. "
        "The JSON object should have the following top-level keys: \n"
        "  - \"observed_features\": (string, a description of visible characteristics and standard features noted)\n"
        "  - \"extracted_information\": (an object containing all extracted fields, e.g., { \"surname\": \"value\", \"given_names\": \"value\", ...})\n"
        "If certain information is not visible or legible, represent its value as null or omit the key within 'extracted_information'. "
        "If MRZ lines are present and legible, include them as a field named \"mrz_lines\" within 'extracted_information'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image, "detail": "high"}
                }
            ]
        }
    ]

    api_response_content = None
    try:
        print(f"[PassportAgent LOG] Sending request to OpenAI model: {PASSPORT_AGENT_MODEL_ID}")
        response = client.chat.completions.create(
            model=PASSPORT_AGENT_MODEL_ID,
            messages=messages,
            temperature=0.1, # Low temperature for more deterministic output
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            api_response_content = response.choices[0].message.content
            print(f"[PassportAgent LOG] Raw JSON response content: {api_response_content}")
            try:
                parsed_json = json.loads(api_response_content)
                # Updated validation for new top-level keys
                if all(key in parsed_json for key in ["observed_features", "extracted_information"]):
                    return parsed_json
                else:
                    print("[PassportAgent ERROR] Returned JSON is missing one or more required top-level keys ('observed_features', 'extracted_information').")
                    print(f"[PassportAgent LOG] Received keys: {list(parsed_json.keys())}")
                    return {"error": "Passport analysis returned incomplete data structure.", "details": api_response_content}

            except json.JSONDecodeError as e:
                print(f"[PassportAgent ERROR] Decoding JSON response from API: {e}. Raw response content: {api_response_content}")
                return {"error": "Failed to decode passport analysis JSON.", "details": api_response_content}
        else:
            print(f"[PassportAgent ERROR] No valid content in API response. Full response: {response}")
            return {"error": "No content from passport analysis API."}

    except Exception as e:
        print(f"[PassportAgent ERROR] An error occurred: {e}. API response content was: {api_response_content if api_response_content else 'N/A'}")
        return {"error": f"Exception during passport analysis: {str(e)}"}

if __name__ == '__main__':
    # This is for direct testing of the passport agent.
    # Ensure OPENAI_API_KEY is set and you have a sample_passport.png
    if os.environ.get("OPENAI_API_KEY"):
        sample_image_path = "path/to/your/sample_passport.png"  # Replace with a real path to a PNG
        if os.path.exists(sample_image_path):
            print(f"--- Testing Passport Agent with: {sample_image_path} ---")
            analysis_result = analyze_passport_image(sample_image_path)
            if analysis_result:
                print("--- Passport Analysis Result ---")
                print(json.dumps(analysis_result, indent=4))
            else:
                print("--- Passport Analysis Failed ---")
        else:
            print(f"Skipping Passport Agent test: Sample image not found at {sample_image_path}")
    else:
        print("Skipping Passport Agent test: OPENAI_API_KEY not set.") 