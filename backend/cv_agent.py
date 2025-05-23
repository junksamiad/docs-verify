import openai
import os
import json
import base64

# --- Model Configuration ---
CV_AGENT_MODEL_ID = 'gpt-4o-2024-08-06'  # Or your preferred capable GPT-4 model
# --- End Model Configuration ---

# Initialize OpenAI client (picks up API key from environment OPENAI_API_KEY)
client = openai.OpenAI()

def encode_image_to_base64_for_cv(image_path: str) -> str | None:
    """
    Encodes an image file to a Base64 string, assuming image_path is a PNG.
    Returns a data URL.
    """
    if not os.path.exists(image_path):
        print(f"[CVAgent ERROR] Image file not found at {image_path} for encoding.")
        return None
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}" # Assuming PNG
    except Exception as e:
        print(f"[CVAgent ERROR] Error encoding image {image_path}: {e}")
        return None

def analyze_cv_image(image_path: str) -> dict | None:
    """
    Analyzes a CV/Resume image for key information, quality, and verification flags.
    image_path is expected to be a path to a processable image (e.g., PNG).
    """
    print(f"[CVAgent LOG] Analyzing CV image: {image_path}")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[CVAgent ERROR] OPENAI_API_KEY environment variable not set.")
        return None

    base64_image = encode_image_to_base64_for_cv(image_path)
    if not base64_image:
        return None

    prompt_text = (
        "You are an AI assistant specialized in analyzing CVs/Resumes from images. "
        "Carefully examine the provided CV image. Your goal is to assess its quality, identify potential issues for manual review, and extract key information. \n\n"
        "1. Image Quality Assessment: Provide a brief summary of the CV image quality (e.g., clarity, lighting, readability, obstructions, glare). Example: \"Image is clear and text is easily legible.\" or \"CV is scanned at a slight angle, some text is blurry.\".\n"
        "2. Manual Verification Flags: Identify and list any specific visual characteristics, anomalies, or inconsistencies that might warrant manual human verification. Examples: \"Font on a specific section appears different from other text.\", \"Edges of the document look uneven or tampered with.\", \"Contact phone number format appears unusual for the claimed region.\", \"Multiple different fonts and formatting styles used inconsistently.\". Also, critically, analyze the work_experience section for any significant unexplained gaps in employment dates (e.g., more than 3-6 months between roles without explanation) and list this as a flag if found, for example: \"Significant unexplained gap in employment dates between 2018-2020.\". If no specific flags are noted (including no significant employment gaps), state \"No specific flags for manual verification noted from visual inspection.\".\n"
        "3. Extracted CV Information: Extract the following details from the CV. If a section or detail is not present, omit the key or use null where appropriate for individual fields.\n"
        "   - personal_details: (object) Contains name (full name if possible), phone_number, email_address, address (if listed), linkedin_profile_url (if listed).\n"
        "   - summary_objective: (string) The professional summary or objective statement, if present.\n"
        "   - work_experience: (array of objects) Each object should contain: company_name, job_title, employment_dates (e.g., \"Jan 2020 - Present\" or \"2018 - 2019\"), responsibilities_achievements (a brief summary or bullet points string).\n"
        "   - education: (array of objects) Each object should contain: institution_name, degree_qualification (e.g., \"BSc Computer Science\"), graduation_date_period (e.g., \"May 2018\" or \"2014 - 2018\").\n"
        "   - skills: (array of strings, listing key skills or technologies. Alternatively, an object with skill categories as keys and arrays of skills as values if clearly structured that way in the CV).\n"
        "   - references_availability: (string) A statement about the availability of references (e.g., \"References available upon request\" or actual reference details if provided).\n\n"
        "Your response MUST be a single, valid JSON object. Do not include any text outside of this JSON object. "
        "The JSON object should have the following top-level keys: \n"
        "  - \"image_quality_summary\": (string, your assessment of the image quality itself)\n"
        "  - \"manual_verification_flags\": (array of strings, each string being a reason for potential manual review; or an empty array if none noted)\n"
        "  - \"cv_data\": (an object containing all the extracted CV sections: personal_details, summary_objective, work_experience, education, skills, references_availability)\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image, "detail": "high"} # Use high detail for CVs
                }
            ]
        }
    ]

    api_response_content = None
    try:
        print(f"[CVAgent LOG] Sending request to OpenAI model: {CV_AGENT_MODEL_ID}")
        response = client.chat.completions.create(
            model=CV_AGENT_MODEL_ID,
            messages=messages,
            temperature=0.2, # Low temperature for more deterministic extraction
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            api_response_content = response.choices[0].message.content
            print(f"[CVAgent LOG] Raw JSON response content: {api_response_content}")
            try:
                parsed_json = json.loads(api_response_content)
                required_keys = ["image_quality_summary", "manual_verification_flags", "cv_data"]
                if all(key in parsed_json for key in required_keys):
                    if not isinstance(parsed_json.get("manual_verification_flags"), list):
                        print("[CVAgent WARNING] 'manual_verification_flags' is not a list. Review agent prompt/response.")
                    if not isinstance(parsed_json.get("cv_data"), dict):
                        print("[CVAgent WARNING] 'cv_data' is not an object/dictionary. Review agent prompt/response.")
                    return parsed_json
                else:
                    print("[CVAgent ERROR] Returned JSON is missing one or more required top-level keys.")
                    print(f"[CVAgent LOG] Expected keys: {required_keys}")
                    print(f"[CVAgent LOG] Received keys: {list(parsed_json.keys())}")
                    return {"error": "CV analysis returned incomplete data structure.", "details": api_response_content}

            except json.JSONDecodeError as e:
                print(f"[CVAgent ERROR] Decoding JSON response from API: {e}. Raw response content: {api_response_content}")
                return {"error": "Failed to decode CV analysis JSON.", "details": api_response_content}
        else:
            print(f"[CVAgent ERROR] No valid content in API response. Full response: {response}")
            return {"error": "No content from CV analysis API."}

    except Exception as e:
        print(f"[CVAgent ERROR] An error occurred: {e}. API response content was: {api_response_content if api_response_content else 'N/A'}")
        return {"error": f"Exception during CV analysis: {str(e)}"}

def analyze_cv_from_text(text_content: str) -> dict | None:
    """
    Analyzes CV/Resume text content for key information, quality, and verification flags.
    """
    print(f"[CVAgent LOG] Analyzing CV text content (length: {len(text_content)})")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[CVAgent ERROR] OPENAI_API_KEY environment variable not set.")
        return None
    if not text_content:
        print("[CVAgent ERROR] Text content for CV analysis cannot be empty.")
        return None

    # The prompt is largely the same, but it refers to "text content" instead of "image"
    # and the "Image Quality Assessment" is rephrased to "Text Readability/Structure Assessment"
    prompt_text = (
        "You are an AI assistant specialized in analyzing CVs/Resumes from text content. "
        "Carefully examine the provided CV text. Your goal is to assess its readability and structure, identify potential issues for manual review, and extract key information. \n\n"
        "1. Text Readability/Structure Assessment: Provide a brief summary of the CV's text quality (e.g., is it well-structured, are there clear headings, any obvious OCR errors if it looks like scanned text?). Example: \"Text is well-structured with clear sections.\" or \"Text appears to be poorly OCRed with many typos.\".\n"
        "2. Manual Verification Flags: Identify and list any specific characteristics, anomalies, or inconsistencies in the text that might warrant manual human verification. Examples: \"Contact phone number format appears unusual.\", \"Inconsistent date formats used throughout.\". Also, critically, analyze the work_experience section for any significant unexplained gaps in employment dates (e.g., more than 3-6 months between roles without explanation) and list this as a flag if found, for example: \"Significant unexplained gap in employment dates between 2018-2020.\". If no specific flags are noted (including no significant employment gaps), state \"No specific flags for manual verification noted from text analysis.\".\n"
        "3. Extracted CV Information: Extract the following details from the CV text. If a section or detail is not present, omit the key or use null where appropriate for individual fields.\n"
        "   - personal_details: (object) Contains name (full name if possible), phone_number, email_address, address (if listed), linkedin_profile_url (if listed).\n"
        "   - summary_objective: (string) The professional summary or objective statement, if present.\n"
        "   - work_experience: (array of objects) Each object should contain: company_name, job_title, employment_dates (e.g., \"Jan 2020 - Present\" or \"2018 - 2019\"), responsibilities_achievements (a brief summary or bullet points string).\n"
        "   - education: (array of objects) Each object should contain: institution_name, degree_qualification (e.g., \"BSc Computer Science\"), graduation_date_period (e.g., \"May 2018\" or \"2014 - 2018\").\n"
        "   - skills: (array of strings, listing key skills or technologies. Alternatively, an object with skill categories as keys and arrays of skills as values if clearly structured that way in the CV).\n"
        "   - references_availability: (string) A statement about the availability of references (e.g., \"References available upon request\" or actual reference details if provided).\n\n"
        "Your response MUST be a single, valid JSON object. Do not include any text outside of this JSON object. "
        "The JSON object should have the following top-level keys: \n"
        "  - \"image_quality_summary\": (string, your assessment of the text readability/structure itself - keep key name for consistency with image version)\n"
        "  - \"manual_verification_flags\": (array of strings, each string being a reason for potential manual review; or an empty array if none noted)\n"
        "  - \"cv_data\": (an object containing all the extracted CV sections: personal_details, summary_objective, work_experience, education, skills, references_availability)\n"
        f"CV Text to Analyze:\n{text_content[:20000]}" # Truncate if very long, adjust as needed
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant that analyzes CV text to extract structured data, assess readability, and identify verification flags. Respond in JSON as per the user's schema description."},
        {"role": "user", "content": prompt_text}
    ]

    api_response_content = None
    try:
        # Corrected f-string for logging
        log_text_snippet = text_content[:200].replace('\n', ' ')
        print(f"[CVAgent LOG] Sending text (first 200 chars: '{log_text_snippet}') to {CV_AGENT_MODEL_ID} for text analysis.")
        response = client.chat.completions.create(
            model=CV_AGENT_MODEL_ID,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            api_response_content = response.choices[0].message.content
            print(f"[CVAgent LOG] Raw JSON response from text analysis: {api_response_content}")
            try:
                parsed_json = json.loads(api_response_content)
                required_keys = ["image_quality_summary", "manual_verification_flags", "cv_data"]
                if all(key in parsed_json for key in required_keys):
                    if not isinstance(parsed_json.get("manual_verification_flags"), list):
                        print("[CVAgent WARNING] 'manual_verification_flags' (from text) is not a list.")
                    if not isinstance(parsed_json.get("cv_data"), dict):
                        print("[CVAgent WARNING] 'cv_data' (from text) is not an object/dictionary.")
                    return parsed_json
                else:
                    print("[CVAgent ERROR] Text-based CV analysis returned JSON missing required keys.")
                    return {"error": "Text-based CV analysis returned incomplete data structure.", "details": api_response_content}
            except json.JSONDecodeError as e:
                print(f"[CVAgent ERROR] Decoding JSON from text-based CV analysis: {e}. Raw: {api_response_content}")
                return {"error": "Failed to decode text-based CV analysis JSON.", "details": api_response_content}
        else:
            print(f"[CVAgent ERROR] No valid content in API response for text-based CV analysis.")
            return {"error": "No content from text-based CV analysis API."}

    except Exception as e:
        print(f"[CVAgent ERROR] An error occurred during text-based CV analysis: {e}.")
        return {"error": f"Exception during text-based CV analysis: {str(e)}"}

if __name__ == '__main__':
    if os.environ.get("OPENAI_API_KEY"):
        # Replace with a real path to a PNG image of a CV
        sample_cv_image_path = "path/to/your/sample_cv.png"
        if os.path.exists(sample_cv_image_path):
            print(f"--- Testing CV Agent with: {sample_cv_image_path} ---")
            analysis_result = analyze_cv_image(sample_cv_image_path)
            if analysis_result:
                print("--- CV Analysis Result ---")
                print(json.dumps(analysis_result, indent=4))
            else:
                print("--- CV Analysis Failed ---")
        else:
            print(f"Skipping CV Agent test: Sample image not found at {sample_cv_image_path}. Please update the path.")
    else:
        print("Skipping CV Agent test: OPENAI_API_KEY not set.") 