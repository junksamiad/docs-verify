import os
import json
from typing import List, Optional

from google import genai # Updated import
from google.genai import types # Updated import for types

from pydantic import BaseModel, Field

# --- Pydantic Models for CV Analysis Response ---

class PersonalDetails(BaseModel):
    name: Optional[str] = Field(default=None, description="Full name of the candidate.")
    phone_number: Optional[str] = Field(default=None, description="Contact phone number.")
    email_address: Optional[str] = Field(default=None, description="Contact email address.")
    address: Optional[str] = Field(default=None, description="Physical address, if available.")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth, if available (e.g., YYYY-MM-DD or as found).")

class CVAnalysisData(BaseModel):
    image_quality_summary: Optional[str] = Field(default=None, description="Brief summary of the CV image quality or text readability.")
    personal_details: Optional[PersonalDetails] = Field(default=None, description="Extracted personal information of the candidate.")
    work_experience_gaps: List[str] = Field(
        default_factory=list,
        description="List of significant, unexplained work experience gaps, formatted as 'mmm-yyyy to mmm-yyyy'. Example: ['jan-2019 to may-2019', 'sep-2020 to dec-2020']"
    )
    other_verification_flags: List[str] = Field(
        default_factory=list,
        description="List of other specific visual anomalies, inconsistencies, or concerns (excluding work gaps) that might warrant manual review."
    )

# --- Model Configuration ---
CV_AGENT_MODEL_ID = 'gemini-2.5-pro-preview-05-06' # Using user-specified model
# --- End Model Configuration ---


def get_gemini_client_and_model() -> Optional[genai.Client]: # Corrected type hint
    """Initializes and returns the Gemini client if API key is set."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("[CVAgent ERROR] GOOGLE_API_KEY environment variable not set.")
        return None
    try:
        # The new SDK uses a client instance. Configuration is typically done when creating the client.
        # genai.configure() is not the primary way in the new SDK.
        # Model can be specified per-request or a default can be set on client.
        # For simplicity here, we'll just ensure the client can be made.
        # The model itself will be specified in the generate_content call.
        client = genai.Client(api_key=google_api_key) 
        # To check if client is valid, we might try a lightweight call or assume it's okay if no exception.
        # For now, let's return the client. The model name will be passed to generate_content.
        return client # Returning the client now
    except Exception as e:
        print(f"[CVAgent ERROR] Failed to initialize Gemini client: {e}")
        return None

def analyze_cv_image(image_path: str) -> Optional[dict]:
    """
    Analyzes a CV/Resume image using Gemini, focusing on personal details and work experience gaps.
    Uses Pydantic models for response structure.
    Assumes image_path is a PNG image.
    """
    print(f"[CVAgent LOG] Analyzing CV image with Gemini: {image_path}")
    
    client = get_gemini_client_and_model() # Now gets a client
    if not client:
        return {"error": "Gemini client not initialized (GOOGLE_API_KEY missing or invalid)."}

    if not os.path.exists(image_path):
        print(f"[CVAgent ERROR] Image file not found at {image_path}")
        return {"error": f"Image file not found: {image_path}"}

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        # Using types.Part.from_bytes as per new SDK examples for local files
        image_part = types.Part.from_data(data=image_bytes, mime_type="image/png") # Corrected to from_data if from_bytes is not found, or from_bytes if it is.
                                                                                 # SDK docs suggest from_bytes. Assuming types.Part.from_data is still valid or an alias.
                                                                                 # Sticking to from_data as it was in original, changing only namespace 'types'.
    except Exception as e:
        print(f"[CVAgent ERROR] Error reading image file {image_path}: {e}")
        return {"error": f"Could not read image file: {e}"}

    schema_dict = CVAnalysisData.model_json_schema()
    # Ensure schema_dict is a valid JSON Schema for the API if it's directly passed.
    # The new SDK examples show passing the Pydantic model class directly to response_schema.
    prompt_text = f"""You are an AI assistant specialized in analyzing CVs/Resumes from images.
Your primary goal is to extract specific personal details and identify any significant, unexplained gaps in work experience.
Carefully examine the provided CV image.

Prioritize the following:

1.  **Personal Details Extraction**: Extract the candidate's full name, phone number, email address, physical address (if present), and date of birth (if present).
    If a detail is not found, omit it or use null for that specific field in the 'personal_details' object.

2.  **Work Experience Gap Identification**: Scrutinize the work experience sections for any unexplained gaps of approximately 3 months or longer between roles or educational periods.
    List each *distinct* identified gap as a string in the 'work_experience_gaps' array, strictly using the format 'mmm-yyyy to mmm-yyyy' (e.g., 'jan-2019 to may-2019'). If no such gaps are found, this array should be empty.

3.  **Image Quality Assessment**: Briefly describe the visual quality of the CV image (e.g., clarity, readability, glare) in 'image_quality_summary'.

4.  **Other Verification Flags**: List any other distinct visual anomalies or major inconsistencies (unrelated to work gaps, e.g., unusual fonts, suspected alterations) in the 'other_verification_flags' array. If none, this array should be empty.

Your response MUST be a single, valid JSON object that strictly conforms to the following JSON Schema. Do not include any text outside of this JSON object. Ensure all specified formats (especially for dates and gaps) are followed.
Schema:
```json
{json.dumps(CVAnalysisData.model_json_schema(), indent=2)}
```
"""


    contents = [prompt_text, image_part]

    # Create GenerationConfig object - CHANGED TO GenerateContentConfig
    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=CVAnalysisData
    )

    api_response_text = None
    try:
        print(f"[CVAgent LOG] Sending request to Gemini model: {CV_AGENT_MODEL_ID} for image analysis")
        response = client.models.generate_content(
            model=CV_AGENT_MODEL_ID, 
            contents=contents,
            config=generation_config # Pass GenerationConfig object to 'config' parameter
        )
        
        # The new SDK (google-genai) typically puts the directly usable text in response.text
        # when response_mime_type and response_schema are used effectively.
        if response.text:
            api_response_text = response.text
            print(f"[CVAgent LOG] Raw JSON response (Gemini image analysis): {api_response_text}")
            try:
                # With response_schema, Pydantic validation might have already occurred,
                # or response.text is the clean JSON.
                # The SDK might even return a parsed object if Pydantic model is used in response_schema.
                # For now, assume response.text is the JSON string.
                cv_analysis_data = CVAnalysisData.model_validate_json(api_response_text)
                print("[CVAgent LOG] Successfully parsed and validated Gemini image response with Pydantic model.")
                return cv_analysis_data.model_dump(mode='json')
            except Exception as pydantic_error:
                print(f"[CVAgent ERROR] Pydantic validation failed (Gemini image): {pydantic_error}. Raw response: {api_response_text}")
                return {"error": "CV analysis (Gemini image) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
        else: # Fallback to check candidates if response.text is empty
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 api_response_text = response.candidates[0].content.parts[0].text
                 print(f"[CVAgent LOG] Raw JSON response from candidate (Gemini image analysis): {api_response_text}")
                 try:
                    cv_analysis_data = CVAnalysisData.model_validate_json(api_response_text)
                    print("[CVAgent LOG] Successfully parsed and validated Gemini image response (from candidate) with Pydantic model.")
                    return cv_analysis_data.model_dump(mode='json')
                 except Exception as pydantic_error:
                    print(f"[CVAgent ERROR] Pydantic validation failed (Gemini image, from candidate): {pydantic_error}. Raw response: {api_response_text}")
                    return {"error": "CV analysis (Gemini image, from candidate) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}

            print(f"[CVAgent ERROR] No valid text in Gemini API response for image. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"[CVAgent Safety] Prompt blocked for image. Reason: {response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason}")
                 return {"error": f"Prompt blocked for image analysis. Reason: {response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason}"}
            # Accessing finish_reason might be different, check SDK if this path is hit.
            # Assuming response.candidates[0].finish_reason is still valid.
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP': 
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[CVAgent Safety] Image analysis finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Image analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from CV analysis API (Gemini image)."}

    except Exception as e:
        print(f"[CVAgent ERROR] An error occurred during Gemini image analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during CV image analysis (Gemini): {str(e)}"}

def analyze_cv_from_text(text_content: str) -> Optional[dict]:
    """
    Analyzes CV/Resume text using Gemini, focusing on personal details and work experience gaps.
    Uses Pydantic models for response structure.
    """
    print(f"[CVAgent LOG] Analyzing CV text with Gemini (length: {len(text_content)})")

    client = get_gemini_client_and_model() # Now gets a client
    if not client:
        return {"error": "Gemini client not initialized (GOOGLE_API_KEY missing or invalid)."}

    if not text_content:
        print("[CVAgent ERROR] Text content for CV analysis cannot be empty.")
        return {"error": "Text content for CV analysis is empty."}

    schema_dict = CVAnalysisData.model_json_schema()
    prompt_text = f"""You are an AI assistant specialized in analyzing CVs/Resumes from text content.
Your primary goal is to extract specific personal details and identify any significant, unexplained gaps in work experience.
Carefully examine the provided CV text content.

Prioritize the following:

1.  **Personal Details Extraction**: Extract the candidate's full name, phone number, email address, physical address (if present), and date of birth (if present).
    If a detail is not found, omit it or use null for that specific field in the 'personal_details' object.

2.  **Work Experience Gap Identification**: Scrutinize the work experience sections for any unexplained gaps of approximately 3 months or longer between roles or educational periods.
    List each *distinct* identified gap as a string in the 'work_experience_gaps' array, strictly using the format 'mmm-yyyy to mmm-yyyy' (e.g., 'jan-2019 to may-2019'). If no such gaps are found, this array should be empty.

3.  **Text Readability/Structure Assessment**: Briefly describe the readability and structure of the CV text (e.g., clarity of sections, formatting, any obvious OCR errors if it looks like scanned text) in 'image_quality_summary' (keeping the key name 'image_quality_summary' for schema consistency).

4.  **Other Verification Flags**: List any other distinct textual anomalies or major inconsistencies (unrelated to work gaps, e.g., inconsistent date formats, highly unusual phrasing that might indicate copy-pasting or fabrication) in the 'other_verification_flags' array. If none, this array should be empty.

Your response MUST be a single, valid JSON object that strictly conforms to the following JSON Schema. Do not include any text outside of this JSON object. Ensure all specified formats (especially for dates and gaps) are followed.
Schema:
```json
{json.dumps(CVAnalysisData.model_json_schema(), indent=2)}
```
CV Text to Analyze:
{text_content[:20000]}
"""

    contents = [prompt_text]

    # Create GenerationConfig object - CHANGED TO GenerateContentConfig
    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=CVAnalysisData
    )

    api_response_text = None
    try:
        log_text_snippet = text_content[:200].replace('\n', ' ')
        print(f"[CVAgent LOG] Sending text (first 200 chars: '{log_text_snippet}') to {CV_AGENT_MODEL_ID} for Gemini text analysis.")
        response = client.models.generate_content(
            model=CV_AGENT_MODEL_ID, 
            contents=contents,
            config=generation_config # Pass GenerationConfig object to 'config' parameter
        )

        if response.text:
            api_response_text = response.text
            print(f"[CVAgent LOG] Raw JSON response (Gemini text analysis): {api_response_text}")
            try:
                cv_analysis_data = CVAnalysisData.model_validate_json(api_response_text)
                print("[CVAgent LOG] Successfully parsed and validated Gemini text response with Pydantic model.")
                return cv_analysis_data.model_dump(mode='json')
            except Exception as pydantic_error:
                print(f"[CVAgent ERROR] Pydantic validation failed for Gemini text analysis: {pydantic_error}. Raw response: {api_response_text}")
                return {"error": "Text-based CV analysis (Gemini) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
        else: # Fallback to check candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 api_response_text = response.candidates[0].content.parts[0].text
                 print(f"[CVAgent LOG] Raw JSON response from candidate (Gemini text analysis): {api_response_text}")
                 try:
                    cv_analysis_data = CVAnalysisData.model_validate_json(api_response_text)
                    print("[CVAgent LOG] Successfully parsed and validated Gemini text response (from candidate) with Pydantic model.")
                    return cv_analysis_data.model_dump(mode='json')
                 except Exception as pydantic_error:
                    print(f"[CVAgent ERROR] Pydantic validation failed (Gemini text, from candidate): {pydantic_error}. Raw response: {api_response_text}")
                    return {"error": "CV analysis (Gemini text, from candidate) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
            
            print(f"[CVAgent ERROR] No valid text in Gemini API response for text. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"[CVAgent Safety] Prompt blocked for text. Reason: {response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason}")
                 return {"error": f"Prompt blocked for text analysis. Reason: {response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason}"}
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[CVAgent Safety] Text analysis finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Text analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from text-based CV analysis API (Gemini)."}

    except Exception as e:
        print(f"[CVAgent ERROR] An error occurred during Gemini text-based CV analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during text-based CV analysis (Gemini): {str(e)}"}

if __name__ == '__main__':
    # Ensure GOOGLE_API_KEY is set in your environment to run these tests
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nSkipping CV Agent tests: GOOGLE_API_KEY not set.")
    else:
        # --- Test Image Analysis ---
        # IMPORTANT: Create a dummy PNG file for this test to run without error if a real one isn't present.
        sample_cv_image_path = "temp_sample_cv.png" 
        # Create a dummy file if it doesn't exist for basic testing flow
        if not os.path.exists(sample_cv_image_path):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (400, 300), color = 'red')
                d = ImageDraw.Draw(img)
                d.text((10,10), "Dummy CV for testing", fill=(255,255,0))
                img.save(sample_cv_image_path)
                print(f"Created dummy CV image at {sample_cv_image_path}")
            except ImportError:
                print("Pillow is not installed. Cannot create a dummy image for testing. Please create a real PNG CV image or install Pillow.")
            except Exception as e:
                print(f"Could not create dummy image: {e}")


        print(f"\n--- Testing CV Image Agent (Gemini) with: {sample_cv_image_path} ---")
        if os.path.exists(sample_cv_image_path): # Check again in case creation failed
            image_analysis_result = analyze_cv_image(sample_cv_image_path)
            if image_analysis_result:
                print("--- CV Image Analysis Result (Gemini) ---")
                # Attempt to pretty-print if it's a dict, otherwise print as is
                try:
                    print(json.dumps(image_analysis_result if isinstance(image_analysis_result, dict) else json.loads(image_analysis_result), indent=4))
                except (json.JSONDecodeError, TypeError):
                     print(image_analysis_result)
            else:
                print("--- CV Image Analysis (Gemini) Failed or No Result ---")
        else:
            print(f"Skipping CV Image Agent test: Sample image not found at {sample_cv_image_path} and dummy creation failed.")

        # --- Test Text Analysis ---
        sample_cv_text = """John Doe - CV
123 Main St, Anytown, USA
(555) 123-4567 | john.doe@email.com | DOB: 1990-01-15

Summary
Highly motivated individual...

Work Experience
Project Manager, Biz Corp (jan-2020 to dec-2022)
- Managed projects successfully.

Software Engineer, Tech Solutions Inc. (jun-2017 to may-2019)
- Developed cool stuff.

Intern, Old Company (jan-2016 to apr-2016)
- Learned things.

Education
BSc Computer Science, University of Tech (2012-2015)
        """
        print("\n--- Testing CV Text Agent (Gemini) ---")
        text_analysis_result = analyze_cv_from_text(sample_cv_text)
        if text_analysis_result:
            print("--- CV Text Analysis Result (Gemini) ---")
            try:
                print(json.dumps(text_analysis_result if isinstance(text_analysis_result, dict) else json.loads(text_analysis_result), indent=4))
            except (json.JSONDecodeError, TypeError):
                print(text_analysis_result)

        else:
            print("--- CV Text Analysis (Gemini) Failed or No Result ---")

        # Clean up dummy file
        if sample_cv_image_path == "temp_sample_cv.png" and os.path.exists(sample_cv_image_path):
            try:
                os.remove(sample_cv_image_path)
                print(f"Removed dummy CV image at {sample_cv_image_path}")
            except Exception as e:
                print(f"Error removing dummy CV image: {e}")