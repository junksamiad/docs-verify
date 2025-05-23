import google.generativeai as genai
import os
import json
from PIL import Image # For reading image dimensions if needed, and basic ops

# --- Model Configuration ---
GEMINI_MODEL_ID = 'gemini-2.0-flash'  # The actual model ID used for the API call
GEMINI_FRIENDLY_NAME = f"Google Gemini ({GEMINI_MODEL_ID})" # Display name for the UI
# --- End Model Configuration ---

# Ensure your GOOGLE_API_KEY environment variable is set.
# You can set it using: export GOOGLE_API_KEY='your_gemini_api_key'
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# Simpler initialization if the key is globally configured for the SDK or picked up automatically

def classify_image_with_gemini(image_path: str, image_mime_type: str, document_types: list[str]) -> str | None:
    """
    Classifies the type of document shown in an image using the Google Gemini API.
    The image path should point to a processed image (e.g., PNG from PDF/HEIC conversion).

    Args:
        image_path: Path to the image file (e.g., PNG).
        image_mime_type: The MIME type of the image file (e.g., 'image/png').
        document_types: A list of strings representing the possible document types.

    Returns:
        The predicted document type as a string, or None if an error occurs.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return None
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    if not os.path.exists(image_path):
        print(f"Error [Gemini]: Image file not found at {image_path}")
        return None
    if not document_types:
        print("Error [Gemini]: Document types list cannot be empty.")
        return None
    if not image_mime_type or not image_mime_type.startswith("image/"):
        print(f"Error [Gemini]: Invalid or missing image MIME type: {image_mime_type}")
        return None # Gemini requires a valid image MIME type

    try:
        print(f"[Gemini LOG] Reading image file: {image_path} with MIME type: {image_mime_type}")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = {
            "mime_type": image_mime_type,
            "data": image_bytes
        }

        # Construct a prompt that asks for JSON output with a specific structure.
        # Gemini doesn't have a direct json_schema enforcement like OpenAI's newer APIs (as of last check for generate_content directly for vision),
        # so we rely on strong prompting for the JSON structure.
        # We will request a JSON string and parse it.
        
        # Define the expected JSON structure for the prompt
        # This helps the model understand what we want.
        example_json_output = json.dumps({"predicted_document_type": "ExampleType"})
        
        prompt = (
            f"Analyze the document in the provided image and classify its type. "
            f"Your response MUST be a single JSON object with exactly one key: 'predicted_document_type'. "
            f"The value for 'predicted_document_type' must be one of the following exact strings: {document_types}. "
            f"For example, your response should look like this: {example_json_output}. "
            f"Do not include any other text, explanations, or markdown formatting outside of this JSON object."
        )
        
        print(f"[Gemini LOG] Sending prompt to Gemini: {prompt[:200]}...") # Log a snippet of the prompt

        # Using a model that supports vision. gemini-1.5-flash is a good general choice.
        # The google_images.md doc refers to `gemini-2.0-flash` but that might be newer/experimental.
        # Let's use `gemini-1.5-flash` which is widely available and good for vision.
        model = genai.GenerativeModel(GEMINI_MODEL_ID) 
        
        # The contents should be a list [image_part, prompt] or [prompt, image_part]
        # Based on docs, text usually comes after the image for single image + text.
        response = model.generate_content([image_part, prompt])
        
        raw_response_text = response.text
        print(f"[Gemini LOG] Raw response text: {raw_response_text}")

        # Attempt to strip markdown code fences if present
        cleaned_response_text = raw_response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[len("```json"):].strip()
        if cleaned_response_text.startswith("```"):
            cleaned_response_text = cleaned_response_text[len("```"):].strip()
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-len("```")].strip()

        try:
            # The response.text should be the JSON string
            parsed_json = json.loads(cleaned_response_text)
            predicted_type = parsed_json.get("predicted_document_type")

            if predicted_type and predicted_type in document_types:
                return predicted_type
            elif predicted_type:
                print(f"Warning [Gemini]: Model returned a type '{predicted_type}' not in the allowed list: {document_types}. Raw JSON: {raw_response_text}")
                return None
            else:
                print(f"Error [Gemini]: 'predicted_document_type' key missing in JSON response: {raw_response_text}")
                return None
        except json.JSONDecodeError as e:
            print(f"Error [Gemini]: Decoding JSON response from API: {e}. Cleaned response: '{cleaned_response_text}'. Raw response: '{raw_response_text}'")
            return None
        except AttributeError:
            # This can happen if response.text is not available (e.g. safety blocks)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Error [Gemini]: Request was blocked. Reason: {response.prompt_feedback.block_reason}")
            else:
                 print(f"Error [Gemini]: Could not extract text from Gemini response. Full response: {response}")
            return None

    except Exception as e:
        print(f"An error occurred while calling the Gemini API or processing its response: {e}")
        # You might want to inspect `e` further if it's a google.api_core.exceptions type for more details
        return None

if __name__ == '__main__':
    # This is a placeholder for direct testing. 
    # Ensure GOOGLE_API_KEY is set in your environment.
    # Replace with a valid path to a PNG image (as conversion is handled upstream)
    
    # Example 1: Test with a placeholder path (update to a real PNG image path)
    sample_png_path = "path/to/your/converted_or_sample_image.png"
    sample_mime_type = "image/png"
    doc_types = ["Invoice", "Receipt", "Letter", "ID Card", "Unknown"]

    if os.environ.get("GOOGLE_API_KEY"):
        if os.path.exists(sample_png_path):
            print(f"\n--- Test Case 1: Classifying '{sample_png_path}' ---")
            classification = classify_image_with_gemini(sample_png_path, sample_mime_type, doc_types)
            print(f"Predicted Document Type (Gemini): {classification}")
        else:
            print(f"\nSkipping Test Case 1: Sample PNG file not found at '{sample_png_path}'. Please update the path.")
        
        # Example 2: Test with a different set of document types
        # (You can use the same image or a different one)
        if os.path.exists(sample_png_path): # Re-use image if it exists
            print(f"\n--- Test Case 2: Classifying '{sample_png_path}' with different categories ---")
            alt_doc_types = ["Passport", "Driver License", "Contract", "Report"]
            classification_alt = classify_image_with_gemini(sample_png_path, sample_mime_type, alt_doc_types)
            print(f"Predicted Document Type (Gemini, alt categories): {classification_alt}")

        # Example 3: Test with empty document types (should be handled)
        print("\n--- Test Case 3: Empty document types list (should return None) ---")
        classification_empty = classify_image_with_gemini(sample_png_path, sample_mime_type, [])
        print(f"Predicted Document Type (Gemini, empty types): {classification_empty}")
    else:
        print("\nSkipping direct tests for gemini_doc_classifier.py: GOOGLE_API_KEY not set.") 