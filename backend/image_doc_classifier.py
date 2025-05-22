from openai import OpenAI
import base64
import os
import mimetypes
import json # Added for parsing JSON response

# Ensure your OPENAI_API_KEY environment variable is set.
# You can set it using: export OPENAI_API_KEY='your_api_key'
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# If the API key is set globally in the environment, this simpler initialization works:
client = OpenAI()

def encode_image_to_base64(image_path: str) -> str | None:
    """
    Encodes an image file to a Base64 string.
    Also infers the MIME type from the file extension.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image"):
        print(f"Error: Could not determine a valid image MIME type for {image_path}")
        # Defaulting to jpeg if unsure, API might handle it.
        # Supported types: PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), Non-animated GIF (.gif)
        mime_type = "image/jpeg" 

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def classify_image_document_type(image_path: str, document_types: list[str], detail: str = "auto") -> str | None:
    """
    Classifies the type of document shown in an image using the OpenAI Responses API
    with vision capabilities and structured JSON output.

    Args:
        image_path: Path to the image file.
        document_types: A list of strings representing the possible document types.
        detail: The level of detail for the model to use ('auto', 'low', 'high').

    Returns:
        The predicted document type as a string, or None if an error occurs.
    """
    if not image_path:
        print("Error: Image path cannot be empty.")
        return None
    if not document_types:
        print("Error: Document types list cannot be empty.")
        return None

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    valid_details = ["auto", "low", "high"]
    if detail not in valid_details:
        print(f"Warning: Invalid detail level '{detail}'. Defaulting to 'auto'. Valid options are: {valid_details}")
        detail = "auto"
    
    # Define the JSON schema for the desired output structure
    # The model should output a JSON object with a key "predicted_document_type"
    # whose value is one of the provided document_types.
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
        f"Analyze the document in the provided image and classify its type. "
        f"You must respond with a JSON object that strictly adheres to the following schema: {json.dumps(output_schema)}. "
        f"The 'predicted_document_type' field in your JSON response must be one of these exact values: {', '.join(document_types)}."
    )

    try:
        response = client.responses.create(
            model="gpt-4o-2024-08-06", # Updated model supporting json_schema
            input=[
                {
                    "role": "user", # Changed role to user for combined image/text input
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "image_url": base64_image,
                            "detail": detail,
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "document_classification_output",
                    "schema": output_schema,
                    "strict": True
                }
            }
        )

        # response.output_text will contain the JSON string
        if response.output_text:
            try:
                parsed_json = json.loads(response.output_text)
                predicted_type = parsed_json.get("predicted_document_type")

                if predicted_type and predicted_type in document_types:
                    return predicted_type
                elif predicted_type:
                    print(f"Warning: Model returned a type '{predicted_type}' not in the allowed list: {document_types}. Raw JSON: {response.output_text}")
                    return None # Or handle as an error/unexpected response
                else:
                    print(f"Error: 'predicted_document_type' key missing in JSON response: {response.output_text}")
                    return None
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response from API: {e}. Raw response: {response.output_text}")
                return None
        elif response.output and response.output[0].content and response.output[0].content[0].type == "refusal":
             print(f"API refused the request: {response.output[0].content[0].refusal}")
             return None
        else:
            print(f"No output_text or refusal found in API response. Full response: {response}")
            return None

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API or processing its response: {e}")
        return None

if __name__ == "__main__":
    # Example Usage:
    # Make sure your OPENAI_API_KEY is set in your environment.
    # e.g., export OPENAI_API_KEY="your_actual_api_key"
    
    # IMPORTANT: Replace with a valid path to an image file for testing.
    # For example, an image of an invoice, a receipt, a letter, etc.
    sample_image_path = "path/to/your/sample_document_image.jpg" 
    detail_level = "auto" # or "low" or "high"
    
    possible_doc_types = ["Invoice", "Receipt", "Letter", "Form", "ID Card", "Presentation Slide", "Passport", "Driver License"]

    if not os.path.exists(sample_image_path):
        print(f"Skipping example: Sample image file not found at '{sample_image_path}'.")
        print("Please update 'sample_image_path' in the script with a valid image path to run the example.")
    else:
        print(f"Classifying document type for image: '{sample_image_path}' with detail: '{detail_level}'")
        classification = classify_image_document_type(sample_image_path, possible_doc_types, detail_level)
        print(f"Predicted Document Type (JSON): {classification}\n")

    # Example with a non-existent image path (should be handled gracefully)
    print("Testing with non-existent image path:")
    classification_non_existent = classify_image_document_type("path/to/non_existent_image.png", possible_doc_types, "high")
    print(f"Predicted Document Type (non-existent image): {classification_non_existent}\n")

    # Example with empty document types list
    if os.path.exists(sample_image_path): # Only run if sample image exists for this test
        print(f"Testing with empty document types list for image: '{sample_image_path}'")
        classification_empty_types = classify_image_document_type(sample_image_path, [], "low")
        print(f"Predicted Document Type (empty types list): {classification_empty_types}\n")
    else:
        print(f"Skipping empty document types test: Sample image file not found at '{sample_image_path}'.") 