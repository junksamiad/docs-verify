import os
import json
from typing import List, Optional, Literal

from google import genai
from google.genai.types import Part, GenerateContentConfig, Blob

from pydantic import BaseModel, Field

# Initialize client (will pick up GOOGLE_API_KEY from env)
client: Optional[genai.Client] = None
if os.environ.get("GOOGLE_API_KEY"):
    try:
        client = genai.Client()
        print("[BankStatementAgent LOG] genai.Client() initialized.")
    except Exception as e:
        print(f"[BankStatementAgent ERROR] Failed to initialize genai.Client(): {e}")
        client = None
else:
    print("Error [BankStatementAgent]: GOOGLE_API_KEY environment variable not set. genai.Client() will not be initialized.")

# --- Pydantic Models for Bank Statement Analysis Response ---

class TransactionItem(BaseModel):
    date: Optional[str] = Field(None, description="Date of the transaction (ISO-8601 format: YYYY-MM-DDTHH:mm:ssZ)")
    description: Optional[str] = Field(None, description="Full transaction description as it appears on the statement")
    amount: Optional[float] = Field(None, description="The absolute value of the transaction amount (always positive)")
    direction: Optional[Literal["paid_in", "paid_out"]] = Field(None, description="Direction of the transaction")
    balance_after: Optional[float] = Field(None, description="Account balance immediately after this transaction")
    category: Optional[Literal[
        "essential_home",
        "essential_household",
        "non_essential_household",
        "salary_income",
        "non_essential_entertainment",
        "gambling",
        "cash_withdrawal",
        "bank_transfer",
        "unknown"
    ]] = Field(None, description="Transaction category")

class BankStatementUK(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the account holder")
    bank_provider: Optional[str] = Field(None, description="Name of the bank (e.g., Barclays, HSBC, Lloyds)")
    sort_code: Optional[str] = Field(None, description="Account sort code (e.g., \"XX-XX-XX\")")
    account_number: Optional[str] = Field(None, description="Account number (e.g., \"XXXXXXXX\")")
    address: Optional[str] = Field(None, description="Full postal address associated with the account holder on the statement")
    statement_start_date: Optional[str] = Field(None, description="Start date of the statement period (ISO-8601 format: YYYY-MM-DDTHH:mm:ssZ)")
    statement_end_date: Optional[str] = Field(None, description="End date of the statement period (ISO-8601 format: YYYY-MM-DDTHH:mm:ssZ)")
    total_paid_in: Optional[float] = Field(None, description="Total amount paid into the account during the statement period")
    total_paid_out: Optional[float] = Field(None, description="Total amount paid out of the account during the statement period")
    transactions: List[TransactionItem] = Field(default_factory=list)

# --- Model Configuration ---
BANK_STATEMENT_AGENT_MODEL_ID = 'gemini-2.5-pro-preview-05-06'
# --- End Model Configuration ---

def analyze_bank_statement_image(image_path: str) -> Optional[dict]:
    """
    Analyzes a Bank Statement image using Gemini.
    Uses Pydantic models for response structure.
    Assumes image_path is a PNG image.
    """
    print(f"[BankStatementAgent LOG] Analyzing Bank Statement image with Gemini: {image_path}")
    
    global client
    if not client:
        print("Error [BankStatementAgent]: Gemini client not initialized. Cannot proceed for image analysis.")
        return {"error": "Gemini API key not configured."}

    if not os.path.exists(image_path):
        print(f"[BankStatementAgent ERROR] Image file not found at {image_path}")
        return {"error": f"Image file not found: {image_path}"}

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        # Manual Part construction for image data
        image_part = Part(
            inline_data=Blob(
                data=image_bytes,
                mime_type="image/png" 
            )
        )
    except Exception as e:
        print(f"[BankStatementAgent ERROR] Error reading image file {image_path}: {e}")
        return {"error": f"Could not read image file: {e}"}

    prompt_text_content = f'''# ROLE: AI Data Extraction Engine (UK Bank Statements)

# TASK:
You are an expert AI data-extraction engine specializing in UK bank statements. Your task is to accurately parse the attached UK bank statement image and return ONLY a single, valid JSON object conforming precisely to the schema defined below. You must extract all specified header information and meticulously list *every* transaction, assigning a category to each one based on the provided rules.

# INPUT:
- UK bank statement image (provided directly, ignore any file_id placeholders in this text if not applicable to direct image input)

# OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- NO markdown formatting (e.g., \`\`\`json ... \`\`\`).
- NO introductory text, explanations, apologies, or conversational filler.
- Adhere strictly to the JSON schema provided below.

# JSON SCHEMA:
```json
{json.dumps(BankStatementUK.model_json_schema(), indent=2)}
```

# TRANSACTION PROCESSING RULES:
1.  **Order:** Transactions MUST be ordered chronologically by date (ascending).
2.  **Completeness:** Include EVERY single transaction listed within the statement period. Do NOT summarise or omit any transactions.
3.  **Exclusions:** NEVER include 'balance brought forward' or 'balance carried forward' entries as transactions.
4.  **Amounts:** Ensure the amount field is always a positive number. Use the direction field to indicate if it was paid in or out.
5.  **Dates:** Format all dates (statement_start_date, statement_end_date, transaction.date) strictly according to the ISO-8601 standard (YYYY-MM-DDTHH:mm:ssZ). If time is not available, use T00:00:00Z.

# TRANSACTION CATEGORIZATION RULES:
- Assign EXACTLY ONE category to EVERY transaction.
- The category value MUST be one of the following strings:
    - essential_home
    - essential_household
    - non_essential_household
    - salary_income
    - non_essential_entertainment
    - gambling
    - cash_withdrawal
    - bank_transfer
    - unknown

- Use the following definitions and hints to assign the correct category. Prioritize specific keywords found in the transaction description. Consider the direction and typical amount patterns.

    1.  **essential_home**:
        - Description hints: Rent, Mortgage, Ground Rent, Service Charge. Look for recurring, often large, payments to letting agents or mortgage providers.
        - Direction: MUST be paid_out.
    2.  **essential_household**:
        - Description hints: Council Tax, Water (e.g., Thames Water, Severn Trent), Electricity/Gas (e.g., E.ON, British Gas, EDF, Octopus), Internet/Broadband (e.g., BT, Virgin Media, TalkTalk, Sky Broadband), TV Licence, Phone/Mobile (e.g., O2, Vodafone, EE, Three).
        - Direction: MUST be paid_out.
    3.  **non_essential_household**:
        - Description hints: Subscription services like Sky TV, Netflix, Spotify, Disney+, Apple Music, Amazon Prime Video. Services like Cleaners, Gardeners, Window Cleaners.
        - Direction: MUST be paid_out.
    4.  **salary_income**:
        - Description hints: Salary, Wages, Payroll, BACS payments from known employers. Look for large, regular incoming payments.
        - Direction: MUST be paid_in.
    5.  **non_essential_entertainment**:
        - Description hints: Restaurants, Pubs, Bars, Cafes, Coffee Shops, Cinema, Theatre, Concerts, Ticketmaster, Eventbrite, Uber, Bolt, Taxis, Just Eat, Deliveroo, Uber Eats, Takeaways. Spending in shops typically associated with leisure (e.g., bookshops, hobby shops - unless clearly household goods).
        - Direction: MUST be paid_out.
    6.  **gambling**:
        - Description hints: Bet, Betting, Bet365, William Hill, Ladbrokes, Coral, Paddy Power, Sky Bet, Casino, Lottery, Scratchcards, Gambling sites.
        - Direction: Can be paid_in (winnings) or paid_out (stakes).
    7.  **cash_withdrawal**:
        - Description hints: ATM, Cash, Withdrawal, WDL, Counter WDL, Cash Advance, Post Office cash.
        - Direction: MUST be paid_out.
    8.  **bank_transfer**:
        - Description hints: Transfer, TRF, TFR, FP, Faster Payment, BACS (if not clearly salary/bill), CHAPS, Payment to/from another individual\'s name (if discernible), Transfers between own accounts (if statement shows this).
        - Direction: Can be paid_in or paid_out.
    9.  **unknown**:
        - Use this category ONLY if the transaction does not clearly fit into any of the categories above based on its description, amount, and direction. This includes ambiguous store purchases, unclear online payments, bank fees/charges (unless specified elsewhere), interest payments/receipts, etc.
# FINAL INSTRUCTION:
Execute the extraction and categorization process diligently. Ensure the final output is solely the valid JSON object adhering to all specified rules and the schema.
Return exactly one JSON document that passes strict JSON.parse validation.
'''
    prompt_part = Part(text=prompt_text_content)

    contents = [image_part, prompt_part]

    generation_config_obj = GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=BankStatementUK,
        max_output_tokens=32000
    )

    api_response_text = None
    try:
        print(f"[BankStatementAgent LOG] Sending image request to Gemini model: {BANK_STATEMENT_AGENT_MODEL_ID} for bank statement analysis")
        response = client.models.generate_content(
            model=BANK_STATEMENT_AGENT_MODEL_ID,
            contents=contents,
            config=generation_config_obj
        )
        
        # Log finish reason if candidates exist
        if response.candidates:
            try:
                finish_reason_name = response.candidates[0].finish_reason.name
                print(f"[BankStatementAgent LOG] Gemini API call (image) finish_reason: {finish_reason_name}")
            except AttributeError:
                print("[BankStatementAgent WARNING] Could not retrieve finish_reason name from response candidate (image).")
        
        if response.text:
            api_response_text = response.text
            print(f"[BankStatementAgent LOG] Raw JSON response (Bank Statement analysis): {api_response_text}")
            try:
                bs_analysis_data = BankStatementUK.model_validate_json(api_response_text)
                print("[BankStatementAgent LOG] Successfully parsed and validated Gemini bank statement response with Pydantic model.")
                return bs_analysis_data.model_dump(mode='json')
            except Exception as pydantic_error:
                print(f"[BankStatementAgent ERROR] Pydantic validation failed (Bank Statement): {pydantic_error}. Raw response: {api_response_text}")
                return {"error": "Bank statement analysis (Gemini) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
        else: # Fallback to check candidates if response.text is empty
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 api_response_text = response.candidates[0].content.parts[0].text
                 print(f"[BankStatementAgent LOG] Raw JSON response from candidate (Bank Statement analysis): {api_response_text}")
                 try:
                    bs_analysis_data = BankStatementUK.model_validate_json(api_response_text)
                    print("[BankStatementAgent LOG] Successfully parsed and validated Gemini bank statement response (from candidate) with Pydantic model.")
                    return bs_analysis_data.model_dump(mode='json')
                 except Exception as pydantic_error:
                    print(f"[BankStatementAgent ERROR] Pydantic validation failed (Bank Statement, from candidate): {pydantic_error}. Raw response: {api_response_text}")
                    return {"error": "Bank statement analysis (Gemini, from candidate) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}

            print(f"[BankStatementAgent ERROR] No valid text in Gemini API response for bank statement. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 reason = response.prompt_feedback.block_reason
                 reason_name = reason.name if hasattr(reason, 'name') else str(reason)
                 print(f"[BankStatementAgent Safety] Prompt blocked for bank statement. Reason: {reason_name}")
                 return {"error": f"Prompt blocked for bank statement analysis. Reason: {reason_name}"}
            
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP': 
                # This log might be redundant if already logged above, but good for error path clarity
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[BankStatementAgent Safety] Bank statement analysis (image) finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Bank statement analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from bank statement analysis API (Gemini)." }

    except Exception as e:
        print(f"[BankStatementAgent ERROR] An error occurred during Gemini bank statement analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during bank statement analysis (Gemini): {str(e)}", "raw_response_snippet": api_response_text[:500] if api_response_text else 'N/A'}

def analyze_bank_statement_pdf(pdf_path: str) -> Optional[dict]:
    """
    Analyzes a Bank Statement PDF using Gemini.
    Uses Pydantic models for response structure.
    """
    print(f"[BankStatementAgent LOG] Analyzing Bank Statement PDF with Gemini: {pdf_path}")
    
    global client
    if not client:
        print("Error [BankStatementAgent]: Gemini client not initialized. Cannot proceed for PDF analysis.")
        return {"error": "Gemini API key not configured."}

    if not os.path.exists(pdf_path):
        print(f"[BankStatementAgent ERROR] PDF file not found at {pdf_path}")
        return {"error": f"PDF file not found: {pdf_path}"}

    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        # Manual Part construction for PDF data
        pdf_part = Part(
            inline_data=Blob(
                data=pdf_bytes,
                mime_type="application/pdf"
            )
        ) 
    except Exception as e:
        print(f"[BankStatementAgent ERROR] Error reading PDF file {pdf_path}: {e}")
        return {"error": f"Could not read PDF file: {e}"}

    prompt_text_content = f'''# ROLE: AI Data Extraction Engine (UK Bank Statements)

# TASK:
You are an expert AI data-extraction engine specializing in UK bank statements. Your task is to accurately parse the attached UK bank statement (which could be an image or a PDF document) and return ONLY a single, valid JSON object conforming precisely to the schema defined below. You must extract all specified header information and meticulously list *every* transaction, assigning a category to each one based on the provided rules.

# INPUT:
- UK bank statement (image or PDF, provided directly)

# OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- NO markdown formatting (e.g., \`\`\`json ... \`\`\`).
- NO introductory text, explanations, apologies, or conversational filler.
- Adhere strictly to the JSON schema provided below.

# JSON SCHEMA:
```json
{json.dumps(BankStatementUK.model_json_schema(), indent=2)}
```

# TRANSACTION PROCESSING RULES:
1.  **Order:** Transactions MUST be ordered chronologically by date (ascending).
2.  **Completeness:** Include EVERY single transaction listed within the statement period. Do NOT summarise or omit any transactions.
3.  **Exclusions:** NEVER include 'balance brought forward' or 'balance carried forward' entries as transactions.
4.  **Amounts:** Ensure the amount field is always a positive number. Use the direction field to indicate if it was paid in or out.
5.  **Dates:** Format all dates (statement_start_date, statement_end_date, transaction.date) strictly according to the ISO-8601 standard (YYYY-MM-DDTHH:mm:ssZ). If time is not available, use T00:00:00Z.

# TRANSACTION CATEGORIZATION RULES:
- Assign EXACTLY ONE category to EVERY transaction.
- The category value MUST be one of the following strings:
    - essential_home
    - essential_household
    - non_essential_household
    - salary_income
    - non_essential_entertainment
    - gambling
    - cash_withdrawal
    - bank_transfer
    - unknown

- Use the following definitions and hints to assign the correct category. Prioritize specific keywords found in the transaction description. Consider the direction and typical amount patterns.

    1.  **essential_home**:
        - Description hints: Rent, Mortgage, Ground Rent, Service Charge. Look for recurring, often large, payments to letting agents or mortgage providers.
        - Direction: MUST be paid_out.
    2.  **essential_household**:
        - Description hints: Council Tax, Water (e.g., Thames Water, Severn Trent), Electricity/Gas (e.g., E.ON, British Gas, EDF, Octopus), Internet/Broadband (e.g., BT, Virgin Media, TalkTalk, Sky Broadband), TV Licence, Phone/Mobile (e.g., O2, Vodafone, EE, Three).
        - Direction: MUST be paid_out.
    3.  **non_essential_household**:
        - Description hints: Subscription services like Sky TV, Netflix, Spotify, Disney+, Apple Music, Amazon Prime Video. Services like Cleaners, Gardeners, Window Cleaners.
        - Direction: MUST be paid_out.
    4.  **salary_income**:
        - Description hints: Salary, Wages, Payroll, BACS payments from known employers. Look for large, regular incoming payments.
        - Direction: MUST be paid_in.
    5.  **non_essential_entertainment**:
        - Description hints: Restaurants, Pubs, Bars, Cafes, Coffee Shops, Cinema, Theatre, Concerts, Ticketmaster, Eventbrite, Uber, Bolt, Taxis, Just Eat, Deliveroo, Uber Eats, Takeaways. Spending in shops typically associated with leisure (e.g., bookshops, hobby shops - unless clearly household goods).
        - Direction: MUST be paid_out.
    6.  **gambling**:
        - Description hints: Bet, Betting, Bet365, William Hill, Ladbrokes, Coral, Paddy Power, Sky Bet, Casino, Lottery, Scratchcards, Gambling sites.
        - Direction: Can be paid_in (winnings) or paid_out (stakes).
    7.  **cash_withdrawal**:
        - Description hints: ATM, Cash, Withdrawal, WDL, Counter WDL, Cash Advance, Post Office cash.
        - Direction: MUST be paid_out.
    8.  **bank_transfer**:
        - Description hints: Transfer, TRF, TFR, FP, Faster Payment, BACS (if not clearly salary/bill), CHAPS, Payment to/from another individual\'s name (if discernible), Transfers between own accounts (if statement shows this).
        - Direction: Can be paid_in or paid_out.
    9.  **unknown**:
        - Use this category ONLY if the transaction does not clearly fit into any of the categories above based on its description, amount, and direction. This includes ambiguous store purchases, unclear online payments, bank fees/charges (unless specified elsewhere), interest payments/receipts, etc.
# FINAL INSTRUCTION:
Execute the extraction and categorization process diligently. Ensure the final output is solely the valid JSON object adhering to all specified rules and the schema.
Return exactly one JSON document that passes strict JSON.parse validation.
'''
    prompt_part = Part(text=prompt_text_content)

    contents = [pdf_part, prompt_part]

    generation_config_obj = GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=BankStatementUK,
        max_output_tokens=32000
    )

    api_response_text = None
    try:
        print(f"[BankStatementAgent LOG] Sending PDF request to Gemini model: {BANK_STATEMENT_AGENT_MODEL_ID} for bank statement analysis")
        response = client.models.generate_content(
            model=BANK_STATEMENT_AGENT_MODEL_ID,
            contents=contents,
            config=generation_config_obj
        )
        
        # Log finish reason if candidates exist
        if response.candidates:
            try:
                finish_reason_name = response.candidates[0].finish_reason.name
                print(f"[BankStatementAgent LOG] Gemini API call (PDF) finish_reason: {finish_reason_name}")
            except AttributeError:
                print("[BankStatementAgent WARNING] Could not retrieve finish_reason name from response candidate (PDF).")
        
        if response.text:
            api_response_text = response.text
            print(f"[BankStatementAgent LOG] Raw JSON response (Bank Statement PDF analysis): {api_response_text}")
            try:
                bs_analysis_data = BankStatementUK.model_validate_json(api_response_text)
                print("[BankStatementAgent LOG] Successfully parsed and validated Gemini bank statement PDF response with Pydantic model.")
                return bs_analysis_data.model_dump(mode='json')
            except Exception as pydantic_error:
                print(f"[BankStatementAgent ERROR] Pydantic validation failed (Bank Statement PDF): {pydantic_error}. Raw response: {api_response_text}")
                return {"error": "Bank statement PDF analysis (Gemini) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
        else: # Fallback to check candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 api_response_text = response.candidates[0].content.parts[0].text
                 print(f"[BankStatementAgent LOG] Raw JSON response from candidate (Bank Statement PDF analysis): {api_response_text}")
                 try:
                    bs_analysis_data = BankStatementUK.model_validate_json(api_response_text)
                    print("[BankStatementAgent LOG] Successfully parsed and validated Gemini bank statement PDF response (from candidate) with Pydantic model.")
                    return bs_analysis_data.model_dump(mode='json')
                 except Exception as pydantic_error:
                    print(f"[BankStatementAgent ERROR] Pydantic validation failed (Bank Statement PDF, from candidate): {pydantic_error}. Raw response: {api_response_text}")
                    return {"error": "Bank statement PDF analysis (Gemini, from candidate) response failed Pydantic validation.", "details": str(pydantic_error), "raw_response": api_response_text}
            
            print(f"[BankStatementAgent ERROR] No valid text in Gemini API response for bank statement PDF. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 reason = response.prompt_feedback.block_reason
                 reason_name = reason.name if hasattr(reason, 'name') else str(reason)
                 print(f"[BankStatementAgent Safety] Prompt blocked for bank statement PDF. Reason: {reason_name}")
                 return {"error": f"Prompt blocked for bank statement PDF analysis. Reason: {reason_name}"}
            if response.candidates and response.candidates[0].finish_reason.name != 'STOP':
                # This log might be redundant if already logged above, but good for error path clarity
                reason_name = response.candidates[0].finish_reason.name
                safety_ratings_str = str(response.candidates[0].safety_ratings)
                print(f"[BankStatementAgent Safety] Bank statement PDF analysis finished with reason: {reason_name}. Details: {safety_ratings_str}")
                return {"error": f"Bank statement PDF analysis did not complete successfully. Finish Reason: {reason_name}"}
            return {"error": "No content from bank statement PDF analysis API (Gemini)." }

    except Exception as e:
        print(f"[BankStatementAgent ERROR] An error occurred during Gemini bank statement PDF analysis: {e}. API response text: {api_response_text if api_response_text else 'N/A'}")
        return {"error": f"Exception during bank statement PDF analysis (Gemini): {str(e)}", "raw_response_snippet": api_response_text[:500] if api_response_text else 'N/A'}

if __name__ == '__main__':
    # Ensure GOOGLE_API_KEY is set in your environment to run these tests
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nSkipping Bank Statement Agent tests: GOOGLE_API_KEY not set.")
    else:
        sample_bs_image_path = "temp_sample_bank_statement.png" 
        if not os.path.exists(sample_bs_image_path):
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (700, 1000), color = 'white') # A bit wider for typical statements
                d = ImageDraw.Draw(img)
                try:
                    # Try to use a common font, fall back to default
                    font = ImageFont.truetype("arial.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()
                
                text_lines = [
                    "My Bank PLC - Bank Statement",
                    "1 Financial Square, London, UK, E1 4XX",
                    " ",
                    "MR JOHN DOE",
                    "123 APPLESEED LANE",
                    "ANYTOWN, AN1 2SH",
                    " ",
                    "Account Number: 12345678",
                    "Sort Code: 01-02-03",
                    "Statement Date: 01 Feb 2024 to 29 Feb 2024",
                    "Currency: GBP",
                    " ",
                    "Date        Description                     Paid Out     Paid In      Balance",
                    "02 Feb 2024 SALARY - COMPANY X                             1500.00     3500.00",
                    "05 Feb 2024 UTILITY BILL - ENERGY CO        100.00                      3400.00",
                    "10 Feb 2024 GROCERIES - SUPERMART            75.50                      3324.50",
                    "15 Feb 2024 TRANSFER TO SAVINGS             200.00                      3124.50",
                    " ",
                    "Total Paid In: 1500.00",
                    "Total Paid Out: 375.50"
                ]
                y_position = 20
                for line in text_lines:
                    d.text((20, y_position), line, fill=(0,0,0), font=font)
                    y_position += 18 # Adjust line spacing
                
                img.save(sample_bs_image_path)
                print(f"Created dummy bank statement image at {sample_bs_image_path}")
            except ImportError:
                print("Pillow is not installed. Cannot create a dummy image for testing. Please create a real PNG bank statement image or install Pillow.")
            except Exception as e:
                print(f"Could not create dummy image: {e}")

        print(f"\n--- Testing Bank Statement Agent (Gemini) with: {sample_bs_image_path} ---")
        if os.path.exists(sample_bs_image_path):
            bs_analysis_result = analyze_bank_statement_image(sample_bs_image_path)
            if bs_analysis_result:
                print("--- Bank Statement Analysis Result (Gemini) ---")
                try:
                    # The result 'bs_analysis_result' should already be a dict from model_dump(mode='json')
                    print(json.dumps(bs_analysis_result, indent=4))
                except (json.JSONDecodeError, TypeError) as e: # Added specific exception var e
                     print(f"Error printing JSON: {e}")
                     print(bs_analysis_result)
            else:
                print("--- Bank Statement Analysis (Gemini) Failed or No Result ---")
        else:
            print(f"Skipping Bank Statement Agent test: Sample image not found at {sample_bs_image_path} and dummy creation failed.")

        # Clean up dummy file
        if sample_bs_image_path == "temp_sample_bank_statement.png" and os.path.exists(sample_bs_image_path):
            try:
                os.remove(sample_bs_image_path)
                print(f"Removed dummy bank statement image at {sample_bs_image_path}")
            except Exception as e:
                print(f"Error removing dummy bank statement image: {e}") 