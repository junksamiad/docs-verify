#!/usr/bin/env python3
"""
Test script for the refactored classifiers using the new OpenAI Responses API
"""

import os
import sys
from openai_doc_classifier import classify_image_document_type
from text_doc_classifier import classify_text_document_type, extract_text_from_document

# Document types to test with
DOC_TYPES = ['CV', 'Passport', 'Drivers Licence', 'Bank Statement', 'Invoice', 'Other']

def test_image_classifier():
    """Test the image classifier with various file types"""
    print("=" * 60)
    print("TESTING IMAGE CLASSIFIER (OpenAI Responses API)")
    print("=" * 60)
    
    test_files = [
        ('test_docs/test.png', 'image/png'),
        ('test_docs/test.jpg', 'image/jpeg'),
        ('test_docs/passport.pdf', 'application/pdf'),
        ('test_docs/cv.pdf', 'application/pdf'),
        ('test_docs/driving_licence.pdf', 'application/pdf'),
        ('test_docs/test.HEIC', 'image/heic'),
    ]
    
    for file_path, mime_type in test_files:
        if os.path.exists(file_path):
            print(f"\nTesting {file_path} ({mime_type})...")
            try:
                result = classify_image_document_type(file_path, mime_type, DOC_TYPES, 'auto')
                print(f"✅ Classification: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"⚠️  Skipping {file_path} - file not found")

def test_text_classifier():
    """Test the text classifier with text files"""
    print("\n" + "=" * 60)
    print("TESTING TEXT CLASSIFIER (OpenAI Responses API)")
    print("=" * 60)
    
    test_files = [
        ('test_docs/test.txt', '.txt'),
        ('test_docs/test.docx', '.docx'),
    ]
    
    for file_path, file_ext in test_files:
        if os.path.exists(file_path):
            print(f"\nTesting {file_path} ({file_ext})...")
            try:
                # Extract text first
                extracted_text = extract_text_from_document(file_path, file_ext)
                if extracted_text:
                    result = classify_text_document_type(extracted_text, DOC_TYPES)
                    if result:
                        print(f"✅ Classification: {result['predicted_document_type']}")
                        print(f"   Text length: {len(result['extracted_text'])} characters")
                    else:
                        print("❌ Classification failed")
                else:
                    print("❌ Text extraction failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"⚠️  Skipping {file_path} - file not found")

if __name__ == "__main__":
    print("Testing Refactored Classifiers with OpenAI Responses API")
    print("Make sure OPENAI_API_KEY is set in your environment")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    test_image_classifier()
    test_text_classifier()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60) 