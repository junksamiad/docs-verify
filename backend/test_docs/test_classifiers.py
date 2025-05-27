#!/usr/bin/env python3
"""
Smoke test script for both PDF classifiers
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from openai_doc_classifier import classify_image_document_type
from gemini_doc_classifier import classify_document_with_gemini

# Test document types from your system
DOC_TYPES = [
    'Bank Statement', 'Utility Bill', 'Council Tax Bill', 'CV', 'P45', 'P60', 
    'NI Letter', 'Pay Slip', 'Passport', 'Right To Work Share Code', 
    'DBS Certificate', 'Police Check Certificate', 'Drivers Licence', 
    'Training Certificate', 'Birth Certificate', 'HMRC Letter', 'DWP Letter', 
    'Other', 'Letter', 'Report', 'Invoice'
]

PDF_FILE = 'test.pdf'
MIME_TYPE = 'application/pdf'

def test_openai_classifier():
    """Test OpenAI classifier (converts PDF to PNG first)"""
    print('🔍 Testing OpenAI Classifier with test.pdf...')
    print('=' * 50)
    
    if not os.path.exists(PDF_FILE):
        print(f'❌ Error: {PDF_FILE} not found!')
        return None
    
    try:
        result = classify_image_document_type(PDF_FILE, MIME_TYPE, DOC_TYPES, 'auto')
        print(f'✅ OpenAI Classification Result: {result}')
        return result
    except Exception as e:
        print(f'❌ OpenAI Classifier Error: {e}')
        return None

def test_gemini_classifier():
    """Test Gemini classifier (handles PDF directly)"""
    print('\n🔍 Testing Gemini Classifier with test.pdf...')
    print('=' * 50)
    
    if not os.path.exists(PDF_FILE):
        print(f'❌ Error: {PDF_FILE} not found!')
        return None
    
    try:
        result = classify_document_with_gemini(PDF_FILE, MIME_TYPE, DOC_TYPES)
        print(f'✅ Gemini Classification Result: {result}')
        return result
    except Exception as e:
        print(f'❌ Gemini Classifier Error: {e}')
        return None

def main():
    """Run both classifier tests"""
    print('🚀 Starting PDF Classifier Smoke Tests')
    print('=' * 60)
    
    # Test OpenAI classifier
    openai_result = test_openai_classifier()
    
    # Test Gemini classifier  
    gemini_result = test_gemini_classifier()
    
    # Summary
    print('\n📊 SMOKE TEST SUMMARY')
    print('=' * 60)
    print(f'OpenAI Result:  {openai_result or "FAILED"}')
    print(f'Gemini Result:  {gemini_result or "FAILED"}')
    
    if openai_result and gemini_result:
        if openai_result == gemini_result:
            print('✅ Both classifiers agree!')
        else:
            print('⚠️  Classifiers disagree - this is normal and expected')
    
    print('\n🎯 Test completed!')

if __name__ == '__main__':
    main() 