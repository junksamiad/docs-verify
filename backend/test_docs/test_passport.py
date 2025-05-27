#!/usr/bin/env python3
"""
Quick test for passport.pdf with both classifiers
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai_doc_classifier import classify_image_document_type
from gemini_doc_classifier import classify_document_with_gemini

# Document types
DOC_TYPES = [
    'Bank Statement', 'Utility Bill', 'Council Tax Bill', 'CV', 'P45', 'P60', 
    'NI Letter', 'Pay Slip', 'Passport', 'Right To Work Share Code', 
    'DBS Certificate', 'Police Check Certificate', 'Drivers Licence', 
    'Training Certificate', 'Birth Certificate', 'HMRC Letter', 'DWP Letter', 
    'Other', 'Letter', 'Report', 'Invoice'
]

def main():
    print('🔍 Testing PASSPORT.PDF with both classifiers...')
    print('=' * 60)

    print('\n📄 Testing OpenAI Classifier...')
    openai_result = classify_image_document_type('passport.pdf', 'application/pdf', DOC_TYPES, 'auto')
    print(f'OpenAI Result: {openai_result}')

    print('\n📄 Testing Gemini Classifier...')
    gemini_result = classify_document_with_gemini('passport.pdf', 'application/pdf', DOC_TYPES)
    print(f'Gemini Result: {gemini_result}')

    print('\n📊 PASSPORT TEST SUMMARY')
    print('=' * 40)
    print(f'OpenAI:  {openai_result or "FAILED"}')
    print(f'Gemini:  {gemini_result or "FAILED"}')
    
    if openai_result and gemini_result:
        if openai_result == gemini_result:
            print('✅ Both classifiers agree!')
        else:
            print('⚠️  Classifiers disagree - this is normal')
    
    print('\n🎯 Passport test completed!')

if __name__ == '__main__':
    main() 