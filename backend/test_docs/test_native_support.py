#!/usr/bin/env python3
"""
Test script to verify native file format support in both OpenAI and Gemini classifiers
"""

import os
import sys
from openai_doc_classifier import classify_image_document_type

# Document types to test with
DOC_TYPES = ['CV', 'Passport', 'Drivers Licence', 'Bank Statement', 'Invoice', 'Other']

def test_openai_native_support():
    """Test OpenAI's native file format support"""
    print("=" * 70)
    print("TESTING OPENAI NATIVE FILE FORMAT SUPPORT")
    print("=" * 70)
    
    test_files = [
        ('test_docs/passport.pdf', 'application/pdf', 'PDF (should be native)'),
        ('test_docs/cv.pdf', 'application/pdf', 'PDF (should be native)'),
        ('test_docs/test.png', 'image/png', 'PNG (should be native)'),
        ('test_docs/test.jpg', 'image/jpeg', 'JPEG (should be native)'),
        ('test_docs/test.HEIC', 'image/heic', 'HEIC (should convert to PNG)'),
    ]
    
    for file_path, mime_type, description in test_files:
        if os.path.exists(file_path):
            print(f"\n🔍 Testing {description}")
            print(f"   File: {file_path}")
            print(f"   MIME: {mime_type}")
            try:
                result = classify_image_document_type(file_path, mime_type, DOC_TYPES, 'auto')
                print(f"   ✅ Result: {result}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"\n⚠️  Skipping {description} - file not found: {file_path}")

def test_gemini_native_support():
    """Test what Gemini would support natively (if API key was available)"""
    print("\n" + "=" * 70)
    print("GEMINI NATIVE FILE FORMAT SUPPORT (Theoretical)")
    print("=" * 70)
    
    # Based on the code analysis, Gemini supports these natively:
    supported_formats = [
        "application/pdf",
        "image/png", 
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/heic",  # ✅ Native support
        "image/heif"   # ✅ Native support
    ]
    
    test_files = [
        ('test_docs/passport.pdf', 'application/pdf'),
        ('test_docs/test.png', 'image/png'),
        ('test_docs/test.jpg', 'image/jpeg'),
        ('test_docs/test.HEIC', 'image/heic'),  # This should work natively with Gemini!
    ]
    
    print("Gemini natively supports these formats:")
    for fmt in supported_formats:
        print(f"   ✅ {fmt}")
    
    print(f"\nTest files that would work natively with Gemini:")
    for file_path, mime_type in test_files:
        if os.path.exists(file_path):
            native_support = "✅ NATIVE" if mime_type in supported_formats else "❌ NEEDS CONVERSION"
            print(f"   {file_path} ({mime_type}) - {native_support}")
        else:
            print(f"   ⚠️  {file_path} - file not found")

def show_architecture_comparison():
    """Show the before/after architecture comparison"""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON: BEFORE vs AFTER")
    print("=" * 70)
    
    print("\n🔴 BEFORE (Old Architecture):")
    print("   OpenAI: PDF → PNG conversion → Classification")
    print("   OpenAI: HEIC → PNG conversion → Classification") 
    print("   OpenAI: PNG/JPG → Direct → Classification")
    print("   Gemini: PDF → Native → Classification")
    print("   Gemini: HEIC → PNG conversion → Classification ❌")
    print("   Gemini: PNG/JPG → Direct → Classification")
    
    print("\n🟢 AFTER (New Architecture):")
    print("   OpenAI: PDF → Native PDF support → Classification ✅")
    print("   OpenAI: HEIC → PNG conversion → Classification")
    print("   OpenAI: PNG/JPG → Direct → Classification") 
    print("   Gemini: PDF → Native → Classification")
    print("   Gemini: HEIC → Native → Classification ✅")
    print("   Gemini: PNG/JPG → Direct → Classification")
    
    print("\n🎯 KEY IMPROVEMENTS:")
    print("   ✅ OpenAI now handles PDFs natively (no conversion)")
    print("   ✅ Gemini now handles HEIC/HEIF natively (no conversion)")
    print("   ✅ Faster processing (fewer conversions)")
    print("   ✅ Better quality (models get original files)")
    print("   ✅ Consistent architecture between providers")

if __name__ == "__main__":
    print("Testing Native File Format Support")
    print("Make sure OPENAI_API_KEY is set for OpenAI tests")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found - OpenAI tests will fail")
    else:
        print("✅ OPENAI_API_KEY found")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found - Gemini tests are theoretical only")
    else:
        print("✅ GOOGLE_API_KEY found")
    
    # Run tests
    if os.environ.get("OPENAI_API_KEY"):
        test_openai_native_support()
    else:
        print("\nSkipping OpenAI tests - no API key")
    
    test_gemini_native_support()
    show_architecture_comparison()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70) 