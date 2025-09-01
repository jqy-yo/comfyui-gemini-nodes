#!/usr/bin/env python3
"""
Test script to diagnose Gemini API issues with single images
"""

import os
import sys
import json
import google.generativeai as genai
from PIL import Image
import io
import base64

def test_gemini_api(api_key, image_path=None):
    """Test Gemini API with different configurations"""
    
    print("=" * 60)
    print("Gemini API Test Script")
    print("=" * 60)
    
    # Test models in order of likelihood to work
    test_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro", 
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash"
    ]
    
    # Configure API
    genai.configure(api_key=api_key)
    
    # Test 1: Simple text generation
    print("\n1. Testing basic text generation...")
    for model_name in test_models:
        try:
            print(f"   Testing {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello, World!'")
            print(f"   ✓ {model_name} works for text generation")
            print(f"     Response: {response.text[:50]}...")
            break
        except Exception as e:
            error_msg = str(e)
            if "500" in error_msg:
                print(f"   ✗ {model_name}: 500 Internal Error - Model may not be available")
            elif "404" in error_msg:
                print(f"   ✗ {model_name}: 404 Not Found - Model doesn't exist")
            else:
                print(f"   ✗ {model_name}: {error_msg[:100]}")
    
    # Test 2: Image/Video processing (if file provided)
    if image_path and os.path.exists(image_path):
        file_ext = os.path.splitext(image_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        
        if is_video:
            print(f"\n2. Testing video processing with: {image_path}")
            print(f"   Video format: {file_ext}")
            
            # Get file size
            file_size = os.path.getsize(image_path)
            print(f"   File size: {file_size / (1024*1024):.2f} MB")
            
            # Test video with different models
            for model_name in test_models:
                try:
                    print(f"\n   Testing {model_name} with video...")
                    model = genai.GenerativeModel(model_name)
                    
                    # Upload the video file
                    print(f"   Uploading video file...")
                    video_file = genai.upload_file(image_path)
                    print(f"   Video uploaded successfully")
                    
                    # Create the prompt with video
                    response = model.generate_content([
                        "Describe this video in one sentence.",
                        video_file
                    ])
                    
                    print(f"   ✓ {model_name} works for video processing!")
                    print(f"     Response: {response.text[:100]}...")
                    
                    # Clean up
                    video_file.delete()
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    if "500" in error_msg:
                        print(f"   ✗ {model_name}: 500 Internal Error")
                        print("     This is the error you're experiencing!")
                        print("     Possible causes:")
                        print("     - Model doesn't support video processing")
                        print("     - Video format/codec not supported")
                        print("     - Video too large or too long")
                        print("     - Try converting to WebM format")
                    elif "404" in error_msg:
                        print(f"   ✗ {model_name}: Model not found")
                    elif "unsupported" in error_msg.lower():
                        print(f"   ✗ {model_name}: Video format not supported")
                    else:
                        print(f"   ✗ {model_name}: {error_msg[:150]}")
        else:
            print(f"\n2. Testing image processing with: {image_path}")
            
            # Load and prepare image
            img = Image.open(image_path)
            print(f"   Image size: {img.size}")
            print(f"   Image format: {img.format}")
        
        # Convert to JPEG if needed
        if img.format != 'JPEG':
            print("   Converting to JPEG...")
            buffer = io.BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
        else:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
        
        print(f"   Image size in bytes: {len(img_bytes)}")
        
        for model_name in test_models:
            try:
                print(f"\n   Testing {model_name} with image...")
                model = genai.GenerativeModel(model_name)
                
                # Create the prompt with image
                response = model.generate_content([
                    "Describe this image in one sentence.",
                    img
                ])
                
                print(f"   ✓ {model_name} works for image processing!")
                print(f"     Response: {response.text[:100]}...")
                
                # Test with different parameters
                print(f"   Testing with custom parameters...")
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=100
                )
                
                response = model.generate_content(
                    ["What is in this image?", img],
                    generation_config=generation_config
                )
                print(f"   ✓ Custom parameters work!")
                
                break
                
            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg:
                    print(f"   ✗ {model_name}: 500 Internal Error")
                    print("     Possible causes:")
                    print("     - Model doesn't support images")
                    print("     - Image too large or in unsupported format")
                    print("     - Regional restrictions")
                elif "404" in error_msg:
                    print(f"   ✗ {model_name}: Model not found")
                else:
                    print(f"   ✗ {model_name}: {error_msg[:150]}")
    
    # Test 3: List available models
    print("\n3. Listing available models...")
    try:
        models = genai.list_models()
        print("   Available models:")
        for model in models:
            if 'gemini' in model.name.lower():
                print(f"   - {model.name}")
                if hasattr(model, 'supported_generation_methods'):
                    print(f"     Supports: {', '.join(model.supported_generation_methods)}")
    except Exception as e:
        print(f"   ✗ Could not list models: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    # Get API key from environment or command line
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        print("Usage: python test_gemini_api.py <API_KEY> [image_path]")
        print("Or set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_gemini_api(api_key, image_path)