#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Gemini Structured Output
Tests different schema formats to identify compatibility issues
"""

import os
import sys
import json
from google import genai
from google.genai import types

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

def test_schema(api_key, schema, prompt, description=""):
    """Test a single schema configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Schema: {json.dumps(schema, indent=2)}")
    print(f"{'='*60}")
    
    try:
        client = genai.Client(api_key=api_key)
        
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=schema
        )
        
        # Test with gemini-2.5-pro as requested
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt],
            config=config
        )
        
        # Try different ways to get the result
        result = None
        
        # Method 1: parsed attribute
        if hasattr(response, 'parsed') and response.parsed:
            result = response.parsed
            print(f"[SUCCESS] Got result from response.parsed: {result}")
        
        # Method 2: text attribute
        elif hasattr(response, 'text') and response.text:
            result = response.text
            print(f"[SUCCESS] Got result from response.text: {result}")
        
        # Method 3: candidates
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            result = part.text
                            print(f"[SUCCESS] Got result from candidates: {result}")
                            break
        
        if result is None:
            print("[FAILED] No result - response returned empty")
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(f"  Finish reason: {candidate.finish_reason}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return None


def main():
    # Get API key from environment or use provided key
    api_key = os.environ.get('GOOGLE_API_KEY', 'AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs')
    
    if not api_key:
        print("No API key provided. Exiting.")
        return
    
    # Test prompt
    test_prompt = "Create a detailed prompt for generating an image of a serene mountain landscape at sunset"
    
    # Test different schema variations
    schemas = [
        {
            "description": "Original schema with description",
            "schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "English prompt for flux model image generation"
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "description": "Simple schema without description",
            "schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                },
                "required": ["prompt"]
            }
        },
        {
            "description": "Schema with prompt as optional",
            "schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                }
            }
        },
        {
            "description": "Minimal schema",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            }
        }
    ]
    
    print("\nTesting Gemini Structured Output with different schemas...")
    print(f"Model: gemini-2.5-pro")
    print(f"Test prompt: {test_prompt}")
    
    results = []
    for test_case in schemas:
        result = test_schema(
            api_key=api_key,
            schema=test_case["schema"],
            prompt=test_prompt,
            description=test_case["description"]
        )
        results.append({
            "description": test_case["description"],
            "success": result is not None,
            "result": result
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "[SUCCESS]" if r["success"] else "[FAILED]"
        print(f"{status} {r['description']}")
        if r["success"] and r["result"]:
            print(f"  Result preview: {str(r['result'])[:100]}...")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    if successful:
        print(f"[SUCCESS] Working schema format found!")
        print(f"  Best option: {successful[0]['description']}")
        print("\nUse this schema in your node:")
        for test_case in schemas:
            if test_case["description"] == successful[0]['description']:
                print(json.dumps(test_case["schema"], indent=2))
    else:
        print("[FAILED] No schema format worked. Possible issues:")
        print("  1. API key may not have access to structured output")
        print("  2. Model may not support structured output") 
        print("  3. Try using gemini-2.0-flash-thinking-exp")


if __name__ == "__main__":
    main()