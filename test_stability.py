#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stability test for Gemini Structured Output
Runs 10 consecutive tests with the same prompt
"""

import os
import sys
import json
import time
from google import genai
from google.genai import types

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

def test_once(client, prompt, schema, max_retries=3):
    """Run a single test with retry mechanism"""
    start_time = time.time()
    last_error = None
    retry_count = 0
    
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=schema
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt],
                config=config
            )
            
            # Get result
            result = None
            if hasattr(response, 'parsed') and response.parsed:
                result = response.parsed
            elif hasattr(response, 'text') and response.text:
                result = response.text
                try:
                    result = json.loads(result)
                except:
                    pass
            
            # If we got a result, return it
            if result:
                elapsed_time = time.time() - start_time
                return result, elapsed_time, None, retry_count
            
            # No result, but no exception - treat as empty response
            last_error = "Empty response from API"
            retry_count += 1
            
            # Wait before retry if not the last attempt
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # Progressive backoff: 1s, 2s, 3s
                
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            
            # Wait before retry if not the last attempt
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
    
    # All retries failed
    elapsed_time = time.time() - start_time
    return None, elapsed_time, last_error, retry_count

def main():
    # API key
    api_key = 'AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs'
    
    # Test configuration
    test_prompt = "你好"
    test_schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "English prompt for flux model image generation"
            }
        },
        "required": ["prompt"]
    }
    
    print("="*70)
    print("STABILITY TEST - Gemini-2.5-pro Structured Output")
    print("="*70)
    print(f"Model: gemini-2.5-pro")
    print(f"Test prompt: {test_prompt}")
    print(f"Schema: {json.dumps(test_schema, indent=2)}")
    print("="*70)
    
    # Create client
    client = genai.Client(api_key=api_key)
    
    # Run 10 tests
    results = []
    total_time = 0
    successful = 0
    failed = 0
    total_retries = 0
    
    print("\nRunning 10 consecutive tests with retry mechanism (max 3 attempts each)...\n")
    
    for i in range(10):
        print(f"Test #{i+1}:", end=" ")
        result, elapsed, error, retries = test_once(client, test_prompt, test_schema)
        
        if result:
            successful += 1
            status = "SUCCESS"
            output = result.get('prompt', 'No prompt field') if isinstance(result, dict) else str(result)
            # Truncate long outputs for display
            if len(output) > 100:
                display_output = output[:97] + "..."
            else:
                display_output = output
            
            retry_info = f" (retries: {retries})" if retries > 0 else ""
            print(f"[{status}] Time: {elapsed:.2f}s{retry_info}")
            print(f"  Output: {display_output}")
        else:
            failed += 1
            status = "FAILED"
            print(f"[{status}] Time: {elapsed:.2f}s (all {retries} attempts failed)")
            if error:
                print(f"  Error: {error[:100]}")
        
        total_time += elapsed
        total_retries += retries
        results.append({
            'test_num': i+1,
            'status': status,
            'time': elapsed,
            'result': result,
            'error': error,
            'retries': retries
        })
        
        # Small delay between requests to avoid rate limiting
        if i < 9:
            time.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tests: 10")
    print(f"Successful: {successful} ({successful*10}%)")
    print(f"Failed: {failed} ({failed*10}%)")
    print(f"Total retries needed: {total_retries}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per request: {total_time/10:.2f}s")
    
    if successful > 0:
        successful_times = [r['time'] for r in results if r['status'] == 'SUCCESS']
        print(f"Average successful request time: {sum(successful_times)/len(successful_times):.2f}s")
    
    # Show retry statistics
    tests_with_retries = [r for r in results if r['retries'] > 0]
    if tests_with_retries:
        print(f"\nRetry Statistics:")
        print(f"  Tests that needed retries: {len(tests_with_retries)}")
        for r in tests_with_retries:
            status_emoji = "✓" if r['status'] == 'SUCCESS' else "✗"
            print(f"    Test #{r['test_num']}: {r['retries']} retries - {status_emoji} {r['status']}")
    
    # Show all outputs
    print("\n" + "="*70)
    print("ALL OUTPUTS")
    print("="*70)
    for r in results:
        if r['status'] == 'SUCCESS' and r['result']:
            output = r['result'].get('prompt', 'No prompt field') if isinstance(r['result'], dict) else str(r['result'])
            print(f"\nTest #{r['test_num']} ({r['time']:.2f}s):")
            print(f"  {output}")
    
    # Stability assessment
    print("\n" + "="*70)
    print("STABILITY ASSESSMENT")
    print("="*70)
    if successful == 10:
        print("[EXCELLENT] 100% success rate - Very stable!")
    elif successful >= 8:
        print("[GOOD] {}% success rate - Mostly stable".format(successful*10))
    elif successful >= 5:
        print("[MODERATE] {}% success rate - Some instability".format(successful*10))
    else:
        print("[POOR] {}% success rate - Unstable, needs investigation".format(successful*10))

if __name__ == "__main__":
    main()