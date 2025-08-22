#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final stability test for improved GeminiStructuredOutput node
"""

import sys
import os
import json
import time

# Mock torch if not available
try:
    import torch
except ImportError:
    class MockTorch:
        def manual_seed(self, seed):
            pass
    sys.modules['torch'] = MockTorch()

# Add the nodes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))

from GeminiStructuredOutput import GeminiStructuredOutput

def run_stability_test():
    """Run 10 consecutive tests to check stability"""
    
    node = GeminiStructuredOutput()
    
    test_prompts = [
        "你好",
        "Create a beautiful landscape",
        "A futuristic city",
        "Abstract art",
        "Nature scene",
        "Portrait photography",
        "Minimalist design",
        "Fantasy world",
        "Science fiction",
        "Traditional art"
    ]
    
    schema = {
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
    print("FINAL STABILITY TEST - GeminiStructuredOutput with Improvements")
    print("="*70)
    print("Model: gemini-2.5-pro")
    print("Tests: 10 different prompts")
    print("="*70)
    
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest #{i}: {prompt[:30]}...")
        test_start = time.time()
        
        try:
            result = node.generate_structured(
                prompt=prompt,
                api_key='AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs',
                model='gemini-2.5-pro',
                output_mode='json_schema',
                schema_json=json.dumps(schema),
                temperature=0.7,
                max_output_tokens=1024,
                seed=42 + i,  # Different seed for each test
                system_instructions="",
                enum_options="",
                top_p=0.95,
                top_k=64,
                property_ordering=""
            )
            
            structured_output, _, _, _ = result
            test_time = time.time() - test_start
            
            if "## Error" not in structured_output and "## API Error" not in structured_output:
                try:
                    parsed = json.loads(structured_output)
                    output_preview = parsed.get('prompt', '')[:50]
                    print(f"  [SUCCESS] {test_time:.1f}s - Output: {output_preview}...")
                    results.append({"status": "SUCCESS", "time": test_time})
                except:
                    print(f"  [FAILED] {test_time:.1f}s - Invalid JSON")
                    results.append({"status": "FAILED", "time": test_time})
            else:
                print(f"  [FAILED] {test_time:.1f}s - API Error")
                
                # Check if fallback was attempted
                if hasattr(node, 'log_messages'):
                    if any("fallback" in log.lower() for log in node.log_messages):
                        print(f"    (Fallback was attempted)")
                
                results.append({"status": "FAILED", "time": test_time})
                
        except Exception as e:
            test_time = time.time() - test_start
            print(f"  [EXCEPTION] {test_time:.1f}s - {str(e)[:50]}")
            results.append({"status": "EXCEPTION", "time": test_time})
        
        # Small delay between tests
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    exceptions = sum(1 for r in results if r["status"] == "EXCEPTION")
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful} ({successful*10}%)")
    print(f"Failed: {failed} ({failed*10}%)")
    print(f"Exceptions: {exceptions} ({exceptions*10}%)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time: {total_time/len(results):.1f}s")
    
    if successful > 0:
        success_times = [r['time'] for r in results if r['status'] == 'SUCCESS']
        print(f"Average success time: {sum(success_times)/len(success_times):.1f}s")
    
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if successful >= 9:
        print("[EXCELLENT] 90%+ success rate")
    elif successful >= 7:
        print("[GOOD] 70%+ success rate")
    elif successful >= 5:
        print("[MODERATE] 50%+ success rate")
    else:
        print("[NEEDS IMPROVEMENT] <50% success rate")

if __name__ == "__main__":
    run_stability_test()