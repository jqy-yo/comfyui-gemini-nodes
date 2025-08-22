#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the updated GeminiStructuredOutput node with retry mechanism
"""

import sys
import os
import json

# Mock torch if not available
try:
    import torch
except ImportError:
    class MockTorch:
        def manual_seed(self, seed):
            pass
    sys.modules['torch'] = MockTorch()
    import torch

# Add the nodes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))

from GeminiStructuredOutput import GeminiStructuredOutput

def test_node():
    """Test the node with retry mechanism"""
    
    # Create node instance
    node = GeminiStructuredOutput()
    
    # Test parameters
    test_cases = [
        {
            "name": "Test 1: Simple prompt",
            "prompt": "你好",
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
            "name": "Test 2: Complex prompt",
            "prompt": "Create a detailed scene of a magical forest",
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
        }
    ]
    
    print("="*70)
    print("Testing GeminiStructuredOutput Node with Retry Mechanism")
    print("="*70)
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-"*40)
        
        try:
            # Call the node's generate_structured method
            result = node.generate_structured(
                prompt=test['prompt'],
                api_key='AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs',
                model='gemini-2.5-pro',
                output_mode='json_schema',
                schema_json=json.dumps(test['schema']),
                temperature=0.7,
                max_output_tokens=1024,
                seed=42,
                system_instructions="",
                enum_options="",
                top_p=0.95,
                top_k=64,
                property_ordering=""
            )
            
            # Result is now a tuple (structured_output, raw_json, debug_request, debug_response)
            structured_output, raw_json, debug_request, debug_response = result
            
            if "## Error" in structured_output or "## API Error" in structured_output:
                print(f"[FAILED]:")
                print(structured_output[:500])
            else:
                print(f"[SUCCESS]")
                print(f"Output: {structured_output[:200]}...")
                
            # Show debug information
            print("\n--- DEBUG: Request Sent ---")
            print(debug_request[:500] + "..." if len(debug_request) > 500 else debug_request)
            
            print("\n--- DEBUG: Response Received ---")
            print(debug_response[:500] + "..." if len(debug_response) > 500 else debug_response)
                
            # Show retry information from logs
            if hasattr(node, 'log_messages'):
                retry_logs = [log for log in node.log_messages if 'Retry' in log or 'attempt' in log]
                if retry_logs:
                    print("\nRetry Information:")
                    for log in retry_logs[:3]:  # Show first 3 retry-related logs
                        print(f"  - {log}")
                        
        except Exception as e:
            print(f"[EXCEPTION]: {str(e)}")
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)

if __name__ == "__main__":
    test_node()