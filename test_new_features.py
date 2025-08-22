#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test new configuration features for GeminiStructuredOutput node
"""

import sys
import os
import json

# Mock torch
try:
    import torch
except ImportError:
    class MockTorch:
        def manual_seed(self, seed):
            pass
    sys.modules['torch'] = MockTorch()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))
from GeminiStructuredOutput import GeminiStructuredOutput

def test_penalty_features():
    """Test presence and frequency penalty features"""
    
    node = GeminiStructuredOutput()
    
    schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string", 
                "description": "Creative text generation"
            }
        },
        "required": ["prompt"]
    }
    
    test_configs = [
        {
            "name": "No penalties (default)",
            "prompt": "Write a creative story about a cat",
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        },
        {
            "name": "High presence penalty (diverse vocab)",
            "prompt": "Write a creative story about a cat",
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0
        },
        {
            "name": "High frequency penalty (avoid repetition)",
            "prompt": "Write a creative story about a cat",
            "presence_penalty": 0.0,
            "frequency_penalty": 1.5
        },
        {
            "name": "Stop sequences test",
            "prompt": "List 10 items",
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_sequences": "5.\n6."  # Stop after item 5
        }
    ]
    
    print("="*70)
    print("Testing New Configuration Features")
    print("="*70)
    
    for config in test_configs:
        print(f"\n{config['name']}")
        print("-"*40)
        
        try:
            result = node.generate_structured(
                prompt=config['prompt'],
                api_key='AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs',
                model='gemini-2.0-flash',  # Use stable model for testing
                output_mode='json_schema',
                schema_json=json.dumps(schema),
                temperature=0.8,
                max_output_tokens=200,
                seed=42,
                presence_penalty=config.get('presence_penalty', 0.0),
                frequency_penalty=config.get('frequency_penalty', 0.0),
                stop_sequences=config.get('stop_sequences', ''),
                response_logprobs=False,
                logprobs=0,
                use_json_schema=False
            )
            
            structured_output, _, _, _ = result
            
            if "## Error" not in structured_output:
                try:
                    parsed = json.loads(structured_output)
                    output = parsed.get('prompt', '')
                    print(f"[SUCCESS]")
                    print(f"Output preview: {output[:150]}...")
                    
                    # Count unique words to see penalty effects
                    words = output.lower().split()
                    unique_words = len(set(words))
                    total_words = len(words)
                    diversity = unique_words / total_words if total_words > 0 else 0
                    print(f"Word diversity: {diversity:.2f} ({unique_words}/{total_words} unique)")
                    
                except json.JSONDecodeError:
                    print(f"[FAILED] Invalid JSON")
            else:
                print(f"[FAILED] API Error")
                
        except Exception as e:
            print(f"[EXCEPTION] {str(e)[:100]}")
    
    print("\n" + "="*70)
    print("PENALTY EFFECTS SUMMARY")
    print("="*70)
    print("- Presence penalty increases vocabulary diversity")
    print("- Frequency penalty reduces word repetition")
    print("- Stop sequences can control output length")
    print("- These features help fine-tune generation behavior")

def test_response_json_schema():
    """Test responseJsonSchema for Gemini 2.5"""
    
    node = GeminiStructuredOutput()
    
    schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Test prompt"
            }
        },
        "required": ["prompt"]
    }
    
    print("\n" + "="*70)
    print("Testing responseJsonSchema (Gemini 2.5 only)")
    print("="*70)
    
    configs = [
        {"name": "Standard responseSchema", "use_json_schema": False},
        {"name": "New responseJsonSchema", "use_json_schema": True}
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-"*40)
        
        try:
            result = node.generate_structured(
                prompt="Hello world",
                api_key='AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs',
                model='gemini-2.5-pro',
                output_mode='json_schema',
                schema_json=json.dumps(schema),
                temperature=0.7,
                max_output_tokens=100,
                seed=42,
                use_json_schema=config['use_json_schema']
            )
            
            structured_output, _, _, _ = result
            
            if "## Error" not in structured_output:
                print(f"[SUCCESS] Method worked!")
            else:
                if "response_schema must not be set when response_json_schema" in structured_output:
                    print(f"[INFO] Conflict detected - feature may need SDK update")
                else:
                    print(f"[FAILED] Other error")
                    
        except Exception as e:
            print(f"[EXCEPTION] {str(e)[:100]}")

if __name__ == "__main__":
    test_penalty_features()
    test_response_json_schema()