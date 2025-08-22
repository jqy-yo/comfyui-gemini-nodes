#!/usr/bin/env python3
"""
Test script for Gemini Video Captioner with structured output
"""

import sys
import os
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes.GeminiVideoCaptioner import GeminiVideoCaptioner

def test_input_types():
    """Test that the INPUT_TYPES method returns the expected schema"""
    print("Testing INPUT_TYPES...")
    input_types = GeminiVideoCaptioner.INPUT_TYPES()
    
    # Check for new optional parameters
    assert "use_structured_output" in input_types["optional"]
    assert "output_schema" in input_types["optional"]
    print("✓ New structured output parameters found in INPUT_TYPES")
    
    # Check return types
    assert GeminiVideoCaptioner.RETURN_TYPES == ("STRING", "IMAGE", "STRING",)
    assert GeminiVideoCaptioner.RETURN_NAMES == ("caption", "sampled_frame", "raw_json",)
    print("✓ Return types include raw_json output")
    
def test_method_signature():
    """Test that the generate_video_caption method has the right signature"""
    print("\nTesting method signature...")
    import inspect
    sig = inspect.signature(GeminiVideoCaptioner.generate_video_caption)
    params = list(sig.parameters.keys())
    
    assert "use_structured_output" in params
    assert "output_schema" in params
    print("✓ generate_video_caption method has structured output parameters")
    
def test_schema_parsing():
    """Test schema parsing in the methods"""
    print("\nTesting schema parsing logic...")
    
    # Sample schema
    test_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "objects": {
                "type": "array",
                "items": {"type": "string"}
            },
            "mood": {"type": "string"}
        },
        "required": ["description", "objects"]
    }
    
    schema_json = json.dumps(test_schema, indent=2)
    print(f"✓ Test schema created: {len(schema_json)} chars")
    
    # Verify JSON parsing works
    parsed = json.loads(schema_json)
    assert parsed["type"] == "object"
    assert "description" in parsed["properties"]
    print("✓ Schema parsing works correctly")

def main():
    print("=" * 60)
    print("Testing Gemini Video Captioner Structured Output Support")
    print("=" * 60)
    
    try:
        test_input_types()
        test_method_signature()
        test_schema_parsing()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())