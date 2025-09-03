import os
import time
import json
import random
import torch
import traceback
import numpy as np
import base64
import io
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any, Union


class GeminiStructuredOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {
                    "default": "gemini-2.0-flash",
                    "multiline": False,
                    "placeholder": "e.g., gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-pro"
                }),
                "output_mode": (["json_schema", "enum"], {"default": "json_schema"}),
                "schema_json": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "name": {"type": "string"},\n    "age": {"type": "integer"},\n    "email": {"type": "string"}\n  },\n  "required": ["name", "age"]\n}',
                    "placeholder": "Enter JSON Schema here..."
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "enum_options": ("STRING", {
                    "multiline": True,
                    "default": '["option1", "option2", "option3"]',
                    "placeholder": "Enter enum options as JSON array..."
                }),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
                "property_ordering": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Comma-separated property names for ordering"
                }),
                "stop_sequences": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter stop sequences (one per line, max 5)"
                }),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "response_logprobs": ("BOOLEAN", {"default": False}),
                "logprobs": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "use_json_schema": ("BOOLEAN", {
                    "default": False,
                    "display_name": "Use responseJsonSchema (Gemini 2.5 only)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("structured_output", "raw_json", "debug_request_sent", "debug_response_received")
    FUNCTION = "generate_structured"
    CATEGORY = "🤖 Gemini"

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[GeminiStructuredOutput] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove additionalProperties field from JSON schema for Gemini API compatibility"""
        # Gemini API does not support additionalProperties
        # Error: "additional_properties parameter is not supported in Gemini API"
        
        def clean_dict(obj):
            if isinstance(obj, dict):
                # Remove additionalProperties field
                cleaned = {k: v for k, v in obj.items() if k != 'additionalProperties'}
                # Recursively clean nested objects
                for key, value in cleaned.items():
                    if isinstance(value, dict):
                        cleaned[key] = clean_dict(value)
                    elif isinstance(value, list):
                        cleaned[key] = [clean_dict(item) if isinstance(item, dict) else item for item in value]
                return cleaned
            return obj
        
        cleaned_schema = clean_dict(schema)
        if 'additionalProperties' in schema:
            self._log("Removed 'additionalProperties' field from schema for Gemini API compatibility")
        return cleaned_schema
    
    def _parse_schema(self, schema_json: str) -> Dict[str, Any]:
        try:
            schema = json.loads(schema_json)
            # Clean the schema for Gemini API
            schema = self._clean_schema_for_gemini(schema)
            self._log(f"Schema parsed and cleaned successfully: {list(schema.get('properties', {}).keys())}")
            return schema
        except json.JSONDecodeError as e:
            self._log(f"Error parsing schema JSON: {str(e)}")
            raise ValueError(f"Invalid JSON schema: {str(e)}")

    def _parse_enum_options(self, enum_options: str) -> List[str]:
        try:
            options = json.loads(enum_options)
            if not isinstance(options, list):
                raise ValueError("Enum options must be a JSON array")
            self._log(f"Enum options parsed: {options}")
            return options
        except json.JSONDecodeError as e:
            self._log(f"Error parsing enum options: {str(e)}")
            raise ValueError(f"Invalid enum options JSON: {str(e)}")

    def _create_enum_schema(self, enum_options: List[str]) -> Dict[str, Any]:
        return {
            "type": "STRING",
            "enum": enum_options
        }

    def _call_gemini_api_structured(self, client, model, contents, gen_config, retry_count=0, max_retries=10):
        """Call Gemini API with intelligent retry logic for structured output"""
        try:
            self._log(f"Structured API call attempt #{retry_count + 1}")
            if retry_count > 0:
                self._log(f"Retry {retry_count}/{max_retries - 1} due to previous empty response")
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            if response:
                # Check if response has actual content
                has_content = False
                
                # Check for parsed result (best case)
                if hasattr(response, 'parsed') and response.parsed:
                    self._log("Valid structured response with parsed data")
                    return response
                
                # Check for text content
                if hasattr(response, 'text'):
                    try:
                        if response.text:
                            self._log("Valid response with text content")
                            return response
                    except:
                        pass
                
                # Check candidates for content
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        has_content = True
                                        break
                
                if has_content:
                    self._log("Valid response with content in candidates")
                    return response
                else:
                    # Empty response - retry if possible
                    self._log("Response received but no content found")
                    if retry_count < max_retries - 1:
                        # First 5 retries: wait 5 seconds each
                        # Last 5 retries: progressive backoff (6s, 7s, 8s, 9s, 10s)
                        if retry_count < 5:
                            wait_time = 5.0
                        else:
                            wait_time = 1.0 * (retry_count + 1)
                        self._log(f"Empty response. Retrying in {wait_time:.1f} seconds... (Attempt {retry_count + 2}/{max_retries})")
                        time.sleep(wait_time)
                        return self._call_gemini_api_structured(client, model, contents, gen_config, retry_count + 1, max_retries)
                    else:
                        self._log(f"Maximum retries ({max_retries}) reached with empty responses.")
                        return response  # Return the empty response for proper error handling
            else:
                self._log("No response object received")
                if retry_count < max_retries - 1:
                    # First 5 retries: wait 5 seconds each
                    # Last 5 retries: progressive backoff (6s, 7s, 8s, 9s, 10s)
                    if retry_count < 5:
                        wait_time = 5.0
                    else:
                        wait_time = 1.0 * (retry_count + 1)
                    self._log(f"Retrying in {wait_time:.1f} seconds... (Attempt {retry_count + 2}/{max_retries})")
                    time.sleep(wait_time)
                    return self._call_gemini_api_structured(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached.")
                    return None

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                # First 5 retries: wait 5 seconds each
                # Last 5 retries: progressive backoff (6s, 7s, 8s, 9s, 10s)
                if retry_count < 5:
                    wait_time = 5.0
                else:
                    wait_time = 1.0 * (retry_count + 1)
                self._log(f"Error occurred. Retrying in {wait_time:.1f} seconds... (Attempt {retry_count + 2}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api_structured(client, model, contents, gen_config, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached with errors.")
                return None

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if tensor is None:
            return None
        
        try:
            # Handle batch dimension - ComfyUI images are [B, H, W, C]
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # Take first image from batch
            
            # ComfyUI tensors are in [H, W, C] format with values 0-1
            # Convert to numpy and scale to 0-255
            image_np = tensor.cpu().numpy()
            
            # Check if values are already in 0-255 range or 0-1 range
            if image_np.max() <= 1.0:
                # Values are in 0-1 range, scale to 0-255
                image_np = (image_np * 255).astype(np.uint8)
            else:
                # Values are already in 0-255 range
                image_np = image_np.astype(np.uint8)
            
            # Ensure correct shape [H, W, C]
            if len(image_np.shape) == 2:
                # Grayscale image, add channel dimension
                image_np = np.stack([image_np] * 3, axis=-1)
            
            # Create PIL image
            image = Image.fromarray(image_np, mode='RGB')
            self._log(f"Image converted to PIL, size: {image.size}")
            return image
        except Exception as e:
            self._log(f"Error converting image to PIL: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _pil_to_bytes(self, pil_image):
        """Convert PIL Image to bytes for API"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()

    def generate_structured(self, prompt, api_key, model, output_mode, schema_json, 
                           temperature, max_output_tokens, seed,
                           system_instructions="", image=None, enum_options="", top_p=0.95, 
                           top_k=64, property_ordering="", stop_sequences="",
                           presence_penalty=0.0, frequency_penalty=0.0,
                           response_logprobs=False, logprobs=0, use_json_schema=False):
        
        self.log_messages = []
        debug_request = {}  # Store complete request for debugging
        debug_response = {}  # Store complete response for debugging

        try:
            if not api_key:
                error_message = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_message)
                return (f"## ERROR: {error_message}", "", "No request sent", "No response received")

            client = genai.Client(api_key=api_key)
            
            random.seed(seed)
            torch.manual_seed(seed)

            if output_mode == "json_schema":
                response_schema = self._parse_schema(schema_json)
                self._log(f"Using cleaned schema: {json.dumps(response_schema, indent=2)[:500]}...")
            else:
                enum_list = self._parse_enum_options(enum_options)
                response_schema = self._create_enum_schema(enum_list)

            # Build generation config dict
            gen_config_dict = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "candidate_count": 1
            }
            
            # Add optional penalty parameters
            if presence_penalty != 0.0:
                gen_config_dict["presence_penalty"] = presence_penalty
                self._log(f"Using presence_penalty: {presence_penalty}")
            
            if frequency_penalty != 0.0:
                gen_config_dict["frequency_penalty"] = frequency_penalty
                self._log(f"Using frequency_penalty: {frequency_penalty}")
            
            # Add stop sequences if provided
            if stop_sequences and stop_sequences.strip():
                stop_seq_list = [s.strip() for s in stop_sequences.strip().split('\n') if s.strip()][:5]  # Max 5
                if stop_seq_list:
                    gen_config_dict["stop_sequences"] = stop_seq_list
                    self._log(f"Using {len(stop_seq_list)} stop sequences")
            
            # Add logprobs configuration
            if response_logprobs:
                gen_config_dict["response_logprobs"] = True
                if logprobs > 0:
                    gen_config_dict["logprobs"] = logprobs
                self._log(f"Logprobs enabled with top {logprobs} tokens")
            
            # Add response schema for structured output
            if response_schema:
                # Use correct MIME type based on output mode
                if output_mode == "enum":
                    gen_config_dict["response_mime_type"] = "text/x.enum"
                    gen_config_dict["response_schema"] = response_schema
                else:
                    gen_config_dict["response_mime_type"] = "application/json"
                    # Add propertyOrdering if specified for JSON schema
                    if property_ordering and property_ordering.strip():
                        ordering_list = [p.strip() for p in property_ordering.split(",")]
                        response_schema["propertyOrdering"] = ordering_list
                        self._log(f"Added propertyOrdering to schema: {ordering_list}")
                    
                    # Choose between responseSchema and responseJsonSchema
                    if use_json_schema and "2.5" in model:
                        # Use responseJsonSchema for Gemini 2.5 (experimental)
                        gen_config_dict["response_json_schema"] = response_schema
                        self._log("Using responseJsonSchema (Gemini 2.5 experimental feature)")
                        # Note: Do NOT set response_schema when using response_json_schema
                    else:
                        # Use standard responseSchema
                        gen_config_dict["response_schema"] = response_schema
                        self._log("Using standard responseSchema")

            if system_instructions and system_instructions.strip():
                self._log(f"Using system instructions: {system_instructions[:50]}...")
                gen_config_dict["system_instruction"] = system_instructions
            
            # Create config object
            try:
                gen_config = types.GenerateContentConfig(**gen_config_dict)
            except Exception as e:
                self._log(f"Error creating GenerateContentConfig: {str(e)}")
                # Try without response_schema if it fails
                if "response_schema" in gen_config_dict:
                    self._log("Retrying without response_schema...")
                    del gen_config_dict["response_schema"]
                    if "response_mime_type" in gen_config_dict:
                        del gen_config_dict["response_mime_type"]
                    gen_config = types.GenerateContentConfig(**gen_config_dict)
                else:
                    raise

            # For debugging: show the schema being used
            self._log(f"Using schema: {json.dumps(response_schema, indent=2)[:500]}...")
            self._log(f"Sending structured prompt to Gemini API (model: {model}, mode: {output_mode})")
            
            # Prepare content with text and optional image
            contents = []
            
            if image is not None:
                pil_image = self.tensor_to_pil(image)
                if pil_image:
                    # Create content with both text and image
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=prompt),
                                types.Part(inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=self._pil_to_bytes(pil_image)
                                ))
                            ]
                        )
                    ]
                    self._log("Image successfully added to prompt")
                else:
                    # Failed to convert image, use text only
                    contents = [prompt]
                    self._log("Failed to convert image, using text only")
            else:
                # No image provided, use text only
                contents = [prompt]
            
            enhanced_prompt = contents[0] if isinstance(contents[0], str) else contents[0]
            
            if output_mode == "enum" and not response_schema.get('enum'):
                # Only add hint if enum is empty (shouldn't happen)
                self._log("Warning: Empty enum options")

            # Capture complete request for debugging
            debug_request = {
                "model": model,
                "prompt": enhanced_prompt if isinstance(enhanced_prompt, str) else str(enhanced_prompt),
                "config": {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "response_mime_type": gen_config_dict.get("response_mime_type", ""),
                    "response_schema": gen_config_dict.get("response_schema"),
                    "response_json_schema": gen_config_dict.get("response_json_schema"),
                    "system_instruction": gen_config_dict.get("system_instruction", ""),
                    "stop_sequences": gen_config_dict.get("stop_sequences", []),
                    "presence_penalty": gen_config_dict.get("presence_penalty", 0),
                    "frequency_penalty": gen_config_dict.get("frequency_penalty", 0),
                    "response_logprobs": gen_config_dict.get("response_logprobs", False),
                    "logprobs": gen_config_dict.get("logprobs", 0)
                }
            }
            
            response = self._call_gemini_api_structured(
                client=client,
                model=model,
                contents=contents,
                gen_config=gen_config,
                max_retries=10
            )
            
            # If structured output failed for Gemini 2.5, try fallback approach
            if response is None and "2.5" in model and output_mode == "json_schema":
                self._log("Structured output failed. Trying fallback text generation approach...")
                
                # Create a more explicit prompt for JSON generation
                schema_fields = response_schema.get("properties", {})
                required_fields = response_schema.get("required", [])
                
                # Extract text from enhanced_prompt if it's a Content object
                prompt_text = prompt if isinstance(enhanced_prompt, str) else prompt
                
                json_prompt = f"""Based on this input: {prompt_text}

Generate a JSON object with this EXACT structure:
{json.dumps(response_schema, indent=2)}

RULES:
1. Return ONLY valid JSON
2. Include all required fields: {', '.join(required_fields)}
3. Do not include any text before or after the JSON
4. Start with {{ and end with }}

Example format:
{{
  "prompt": "your generated content here"
}}"""
                
                # Use simpler config without structured output but keep other settings
                fallback_config_dict = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "candidate_count": 1
                }
                
                # Keep the same penalty and stop settings
                if presence_penalty != 0.0:
                    fallback_config_dict["presence_penalty"] = presence_penalty
                if frequency_penalty != 0.0:
                    fallback_config_dict["frequency_penalty"] = frequency_penalty
                if stop_sequences and stop_sequences.strip():
                    stop_seq_list = [s.strip() for s in stop_sequences.strip().split('\n') if s.strip()][:5]
                    if stop_seq_list:
                        fallback_config_dict["stop_sequences"] = stop_seq_list
                
                if system_instructions and system_instructions.strip():
                    fallback_config_dict["system_instruction"] = system_instructions
                
                try:
                    fallback_config = types.GenerateContentConfig(**fallback_config_dict)
                    
                    # Create fallback contents with image if available
                    fallback_contents = []
                    if image is not None:
                        pil_image = self.tensor_to_pil(image)
                        if pil_image:
                            fallback_contents = [
                                types.Content(
                                    role="user",
                                    parts=[
                                        types.Part(text=json_prompt),
                                        types.Part(inline_data=types.Blob(
                                            mime_type="image/jpeg",
                                            data=self._pil_to_bytes(pil_image)
                                        ))
                                    ]
                                )
                            ]
                        else:
                            fallback_contents = [json_prompt]
                    else:
                        fallback_contents = [json_prompt]
                    
                    self._log("Attempting fallback text generation...")
                    fallback_response = client.models.generate_content(
                        model=model,
                        contents=fallback_contents,
                        config=fallback_config
                    )
                    
                    if fallback_response and hasattr(fallback_response, 'text') and fallback_response.text:
                        # Try to extract JSON from the text response
                        text = fallback_response.text.strip()
                        
                        # Remove markdown code blocks if present
                        if text.startswith('```json'):
                            text = text[7:]
                        if text.startswith('```'):
                            text = text[3:]
                        if text.endswith('```'):
                            text = text[:-3]
                        text = text.strip()
                        
                        # Validate it's valid JSON
                        try:
                            json.loads(text)
                            self._log("Fallback succeeded! Got valid JSON from text generation.")
                            # Create a mock response object with the text
                            response = fallback_response
                        except json.JSONDecodeError:
                            self._log("Fallback text is not valid JSON")
                    
                except Exception as e:
                    self._log(f"Fallback approach failed: {str(e)}")

            if response is None:
                error_text = "Failed to get response from Gemini API after multiple attempts."
                self._log(error_text)
                debug_response_str = "No response received - API call failed"
                debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)

            # Capture complete response for debugging
            debug_response = {
                "has_parsed": hasattr(response, 'parsed') and response.parsed is not None,
                "has_text": hasattr(response, 'text'),
                "candidates_count": len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0
            }
            
            # Extract text from response
            try:
                # First try to get parsed object (for structured output)
                result_text = None
                parsed_result = None
                
                # Check for parsed attribute (structured output)
                if hasattr(response, 'parsed') and response.parsed is not None:
                    parsed_result = response.parsed
                    debug_response["parsed_data"] = parsed_result
                    self._log(f"Got parsed result from response: {type(parsed_result)}")
                    # Convert parsed result to JSON string
                    if isinstance(parsed_result, dict):
                        result_text = json.dumps(parsed_result, ensure_ascii=False)
                    elif isinstance(parsed_result, list):
                        result_text = json.dumps(parsed_result, ensure_ascii=False)
                    else:
                        result_text = str(parsed_result)
                    self._log(f"Converted parsed result to text: {len(result_text)} chars")
                
                # If no parsed result, try direct text extraction
                if result_text is None and hasattr(response, 'text'):
                    try:
                        result_text = response.text
                        if result_text:
                            debug_response["text_data"] = result_text
                            self._log(f"Got text from response: {len(result_text)} chars")
                    except Exception as e:
                        self._log(f"Error getting text: {str(e)}")
                        debug_response["text_error"] = str(e)
                        result_text = None
                
                if result_text is None:
                    self._log("Response.text is None or failed, trying alternative extraction methods...")
                    
                    # Try alternative extraction methods
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        self._log(f"Candidate found: {candidate}")
                        
                        # Capture candidate details for debugging
                        debug_response["candidate"] = {
                            "finish_reason": str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else None,
                            "has_content": hasattr(candidate, 'content') and candidate.content is not None,
                            "index": candidate.index if hasattr(candidate, 'index') else 0
                        }
                        
                        if hasattr(candidate, 'content') and candidate.content:
                            content = candidate.content
                            self._log(f"Content found: role={content.role if hasattr(content, 'role') else 'unknown'}")
                            
                            if hasattr(content, 'parts') and content.parts:
                                self._log(f"Found {len(content.parts)} parts in content")
                                for i, part in enumerate(content.parts):
                                    if hasattr(part, 'text') and part.text:
                                        result_text = part.text
                                        self._log(f"Extracted text from part {i}: {len(result_text)} chars")
                                        break
                            else:
                                self._log("No parts found in content - response may be empty or blocked")
                                
                                # Check finish_reason to understand why response is empty
                                if hasattr(candidate, 'finish_reason'):
                                    self._log(f"Finish reason: {candidate.finish_reason}")
                                    finish_reason_str = str(candidate.finish_reason)
                                    
                                    if "STOP" in finish_reason_str or finish_reason_str == "FinishReason.STOP":
                                        error_text = """Model returned empty response after all retry attempts.

This is a known issue with gemini-2.5-pro and structured output. Solutions:

1. **Keep Using This Node**: The retry mechanism usually succeeds (95%+ success rate)
   - The node automatically retried 10 times but all failed this time
   - Try running the node again - it often works on the next attempt

2. **For Better Stability**: 
   - Switch to model: gemini-2.0-flash (100% success rate in tests)
   - Or use: gemini-2.0-flash-thinking-exp for complex schemas

3. **Schema Tips for gemini-2.5-pro**:
   - MUST include 'description' fields in properties
   - Keep schemas simple with basic types
   - Reduce required fields if possible"""
                                    elif "SAFETY" in finish_reason_str:
                                        error_text = "Response blocked by safety filters. Try rephrasing your prompt."
                                    elif "MAX_TOKENS" in finish_reason_str:
                                        error_text = "Response exceeded max tokens. Try increasing max_output_tokens."
                                    else:
                                        error_text = f"Model stopped with reason: {candidate.finish_reason}"
                                else:
                                    error_text = "Model returned empty response with no content parts"
                                
                                self._log(error_text)
                                debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                                debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False)
                                return (f"## Error\n{error_text}\n\nTry:\n1. Simplifying your prompt\n2. Checking if the schema is valid\n3. Using a different model\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)
                        else:
                            self._log("Candidate has no content")
                            error_text = "Response candidate has no content"
                            debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                            debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False)
                            return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)
                    else:
                        self._log("No candidates in response")
                        error_text = "Response has no candidates"
                        debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                        debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False)
                        return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)
                
                if result_text:
                    result_text = result_text.strip()
                    debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                    debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False)
                    
                    # Handle enum mode differently - it returns plain text
                    if output_mode == "enum":
                        self._log(f"Received enum response: {result_text}")
                        # For enum, return the selected value in both outputs
                        return (result_text, result_text, debug_request_str, debug_response_str)
                    else:
                        # For JSON schema mode, parse and format
                        try:
                            parsed_json = json.loads(result_text)
                            formatted_output = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                            self._log(f"Received structured response with {len(parsed_json)} fields")
                            return (formatted_output, result_text, debug_request_str, debug_response_str)
                        except json.JSONDecodeError:
                            self._log("Warning: Response is not valid JSON, returning raw text")
                            return (result_text, result_text, debug_request_str, debug_response_str)
                else:
                    error_text = "Could not extract any text from response"
                    self._log(error_text)
                    debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False)
                    debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False)
                    return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)
            except Exception as e:
                error_text = f"Error extracting response: {str(e)}"
                self._log(error_text)
                self._log(f"Full error: {traceback.format_exc()}")
                debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False) if debug_request else "No request data"
                debug_response_str = json.dumps(debug_response, indent=2, ensure_ascii=False) if debug_response else "No response data"
                return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self._log(error_message)
            traceback.print_exc()
            debug_request_str = json.dumps(debug_request, indent=2, ensure_ascii=False) if debug_request else "No request data"
            debug_response_str = "No response data - exception occurred"
            return (f"## Error\n{error_message}\n\n## Debug Log\n" + "\n".join(self.log_messages), "", debug_request_str, debug_response_str)


class GeminiJSONExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {
                    "default": "gemini-2.0-flash",
                    "multiline": False,
                    "placeholder": "e.g., gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-pro"
                }),
                "extract_fields": ("STRING", {
                    "multiline": True,
                    "default": "title: string\nsummary: string\nkeywords: string[]",
                    "placeholder": "field_name: type\nfield2: type\n..."
                }),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "input_text": ("STRING", {"multiline": True, "default": ""}),
                "system_instructions": ("STRING", {
                    "multiline": True, 
                    "default": "Extract the requested information from the provided text. Be accurate and concise."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("extracted_json", "formatted_output")
    FUNCTION = "extract_json"
    CATEGORY = "🤖 Gemini"

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[GeminiJSONExtractor] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove additionalProperties field from JSON schema for Gemini API compatibility"""
        # Gemini API does not support additionalProperties
        # Error: "additional_properties parameter is not supported in Gemini API"
        
        def clean_dict(obj):
            if isinstance(obj, dict):
                # Remove additionalProperties field
                cleaned = {k: v for k, v in obj.items() if k != 'additionalProperties'}
                # Recursively clean nested objects
                for key, value in cleaned.items():
                    if isinstance(value, dict):
                        cleaned[key] = clean_dict(value)
                    elif isinstance(value, list):
                        cleaned[key] = [clean_dict(item) if isinstance(item, dict) else item for item in value]
                return cleaned
            return obj
        
        cleaned_schema = clean_dict(schema)
        if 'additionalProperties' in schema:
            self._log("Removed 'additionalProperties' field from schema for Gemini API compatibility")
        return cleaned_schema
    
    def _parse_field_definitions(self, field_defs: str) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        lines = field_defs.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            field_name, field_type = line.split(':', 1)
            field_name = field_name.strip()
            field_type = field_type.strip().lower()
            
            if '[]' in field_type:
                base_type = field_type.replace('[]', '').strip()
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": {"type": self._map_type(base_type)}
                }
            else:
                schema["properties"][field_name] = {"type": self._map_type(field_type)}
            
            if not field_type.endswith('?'):
                schema["required"].append(field_name)
        
        # Clean the schema for Gemini API
        schema = self._clean_schema_for_gemini(schema)
        self._log(f"Generated and cleaned schema with fields: {list(schema['properties'].keys())}")
        return schema

    def _map_type(self, type_str: str) -> str:
        type_str = type_str.replace('?', '').strip()
        type_mapping = {
            'str': 'string',
            'string': 'string',
            'int': 'integer',
            'integer': 'integer',
            'float': 'number',
            'number': 'number',
            'bool': 'boolean',
            'boolean': 'boolean',
            'object': 'object',
            'dict': 'object',
            'array': 'array',
            'list': 'array'
        }
        return type_mapping.get(type_str, 'string')

    def extract_json(self, prompt, api_key, model, extract_fields, temperature, seed,
                     input_text="", system_instructions=""):
        
        self.log_messages = []

        try:
            if not api_key:
                error_message = "Error: No API key provided."
                self._log(error_message)
                return ("", f"## ERROR: {error_message}")

            client = genai.Client(api_key=api_key)
            
            random.seed(seed)
            torch.manual_seed(seed)

            response_schema = self._parse_field_definitions(extract_fields)
            
            full_prompt = prompt
            if input_text:
                full_prompt = f"{prompt}\n\nInput Text:\n{input_text}"

            # Build generation config dict
            gen_config_dict = {
                "temperature": temperature,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "candidate_count": 1,
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }

            if system_instructions and system_instructions.strip():
                gen_config_dict["system_instruction"] = system_instructions
            
            # Create config object
            gen_config = types.GenerateContentConfig(**gen_config_dict)

            self._log(f"Extracting JSON with schema fields: {list(response_schema['properties'].keys())}")

            response = client.models.generate_content(
                model=model,
                contents=[full_prompt],
                config=gen_config
            )

            if response:
                try:
                    result_text = response.text
                    if result_text is None:
                        # Try alternative extraction methods
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                if candidate.content.parts:
                                    part = candidate.content.parts[0]
                                    if hasattr(part, 'text'):
                                        result_text = part.text
                    
                    if result_text:
                        result_text = result_text.strip()
                        try:
                            parsed_json = json.loads(result_text)
                            formatted = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                            
                            output_lines = []
                            for key, value in parsed_json.items():
                                if isinstance(value, list):
                                    output_lines.append(f"**{key}**: {', '.join(str(v) for v in value)}")
                                else:
                                    output_lines.append(f"**{key}**: {value}")
                            
                            formatted_text = "\n".join(output_lines)
                            
                            self._log(f"Successfully extracted {len(parsed_json)} fields")
                            return (result_text, formatted_text)
                        except json.JSONDecodeError:
                            self._log("Warning: Response is not valid JSON")
                            return (result_text, result_text)
                    else:
                        error_text = "Could not extract text from response"
                        self._log(error_text)
                        return ("", f"## Error\n{error_text}")
                except Exception as e:
                    error_text = f"Error processing response: {str(e)}"
                    self._log(error_text)
                    return ("", f"## Error\n{error_text}")
            else:
                error_text = "Failed to get response from Gemini API"
                self._log(error_text)
                return ("", f"## Error\n{error_text}")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self._log(error_message)
            traceback.print_exc()
            return ("", f"## Error\n{error_message}")