import os
import time
import json
import random
import torch
import traceback
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("structured_output", "raw_json")
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

    def _parse_schema(self, schema_json: str) -> Dict[str, Any]:
        try:
            schema = json.loads(schema_json)
            self._log(f"Schema parsed successfully: {list(schema.get('properties', {}).keys())}")
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
            "type": "object",
            "properties": {
                "selection": {
                    "type": "string",
                    "enum": enum_options
                }
            },
            "required": ["selection"]
        }

    def _call_gemini_api_structured(self, client, model, contents, gen_config, retry_count=0, max_retries=3):
        try:
            self._log(f"Structured API call attempt #{retry_count + 1}")
            self._log(f"Using model: {model}")
            self._log(f"Config keys: {list(gen_config.__dict__.keys()) if hasattr(gen_config, '__dict__') else 'N/A'}")
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            if response:
                self._log("Valid structured response received")
                return response
            else:
                self._log("No response received")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    return self._call_gemini_api_structured(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached.")
                    return None

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api_structured(client, model, contents, gen_config, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached.")
                return None

    def generate_structured(self, prompt, api_key, model, output_mode, schema_json, 
                           temperature, max_output_tokens, seed,
                           system_instructions="", enum_options="", top_p=0.95, 
                           top_k=64, property_ordering=""):
        
        self.log_messages = []

        try:
            if not api_key:
                error_message = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_message)
                return (f"## ERROR: {error_message}", "")

            client = genai.Client(api_key=api_key)
            
            random.seed(seed)
            torch.manual_seed(seed)

            if output_mode == "json_schema":
                response_schema = self._parse_schema(schema_json)
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
            
            # Add response schema for structured output
            if response_schema:
                gen_config_dict["response_mime_type"] = "application/json"
                gen_config_dict["response_schema"] = response_schema

            if property_ordering and property_ordering.strip():
                ordering_list = [p.strip() for p in property_ordering.split(",")]
                # Note: Property ordering may not be supported in the current SDK version
                self._log(f"Property ordering requested (may not be supported): {ordering_list}")

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
            
            # Enhance prompt for better structured output generation
            enhanced_prompt = prompt
            if output_mode == "json_schema":
                # Add a hint about the expected JSON structure
                schema_hint = f"\n\nPlease respond with a valid JSON object that follows this structure:\n{json.dumps(response_schema, indent=2)[:500]}"
                enhanced_prompt = prompt + schema_hint
                self._log("Added schema hint to prompt for better generation")

            response = self._call_gemini_api_structured(
                client=client,
                model=model,
                contents=[enhanced_prompt],
                gen_config=gen_config,
                max_retries=3
            )

            if response is None:
                error_text = "Failed to get response from Gemini API after multiple attempts."
                self._log(error_text)
                return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")

            # Extract text from response
            try:
                # First try direct text extraction
                result_text = None
                if hasattr(response, 'text'):
                    try:
                        result_text = response.text
                        if result_text:
                            self._log(f"Successfully got text from response: {len(result_text)} chars")
                    except Exception as e:
                        self._log(f"Error getting text: {str(e)}")
                        result_text = None
                
                if result_text is None:
                    self._log("Response.text is None or failed, trying alternative extraction methods...")
                    
                    # Try alternative extraction methods
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        self._log(f"Candidate found: {candidate}")
                        
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
                                        error_text = "Model returned empty response. Possible issues:\n1. The schema may be too complex or incompatible\n2. The prompt doesn't match the expected schema\n3. The model cannot generate valid JSON for this schema\n\nTry simplifying the schema or using a more capable model like gemini-2.0-flash-thinking-exp"
                                    elif "SAFETY" in finish_reason_str:
                                        error_text = "Response blocked by safety filters. Try rephrasing your prompt."
                                    elif "MAX_TOKENS" in finish_reason_str:
                                        error_text = "Response exceeded max tokens. Try increasing max_output_tokens."
                                    else:
                                        error_text = f"Model stopped with reason: {candidate.finish_reason}"
                                else:
                                    error_text = "Model returned empty response with no content parts"
                                
                                self._log(error_text)
                                return (f"## Error\n{error_text}\n\nTry:\n1. Simplifying your prompt\n2. Checking if the schema is valid\n3. Using a different model\n\n## Debug Log\n" + "\n".join(self.log_messages), "")
                        else:
                            self._log("Candidate has no content")
                            error_text = "Response candidate has no content"
                            return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")
                    else:
                        self._log("No candidates in response")
                        error_text = "Response has no candidates"
                        return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")
                
                if result_text:
                    result_text = result_text.strip()
                    try:
                        parsed_json = json.loads(result_text)
                        formatted_output = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                        self._log(f"Received structured response with {len(parsed_json)} fields")
                        return (formatted_output, result_text)
                    except json.JSONDecodeError:
                        self._log("Warning: Response is not valid JSON, returning raw text")
                        return (result_text, result_text)
                else:
                    error_text = "Could not extract any text from response"
                    self._log(error_text)
                    return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")
            except Exception as e:
                error_text = f"Error extracting response: {str(e)}"
                self._log(error_text)
                self._log(f"Full error: {traceback.format_exc()}")
                return (f"## Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self._log(error_message)
            traceback.print_exc()
            return (f"## Error\n{error_message}\n\n## Debug Log\n" + "\n".join(self.log_messages), "")


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
        
        self._log(f"Generated schema with fields: {list(schema['properties'].keys())}")
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