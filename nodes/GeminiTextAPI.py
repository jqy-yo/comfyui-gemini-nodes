import os
import time
import random
import torch
import traceback
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any


class GeminiTextAPI:
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
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),  # INT32 max value
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
                "api_version": (["auto", "v1", "v1beta", "v1alpha"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "🤖 Gemini"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[GeminiTextAPI] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
        
    def _call_gemini_api(self, client, model, contents, gen_config, retry_count=0, max_retries=3):
        """Call Gemini API with retry logic using the updated generate_content method"""
        try:
            self._log(f"API call attempt #{retry_count + 1}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            # Check if response is valid
            if response:
                self._log("Valid API response received")
                # Log response details for debugging
                self._log(f"Response type: {type(response)}")
                if hasattr(response, 'text'):
                    self._log(f"Has text attribute: {response.text is not None}")
                if hasattr(response, 'candidates'):
                    self._log(f"Has candidates: {len(response.candidates) if response.candidates else 0}")
                return response
            else:
                self._log("No response received")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait 2 seconds before retry
                    return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                    return None

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff: 2s, 4s, 6s...
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                return None

    def generate_text(self, prompt, api_key, model, temperature, max_output_tokens, seed,
                      system_instructions="", top_p=0.95, top_k=64, api_version="auto"):
        """Generate text response from Gemini API using the new client structure"""
        # Reset log messages
        self.log_messages = []

        try:
            # Check if API key is provided
            if not api_key:
                error_message = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_message)
                return (f"## ERROR: {error_message}\n\nPlease provide a valid Google API key.",)

            # Create client instance with API key
            # Note: The google-genai SDK automatically handles API version selection
            # The api_version parameter is provided for future compatibility
            if api_version != "auto":
                self._log(f"Note: Using google-genai SDK which automatically handles API versions. Requested version: {api_version}")
            
            client = genai.Client(api_key=api_key)

            # Set random seeds for reproducibility
            random.seed(seed)
            torch.manual_seed(seed)

            # Build generation config
            gen_config_dict = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "candidate_count": 1
            }
            
            # Add system instructions if provided
            if system_instructions and system_instructions.strip():
                self._log(f"Using system instructions: {system_instructions[:50]}...")
                gen_config_dict["system_instruction"] = system_instructions
            
            # Create config object
            gen_config = types.GenerateContentConfig(**gen_config_dict)

            self._log(f"Sending prompt to Gemini API (model: {model}, temp: {temperature})")

            # Make API call with contents parameter
            response = self._call_gemini_api(
                client=client,
                model=model,
                contents=[prompt],  # Contents expects a list
                gen_config=gen_config,
                max_retries=3
            )

            # Check if we got a valid response
            if response is None:
                error_text = "Failed to get response from Gemini API after multiple attempts."
                self._log(error_text)
                return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)

            # Extract and return the raw text from the response
            try:
                # Try to get text directly first
                result_text = None
                if hasattr(response, 'text'):
                    result_text = response.text
                
                if result_text is None:
                    self._log("Response.text is None, checking response structure...")
                    self._log(f"Response type: {type(response)}")
                    
                    # Try to extract from candidates
                    if hasattr(response, 'candidates') and response.candidates:
                        self._log(f"Number of candidates: {len(response.candidates)}")
                        candidate = response.candidates[0]
                        
                        # Log candidate details
                        if hasattr(candidate, 'finish_reason'):
                            self._log(f"Finish reason: {candidate.finish_reason}")
                        
                        if hasattr(candidate, 'content') and candidate.content:
                            content = candidate.content
                            self._log(f"Content role: {content.role if hasattr(content, 'role') else 'unknown'}")
                            
                            if hasattr(content, 'parts') and content.parts:
                                self._log(f"Found {len(content.parts)} parts in content")
                                for i, part in enumerate(content.parts):
                                    if hasattr(part, 'text') and part.text:
                                        result_text = part.text
                                        self._log(f"Extracted text from part {i}: {len(result_text)} chars")
                                        break
                            else:
                                self._log("No parts found in content - response may be empty or blocked")
                                
                                # Provide helpful error message based on finish_reason
                                if hasattr(candidate, 'finish_reason'):
                                    if str(candidate.finish_reason) == "STOP":
                                        error_text = "Model returned empty response (finish_reason=STOP). The prompt may have been blocked or the model had nothing to generate."
                                    elif "SAFETY" in str(candidate.finish_reason):
                                        error_text = "Response blocked by safety filters. Try rephrasing your prompt."
                                    else:
                                        error_text = f"Model stopped with reason: {candidate.finish_reason}"
                                else:
                                    error_text = "Model returned empty response with no content"
                                
                                return (f"## API Error\n{error_text}\n\nTry:\n1. Rephrasing your prompt\n2. Using a different model\n3. Checking if the content violates safety guidelines\n\n## Debug Log\n" + "\n".join(self.log_messages),)
                        else:
                            self._log("Candidate has no content")
                            error_text = "Response candidate has no content"
                            return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)
                    else:
                        self._log("No candidates in response")
                        error_text = "Response has no candidates"
                        return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)
                
                # Check if we successfully got text
                if result_text:
                    result_text = result_text.strip()
                    self._log(f"Successfully received response ({len(result_text)} characters)")
                    return (result_text,)
                else:
                    error_text = "Could not extract any text from response"
                    self._log(error_text)
                    return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)
                
            except Exception as e:
                error_text = f"Error extracting text: {str(e)}"
                self._log(error_text)
                self._log(f"Full error: {traceback.format_exc()}")
                return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self._log(error_message)
            traceback.print_exc()

            # Return error message and debug log
            return (f"## Error\n{error_message}\n\n## Debug Log\n" + "\n".join(self.log_messages),)