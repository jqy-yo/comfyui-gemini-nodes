import os
import tempfile
import numpy as np
import torch
import base64
import json
import time
from PIL import Image
from comfy.utils import ProgressBar
import traceback
from google import genai
from google.genai import types
import io


class GeminiImageProcessor:
    """
    Universal node for image processing using Google's Gemini API.
    Supports: image analysis, modification suggestions, generation prompts, and text extraction.
    """
    
    def _clean_schema_for_gemini(self, schema: dict) -> dict:
        """Remove additionalProperties field from JSON schema for Gemini API compatibility"""
        def clean_dict(obj):
            if isinstance(obj, dict):
                cleaned = {k: v for k, v in obj.items() if k != 'additionalProperties'}
                for key, value in cleaned.items():
                    if isinstance(value, dict):
                        cleaned[key] = clean_dict(value)
                    elif isinstance(value, list):
                        cleaned[key] = [clean_dict(item) if isinstance(item, dict) else item for item in value]
                return cleaned
            return obj
        
        cleaned_schema = clean_dict(schema)
        if 'additionalProperties' in schema:
            print("[GeminiImageProcessor] Removed 'additionalProperties' field from schema for Gemini API compatibility")
        return cleaned_schema

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE", {}),
                "mode": (["analyze", "modify", "generate_prompt", "extract_text", "structured_output"], {
                    "default": "analyze"
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "placeholder": "Enter your prompt based on the selected mode..."
                }),
                "model": ("STRING", {
                    "default": "gemini-1.5-flash",
                    "multiline": False,
                    "placeholder": "e.g., gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 50, "max": 8192, "step": 10}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "reference_image": ("IMAGE", {}),  # For modification mode
                "output_schema": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "main_subject": {"type": "string"},\n    "description": {"type": "string"},\n    "colors": {\n      "type": "array",\n      "items": {"type": "string"}\n    },\n    "mood": {"type": "string"},\n    "style": {"type": "string"}\n  },\n  "required": ["main_subject", "description"]\n}',
                    "placeholder": "JSON Schema for structured output (only for structured_output mode)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "processed_image", "raw_json", "api_request", "api_response")
    FUNCTION = "process_image"
    CATEGORY = "🤖 Gemini"

    def get_mode_prompt(self, mode, user_prompt):
        """Generate appropriate prompt based on mode"""
        mode_prompts = {
            "analyze": user_prompt or "Provide a detailed analysis of this image including subjects, composition, colors, mood, and any notable details.",
            "modify": f"Based on this image, suggest detailed modifications: {user_prompt}\nProvide specific editing instructions that could be applied.",
            "generate_prompt": f"Create a detailed text-to-image generation prompt based on this image. {user_prompt}\nInclude style, composition, lighting, colors, and all visual elements.",
            "extract_text": f"Extract and transcribe all text visible in this image. {user_prompt}\nInclude formatting and layout information where relevant.",
            "structured_output": user_prompt or "Analyze this image and provide structured information."
        }
        return mode_prompts.get(mode, user_prompt)

    def prepare_image_for_api(self, image_tensor):
        """Convert tensor image to base64 for API"""
        # Handle batch dimension
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # Convert tensor to numpy array (0-255 range)
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img_np)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        # Encode to base64
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return base64_image, img

    def _send_api_request(self, client, model, contents, config, use_structured=False, retry_count=0, max_retries=3):
        """Send API request to Gemini with retry logic"""
        # Normalize model name
        if model and not model.startswith('models/'):
            model = f"models/{model}"
        
        try:
            print(f"[GeminiImageProcessor] Sending request to {model}...")
            
            # Log request details for debugging
            print(f"[DEBUG] Model: {model}")
            print(f"[DEBUG] Content parts: {len(contents) if isinstance(contents, list) else 1}")
            print(f"[DEBUG] Temperature: {config.temperature if hasattr(config, 'temperature') else 'N/A'}")
            print(f"[DEBUG] Max tokens: {config.max_output_tokens if hasattr(config, 'max_output_tokens') else 'N/A'}")
            
            # Store complete request data that's being sent to API
            api_request = {
                "model": model,
                "contents": contents,  # Full contents including images
                "config": {}
            }
            
            # Add all config parameters
            if hasattr(config, 'temperature'):
                api_request["config"]["temperature"] = config.temperature
            if hasattr(config, 'max_output_tokens'):
                api_request["config"]["max_output_tokens"] = config.max_output_tokens
            if hasattr(config, 'top_p'):
                api_request["config"]["top_p"] = config.top_p
            if hasattr(config, 'top_k'):
                api_request["config"]["top_k"] = config.top_k
            if hasattr(config, 'candidate_count'):
                api_request["config"]["candidate_count"] = config.candidate_count
            if hasattr(config, 'seed'):
                api_request["config"]["seed"] = config.seed
            if hasattr(config, 'response_mime_type'):
                api_request["config"]["response_mime_type"] = config.response_mime_type
            if hasattr(config, 'response_schema'):
                api_request["config"]["response_schema"] = config.response_schema
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            # Process response
            if response:
                api_response = {"text": None, "candidates": []}
                
                if hasattr(response, 'text'):
                    api_response["text"] = response.text
                
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        candidate_data = {
                            "finish_reason": str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else None,
                            "content": {"parts": []}
                        }
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text'):
                                        candidate_data["content"]["parts"].append({"text": part.text})
                        api_response["candidates"].append(candidate_data)
                
                # Extract text from response
                if response.text:
                    print(f"[GeminiImageProcessor] Response received: {len(response.text)} chars")
                    return (response.text, api_request, api_response)
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            text = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else ""
                            return (text, api_request, api_response)
                
                # No valid response
                if retry_count < max_retries - 1:
                    print(f"[GeminiImageProcessor] No valid response, retrying... ({retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))
                    return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
                
                return ("Error: No response from API", api_request, api_response)
            
            else:
                if retry_count < max_retries - 1:
                    print(f"[GeminiImageProcessor] Empty response, retrying... ({retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))
                    return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
                return ("Error: Empty response from API", api_request, {})
                
        except Exception as e:
            error_msg = f"API error: {str(e)}"
            print(f"[GeminiImageProcessor] {error_msg}")
            
            # Special handling for 500 errors
            if "500 INTERNAL" in str(e) or "Internal error" in str(e):
                print("[GeminiImageProcessor] ⚠️ 500 Internal Server Error")
                print(f"[GeminiImageProcessor] Model: {model}")
                
                if "lite" in model.lower():
                    print("[GeminiImageProcessor] Lite models have limited capabilities")
                    print("[GeminiImageProcessor] Try using gemini-1.5-flash or gemini-2.0-flash")
                
                error_detail = f"Error: 500 Internal Server Error with {model}. "
                if "lite" in model.lower():
                    error_detail += "Lite models have limited capabilities. Try gemini-1.5-flash instead."
                else:
                    error_detail += "The model might not support this operation."
                
                return (error_detail, api_request, {})
            
            # Retry for other errors
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)
                print(f"[GeminiImageProcessor] Retrying in {wait_time}s... ({retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
            
            traceback.print_exc()
            return (f"Error: {error_msg}", api_request if 'api_request' in locals() else {}, {})

    def process_image(self, api_key, image, mode, prompt, model, temperature, max_output_tokens, 
                     top_p, top_k, seed, reference_image=None, output_schema=None):
        """Process image based on selected mode"""
        
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Prepare the main image
        print(f"[GeminiImageProcessor] Processing image in '{mode}' mode with model: {model}")
        base64_image, pil_image = self.prepare_image_for_api(image)
        
        # Get appropriate prompt for the mode
        final_prompt = self.get_mode_prompt(mode, prompt)
        
        # Build content parts
        content_parts = []
        
        # Add prompt
        content_parts.append({"text": final_prompt})
        
        # Add main image
        content_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64_image
            }
        })
        
        # Add reference image if provided (for modification mode)
        if reference_image is not None and mode == "modify":
            print("[GeminiImageProcessor] Adding reference image for modification comparison")
            ref_base64, _ = self.prepare_image_for_api(reference_image)
            content_parts.append({"text": "Reference image for modifications:"})
            content_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": ref_base64
                }
            })
        
        # Handle structured output mode
        use_structured = (mode == "structured_output")
        response_schema = None
        
        if use_structured and output_schema:
            try:
                response_schema = json.loads(output_schema)
                response_schema = self._clean_schema_for_gemini(response_schema)
                print("[GeminiImageProcessor] Using structured output with schema")
            except json.JSONDecodeError as e:
                print(f"[GeminiImageProcessor] Warning: Invalid JSON schema: {e}")
                use_structured = False
        
        # Create Gemini client
        client = genai.Client(api_key=api_key)
        
        # Create generation config
        gen_config_dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "candidate_count": 1
        }
        
        if seed > 0:
            gen_config_dict["seed"] = seed
        
        if use_structured and response_schema:
            gen_config_dict["response_mime_type"] = "application/json"
            gen_config_dict["response_schema"] = response_schema
        
        gen_config = types.GenerateContentConfig(**gen_config_dict)
        
        # Send API request
        result = self._send_api_request(client, model, content_parts, gen_config, use_structured)
        
        # Parse result
        if isinstance(result, tuple) and len(result) == 3:
            output_text, api_request, api_response = result
        else:
            output_text = str(result)
            api_request, api_response = {}, {}
        
        # Format output based on mode
        raw_json = ""
        if mode == "structured_output" or use_structured:
            try:
                parsed = json.loads(output_text)
                raw_json = output_text
                output_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                raw_json = ""
        
        # Return original image as processed_image (since we don't actually modify pixels)
        # In a real implementation, you might want to apply the suggested modifications
        processed_image = image
        
        # Add mode-specific formatting
        if mode == "modify":
            output_text = f"=== MODIFICATION SUGGESTIONS ===\n\n{output_text}"
        elif mode == "generate_prompt":
            output_text = f"=== GENERATION PROMPT ===\n\n{output_text}"
        elif mode == "extract_text":
            output_text = f"=== EXTRACTED TEXT ===\n\n{output_text}"
        elif mode == "analyze":
            output_text = f"=== IMAGE ANALYSIS ===\n\n{output_text}"
        
        print(f"[GeminiImageProcessor] Processing complete")
        
        # Format api_request for display (convert image data to summary)
        display_request = self._format_request_for_display(api_request)
        
        return (
            output_text,
            processed_image,
            raw_json,
            json.dumps(display_request, indent=2, ensure_ascii=False),
            json.dumps(api_response, indent=2, ensure_ascii=False)
        )
    
    def _format_request_for_display(self, request):
        """Format API request for readable display"""
        if not request:
            return {}
        
        formatted = {
            "model": request.get("model", "N/A"),
            "contents": [],
            "config": request.get("config", {})
        }
        
        # Process contents to make them more readable
        if "contents" in request and request["contents"]:
            for item in request["contents"]:
                if isinstance(item, dict):
                    if "text" in item:
                        formatted["contents"].append({
                            "type": "text",
                            "text": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
                        })
                    elif "inline_data" in item:
                        formatted["contents"].append({
                            "type": "image",
                            "mime_type": item["inline_data"].get("mime_type", "unknown"),
                            "data_size": len(item["inline_data"].get("data", "")) if "data" in item["inline_data"] else 0,
                            "data_preview": "[BASE64_IMAGE_DATA]"
                        })
        
        return formatted


class GeminiImageBatchProcessor:
    """
    Batch processing node for multiple images using Google's Gemini API.
    Processes multiple images in a single request for comparative analysis or batch operations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "images": ("IMAGE", {}),  # Expects batch of images
                "mode": (["compare", "sequence_analysis", "batch_describe", "find_differences"], {
                    "default": "batch_describe"
                }),
                "prompt": ("STRING", {
                    "default": "Analyze these images.",
                    "multiline": True,
                }),
                "model": ("STRING", {
                    "default": "gemini-1.5-flash",
                    "multiline": False,
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 50, "max": 8192, "step": 10}),
                "process_individually": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("analysis_text", "first_image", "api_request", "api_response")
    FUNCTION = "process_batch"
    CATEGORY = "🤖 Gemini"
    
    def get_batch_prompt(self, mode, user_prompt, num_images):
        """Generate appropriate prompt for batch processing"""
        batch_prompts = {
            "compare": f"Compare and contrast these {num_images} images. {user_prompt}\nHighlight similarities and differences.",
            "sequence_analysis": f"Analyze these {num_images} images as a sequence. {user_prompt}\nDescribe the progression or changes.",
            "batch_describe": f"Describe each of these {num_images} images. {user_prompt}",
            "find_differences": f"Find and list all differences between these {num_images} images. {user_prompt}",
        }
        return batch_prompts.get(mode, user_prompt)
    
    def process_batch(self, api_key, images, mode, prompt, model, temperature, max_output_tokens, process_individually):
        """Process batch of images"""
        
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Get number of images
        num_images = images.shape[0] if len(images.shape) == 4 else 1
        print(f"[GeminiImageBatchProcessor] Processing {num_images} images in '{mode}' mode")
        
        # Prepare all images
        image_parts = []
        for i in range(num_images):
            img_tensor = images[i] if len(images.shape) == 4 else images
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            if process_individually:
                image_parts.append({
                    "text": f"Image {i + 1}:",
                    "image": base64_image
                })
            else:
                image_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                })
        
        # Create content
        final_prompt = self.get_batch_prompt(mode, prompt, num_images)
        
        if process_individually:
            # Process each image separately and combine results
            results = []
            client = genai.Client(api_key=api_key)
            
            for i, img_data in enumerate(image_parts):
                content = [
                    {"text": f"Image {i + 1} analysis:\n{prompt}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_data["image"]}}
                ]
                
                gen_config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens // num_images,  # Divide tokens among images
                    candidate_count=1
                )
                
                try:
                    response = client.models.generate_content(
                        model=f"models/{model}" if not model.startswith("models/") else model,
                        contents=content,
                        config=gen_config
                    )
                    results.append(f"=== Image {i + 1} ===\n{response.text}\n")
                except Exception as e:
                    results.append(f"=== Image {i + 1} ===\nError: {str(e)}\n")
            
            output_text = "\n".join(results)
            api_request = {
                "mode": "individual",
                "num_images": num_images,
                "model": f"models/{model}" if not model.startswith("models/") else model,
                "temperature": temperature,
                "max_output_tokens_per_image": max_output_tokens // num_images
            }
            api_response = {"combined_results": len(results)}
            
        else:
            # Process all images in single request
            content_parts = [{"text": final_prompt}]
            content_parts.extend(image_parts)
            
            # Send to API
            client = genai.Client(api_key=api_key)
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                candidate_count=1
            )
            
            try:
                response = client.models.generate_content(
                    model=f"models/{model}" if not model.startswith("models/") else model,
                    contents=content_parts,
                    config=gen_config
                )
                output_text = response.text
                api_request = {
                    "mode": "batch",
                    "num_images": num_images,
                    "model": f"models/{model}" if not model.startswith("models/") else model,
                    "contents": [
                        {"type": "text", "content": final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt},
                        {"type": "images", "count": num_images, "format": "JPEG"}
                    ],
                    "config": {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "candidate_count": 1
                    }
                }
                api_response = {"success": True, "text_length": len(response.text)}
            except Exception as e:
                output_text = f"Error processing batch: {str(e)}"
                api_request = {
                    "mode": "batch",
                    "num_images": num_images,
                    "model": f"models/{model}" if not model.startswith("models/") else model,
                    "error": "Failed to send request"
                }
                api_response = {"error": str(e)}
        
        # Return first image for preview
        first_image = images[0].unsqueeze(0) if len(images.shape) == 4 else images.unsqueeze(0)
        
        return (
            output_text,
            first_image,
            json.dumps(api_request, indent=2),
            json.dumps(api_response, indent=2)
        )