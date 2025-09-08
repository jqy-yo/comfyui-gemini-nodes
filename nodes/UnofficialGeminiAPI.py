import os
import time
import json
import requests
import base64
from PIL import Image
import numpy as np
import torch
from typing import Optional, List, Dict, Any
import io


class UnofficialGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "sk-xxxx"}),
                "api_format": (["OpenAI Compatible", "Gemini Native"], {"default": "OpenAI Compatible"}),
                "model": ("STRING", {
                    "default": "claude-3-5-sonnet-20240620",
                    "multiline": False,
                    "placeholder": "e.g., claude-3-5-sonnet, gpt-4o, gemini-2.5-flash-image-preview"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "base_url": ("STRING", {
                    "default": "https://www.chataiapi.com/v1",
                    "multiline": False,
                    "placeholder": "API base URL"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("response", "image", "api_request", "api_response")
    FUNCTION = "call_unofficial_api"
    CATEGORY = "🤖 Gemini/Unofficial"

    def __init__(self):
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}
        self.last_image = None

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[UnofficialGeminiAPI] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def tensor_to_base64(self, tensor):
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
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            self._log(f"Image converted to base64, size: {image.size}")
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            self._log(f"Error converting image to base64: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return None
    
    def base64_to_tensor(self, base64_str):
        """Convert base64 string to ComfyUI tensor format [B, H, W, C]"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(base64_str)
            
            # Open image with PIL
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Add batch dimension [H, W, C] -> [1, H, W, C]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self._log(f"Image decoded from base64, size: {image.size}")
            return img_tensor
            
        except Exception as e:
            self._log(f"Error converting base64 to tensor: {str(e)}")
            return None
    
    def create_empty_image(self, width=512, height=512):
        """Create an empty black image tensor"""
        # Create black image
        img_array = np.zeros((height, width, 3), dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    def call_unofficial_api(self, prompt, api_key, api_format, model, temperature, max_tokens, base_url, seed,
                           system_prompt="You are a helpful assistant.", 
                           image1=None, image2=None, image3=None, image4=None, image5=None,
                           top_p=0.95, frequency_penalty=0.0, presence_penalty=0.0):
        
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}

        try:
            if not api_key:
                error_message = "Error: No API key provided. Please enter your API key."
                self._log(error_message)
                empty_image = self.create_empty_image()
                return (f"## ERROR: {error_message}", empty_image, "{}", "{}")

            # Collect all non-None images
            images = []
            for i, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    image_base64 = self.tensor_to_base64(img)
                    if image_base64:
                        images.append(image_base64)
                        self._log(f"Image {i} successfully converted to base64")
                    else:
                        self._log(f"Failed to convert image {i}")
            
            # Build request based on API format
            if api_format == "Gemini Native":
                # Gemini Native API format
                parts = [{"text": prompt}]
                
                # Add images to parts if any
                for img_base64 in images:
                    # Remove data URL prefix for Gemini format
                    if img_base64.startswith('data:image'):
                        img_data = img_base64.split(',')[1] if ',' in img_base64 else img_base64
                    else:
                        img_data = img_base64
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_data
                        }
                    })
                
                # Add system prompt to the beginning if provided
                if system_prompt and system_prompt != "You are a helpful assistant.":
                    parts.insert(0, {"text": system_prompt + "\n\n"})
                
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": parts
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "topP": top_p
                    }
                }
                
                # Construct URL for Gemini native format
                # Check if base_url already contains the model endpoint
                if ':generateContent' in base_url:
                    url = base_url
                else:
                    # Construct the URL with model name
                    base = base_url.rstrip('/')
                    if '/v1' in base:
                        url = f"{base}/models/{model}:generateContent"
                    else:
                        url = f"{base}/v1/models/{model}:generateContent"
                
                # Add API key to URL or headers based on provider
                if '?key=' not in url:
                    url = f"{url}?key={api_key}"
                    headers = {'Content-Type': 'application/json'}
                else:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {api_key}'
                    }
                
                self._log(f"Using Gemini Native API format")
                self._log(f"Sending {len(images)} image(s) with prompt") if images else None
                
            else:
                # OpenAI Compatible format
                messages = []
                
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                # Build user message with text and images
                if images:
                    content = [{"type": "text", "text": prompt}]
                    for img_base64 in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_base64}
                        })
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                    self._log(f"Sending {len(images)} image(s) with prompt")
                else:
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "seed": seed if seed > 0 else None
                }
                
                # Remove None values from payload
                payload = {k: v for k, v in payload.items() if v is not None}
                
                url = f"{base_url.rstrip('/')}/chat/completions"
                
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                    'User-Agent': 'ComfyUI-Gemini-Nodes/1.0.0',
                    'Content-Type': 'application/json'
                }
                
                self._log(f"Using OpenAI Compatible API format")
            
            self.api_request = payload
            self._log(f"Sending request to {url}")
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                self.api_response = data
                
                # Handle response based on API format
                if api_format == "Gemini Native":
                    # Handle Gemini native response format
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        content_obj = candidate.get('content', {})
                        parts = content_obj.get('parts', [])
                        
                        # Extract text and images from parts
                        text_content = ""
                        output_images = []
                        
                        for part in parts:
                            if 'text' in part:
                                text_content += part['text']
                            elif 'inlineData' in part:
                                inline_data = part['inlineData']
                                mime_type = inline_data.get('mimeType', '')
                                img_data = inline_data.get('data', '')
                                
                                if img_data and 'image' in mime_type:
                                    # Convert base64 to tensor
                                    try:
                                        # Add data URL prefix if not present
                                        if not img_data.startswith('data:'):
                                            img_data = f"data:{mime_type};base64,{img_data}"
                                        
                                        img_tensor = self.base64_to_tensor(img_data)
                                        if img_tensor is not None:
                                            output_images.append(img_tensor)
                                            self._log(f"Extracted image from response (type: {mime_type})")
                                    except Exception as e:
                                        self._log(f"Failed to process inline image: {e}")
                        
                        # Use the first image if available, otherwise create empty
                        if output_images:
                            output_image = output_images[0]
                            self._log(f"Using first of {len(output_images)} generated images")
                        else:
                            output_image = self.create_empty_image()
                            self._log("No images found in Gemini response")
                        
                        api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                        api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                        
                        return (text_content.strip(), output_image, api_request_str, api_response_str)
                    else:
                        error_msg = "No valid candidates in Gemini native response"
                        self._log(error_msg)
                        empty_image = self.create_empty_image()
                        return (f"## ERROR: {error_msg}", 
                               empty_image,
                               json.dumps(self.api_request, indent=2), 
                               json.dumps(data, indent=2))
                
                elif 'choices' in data and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    message = choice.get('message', {})
                    content = message.get('content', '')
                    
                    self._log("API call successful")
                    self._log(f"Response structure: choices[0] keys: {choice.keys()}")
                    self._log(f"Message keys: {message.keys()}")
                    
                    # Check if response contains an image URL or base64 image
                    output_image = None
                    
                    # Check for images in message structure (some APIs return images separately)
                    if 'images' in message:
                        images_data = message['images']
                        self._log(f"Found 'images' field in message: {type(images_data)}")
                        if isinstance(images_data, list) and len(images_data) > 0:
                            # Take the first image
                            img_data = images_data[0]
                            if isinstance(img_data, str):
                                if img_data.startswith('data:image'):
                                    output_image = self.base64_to_tensor(img_data)
                                    self._log("Image extracted from message.images field")
                                elif img_data.startswith('http'):
                                    self._log(f"Image URL found in message.images: {img_data}")
                    
                    # Check for image_url in message (some APIs use this format)
                    elif 'image_url' in message:
                        img_url = message['image_url']
                        self._log(f"Found 'image_url' field in message: {img_url}")
                        if isinstance(img_url, str) and img_url.startswith('data:image'):
                            output_image = self.base64_to_tensor(img_url)
                            self._log("Image extracted from message.image_url field")
                    
                    # Check in content field
                    elif isinstance(content, str):
                        # Check for image URL in response
                        if 'http' in content and ('.png' in content or '.jpg' in content or '.jpeg' in content):
                            # Extract URL from content
                            import re
                            url_pattern = r'(https?://[^\s]+\.(?:png|jpg|jpeg))'
                            match = re.search(url_pattern, content)
                            if match:
                                img_url = match.group(1)
                                self._log(f"Image URL detected in content: {img_url}")
                                # Note: Would need to download the image here
                        # Check for base64 image in response  
                        elif 'base64,' in content:
                            # Extract base64 string
                            import re
                            base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                            match = re.search(base64_pattern, content)
                            if match:
                                base64_str = match.group(0)
                                output_image = self.base64_to_tensor(base64_str)
                                self._log("Base64 image extracted from content")
                    
                    # Check for content as a dict/list (some APIs return structured content)
                    elif isinstance(content, (dict, list)):
                        self._log(f"Content is structured data: {type(content)}")
                        # Handle structured content that might contain images
                        if isinstance(content, dict) and 'image' in content:
                            img_data = content['image']
                            if isinstance(img_data, str) and img_data.startswith('data:image'):
                                output_image = self.base64_to_tensor(img_data)
                                self._log("Image extracted from structured content")
                    
                    # Check in the entire data response for other possible image fields
                    if output_image is None:
                        # Check usage field for image tokens (Gemini specific)
                        usage = data.get('usage', {})
                        if 'completion_tokens' in usage:
                            completion_tokens = usage.get('completion_tokens', 0)
                            # If completion_tokens is high (>1000), might indicate an image was generated
                            if completion_tokens > 1000:
                                self._log(f"High completion tokens ({completion_tokens}) detected, may indicate image generation")
                        
                        # Log full response structure for debugging
                        self._log(f"Full response keys: {data.keys()}")
                        if 'data' in data:
                            self._log(f"Found 'data' field in response: {type(data['data'])}")
                        if 'images' in data:
                            self._log(f"Found 'images' field in response: {type(data['images'])}")
                    
                    # Convert content to string if it's not
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    
                    # If no image found in response, create empty image
                    if output_image is None:
                        self._log("No image found in response, creating empty placeholder")
                        output_image = self.create_empty_image()
                    
                    api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                    api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                    
                    return (content, output_image, api_request_str, api_response_str)
                else:
                    error_msg = "No valid response content from API"
                    self._log(error_msg)
                    empty_image = self.create_empty_image()
                    return (f"## ERROR: {error_msg}", 
                           empty_image,
                           json.dumps(self.api_request, indent=2), 
                           json.dumps(data, indent=2))
            else:
                error_msg = f"API request failed with status code: {response.status_code}"
                self._log(error_msg)
                self._log(f"Response: {response.text}")
                empty_image = self.create_empty_image()
                return (f"## ERROR: {error_msg}\n\nResponse: {response.text}", 
                       empty_image,
                       json.dumps(self.api_request, indent=2), 
                       response.text)

        except requests.exceptions.Timeout:
            error_msg = "API request timed out after 120 seconds"
            self._log(error_msg)
            empty_image = self.create_empty_image()
            return (f"## ERROR: {error_msg}", 
                   empty_image,
                   json.dumps(self.api_request, indent=2), 
                   "{}")
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self._log(error_msg)
            empty_image = self.create_empty_image()
            return (f"## ERROR: {error_msg}", 
                   empty_image,
                   json.dumps(self.api_request, indent=2), 
                   "{}")
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            empty_image = self.create_empty_image()
            return (f"## ERROR: {error_msg}", 
                   empty_image,
                   json.dumps(self.api_request, indent=2), 
                   "{}")


class UnofficialGeminiStreamAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "sk-xxxx"}),
                "api_format": (["OpenAI Compatible", "Gemini Native"], {"default": "OpenAI Compatible"}),
                "model": ("STRING", {
                    "default": "claude-3-5-sonnet-20240620",
                    "multiline": False,
                    "placeholder": "e.g., claude-3-5-sonnet, gpt-4o, gemini-2.5-flash-image-preview"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "base_url": ("STRING", {
                    "default": "https://www.chataiapi.com/v1",
                    "multiline": False,
                    "placeholder": "API base URL"
                }),
                "stream": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "image", "stream_log", "api_request", "api_response")
    FUNCTION = "call_unofficial_stream_api"
    CATEGORY = "🤖 Gemini/Unofficial"

    def __init__(self):
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}
        self.last_image = None
        self.stream_chunks = []

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[UnofficialStreamAPI] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def tensor_to_base64(self, tensor):
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
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            self._log(f"Image converted to base64, size: {image.size}")
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            self._log(f"Error converting image to base64: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return None
    
    def base64_to_tensor(self, base64_str):
        """Convert base64 string to ComfyUI tensor format [B, H, W, C]"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(base64_str)
            
            # Open image with PIL
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Add batch dimension [H, W, C] -> [1, H, W, C]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self._log(f"Image decoded from base64, size: {image.size}")
            return img_tensor
            
        except Exception as e:
            self._log(f"Error converting base64 to tensor: {str(e)}")
            return None
    
    def create_empty_image(self, width=512, height=512):
        """Create an empty black image tensor"""
        # Create black image
        img_array = np.zeros((height, width, 3), dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    def call_unofficial_stream_api(self, prompt, api_key, api_format, model, temperature, max_tokens, 
                                  base_url, stream, seed,
                                  system_prompt="You are a helpful assistant.", 
                                  image1=None, image2=None, image3=None, image4=None, image5=None,
                                  top_p=0.95):
        
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}
        self.stream_chunks = []

        try:
            if not api_key:
                error_message = "Error: No API key provided. Please enter your API key."
                self._log(error_message)
                empty_image = self.create_empty_image()
                return (f"## ERROR: {error_message}", empty_image, "", "{}", "{}")

            # Collect all non-None images
            images = []
            for i, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    image_base64 = self.tensor_to_base64(img)
                    if image_base64:
                        images.append(image_base64)
                        self._log(f"Image {i} successfully converted to base64")
                    else:
                        self._log(f"Failed to convert image {i}")
            
            # Build request based on API format
            if api_format == "Gemini Native":
                # Gemini Native API format (Note: Gemini native doesn't support streaming for image generation)
                parts = [{"text": prompt}]
                
                # Add images to parts if any
                for img_base64 in images:
                    # Remove data URL prefix for Gemini format
                    if img_base64.startswith('data:image'):
                        img_data = img_base64.split(',')[1] if ',' in img_base64 else img_base64
                    else:
                        img_data = img_base64
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_data
                        }
                    })
                
                # Add system prompt to the beginning if provided
                if system_prompt and system_prompt != "You are a helpful assistant.":
                    parts.insert(0, {"text": system_prompt + "\n\n"})
                
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": parts
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "topP": top_p
                    }
                }
                
                # Construct URL for Gemini native format
                if ':streamGenerateContent' in base_url or ':generateContent' in base_url:
                    url = base_url
                else:
                    # Construct the URL with model name
                    base = base_url.rstrip('/')
                    endpoint = ":streamGenerateContent" if stream else ":generateContent"
                    if '/v1' in base:
                        url = f"{base}/models/{model}{endpoint}"
                    else:
                        url = f"{base}/v1/models/{model}{endpoint}"
                
                # Add API key to URL or headers
                if '?key=' not in url:
                    url = f"{url}?key={api_key}"
                    headers = {'Content-Type': 'application/json'}
                else:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {api_key}'
                    }
                
                self._log(f"Using Gemini Native API format (stream={stream})")
                self._log(f"Sending {len(images)} image(s) with prompt") if images else None
                
            else:
                # OpenAI Compatible format
                messages = []
                
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                # Build user message with text and images
                if images:
                    content = [{"type": "text", "text": prompt}]
                    for img_base64 in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_base64}
                        })
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                    self._log(f"Sending {len(images)} image(s) with prompt")
                else:
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": stream,
                    "seed": seed if seed > 0 else None
                }
                
                # Remove None values from payload
                payload = {k: v for k, v in payload.items() if v is not None}
                
                url = f"{base_url.rstrip('/')}/chat/completions"
                
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                    'User-Agent': 'ComfyUI-Gemini-Nodes/1.0.0',
                    'Content-Type': 'application/json'
                }
                
                self._log(f"Using OpenAI Compatible API format (stream={stream})")
            
            self.api_request = payload
            self._log(f"Sending request to {url}")
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                stream=stream,
                timeout=120
            )
            
            if response.status_code == 200:
                if api_format == "Gemini Native" and stream:
                    # Handle Gemini native streaming
                    full_content = ""
                    stream_log = []
                    output_images = []
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                self.stream_chunks.append(data)
                                
                                if 'candidates' in data and len(data['candidates']) > 0:
                                    candidate = data['candidates'][0]
                                    content_obj = candidate.get('content', {})
                                    parts = content_obj.get('parts', [])
                                    
                                    for part in parts:
                                        if 'text' in part:
                                            chunk = part['text']
                                            full_content += chunk
                                            stream_log.append(f"Text: {chunk[:50]}..." if len(chunk) > 50 else f"Text: {chunk}")
                                        elif 'inlineData' in part:
                                            inline_data = part['inlineData']
                                            mime_type = inline_data.get('mimeType', '')
                                            if 'image' in mime_type:
                                                stream_log.append(f"Image received: {mime_type}")
                                                # Process image
                                                img_data = inline_data.get('data', '')
                                                if img_data:
                                                    try:
                                                        if not img_data.startswith('data:'):
                                                            img_data = f"data:{mime_type};base64,{img_data}"
                                                        img_tensor = self.base64_to_tensor(img_data)
                                                        if img_tensor is not None:
                                                            output_images.append(img_tensor)
                                                    except Exception as e:
                                                        self._log(f"Failed to process streaming image: {e}")
                            except json.JSONDecodeError as e:
                                self._log(f"Failed to parse stream chunk: {e}")
                    
                    # Use the first image if available
                    output_image = output_images[0] if output_images else self.create_empty_image()
                    
                    self.api_response = {
                        "stream": True,
                        "chunks": len(self.stream_chunks),
                        "total_content": full_content,
                        "images_count": len(output_images)
                    }
                    
                    api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                    api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                    stream_log_str = "\n".join(stream_log[:50])
                    
                    return (full_content, output_image, stream_log_str, api_request_str, api_response_str)
                
                elif stream:
                    # Handle OpenAI compatible streaming
                    full_content = ""
                    stream_log = []
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str == "[DONE]":
                                    self._log("Stream completed")
                                    stream_log.append("[STREAM COMPLETED]")
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    self.stream_chunks.append(data)
                                    
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            chunk = delta['content']
                                            full_content += chunk
                                            stream_log.append(f"Chunk: {chunk}")
                                except json.JSONDecodeError:
                                    self._log(f"Failed to parse stream chunk: {data_str}")
                    
                    self.api_response = {
                        "stream": True,
                        "chunks": len(self.stream_chunks),
                        "total_content": full_content
                    }
                    
                    api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                    api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                    stream_log_str = "\n".join(stream_log[:50])
                    
                    output_image = self.create_empty_image()
                    return (full_content, output_image, stream_log_str, api_request_str, api_response_str)
                
                else:
                    # Non-streaming response
                    data = response.json()
                    self.api_response = data
                    
                    if api_format == "Gemini Native":
                        # Handle Gemini native non-streaming response
                        if 'candidates' in data and len(data['candidates']) > 0:
                            candidate = data['candidates'][0]
                            content_obj = candidate.get('content', {})
                            parts = content_obj.get('parts', [])
                            
                            text_content = ""
                            output_images = []
                            
                            for part in parts:
                                if 'text' in part:
                                    text_content += part['text']
                                elif 'inlineData' in part:
                                    inline_data = part['inlineData']
                                    mime_type = inline_data.get('mimeType', '')
                                    img_data = inline_data.get('data', '')
                                    
                                    if img_data and 'image' in mime_type:
                                        try:
                                            if not img_data.startswith('data:'):
                                                img_data = f"data:{mime_type};base64,{img_data}"
                                            img_tensor = self.base64_to_tensor(img_data)
                                            if img_tensor is not None:
                                                output_images.append(img_tensor)
                                                self._log(f"Extracted image from response (type: {mime_type})")
                                        except Exception as e:
                                            self._log(f"Failed to process inline image: {e}")
                            
                            output_image = output_images[0] if output_images else self.create_empty_image()
                            
                            api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                            api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                            
                            return (text_content.strip(), output_image, "Non-stream response", api_request_str, api_response_str)
                    
                    elif 'choices' in data and len(data['choices']) > 0:
                        # Handle OpenAI compatible non-streaming response
                        content = data['choices'][0]['message']['content']
                        self._log("API call successful")
                        
                        api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                        api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                        
                        output_image = self.create_empty_image()
                        return (content, output_image, "Non-stream response", api_request_str, api_response_str)
                    
                    else:
                        error_msg = "No valid response content from API"
                        self._log(error_msg)
                        empty_image = self.create_empty_image()
                        return (f"## ERROR: {error_msg}", empty_image, "", 
                               json.dumps(self.api_request, indent=2), 
                               json.dumps(data, indent=2))
            else:
                error_msg = f"API request failed with status code: {response.status_code}"
                self._log(error_msg)
                self._log(f"Response: {response.text}")
                empty_image = self.create_empty_image()
                return (f"## ERROR: {error_msg}\n\nResponse: {response.text}", empty_image, "", 
                       json.dumps(self.api_request, indent=2), 
                       response.text)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            empty_image = self.create_empty_image()
            return (f"## ERROR: {error_msg}", empty_image, "", 
                   json.dumps(self.api_request, indent=2), 
                   "{}")


NODE_CLASS_MAPPINGS = {
    "UnofficialGeminiAPI": UnofficialGeminiAPI,
    "UnofficialGeminiStreamAPI": UnofficialGeminiStreamAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnofficialGeminiAPI": "Unofficial API Call",
    "UnofficialGeminiStreamAPI": "Unofficial Stream API Call",
}