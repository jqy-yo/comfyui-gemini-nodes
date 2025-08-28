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
                "model": ("STRING", {
                    "default": "claude-3-5-sonnet-20240620",
                    "multiline": False,
                    "placeholder": "e.g., claude-3-5-sonnet, gpt-4o, gemini-pro"
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
                "image": ("IMAGE",),
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

    def call_unofficial_api(self, prompt, api_key, model, temperature, max_tokens, base_url, seed,
                           system_prompt="You are a helpful assistant.", 
                           image=None, top_p=0.95, 
                           frequency_penalty=0.0, presence_penalty=0.0):
        
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}

        try:
            if not api_key:
                error_message = "Error: No API key provided. Please enter your API key."
                self._log(error_message)
                empty_image = self.create_empty_image()
                return (f"## ERROR: {error_message}", empty_image, "{}", "{}")

            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            if image is not None:
                image_base64 = self.tensor_to_base64(image)
                if image_base64:
                    self._log("Image successfully converted to base64")
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            }
                        ]
                    })
                else:
                    self._log("Failed to convert image, sending text only")
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
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

            self.api_request = payload
            self._log(f"Sending request to {base_url}/chat/completions")
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'User-Agent': 'ComfyUI-Gemini-Nodes/1.0.0',
                'Content-Type': 'application/json'
            }

            url = f"{base_url.rstrip('/')}/chat/completions"
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                self.api_response = data
                
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    self._log("API call successful")
                    
                    # Check if response contains an image URL or base64 image
                    output_image = None
                    if isinstance(content, str):
                        # Check for image URL in response
                        if 'http' in content and ('.png' in content or '.jpg' in content or '.jpeg' in content):
                            # Extract URL from content (you may need to parse this based on actual response format)
                            self._log("Image URL detected in response")
                        # Check for base64 image in response  
                        elif 'base64,' in content:
                            # Extract base64 string
                            import re
                            base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                            match = re.search(base64_pattern, content)
                            if match:
                                base64_str = match.group(0)
                                output_image = self.base64_to_tensor(base64_str)
                                self._log("Base64 image extracted from response")
                    
                    # If no image found in response, create empty image
                    if output_image is None:
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
                "model": ("STRING", {
                    "default": "claude-3-5-sonnet-20240620",
                    "multiline": False,
                    "placeholder": "e.g., claude-3-5-sonnet, gpt-4o, gemini-pro"
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
                "image": ("IMAGE",),
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

    def call_unofficial_stream_api(self, prompt, api_key, model, temperature, max_tokens, 
                                  base_url, stream, seed,
                                  system_prompt="You are a helpful assistant.", 
                                  image=None, top_p=0.95):
        
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

            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            if image is not None:
                image_base64 = self.tensor_to_base64(image)
                if image_base64:
                    self._log("Image successfully converted to base64")
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            }
                        ]
                    })
                else:
                    self._log("Failed to convert image, sending text only")
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
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

            self.api_request = payload
            self._log(f"Sending {'stream' if stream else 'non-stream'} request to {base_url}/chat/completions")
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'User-Agent': 'ComfyUI-Gemini-Nodes/1.0.0',
                'Content-Type': 'application/json'
            }

            url = f"{base_url.rstrip('/')}/chat/completions"
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                stream=stream,
                timeout=120
            )
            
            if response.status_code == 200:
                if stream:
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
                    
                    # Create empty image for now (could be enhanced to extract images from stream)
                    output_image = self.create_empty_image()
                    return (full_content, output_image, stream_log_str, api_request_str, api_response_str)
                else:
                    data = response.json()
                    self.api_response = data
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        self._log("API call successful")
                        
                        api_request_str = json.dumps(self.api_request, indent=2, ensure_ascii=False)
                        api_response_str = json.dumps(self.api_response, indent=2, ensure_ascii=False)
                        
                        # Check for image in response
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