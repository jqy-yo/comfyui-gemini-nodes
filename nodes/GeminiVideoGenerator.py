import os
import time
import json
import torch
import numpy as np
import tempfile
import traceback
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, Tuple, Dict, Any
from io import BytesIO
import cv2


class GeminiVideoGenerator:
    """
    Generate videos using Google's Veo models through Gemini API.
    Supports text-to-video and image-to-video generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic drone shot of a red convertible driving along a coastal road at sunset"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["veo-3.0-generate-preview", "veo-3.0-fast-generate-preview", "veo-2.0-generate-001"], {
                    "default": "veo-3.0-generate-preview"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {
                    "default": "16:9"
                }),
                "person_generation": (["default", "allow", "dont_allow"], {
                    "default": "default"
                }),
                "max_wait_minutes": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "poll_interval_seconds": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 60,
                    "step": 5,
                    "display": "number"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Elements to exclude from the video (e.g., blurry, low quality, distorted)"
                }),
                "initial_image": ("IMAGE", {
                    "tooltip": "Optional starting image for video generation"
                }),
                "save_path": ("STRING", {
                    "default": "",
                    "placeholder": "Optional path to save the generated video (e.g., /path/to/output.mp4)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "preview_frame", "generation_info", "api_request", "api_response")
    FUNCTION = "generate_video"
    CATEGORY = "🤖 Gemini"

    def __init__(self):
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}

    def _log(self, message):
        """Log messages for debugging"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[GeminiVideoGenerator] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _tensor_to_pil(self, image_tensor):
        """Convert ComfyUI image tensor to PIL Image for API"""
        try:
            if image_tensor is None:
                return None
                
            # Ensure tensor is in correct format [1, H, W, 3]
            if len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1:
                # Get first frame image
                image_np = image_tensor[0].cpu().numpy()
                
                # Convert to uint8 format for PIL
                image_np = (image_np * 255).astype(np.uint8)
                
                # Create PIL image
                pil_image = Image.fromarray(image_np)
                
                self._log(f"Converted tensor to PIL image: {pil_image.width}x{pil_image.height}")
                return pil_image
            else:
                self._log(f"Image format incorrect: {image_tensor.shape}")
                return None
        except Exception as e:
            self._log(f"Error converting tensor to PIL: {str(e)}")
            return None

    def _extract_video_frame(self, video_path):
        """Extract a preview frame from the generated video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get middle frame
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to ComfyUI tensor format [1, H, W, 3]
                img_array = frame_rgb.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                
                self._log(f"Extracted preview frame from video: {frame_rgb.shape}")
                return img_tensor
            else:
                self._log("Failed to extract frame from video")
                return self._create_placeholder_image()
                
        except Exception as e:
            self._log(f"Error extracting video frame: {str(e)}")
            return self._create_placeholder_image()

    def _create_placeholder_image(self, width=1280, height=720, message="Video Generated"):
        """Create a placeholder image when video frame extraction fails"""
        try:
            # Create a dark gray image with text
            image = Image.new('RGB', (width, height), color=(64, 64, 64))
            
            # Draw text in the center
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to use a better font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), message, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            position = ((width - text_width) // 2, (height - text_height) // 2)
            
            # Draw text
            draw.text(position, message, fill=(255, 255, 255), font=font)
            
            # Add info text
            info_text = f"Video saved. Check video_path output."
            info_bbox = draw.textbbox((0, 0), info_text, font=font)
            info_width = info_bbox[2] - info_bbox[0]
            info_position = ((width - info_width) // 2, height // 2 + 60)
            draw.text(info_position, info_text, fill=(200, 200, 200), font=font)
            
            # Convert to tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            return img_tensor
            
        except Exception as e:
            self._log(f"Error creating placeholder image: {str(e)}")
            # Return a simple black image as last resort
            img_array = np.zeros((height, width, 3), dtype=np.float32)
            return torch.from_numpy(img_array).unsqueeze(0)


    def generate_video(self, prompt, api_key, model, aspect_ratio, person_generation,
                       max_wait_minutes, poll_interval_seconds,
                       negative_prompt="", initial_image=None, save_path=""):
        """Generate video using Veo models"""
        
        # Reset logging and API tracking
        self.log_messages = []
        self.api_request = {}
        self.api_response = {}
        
        try:
            # Validate API key
            if not api_key:
                error_msg = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_msg)
                return ("", self._create_placeholder_image(), error_msg, "{}", "{}")
            
            # Create client
            self._log(f"Initializing Gemini client with model: {model}")
            client = genai.Client(api_key=api_key)
            
            # Prepare generation config using GenerateVideosConfig
            config_params = {
                "aspect_ratio": aspect_ratio,  # Keep original format like "16:9"
            }
            
            # Add negative prompt if provided
            if negative_prompt and negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()
                self._log(f"Using negative prompt: {negative_prompt[:100]}...")
            
            # Create config object
            generation_config = types.GenerateVideosConfig(**config_params)
            
            # Prepare generation parameters
            generation_params = {
                "model": model,
                "prompt": prompt,
                "config": generation_config
            }
            
            # Process initial image if provided
            if initial_image is not None:
                # Convert image tensor to PIL for API
                pil_image = self._tensor_to_pil(initial_image)
                if pil_image:
                    generation_params["image"] = pil_image
                    self._log("Using initial image for video generation")
            
            # Store API request
            self.api_request = {
                "model": model,
                "config": {
                    "prompt": prompt,
                    "aspectRatio": aspect_ratio,
                    "negativePrompt": negative_prompt if negative_prompt else None,
                    "personGeneration": person_generation,
                    "hasInitialImage": initial_image is not None
                }
            }
            
            # Start video generation (asynchronous operation)
            self._log(f"Starting video generation with prompt: {prompt[:100]}...")
            
            try:
                # Create the generation request with proper parameters
                operation = client.models.generate_videos(**generation_params)
                
                self._log(f"Video generation started. Operation: {operation.name if hasattr(operation, 'name') else 'Unknown'}")
                
                # Poll for completion
                max_wait_seconds = int(max_wait_minutes * 60)
                start_time = time.time()
                poll_count = 0
                
                self._log(f"Starting to poll for video generation completion (max wait: {max_wait_minutes} minutes)")
                
                while True:
                    elapsed_time = time.time() - start_time
                    poll_count += 1
                    
                    # Check timeout
                    if elapsed_time > max_wait_seconds:
                        # Determine final operation state for debugging
                        final_state = "unknown"
                        if hasattr(operation, 'done'):
                            final_state = f"done={operation.done}"
                        elif hasattr(operation, 'status'):
                            final_state = f"status={operation.status}"
                        
                        error_msg = f"Video generation timed out after {max_wait_minutes} minutes. Try increasing max_wait_minutes parameter."
                        self._log(error_msg)
                        self._log(f"Total polls: {poll_count}, Last operation state: {final_state}")
                        self._log(f"Consider setting max_wait_minutes to 15-20 for Veo 3.0 models")
                        return ("", self._create_placeholder_image(), error_msg, 
                               json.dumps(self.api_request, indent=2), 
                               json.dumps({"error": "timeout", "elapsed_minutes": elapsed_time/60, "polls": poll_count, "final_state": final_state}, indent=2))
                    
                    # Check operation status - try multiple methods
                    is_done = False
                    
                    # Debug: Log operation type and attributes on first poll
                    if poll_count == 1:
                        self._log(f"Operation type: {type(operation)}")
                        if hasattr(operation, '__dict__'):
                            attrs = list(operation.__dict__.keys())
                            self._log(f"Operation attributes: {attrs[:10]}")  # Show first 10 attributes
                    
                    # Try different methods to check if done
                    if hasattr(operation, 'done'):
                        is_done = operation.done
                        if poll_count % 3 == 0:  # Log every 3rd poll
                            self._log(f"operation.done = {operation.done}")
                    elif hasattr(operation, 'is_done'):
                        is_done = operation.is_done()
                    elif hasattr(operation, 'status'):
                        is_done = operation.status == 'DONE'
                        if poll_count % 3 == 0:
                            self._log(f"operation.status = {operation.status}")
                    
                    if is_done:
                        self._log(f"Video generation completed after {elapsed_time:.1f} seconds!")
                        break
                    
                    # Calculate progress estimate (video generation typically takes 3-8 minutes)
                    estimated_progress = min(100, (elapsed_time / (5 * 60)) * 100)  # Assume 5 minutes typical
                    
                    # Wait before next poll
                    remaining_time = max_wait_seconds - elapsed_time
                    self._log(f"Poll #{poll_count}: Video generation in progress... (~{estimated_progress:.0f}% | {elapsed_time:.0f}s elapsed, {remaining_time:.0f}s remaining)")
                    
                    # Show additional info every 5th poll
                    if poll_count % 5 == 0:
                        self._log(f"Still processing... Veo video generation typically takes 3-8 minutes for high quality.")
                    
                    time.sleep(poll_interval_seconds)
                    
                    # Refresh operation status by getting it again
                    try:
                        operation_name = operation.name if hasattr(operation, 'name') else str(operation)
                        self._log(f"Refreshing operation status: {operation_name}")
                        
                        # Try to get updated operation status
                        refreshed_operation = client.operations.get(name=operation_name)
                        if refreshed_operation:
                            operation = refreshed_operation
                            
                            # Check if operation has error
                            if hasattr(operation, 'error') and operation.error:
                                error_msg = f"Operation error: {operation.error}"
                                self._log(error_msg)
                                return ("", self._create_placeholder_image(), error_msg,
                                       json.dumps(self.api_request, indent=2),
                                       json.dumps({"error": str(operation.error)}, indent=2))
                                   
                    except AttributeError as e:
                        # The operation object might not have the expected attributes
                        self._log(f"Warning: Operation refresh not available: {str(e)}")
                        # Check if we can access done status directly
                        try:
                            # Try alternative method to check status
                            if hasattr(operation, '__dict__'):
                                self._log(f"Operation attributes: {list(operation.__dict__.keys())}")
                        except:
                            pass
                    except Exception as e:
                        self._log(f"Warning: Error refreshing operation status: {str(e)}")
                        # Continue anyway, the operation might still be running
                
                # Get the result from the completed operation
                if operation.done:
                    # Store API response
                    self.api_response = {
                        "status": "completed",
                        "videos": []
                    }
                    
                    # Access the response from the operation
                    if hasattr(operation, 'response'):
                        response = operation.response
                        
                        # Check for generated videos
                        if hasattr(response, 'generated_videos') and response.generated_videos:
                            video = response.generated_videos[0]  # Get first video
                            
                            video_info = {
                                "video_file": None,
                                "metadata": {}
                            }
                            
                            # Get video file reference
                            if hasattr(video, 'video'):
                                video_file = video.video
                                video_info["video_file"] = str(video_file) if video_file else None
                                
                                # Download the video using the file reference
                                if video_file:
                                    try:
                                        # Download using client.files.download
                                        self._log(f"Downloading video file: {video_file}")
                                        downloaded_file = client.files.download(file=video_file)
                                        
                                        # Save to specified path or temp location
                                        if not save_path:
                                            temp_dir = tempfile.gettempdir()
                                            timestamp = int(time.time())
                                            save_path = os.path.join(temp_dir, f"gemini_veo_video_{timestamp}.mp4")
                                        
                                        # Save the downloaded content
                                        downloaded_file.save(save_path)
                                        self._log(f"Video saved to: {save_path}")
                                        video_path = save_path
                                    except Exception as e:
                                        self._log(f"Error downloading video: {str(e)}")
                                        video_path = None
                                else:
                                    video_path = None
                            
                            self.api_response["videos"].append(video_info)
                            
                            if video_path and os.path.exists(video_path):
                                # Extract preview frame
                                preview_frame = self._extract_video_frame(video_path)
                                
                                # Generate info
                                generation_info = f"Video generated successfully!\n"
                                generation_info += f"Model: {model}\n"
                                generation_info += f"Aspect Ratio: {aspect_ratio}\n"
                                generation_info += f"Duration: ~8 seconds\n"
                                generation_info += f"Resolution: 720p\n"
                                generation_info += f"Path: {video_path}"
                                
                                return (video_path, preview_frame, generation_info,
                                       json.dumps(self.api_request, indent=2),
                                       json.dumps(self.api_response, indent=2))
                            else:
                                error_msg = "Failed to download or save generated video"
                                self._log(error_msg)
                                return ("", self._create_placeholder_image(), error_msg,
                                       json.dumps(self.api_request, indent=2),
                                       json.dumps(self.api_response, indent=2))
                        else:
                            error_msg = "No generated videos in response"
                            self._log(error_msg)
                            return ("", self._create_placeholder_image(), error_msg,
                                   json.dumps(self.api_request, indent=2),
                                   json.dumps(self.api_response, indent=2))
                    else:
                        error_msg = "Operation completed but no response data"
                        self._log(error_msg)
                        return ("", self._create_placeholder_image(), error_msg,
                               json.dumps(self.api_request, indent=2),
                               json.dumps(self.api_response, indent=2))
                else:
                    error_msg = "Operation did not complete successfully"
                    self._log(error_msg)
                    return ("", self._create_placeholder_image(), error_msg,
                           json.dumps(self.api_request, indent=2),
                           json.dumps({"error": "operation_not_done"}, indent=2))
                    
            except Exception as e:
                error_msg = f"API call error: {str(e)}"
                self._log(error_msg)
                self._log(f"Full error: {traceback.format_exc()}")
                
                # Check if it's a model access error
                if "not found" in str(e).lower() or "permission" in str(e).lower():
                    error_msg += "\n\nNote: Veo models may require special access. Please check:\n"
                    error_msg += "1. Your API key has access to Veo models\n"
                    error_msg += "2. The model name is correct\n"
                    error_msg += "3. Your region supports video generation"
                
                return ("", self._create_placeholder_image(), error_msg,
                       json.dumps(self.api_request, indent=2),
                       json.dumps({"error": str(e)}, indent=2))
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._log(error_msg)
            traceback.print_exc()
            
            return ("", self._create_placeholder_image(), error_msg,
                   json.dumps(self.api_request, indent=2) if self.api_request else "{}",
                   json.dumps({"error": str(e)}, indent=2))


# Node registration
NODE_CLASS_MAPPINGS = {
    "GeminiVideoGenerator": GeminiVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiVideoGenerator": "🎬 Gemini Video Generator (Veo)"
}