import os
import tempfile
import cv2
import numpy as np
import torch
import base64
import json
import time
import mimetypes
import shutil
from PIL import Image
from datetime import timedelta
from comfy.utils import ProgressBar
import traceback
from google import genai
from google.genai import types


class GeminiVideoCaptioner:
    """
    Node for captioning videos using Google's Gemini API.

    Note: All videos (from file or image batch) are converted to WebM format with a size limit
    of just under 30MB to ensure compatibility with the Gemini API payload limitations.
    Video quality will be adjusted automatically to meet the size requirement.
    """
    
    def _clean_schema_for_gemini(self, schema: dict) -> dict:
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
            print("[GeminiVideoCaptioner] Removed 'additionalProperties' field from schema for Gemini API compatibility")
        return cleaned_schema

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {
                    "default": "gemini-2.0-flash",
                    "multiline": False,
                    "placeholder": "e.g., gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-pro"
                }),
                "frames_per_second": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "max_duration_minutes": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 45.0, "step": 0.1}),
                "prompt": ("STRING", {
                    "default": "Describe this video scene in detail. Include any important actions, subjects, settings, and atmosphere.",
                    "multiline": True
                }),
                "process_audio": (["false", "true"], {"default": "false"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 50, "max": 8192, "step": 10}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),  # INT32 max value
            },
            "optional": {
                "video_path": ("STRING", {"default": ""}),
                "image": ("IMAGE", {}),
                "api_version": (["auto", "v1", "v1beta", "v1alpha"], {"default": "auto"}),
                "use_structured_output": ("BOOLEAN", {"default": False}),
                "output_schema": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "description": {"type": "string"},\n    "objects": {\n      "type": "array",\n      "items": {"type": "string"}\n    },\n    "actions": {\n      "type": "array",\n      "items": {"type": "string"}\n    },\n    "mood": {"type": "string"},\n    "technical_details": {"type": "string"}\n  },\n  "required": ["description", "objects", "actions", "mood"]\n}',
                    "placeholder": "Enter JSON Schema for structured output..."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "sampled_frame", "raw_json", "api_request", "api_response")
    FUNCTION = "generate_video_caption"
    CATEGORY = "🤖 Gemini"

    def generate_video_caption(self, api_key, model, frames_per_second, max_duration_minutes,
                               prompt, process_audio, temperature, max_output_tokens, top_p, top_k, seed,
                               video_path=None, image=None, api_version="auto", use_structured_output=False, output_schema=None):
        if not api_key:
            raise ValueError("Gemini API key is required")

        # Check if we have either a video path or image input
        if (not video_path or not os.path.exists(video_path)) and image is None:
            raise ValueError("Either a valid video path or image input is required")

        # Validate model-specific limitations
        process_audio = process_audio == "true"
        if model == "gemini-1.0-pro-vision" and process_audio:
            print("[Warning] Gemini 1.0 Pro Vision does not support audio processing. Ignoring audio.")
            process_audio = False

        # Calculate max frames based on model and duration limits
        max_seconds = int(max_duration_minutes * 60)
        if model == "gemini-1.0-pro-vision":
            max_seconds = min(max_seconds, 120)  # 2 minutes max for Gemini 1.0
        else:
            max_seconds = min(max_seconds, 45 * 60)  # 45 minutes max for Gemini 1.5+

        # Process based on input type (video file or image batch)
        if video_path and os.path.exists(video_path):
            # Processing a video file
            mime_type = mimetypes.guess_type(video_path)[0]
            if not mime_type or not mime_type.startswith("video/"):
                if mime_type:
                    raise ValueError(f"Unsupported file type: {mime_type}. Please provide a video file.")
                else:
                    # Try to infer from extension
                    ext = os.path.splitext(video_path)[1].lower()
                    if ext in ['.mp4', '.avi', '.mov', '.webm', '.wmv', '.flv', '.mpg', '.mpeg']:
                        mime_type = f"video/{ext[1:]}"
                    else:
                        raise ValueError(f"Unrecognized video format: {ext}. Please provide a supported video file.")

            # Extract video info
            video_info = self.get_video_info(video_path)
            video_duration = min(video_info['duration'], max_seconds)
            print(f"[GeminiVideoCaptioner] Video duration: {timedelta(seconds=video_duration)}")

            # Convert to WebM with size limit for direct API submission
            print(f"[GeminiVideoCaptioner] Converting video to WebM format...")
            webm_path = self.convert_to_webm(video_path)

            # Extract a middle frame for preview
            cap = cv2.VideoCapture(video_path)
            middle_frame_pos = int(video_info['frame_count'] / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
            ret, middle_frame = cap.read()
            cap.release()

            if not ret:
                # Fallback to first frame if middle frame extraction fails
                cap = cv2.VideoCapture(video_path)
                ret, middle_frame = cap.read()
                cap.release()

            if ret:
                # Convert BGR to RGB
                middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)
                sample_frame_np = np.array(middle_frame).astype(np.float32) / 255.0
                sample_frame_tensor = torch.from_numpy(sample_frame_np)[None,]
            else:
                # Create a blank frame if extraction fails
                sample_frame_np = np.zeros((512, 512, 3), dtype=np.float32)
                sample_frame_tensor = torch.from_numpy(sample_frame_np)[None,]

            # If WebM conversion failed, fall back to frame extraction
            raw_json = ""  # Initialize raw_json
            if webm_path is None:
                print(f"[GeminiVideoCaptioner] WebM conversion failed, falling back to frame extraction...")
                frames, frame_timestamps = self.extract_frames(video_path, frames_per_second, max_seconds)
                if not frames:
                    raise ValueError(f"Failed to extract frames from video: {video_path}")

                print(f"[GeminiVideoCaptioner] Extracted {len(frames)} frames at {frames_per_second} fps")

                # Generate captions for frames
                caption, raw_json = self.get_caption_with_frames(
                    api_key,
                    frames,
                    prompt,
                    model,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                    seed,
                    api_version,
                    use_structured_output,
                    output_schema
                )
            else:
                # Use the WebM file for captioning
                mime_type = "video/webm"
                print(f"[GeminiVideoCaptioner] Using WebM video for captioning")

                # Get caption using the WebM file
                result = self.get_caption_with_video_file(
                    api_key,
                    webm_path,
                    mime_type,
                    prompt,
                    model,
                    process_audio,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                    seed,
                    api_version,
                    use_structured_output,
                    output_schema
                )

                # Clean up temporary WebM file
                os.unlink(webm_path)
                
                # Handle result based on format
                if isinstance(result, tuple) and len(result) == 4:
                    caption, raw_json, api_request, api_response = result
                elif isinstance(result, tuple) and len(result) == 3:
                    caption, api_request, api_response = result
                    raw_json = ""
                elif isinstance(result, tuple) and len(result) == 2:
                    caption, raw_json = result
                    api_request, api_response = {}, {}
                else:
                    caption = result if isinstance(result, str) else str(result)
                    raw_json = ""
                    api_request, api_response = {}, {}

            return (caption, sample_frame_tensor, raw_json, json.dumps(api_request, indent=2), json.dumps(api_response, indent=2))

        else:
            # Processing an image batch
            if image is None:
                raise ValueError("Image input is required when no video path is provided")

            print(f"[GeminiVideoCaptioner] Processing image batch with {image.shape[0]} frames")

            # Convert tensor images to numpy arrays
            frames = []

            # Extract frames from the image tensor
            for i in range(image.shape[0]):
                # Convert tensor to numpy array (0-255 range)
                img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
                frames.append(img_np)

            # Choose a middle frame as sample output
            middle_idx = len(frames) // 2
            sample_frame_tensor = image[middle_idx].unsqueeze(0) if image.shape[0] > 0 else None
            raw_json = ""  # Initialize raw_json

            # If only one frame is provided, always send as JPEG directly (more efficient)
            if len(frames) == 1:
                print(f"[GeminiVideoCaptioner] Single frame input detected for {model}")
                print(f"[GeminiVideoCaptioner] Using direct JPEG processing (more efficient for single images)")
                
                # Get image dimensions for debugging
                if len(frames) > 0:
                    h, w = frames[0].shape[:2] if len(frames[0].shape) >= 2 else (0, 0)
                    print(f"[GeminiVideoCaptioner] Image dimensions: {w}x{h}")
                
                try:
                    result = self.get_caption_with_frames(
                        api_key,
                        frames,  # This will be a list with one frame
                        prompt,
                        model,
                        temperature,
                        max_output_tokens,
                        top_p,
                        top_k,
                        seed,
                        api_version,
                        use_structured_output,
                        output_schema
                    )
                    
                    # Handle different return formats
                    if isinstance(result, tuple):
                        if len(result) == 4:
                            caption, raw_json, api_request, api_response = result
                        elif len(result) == 3:
                            caption, api_request, api_response = result
                            raw_json = ""
                        else:
                            caption = result[0] if len(result) > 0 else "Error"
                            raw_json = result[1] if len(result) > 1 else ""
                    else:
                        caption = str(result)
                        raw_json = ""
                        
                except Exception as e:
                    error_msg = str(e)
                    if "500 INTERNAL" in error_msg:
                        caption = f"Error: Image processing failed with 500 error. Model '{model}' may have issues with this image."
                        print("[GeminiVideoCaptioner] ⚠️ Tips for resolving 500 error with single image:")
                        print(f"  1. Current model: {model}")
                        print("  2. Try using gemini-1.5-flash instead of experimental models")
                        print("  3. Check image size - very large images (>10MB) may cause issues")
                        print("  4. Ensure image format is standard (JPEG, PNG)")
                        print("  5. Some models may have temporary issues - try again later")
                    else:
                        caption = f"Error processing single image: {error_msg}"
                    raw_json = ""
            else:
                # Original logic: try to create WebM, then send video file or fallback to frames
                print(f"[GeminiVideoCaptioner] Creating WebM video from {len(frames)} frames...")
                webm_path = self.create_webm_from_frames(frames, fps=frames_per_second)

                if webm_path is None:
                    print(f"[GeminiVideoCaptioner] WebM creation failed, falling back to frame processing...")
                    caption, raw_json = self.get_caption_with_frames(
                        api_key,
                        frames,
                        prompt,
                        model,
                        temperature,
                        max_output_tokens,
                        top_p,
                        top_k,
                        seed,
                        api_version,
                        use_structured_output,
                        output_schema
                    )
                else:
                    mime_type = "video/webm"
                    print(f"[GeminiVideoCaptioner] Using WebM video for captioning")
                    result = self.get_caption_with_video_file(
                        api_key,
                        webm_path,
                        mime_type,
                        prompt,
                        model,
                        process_audio,
                        temperature,
                        max_output_tokens,
                        top_p,
                        top_k,
                        seed,
                        api_version,
                        use_structured_output,
                        output_schema
                    )
                    os.unlink(webm_path) # Clean up temporary WebM file
                    
                    # Handle result based on format
                    if isinstance(result, tuple) and len(result) == 4:
                        caption, raw_json, api_request, api_response = result
                    elif isinstance(result, tuple) and len(result) == 3:
                        caption, api_request, api_response = result
                        raw_json = ""
                    elif isinstance(result, tuple) and len(result) == 2:
                        caption, raw_json = result
                        api_request, api_response = {}, {}
                    else:
                        caption = result if isinstance(result, str) else str(result)
                        raw_json = ""
                        api_request, api_response = {}, {}

            return (caption, sample_frame_tensor, raw_json, json.dumps(api_request, indent=2), json.dumps(api_response, indent=2))

    def get_video_info(self, video_path):
        """Get basic information about the video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }

    def extract_frames(self, video_path, target_fps, max_seconds):
        """Extract frames at specified fps from a video file, up to max_seconds"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate how many frames to skip to achieve target_fps
        if original_fps <= 0:
            print("[Warning] Could not determine video FPS. Using 30 fps as default.")
            original_fps = 30

        frame_interval = max(1, round(original_fps / target_fps))

        # Calculate total duration and limit frames
        duration = min(frame_count / original_fps, max_seconds)
        max_frames = int(duration * target_fps)

        frames = []
        timestamps = []  # in seconds

        progress = ProgressBar(max_frames)
        frame_idx = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break

            # Process frame at intervals to achieve target_fps
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                # Calculate timestamp in seconds
                timestamp = frame_idx / original_fps
                timestamps.append(timestamp)

                progress.update_absolute(frame_count)
                frame_count += 1

            frame_idx += 1

        cap.release()
        return frames, timestamps

    def get_gemini_caption(self, api_key, frames, timestamps, video_path, mime_type, prompt, model,
                           process_audio, temperature, max_output_tokens, top_p, top_k, seed):
        """Send frames to Gemini API and get caption response"""

        # For newer models that support direct video input
        if model in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
            return self.get_caption_with_video_file(
                api_key, video_path, mime_type, prompt, model, process_audio,
                temperature, max_output_tokens, top_p, top_k, seed
            )
        else:
            # For gemini-1.0-pro-vision, we need to send individual frames
            return self.get_caption_with_frames(
                api_key, frames, prompt, model, temperature, max_output_tokens,
                top_p, top_k, seed
            )

    def get_caption_with_frames(self, api_key, frames, prompt, model, temperature, max_output_tokens, top_p, top_k,
                                seed, api_version="auto", use_structured_output=False, output_schema=None):
        """Send individual frames to Gemini API using google-genai SDK"""
        # Create client with API key
        client = genai.Client(api_key=api_key)
        
        print(f"[GeminiVideoCaptioner] Using google-genai SDK for {model}")

        content_parts = []

        # Set frame limit based on model
        max_frames = 20  # Default for Gemini 1.0

        if "gemini-2.5" in model or "models/gemini-2.5" in model:
            max_frames = min(len(frames), 100)  # Higher limit for 2.5 models
            print(f"[GeminiVideoCaptioner] Using max_frames={max_frames} for Gemini 2.5 model")
        elif model.startswith("gemini-1.5") or model.startswith("gemini-2.0"):
            max_frames = min(len(frames), 60)  # Can handle more frames for newer models
        else:
            max_frames = min(len(frames), 20)  # Limit to 20 frames for 1.0
        
        # For 2.5 models or newer, put the frames first, then the prompt
        if not (model.startswith("gemini-1.0") or "gemini-1.0" in model):
            # For newer models, we should put the frames first, then the prompt
            # This follows Google's recommendation for best results
            pass
        else:
            # Put the prompt first for older models
            content_parts.append({"text": prompt})

        frames_to_process = frames[:max_frames]

        print(
            f"[GeminiVideoCaptioner] Processing {len(frames_to_process)} out of {len(frames)} total frames for {model}")
        progress = ProgressBar(len(frames_to_process))

        # Add frames as mimetype image/jpeg
        for i, frame in enumerate(frames_to_process):
            # Convert numpy array to PIL Image
            img = Image.fromarray(frame)

            # Save image to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                img.save(temp, format="JPEG")
                temp_filename = temp.name

            # Read the image file and encode to base64
            with open(temp_filename, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Add to content parts
            content_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            })

            # Delete temp file
            os.unlink(temp_filename)
            progress.update_absolute(i)

        # For newer models, add the prompt after the frames
        if not (model.startswith("gemini-1.0") or "gemini-1.0" in model):
            content_parts.append({"text": prompt})

        # Parse schema if structured output is enabled
        response_schema = None
        if use_structured_output and output_schema:
            try:
                response_schema = json.loads(output_schema)
                # Clean the schema for Gemini API
                response_schema = self._clean_schema_for_gemini(response_schema)
                print(f"[GeminiVideoCaptioner] Using structured output with cleaned schema")
            except json.JSONDecodeError as e:
                print(f"[GeminiVideoCaptioner] Warning: Invalid JSON schema, falling back to regular output: {e}")
                use_structured_output = False

        # Create generation config for SDK
        gen_config_dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "candidate_count": 1
        }
        
        # Add structured output if enabled
        if use_structured_output and response_schema:
            gen_config_dict["response_mime_type"] = "application/json"
            gen_config_dict["response_schema"] = response_schema
        
        gen_config = types.GenerateContentConfig(**gen_config_dict)
        
        # Send request using SDK
        print(f"[GeminiVideoCaptioner] Sending request to Gemini API ({model})...")
        result = self._send_api_request(client, model, content_parts, gen_config, use_structured_output)
        
        # Handle new return format with API request/response
        if isinstance(result, tuple) and len(result) >= 3:
            # Result includes API request and response
            if use_structured_output:
                if len(result) == 4:  # (formatted, raw_json, api_request, api_response)
                    return result
                else:  # (text, api_request, api_response)
                    text = result[0]
                    try:
                        parsed = json.loads(text)
                        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                        return (formatted, text, result[1], result[2])
                    except:
                        return (text, text, result[1], result[2])
            else:
                # For non-structured, add empty raw_json
                return (result[0], "", result[1], result[2])
        else:
            # Fallback for old format (shouldn't happen)
            if use_structured_output and isinstance(result, tuple):
                return result + ({}, {})  # Add empty request/response
            elif use_structured_output:
                return (result, result, {}, {})
            else:
                return (result, "", {}, {})

    def get_caption_with_video_file(self, api_key, video_path, mime_type, prompt, model,
                                    process_audio, temperature, max_output_tokens, top_p, top_k, seed, api_version="auto",
                                    use_structured_output=False, output_schema=None):
        """Send entire video file to Gemini API using google-genai SDK (for Gemini 1.5+ models)"""
        # Create client with API key
        client = genai.Client(api_key=api_key)
        
        print(f"[GeminiVideoCaptioner] Using google-genai SDK for {model}")

        # For newer models (1.5+), we can send the entire video file
        print(f"[GeminiVideoCaptioner] Processing video file directly with {model}")

        # Check file size
        file_size = os.path.getsize(video_path)
        max_request_size = 30 * 1024 * 1024  # 30MB

        if file_size > max_request_size:
            print(
                f"[GeminiVideoCaptioner] Warning: Video file size ({file_size / (1024 * 1024):.2f}MB) exceeds the direct API request limit of 30MB.")

            # Try to compress the video to WebM if not already
            if not mime_type or mime_type != "video/webm":
                print(f"[GeminiVideoCaptioner] Attempting to compress to WebM format...")
                webm_path = self.convert_to_webm(video_path)

                if webm_path:
                    # Use the compressed WebM file instead
                    video_path = webm_path
                    mime_type = "video/webm"
                    file_size = os.path.getsize(video_path)
                    print(f"[GeminiVideoCaptioner] Compressed to WebM: {file_size / (1024 * 1024):.2f}MB")

            # If still too large, fall back to frame extraction
            if file_size > max_request_size:
                print(f"[GeminiVideoCaptioner] Video still too large, falling back to frame extraction...")
                frames, timestamps = self.extract_frames(video_path, 1.0, 300)  # 1 fps, max 5 minutes

                # Clean up temporary WebM file if we created one
                if 'webm_path' in locals() and webm_path:
                    os.unlink(webm_path)

                if not frames:
                    return "Error: Failed to extract frames from large video file"

                return self.get_caption_with_frames(
                    api_key, frames, prompt, model, temperature, max_output_tokens, top_p, top_k, seed, api_version
                )

        # Read video file and encode to base64
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
            base64_video = base64.b64encode(video_data).decode("utf-8")

        # Prepare content parts for SDK
        content_parts = [{
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_video
            }
        }, {
            "text": prompt
        }]

        # Parse schema if structured output is enabled
        response_schema = None
        if use_structured_output and output_schema:
            try:
                response_schema = json.loads(output_schema)
                # Clean the schema for Gemini API
                response_schema = self._clean_schema_for_gemini(response_schema)
                print(f"[GeminiVideoCaptioner] Using structured output with cleaned schema")
            except json.JSONDecodeError as e:
                print(f"[GeminiVideoCaptioner] Warning: Invalid JSON schema, falling back to regular output: {e}")
                use_structured_output = False

        # Create generation config
        gen_config_dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "candidate_count": 1
        }
        
        # Add structured output if enabled
        if use_structured_output and response_schema:
            gen_config_dict["response_mime_type"] = "application/json"
            gen_config_dict["response_schema"] = response_schema
        
        gen_config = types.GenerateContentConfig(**gen_config_dict)

        # Send request using SDK
        print(f"[GeminiVideoCaptioner] Sending video to Gemini API ({model})...")
        result = self._send_api_request(client, model, content_parts, gen_config, use_structured_output)
        
        # Handle new return format with API request/response
        if isinstance(result, tuple) and len(result) >= 3:
            # Result includes API request and response
            if use_structured_output:
                if len(result) == 4:  # (formatted, raw_json, api_request, api_response)
                    return result
                else:  # (text, api_request, api_response)
                    text = result[0]
                    try:
                        parsed = json.loads(text)
                        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                        return (formatted, text, result[1], result[2])
                    except:
                        return (text, text, result[1], result[2])
            else:
                # For non-structured, add empty raw_json
                return (result[0], "", result[1], result[2])
        else:
            # Fallback for old format (shouldn't happen)
            if use_structured_output and isinstance(result, tuple):
                return result + ({}, {})  # Add empty request/response
            elif use_structured_output:
                return (result, result, {}, {})
            else:
                return (result, "", {}, {})

    def convert_to_webm(self, input_path, max_size_mb=29):
        """Convert any video to WebM format with size limit using OpenCV

        Args:
            input_path: Path to input video file
            max_size_mb: Maximum size in MB for the output WebM file

        Returns:
            Path to the converted WebM file
        """
        output_path = None
        cap = None
        out = None
        successful_conversion = False

        try:
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
                output_path = temp_webm.name

            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # Cap duration if very long video (45 minutes max)
            max_duration = 45 * 60  # 45 minutes in seconds
            max_frames = int(min(duration, max_duration) * fps)

            # Start with high quality and gradually reduce
            max_size_bytes = max_size_mb * 1024 * 1024
            size_factors = [1.0, 0.75, 0.5, 0.25, 0.125]  # Progressive size reduction

            for size_factor in size_factors:
                # Calculate new dimensions
                new_width = int(width * size_factor)
                new_height = int(height * size_factor)

                # Ensure dimensions are even by flooring to the nearest even number, min 2
                new_width = (new_width // 2) * 2
                new_width = max(2, new_width)
                new_height = (new_height // 2) * 2
                new_height = max(2, new_height)

                # Try different quality settings
                for quality in [95, 80, 60, 40, 20]:
                    print(f"[GeminiVideoCaptioner] Converting to WebM: {new_width}x{new_height}, quality={quality}")

                    try:
                        # Create VideoWriter object
                        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # WebM codec
                        out = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            fps,
                            (new_width, new_height),
                            isColor=True
                        )

                        if not out.isOpened():
                            raise Exception("Failed to open VideoWriter")

                        frame_count = 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

                        while True:
                            ret, frame = cap.read()
                            if not ret or frame_count >= max_frames:
                                break

                            # Resize frame if needed
                            if size_factor != 1.0:
                                frame = cv2.resize(frame, (new_width, new_height))

                            out.write(frame)
                            frame_count += 1

                        # Release VideoWriter
                        if out is not None:
                            out.release()
                            out = None

                        # Check file size
                        file_size = os.path.getsize(output_path)
                        if file_size <= max_size_bytes:
                            print(f"[GeminiVideoCaptioner] Created WebM: {file_size / 1024 / 1024:.2f}MB")
                            successful_conversion = True
                            return output_path

                        print(
                            f"[GeminiVideoCaptioner] File too large ({file_size / 1024 / 1024:.2f}MB), retrying with lower quality")

                    except Exception as e:
                        print(f"[GeminiVideoCaptioner] Error during conversion: {e}")
                        if out is not None:
                            out.release()
                            out = None
                        continue

            print("[GeminiVideoCaptioner] Could not create WebM within size limit")
            return None

        except Exception as e:
            print(f"[GeminiVideoCaptioner] Error converting to WebM: {e}")
            return None

        finally:
            # Cleanup resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if not successful_conversion and output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception as e:
                    print(f"[GeminiVideoCaptioner] Warning: Failed to delete temp file {output_path} in convert_to_webm finally: {e}")
                    pass

    def create_webm_from_frames(self, frames, fps=30, max_size_mb=29):
        """Create a WebM video from a list of frames using OpenCV

        Args:
            frames: List of numpy arrays representing frames
            fps: Frames per second
            max_size_mb: Maximum size in MB for the output WebM file

        Returns:
            Path to the created WebM file or None if creation fails
        """
        if not frames:
            return None

        output_path = None
        out = None

        try:
            # Create a temporary file for the output WebM
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
                output_path = temp_webm.name

            # Get dimensions from the first frame
            height, width = frames[0].shape[:2]

            # Ensure dimensions are even by flooring to the nearest even number, min 2
            width = (width // 2) * 2
            width = max(2, width)
            height = (height // 2) * 2
            height = max(2, height)

            # Set size reduction factors and quality
            size_factors = [1.0, 0.75, 0.5, 0.25, 0.125]
            max_size_bytes = max_size_mb * 1024 * 1024

            # Ensure frames are in correct format
            processed_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    # Convert from float [0-1] to uint8 [0-255] if needed
                    frame = (frame * 255).astype(np.uint8)
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(frame)

            # Try different size factors and quality settings
            for size_factor in size_factors:
                # Calculate new dimensions
                new_width = int(width * size_factor)
                new_height = int(height * size_factor)

                # Ensure dimensions are even by flooring to the nearest even number, min 2
                new_width = (new_width // 2) * 2
                new_width = max(2, new_width)
                new_height = (new_height // 2) * 2
                new_height = max(2, new_height)

                # Try different quality settings
                for quality in [95, 80, 60, 40, 20]:
                    print(f"[GeminiVideoCaptioner] Creating WebM: {new_width}x{new_height}, quality={quality}")

                    try:
                        # Create VideoWriter object
                        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # WebM codec
                        out = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            fps,
                            (new_width, new_height),
                            isColor=True
                        )

                        if not out.isOpened():
                            raise Exception("Failed to open VideoWriter")

                        for frame in processed_frames:
                            # Resize frame if needed
                            if size_factor != 1.0:
                                frame = cv2.resize(frame, (new_width, new_height))

                            # Apply quality compression
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                            _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

                            # Write frame
                            out.write(frame)

                        # Release VideoWriter
                        if out is not None:
                            out.release()
                            out = None

                        # Check file size
                        file_size = os.path.getsize(output_path)
                        if file_size <= max_size_bytes:
                            print(
                                f"[GeminiVideoCaptioner] Created WebM: {file_size / 1024 / 1024:.2f}MB with dimensions {new_width}x{new_height}")
                            return output_path

                        print(
                            f"[GeminiVideoCaptioner] File too large ({file_size / 1024 / 1024:.2f}MB), retrying with lower quality")

                    except Exception as e:
                        print(f"[GeminiVideoCaptioner] Error while creating video: {e}")
                        if out is not None:
                            out.release()
                            out = None
                        continue

            # If we couldn't get under the size limit, cleanup and return None
            print("[GeminiVideoCaptioner] Couldn't create WebM within size limit")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None

        except Exception as e:
            print(f"[GeminiVideoCaptioner] Error creating WebM from frames: {e}")
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return None

        finally:
            # Cleanup resources
            if out is not None:
                out.release()

    def _send_api_request(self, client, model, contents, config, use_structured=False, retry_count=0, max_retries=3):
        """Helper method to send API request using google-genai SDK"""
        try:
            # Debug: Log request details
            print(f"[DEBUG] Model: {model}")
            print(f"[DEBUG] Content parts count: {len(contents) if isinstance(contents, list) else 1}")
            if isinstance(contents, list):
                for i, part in enumerate(contents[:3]):  # Log first 3 parts
                    if isinstance(part, dict):
                        if 'text' in part:
                            print(f"[DEBUG] Part {i}: Text - {part['text'][:100]}...")
                        elif 'inline_data' in part:
                            print(f"[DEBUG] Part {i}: Image/Video - {part['inline_data'].get('mime_type', 'unknown')}")
            print(f"[DEBUG] Temperature: {config.temperature if hasattr(config, 'temperature') else 'N/A'}")
            print(f"[DEBUG] Max tokens: {config.max_output_tokens if hasattr(config, 'max_output_tokens') else 'N/A'}")
            print(f"[DEBUG] Structured output: {use_structured}")
            # Debug logging for request
            print(f"[DEBUG] Sending request using google-genai SDK")
            print(f"[DEBUG] Model: {model}")
            print(f"[DEBUG] Number of content parts: {len(contents)}")
            print(f"[DEBUG] Structured output: {use_structured}")
            
            # Store API request data
            api_request = {
                "model": model,
                "contents": [str(c) if not hasattr(c, '_pb') else "<video/image data>" for c in contents],
                "config": {
                    "temperature": config.temperature if hasattr(config, 'temperature') else None,
                    "max_output_tokens": config.max_output_tokens if hasattr(config, 'max_output_tokens') else None,
                    "top_p": config.top_p if hasattr(config, 'top_p') else None,
                    "top_k": config.top_k if hasattr(config, 'top_k') else None,
                    "response_mime_type": config.response_mime_type if hasattr(config, 'response_mime_type') else None,
                    "response_schema": str(config.response_schema) if hasattr(config, 'response_schema') else None
                }
            }

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            # Check if response is valid
            if response:
                print("[DEBUG] Valid API response received")
                print(f"[DEBUG] Response type: {type(response)}")
                
                # Store API response data
                api_response = {
                    "text": None,
                    "candidates": []
                }
                
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
                
                # For structured output, check for parsed result first
                if use_structured and hasattr(response, 'parsed') and response.parsed:
                    print(f"[DEBUG] Got parsed structured result")
                    parsed_result = response.parsed
                    if isinstance(parsed_result, dict):
                        formatted = json.dumps(parsed_result, indent=2, ensure_ascii=False)
                        raw = json.dumps(parsed_result, ensure_ascii=False)
                        return (formatted, raw, api_request, api_response)
                    else:
                        return (str(parsed_result), str(parsed_result), api_request, api_response)
                
                # Try to get text directly
                if hasattr(response, 'text'):
                    text = response.text
                    if text:
                        print(f"[DEBUG] Successfully extracted text: {len(text)} chars")
                        if use_structured:
                            # Try to parse as JSON for structured output
                            try:
                                parsed = json.loads(text)
                                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                                return (formatted, text, api_request, api_response)
                            except:
                                return (text, text, api_request, api_response)
                        return (text, api_request, api_response)
                
                # Try to extract from candidates
                if hasattr(response, 'candidates') and response.candidates:
                    print(f"[DEBUG] Number of candidates: {len(response.candidates)}")
                    candidate = response.candidates[0]
                    
                    if hasattr(candidate, 'finish_reason'):
                        print(f"[DEBUG] Finish reason: {candidate.finish_reason}")
                    
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        if hasattr(content, 'parts') and content.parts:
                            for i, part in enumerate(content.parts):
                                if hasattr(part, 'text') and part.text:
                                    text = part.text
                                    print(f"[DEBUG] Extracted text from part {i}: {len(text)} chars")
                                    if use_structured:
                                        # Try to parse as JSON for structured output
                                        try:
                                            parsed = json.loads(text)
                                            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                                            return (formatted, text, api_request, api_response)
                                        except:
                                            return (text, text, api_request, api_response)
                                    return (text, api_request, api_response)
                        else:
                            # Handle safety blocks or empty responses
                            error_msg = ""
                            if hasattr(candidate, 'finish_reason'):
                                if str(candidate.finish_reason) == "STOP":
                                    error_msg = "Error: Model returned empty response. The prompt may have been blocked or the model had nothing to generate."
                                elif "SAFETY" in str(candidate.finish_reason):
                                    error_msg = "Error: Response blocked by safety filters. Try rephrasing your prompt."
                                else:
                                    error_msg = f"Error: Model stopped with reason: {candidate.finish_reason}"
                            else:
                                error_msg = "Error: Model returned empty response"
                            
                            if use_structured:
                                return (error_msg, "", api_request, api_response)
                            return (error_msg, api_request, api_response)
                
                # No valid response found
                print(f"[DEBUG] Could not extract text from response")
                if retry_count < max_retries - 1:
                    print(f"[DEBUG] Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))  # Progressive backoff
                    return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
                if use_structured:
                    return ("Failed to get caption from Gemini API: unexpected response format", "", api_request, api_response)
                return ("Failed to get caption from Gemini API: unexpected response format", api_request, api_response)
            
            else:
                print("[DEBUG] No response received")
                if retry_count < max_retries - 1:
                    print(f"[DEBUG] Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))
                    return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
                if use_structured:
                    return ("Error: No response from Gemini API after multiple attempts", "", api_request, {})
                return ("Error: No response from Gemini API after multiple attempts", api_request, {})

        except Exception as e:
            error_msg = f"API call error: {str(e)}"
            print(f"[GeminiVideoCaptioner] {error_msg}")
            
            # Special handling for 500 INTERNAL errors
            if "500 INTERNAL" in str(e) or "Internal error encountered" in str(e):
                print("[GeminiVideoCaptioner] ⚠️ Gemini API returned 500 Internal Server Error")
                print(f"[GeminiVideoCaptioner] Model being used: {model}")
                
                # Special handling for specific models
                if "gemini-2.5-flash-lite" in model:
                    print("[GeminiVideoCaptioner] ⚠️ KNOWN ISSUE: gemini-2.5-flash-lite has limited capabilities")
                    print("[GeminiVideoCaptioner] This model may not support:")
                    print("  - Video processing (only images)")
                    print("  - Large content payloads")
                    print("  - Certain structured output formats")
                    print("[GeminiVideoCaptioner] RECOMMENDED ALTERNATIVES:")
                    print("  1. Use gemini-1.5-flash for general purpose")
                    print("  2. Use gemini-2.5-flash for better performance")
                    print("  3. Use gemini-2.0-flash for video processing")
                elif "exp" in model:
                    print("[GeminiVideoCaptioner] ⚠️ Experimental model detected")
                    print("[GeminiVideoCaptioner] Experimental models may be unstable")
                    print("[GeminiVideoCaptioner] Try using stable versions instead")
                else:
                    print("[GeminiVideoCaptioner] Possible causes:")
                    print("  1. The model might not support the requested operation")
                    print("  2. Content type (video/image) not supported by this model")
                    print("  3. Request payload too large for this model")
                    print("  4. Temporary server issues on Google's side")
                
                # Don't retry 500 errors as they're usually not transient
                error_detail = f"Error: Gemini API Internal Server Error (500) with model '{model}'."
                
                if "gemini-2.5-flash-lite" in model:
                    error_detail += " This lite model has limited capabilities. Try using gemini-1.5-flash or gemini-2.5-flash instead."
                elif "exp" in model:
                    error_detail += " Experimental models may be unstable. Try using stable versions."
                else:
                    error_detail += " The model might not support this operation or content type."
                
                if use_structured:
                    return (error_detail, "", api_request, {})
                return (error_detail, api_request, {})
            
            traceback.print_exc()  # Print full stack trace for debugging
            
            # Retry on other errors
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff
                print(f"[GeminiVideoCaptioner] Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._send_api_request(client, model, contents, config, use_structured, retry_count + 1, max_retries)
            
            if use_structured:
                return (f"Error: {error_msg}", "", {}, {})
            return (f"Error: {error_msg}", {}, {})