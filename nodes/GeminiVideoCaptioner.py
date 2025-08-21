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
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("caption", "sampled_frame",)
    FUNCTION = "generate_video_caption"
    CATEGORY = "🤖 Gemini"

    def generate_video_caption(self, api_key, model, frames_per_second, max_duration_minutes,
                               prompt, process_audio, temperature, max_output_tokens, top_p, top_k, seed,
                               video_path=None, image=None, api_version="auto"):
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
            if webm_path is None:
                print(f"[GeminiVideoCaptioner] WebM conversion failed, falling back to frame extraction...")
                frames, frame_timestamps = self.extract_frames(video_path, frames_per_second, max_seconds)
                if not frames:
                    raise ValueError(f"Failed to extract frames from video: {video_path}")

                print(f"[GeminiVideoCaptioner] Extracted {len(frames)} frames at {frames_per_second} fps")

                # Generate captions for frames
                caption = self.get_caption_with_frames(
                    api_key,
                    frames,
                    prompt,
                    model,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                    seed,
                    api_version
                )
            else:
                # Use the WebM file for captioning
                mime_type = "video/webm"
                print(f"[GeminiVideoCaptioner] Using WebM video for captioning")

                # Get caption using the WebM file
                caption = self.get_caption_with_video_file(
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
                    api_version
                )

                # Clean up temporary WebM file
                os.unlink(webm_path)

            return (caption, sample_frame_tensor)

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

            # If only one frame is provided and it's a newer model, send as JPEG directly
            if len(frames) == 1 and model in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
                print(f"[GeminiVideoCaptioner] Single frame input for {model}, using direct frame processing (JPEG).")
                caption = self.get_caption_with_frames(
                    api_key,
                    frames,  # This will be a list with one frame
                    prompt,
                    model,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                    seed,
                    api_version
                )
            else:
                # Original logic: try to create WebM, then send video file or fallback to frames
                print(f"[GeminiVideoCaptioner] Creating WebM video from {len(frames)} frames...")
                webm_path = self.create_webm_from_frames(frames, fps=frames_per_second)

                if webm_path is None:
                    print(f"[GeminiVideoCaptioner] WebM creation failed, falling back to frame processing...")
                    caption = self.get_caption_with_frames(
                        api_key,
                        frames,
                        prompt,
                        model,
                        temperature,
                        max_output_tokens,
                        top_p,
                        top_k,
                        seed,
                        api_version
                    )
                else:
                    mime_type = "video/webm"
                    print(f"[GeminiVideoCaptioner] Using WebM video for captioning")
                    caption = self.get_caption_with_video_file(
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
                        api_version
                    )
                    os.unlink(webm_path) # Clean up temporary WebM file

            return (caption, sample_frame_tensor)

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
                                seed, api_version="auto"):
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

        # Create generation config for SDK
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            candidate_count=1
        )
        
        # Send request using SDK
        print(f"[GeminiVideoCaptioner] Sending request to Gemini API ({model})...")
        return self._send_api_request(client, model, content_parts, gen_config)

    def get_caption_with_video_file(self, api_key, video_path, mime_type, prompt, model,
                                    process_audio, temperature, max_output_tokens, top_p, top_k, seed, api_version="auto"):
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

        # Create generation config
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            candidate_count=1
        )

        # Send request using SDK
        print(f"[GeminiVideoCaptioner] Sending video to Gemini API ({model})...")
        return self._send_api_request(client, model, content_parts, gen_config)

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

    def _send_api_request(self, client, model, contents, config, retry_count=0, max_retries=3):
        """Helper method to send API request using google-genai SDK"""
        try:
            # Debug logging for request
            print(f"[DEBUG] Sending request using google-genai SDK")
            print(f"[DEBUG] Model: {model}")
            print(f"[DEBUG] Number of content parts: {len(contents)}")

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            # Check if response is valid
            if response:
                print("[DEBUG] Valid API response received")
                print(f"[DEBUG] Response type: {type(response)}")
                
                # Try to get text directly
                if hasattr(response, 'text'):
                    text = response.text
                    if text:
                        print(f"[DEBUG] Successfully extracted text: {len(text)} chars")
                        return text
                
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
                                    return text
                        else:
                            # Handle safety blocks or empty responses
                            if hasattr(candidate, 'finish_reason'):
                                if str(candidate.finish_reason) == "STOP":
                                    return "Error: Model returned empty response. The prompt may have been blocked or the model had nothing to generate."
                                elif "SAFETY" in str(candidate.finish_reason):
                                    return "Error: Response blocked by safety filters. Try rephrasing your prompt."
                                else:
                                    return f"Error: Model stopped with reason: {candidate.finish_reason}"
                            return "Error: Model returned empty response"
                
                # No valid response found
                print(f"[DEBUG] Could not extract text from response")
                if retry_count < max_retries - 1:
                    print(f"[DEBUG] Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))  # Progressive backoff
                    return self._send_api_request(client, model, contents, config, retry_count + 1, max_retries)
                return "Failed to get caption from Gemini API: unexpected response format"
            
            else:
                print("[DEBUG] No response received")
                if retry_count < max_retries - 1:
                    print(f"[DEBUG] Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2 * (retry_count + 1))
                    return self._send_api_request(client, model, contents, config, retry_count + 1, max_retries)
                return "Error: No response from Gemini API after multiple attempts"

        except Exception as e:
            error_msg = f"API call error: {str(e)}"
            print(f"[GeminiVideoCaptioner] {error_msg}")
            traceback.print_exc()  # Print full stack trace for debugging
            
            # Retry on error
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff
                print(f"[GeminiVideoCaptioner] Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._send_api_request(client, model, contents, config, retry_count + 1, max_retries)
            
            return f"Error: {error_msg}"