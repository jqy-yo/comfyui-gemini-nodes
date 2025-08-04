# ComfyUI Gemini Nodes

A collection of custom nodes for integrating Google Gemini API with ComfyUI, providing powerful AI capabilities for text generation, image generation, and video analysis.

## Features

### 🤖 Gemini Text API
- Generate text using any Gemini model (flexible text input for model selection)
- Supports all Gemini models including gemini-2.5-flash-lite, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro, etc.
- Configurable generation parameters (temperature, max tokens, top_p, top_k)
- Custom system instructions support
- Comprehensive error handling and retry logic

### 🎨 Gemini Image Editor
- Generate images using any Gemini model with image generation capability
- Flexible model input supports gemini-2.5-flash, models/gemini-2.0-flash-exp, imagen-3.0-generate-001, etc.
- Support for up to 4 reference images as input
- Batch generation (up to 8 images)
- Automatic image resizing (minimum 1024x1024)
- Asynchronous parallel processing for efficiency

### 🚀 Gemini Image Gen Advanced
- Multi-slot input system (up to 100 input combinations)
- Independent image and prompt for each slot
- Asynchronous parallel API calls
- Batch processing with progress tracking
- Automatic image padding and format conversion

### 🎬 Gemini Video Captioner
- Generate descriptive captions for videos
- Support for video file paths or image batch input
- Automatic video format conversion (WebM)
- Intelligent file size control (under 30MB)
- Frame extraction and timestamp processing

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-gemini-nodes.git
```

2. Install required dependencies:
```bash
cd comfyui-gemini-nodes
pip install -r requirements.txt
```

3. Set up your Google Gemini API key as an environment variable:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Configuration

### API Key Setup

The nodes look for the Gemini API key in the following order:
1. Environment variable: `GOOGLE_API_KEY`
2. ComfyUI settings file

For security, we recommend using environment variables.

### Model Selection

All nodes now support flexible model input via text field. You can enter any valid Gemini model name:

- **Text Generation**: Any Gemini model (e.g., gemini-2.5-flash-lite, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro)
- **Image Generation**: Any model with image generation capability (e.g., gemini-2.5-flash, models/gemini-2.0-flash-exp, imagen-3.0-generate-001)
- **Video Analysis**: Any model with video analysis capability (e.g., gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-pro)

Simply type the model name in the model field. The nodes will automatically handle the API endpoints and configurations.

## Usage

After installation, the nodes will appear in ComfyUI under the "🤖 Gemini" category:

1. **Gemini Text API**: For text generation tasks
2. **Gemini Image Editor**: For image generation with reference images
3. **Gemini Image Gen Advanced**: For complex multi-input image generation
4. **Gemini Video Captioner**: For video analysis and captioning

## Examples

### Text Generation
- Connect a prompt to the Gemini Text API node
- Configure temperature and other parameters
- Get generated text output

### Image Generation
- Input reference images (optional)
- Provide text prompt
- Configure batch size and model
- Receive generated images

### Video Captioning
- Input video file path or image sequence
- Configure prompt and model
- Get descriptive captions

## Requirements

- ComfyUI
- Python 3.8+
- google-generativeai
- Pillow (PIL)
- numpy
- torch
- cv2 (opencv-python)
- moviepy
- aiohttp

## Troubleshooting

### API Key Issues
- Ensure your API key is correctly set in environment variables
- Check API key permissions for the models you're trying to use

### Model Access
- Some models require specific API access levels
- Imagen models may have regional restrictions

### Memory Issues
- Large batch sizes may cause memory issues
- Reduce batch size or image resolution if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Original nodes created as part of the Fill-Nodes collection, now separated for focused Gemini API integration.

## Support

For issues, questions, or contributions, please open an issue on GitHub.