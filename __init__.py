"""
ComfyUI Gemini Nodes
A collection of nodes for integrating Google Gemini API with ComfyUI
"""

from .nodes.GeminiImageEditor import GeminiImageEditor
from .nodes.GeminiTextAPI import GeminiTextAPI
from .nodes.GeminiImageGenADV import GeminiImageGenADV
from .nodes.GeminiVideoCaptioner import GeminiVideoCaptioner
from .nodes.GeminiVideoGenerator import GeminiVideoGenerator
from .nodes.GeminiStructuredOutput import GeminiStructuredOutput, GeminiJSONExtractor
from .nodes.GeminiFieldExtractor import GeminiFieldExtractor, GeminiJSONParser
from .nodes.UnofficialGeminiAPI import UnofficialGeminiAPI, UnofficialGeminiStreamAPI
from .nodes.GeminiImageProcessor import GeminiImageProcessor, GeminiImageBatchProcessor

NODE_CLASS_MAPPINGS = {
    "GeminiImageEditor": GeminiImageEditor,
    "GeminiTextAPI": GeminiTextAPI,
    "GeminiImageGenADV": GeminiImageGenADV,
    "GeminiVideoCaptioner": GeminiVideoCaptioner,
    "GeminiVideoGenerator": GeminiVideoGenerator,
    "GeminiStructuredOutput": GeminiStructuredOutput,
    "GeminiJSONExtractor": GeminiJSONExtractor,
    "GeminiFieldExtractor": GeminiFieldExtractor,
    "GeminiJSONParser": GeminiJSONParser,
    "UnofficialGeminiAPI": UnofficialGeminiAPI,
    "UnofficialGeminiStreamAPI": UnofficialGeminiStreamAPI,
    "GeminiImageProcessor": GeminiImageProcessor,
    "GeminiImageBatchProcessor": GeminiImageBatchProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEditor": "Gemini Image Editor",
    "GeminiTextAPI": "Gemini Text API",
    "GeminiImageGenADV": "Gemini Image Gen Advanced",
    "GeminiVideoCaptioner": "Gemini Video Captioner",
    "GeminiVideoGenerator": "Gemini Video Generator (Veo)",
    "GeminiStructuredOutput": "Gemini Structured Output",
    "GeminiJSONExtractor": "Gemini JSON Extractor",
    "GeminiFieldExtractor": "Gemini Field Extractor",
    "GeminiJSONParser": "Gemini JSON Parser",
    "UnofficialGeminiAPI": "Unofficial API Call",
    "UnofficialGeminiStreamAPI": "Unofficial Stream API Call",
    "GeminiImageProcessor": "Gemini Image Processor",
    "GeminiImageBatchProcessor": "Gemini Image Batch Processor",
}

# ASCII Art
ascii_art = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██╗                ║
║  ██╔════╝ ██╔════╝████╗ ████║██║████╗  ██║██║                ║
║  ██║  ███╗█████╗  ██╔████╔██║██║██╔██╗ ██║██║                ║
║  ██║   ██║██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║                ║
║  ╚██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║██║                ║
║   ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝                ║
║                                                               ║
║            ComfyUI Nodes for Google Gemini API                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
print(ascii_art)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]