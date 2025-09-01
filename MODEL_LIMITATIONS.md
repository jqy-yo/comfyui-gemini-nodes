# Gemini Model Limitations and Capabilities

## Known Model Limitations

### gemini-2.5-flash-lite
**Status:** Limited Capabilities
**Known Issues:**
- May return 500 errors with video content
- Limited support for large images
- Reduced context window compared to full models
- May not support all structured output formats

**Recommended Use Cases:**
- Simple text generation
- Small image analysis (< 1MB)
- Quick responses with low complexity

**NOT Recommended For:**
- Video processing
- Large image batches
- Complex structured outputs
- Long context conversations

### gemini-2.0-flash-exp
**Status:** Experimental
**Known Issues:**
- May be unstable or unavailable
- Features subject to change
- Regional availability varies

### Working Models for Video/Image Processing

#### Best for Video:
1. **gemini-1.5-flash** - Stable, good performance
2. **gemini-2.0-flash** - Latest stable video support
3. **gemini-1.5-pro** - Higher quality but slower

#### Best for Images:
1. **gemini-1.5-flash** - Fast and reliable
2. **gemini-2.5-flash** - Latest features
3. **gemini-1.5-pro** - Best quality

## Troubleshooting 500 Errors

If you get a 500 error with a specific model:

1. **Check content type compatibility:**
   - Lite models may not support video
   - Some models have size limits

2. **Verify in test script:**
   ```bash
   python test_gemini_api.py YOUR_API_KEY path/to/content
   ```

3. **Try with smaller content:**
   - Reduce image size
   - Shorten video duration
   - Simplify prompts

4. **Use appropriate model for task:**
   - Don't force lite models for heavy tasks
   - Use stable models for production

## Model Selection Guide

| Task | Recommended Model | Avoid |
|------|------------------|--------|
| Video Analysis | gemini-1.5-flash | gemini-2.5-flash-lite |
| Large Images | gemini-1.5-pro | gemini-2.5-flash-lite |
| Quick Text | gemini-2.5-flash-lite | - |
| Production Use | gemini-1.5-flash | *-exp models |