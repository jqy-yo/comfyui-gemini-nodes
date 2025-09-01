# Gemini Nodes Troubleshooting Guide

## Common Issues and Solutions

### 1. 500 INTERNAL Server Error

**Symptoms:**
```
google.genai.errors.ServerError: 500 INTERNAL. {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}
```

**Possible Causes and Solutions:**

#### A. Regional Restrictions
Some Gemini models (especially video-related models) may not be available in all regions.

**Solutions:**
1. Check if your server is in a supported region for Gemini API
2. Try using a different model (e.g., `gemini-1.5-flash` instead of `gemini-2.0-flash-exp`)
3. Consider using a VPN or proxy if allowed by your organization

#### B. Model Access Permissions
Your API key might not have access to certain models.

**Solutions:**
1. Verify your API key has access to the model you're trying to use
2. Check Google AI Studio to see which models are available for your account
3. Try using a more basic model first to confirm API connectivity

#### C. Request Size or Content Issues
Large videos or certain content types might trigger server errors.

**Solutions:**
1. Try with a smaller video file first
2. Ensure video format is supported (MP4, MOV, AVI, etc.)
3. Check if the video duration is within limits (typically under 2 minutes for most models)

### 2. JSON Parse Errors After API Failure

**Symptoms:**
```
[GeminiFieldExtractor] JSON parse error: Expecting value: line 1 column 1 (char 0)
```

This usually occurs when the API request fails and returns an error message instead of JSON data.

**Solution:**
The updated nodes now handle this more gracefully and will show the actual API error instead of a JSON parse error.

### 3. Testing Your Setup

To verify your Gemini API setup is working:

1. **Test with a simple text prompt first:**
   - Use the GeminiTextGenerator node with a basic prompt
   - If this works, your API key and basic connectivity are fine

2. **Test with a small image:**
   - Use GeminiImageCaptioner with a small image
   - If this works, multimodal features are available

3. **Test with video:**
   - Start with a very short video (< 10 seconds)
   - Use a common format like MP4
   - If this fails with 500 errors, video features might not be available in your region

### 4. Alternative Solutions

If you continue to experience 500 errors with video models:

1. **Use image-based captioning:**
   - Extract frames from your video
   - Use GeminiImageCaptioner on individual frames
   - Combine the results

2. **Try different models:**
   - `gemini-1.5-flash`: Good general-purpose model
   - `gemini-1.5-pro`: More capable but might have more restrictions
   - `gemini-2.0-flash-exp`: Latest but might not be available everywhere

3. **Check server location:**
   - L4 servers might be hosted in regions with different API availability
   - Consider using a different server location if possible

### 5. Debug Information

The updated nodes now provide more detailed error messages including:
- Specific model being used
- Possible causes for failures
- Suggestions for resolution

### 6. Getting Help

If issues persist:
1. Check the [Google AI Studio](https://aistudio.google.com/) for model availability
2. Review [Gemini API documentation](https://ai.google.dev/gemini-api/docs) for regional availability
3. Open an issue on the [GitHub repository](https://github.com/your-repo/comfyui-gemini-nodes) with:
   - Full error message
   - Model being used
   - Server location (if known)
   - Video/image specifications (size, format, duration)