#!/usr/bin/env python3
"""
Test script for Granite Vision API connection.
This script tests the connection to the Granite Vision API endpoint
using the OpenAI-compatible API format.
"""

import os
import json
import base64
import requests
import logging
from PIL import Image, ImageDraw, ImageOps
import io
from dotenv import load_dotenv

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'granite_vision_test.log'))
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def encode_image(image_path):
    """
    Encode an image file to base64 for API requests.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image with data URI prefix
    """
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoding}"

def test_text_only_request():
    """Test a text-only request to the Granite Vision API"""
    logger.info("Testing text-only request to Granite Vision API...")
    
    base_endpoint = os.getenv("GRANITE_VISION_ENDPOINT")
    completions_path = os.getenv("OPENAI_COMPLETIONS", "/v1/chat/completions")
    endpoint = f"{base_endpoint.rstrip('/')}{completions_path}"
    model_name = os.getenv("GRANITE_VISION_MODEL_NAME", "granite-vision-3-2")
    api_key = os.getenv("GRANITE_VISION_MODEL_API_KEY", "")
    
    if not endpoint:
        logger.error("GRANITE_VISION_ENDPOINT not found in environment variables")
        return False
    
    logger.info(f"Using endpoint: {endpoint}")
    logger.info(f"Using model: {model_name}")
    
    # Create a simple text prompt
    prompt = "What is the capital of France?"
    
    # Prepare the request payload (OpenAI-compatible format)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to headers if provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Make the API request
        logger.info("Sending request to API...")
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload)
        )
        
        # Log the response status and content
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("API request successful!")
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            logger.error(f"API request failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error making API request: {e}")
        return False

def test_image_request():
    """Test an image request to the Granite Vision API"""
    logger.info("Testing image request to Granite Vision API...")
    
    base_endpoint = os.getenv("GRANITE_VISION_ENDPOINT")
    completions_path = os.getenv("OPENAI_COMPLETIONS", "/v1/chat/completions")
    endpoint = f"{base_endpoint.rstrip('/')}{completions_path}"
    model_name = os.getenv("GRANITE_VISION_MODEL_NAME", "granite-vision-3-2")
    api_key = os.getenv("GRANITE_VISION_MODEL_API_KEY", "")
    
    if not endpoint:
        logger.error("GRANITE_VISION_ENDPOINT not found in environment variables")
        return False
    
    # Create a simple test image
    logger.info("Creating a simple test image for the API request...")
    try:
        # Create a simple test image with a table
        img = Image.new('RGB', (500, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple table
        draw.rectangle([(50, 50), (450, 250)], outline=(0, 0, 0))
        
        # Draw table grid
        for i in range(1, 3):
            # Horizontal lines
            draw.line([(50, 50 + i * 50), (450, 50 + i * 50)], fill=(0, 0, 0), width=1)
            # Vertical lines
            draw.line([(50 + i * 133, 50), (50 + i * 133, 250)], fill=(0, 0, 0), width=1)
        
        # Add some text
        draw.text((90, 70), "Header 1", fill=(0, 0, 0))
        draw.text((220, 70), "Header 2", fill=(0, 0, 0))
        draw.text((350, 70), "Header 3", fill=(0, 0, 0))
        
        draw.text((90, 120), "Data 1", fill=(0, 0, 0))
        draw.text((220, 120), "Data 2", fill=(0, 0, 0))
        draw.text((350, 120), "Data 3", fill=(0, 0, 0))
        
        draw.text((90, 170), "Data 4", fill=(0, 0, 0))
        draw.text((220, 170), "Data 5", fill=(0, 0, 0))
        draw.text((350, 170), "Data 6", fill=(0, 0, 0))
        
        # Save the image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(base_dir, "test_table.png")
        img.save(test_image_path)
        
        logger.info(f"Created test image at: {test_image_path}")
        
        # Encode the image
        image_uri = encode_image(test_image_path)
        
        # Create a prompt with the image
        prompt = f"Describe what you see in this image."
        
        # Prepare the request payload (OpenAI-compatible format)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key to headers if provided
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Make the API request
        logger.info("Sending image request to API...")
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload)
        )
        
        # Log the response status and content
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("API request successful!")
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            logger.error(f"API request failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error making image API request: {e}")
        return False

def main():
    """Main function to run the tests"""
    logger.info("Starting Granite Vision API connection test")
    
    # Test text-only request
    text_success = test_text_only_request()
    
    # Test image request
    image_success = test_image_request()
    
    if text_success:
        logger.info("[PASSED] Text-only API request test PASSED")
    else:
        logger.info("[FAILED] Text-only API request test FAILED")
    
    if image_success:
        logger.info("[PASSED] Image API request test PASSED")
    else:
        logger.info("[FAILED] Image API request test FAILED")
    
    if text_success or image_success:
        logger.info("[SUCCESS] At least one test PASSED - API connection is working")
    else:
        logger.info("[ERROR] All tests FAILED - API connection is not working")

if __name__ == "__main__":
    main()