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
from PIL import Image, ImageOps
import io
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('granite_vision_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def encode_image(image_path, is_pdf=False):
    """
    Encode an image file to base64 for API requests.
    If the file is a PDF, convert the first page to an image.
    
    Args:
        image_path (str): Path to the image or PDF file
        is_pdf (bool): Whether the file is a PDF
        
    Returns:
        str: Base64 encoded image with data URI prefix
    """
    if is_pdf:
        try:
            from pdf2image import convert_from_path
            # Convert first page of PDF to image
            images = convert_from_path(image_path, first_page=1, last_page=1)
            if not images:
                logger.error("Failed to convert PDF to image")
                return None
            image = images[0]
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return None
    else:
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
    
    endpoint = os.getenv("GRANITE_VISION_ENDPOINT")
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
    
    endpoint = os.getenv("GRANITE_VISION_ENDPOINT")
    model_name = os.getenv("GRANITE_VISION_MODEL_NAME", "granite-vision-3-2")
    api_key = os.getenv("GRANITE_VISION_MODEL_API_KEY", "")
    
    if not endpoint:
        logger.error("GRANITE_VISION_ENDPOINT not found in environment variables")
        return False
    
    # Look for a sample PDF in the table_pdfs directory
    pdf_files = [
        "/workspace/Table_Parser_Tester/table_pdfs/create_tables/table_pdfs/basic_tables.pdf",
        "/workspace/Table_Parser_Tester/table_pdfs/create_tables/table_pdfs/moderate_tables.pdf"
    ]
    
    # Use the first PDF file that exists
    pdf_file = None
    for file_path in pdf_files:
        if os.path.exists(file_path):
            pdf_file = file_path
            break
    
    if not pdf_file:
        logger.error("No PDF files found in the expected locations")
        return False
    
    logger.info(f"Using PDF file: {pdf_file}")
    
    try:
        # Install pdf2image if not already installed
        try:
            import pdf2image
        except ImportError:
            logger.info("Installing pdf2image...")
            os.system("pip install pdf2image")
            
            # Check if poppler is installed (required by pdf2image)
            try:
                os.system("apt-get update && apt-get install -y poppler-utils")
            except:
                logger.warning("Could not install poppler-utils. If pdf2image fails, install it manually.")
        
        # Encode the PDF as an image
        image_uri = encode_image(pdf_file, is_pdf=True)
        
        if not image_uri:
            logger.error("Failed to encode PDF as image")
            return False
        
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
    
    # Test image request if we have sample images
    image_success = test_image_request()
    
    if text_success:
        logger.info("✅ Text-only API request test PASSED")
    else:
        logger.info("❌ Text-only API request test FAILED")
    
    if image_success:
        logger.info("✅ Image API request test PASSED")
    else:
        logger.info("❌ Image API request test FAILED")
    
    if text_success or image_success:
        logger.info("✅ At least one test PASSED - API connection is working")
    else:
        logger.info("❌ All tests FAILED - API connection is not working")

if __name__ == "__main__":
    main()