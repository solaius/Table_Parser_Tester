#!/usr/bin/env python3
"""
Test script for checking CUDA availability.
This script checks if CUDA is available and prints information about the GPU.
"""

import os
import sys
import logging
import torch
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'cuda_test.log'))
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def test_cuda():
    """Test CUDA availability and print GPU information."""
    logger.info("Testing CUDA availability...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        logger.info(f"CUDA version: {cuda_version}")
        
        # Get number of GPUs
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"CUDA device name: {device_name}")
        
        # Get device properties
        device_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"Device properties: {device_properties}")
        
        # Test a simple tensor operation on GPU
        try:
            logger.info("Testing tensor operation on GPU...")
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()
            z = x + y
            logger.info(f"Tensor operation successful. Result shape: {z.shape}, device: {z.device}")
            logger.info("[PASSED] CUDA is working correctly")
            return True
        except Exception as e:
            logger.error(f"Error performing tensor operation on GPU: {e}")
            logger.info("[FAILED] CUDA test failed")
            return False
    else:
        # Check if USE_GPU is set to true in .env
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        if use_gpu:
            logger.warning("USE_GPU is set to true in .env, but CUDA is not available")
            logger.info("Please check your CUDA installation or set USE_GPU=false in .env")
        else:
            logger.info("USE_GPU is set to false in .env, using CPU as expected")
        
        logger.info("[SKIPPED] CUDA test skipped (not available)")
        return False

def main():
    """Main function to run the CUDA test."""
    logger.info("Starting CUDA availability test")
    
    # Test CUDA
    result = test_cuda()
    
    if result:
        logger.info("CUDA test completed successfully")
    else:
        if torch.cuda.is_available():
            logger.warning("CUDA is available but test failed")
        else:
            logger.info("CUDA is not available, using CPU")
    
    # Print PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Print Python version
    logger.info(f"Python version: {sys.version}")
    
    # Print environment variables related to CUDA
    logger.info("Environment variables related to CUDA:")
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES"]:
        logger.info(f"{var}: {os.environ.get(var, 'Not set')}")

if __name__ == "__main__":
    main()