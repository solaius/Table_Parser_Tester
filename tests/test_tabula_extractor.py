#!/usr/bin/env python3
"""
Test script for TabulaExtractor.
This script tests the TabulaExtractor with a sample PDF.
"""

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdf_table_extractor.tabula_extractor import TabulaExtractor

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'tabula_extractor_test.log'))
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to test the TabulaExtractor."""
    logger.info("Starting TabulaExtractor test")
    
    # Find a sample PDF
    pdf_files = [
        "table_pdfs/basic_tables.pdf",
        "table_pdfs/moderate_tables.pdf"
    ]
    
    # Use the first PDF file that exists
    pdf_file = None
    for file_path in pdf_files:
        if os.path.exists(file_path):
            pdf_file = file_path
            logger.info(f"Found PDF file at: {file_path}")
            break
    
    if not pdf_file:
        logger.error("No PDF files found in the expected locations")
        return
    
    logger.info(f"Using PDF file: {pdf_file}")
    
    try:
        # Initialize the extractor
        extractor = TabulaExtractor()
        
        # Extract tables from the PDF
        logger.info("Extracting tables from PDF...")
        tables = extractor.extract_tables(pdf_file)
        
        # Check if any tables were extracted
        if tables and len(tables) > 0:
            logger.info(f"✅ Successfully extracted {len(tables)} tables from the PDF")
            
            # Print the first table as a preview
            if len(tables) > 0:
                logger.info("First table preview:")
                logger.info(tables[0])
        else:
            logger.warning("❌ No tables were extracted from the PDF")
        
        # Test get_raw_output method if available
        if hasattr(extractor, 'get_raw_output'):
            raw_output = extractor.get_raw_output(pdf_file)
            if raw_output and len(raw_output) > 0:
                logger.info(f"✅ Successfully got raw output with {len(raw_output)} items")
            else:
                logger.warning("❌ Raw output is empty or not available")
        
    except Exception as e:
        logger.error(f"Error testing TabulaExtractor: {e}")

if __name__ == "__main__":
    main()