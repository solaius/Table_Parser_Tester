#!/usr/bin/env python3
"""
Test script for CamelotExtractor.
This script tests the CamelotExtractor with a sample PDF.
"""

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from pdf_table_extractor.camelot_extractor import CamelotExtractor

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'camelot_extractor_test.log'))
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to test the CamelotExtractor."""
    logger.info("Starting CamelotExtractor test")
    
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
        # Test Lattice flavor
        logger.info("Testing CamelotExtractor with Lattice flavor...")
        extractor_lattice = CamelotExtractor(flavor='lattice')
        
        # Extract tables from the PDF
        logger.info("Extracting tables from PDF using Lattice method...")
        tables_lattice = extractor_lattice.extract_tables(pdf_file)
        
        # Check if any tables were extracted
        if tables_lattice and len(tables_lattice) > 0:
            logger.info(f"✅ Successfully extracted {len(tables_lattice)} tables from the PDF using Lattice method")
            
            # Print the first table as a preview
            if len(tables_lattice) > 0:
                logger.info("First table preview (Lattice):")
                logger.info(tables_lattice[0])
        else:
            logger.warning("❌ No tables were extracted from the PDF using Lattice method")
        
        # Test Stream flavor
        logger.info("Testing CamelotExtractor with Stream flavor...")
        extractor_stream = CamelotExtractor(flavor='stream')
        
        # Extract tables from the PDF
        logger.info("Extracting tables from PDF using Stream method...")
        tables_stream = extractor_stream.extract_tables(pdf_file)
        
        # Check if any tables were extracted
        if tables_stream and len(tables_stream) > 0:
            logger.info(f"✅ Successfully extracted {len(tables_stream)} tables from the PDF using Stream method")
            
            # Print the first table as a preview
            if len(tables_stream) > 0:
                logger.info("First table preview (Stream):")
                logger.info(tables_stream[0])
        else:
            logger.warning("❌ No tables were extracted from the PDF using Stream method")
        
        # Test get_raw_output method if available
        if hasattr(extractor_lattice, 'get_raw_output'):
            raw_output = extractor_lattice.get_raw_output(pdf_file)
            if raw_output and len(raw_output) > 0:
                logger.info(f"✅ Successfully got raw output with {len(raw_output)} items")
            else:
                logger.warning("❌ Raw output is empty or not available")
        
    except Exception as e:
        logger.error(f"Error testing CamelotExtractor: {e}")

if __name__ == "__main__":
    main()