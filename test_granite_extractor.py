#!/usr/bin/env python3
"""
Test script for DoclingGraniteVisionExtractor.
This script tests the DoclingGraniteVisionExtractor with a sample PDF.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from pdf_table_extractor.docling_granitevision import DoclingGraniteVisionExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('granite_extractor_test.log')
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main function to test the DoclingGraniteVisionExtractor"""
    logger.info("Starting DoclingGraniteVisionExtractor test")
    
    # Find a sample PDF
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
        return
    
    logger.info(f"Using PDF file: {pdf_file}")
    
    try:
        # Initialize the extractor
        extractor = DoclingGraniteVisionExtractor()
        
        # Extract tables from the PDF
        logger.info("Extracting tables from PDF...")
        tables = extractor.extract_tables(pdf_file)
        
        # Check if any tables were extracted
        if tables:
            logger.info(f"✅ Successfully extracted {len(tables)} tables from the PDF")
            
            # Print the first table
            if len(tables) > 0:
                logger.info(f"First table preview:\n{tables[0].head()}")
        else:
            logger.warning("❌ No tables were extracted from the PDF")
        
        # Get raw output
        raw_output = extractor.get_raw_output(pdf_file)
        if raw_output:
            logger.info(f"✅ Successfully got raw output with {len(raw_output)} items")
        else:
            logger.warning("❌ No raw output was returned")
            
    except Exception as e:
        logger.error(f"Error testing DoclingGraniteVisionExtractor: {e}")

if __name__ == "__main__":
    main()