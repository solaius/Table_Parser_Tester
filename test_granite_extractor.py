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
    
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find a sample PDF - try different relative paths
    base_paths = [
        script_dir,  # Current script directory
        os.path.dirname(script_dir),  # Project root
        os.getcwd(),  # Current working directory
    ]
    
    relative_paths = [
        "table_pdfs/create_tables/table_pdfs/basic_tables.pdf",
        "table_pdfs/create_tables/table_pdfs/moderate_tables.pdf",
        "table_pdfs/basic_tables.pdf",
        "table_pdfs/moderate_tables.pdf"
    ]
    
    # Generate all possible combinations of base paths and relative paths
    pdf_files = []
    for base_path in base_paths:
        for rel_path in relative_paths:
            pdf_files.append(os.path.join(base_path, rel_path))
    
    # Add absolute paths as fallback
    pdf_files.extend([
        "/workspace/Table_Parser_Tester/table_pdfs/create_tables/table_pdfs/basic_tables.pdf",
        "/workspace/Table_Parser_Tester/table_pdfs/create_tables/table_pdfs/moderate_tables.pdf"
    ])
    
    # Use the first PDF file that exists
    pdf_file = None
    for file_path in pdf_files:
        if os.path.exists(file_path):
            pdf_file = file_path
            logger.info(f"Found PDF file at: {file_path}")
            break
    
    if not pdf_file:
        logger.warning("No PDF files found in the expected locations. Attempting to create a test PDF...")
        try:
            # Try to create a simple PDF with a table using reportlab
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            
            # Create a test PDF path
            test_pdf_path = os.path.join(script_dir, "test_table.pdf")
            
            # Create the PDF
            doc = SimpleDocTemplate(test_pdf_path, pagesize=letter)
            elements = []
            
            # Create table data
            data = [
                ['Header 1', 'Header 2', 'Header 3'],
                ['Row 1, Col 1', 'Row 1, Col 2', 'Row 1, Col 3'],
                ['Row 2, Col 1', 'Row 2, Col 2', 'Row 2, Col 3'],
                ['Row 3, Col 1', 'Row 3, Col 2', 'Row 3, Col 3']
            ]
            
            # Create the table
            table = Table(data)
            
            # Add style
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            table.setStyle(style)
            
            # Add the table to the elements
            elements.append(table)
            
            # Build the PDF
            doc.build(elements)
            
            logger.info(f"Created test PDF at: {test_pdf_path}")
            pdf_file = test_pdf_path
        except Exception as e:
            logger.error(f"Failed to create test PDF: {e}")
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