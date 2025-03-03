import os
import sys
import pandas as pd
import tabula
import pdfplumber
from pdfminer.high_level import extract_text
from docling.document_converter import DocumentConverter
import camelot
import torch
import warnings
import pytesseract
from transformers import DetrImageProcessor, TableTransformerForObjectDetection, logging as transformers_logging
from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()  # Only show errors, not warnings

def convert_table_to_markdown(df):
    """
    Convert a pandas DataFrame to Markdown format.
    Uses DataFrame.to_markdown() if available.
    """
    try:
        md = df.to_markdown(index=False)
    except Exception:
        # Fallback if to_markdown is not available
        headers = df.columns.tolist()
        md_header = "| " + " | ".join(headers) + " |\n"
        md_separator = "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        md_rows = "\n".join(rows)
        md = md_header + md_separator + md_rows
    
    return md

def extract_with_tabula(pdf_path):
    print(f"\nExtracting tables from {pdf_path} using Tabula...")
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    print(f"Found {len(tables)} table(s)")
    
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}:")
        md = convert_table_to_markdown(table)
        print(md)
    
    return tables

def extract_with_pdfplumber(pdf_path):
    print(f"\nExtracting tables from {pdf_path} using pdfplumber...")
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_table = page.extract_table()
            if raw_table:
                print(f"Found table on page {page_num+1}")
                # Assume first row as header and the rest as data
                table_df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
                tables.append(table_df)
    
    print(f"Found {len(tables)} table(s)")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}:")
        md = convert_table_to_markdown(table)
        print(md)
    
    return tables

def extract_with_pdfminer(pdf_path):
    print(f"\nExtracting text from {pdf_path} using PDFMiner...")
    text = extract_text(pdf_path)
    print("Extracted text (first 500 characters):")
    print(text[:500])
    print("...")
    
    # PDFMiner doesn't extract tables directly, so we return an empty list
    return []

def extract_with_docling(pdf_path):
    print(f"\nExtracting tables from {pdf_path} using Docling...")
    
    # Create a document converter
    converter = DocumentConverter()
    
    # Load the PDF
    result = converter.convert(pdf_path)
    
    # Access the document
    doc = result.document
    
    # Extract tables from the document
    tables = []
    
    # Check if the document has tables
    if hasattr(doc, 'tables') and doc.tables:
        print(f"Found {len(doc.tables)} table(s) in the document")
        
        for i, table in enumerate(doc.tables):
            # Use the built-in method to export to DataFrame if available
            if hasattr(table, 'export_to_dataframe'):
                df = table.export_to_dataframe()
                tables.append(df)
                print(f"Extracted table {i+1}")
    
    print(f"Found {len(tables)} table(s) in total")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}:")
        md = convert_table_to_markdown(table)
        print(md)
    
    return tables

def extract_with_camelot(pdf_path, flavor='lattice'):
    print(f"\nExtracting tables from {pdf_path} using Camelot ({flavor})...")
    
    # Extract tables using Camelot
    tables = camelot.read_pdf(pdf_path, flavor=flavor, pages='all')
    
    print(f"Found {len(tables)} table(s)")
    
    # Convert Camelot tables to pandas DataFrames
    dataframes = []
    for i, table in enumerate(tables):
        # Get the table as a pandas DataFrame
        df = table.df
        
        # Print parsing report
        print(f"\nTable {i+1} parsing report:")
        print(f"  Accuracy: {table.parsing_report['accuracy']:.2f}%")
        print(f"  Whitespace: {table.parsing_report['whitespace']:.2f}%")
        print(f"  Page: {table.parsing_report['page']}")
        
        # Print the table as Markdown
        print(f"\nTable {i+1}:")
        md = convert_table_to_markdown(df)
        print(md)
        
        # Add the table to the list
        dataframes.append(df)
    
    return dataframes

def extract_with_table_transformer(pdf_path):
    print(f"\nExtracting tables from {pdf_path} using Table Transformer...")
    
    # Load the model and processor with updated parameters to avoid deprecation warnings
    processor = DetrImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection",
        do_resize=True
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection",
        ignore_mismatched_sizes=True  # Ignore mismatched sizes to suppress warnings
    )
    
    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)
    print(f"Converted {len(images)} page(s)")
    
    # List to store all extracted tables
    all_tables = []
    
    # Process each page
    for page_num, image in enumerate(images):
        print(f"\nProcessing page {page_num+1}...")
        
        # Process the image with the model
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert outputs to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        # Check if any tables were detected
        if len(results["scores"]) == 0:
            print(f"No tables detected on page {page_num+1}")
            continue
        
        print(f"Detected {len(results['scores'])} table(s) on page {page_num+1}")
        
        # Extract table regions
        for table_num, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            print(f"Table {table_num+1} - Score: {score.item():.2f}")
            box = [round(i) for i in box.tolist()]
            
            # Extract the table region from the image
            table_image = image.crop((box[0], box[1], box[2], box[3]))
            
            # Convert table image to grayscale for better processing
            table_image_np = np.array(table_image)
            table_image_gray = cv2.cvtColor(table_image_np, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding to enhance table structure
            table_image_binary = cv2.adaptiveThreshold(
                table_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours to detect cells
            contours, _ = cv2.findContours(
                table_image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours to find table cells
            cells = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small contours and very large ones
                if w > 20 and h > 20 and w < table_image_np.shape[1] * 0.9 and h < table_image_np.shape[0] * 0.9:
                    cells.append((x, y, x + w, y + h))
            
            # If no cells were detected, try a different approach or skip
            if not cells:
                print(f"No cells detected in table {table_num+1}")
                continue
            
            print(f"Detected {len(cells)} cells in table {table_num+1}")
            
            # Sort cells by y-coordinate (row) and then by x-coordinate (column)
            cells.sort(key=lambda cell: (cell[1], cell[0]))
            
            # Group cells into rows based on y-coordinate
            rows = []
            current_row = [cells[0]]
            row_y = cells[0][1]
            
            for cell in cells[1:]:
                # If the cell is in the same row (y-coordinate within a threshold)
                if abs(cell[1] - row_y) < 20:
                    current_row.append(cell)
                else:
                    # Sort the current row by x-coordinate
                    current_row.sort(key=lambda c: c[0])
                    rows.append(current_row)
                    # Start a new row
                    current_row = [cell]
                    row_y = cell[1]
            
            # Add the last row
            if current_row:
                current_row.sort(key=lambda c: c[0])
                rows.append(current_row)
            
            print(f"Grouped cells into {len(rows)} rows")
            
            # Create a DataFrame from the extracted cells
            table_data = []
            for row in rows:
                row_data = []
                for cell in row:
                    # Extract text from the cell using OCR
                    cell_image = table_image.crop((cell[0], cell[1], cell[2], cell[3]))
                    
                    # Preprocess the cell image to improve OCR accuracy
                    cell_np = np.array(cell_image)
                    
                    # Try multiple preprocessing techniques for better OCR results
                    ocr_results = []
                    
                    # 1. Try with original image
                    try:
                        text1 = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
                        if text1:
                            ocr_results.append(text1)
                    except Exception:
                        pass
                    
                    # 2. Try with grayscale image
                    try:
                        if len(cell_np.shape) == 3 and cell_np.shape[2] == 3:
                            cell_gray = cv2.cvtColor(cell_np, cv2.COLOR_RGB2GRAY)
                        else:
                            cell_gray = cell_np
                            
                        cell_gray_pil = Image.fromarray(cell_gray)
                        text2 = pytesseract.image_to_string(cell_gray_pil, config='--psm 6').strip()
                        if text2:
                            ocr_results.append(text2)
                    except Exception:
                        pass
                    
                    # 3. Try with binary image (black text on white background)
                    try:
                        _, cell_binary = cv2.threshold(cell_gray, 150, 255, cv2.THRESH_BINARY)
                        cell_binary_pil = Image.fromarray(cell_binary)
                        text3 = pytesseract.image_to_string(cell_binary_pil, config='--psm 6').strip()
                        if text3:
                            ocr_results.append(text3)
                    except Exception:
                        pass
                    
                    # 4. Try with inverted binary image (white text on black background)
                    try:
                        _, cell_binary_inv = cv2.threshold(cell_gray, 150, 255, cv2.THRESH_BINARY_INV)
                        cell_binary_inv_pil = Image.fromarray(cell_binary_inv)
                        text4 = pytesseract.image_to_string(cell_binary_inv_pil, config='--psm 6').strip()
                        if text4:
                            ocr_results.append(text4)
                    except Exception:
                        pass
                    
                    # 5. Try with adaptive thresholding
                    try:
                        cell_adaptive = cv2.adaptiveThreshold(
                            cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2
                        )
                        cell_adaptive_pil = Image.fromarray(cell_adaptive)
                        text5 = pytesseract.image_to_string(cell_adaptive_pil, config='--psm 6').strip()
                        if text5:
                            ocr_results.append(text5)
                    except Exception:
                        pass
                    
                    # 6. Try with resized image (2x)
                    try:
                        h, w = cell_gray.shape
                        cell_resized = cv2.resize(cell_gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                        cell_resized_pil = Image.fromarray(cell_resized)
                        text6 = pytesseract.image_to_string(cell_resized_pil, config='--psm 6').strip()
                        if text6:
                            ocr_results.append(text6)
                    except Exception:
                        pass
                    
                    # Choose the best result (longest text or first non-empty)
                    if ocr_results:
                        # Sort by length and take the longest result
                        ocr_results.sort(key=len, reverse=True)
                        cell_text = ocr_results[0]
                    else:
                        # Fallback to placeholder if all OCR attempts fail
                        cell_text = f"Cell ({cell[0]}, {cell[1]})"
                    
                    row_data.append(cell_text)
                table_data.append(row_data)
            
            # Create a DataFrame
            if table_data:
                df = pd.DataFrame(table_data)
                all_tables.append(df)
                
                # Print the table as Markdown
                print(f"\nTable {table_num+1} content:")
                md = convert_table_to_markdown(df)
                print(md)
    
    print(f"\nExtracted {len(all_tables)} table(s) in total")
    return all_tables

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli_test.py <pdf_path> [engine]")
        print("Available engines: tabula, pdfplumber, pdfminer, docling, camelot-lattice, camelot-stream, table-transformer (default: tabula)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else "tabula"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    if engine == "tabula":
        extract_with_tabula(pdf_path)
    elif engine == "pdfplumber":
        extract_with_pdfplumber(pdf_path)
    elif engine == "pdfminer":
        extract_with_pdfminer(pdf_path)
    elif engine == "docling":
        extract_with_docling(pdf_path)
    elif engine == "camelot-lattice":
        extract_with_camelot(pdf_path, flavor='lattice')
    elif engine == "camelot-stream":
        extract_with_camelot(pdf_path, flavor='stream')
    elif engine == "table-transformer":
        extract_with_table_transformer(pdf_path)
    else:
        print(f"Unknown engine: {engine}")
        print("Available engines: tabula, pdfplumber, pdfminer, docling, camelot-lattice, camelot-stream, table-transformer")
        sys.exit(1)

if __name__ == "__main__":
    main()