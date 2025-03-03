import os
import sys
import pandas as pd
import tabula
import pdfplumber
from pdfminer.high_level import extract_text
from docling.document_converter import DocumentConverter
import camelot

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli_test.py <pdf_path> [engine]")
        print("Available engines: tabula, pdfplumber, pdfminer, docling, camelot-lattice, camelot-stream (default: tabula)")
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
    else:
        print(f"Unknown engine: {engine}")
        print("Available engines: tabula, pdfplumber, pdfminer, docling, camelot-lattice, camelot-stream")
        sys.exit(1)

if __name__ == "__main__":
    main()