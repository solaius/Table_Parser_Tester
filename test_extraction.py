import tabula
import pandas as pd
import pdfplumber
from pdfminer.high_level import extract_text

def test_tabula_extraction(pdf_path):
    print(f"\n=== Testing Tabula Extraction on {pdf_path} ===")
    try:
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        print(f"Found {len(tables)} table(s)")
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            print(table)
    except Exception as e:
        print(f"Error with Tabula: {e}")

def test_pdfplumber_extraction(pdf_path):
    print(f"\n=== Testing pdfplumber Extraction on {pdf_path} ===")
    try:
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
            print(table)
    except Exception as e:
        print(f"Error with pdfplumber: {e}")

def test_pdfminer_extraction(pdf_path):
    print(f"\n=== Testing PDFMiner Extraction on {pdf_path} ===")
    try:
        text = extract_text(pdf_path)
        print("Extracted text (first 500 characters):")
        print(text[:500])
        print("...")
    except Exception as e:
        print(f"Error with PDFMiner: {e}")

if __name__ == "__main__":
    # Test with the sample PDFs
    pdfs = ['/workspace/sample_table.pdf', '/workspace/multi_table.pdf']
    
    for pdf_path in pdfs:
        print(f"\n\n{'='*50}")
        print(f"Testing extraction on: {pdf_path}")
        print(f"{'='*50}")
        
        test_tabula_extraction(pdf_path)
        test_pdfplumber_extraction(pdf_path)
        test_pdfminer_extraction(pdf_path)