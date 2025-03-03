import argparse
from pdf_table_extractor.tabula_extractor import TabulaExtractor
from pdf_table_extractor.pdfplumber_extractor import PdfPlumberExtractor
from pdf_table_extractor.pdfminer_extractor import PDFMinerExtractor
from pdf_table_extractor.docling_extractor import DoclingExtractor
from pdf_table_extractor.camelot_extractor import CamelotExtractor
from pdf_table_extractor.tatr_extractor import TableTransformerExtractor

def main():
    parser = argparse.ArgumentParser(description="PDF Table Extraction Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--engine", choices=["tabula", "pdfplumber", "pdfminer", "docling", 
                                         "camelot-lattice", "camelot-stream", "table-transformer"], 
                        default="tabula", help="Choose the extraction engine")
    args = parser.parse_args()

    if args.engine == "tabula":
        extractor = TabulaExtractor()
    elif args.engine == "pdfplumber":
        extractor = PdfPlumberExtractor()
    elif args.engine == "pdfminer":
        extractor = PDFMinerExtractor()
    elif args.engine == "docling":
        extractor = DoclingExtractor()
    elif args.engine == "camelot-lattice":
        extractor = CamelotExtractor(flavor='lattice')
    elif args.engine == "camelot-stream":
        extractor = CamelotExtractor(flavor='stream')
    elif args.engine == "table-transformer":
        extractor = TableTransformerExtractor()

    tables = extractor.extract_tables(args.pdf_path)
    print(f"Extracted {len(tables)} table(s) using {args.engine}")

    for i, table in enumerate(tables):
        print(f"Table {i+1}:\n", table)

if __name__ == "__main__":
    main()