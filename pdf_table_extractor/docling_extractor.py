import pandas as pd
from docling.document_converter import DocumentConverter
from pdf_table_extractor.extractor_base import PDFTableExtractor

class DoclingExtractor(PDFTableExtractor):
    """
    PDF Table extractor implementation using Docling library.
    Docling is a document processing library that can extract tables from PDFs.
    """
    
    def extract_tables(self, pdf_path: str) -> list:
        """
        Extract tables from a PDF using Docling.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of pandas DataFrames representing the tables.
        """
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
            for table in doc.tables:
                # Use the built-in method to export to DataFrame if available
                if hasattr(table, 'export_to_dataframe'):
                    df = table.export_to_dataframe()
                    tables.append(df)
        
        return tables