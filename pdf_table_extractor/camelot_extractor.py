import camelot
import pandas as pd
from pdf_table_extractor.extractor_base import PDFTableExtractor

class CamelotExtractor(PDFTableExtractor):
    """
    PDF Table extractor implementation using Camelot library.
    Camelot is a Python library that can extract tables from PDFs.
    It has two extraction methods: Lattice and Stream.
    """
    
    def __init__(self, flavor='lattice'):
        """
        Initialize the Camelot extractor with a specific flavor.
        
        Args:
            flavor (str): The extraction method to use, either 'lattice' or 'stream'.
                          'lattice' is used for tables with clearly demarcated lines.
                          'stream' is used for tables with whitespaces between cells.
        """
        self.flavor = flavor
        
    def extract_tables(self, pdf_path: str) -> list:
        """
        Extract tables from a PDF using Camelot.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of pandas DataFrames representing the tables.
        """
        # Extract tables using Camelot
        tables = camelot.read_pdf(pdf_path, flavor=self.flavor, pages='all')
        
        # Convert Camelot tables to pandas DataFrames
        dataframes = []
        for table in tables:
            # Get the table as a pandas DataFrame
            df = table.df
            
            # Add the table to the list
            dataframes.append(df)
        
        return dataframes