from abc import ABC, abstractmethod

class PDFTableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, pdf_path: str) -> list:
        """
        Extract tables from a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of tables (e.g., pandas DataFrames).
        """
        pass