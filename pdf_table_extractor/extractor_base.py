from abc import ABC, abstractmethod
import json

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
    
    def get_raw_output(self, pdf_path: str) -> list:
        """
        Get the raw output from the extractor.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of raw outputs for each table.
        """
        # Default implementation returns None for each table
        # Subclasses should override this method to provide the raw output
        tables = self.extract_tables(pdf_path)
        return [None] * len(tables)
    
    def get_raw_output_as_json(self, pdf_path: str) -> list:
        """
        Get the raw output from the extractor as JSON strings.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of JSON strings representing the raw output for each table.
        """
        raw_outputs = self.get_raw_output(pdf_path)
        json_outputs = []
        
        for output in raw_outputs:
            try:
                if output is None:
                    json_outputs.append("No raw output available for this extractor")
                else:
                    # Try to convert to JSON
                    json_str = json.dumps(output, indent=2)
                    json_outputs.append(json_str)
            except Exception as e:
                json_outputs.append(f"Error converting raw output to JSON: {str(e)}")
        
        return json_outputs