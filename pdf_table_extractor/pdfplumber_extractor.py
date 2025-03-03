import pdfplumber
import pandas as pd
from pdf_table_extractor.extractor_base import PDFTableExtractor

class PdfPlumberExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw_table = page.extract_table()
                if raw_table:
                    # Assume first row as header and the rest as data
                    table_df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
                    tables.append(table_df)
        return tables