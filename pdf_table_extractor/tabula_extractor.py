import tabula
from pdf_table_extractor.extractor_base import PDFTableExtractor

class TabulaExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
        # Use Tabula to read all tables from the PDF
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        return tables