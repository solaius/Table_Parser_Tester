from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import markdown
import tabula
import pdfplumber
from pdfminer.high_level import extract_text
from docling.document_converter import DocumentConverter
import camelot

app = Flask(__name__, template_folder='pdf_table_extractor/templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class PDFTableExtractor:
    def extract_tables(self, pdf_path: str) -> list:
        """
        Extract tables from a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of tables (e.g., pandas DataFrames).
        """
        pass

class TabulaExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
        # Use Tabula to read all tables from the PDF
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        return tables

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

class PDFMinerExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
        # PDFMiner is better for text extraction.
        text = extract_text(pdf_path)
        # Implement custom logic to parse 'text' into table structures.
        tables = []
        # TODO: Add custom table parsing logic here
        return tables

class DoclingExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
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

class CamelotExtractor(PDFTableExtractor):
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Choose extractor based on user selection (default: tabula)
        engine = request.form.get('engine', 'tabula')
        if engine == 'tabula':
            extractor = TabulaExtractor()
        elif engine == 'pdfplumber':
            extractor = PdfPlumberExtractor()
        elif engine == 'pdfminer':
            extractor = PDFMinerExtractor()
        elif engine == 'docling':
            extractor = DoclingExtractor()
        elif engine == 'camelot-lattice':
            extractor = CamelotExtractor(flavor='lattice')
        elif engine == 'camelot-stream':
            extractor = CamelotExtractor(flavor='stream')
        else:
            extractor = TabulaExtractor()
        
        tables = extractor.extract_tables(file_path)
        md_tables = []
        rendered_tables = []
        
        for i, df in enumerate(tables):
            if not isinstance(df, pd.DataFrame):
                continue
            md = convert_table_to_markdown(df)
            # Convert Markdown to HTML (using markdown extension for tables)
            html = markdown.markdown(md, extensions=['tables'])
            md_tables.append(md)
            rendered_tables.append(html)
        
        return render_template('results.html', 
                              md_tables=md_tables, 
                              rendered_tables=rendered_tables,
                              num_tables=len(md_tables))
    
    return render_template('index.html')

if __name__ == '__main__':
    # To allow the app to be embedded in iframes, we need to set CORS headers
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    app.run(host='0.0.0.0', port=54656, debug=True)