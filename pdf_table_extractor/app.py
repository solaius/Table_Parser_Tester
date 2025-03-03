from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import markdown
import zipfile

from pdf_table_extractor.tabula_extractor import TabulaExtractor
from pdf_table_extractor.pdfplumber_extractor import PdfPlumberExtractor
from pdf_table_extractor.pdfminer_extractor import PDFMinerExtractor
from pdf_table_extractor.docling_extractor import DoclingExtractor
from pdf_table_extractor.camelot_extractor import CamelotExtractor
from pdf_table_extractor.tatr_extractor import TableTransformerExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add CORS headers to allow embedding in iframes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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
        elif engine == 'table-transformer':
            extractor = TableTransformerExtractor()
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
    app.run(host='0.0.0.0', port=54656, debug=True)