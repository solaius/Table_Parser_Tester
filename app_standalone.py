from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import markdown
import tabula
import pdfplumber
from pdfminer.high_level import extract_text
from docling.document_converter import DocumentConverter
import camelot
import torch
import warnings
import pytesseract
from transformers import DetrImageProcessor, TableTransformerForObjectDetection, logging as transformers_logging
from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()  # Only show errors, not warnings

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

class TableTransformerExtractor(PDFTableExtractor):
    """
    PDF Table extractor implementation using Microsoft's Table Transformer (TATR) model.
    This extractor uses the table-transformer-detection model from Hugging Face to detect
    tables in PDF documents and then extracts their content.
    """
    
    def __init__(self):
        """
        Initialize the Table Transformer extractor.
        Loads the pre-trained model from Hugging Face.
        """
        # Load the model and processor with updated parameters to avoid deprecation warnings
        self.processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection",
            do_resize=True
        )
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection",
            ignore_mismatched_sizes=True  # Ignore mismatched sizes to suppress warnings
        )
        
    def extract_tables(self, pdf_path: str) -> list:
        """
        Extract tables from a PDF using Table Transformer.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of pandas DataFrames representing the tables.
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        # List to store all extracted tables
        all_tables = []
        
        # Process each page
        for page_num, image in enumerate(images):
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Process the image with the model
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            
            # Convert outputs to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            # Extract table regions
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i) for i in box.tolist()]
                
                # Extract the table region from the image
                table_image = image.crop((box[0], box[1], box[2], box[3]))
                
                # Convert table image to grayscale for better OCR
                table_image_np = np.array(table_image)
                table_image_gray = cv2.cvtColor(table_image_np, cv2.COLOR_RGB2GRAY)
                
                # Apply adaptive thresholding to enhance table structure
                table_image_binary = cv2.adaptiveThreshold(
                    table_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Find contours to detect cells
                contours, _ = cv2.findContours(
                    table_image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Filter contours to find table cells
                cells = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter out very small contours and very large ones
                    if w > 20 and h > 20 and w < table_image_np.shape[1] * 0.9 and h < table_image_np.shape[0] * 0.9:
                        cells.append((x, y, x + w, y + h))
                
                # If no cells were detected, try a different approach or skip
                if not cells:
                    continue
                
                # Sort cells by y-coordinate (row) and then by x-coordinate (column)
                cells.sort(key=lambda cell: (cell[1], cell[0]))
                
                # Group cells into rows based on y-coordinate
                rows = []
                current_row = [cells[0]]
                row_y = cells[0][1]
                
                for cell in cells[1:]:
                    # If the cell is in the same row (y-coordinate within a threshold)
                    if abs(cell[1] - row_y) < 20:
                        current_row.append(cell)
                    else:
                        # Sort the current row by x-coordinate
                        current_row.sort(key=lambda c: c[0])
                        rows.append(current_row)
                        # Start a new row
                        current_row = [cell]
                        row_y = cell[1]
                
                # Add the last row
                if current_row:
                    current_row.sort(key=lambda c: c[0])
                    rows.append(current_row)
                
                # Create a DataFrame from the extracted cells
                table_data = []
                for row in rows:
                    row_data = []
                    for cell in row:
                        # Extract text from the cell using OCR
                        cell_image = table_image.crop((cell[0], cell[1], cell[2], cell[3]))
                        
                        # Preprocess the cell image to improve OCR accuracy
                        cell_np = np.array(cell_image)
                        
                        # Try multiple preprocessing techniques for better OCR results
                        ocr_results = []
                        
                        # 1. Try with original image
                        try:
                            text1 = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
                            if text1:
                                ocr_results.append(text1)
                        except Exception:
                            pass
                        
                        # 2. Try with grayscale image
                        try:
                            if len(cell_np.shape) == 3 and cell_np.shape[2] == 3:
                                cell_gray = cv2.cvtColor(cell_np, cv2.COLOR_RGB2GRAY)
                            else:
                                cell_gray = cell_np
                                
                            cell_gray_pil = Image.fromarray(cell_gray)
                            text2 = pytesseract.image_to_string(cell_gray_pil, config='--psm 6').strip()
                            if text2:
                                ocr_results.append(text2)
                        except Exception:
                            pass
                        
                        # 3. Try with binary image (black text on white background)
                        try:
                            _, cell_binary = cv2.threshold(cell_gray, 150, 255, cv2.THRESH_BINARY)
                            cell_binary_pil = Image.fromarray(cell_binary)
                            text3 = pytesseract.image_to_string(cell_binary_pil, config='--psm 6').strip()
                            if text3:
                                ocr_results.append(text3)
                        except Exception:
                            pass
                        
                        # 4. Try with inverted binary image (white text on black background)
                        try:
                            _, cell_binary_inv = cv2.threshold(cell_gray, 150, 255, cv2.THRESH_BINARY_INV)
                            cell_binary_inv_pil = Image.fromarray(cell_binary_inv)
                            text4 = pytesseract.image_to_string(cell_binary_inv_pil, config='--psm 6').strip()
                            if text4:
                                ocr_results.append(text4)
                        except Exception:
                            pass
                        
                        # 5. Try with adaptive thresholding
                        try:
                            cell_adaptive = cv2.adaptiveThreshold(
                                cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2
                            )
                            cell_adaptive_pil = Image.fromarray(cell_adaptive)
                            text5 = pytesseract.image_to_string(cell_adaptive_pil, config='--psm 6').strip()
                            if text5:
                                ocr_results.append(text5)
                        except Exception:
                            pass
                        
                        # 6. Try with resized image (2x)
                        try:
                            h, w = cell_gray.shape
                            cell_resized = cv2.resize(cell_gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                            cell_resized_pil = Image.fromarray(cell_resized)
                            text6 = pytesseract.image_to_string(cell_resized_pil, config='--psm 6').strip()
                            if text6:
                                ocr_results.append(text6)
                        except Exception:
                            pass
                        
                        # Choose the best result (longest text or first non-empty)
                        if ocr_results:
                            # Sort by length and take the longest result
                            ocr_results.sort(key=len, reverse=True)
                            cell_text = ocr_results[0]
                        else:
                            # Fallback to placeholder if all OCR attempts fail
                            cell_text = f"Cell ({cell[0]}, {cell[1]})"
                        
                        row_data.append(cell_text)
                    table_data.append(row_data)
                
                # Create a DataFrame
                if table_data:
                    df = pd.DataFrame(table_data)
                    all_tables.append(df)
        
        return all_tables

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
    # To allow the app to be embedded in iframes, we need to set CORS headers
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 54800))
    # Use DEBUG environment variable if available
    debug_mode = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"Starting web app on port {port} with debug mode {'enabled' if debug_mode else 'disabled'}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)