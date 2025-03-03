# PDF Table Extraction Tool

A modular PDF table extraction tool with a web UI that supports multiple extraction engines.

> **Note:** If you encounter any issues, please check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide.

## Features

- Extract tables from PDF files using different extraction engines:
  - Tabula
  - pdfplumber
  - PDFMiner
  - Docling
  - Camelot (with both Lattice and Stream methods)
  - Table Transformer (TATR) - Microsoft's deep learning model for table detection with OCR text extraction
- Web interface for easy uploading and viewing of results
- View both raw Markdown and rendered HTML tables
- Modular design for easy extension with additional extraction engines

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-table-extractor.git
   cd pdf-table-extractor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Java (required for Tabula):
   
   **Ubuntu/Debian:**
   ```
   sudo apt-get update && sudo apt-get install -y default-jre
   ```
   
   **macOS:**
   ```
   brew install openjdk
   ```
   
   **Windows:**
   Download and install Java from [java.com](https://www.java.com/download/)

4. Install Tesseract OCR (required for Table Transformer):

   **Ubuntu/Debian:**
   ```
   sudo apt-get update && sudo apt-get install -y tesseract-ocr
   ```
   
   **macOS:**
   ```
   brew install tesseract
   ```
   
   **Windows:**
   Download and install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

5. Run the web application:
   ```
   python run_web_app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:54656
   ```

## Usage

### Web Interface

1. Upload a PDF file using the web interface
2. Select the extraction engine (Tabula, pdfplumber, or PDFMiner)
3. Click "Extract Tables"
4. View the extracted tables in both Markdown and rendered HTML formats

### Command Line

You can also use the command-line interface:

```
python -m pdf_table_extractor.main path/to/your/file.pdf --engine tabula
```

Available engines: `tabula`, `pdfplumber`, `pdfminer`, `docling`, `camelot-lattice`, `camelot-stream`, `table-transformer`

## Project Structure

```
pdf_table_extractor/
├── __init__.py
├── extractor_base.py          # Abstract base class/interface
├── tabula_extractor.py        # Tabula implementation
├── pdfplumber_extractor.py    # pdfplumber implementation
├── pdfminer_extractor.py      # PDFMiner implementation
├── docling_extractor.py       # Docling implementation
├── camelot_extractor.py       # Camelot implementation
├── tatr_extractor.py          # Table Transformer implementation
├── main.py                    # CLI interface
├── app.py                     # Flask-based web UI
└── templates/
    ├── index.html             # File upload form
    └── results.html           # Display extracted tables
```

## Extending with New Extractors

To add a new extraction engine, create a new class that inherits from `PDFTableExtractor` and implements the `extract_tables` method:

```python
from pdf_table_extractor.extractor_base import PDFTableExtractor

class NewExtractor(PDFTableExtractor):
    def extract_tables(self, pdf_path: str) -> list:
        # Implement your extraction logic here
        tables = []
        # ...
        return tables
```

Then update the web UI and CLI to include your new extractor.

## Dependencies

- tabula-py
- pdfplumber
- pdfminer.six
- docling
- camelot-py
- opencv-python
- pypdfium2
- transformers
- torch
- timm
- pdf2image
- Pillow
- pytesseract
- pandas
- Flask
- Markdown
- tabulate

## License

MIT