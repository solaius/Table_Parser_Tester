# PDF Table Extraction Tool

A modular PDF table extraction tool with a web UI that supports multiple extraction engines, including AI-powered table extraction with Granite Vision.

> **Note:** If you encounter any issues, please check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide.

## Features

- Extract tables from PDF files using different extraction engines:
  - Tabula
  - pdfplumber
  - PDFMiner
  - Docling
  - Camelot (with both Lattice and Stream methods)
  - Table Transformer (TATR) - Microsoft's deep learning model for table detection with OCR text extraction
  - **NEW: Docling with Granite Vision** - IBM's multimodal AI model for enhanced table extraction
- GPU acceleration support for faster processing
- Web interface for easy uploading and viewing of results
- View both raw Markdown and rendered HTML tables
- Modular design for easy extension with additional extraction engines

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/solaius/Table_Parser_Tester.git
   cd Table_Parser_Tester
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

5. (Optional) Set up environment variables in a `.env` file:
   ```
   # Copy the example environment file
   cp .env_example .env
   
   # Edit the .env file with your settings
   # For Granite Vision API endpoint
   GRANITE_VISION_ENDPOINT=https://your-granite-vision-endpoint.com
   OPENAI_COMPLETIONS=/v1/chat/completions
   GRANITE_VISION_MODEL_NAME=granite-vision-3-2
   
   # For GPU acceleration
   USE_GPU=true  # Set to false to use CPU only
   GPU_DEVICE=0  # Device ID (usually 0 for the first GPU)
   
   # Web app settings
   PORT=54800
   ```

6. Run the web application:
   ```
   python run_web_app.py
   ```

7. Open your browser and navigate to:
   ```
   http://localhost:54800  # Or the port you specified in .env
   ```

## Usage

### Web Interface

1. Upload a PDF file using the web interface
2. Select the extraction engine:
   - Tabula
   - pdfplumber
   - PDFMiner
   - Docling
   - Camelot (Lattice or Stream)
   - Table Transformer
   - **Docling with Granite Vision**
3. Click "Extract Tables"
4. View the extracted tables in both Markdown and rendered HTML formats

### Command Line

You can also use the command-line interface:

```
python -m pdf_table_extractor.main path/to/your/file.pdf --engine tabula
```

Available engines: `tabula`, `pdfplumber`, `pdfminer`, `docling`, `camelot-lattice`, `camelot-stream`, `table-transformer`, `docling-granite-vision`

### Testing Extractors and API Connections

To test all extractors and API connections:

```bash
# Run all tests
python tests/run_all_tests.py --all

# Run a specific test
python tests/run_all_tests.py --test test_cuda.py

# List available tests
python tests/run_all_tests.py --list

# Test Granite Vision API connection
python tests/granite_vision_test.py
```

The test suite includes tests for all extractors and CUDA availability. Each test will generate a log file in the `logs` directory.

### GPU Acceleration Setup

To enable GPU acceleration for faster processing:

1. Install CUDA and cuDNN (see below)
2. Install PyTorch with CUDA support:
   ```
   # For CUDA 12.6 (recommended)
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Set `USE_GPU=true` in your `.env` file
4. Specify the GPU device ID with `GPU_DEVICE=0` (use 0 for the first GPU)

#### Installing CUDA and cuDNN

**For Windows:**
1. Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Download and install [cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA Developer account)

**For Linux:**
```bash
# Example for Ubuntu
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit
# Install cuDNN following NVIDIA's instructions
```

**For macOS:**
Note that recent macOS versions with Apple Silicon use Metal instead of CUDA. For Intel Macs, follow NVIDIA's instructions.

## Project Structure

```
Table_Parser_Tester/
├── pdf_table_extractor/
│   ├── __init__.py
│   ├── extractor_base.py          # Abstract base class/interface
│   ├── tabula_extractor.py        # Tabula implementation
│   ├── pdfplumber_extractor.py    # pdfplumber implementation
│   ├── pdfminer_extractor.py      # PDFMiner implementation
│   ├── docling_extractor.py       # Docling implementation
│   ├── docling_granitevision.py   # Docling with Granite Vision implementation
│   ├── camelot_extractor.py       # Camelot implementation
│   ├── tatr_extractor.py          # Table Transformer implementation
│   ├── main.py                    # CLI interface
│   ├── app.py                     # Flask-based web UI
│   └── templates/
│       ├── index.html             # File upload form
│       └── results.html           # Display extracted tables
├── table_pdfs/                    # Sample PDF files for testing
├── tests/                         # Test scripts for all extractors
│   ├── test_cuda.py               # Test CUDA availability
│   ├── test_tabula_extractor.py   # Test Tabula extractor
│   ├── test_pdfplumber_extractor.py # Test PDFPlumber extractor
│   ├── test_pdfminer_extractor.py # Test PDFMiner extractor
│   ├── test_docling_extractor.py  # Test Docling extractor
│   ├── test_docling_granitevision.py # Test Docling with Granite Vision
│   ├── test_camelot_extractor.py  # Test Camelot extractor
│   ├── test_tatr_extractor.py     # Test Table Transformer extractor
│   ├── granite_vision_test.py     # Test Granite Vision API
│   ├── test_granite_extractor.py  # Test Granite Vision extractor
│   └── run_all_tests.py           # Run all tests
├── logs/                          # Log files directory
├── run_web_app.py                 # Web application runner
├── app_standalone.py              # Standalone application
├── requirements.txt               # Project dependencies
├── .env_example                   # Example environment variables
├── CURRENT_STATE.md               # Current state of the project
├── TROUBLESHOOTING.md             # Troubleshooting guide
└── README.md                      # This file
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
- reportlab
- requests
- python-dotenv
- langchain_community
- langchain_huggingface
- replicate

### Optional Dependencies for GPU Acceleration

- CUDA Toolkit
- cuDNN
- PyTorch with CUDA support

## License

MIT