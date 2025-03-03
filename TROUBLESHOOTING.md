# Troubleshooting Guide

## Common Issues and Solutions

### 1. Flask `allow_iframe` Error

**Error:**
```
TypeError: run_simple() got an unexpected keyword argument 'allow_iframe'
```

**Solution:**
The `allow_iframe=True` parameter is not a valid parameter for Flask's `app.run()` method. Remove this parameter from the `run_web_app.py` file:

```python
# Incorrect
app.run(host='0.0.0.0', port=8009, debug=True, allow_iframe=True)

# Correct
app.run(host='0.0.0.0', port=8009, debug=True)
```

To enable iframe embedding, use CORS headers instead:

```python
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
```

### 2. Tabula Java Dependency Error

**Error:**
```
Error with Tabula: `java` command is not found from this Python process.
Please ensure Java is installed and PATH is set for `java`
```

**Solution:**
Tabula requires Java to be installed. Install Java using your system's package manager:

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y default-jre
```

**macOS:**
```bash
brew install openjdk
```

**Windows:**
Download and install Java from [java.com](https://www.java.com/download/)

### 3. Import Errors

**Error:**
```
ImportError: cannot import name 'X' from 'pdf_table_extractor.Y'
```

**Solution:**
Make sure all the import statements use the correct package structure. If you're running the code from outside the project directory, use absolute imports:

```python
# Change this:
from .tabula_extractor import TabulaExtractor

# To this:
from pdf_table_extractor.tabula_extractor import TabulaExtractor
```

### 4. No Tables Found

**Issue:**
The extractor doesn't find any tables in the PDF.

**Solution:**
- Try a different extraction engine (tabula, pdfplumber, pdfminer)
- Check if the PDF actually contains tables in a format that can be recognized
- For complex PDFs, you might need to customize the extraction parameters

### 5. Markdown Rendering Issues

**Issue:**
Tables don't render correctly in the web UI.

**Solution:**
Make sure you have the `markdown` package installed with the `tables` extension:

```bash
pip install markdown
```

If you're still having issues, try installing `tabulate` for better Markdown table support:

```bash
pip install tabulate
```