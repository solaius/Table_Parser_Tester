import pandas as pd
import io
import base64
import os
import sys
import logging
from PIL import Image, ImageOps
from typing import List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='granite_vision.log',
    filemode='a'
)

# Load environment variables from .env file
load_dotenv()

from pdf_table_extractor.extractor_base import PDFTableExtractor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import TableItem
from docling_core.types.doc.document import DocItemLabel

# Check Python version
assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12."

class DoclingGraniteVisionExtractor(PDFTableExtractor):
    """
    PDF Table extractor implementation using IBM's Docling and Granite Vision 3.2 2b model.
    This extractor uses a multimodal approach to extract tables from PDFs:
    1. Docling for document processing and initial table detection
    2. Granite Vision 3.2 2b model for enhanced table understanding and extraction
    """
    
    def __init__(self, api_endpoint: Optional[str] = None, replicate_api_token: Optional[str] = None):
        """
        Initialize the DoclingGraniteVisionExtractor.
        
        Args:
            api_endpoint (str, optional): Custom API endpoint for Granite Vision model.
                Defaults to the value from .env file or using Replicate if None.
            replicate_api_token (str, optional): Replicate API token for accessing the model.
                If None, will try to get from environment variable REPLICATE_API_TOKEN.
        """
        # Get API endpoint from parameters or environment variable
        if api_endpoint is None:
            self.api_endpoint = os.getenv("GRANITE_VISION_ENDPOINT")
        else:
            self.api_endpoint = api_endpoint
        
        # Get Replicate API token from parameters or environment variable
        if replicate_api_token is None:
            self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
        else:
            self.replicate_api_token = replicate_api_token
            
        # Initialize the vision model client
        if self.api_endpoint:
            # Use custom endpoint if provided
            self._init_custom_endpoint()
        else:
            # Use Replicate if no custom endpoint
            self._init_replicate()
    
    def _init_replicate(self):
        """Initialize the Replicate client for Granite Vision model."""
        try:
            from langchain_community.llms import Replicate
            from ibm_granite_community.notebook_utils import get_env_var
            
            if not self.replicate_api_token:
                self.replicate_api_token = get_env_var("REPLICATE_API_TOKEN")
                
            self.vision_model = Replicate(
                model=os.getenv("GRANITE_VISION_MODEL_NAME", "granite-vision-3-2"),
                replicate_api_token=self.replicate_api_token,
                model_kwargs={
                    "max_tokens": 1000,
                    "min_tokens": 100,
                },
            )
            self.use_replicate = True
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not initialize Replicate client: {e}")
            logging.info("Falling back to basic Docling extraction without Granite Vision.")
            self.use_replicate = False
    
    def _init_custom_endpoint(self):
        """Initialize client for custom API endpoint."""
        try:
            import requests
            self.use_replicate = False
            self.use_custom_endpoint = True
            
            # Get model name from environment variable or use default
            self.model_name = os.getenv("GRANITE_VISION_MODEL_NAME", "granite-vision-3-2")
            
            # Get API key from environment variable
            self.api_key = os.getenv("GRANITE_VISION_MODEL_API_KEY", "")
            
            # Log successful initialization
            logging.info(f"Successfully initialized custom endpoint client for {self.model_name}")
            logging.info(f"Using endpoint: {self.api_endpoint}")
            
            # Test connection
            self._test_endpoint_connection()
            
        except ImportError:
            logging.warning("Could not import requests library for custom endpoint.")
            logging.info("Falling back to basic Docling extraction without Granite Vision.")
            self.use_custom_endpoint = False
            
    def _test_endpoint_connection(self):
        """Test the connection to the custom endpoint."""
        try:
            import requests
            import json
            
            # Simple test prompt
            prompt = "Hello, are you working?"
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key to headers if provided
            if hasattr(self, 'api_key') and self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make the API request
            logging.info("Testing connection to Granite Vision API...")
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info("✅ Connection to Granite Vision API successful!")
                return True
            else:
                logging.warning(f"❌ Connection to Granite Vision API failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Error testing endpoint connection: {e}")
            return False
    
    def _encode_image(self, image: Image.Image, format: str = "png") -> str:
        """
        Encode an image to base64 for API requests.
        
        Args:
            image (Image.Image): PIL Image to encode
            format (str): Image format (default: png)
            
        Returns:
            str: Base64 encoded image with data URI prefix
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format)
        encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format};base64,{encoding}"
    
    def _process_table_with_vision(self, table, image_data=None):
        """
        Process a table using the Granite Vision model.
        
        Args:
            table: The table object from Docling
            image_data: Optional image data if available
            
        Returns:
            pandas.DataFrame: The processed table as a DataFrame
        """
        # If we don't have vision capabilities, fall back to basic export
        if not hasattr(self, 'use_replicate') or not self.use_replicate:
            if hasattr(self, 'use_custom_endpoint') and self.use_custom_endpoint:
                return self._process_with_custom_endpoint(table, image_data)
            else:
                # Fall back to basic Docling table export
                if hasattr(table, 'export_to_dataframe'):
                    return table.export_to_dataframe()
                else:
                    # Create empty DataFrame if export method not available
                    return pd.DataFrame()
        
        # Convert table to markdown for the vision model
        table_markdown = table.export_to_markdown()
        
        # If we have image data, use it for multimodal understanding
        if image_data:
            try:
                image = Image.open(io.BytesIO(image_data))
                image_uri = self._encode_image(image)
                
                # Create a prompt that includes both the image and initial table markdown
                prompt = f"""
                I need to extract and structure this table accurately. 
                Here's the image of the table: {image_uri}
                
                I have an initial extraction in markdown format:
                {table_markdown}
                
                Please analyze the image and correct any errors in the table structure.
                Return the corrected table in markdown format.
                """
                
                # Get improved table from vision model
                improved_table_md = self.vision_model(prompt)
                
                # Convert the markdown back to DataFrame
                # This is a simple conversion and might need improvement
                try:
                    # Try to parse the markdown table
                    from io import StringIO
                    # Extract just the table part (between lines with |)
                    table_lines = []
                    capture = False
                    for line in improved_table_md.split('\n'):
                        if '|' in line:
                            capture = True
                            table_lines.append(line)
                        elif capture and '|' not in line:
                            # Stop capturing if we've moved past the table
                            break
                    
                    if table_lines:
                        # Clean up the markdown table for pandas
                        cleaned_lines = []
                        for line in table_lines:
                            # Remove leading/trailing whitespace and | characters
                            cleaned = line.strip()
                            if cleaned.startswith('|'):
                                cleaned = cleaned[1:]
                            if cleaned.endswith('|'):
                                cleaned = cleaned[:-1]
                            cleaned_lines.append(cleaned)
                        
                        # Skip separator line (the one with dashes)
                        header = cleaned_lines[0].split('|')
                        data = [row.split('|') for row in cleaned_lines if '---' not in row and row != cleaned_lines[0]]
                        
                        # Create DataFrame
                        df = pd.DataFrame(data, columns=[h.strip() for h in header])
                        return df
                    else:
                        # Fall back to original table if parsing fails
                        if hasattr(table, 'export_to_dataframe'):
                            return table.export_to_dataframe()
                        else:
                            return pd.DataFrame()
                except Exception as e:
                    logging.error(f"Error converting vision model output to DataFrame: {e}")
                    # Fall back to original table
                    if hasattr(table, 'export_to_dataframe'):
                        return table.export_to_dataframe()
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error processing table with vision model: {e}")
                # Fall back to basic Docling table export
                if hasattr(table, 'export_to_dataframe'):
                    return table.export_to_dataframe()
                else:
                    return pd.DataFrame()
        else:
            # No image data, use text-only approach
            prompt = f"""
            I need to extract and structure this table accurately.
            Here's the table in markdown format:
            {table_markdown}
            
            Please analyze and correct any errors in the table structure.
            Return the corrected table in markdown format.
            """
            
            try:
                # Get improved table from vision model (text-only in this case)
                improved_table_md = self.vision_model(prompt)
                
                # Convert the markdown back to DataFrame (same as above)
                from io import StringIO
                table_lines = []
                capture = False
                for line in improved_table_md.split('\n'):
                    if '|' in line:
                        capture = True
                        table_lines.append(line)
                    elif capture and '|' not in line:
                        break
                
                if table_lines:
                    cleaned_lines = []
                    for line in table_lines:
                        cleaned = line.strip()
                        if cleaned.startswith('|'):
                            cleaned = cleaned[1:]
                        if cleaned.endswith('|'):
                            cleaned = cleaned[:-1]
                        cleaned_lines.append(cleaned)
                    
                    header = cleaned_lines[0].split('|')
                    data = [row.split('|') for row in cleaned_lines if '---' not in row and row != cleaned_lines[0]]
                    
                    df = pd.DataFrame(data, columns=[h.strip() for h in header])
                    return df
                else:
                    if hasattr(table, 'export_to_dataframe'):
                        return table.export_to_dataframe()
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error processing table with text-only approach: {e}")
                if hasattr(table, 'export_to_dataframe'):
                    return table.export_to_dataframe()
                else:
                    return pd.DataFrame()
    
    def _process_with_custom_endpoint(self, table, image_data=None):
        """
        Process a table using a custom API endpoint.
        
        Args:
            table: The table object from Docling
            image_data: Optional image data if available
            
        Returns:
            pandas.DataFrame: The processed table as a DataFrame
        """
        import requests
        import json
        
        # Convert table to markdown for the vision model
        table_markdown = table.export_to_markdown()
        
        # If we have image data, use it for multimodal understanding
        if image_data:
            try:
                image = Image.open(io.BytesIO(image_data))
                image_uri = self._encode_image(image)
                
                # Create a prompt that includes both the image and initial table markdown
                prompt = f"""
                I need to extract and structure this table accurately. 
                Here's the image of the table: {image_uri}
                
                I have an initial extraction in markdown format:
                {table_markdown}
                
                Please analyze the image and correct any errors in the table structure.
                Return the corrected table in markdown format.
                """
                
                # Prepare the request payload (OpenAI-compatible format)
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                # Make the API request
                response = requests.post(
                    self.api_endpoint,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    improved_table_md = result["choices"][0]["message"]["content"]
                    
                    # Convert the markdown back to DataFrame
                    table_lines = []
                    capture = False
                    for line in improved_table_md.split('\n'):
                        if '|' in line:
                            capture = True
                            table_lines.append(line)
                        elif capture and '|' not in line:
                            break
                    
                    if table_lines:
                        cleaned_lines = []
                        for line in table_lines:
                            cleaned = line.strip()
                            if cleaned.startswith('|'):
                                cleaned = cleaned[1:]
                            if cleaned.endswith('|'):
                                cleaned = cleaned[:-1]
                            cleaned_lines.append(cleaned)
                        
                        header = cleaned_lines[0].split('|')
                        data = [row.split('|') for row in cleaned_lines if '---' not in row and row != cleaned_lines[0]]
                        
                        df = pd.DataFrame(data, columns=[h.strip() for h in header])
                        return df
                    else:
                        if hasattr(table, 'export_to_dataframe'):
                            return table.export_to_dataframe()
                        else:
                            return pd.DataFrame()
                else:
                    logging.warning(f"API request failed with status code {response.status_code}: {response.text}")
                    if hasattr(table, 'export_to_dataframe'):
                        return table.export_to_dataframe()
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error processing table with custom endpoint: {e}")
                if hasattr(table, 'export_to_dataframe'):
                    return table.export_to_dataframe()
                else:
                    return pd.DataFrame()
        else:
            # No image data, use text-only approach with custom endpoint
            prompt = f"""
            I need to extract and structure this table accurately.
            Here's the table in markdown format:
            {table_markdown}
            
            Please analyze and correct any errors in the table structure.
            Return the corrected table in markdown format.
            """
            
            try:
                # Prepare the request payload (OpenAI-compatible format)
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
                
                # Make the API request
                response = requests.post(
                    self.api_endpoint,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    improved_table_md = result["choices"][0]["message"]["content"]
                    
                    # Convert the markdown back to DataFrame
                    table_lines = []
                    capture = False
                    for line in improved_table_md.split('\n'):
                        if '|' in line:
                            capture = True
                            table_lines.append(line)
                        elif capture and '|' not in line:
                            break
                    
                    if table_lines:
                        cleaned_lines = []
                        for line in table_lines:
                            cleaned = line.strip()
                            if cleaned.startswith('|'):
                                cleaned = cleaned[1:]
                            if cleaned.endswith('|'):
                                cleaned = cleaned[:-1]
                            cleaned_lines.append(cleaned)
                        
                        header = cleaned_lines[0].split('|')
                        data = [row.split('|') for row in cleaned_lines if '---' not in row and row != cleaned_lines[0]]
                        
                        df = pd.DataFrame(data, columns=[h.strip() for h in header])
                        return df
                    else:
                        if hasattr(table, 'export_to_dataframe'):
                            return table.export_to_dataframe()
                        else:
                            return pd.DataFrame()
                else:
                    logging.warning(f"API request failed with status code {response.status_code}: {response.text}")
                    if hasattr(table, 'export_to_dataframe'):
                        return table.export_to_dataframe()
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error processing table with text-only approach using custom endpoint: {e}")
                if hasattr(table, 'export_to_dataframe'):
                    return table.export_to_dataframe()
                else:
                    return pd.DataFrame()
    
    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF using Docling and Granite Vision.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of pandas DataFrames representing the tables.
        """
        # Create PDF pipeline options with image generation enabled
        pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
        
        # Create a document converter with the specified options
        converter = DocumentConverter(format_options=format_options)
        
        try:
            # Convert the PDF to a Docling document
            result = converter.convert(pdf_path)
            doc = result.document
            
            # Store the document for later use in get_raw_output
            self._last_document = doc
            
            # Extract tables from the document
            tables = []
            
            # Check if the document has tables
            if hasattr(doc, 'tables') and doc.tables:
                for table in doc.tables:
                    # Process only items labeled as tables
                    if hasattr(table, 'label') and table.label == DocItemLabel.TABLE:
                        # Try to get the image data for the table if available
                        image_data = None
                        if hasattr(table, 'get_image') and callable(getattr(table, 'get_image', None)):
                            try:
                                image_data = table.get_image()
                            except:
                                pass
                        
                        # Process the table with Granite Vision if available
                        df = self._process_table_with_vision(table, image_data)
                        if not df.empty:
                            tables.append(df)
            
            return tables
        except Exception as e:
            logging.error(f"Error extracting tables with Docling and Granite Vision: {e}")
            self._last_document = None
            return []
    
    def get_raw_output(self, pdf_path: str) -> list:
        """
        Get the raw output from the extractor.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            list: A list of raw outputs for each table.
        """
        # If we haven't processed this PDF yet, do it now
        if not hasattr(self, '_last_document') or self._last_document is None:
            self.extract_tables(pdf_path)
        
        # If we still don't have a document, return empty list
        if not hasattr(self, '_last_document') or self._last_document is None:
            return []
        
        raw_outputs = []
        
        # Check if the document has tables
        if hasattr(self._last_document, 'tables') and self._last_document.tables:
            for table in self._last_document.tables:
                # Process only items labeled as tables
                if hasattr(table, 'label') and table.label == DocItemLabel.TABLE:
                    # Get the raw table data
                    raw_data = {}
                    
                    # Add table metadata
                    if hasattr(table, 'get_ref') and callable(getattr(table, 'get_ref', None)):
                        try:
                            raw_data['ref'] = table.get_ref().cref
                        except:
                            pass
                    
                    # Add table page number if available
                    if hasattr(table, 'page_number'):
                        raw_data['page_number'] = table.page_number
                    
                    # Add table bounding box if available
                    if hasattr(table, 'bbox') and table.bbox:
                        raw_data['bbox'] = {
                            'x0': table.bbox.x0 if hasattr(table.bbox, 'x0') else None,
                            'y0': table.bbox.y0 if hasattr(table.bbox, 'y0') else None,
                            'x1': table.bbox.x1 if hasattr(table.bbox, 'x1') else None,
                            'y1': table.bbox.y1 if hasattr(table.bbox, 'y1') else None
                        }
                    
                    # Add table caption if available
                    if hasattr(table, 'caption') and table.caption:
                        raw_data['caption'] = table.caption
                    
                    # Add table cells
                    if hasattr(table, 'cells') and table.cells:
                        cells_data = []
                        for row_idx, row in enumerate(table.cells):
                            row_data = []
                            for col_idx, cell in enumerate(row):
                                cell_data = {
                                    'row': row_idx,
                                    'col': col_idx,
                                    'text': cell.text if hasattr(cell, 'text') else '',
                                    'rowspan': cell.rowspan if hasattr(cell, 'rowspan') else 1,
                                    'colspan': cell.colspan if hasattr(cell, 'colspan') else 1
                                }
                                # Add additional cell properties if available
                                if hasattr(cell, 'bbox') and cell.bbox:
                                    cell_data['bbox'] = {
                                        'x0': cell.bbox.x0 if hasattr(cell.bbox, 'x0') else None,
                                        'y0': cell.bbox.y0 if hasattr(cell.bbox, 'y0') else None,
                                        'x1': cell.bbox.x1 if hasattr(cell.bbox, 'x1') else None,
                                        'y1': cell.bbox.y1 if hasattr(cell.bbox, 'y1') else None
                                    }
                                if hasattr(cell, 'is_header') and cell.is_header:
                                    cell_data['is_header'] = True
                                row_data.append(cell_data)
                            cells_data.append(row_data)
                        raw_data['cells'] = cells_data
                        
                        # Add table dimensions
                        raw_data['dimensions'] = {
                            'rows': len(table.cells),
                            'columns': len(table.cells[0]) if table.cells else 0
                        }
                    
                    # Add table as markdown
                    if hasattr(table, 'export_to_markdown') and callable(getattr(table, 'export_to_markdown', None)):
                        try:
                            raw_data['markdown'] = table.export_to_markdown()
                        except:
                            pass
                    
                    raw_outputs.append(raw_data)
        return raw_outputs
