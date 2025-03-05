import os
from dotenv import load_dotenv
from pdf_table_extractor.app import app

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    # Use the port from runtime information (50213 or 59476)
    # Use DEBUG environment variable if available
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=50213, debug=debug_mode)