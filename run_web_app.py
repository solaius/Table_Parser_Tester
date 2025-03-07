import os
from dotenv import load_dotenv
from pdf_table_extractor.app import app

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 54800))
    # Use DEBUG environment variable if available
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"Starting web app on port {port} with debug mode {'enabled' if debug_mode else 'disabled'}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)