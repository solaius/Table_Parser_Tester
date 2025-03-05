import os
from pdf_table_extractor.app import app

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    # Use the port from runtime information (50213 or 59476)
    app.run(host='0.0.0.0', port=50213, debug=True)