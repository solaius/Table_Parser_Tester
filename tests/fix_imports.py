#!/usr/bin/env python3
"""
Script to fix imports in test files.
"""

import os
import re

def fix_file(file_path):
    """Fix imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove any \n# lines
    content = re.sub(r'\\n#[^\n]*\n', '\n', content)
    
    # Check if the import path fix is already there
    if "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))" not in content:
        # Add the import path fix after the imports
        content = re.sub(
            r'(from dotenv import load_dotenv\n)',
            r'\1\n# Add parent directory to Python path\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\')))\n\n',
            content
        )
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Main function."""
    # Get all test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.startswith('test_') and f.endswith('.py')
    ]
    
    # Fix each file
    for file_path in test_files:
        print(f"Fixing {file_path}...")
        fix_file(file_path)

if __name__ == "__main__":
    main()