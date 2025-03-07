#!/usr/bin/env python3
"""
Script to fix the DataFrame creation in docling_granitevision.py
"""

import re

def fix_dataframe_creation(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the DataFrame creation code
    pattern = r"""header = cleaned_lines\[0\]\.split\('\|'\)
\s+data = \[row\.split\('\|'\) for row in cleaned_lines if '---' not in row and row != cleaned_lines\[0\]\]
\s+
\s+df = pd\.DataFrame\(data, columns=\[h\.strip\(\) for h in header\]\)"""
    
    # Replacement code with proper column handling
    replacement = """header = [h.strip() for h in cleaned_lines[0].split('|')]
                        data = []
                        
                        # Process each row, ensuring it has the same number of columns as the header
                        for row in cleaned_lines:
                            if '---' not in row and row != cleaned_lines[0]:
                                row_data = [cell.strip() for cell in row.split('|')]
                                # Ensure the row has the same number of columns as the header
                                if len(row_data) < len(header):
                                    # Add empty cells if needed
                                    row_data.extend([''] * (len(header) - len(row_data)))
                                elif len(row_data) > len(header):
                                    # Truncate if there are too many columns
                                    row_data = row_data[:len(header)]
                                data.append(row_data)
                        
                        df = pd.DataFrame(data, columns=header)"""
    
    # Replace all occurrences
    updated_content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {file_path}")

if __name__ == "__main__":
    fix_dataframe_creation("/workspace/Table_Parser_Tester/pdf_table_extractor/docling_granitevision.py")