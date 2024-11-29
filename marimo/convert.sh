#!/bin/bash

# Check if any .py files exist
if ! ls *.py 1> /dev/null 2>&1; then
    echo "No Python files found in the current directory."
    exit 1
fi

# Create a directory for HTML outputs if it doesn't exist
mkdir -p marimo_html

# Loop through all .py files
for file in *.py; do
    # Skip if file is not a regular file
    [ -f "$file" ] || continue

    # Extract filename without extension
    filename=$(basename "$file" .py)

    # Convert to HTML
    echo "Converting $file to HTML..."
    marimo export html "$file" -o "marimo_html/${filename}.html"
done

echo "Conversion complete. HTML files are in the marimo_html directory."
