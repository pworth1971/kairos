#!/bin/bash

# Directory containing the files
DIRECTORY="/Users/pietro/code/kairos/DARPA/CADETS_E5/Data/bin/"

# Loop through all files containing 'bin' in their name
for file in "$DIRECTORY"*bin*; do
    # Check if the file exists to avoid errors
    if [[ -f "$file" ]]; then
        echo "Processing file: $file"
        ./json_consumer.sh "$file"
    else
        echo "No matching files found."
    fi
done
