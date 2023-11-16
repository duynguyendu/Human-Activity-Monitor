#!/bin/bash

# Directory containing the folders to process
base_dir="/home/ht0710/Documents/GitHub/Human-Activity-Recognition/data/processed/0.3"

# Iterate through the folders in the base directory
for folder in "$base_dir"/*; do
    # Check if the path is a directory
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"

		zip -r9 "$folder.zip" "$folder"
        # python /home/ht0710/Documents/GitHub/Human-Activity-Recognition/tools/labeling/name_fix.py -p "$folder"
        # python /home/ht0710/Documents/GitHub/Human-Activity-Recognition/tools/labeling/divide.py -p "$folder" -bp 10000
    fi
done
