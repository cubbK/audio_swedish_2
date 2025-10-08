#!/bin/bash

# Function to process a single tar file
process_tar_file() {
    local TAR_FILE="$1"
    local BASE_NAME="${TAR_FILE%.tar}"
    local SORTED_TAR="${BASE_NAME}_sorted.tar"
    
    echo "Processing: $TAR_FILE"
    
    # Check if input file exists
    if [ ! -f "$TAR_FILE" ]; then
        echo "Error: File '$TAR_FILE' not found!"
        return 1
    fi
    
    # 1. Skapa en tillfällig katalog
    mkdir "tmp_sorted_$$"
    cd "tmp_sorted_$$"
    
    # 2. Extrahera gamla tar-filen
    tar -xf "../$TAR_FILE"
    
    # 3. Skapa en ny tar i sorterad ordning
    ls | sort | tar -cf "../$SORTED_TAR" -T -
    
    # 4. (valfritt) Verifiera
    echo "Contents of $SORTED_TAR:"
    tar -tf "../$SORTED_TAR" | head
    
    cd ..
    rm -rf "tmp_sorted_$$"
    rm "$TAR_FILE"
    
    echo "Created sorted tar file: $SORTED_TAR"
    echo "----------------------------------------"
}

# Main script logic
if [ $# -eq 0 ]; then
    # No arguments - process all .tar files in current directory
    echo "Processing all .tar files in current directory..."
    
    tar_files=(*.tar)
    if [ ! -e "${tar_files[0]}" ]; then
        echo "No .tar files found in current directory!"
        exit 1
    fi
    
    for tar_file in *.tar; do
        if [ -f "$tar_file" ]; then
            process_tar_file "$tar_file"
        fi
    done
    
    echo "All tar files processed successfully!"
else
    # Arguments provided - process specific files
    echo "Processing specified tar files..."
    for tar_file in "$@"; do
        process_tar_file "$tar_file"
    done
fi