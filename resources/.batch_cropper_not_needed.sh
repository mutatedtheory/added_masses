#!/bin/bash

# Function to display a progress bar
progress_bar() {
    local width=40
    local percent=$(( $1 * 100 / $2 ))
    local completed=$(( $width * $percent / 100 ))
    printf "\r[%-${width}s] %d%%" "$(printf '='%.0s $(seq 1 $completed))" "$percent"
}

# Function to restore files on script abort
restore_files() {
    echo "Restoring files..."
    for fname in "${moved_files[@]}"; do
        mv "$backup_dir/${fname}" "$fname"
    done
    echo "Files restored."
    cleanup_and_exit
}

# Function to clean up and exit
cleanup_and_exit() {
    printf "\nAborting the script...\n"
    exit 1
}

# Set up the signal handler
trap restore_files INT

# Confirm with the user before proceeding
read -p "This script will move and crop image files. Continue? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    cleanup_and_exit
fi

backup_dir="../images_backup_$(date +"%Y%m%d%H%M%S")"
mkdir -p "$backup_dir"

file_count=$(ls -1 *.png | wc -l)
current_file=0

# Array to store moved files
declare -a moved_files

for fname in *.png; do
    ((current_file++))
    if mv "${fname}" "$backup_dir/${fname}" && \
        convert "$backup_dir/${fname}" -crop 1683x1683+214+61 "${fname}"; then
        progress_bar $current_file $file_count
        moved_files+=("${fname}")
    else
        echo "Error processing ${fname}. Skipping..."
    fi
done

printf "\nDone!\n"
