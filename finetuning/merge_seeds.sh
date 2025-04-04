#!/bin/bash

# Set the source and destination directories
source_dir="path/to/finetuning/runs/output"
destination_dir="path/to/finetuning/runs/output/merged_seeds"

# Loop through all the files in the source directory
for file in "$source_dir"/*/*/*/*/*/*; do
    # Extract the necessary information from the file path 
    filepath=$(dirname "$file") # Set the file path to the directory of the current file in the loop
    filename=$(basename "$file")
    seed=$(echo "$filepath" | cut -d/ -f10 )
    dataset_name=$(echo "$filepath" | cut -d/ -f11 )
    loss_configuration=$(echo "$filepath" | cut -d/ -f12 )
    # Create the destination directory if it doesn't exist
    mkdir -p "$destination_dir/$dataset_name/$loss_configuration"

    #echo "$file"
    #echo "$destination_dir/$dataset_name/$loss_configuration/$filename"_"$seed"
    #break
    # Move the file to the destination directory with the modified filename
    cp "$file" "$destination_dir/$dataset_name/$loss_configuration/$seed"_"$filename"
done
