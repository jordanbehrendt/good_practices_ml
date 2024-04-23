#!/bin/bash

# Set the source and destination directories
source_dir="/media/leon/Samsung_T5/Uni/good_practices_ml/runs/all_seeds"
destination_dir="/media/leon/Samsung_T5/Uni/good_practices_ml/runs/merged_seeeds"

# Loop through all the files in the source directory
for file in "$source_dir"/*/*/*/*/*; do
    # Extract the necessary information from the file path 
    # /home/lbrenig/Documents/Uni/GPML/good_practices_ml/runs/seed_3838/geo_strongly_balanced/starting_regional_loss_portion-0.0/regional_loss_decline-1.0/events.out.tfevents.1709967840.lbrenig-MS-7D09.14941.68
    filepath=$(dirname "$file") # Set the file path to the directory of the current file in the loop
    filename=$(basename "$file")
    seed=$(echo "$filepath" | cut -d/ -f9 )
    dataset_name=$(echo "$filepath" | cut -d/ -f10 )
    loss_configuration=$(echo "$filepath" | cut -d/ -f11 )
    # Create the destination directory if it doesn't exist
    mkdir -p "$destination_dir/$dataset_name/$loss_configuration"

    # Move the file to the destination directory with the modified filename
    cp "$file" "$destination_dir/$dataset_name/$loss_configuration/$filename"_"$seed"
done
