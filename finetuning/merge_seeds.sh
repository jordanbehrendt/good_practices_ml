#!/bin/bash

# Set the source and destination directories
source_dir="/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/finetuning/runs/all_seeds2"
destination_dir="/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/finetuning/runs/merged_seeds2"

# Loop through all the files in the source directory
for file in "$source_dir"/*/*/*/*/*/*; do
    # Extract the necessary information from the file path 
    # /home/lbrenig/Documents/Uni/GPML/good_practices_ml/finetuning/runs/seed_3838/geo_strongly_balanced/starting_regional_loss_portion-0.0/regional_loss_decline-1.0/events.out.tfevents.1709967840.lbrenig-MS-7D09.14941.68
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
