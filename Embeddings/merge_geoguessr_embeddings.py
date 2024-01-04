import os
import pandas as pd

# Directory containing CSV files
directory = '/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/Embeddings/Image/'

# Get a list of filenames that start with "geoguessr" and end with ".csv"
file_list = [file for file in os.listdir(directory) if file.startswith('geoguessr') and file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in file_list:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Use combined_df for further analysis or processing
combined_df.to_csv(f"Image/geoguessr_embeddings.csv", index=False)