import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import os
sys.path.append('./scripts')
from scripts import load_dataset
import sklearn.model_selection
import ast
from sklearn.metrics.pairwise import cosine_similarity

def balance_data(max_num, unbalanced_df, country_list):
    balanced_array = []
    for index,row in country_list.iterrows():
        country_df = unbalanced_df.loc[unbalanced_df["label"] == row["Country"]]
        country_samples_count = len(country_df.index)
        if country_samples_count > max_num:
            drop_indices = np.random.choice(country_df.index,country_samples_count-max_num, replace=False)
            replace_df = country_df.drop(drop_indices)
            balanced_array.append(replace_df)
        else:
            balanced_array.append(country_df)
    balanced_df = pd.concat(balanced_array, ignore_index=True)
    return balanced_df

def save_embeddings(path, name, dataframe):
    dataframe.to_csv(f"{path}/{name}.csv", index=False)

def save_batches(path, name, dataframe):
    batch_size = 1000
    num_batches = len(dataframe) // batch_size

    # Iterate over the DataFrame in batches
    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataframe))
        
        batch_df = dataframe.iloc[start_idx:end_idx]
                
        # Save or process the batch as needed
        batch_df.to_csv(f"{path}/{name}_{i}.csv", index=False)


# Training Directory containing CSV files
weakly_balanced_directory = '/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/Embeddings/Training/Weakly_Balanced'
weakly_balanced_replace_directory = '/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/Embeddings/Training/Weakly_Balanced_Replace'

# Get a list of filenames that start with "geoguessr" and end with ".csv"
weakly_balanced_file_list = [file for file in os.listdir(weakly_balanced_directory) if file.endswith('.csv')]
weakly_balanced_replace_file_list = [file for file in os.listdir(weakly_balanced_replace_directory) if file.endswith('.csv')]
print('loaded databases')

# Initialize an empty list to store DataFrames
weakly_balanced_dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in weakly_balanced_file_list:
    file_path = os.path.join(weakly_balanced_directory, file)
    df = pd.read_csv(file_path)
    weakly_balanced_dfs.append(df)
print('combined weakly_balanced databases')

# Concatenate all DataFrames in the list into a single DataFrame
combined_weakly_balanced_df = pd.concat(weakly_balanced_dfs, ignore_index=True)

weakly_balanced_replace_dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in weakly_balanced_replace_file_list:
    file_path = os.path.join(weakly_balanced_replace_directory, file)
    df = pd.read_csv(file_path)
    weakly_balanced_replace_dfs.append(df)
print('combined weakly_balanced_replace databases')

# Concatenate all DataFrames in the list into a single DataFrame
combined_weakly_balanced_replace_df = pd.concat(weakly_balanced_replace_dfs, ignore_index=True)

country_list = pd.read_csv("/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/data_finding/country_list.csv")

# Create Balanced Dataset
strongly_balanced_df = balance_data(max_num=200,country_list=country_list, unbalanced_df=combined_weakly_balanced_df)
strongly_balanced_replace_df = balance_data(max_num=200,country_list=country_list, unbalanced_df=combined_weakly_balanced_replace_df)

#Save Training Embeddings
save_batches('./Embeddings/Training/Strongly_Balanced','strongly_balanced', strongly_balanced_df)
save_batches('./Embeddings/Training/Strongly_Balanced_Replace','strongly_balanced_replace', strongly_balanced_replace_df)