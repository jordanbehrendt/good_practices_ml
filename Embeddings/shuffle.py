
import pandas as pd
import os

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

# balanced_directory = './Embeddings/Training/Balanced'
# balanced_file_list = [file for file in os.listdir(balanced_directory)]

# # Initialize an empty list to store DataFrames
# balanced_dfs = []

# # Iterate through the files, read them as DataFrames, and append to the list
# for file in balanced_file_list:
#     file_path = os.path.join(balanced_directory, file)
#     df = pd.read_csv(file_path)
#     balanced_dfs.append(df)

# balanced_df = pd.concat(balanced_dfs, ignore_index=True)
# balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
# save_batches(balanced_directory,'balanced', balanced_df)


# unbalanced_directory = './Embeddings/Training/Unbalanced'
# unbalanced_file_list = [file for file in os.listdir(unbalanced_directory)]

# # Initialize an empty list to store DataFrames
# unbalanced_dfs = []

# # Iterate through the files, read them as DataFrames, and append to the list
# for file in unbalanced_file_list:
#     file_path = os.path.join(unbalanced_directory, file)
#     df = pd.read_csv(file_path)
#     unbalanced_dfs.append(df)

# unbalanced_df = pd.concat(unbalanced_dfs, ignore_index=True)
# unbalanced_df = unbalanced_df.sample(frac=1).reset_index(drop=True)
# save_batches(unbalanced_directory,'unbalanced', unbalanced_df)



# replace_directory = './Embeddings/Training/Replace'
# replace_file_list = [file for file in os.listdir(replace_directory)]

# # Initialize an empty list to store DataFrames
# replace_dfs = []

# # Iterate through the files, read them as DataFrames, and append to the list
# for file in replace_file_list:
#     file_path = os.path.join(replace_directory, file)
#     df = pd.read_csv(file_path)
#     replace_dfs.append(df)

# replace_df = pd.concat(replace_dfs, ignore_index=True)
# replace_df = replace_df.sample(frac=1).reset_index(drop=True)
# save_batches(replace_directory,'replace', replace_df)



strongly_balanced_replace_directory = './Embeddings/Training/Strongly_Balanced_Replace'
strongly_balanced_replace_file_list = [file for file in os.listdir(strongly_balanced_replace_directory)]

# Initialize an empty list to store DataFrames
strongly_balanced_replace_dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in strongly_balanced_replace_file_list:
    file_path = os.path.join(strongly_balanced_replace_directory, file)
    df = pd.read_csv(file_path)
    strongly_balanced_replace_dfs.append(df)

strongly_balanced_replace_df = pd.concat(strongly_balanced_replace_dfs, ignore_index=True)
strongly_balanced_replace_df = strongly_balanced_replace_df.sample(frac=1).reset_index(drop=True)
save_batches(strongly_balanced_replace_directory,'strongly_balanced_replace', strongly_balanced_replace_df)



strongly_balanced_directory = './Embeddings/Training/Strongly_Balanced'
strongly_balanced_file_list = [file for file in os.listdir(strongly_balanced_directory)]

# Initialize an empty list to store DataFrames
strongly_balanced_dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in strongly_balanced_file_list:
    file_path = os.path.join(strongly_balanced_directory, file)
    df = pd.read_csv(file_path)
    strongly_balanced_dfs.append(df)

strongly_balanced_df = pd.concat(strongly_balanced_dfs, ignore_index=True)
strongly_balanced_df = strongly_balanced_df.sample(frac=1).reset_index(drop=True)
save_batches(strongly_balanced_directory,'strongly_balanced', strongly_balanced_df)