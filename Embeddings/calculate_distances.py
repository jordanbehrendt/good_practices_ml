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

def calculate_distances(embedding_str):
    start = embedding_str.find('[[')
    end = embedding_str.find(']]')+2
    embedding_str = embedding_str[start:end]
    image_embedding = torch.tensor(ast.literal_eval(embedding_str))
    image_embedding_values = np.array(image_embedding.flatten().tolist()).reshape(1, -1)

    prompt_distances = []
    # Reshape the vectors to be 2D arrays for sklearn's cosine_similarity
    #image_embedding = image_embedding.reshape(1, -1)
    for prompt_embedding in prompt_embeddings:
        prompt_embedding_values = np.array(prompt_embedding.flatten().tolist()).reshape(1, -1)
        # Calculate Cosine Similarity         
        prompt_distances.append(cosine_similarity(image_embedding_values, prompt_embedding_values)[0,0])

    model_input = np.concatenate((image_embedding_values[0], np.array(prompt_distances))).astype(np.float32)
    return model_input.tobytes()

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



# Embeddings of Prompt
prompt_embeddings = torch.load('./Embeddings/Prompt/prompt_image_shows_embedding.pt')

# Directory containing CSV files
directory = '/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/Embeddings/Image'

# Get a list of filenames that start with "geoguessr" and end with ".csv"
geoguessr_file_list = [file for file in os.listdir(directory) if file.startswith('geoguessr') and file.endswith('.csv')]
tourist_df = pd.read_csv('/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/Embeddings/Image/bigfoto_embeddings.csv')
aerial_df = pd.read_csv('/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/Embeddings/Image/aerial_map_embeddings.csv')
print('loaded databases')

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in geoguessr_file_list:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)
print('combined geoguessr databases')

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Calculate distances of all aerial samples
aerial_distances = []
for index, row in aerial_df.iterrows():
    aerial_distances.append(calculate_distances(row['Embedding']))

aerial_df['model_input'] = aerial_distances

# Calculate distances of all geoguessr samples
geoguessr_distances = []
for index, row in combined_df.iterrows():
    geoguessr_distances.append(calculate_distances(row['Embedding']))

combined_df['model_input'] = geoguessr_distances


# Calculate distances of all tourist samples
tourist_distances = []
for index, row in tourist_df.iterrows():
    tourist_distances.append(calculate_distances(row['Embedding']))

tourist_df['model_input'] = tourist_distances


#Save Training Embeddings
save_batches('./Embeddings/Image/','geoguessr_embeddings_batch', combined_df)
save_batches('./Embeddings/Image/','aerial_map_embeddings', aerial_df)
save_batches('./Embeddings/Image/','bigfoto_embeddings', tourist_df)