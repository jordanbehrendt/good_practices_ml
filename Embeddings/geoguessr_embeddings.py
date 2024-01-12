import clip
import torch
import pandas as pd
import sys
import PIL
sys.path.append("./../")
from scripts import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)

geoguessr_df = load_dataset.load_data(DATA_PATH="/share/temp/bjordan/good_practices_in_machine_learning/compressed_dataset/")
print(geoguessr_df)

batch_size = 1000
num_batches = len(geoguessr_df) // batch_size

# Iterate over the DataFrame in batches
for i in range(num_batches + 1):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(geoguessr_df))
    
    batch_df = geoguessr_df.iloc[start_idx:end_idx]
    
    # Apply the encoding function to each row in the batch
    batch_df["Embedding"] = batch_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
    
    # Save or process the batch as needed
    batch_df.to_csv(f"Image/geoguessr_embeddings_batch_{i}.csv", index=False)

#geoguessr_df["Embedding"] = geoguessr_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
#geoguessr_df.to_csv("Image/geoguessr_embeddings.csv")