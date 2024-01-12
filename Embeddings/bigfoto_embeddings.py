import clip
import torch
import pandas as pd
import sys
import PIL
sys.path.append("./../")
from scripts import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)

bigfoto_df = load_dataset.load_data(DATA_PATH="/share/temp/bjordan/good_practices_in_machine_learning/BigFoto/")
print(bigfoto_df)

bigfoto_df["Embedding"] = bigfoto_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
bigfoto_df.to_csv("Image/bigfoto_embeddings.csv")